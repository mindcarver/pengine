import copy
import hashlib
import json
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx
import pytest
from langchain.agents import create_agent
from langchain.agents.middleware.types import ModelRequest, ModelResponse, ToolCallRequest
from langchain.agents.structured_output import (
    OutputToolBinding,
    StructuredOutputValidationError,
    ToolStrategy,
)
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import StructuredTool
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from persona_factory import create_persona_package
from pydantic import BaseModel, Field, ValidationError, field_validator

from pengine.agents import (
    _EPISODE_PLANNER_PROMPT,
    _SCRIPT_WRITER_PROMPT,
    _SPECIALIST_SKILL_SOURCES,
    _STORY_ARCHITECT_PROMPT,
    SKILLED_WRITE_PERMISSIONS,
    VIRTUAL_FILE_PERMISSIONS,
    AgentProtocolError,
    CanonReviewerResult,
    ContentReviewRejectedError,
    DeepAgentWorkflow,
    EpisodePlannerResult,
    EpisodeReviewerResult,
    OutlineRepairPatch,
    QualityGateRejectedError,
    QualityReviewerResult,
    ScriptWriterResult,
    StageGuardMiddleware,
    StoryArchitectResult,
    StoryArtifactRepairPatch,
    StructuredResultMiddleware,
    WorkflowCompletion,
    _apply_outline_repair_patch,
    _apply_story_artifact_repair_patch,
    _arithmetic_tool,
    _calculate_arithmetic,
    _canon_issue_ledger,
    _drop_dangling_tool_call_messages,
    _language_retry_fingerprint,
    _language_retry_matches,
    _merge_story_canon_reviews,
    _outline_repair_context,
    _outline_repair_result,
    _request_with_canonical_workspace,
    _story_patch_correction,
    _story_repair_context,
    _structured_output_retry_message,
    _structured_result_validation_correction,
    _supervisor_prompt,
    _validate_outline_repair_patch_targets,
    _validate_result_language,
    flatten_cr_candidate,
)
from pengine.config import Settings
from pengine.continuity import (
    EpisodeStateDelta,
    StoryContract,
    render_story_contract_markdown,
    story_contract_sha256,
)
from pengine.language import SIMPLIFIED_CHINESE, language_instruction
from pengine.personas import PersonaCatalog
from pengine.repository import Repository
from pengine.schemas import CreateCreationRequest, EpisodeDraft, EpisodePlan, InternalStage
from pengine.series_bible import build_series_bible, project_series_bible
from pengine.skill_assets import load_agent_skill_files
from pengine.worker import Worker


class ToolCallingFakeModel(FakeMessagesListChatModel):
    bound_tool_names: list[list[str]] = Field(default_factory=list)
    bound_tool_descriptions: list[list[str]] = Field(default_factory=list)

    def bind_tools(
        self,
        tools: Sequence[Any],
        **_: Any,
    ) -> "ToolCallingFakeModel":
        self.bound_tool_names.append([_tool_name(tool) for tool in tools])
        self.bound_tool_descriptions.append([getattr(tool, "description", "") for tool in tools])
        return self


def _fake_workflow(
    *,
    model: ToolCallingFakeModel,
    checkpointer: Any,
    recursion_limit: int = 80,
    provider_profile_key: str = "toolcallingfakemodel",
) -> DeepAgentWorkflow:
    return DeepAgentWorkflow(
        generation_model=model,
        review_model=model,
        checkpointer=checkpointer,
        recursion_limit=recursion_limit,
        generation_provider_profile_key=provider_profile_key,
        review_provider_profile_key=provider_profile_key,
    )


def _tool_name(tool: Any) -> str:
    if hasattr(tool, "name"):
        return tool.name
    if isinstance(tool, dict):
        if "name" in tool:
            return tool["name"]
        return tool.get("function", {}).get("name", "")
    return ""


def _tool_call(name: str, args: dict[str, Any], index: int) -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[
            {
                "name": name,
                "args": args,
                "id": f"call-{index}",
                "type": "tool_call",
            }
        ],
    )


def _story_contract(episode_count: int = 1) -> StoryContract:
    facts = [
        {
            "fact_id": f"fact_ep{episode}",
            "subject": "测试人物",
            "predicate": "确认事实",
            "kind": "text",
            "value": f"事实{episode}",
            "first_revealed_episode": episode,
        }
        for episode in range(1, episode_count + 1)
    ]
    return StoryContract.model_validate(
        {
            "version": 1,
            "episode_count": episode_count,
            "characters": [
                {
                    "character_id": "test_character",
                    "name": "测试人物",
                    "role": "主角",
                    "initial_known_fact_ids": [],
                }
            ],
            "relationships": [],
            "facts": facts,
            "timeline": [
                {
                    "event_id": f"event_ep{episode}",
                    "order": episode,
                    "when": f"episode-{episode}",
                    "participant_ids": ["test_character"],
                    "fact_ids": [f"fact_ep{episode}"],
                }
                for episode in range(1, episode_count + 1)
            ],
            "knowledge_states": [
                {
                    "episode_number": episode,
                    "character_id": "test_character",
                    "known_fact_ids": [f"fact_ep{known}" for known in range(1, episode + 1)],
                }
                for episode in range(1, episode_count + 1)
            ],
            "clues": [],
            "prohibitions": ["不得增加人物"],
            "episode_obligations": [
                {
                    "obligation_id": f"obligation_ep{episode}",
                    "episode_number": episode,
                    "new_information_fact_ids": [f"fact_ep{episode}"],
                    "end_hook": f"钩子{episode}",
                    "required_clue_ids": [],
                }
                for episode in range(1, episode_count + 1)
            ],
        }
    )


def _state_delta(contract: StoryContract, episode_number: int) -> dict[str, Any]:
    contract_hash = story_contract_sha256(contract)
    return {
        "episode_number": episode_number,
        "contract_sha256": contract_hash,
        "established_fact_ids": [f"fact_ep{episode_number}"],
        "knowledge_gains": [
            {
                "character_id": "test_character",
                "fact_ids": [f"fact_ep{episode_number}"],
            }
        ],
        "introduced_clue_ids": [],
        "resolved_clue_ids": [],
        "satisfied_obligation_ids": [f"obligation_ep{episode_number}"],
        "evidence": [
            {"target_id": f"fact_ep{episode_number}", "excerpt": f"事实{episode_number}"},
            {
                "target_id": f"obligation_ep{episode_number}",
                "excerpt": f"钩子{episode_number}",
            },
        ],
        "handoff": f"第{episode_number}集结束",
    }


def _successful_responses() -> list[AIMessage]:
    contract = _story_contract()
    stages = [
        (
            "selecting_l0_variant",
            "story_architect",
            "StoryArchitectResult",
            {
                "stage": "selecting_l0_variant",
                "content": None,
                "character_biographies": None,
                "relationship_logic": None,
                "selected_l0_variant": "主动选择",
                "selection_rationale": "契合故事",
            },
        ),
        (
            "generating_story_outline",
            "story_architect",
            "StoryArchitectResult",
            {
                "stage": "generating_story_outline",
                "content": "故事大纲",
                "character_biographies": None,
                "relationship_logic": None,
                "selected_l0_variant": None,
                "selection_rationale": None,
            },
        ),
        (
            "generating_character_relationships",
            "story_architect",
            "StoryArchitectResult",
            {
                "stage": "generating_character_relationships",
                "content": None,
                "character_biographies": "人物小传",
                "relationship_logic": "关系逻辑",
                "selected_l0_variant": None,
                "selection_rationale": None,
            },
        ),
        (
            "generating_episode_outline",
            "episode_planner",
            "EpisodePlannerResult",
            {
                "stage": "generating_episode_outline",
                "content": "分集大纲",
                "episode_count": 1,
                "episodes": [{"episode_number": 1, "plan": "第一集计划"}],
                "story_contract": contract.model_dump(mode="json"),
            },
        ),
        (
            "generating_episode_scripts",
            "script_writer",
            "ScriptWriterResult",
            {
                "stage": "generating_episode_scripts",
                "episode_number": 1,
                "content": "事实1\n钩子1",
                "state_delta": _state_delta(contract, 1),
            },
        ),
        (
            "accepting_l0",
            "quality_reviewer",
            "QualityReviewerResult",
            {
                "stage": "accepting_l0",
                "passed": True,
                "evidence": "符合 L0",
                "feedback_handling": [],
            },
        ),
        (
            "accepting_l4",
            "quality_reviewer",
            "QualityReviewerResult",
            {
                "stage": "accepting_l4",
                "passed": True,
                "evidence": "符合 L4",
                "feedback_handling": [],
            },
        ),
    ]
    responses: list[AIMessage] = []
    index = 0
    for stage, subagent, schema, payload in stages:
        responses.append(
            _tool_call(
                "task",
                {
                    "description": f"[stage={stage}] execute the stage",
                    "subagent_type": subagent,
                },
                index,
            )
        )
        responses.append(_tool_call(schema, payload, index + 1))
        index += 2
        if stage == "generating_story_outline":
            # Outline stage: single-lens canon review (1 review call).
            responses.append(
                _tool_call(
                    "CanonReviewerResult",
                    {"passed": True, "evidence": "故事大纲一致", "issues": []},
                    index,
                )
            )
            index += 1
        if stage == "generating_character_relationships":
            # Character + relationships stage: two-lens canon review (2 review calls).
            for _ in range(2):
                responses.append(
                    _tool_call(
                        "CanonReviewerResult",
                        {"passed": True, "evidence": "故事工件一致", "issues": []},
                        index,
                    )
                )
                index += 1
        if stage == "generating_episode_outline":
            responses.append(
                _tool_call(
                    "CanonReviewerResult",
                    {"passed": True, "evidence": "合同一致", "issues": []},
                    index,
                )
            )
            index += 1
        if stage == "generating_episode_scripts":
            responses.append(
                _tool_call(
                    "EpisodeReviewerResult",
                    {"passed": True, "evidence": "分集一致", "issues": []},
                    index,
                )
            )
            index += 1
    responses.append(
        _tool_call(
            "WorkflowCompletion",
            {"completed": True},
            index,
        )
    )
    return responses


def _index_of_tool_call(responses: list[AIMessage], name: str, *, occurrence: int = 1) -> int:
    """1-based occurrence index of the Nth AIMessage carrying a tool call named ``name``."""
    seen = 0
    for index, message in enumerate(responses):
        if message.tool_calls and message.tool_calls[0]["name"] == name:
            seen += 1
            if seen == occurrence:
                return index
    raise AssertionError(f"No tool call named {name!r} (occurrence {occurrence}) in responses")


def _successful_responses_unified() -> list[AIMessage]:
    """The unified SeriesBible flow response sequence.

    In the unified path the writer relies on deterministic per-episode validation and
    the declared structural milestone/final reviews, so the per-episode
    ``episode_reviewer`` response is replaced by the bound final ``series_reviewer``
    result for the single-episode series.
    """
    responses = _successful_responses()
    episode_review_index = _index_of_tool_call(responses, "EpisodeReviewerResult", occurrence=1)
    unified: list[AIMessage] = []
    for index, response in enumerate(responses):
        if index == episode_review_index:
            unified.append(
                _tool_call(
                    "StructuralReviewResult",
                    {"passed": True, "category": "pass", "evidence": "全系列一致"},
                    index,
                )
            )
        else:
            unified.append(response)
    return unified


def _episode_hook_kwargs(
    *,
    episode_drafts: list[EpisodeDraft] | None = None,
) -> tuple[dict[str, Any], list[int]]:
    committed = {draft.episode_number: draft for draft in episode_drafts or []}
    attempts: list[int] = []

    async def before_episode(plan: EpisodePlan) -> int:
        attempts.append(plan.episode_number)
        return 1

    async def commit_episode(
        episode_number: int,
        content: str,
        episode_lock=None,
        **kwargs,
    ) -> EpisodeDraft:
        existing = committed.get(episode_number)
        content_hash = hashlib.sha256(content.encode()).hexdigest()
        if existing is not None:
            assert existing.content_sha256 == content_hash
            return existing
        draft = EpisodeDraft(
            episode_number=episode_number,
            content=content,
            content_sha256=content_hash,
            completed_at=datetime(2026, 7, 31, tzinfo=UTC),
            contract_sha256=(episode_lock.contract_sha256 if episode_lock else None),
            state_delta=(episode_lock.state_delta if episode_lock else None),
            series_state=(episode_lock.series_state if episode_lock else None),
            series_state_sha256=(episode_lock.series_state_sha256 if episode_lock else None),
            semantic_review=(episode_lock.semantic_review if episode_lock else None),
            repair_rounds=(episode_lock.repair_rounds if episode_lock else None),
        )
        committed[episode_number] = draft
        return draft

    async def assemble_episode_scripts() -> str:
        return "\n\n---\n\n".join(
            f"第 {episode_number} 集\n{draft.content}"
            for episode_number, draft in sorted(committed.items())
        )

    return (
        {
            "episode_drafts": list(committed.values()),
            "before_episode": before_episode,
            "commit_episode": commit_episode,
            "assemble_episode_scripts": assemble_episode_scripts,
        },
        attempts,
    )


def test_story_architect_schema_exposes_stage_specific_field_contract() -> None:
    properties = StoryArchitectResult.model_json_schema()["properties"]

    # The outline stage uses the single ``content`` field; the c+r stage uses the
    # two ``character_biographies`` / ``relationship_logic`` fields. The old
    # ``story_outline`` property is gone.
    assert "story_outline" not in properties
    assert "generating_story_outline" in properties["content"]["description"]
    assert "Must be null for selecting_l0_variant" in properties["content"]["description"]
    for cr_field in ("character_biographies", "relationship_logic"):
        assert "generating_character_relationships" in properties[cr_field]["description"]
        assert "Must be null for selecting_l0_variant" in properties[cr_field]["description"]
    assert "selecting_l0_variant" in properties["selected_l0_variant"]["description"]
    assert "do not add an English translation" in (properties["selected_l0_variant"]["description"])
    assert "Must be null" in properties["selected_l0_variant"]["description"]
    assert "selecting_l0_variant" in properties["selection_rationale"]["description"]
    assert "without an English translation" in (properties["selection_rationale"]["description"])
    assert "Must be null" in properties["selection_rationale"]["description"]


def test_quality_reviewer_schema_exposes_gate_decision_contract() -> None:
    properties = QualityReviewerResult.model_json_schema()["properties"]

    assert properties["passed"]["type"] == "boolean"
    assert "concrete evidence" in properties["passed"]["description"]
    assert "accepting_l0" in properties["feedback_handling"]["description"]
    assert "initial run" in properties["feedback_handling"]["description"]
    assert "revision" in properties["feedback_handling"]["description"]


def test_outline_repair_schema_cannot_repeat_the_full_candidate() -> None:
    properties = OutlineRepairPatch.model_json_schema()["properties"]

    assert set(properties) == {"stage", "content_replacements", "json_edits"}
    assert not {"content", "episode_count", "episodes", "story_contract"} & set(properties)


def test_story_repair_schema_is_line_addressed_and_cannot_repeat_the_candidate() -> None:
    schema = StoryArtifactRepairPatch.model_json_schema()
    properties = schema["properties"]
    replacement_ref = properties["line_replacements"]["items"]["$ref"].split("/")[-1]
    replacement_properties = schema["$defs"][replacement_ref]["properties"]

    assert set(properties) == {"stage", "line_replacements"}
    assert set(replacement_properties) == {"start_line", "end_line", "replacement"}
    assert "maxItems" not in properties["line_replacements"]
    assert "content" not in properties


def test_script_writer_schema_requires_non_null_state_delta() -> None:
    schema = ScriptWriterResult.model_json_schema()

    assert "state_delta" in schema["required"]
    assert "anyOf" not in schema["properties"]["state_delta"]
    assert schema["properties"]["state_delta"]["$ref"]
    assert "complete verbatim screenplay" in schema["properties"]["content"]["description"]
    assert "completion summary" in schema["properties"]["content"]["description"]


def test_script_writer_accepts_json_encoded_state_delta_from_tool_call() -> None:
    contract = _story_contract()
    state_delta = _state_delta(contract, 1)

    result = ScriptWriterResult.model_validate(
        {
            "stage": "generating_episode_scripts",
            "episode_number": 1,
            "content": "第一集完整剧本",
            "state_delta": json.dumps(state_delta, ensure_ascii=False),
        }
    )

    assert result.state_delta.model_dump(mode="json") == state_delta


def test_script_writer_tool_binding_accepts_json_encoded_state_delta_without_retry() -> None:
    contract = _story_contract()
    state_delta = _state_delta(contract, 1)
    strategy = ToolStrategy(ScriptWriterResult)
    binding = OutputToolBinding.from_schema_spec(strategy.schema_specs[0])

    result = binding.parse(
        {
            "stage": "generating_episode_scripts",
            "episode_number": 1,
            "content": "第一集完整剧本",
            "state_delta": json.dumps(state_delta, ensure_ascii=False),
        }
    )

    assert isinstance(result, ScriptWriterResult)
    assert result.state_delta.model_dump(mode="json") == state_delta


@pytest.mark.parametrize(
    "encoded", ["{not-json}", "[]", "null", '"text"', '"{\\"episode_number\\": 1}"']
)
def test_script_writer_rejects_invalid_json_encoded_state_delta(encoded: str) -> None:
    with pytest.raises(ValidationError):
        ScriptWriterResult.model_validate(
            {
                "stage": "generating_episode_scripts",
                "episode_number": 1,
                "content": "第一集完整剧本",
                "state_delta": encoded,
            }
        )


def test_story_artifact_patch_repairs_only_numbered_minimal_lines() -> None:
    content = flatten_cr_candidate(
        character_biographies="人物小传确认程远二十二岁，比程屿大两岁。",
        relationship_logic=(
            "程远在海难时二十四岁，比程屿大约六岁。\n"
            "兄弟二人的年龄差决定了程屿对兄长的依赖，也影响调查中的选择。\n"
            "其余人物关系与已批准的小传保持不变，并继续约束后续情节与人物行为。"
        ),
    )
    relationship_conflict_line = (
        content.split("\n").index("程远在海难时二十四岁，比程屿大约六岁。") + 1
    )
    review = CanonReviewerResult(
        passed=False,
        evidence="年龄与人物小传冲突",
        issues=[
            {
                "code": "relative_age_conflict",
                "message": "候选年龄应与小传的二十二岁、相差两岁一致",
                "script_excerpt": "程远在海难时二十四岁，比程屿大约六岁。",
            }
        ],
    )
    patch = StoryArtifactRepairPatch.model_validate(
        {
            "stage": "generating_character_relationships",
            "line_replacements": [
                {
                    "start_line": relationship_conflict_line,
                    "end_line": relationship_conflict_line,
                    "replacement": "程远在海难时二十二岁，比程屿大两岁。",
                }
            ],
        }
    )

    context = _story_repair_context(
        stage=InternalStage.GENERATING_CHARACTER_RELATIONSHIPS,
        content=content,
        review=review,
    )
    repaired = _apply_story_artifact_repair_patch(
        stage=InternalStage.GENERATING_CHARACTER_RELATIONSHIPS,
        content=content,
        patch=patch,
    )

    assert context["candidate_lines"][relationship_conflict_line - 1] == {
        "line_number": relationship_conflict_line,
        "text": "程远在海难时二十四岁，比程屿大约六岁。",
    }
    assert context["confirmed_issues"][0]["code"] == "relative_age_conflict"
    assert "二十二岁，比程屿大两岁" in repaired.relationship_logic
    assert "二十四岁" not in repaired.relationship_logic


def test_story_artifact_patch_rejects_invalid_overlapping_or_unchanged_lines() -> None:
    content = "标题\n年龄冲突。\n关系冲突。\n其余关系文本保持不变并提供足够上下文。"
    invalid_range = StoryArtifactRepairPatch.model_validate(
        {
            "stage": "generating_story_outline",
            "line_replacements": [{"start_line": 9, "end_line": 9, "replacement": "年龄一致。"}],
        }
    )
    overlapping = StoryArtifactRepairPatch.model_validate(
        {
            "stage": "generating_story_outline",
            "line_replacements": [
                {"start_line": 2, "end_line": 3, "replacement": "关系一致。"},
                {"start_line": 3, "end_line": 3, "replacement": "年龄一致。"},
            ],
        }
    )
    unchanged = StoryArtifactRepairPatch.model_validate(
        {
            "stage": "generating_story_outline",
            "line_replacements": [{"start_line": 2, "end_line": 2, "replacement": "年龄冲突。"}],
        }
    )

    with pytest.raises(ValueError, match="story_repair_line_range_invalid"):
        _apply_story_artifact_repair_patch(
            stage=InternalStage.GENERATING_STORY_OUTLINE,
            content=content,
            patch=invalid_range,
        )
    with pytest.raises(ValueError, match="overlapping_story_line_replacement"):
        _apply_story_artifact_repair_patch(
            stage=InternalStage.GENERATING_STORY_OUTLINE,
            content=content,
            patch=overlapping,
        )
    with pytest.raises(ValueError, match="story_repair_line_did_not_change"):
        _apply_story_artifact_repair_patch(
            stage=InternalStage.GENERATING_STORY_OUTLINE,
            content=content,
            patch=unchanged,
        )

    correction = _story_patch_correction(
        ValueError("story_repair_line_range_invalid"),
        content=content,
        patch=invalid_range,
    )
    assert "Candidate has 4 numbered lines" in correction
    assert '"start_line": 9' in correction


def test_story_artifact_patch_discards_harmless_no_op_alongside_real_repairs() -> None:
    content = (
        "台风预警在七月二十日晚发布。\n"
        "林夏必须在封岛前查清旧表来历。\n"
        "其余人物身份、动机、知识来源、时间线与证物约束均保持不变。"
    )
    outline_lines = content.split("\n")
    date_line = outline_lines.index("台风预警在七月二十日晚发布。") + 1
    closing_line = (
        outline_lines.index("其余人物身份、动机、知识来源、时间线与证物约束均保持不变。") + 1
    )
    patch = StoryArtifactRepairPatch.model_validate(
        {
            "stage": "generating_story_outline",
            "line_replacements": [
                {
                    "start_line": date_line,
                    "end_line": date_line,
                    "replacement": "台风预警在七月十九日晚发布。",
                },
                {
                    "start_line": closing_line,
                    "end_line": closing_line,
                    "replacement": "其余人物身份、动机、知识来源、时间线与证物约束均保持不变。",
                },
            ],
        }
    )

    repaired = _apply_story_artifact_repair_patch(
        stage=InternalStage.GENERATING_STORY_OUTLINE,
        content=content,
        patch=patch,
    )

    assert "七月十九日晚" in repaired.content
    assert repaired.content.endswith("其余人物身份、动机、知识来源、时间线与证物约束均保持不变。")


def test_story_artifact_patch_enforces_total_old_or_new_line_change_budget() -> None:
    content = "第一行。\n第二行。\n第三行。\n第四行。"
    patch = StoryArtifactRepairPatch.model_validate(
        {
            "stage": "generating_story_outline",
            "line_replacements": [
                {
                    "start_line": 2,
                    "end_line": 2,
                    "replacement": "这是一个长度接近完整候选的新事实插入，不能绕过预算。",
                }
            ],
        }
    )

    with pytest.raises(ValueError, match="story_repair_patch_not_minimal"):
        _apply_story_artifact_repair_patch(
            stage=InternalStage.GENERATING_STORY_OUTLINE,
            content=content,
            patch=patch,
        )


def test_story_artifact_patch_allows_in_scope_multi_line_outline_repair() -> None:
    # Delivery #57: a canon repair that legitimately fixes several lines of a short
    # story outline is bounded line addressing, not an over-scope rewrite. It must
    # apply while each replacement stays smaller than the whole candidate.
    content = (
        "开局：林夏在台风夜回到旧屋。\n"
        "中段：林夏发现旧表与父亲失踪有关。\n"
        "结尾：林夏决定调查旧表来历。\n"
        "其余人物设定与已批准大纲保持一致。"
    )
    outline_lines = content.split("\n")
    opening_line = outline_lines.index("开局：林夏在台风夜回到旧屋。") + 1
    middle_line = outline_lines.index("中段：林夏发现旧表与父亲失踪有关。") + 1
    ending_line = outline_lines.index("结尾：林夏决定调查旧表来历。") + 1
    patch = StoryArtifactRepairPatch.model_validate(
        {
            "stage": "generating_story_outline",
            "line_replacements": [
                {
                    "start_line": opening_line,
                    "end_line": opening_line,
                    "replacement": "开局：林夏在台风夜赶回旧屋，发现门锁被换。",
                },
                {
                    "start_line": middle_line,
                    "end_line": middle_line,
                    "replacement": "中段：林夏查明旧表与父亲失踪直接相关。",
                },
                {
                    "start_line": ending_line,
                    "end_line": ending_line,
                    "replacement": "结尾：林夏决定留在岛上继续追查真相。",
                },
            ],
        }
    )

    repaired = _apply_story_artifact_repair_patch(
        stage=InternalStage.GENERATING_STORY_OUTLINE,
        content=content,
        patch=patch,
    )
    assert "发现门锁被换" in repaired.content
    assert "直接相关" in repaired.content
    assert "继续追查真相" in repaired.content
    assert "其余人物设定与已批准大纲保持一致" in repaired.content
    assert "其余人物设定与已批准大纲保持一致" in content  # closing line is untouched


def test_canon_review_rejects_non_blocking_suggestions_as_issues() -> None:
    with pytest.raises(
        ValidationError,
        match="Canon review issues must contain only blocking contradictions",
    ):
        CanonReviewerResult.model_validate(
            {
                "passed": False,
                "evidence": "这里只是偏好建议。",
                "issues": [
                    {
                        "code": "optional_naming",
                        "message": "这不是逻辑矛盾，仅建议换一个名字，不构成失败。",
                    }
                ],
            }
        )


def _prior_story_review() -> CanonReviewerResult:
    return CanonReviewerResult(
        passed=False,
        evidence="上轮确认人物年龄冲突。",
        issues=[
            {
                "code": "relative_age_conflict",
                "message": "上游小传为二十二岁。",
                "script_excerpt": "人物年龄二十四岁。",
            }
        ],
    )


def _passing_canon_review_with_closure(
    prior: CanonReviewerResult,
    *,
    status: str | None,
) -> CanonReviewerResult:
    closures = []
    if status is not None:
        closures = [
            {
                "issue_id": _canon_issue_ledger(prior)[0]["issue_id"],
                "status": status,
                "evidence": "当前候选已写为二十二岁，并与批准小传一致。",
            }
        ]
    return CanonReviewerResult.model_validate(
        {
            "passed": True,
            "evidence": "本 lens 未发现新的矛盾。",
            "issues": [],
            "prior_issue_closures": closures,
        }
    )


def _resolved_prior_story_closures(request: ToolCallRequest) -> list[dict[str, str]]:
    previous = request.state.get("files", {}).get("/workspace/previous_story_review.json")
    if not isinstance(previous, Mapping):
        return []
    payload = json.loads(previous["content"])
    return [
        {
            "issue_id": entry["issue_id"],
            "status": "resolved",
            "evidence": "完整当前候选已按批准上游修正该冲突。",
        }
        for entry in payload["issue_ledger"]
    ]


def test_story_canon_missing_closure_cannot_pass() -> None:
    prior = _prior_story_review()

    merged = _merge_story_canon_reviews(
        [
            _passing_canon_review_with_closure(prior, status=None),
            _passing_canon_review_with_closure(prior, status=None),
        ],
        prior,
    )

    assert merged.passed is False
    assert [issue.code for issue in merged.issues] == ["relative_age_conflict"]
    assert "missing" in merged.evidence


def test_story_canon_one_resolved_and_one_missing_cannot_pass() -> None:
    prior = _prior_story_review()

    merged = _merge_story_canon_reviews(
        [
            _passing_canon_review_with_closure(prior, status="resolved"),
            _passing_canon_review_with_closure(prior, status=None),
        ],
        prior,
    )

    assert merged.passed is False
    assert [issue.code for issue in merged.issues] == ["relative_age_conflict"]


def test_story_canon_both_resolved_allow_prior_issue_to_close() -> None:
    prior = _prior_story_review()

    merged = _merge_story_canon_reviews(
        [
            _passing_canon_review_with_closure(prior, status="resolved"),
            _passing_canon_review_with_closure(prior, status="resolved"),
        ],
        prior,
    )

    assert merged.passed is True
    assert merged.issues == []


def test_story_canon_unresolved_closure_retains_prior_issue() -> None:
    prior = _prior_story_review()

    merged = _merge_story_canon_reviews(
        [
            _passing_canon_review_with_closure(prior, status="resolved"),
            _passing_canon_review_with_closure(prior, status="unresolved"),
        ],
        prior,
    )

    assert merged.passed is False
    assert [issue.code for issue in merged.issues] == ["relative_age_conflict"]
    assert "unresolved" in merged.evidence


@pytest.mark.asyncio
async def test_story_repair_allows_two_bounded_targeted_patch_corrections() -> None:
    corrections: list[str | None] = []
    content = (
        "- **林守诚（父亲）**：兼岛上唯一修表师。\n"
        "其余人物身份、关系、时间线与证物约束均保持不变。\n"
        "本段补充足够上下文，确保修复只触及冲突限定词而不重写整个故事工件。"
    )
    conflict_line = content.split("\n").index("- **林守诚（父亲）**：兼岛上唯一修表师。") + 1
    line_count = len(content.split("\n"))

    async def before_stage(_: InternalStage) -> int:
        return 1

    async def approve_stage(_: InternalStage, __: Mapping[str, Any]) -> None:
        return None

    async def generate_story_patch(
        stage: InternalStage,
        _: str,
        __: CanonReviewerResult,
        ___: int,
        correction: str | None,
    ) -> StoryArtifactRepairPatch:
        corrections.append(correction)
        line_number = conflict_line if len(corrections) == 3 else 99
        return StoryArtifactRepairPatch.model_validate(
            {
                "stage": stage.value,
                "line_replacements": [
                    {
                        "start_line": line_number,
                        "end_line": line_number,
                        "replacement": "- **林守诚（父亲）**：兼岛上修表师。",
                    }
                ],
            }
        )

    middleware = StageGuardMiddleware(
        before_stage=before_stage,
        approve_stage=approve_stage,
        approved_stages=set(),
        generate_story_patch=generate_story_patch,
    )
    review = CanonReviewerResult(
        passed=False,
        evidence="唯一修表师身份冲突。",
        issues=[
            {
                "code": "only_repairer_conflict",
                "message": "删除唯一限定词。",
                "script_excerpt": "林守诚（父亲）：兼岛上唯一修表师。",
            }
        ],
    )

    repaired = await middleware._invoke_story_artifact_repair(
        request=None,
        handler=None,
        stage=InternalStage.GENERATING_STORY_OUTLINE,
        content=content,
        review=review,
        repair_round=1,
    )

    assert len(corrections) == 3
    assert corrections[0] is None
    assert all(f"Candidate has {line_count} numbered lines" in value for value in corrections[1:])
    assert "唯一" not in repaired.content


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure",
    [
        TimeoutError("story repair timed out"),
        httpx.ReadTimeout(
            "relay timed out",
            request=httpx.Request("POST", "https://relay.example/v1/chat/completions"),
        ),
    ],
)
async def test_story_repair_preserves_transport_failures_without_retry(failure: Exception) -> None:
    calls = 0

    async def before_stage(_: InternalStage) -> int:
        return 1

    async def approve_stage(_: InternalStage, __: Mapping[str, Any]) -> None:
        return None

    async def unavailable_patch(
        _: InternalStage,
        __: str,
        ___: CanonReviewerResult,
        ____: int,
        _____: str | None,
    ) -> Any:
        nonlocal calls
        calls += 1
        raise failure

    middleware = StageGuardMiddleware(
        before_stage=before_stage,
        approve_stage=approve_stage,
        approved_stages=set(),
        generate_story_patch=unavailable_patch,
    )
    review = CanonReviewerResult(
        passed=False,
        evidence="需要修复。",
        issues=[{"code": "stale_text", "message": "替换旧文字。"}],
    )

    with pytest.raises(type(failure)) as caught:
        await middleware._invoke_story_artifact_repair(
            request=None,
            handler=None,
            stage=InternalStage.GENERATING_STORY_OUTLINE,
            content="原始故事大纲。",
            review=review,
            repair_round=1,
        )

    assert caught.value is failure
    assert calls == 1


@pytest.mark.asyncio
async def test_loop_relay_retry_survives_transient_interruption() -> None:
    """A transient relay interruption inside the review loop retries the same
    call instead of propagating and resetting the loop's repair_rounds."""
    from pengine.agents import _with_loop_relay_retry

    calls = 0

    async def flaky_call() -> str:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise httpx.ReadTimeout(
                "relay timed out",
                request=httpx.Request("POST", "https://relay.example/v1/chat/completions"),
            )
        return "recovered"

    # Patch the retry delay to zero so the test does not sleep.
    import pengine.agents as agents_module

    original = agents_module.retryable_relay_interruption
    agents_module.retryable_relay_interruption = lambda exc: type(
        "RRI", (), {"retry_delay_seconds": 0}
    )()
    try:
        result = await _with_loop_relay_retry(flaky_call)
    finally:
        agents_module.retryable_relay_interruption = original

    assert result == "recovered"
    assert calls == 2


@pytest.mark.asyncio
async def test_loop_relay_retry_propagates_after_max_retries() -> None:
    """After max_retries exhausted relay failures, the interruption propagates
    so the worker recovers the stage as before (no regression)."""
    from pengine.agents import _with_loop_relay_retry

    calls = 0

    async def always_fails() -> str:
        nonlocal calls
        calls += 1
        raise httpx.ReadTimeout(
            "relay timed out",
            request=httpx.Request("POST", "https://relay.example/v1/chat/completions"),
        )

    import pengine.agents as agents_module

    original = agents_module.retryable_relay_interruption
    agents_module.retryable_relay_interruption = lambda exc: type(
        "RRI", (), {"retry_delay_seconds": 0}
    )()
    try:
        with pytest.raises(httpx.ReadTimeout):
            await _with_loop_relay_retry(always_fails, max_retries=2)
    finally:
        agents_module.retryable_relay_interruption = original

    assert calls == 3


@pytest.mark.asyncio
async def test_loop_relay_retry_propagates_non_relay_errors_immediately() -> None:
    """A non-relay error (e.g. AgentProtocolError) propagates immediately
    without retrying, so structured-output failures are not masked."""
    from pengine.agents import _with_loop_relay_retry

    calls = 0

    async def protocol_error() -> str:
        nonlocal calls
        calls += 1
        raise ValueError("not a relay error")

    with pytest.raises(ValueError, match="not a relay error"):
        await _with_loop_relay_retry(protocol_error)

    assert calls == 1


def test_canonical_workspace_replaces_stale_story_files_and_preserves_scratch() -> None:
    request = ToolCallRequest(
        tool_call={"name": "task", "args": {}, "id": "canonical", "type": "tool_call"},
        tool=None,
        state={
            "files": {
                "/workspace/story_outline.md": {"content": "旧大纲", "encoding": "utf-8"},
                "/workspace/character_biographies.md": {
                    "content": "未批准的旧小传",
                    "encoding": "utf-8",
                },
                "/workspace/scratch.md": {"content": "保留", "encoding": "utf-8"},
            }
        },
        runtime=None,
    )
    approved = {
        InternalStage.SELECTING_L0_VARIANT: {
            "stage": "selecting_l0_variant",
            "selected_l0_variant": "主动选择",
            "selection_rationale": "契合创意",
            "content": None,
        },
        InternalStage.GENERATING_STORY_OUTLINE: {
            "stage": "generating_story_outline",
            "content": "已批准大纲",
            "character_biographies": None,
            "relationship_logic": None,
            "selected_l0_variant": None,
            "selection_rationale": None,
        },
        InternalStage.GENERATING_CHARACTER_RELATIONSHIPS: {
            "stage": "generating_character_relationships",
            "content": None,
            "character_biographies": "已批准小传",
            "relationship_logic": "已批准关系",
            "selected_l0_variant": None,
            "selection_rationale": None,
        },
    }

    normalized = _request_with_canonical_workspace(request, approved)
    files = normalized.state["files"]

    assert files["/workspace/story_outline.md"]["content"] == "已批准大纲"
    assert files["/workspace/character_biographies.md"]["content"] == "已批准小传"
    assert files["/workspace/relationship_logic.md"]["content"] == "已批准关系"
    assert files["/workspace/scratch.md"]["content"] == "保留"
    assert "已批准大纲" in files["/workspace/approved-checkpoints.json"]["content"]


def test_outline_repair_context_excludes_unrelated_contract_and_frozen_upstream() -> None:
    contract = _story_contract(episode_count=2).model_dump(mode="json")
    contract["prohibitions"].append("DO_NOT_SEND_" * 4_000)
    candidate = {
        "stage": "generating_episode_outline",
        "content": "第一集发现事实一。第二集发现事实二。",
        "episode_count": 2,
        "episodes": [
            {"episode_number": 1, "plan": "第一集计划"},
            {"episode_number": 2, "plan": "第二集计划"},
        ],
        "story_contract": contract,
    }
    review = CanonReviewerResult(
        passed=False,
        evidence="事实一被角色过早知晓。",
        issues=[
            {
                "code": "knowledge_overcommit",
                "message": "修正所有引用事实一的知识状态。",
                "contract_refs": ["fact_ep1", "knowledge_states"],
            }
        ],
    )

    context = _outline_repair_context(candidate, review)
    serialized_context = json.dumps(context, ensure_ascii=False, separators=(",", ":"))
    serialized_candidate = json.dumps(candidate, ensure_ascii=False, separators=(",", ":"))
    target_by_path = {item["path"]: item for item in context["contract_targets"]}

    assert "frozen_upstream" not in context
    assert "candidate" not in context
    assert "DO_NOT_SEND" not in serialized_context
    assert len(serialized_context) * 2 < len(serialized_candidate)
    assert context["matched_contract_refs"] == ["fact_ep1"]
    assert context["matched_collection_scopes"] == ["knowledge_states"]
    assert context["unmatched_contract_refs"] == []
    assert context["readable_outline"]["value"] == candidate["content"]
    assert [item["value"] for item in context["episode_plans"]] == candidate["episodes"]
    assert set(target_by_path) == {
        "/story_contract/facts/0",
        "/story_contract/knowledge_states/0",
        "/story_contract/knowledge_states/1",
    }
    assert target_by_path["/story_contract/facts/0"]["editable"] is False
    assert target_by_path["/story_contract/knowledge_states/0"]["editable"] is True


def test_outline_repair_context_exposes_fact_dependency_closure() -> None:
    contract = _story_contract(episode_count=2).model_dump(mode="json")
    candidate = {
        "stage": "generating_episode_outline",
        "content": "第一集只发现刻字，第二集确认事实一。",
        "episode_count": 2,
        "episodes": [
            {"episode_number": 1, "plan": "第一集计划"},
            {"episode_number": 2, "plan": "第二集计划"},
        ],
        "story_contract": contract,
    }
    review = CanonReviewerResult(
        passed=False,
        evidence="事实一被提前到第一集揭示。",
        issues=[
            {
                "code": "premature_reveal",
                "message": "fact_ep1 应改到第二集揭示并同步义务与知识状态。",
                "contract_refs": ["fact_ep1"],
            }
        ],
    )

    context = _outline_repair_context(candidate, review)
    target_by_path = {item["path"]: item for item in context["contract_targets"]}

    assert set(target_by_path) == {
        "/story_contract/facts/0",
        "/story_contract/timeline/0",
        "/story_contract/knowledge_states/0",
        "/story_contract/knowledge_states/1",
        "/story_contract/episode_obligations/0",
        "/story_contract/episode_obligations/1",
    }
    assert all(item["editable"] is True for item in target_by_path.values())


def test_outline_repair_patch_targets_only_exposed_editable_nodes() -> None:
    contract = _story_contract().model_dump(mode="json")
    contract["characters"][0]["initial_known_fact_ids"] = ["fact_ep1"]
    candidate = {
        "stage": "generating_episode_outline",
        "content": "第一集发现事实一。",
        "episode_count": 1,
        "episodes": [{"episode_number": 1, "plan": "第一集计划"}],
        "story_contract": contract,
    }
    review = CanonReviewerResult(
        passed=False,
        evidence="知识状态过度承诺。",
        issues=[
            {
                "code": "knowledge_overcommit",
                "message": "test_character 不应知道 fact_ep1。",
                "contract_refs": ["fact_ep1", "knowledge_states"],
            }
        ],
    )
    context = _outline_repair_context(candidate, review)
    target_by_path = {item["path"]: item for item in context["contract_targets"]}
    allowed = OutlineRepairPatch.model_validate(
        {
            "stage": "generating_episode_outline",
            "json_edits": [
                {
                    "op": "replace",
                    "path": "/story_contract/knowledge_states/0/known_fact_ids",
                    "expected": ["fact_ep1"],
                    "value": [],
                }
            ],
        }
    )
    forbidden = OutlineRepairPatch.model_validate(
        {
            "stage": "generating_episode_outline",
            "json_edits": [
                {
                    "op": "remove",
                    "path": "/story_contract/facts/0",
                    "expected": contract["facts"][0],
                    "value": None,
                }
            ],
        }
    )

    assert target_by_path["/story_contract/characters/0"]["editable"] is True
    _validate_outline_repair_patch_targets(allowed, context)
    with pytest.raises(ValueError, match="target_not_exposed"):
        _validate_outline_repair_patch_targets(forbidden, context)


def test_outline_repair_result_reports_output_truncation() -> None:
    raw = AIMessage(content="", response_metadata={"finish_reason": "length"})

    with pytest.raises(AgentProtocolError) as error:
        _outline_repair_result({"raw": raw, "parsed": None, "parsing_error": None})

    assert "token limit" in str(error.value)
    assert error.value.safe_message == "分集大纲修复补丁输出被模型截断。"


@pytest.mark.asyncio
async def test_structured_result_middleware_forces_result_after_prose() -> None:
    class Result(BaseModel):
        value: str

    calls: list[ModelRequest] = []

    async def handler(request: ModelRequest) -> ModelResponse:
        calls.append(request)
        if len(calls) == 1:
            return ModelResponse(
                result=[AIMessage(content="I finished the work in prose.")],
                structured_response=None,
            )
        return ModelResponse(
            result=[AIMessage(content="", tool_calls=[])],
            structured_response=Result(value="done"),
        )

    request = ModelRequest(
        model=FakeMessagesListChatModel(responses=[]),
        messages=[],
        tools=[{"type": "function", "function": {"name": "work"}}],
        response_format=ToolStrategy(Result),
    )

    response = await StructuredResultMiddleware().awrap_model_call(request, handler)

    assert response.structured_response == Result(value="done")
    assert len(calls) == 2
    assert calls[1].tools == []
    assert isinstance(calls[1].messages[-1], HumanMessage)
    assert "exactly one valid Result tool call" in calls[1].messages[-1].content


@pytest.mark.asyncio
async def test_structured_result_middleware_corrects_invalid_forced_result() -> None:
    class Result(BaseModel):
        value: str = Field(min_length=1)

    model = ToolCallingFakeModel(
        responses=[
            AIMessage(content="The work is complete in prose."),
            _tool_call("Result", {}, 1),
            _tool_call("Result", {"value": "done"}, 2),
        ]
    )
    work_tool = StructuredTool.from_function(
        lambda value: value,
        name="work",
        description="A working tool unavailable during structured correction.",
    )
    agent = create_agent(
        model,
        tools=[work_tool],
        middleware=[StructuredResultMiddleware()],
        response_format=ToolStrategy(Result),
    )

    result = await agent.ainvoke({"messages": [HumanMessage(content="Complete the task.")]})

    assert result["structured_response"] == Result(value="done")
    assert model.bound_tool_names[-3:] == [
        ["work", "Result"],
        ["Result"],
        ["Result"],
    ]


@pytest.mark.asyncio
async def test_structured_result_middleware_rejects_work_tool_after_forcing_result() -> None:
    class Result(BaseModel):
        value: str

    calls: list[ModelRequest] = []

    async def handler(request: ModelRequest) -> ModelResponse:
        calls.append(request)
        if len(calls) == 1:
            return ModelResponse(
                result=[AIMessage(content="I finished the work in prose.")],
                structured_response=None,
            )
        return ModelResponse(
            result=[
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "work",
                            "args": {"value": "hallucinated"},
                            "id": "work-after-force",
                            "type": "tool_call",
                        }
                    ],
                )
            ],
            structured_response=None,
        )

    request = ModelRequest(
        model=FakeMessagesListChatModel(responses=[]),
        messages=[],
        tools=[{"type": "function", "function": {"name": "work"}}],
        response_format=ToolStrategy(Result),
    )

    with pytest.raises(AgentProtocolError, match="invalid structured output"):
        await StructuredResultMiddleware().awrap_model_call(request, handler)

    assert len(calls) == 2
    assert calls[1].tools == []


@pytest.mark.asyncio
async def test_structured_result_middleware_removes_work_tools_after_schema_error() -> None:
    class Result(BaseModel):
        value: str

    error_message = ToolMessage(
        content="Correct the schema.",
        tool_call_id="call-9",
        name="Result",
    )
    request = ModelRequest(
        model=FakeMessagesListChatModel(responses=[]),
        messages=[
            _tool_call("Result", {"value": "invalid"}, 9),
            error_message,
        ],
        tools=[{"type": "function", "function": {"name": "work"}}],
        response_format=ToolStrategy(Result),
    )

    async def handler(candidate: ModelRequest) -> ModelResponse:
        assert candidate.tools == []
        return ModelResponse(
            result=[AIMessage(content="", tool_calls=[])],
            structured_response=Result(value="corrected"),
        )

    response = await StructuredResultMiddleware().awrap_model_call(request, handler)

    assert response.structured_response == Result(value="corrected")


@pytest.mark.asyncio
async def test_structured_result_middleware_reconstructs_safe_validation_details() -> None:
    valid_args = {
        "stage": "generating_episode_outline",
        "content": "分集大纲",
        "episode_count": 1,
        "episodes": [{"episode_number": 1, "plan": "第一集计划"}],
        "story_contract": _story_contract().model_dump(mode="json"),
    }
    invalid_args = copy.deepcopy(valid_args)
    invalid_args["story_contract"]["facts"][0]["unit"] = "次"
    invalid_message = _tool_call("EpisodePlannerResult", invalid_args, 77)
    generic_error = ToolMessage(
        content=(
            "Return exactly one valid structured result tool call for the requested stage. "
            "Do not return the result as prose."
        ),
        tool_call_id="call-77",
        name="EpisodePlannerResult",
    )
    response_format = ToolStrategy(EpisodePlannerResult)
    correction = _structured_result_validation_correction(
        response_format,
        [invalid_message, generic_error],
    )

    assert correction is not None
    assert "Non-numeric facts cannot declare a unit" in correction.content

    request = ModelRequest(
        model=FakeMessagesListChatModel(responses=[]),
        messages=[invalid_message, generic_error],
        tools=[{"type": "function", "function": {"name": "work"}}],
        response_format=response_format,
    )

    async def handler(candidate: ModelRequest) -> ModelResponse:
        assert candidate.tools == []
        assert isinstance(candidate.messages[-1], HumanMessage)
        assert "Non-numeric facts cannot declare a unit" in candidate.messages[-1].content
        return ModelResponse(
            result=[AIMessage(content="", tool_calls=[])],
            structured_response=EpisodePlannerResult.model_validate(valid_args),
        )

    response = await StructuredResultMiddleware().awrap_model_call(request, handler)

    assert isinstance(response.structured_response, EpisodePlannerResult)


@pytest.mark.asyncio
async def test_structured_result_middleware_preserves_validation_details_after_prose() -> None:
    valid_args = {
        "stage": "generating_episode_outline",
        "content": "分集大纲",
        "episode_count": 1,
        "episodes": [{"episode_number": 1, "plan": "第一集计划"}],
        "story_contract": _story_contract().model_dump(mode="json"),
    }
    invalid_args = copy.deepcopy(valid_args)
    invalid_args["story_contract"]["facts"][0]["unit"] = "次"
    request = ModelRequest(
        model=FakeMessagesListChatModel(responses=[]),
        messages=[
            _tool_call("EpisodePlannerResult", invalid_args, 78),
            ToolMessage(
                content="Return a valid structured result.",
                tool_call_id="call-78",
                name="EpisodePlannerResult",
            ),
        ],
        tools=[{"type": "function", "function": {"name": "work"}}],
        response_format=ToolStrategy(EpisodePlannerResult),
    )
    calls: list[ModelRequest] = []

    async def handler(candidate: ModelRequest) -> ModelResponse:
        calls.append(candidate)
        if len(calls) == 1:
            assert "Non-numeric facts cannot declare a unit" in candidate.messages[-1].content
            return ModelResponse(
                result=[AIMessage(content="I corrected it in prose.")],
                structured_response=None,
            )
        assert any(
            isinstance(message, HumanMessage)
            and "Non-numeric facts cannot declare a unit" in message.content
            for message in candidate.messages
        )
        assert any(
            isinstance(message, AIMessage) and message.content == "I corrected it in prose."
            for message in candidate.messages
        )
        return ModelResponse(
            result=[AIMessage(content="", tool_calls=[])],
            structured_response=EpisodePlannerResult.model_validate(valid_args),
        )

    response = await StructuredResultMiddleware().awrap_model_call(request, handler)

    assert isinstance(response.structured_response, EpisodePlannerResult)
    assert len(calls) == 2


@pytest.mark.asyncio
async def test_structured_result_middleware_allows_layered_schema_corrections() -> None:
    class Result(BaseModel):
        value: str

    messages: list[AIMessage | ToolMessage] = []
    for index in range(2):
        messages.extend(
            [
                _tool_call("Result", {"value": "invalid"}, index),
                ToolMessage(
                    content=f"Correct schema layer {index + 1}.",
                    tool_call_id=f"call-{index}",
                    name="Result",
                ),
            ]
        )
    request = ModelRequest(
        model=FakeMessagesListChatModel(responses=[]),
        messages=messages,
        tools=[{"type": "function", "function": {"name": "work"}}],
        response_format=ToolStrategy(Result),
    )

    async def handler(candidate: ModelRequest) -> ModelResponse:
        assert candidate.tools == []
        return ModelResponse(
            result=[AIMessage(content="", tool_calls=[])],
            structured_response=Result(value="corrected"),
        )

    response = await StructuredResultMiddleware().awrap_model_call(request, handler)

    assert response.structured_response == Result(value="corrected")


@pytest.mark.asyncio
async def test_structured_result_middleware_budgets_errors_by_assistant_turn() -> None:
    class Result(BaseModel):
        value: int

    first_turn = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "Result",
                "args": {"value": "invalid-a"},
                "id": "first-a",
                "type": "tool_call",
            },
            {
                "name": "Result",
                "args": {"value": "invalid-b"},
                "id": "first-b",
                "type": "tool_call",
            },
        ],
    )
    second_turn = _tool_call("Result", {"value": "invalid-c"}, 3)
    messages: list[AIMessage | ToolMessage] = [
        first_turn,
        ToolMessage(content="Correct schema.", tool_call_id="first-a", name="Result"),
        ToolMessage(content="Correct schema.", tool_call_id="first-b", name="Result"),
        second_turn,
        ToolMessage(content="Correct schema.", tool_call_id="call-3", name="Result"),
    ]
    request = ModelRequest(
        model=FakeMessagesListChatModel(responses=[]),
        messages=messages,
        tools=[{"type": "function", "function": {"name": "work"}}],
        response_format=ToolStrategy(Result),
    )

    async def handler(candidate: ModelRequest) -> ModelResponse:
        assert candidate.tools == []
        return ModelResponse(
            result=[AIMessage(content="", tool_calls=[])],
            structured_response=Result(value=7),
        )

    response = await StructuredResultMiddleware().awrap_model_call(request, handler)

    assert response.structured_response == Result(value=7)


@pytest.mark.asyncio
async def test_structured_result_middleware_stops_after_three_schema_errors() -> None:
    class Result(BaseModel):
        value: str

    messages: list[AIMessage | ToolMessage] = []
    for index in range(3):
        messages.extend(
            [
                _tool_call("Result", {"value": "invalid"}, index),
                ToolMessage(
                    content=f"Correct schema layer {index + 1}.",
                    tool_call_id=f"call-{index}",
                    name="Result",
                ),
            ]
        )
    request = ModelRequest(
        model=FakeMessagesListChatModel(responses=[]),
        messages=messages,
        tools=[],
        response_format=ToolStrategy(Result),
    )

    async def handler(_: ModelRequest) -> ModelResponse:
        raise AssertionError("The exhausted request must not call the model again")

    with pytest.raises(AgentProtocolError, match="invalid structured output") as error:
        await StructuredResultMiddleware().awrap_model_call(request, handler)

    assert error.value.safe_message == "模型未返回有效的结构化结果。"


@pytest.mark.asyncio
async def test_structured_result_middleware_stops_repeated_work_tool_loop() -> None:
    class Result(BaseModel):
        value: str

    messages: list[AIMessage | ToolMessage] = []
    for index in range(3):
        tool_call_id = f"work-{index}"
        messages.extend(
            [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "write_todos",
                            "args": {"todos": [{"content": "完成大纲", "status": "pending"}]},
                            "id": tool_call_id,
                            "type": "tool_call",
                        }
                    ],
                ),
                ToolMessage(
                    content="Todo list updated.",
                    tool_call_id=tool_call_id,
                    name="write_todos",
                ),
            ]
        )
    request = ModelRequest(
        model=FakeMessagesListChatModel(responses=[]),
        messages=messages,
        tools=[{"type": "function", "function": {"name": "write_todos"}}],
        response_format=ToolStrategy(Result),
    )

    async def handler(candidate: ModelRequest) -> ModelResponse:
        assert candidate.tools == []
        assert isinstance(candidate.messages[-1], HumanMessage)
        assert "repeated without progress" in candidate.messages[-1].content
        return ModelResponse(
            result=[AIMessage(content="", tool_calls=[])],
            structured_response=Result(value="done"),
        )

    response = await StructuredResultMiddleware().awrap_model_call(request, handler)

    assert response.structured_response == Result(value="done")


@pytest.mark.asyncio
async def test_structured_result_middleware_stops_alternating_work_tool_loop() -> None:
    class Result(BaseModel):
        value: str

    messages: list[AIMessage | ToolMessage] = []
    for index, left in enumerate(("9", "26", "9", "26", "9")):
        tool_call_id = f"arithmetic-{index}"
        messages.extend(
            [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "calculate_arithmetic",
                            "args": {"left": left, "operation": "add", "right": "1"},
                            "id": tool_call_id,
                            "type": "tool_call",
                        }
                    ],
                ),
                ToolMessage(
                    content=str(int(left) + 1),
                    tool_call_id=tool_call_id,
                    name="calculate_arithmetic",
                ),
            ]
        )
    request = ModelRequest(
        model=FakeMessagesListChatModel(responses=[]),
        messages=messages,
        tools=[{"type": "function", "function": {"name": "calculate_arithmetic"}}],
        response_format=ToolStrategy(Result),
    )

    async def handler(candidate: ModelRequest) -> ModelResponse:
        assert candidate.tools == []
        assert isinstance(candidate.messages[-1], HumanMessage)
        assert "working-tool loop was detected" in candidate.messages[-1].content
        return ModelResponse(
            result=[AIMessage(content="", tool_calls=[])],
            structured_response=Result(value="done"),
        )

    response = await StructuredResultMiddleware().awrap_model_call(request, handler)

    assert response.structured_response == Result(value="done")


@pytest.mark.asyncio
async def test_structured_result_middleware_stops_after_work_tool_turn_budget() -> None:
    class Result(BaseModel):
        value: str

    messages: list[AIMessage | ToolMessage] = []
    for index in range(24):
        messages.append(
            _tool_call("read_file", {"file_path": f"/workspace/review-{index}.md"}, index)
        )
        messages.append(ToolMessage(content="ok", tool_call_id=f"call-{index}", name="read_file"))
    request = ModelRequest(
        model=FakeMessagesListChatModel(responses=[]),
        messages=messages,
        tools=[{"type": "function", "function": {"name": "read_file"}}],
        response_format=ToolStrategy(Result),
    )

    async def handler(candidate: ModelRequest) -> ModelResponse:
        assert candidate.tools == []
        assert isinstance(candidate.messages[-1], HumanMessage)
        assert "working-tool turn budget is exhausted" in candidate.messages[-1].content
        return ModelResponse(
            result=[AIMessage(content="", tool_calls=[])],
            structured_response=Result(value="done"),
        )

    response = await StructuredResultMiddleware().awrap_model_call(request, handler)

    assert response.structured_response == Result(value="done")


@pytest.mark.asyncio
async def test_structured_result_middleware_allows_normal_multi_tool_review() -> None:
    class Result(BaseModel):
        value: str

    messages: list = []
    for turn in range(8):
        calls = [
            {
                "name": "read_file",
                "args": {"file_path": f"/workspace/review-{turn}-{call}.md"},
                "id": f"review-{turn}-{call}",
                "type": "tool_call",
            }
            for call in range(3)
        ]
        messages.append(AIMessage(content="", tool_calls=calls))
        messages.extend(
            ToolMessage(content="ok", tool_call_id=f"review-{turn}-{call}") for call in range(3)
        )
    request = ModelRequest(
        model=FakeMessagesListChatModel(responses=[]),
        messages=messages,
        tools=[{"type": "function", "function": {"name": "read_file"}}],
        response_format=ToolStrategy(Result),
    )

    async def handler(candidate: ModelRequest) -> ModelResponse:
        assert candidate.tools == request.tools
        assert candidate.messages == request.messages
        return ModelResponse(
            result=[AIMessage(content="", tool_calls=[])],
            structured_response=Result(value="done"),
        )

    response = await StructuredResultMiddleware().awrap_model_call(request, handler)

    assert response.structured_response == Result(value="done")


@pytest.mark.asyncio
async def test_structured_result_middleware_keeps_only_langchain_result_tool() -> None:
    class Result(BaseModel):
        value: str

    model = ToolCallingFakeModel(
        responses=[
            AIMessage(content="The work is complete in prose."),
            _tool_call("Result", {"value": "done"}, 1),
        ]
    )
    work_tool = StructuredTool.from_function(
        lambda value: value,
        name="work",
        description="A working tool that must not be available during forced completion.",
    )
    agent = create_agent(
        model,
        tools=[work_tool],
        middleware=[StructuredResultMiddleware()],
        response_format=ToolStrategy(Result),
    )

    result = await agent.ainvoke({"messages": [HumanMessage(content="Complete the task.")]})

    assert result["structured_response"] == Result(value="done")
    assert model.bound_tool_names[-2] == ["work", "Result"]
    assert model.bound_tool_names[-1] == ["Result"]


@pytest.mark.asyncio
async def test_structured_result_middleware_recovers_truncated_output() -> None:
    class Result(BaseModel):
        value: str

    calls: list[ModelRequest] = []

    async def handler(request: ModelRequest) -> ModelResponse:
        calls.append(request)
        if len(calls) == 1:
            return ModelResponse(
                result=[
                    AIMessage(
                        content="The review evidence is long and continues past the limit...",
                        response_metadata={"finish_reason": "length"},
                    )
                ],
                structured_response=None,
            )
        return ModelResponse(
            result=[AIMessage(content="", tool_calls=[])],
            structured_response=Result(value="done"),
        )

    request = ModelRequest(
        model=FakeMessagesListChatModel(responses=[]),
        messages=[],
        tools=[{"type": "function", "function": {"name": "work"}}],
        response_format=ToolStrategy(Result),
    )

    response = await StructuredResultMiddleware().awrap_model_call(request, handler)

    assert response.structured_response == Result(value="done")
    assert len(calls) == 2
    assert calls[1].tools == []
    assert isinstance(calls[1].messages[-1], HumanMessage)
    assert "truncated" in calls[1].messages[-1].content
    assert "Result tool call" in calls[1].messages[-1].content


@pytest.mark.asyncio
async def test_structured_result_middleware_classifies_persistent_truncation() -> None:
    class Result(BaseModel):
        value: str

    calls: list[ModelRequest] = []

    async def handler(request: ModelRequest) -> ModelResponse:
        calls.append(request)
        return ModelResponse(
            result=[
                AIMessage(
                    content="Truncated review evidence that never fits within the limit...",
                    response_metadata={"finish_reason": "length"},
                )
            ],
            structured_response=None,
        )

    request = ModelRequest(
        model=FakeMessagesListChatModel(responses=[]),
        messages=[],
        tools=[{"type": "function", "function": {"name": "work"}}],
        response_format=ToolStrategy(Result),
    )

    with pytest.raises(AgentProtocolError) as error:
        await StructuredResultMiddleware().awrap_model_call(request, handler)

    assert "truncated" in str(error.value)
    assert error.value.safe_message == "结构化评审输出被模型截断。"
    assert error.value.repair_instruction is not None


@pytest.mark.asyncio
async def test_structured_result_middleware_recovers_truncated_tool_call() -> None:
    class Result(BaseModel):
        value: str

    calls: list[ModelRequest] = []

    async def handler(request: ModelRequest) -> ModelResponse:
        calls.append(request)
        if len(calls) == 1:
            return ModelResponse(
                result=[
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "Result",
                                "args": {"value": "trunc"},
                                "id": "truncated-result",
                                "type": "tool_call",
                            }
                        ],
                        response_metadata={"finish_reason": "length"},
                    )
                ],
                structured_response=None,
            )
        return ModelResponse(
            result=[AIMessage(content="", tool_calls=[])],
            structured_response=Result(value="done"),
        )

    request = ModelRequest(
        model=FakeMessagesListChatModel(responses=[]),
        messages=[],
        tools=[{"type": "function", "function": {"name": "work"}}],
        response_format=ToolStrategy(Result),
    )

    response = await StructuredResultMiddleware().awrap_model_call(request, handler)

    assert response.structured_response == Result(value="done")
    assert len(calls) == 2
    assert calls[1].tools == []
    assert isinstance(calls[1].messages[-1], HumanMessage)
    assert "truncated" in calls[1].messages[-1].content
    assert not any(
        isinstance(message, AIMessage) and message.tool_calls for message in calls[1].messages
    )


def test_drop_dangling_tool_call_messages_cleans_truncated_history() -> None:
    messages = [
        HumanMessage(content="start"),
        AIMessage(
            content="",
            tool_calls=[
                {"name": "work", "args": {}, "id": "call-1", "type": "tool_call"},
                {
                    "name": "Result",
                    "args": {"value": "trunc"},
                    "id": "call-2",
                    "type": "tool_call",
                },
            ],
        ),
        ToolMessage(content="ok", tool_call_id="call-1"),
        HumanMessage(content="next"),
    ]
    cleaned = _drop_dangling_tool_call_messages(messages)
    # call-2 was never answered by a ToolMessage (truncated tool call): the
    # assistant message that declared it must be dropped so the next provider
    # request never carries a dangling tool_calls message.
    assert cleaned == [messages[0], messages[3]]
    assert not any(isinstance(message, AIMessage) and message.tool_calls for message in cleaned)


def test_drop_dangling_tool_call_messages_keeps_answered_calls() -> None:
    messages = [
        HumanMessage(content="start"),
        AIMessage(
            content="",
            tool_calls=[
                {"name": "work", "args": {}, "id": "call-1", "type": "tool_call"},
            ],
        ),
        ToolMessage(content="ok", tool_call_id="call-1"),
        HumanMessage(content="done"),
    ]
    cleaned = _drop_dangling_tool_call_messages(messages)
    assert cleaned == messages


def test_drop_dangling_tool_call_messages_removes_orphaned_tool_messages() -> None:
    messages = [
        HumanMessage(content="start"),
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "Result",
                    "args": {"value": "trunc"},
                    "id": "call-x",
                    "type": "tool_call",
                },
            ],
        ),
        ToolMessage(content="orphan", tool_call_id="call-y"),
    ]
    cleaned = _drop_dangling_tool_call_messages(messages)
    # call-x was truncated (no ToolMessage answered it) and call-y has no
    # declaring assistant tool call: both are removed from the transcript.
    assert cleaned == [messages[0]]


def test_drop_dangling_tool_call_messages_cleans_invalid_tool_calls() -> None:
    """A malformed tool call classified as invalid_tool_calls (not truncation)
    dangles identically: langchain-openai re-serializes it as an assistant
    tool_calls entry on the next request, but no ToolMessage answers it, so the
    provider rejects with HTTP 400 "insufficient tool messages following
    tool_calls message" (Issue #52 graph revision 12; E2E
    20260805T042536Z-1180ba17)."""
    messages = [
        HumanMessage(content="review the outline"),
        AIMessage(
            content="",
            tool_calls=[],
            invalid_tool_calls=[
                {
                    "name": "CanonReviewerResult",
                    "args": '{"broken": json}',  # malformed -> classified invalid
                    "id": "invalid-1",
                    "type": "invalid_tool_call",
                },
            ],
        ),
        HumanMessage(content="retry"),
    ]
    cleaned = _drop_dangling_tool_call_messages(messages)
    # The AIMessage carrying only an invalid_tool_call is dropped so the next
    # provider request never carries a dangling tool_calls entry.
    assert cleaned == [messages[0], messages[2]]
    assert not any(
        isinstance(message, AIMessage)
        and (message.tool_calls or getattr(message, "invalid_tool_calls", []))
        for message in cleaned
    )


def test_drop_dangling_tool_call_messages_cleans_mixed_valid_and_invalid_calls() -> None:
    """When an AIMessage carries both a answered valid call and an unanswered
    invalid call, the whole message is dropped (partial keep would re-serialize
    the invalid call)."""
    messages = [
        HumanMessage(content="start"),
        AIMessage(
            content="",
            tool_calls=[
                {"name": "work", "args": {}, "id": "valid-1", "type": "tool_call"},
            ],
            invalid_tool_calls=[
                {
                    "name": "Result",
                    "args": "{bad",
                    "id": "invalid-2",
                    "type": "invalid_tool_call",
                },
            ],
        ),
        ToolMessage(content="ok", tool_call_id="valid-1"),
        HumanMessage(content="next"),
    ]
    cleaned = _drop_dangling_tool_call_messages(messages)
    # invalid-2 is unanswered -> the AIMessage dangles -> dropped along with
    # its orphaned valid-1 ToolMessage.
    assert cleaned == [messages[0], messages[3]]


@pytest.mark.asyncio
async def test_structured_result_middleware_cleans_dangling_history_before_retry() -> None:
    class Result(BaseModel):
        value: str

    calls: list[ModelRequest] = []

    async def handler(request: ModelRequest) -> ModelResponse:
        calls.append(request)
        return ModelResponse(
            result=[AIMessage(content="", tool_calls=[])],
            structured_response=Result(value="done"),
        )

    request = ModelRequest(
        model=FakeMessagesListChatModel(responses=[]),
        messages=[
            HumanMessage(content="start"),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "work",
                        "args": {"expression": "1+1"},
                        "id": "call-stale",
                        "type": "tool_call",
                    },
                ],
            ),
            HumanMessage(content="continue"),
        ],
        tools=[{"type": "function", "function": {"name": "work"}}],
        response_format=ToolStrategy(Result),
    )

    response = await StructuredResultMiddleware().awrap_model_call(request, handler)

    assert response.structured_response == Result(value="done")
    assert len(calls) == 1
    assert not any(
        isinstance(message, AIMessage) and message.tool_calls for message in calls[0].messages
    )
    assert calls[0].messages[0] == request.messages[0]
    assert calls[0].messages[-1] == request.messages[-1]


def test_outline_repair_patch_applies_only_guarded_minimal_edits() -> None:
    contract = _story_contract()
    candidate = {
        "stage": "generating_episode_outline",
        "content": "旧分集大纲。其余内容保持。",
        "episode_count": 1,
        "episodes": [{"episode_number": 1, "plan": "第一集计划"}],
        "story_contract": contract.model_dump(mode="json"),
    }
    patch = OutlineRepairPatch.model_validate(
        {
            "stage": "generating_episode_outline",
            "content_replacements": [{"old": "旧分集大纲", "new": "修复后的分集大纲"}],
            "json_edits": [
                {
                    "op": "replace",
                    "path": "/story_contract/characters/0/role",
                    "expected": "主角",
                    "value": "调查真相的主角",
                }
            ],
        }
    )

    repaired = _apply_outline_repair_patch(
        candidate,
        patch,
        output_language=SIMPLIFIED_CHINESE,
    )

    assert repaired.content == "修复后的分集大纲。其余内容保持。"
    assert repaired.story_contract.characters[0].role == "调查真相的主角"
    assert candidate["content"] == "旧分集大纲。其余内容保持。"
    assert candidate["story_contract"]["characters"][0]["role"] == "主角"


def test_outline_repair_patch_rejects_a_stale_expected_value() -> None:
    contract = _story_contract()
    candidate = {
        "stage": "generating_episode_outline",
        "content": "分集大纲",
        "episode_count": 1,
        "episodes": [{"episode_number": 1, "plan": "第一集计划"}],
        "story_contract": contract.model_dump(mode="json"),
    }
    patch = OutlineRepairPatch.model_validate(
        {
            "stage": "generating_episode_outline",
            "json_edits": [
                {
                    "op": "replace",
                    "path": "/story_contract/characters/0/role",
                    "expected": "错误旧值",
                    "value": "调查真相的主角",
                }
            ],
        }
    )

    with pytest.raises(ValueError, match="patch_target_mismatch"):
        _apply_outline_repair_patch(candidate, patch)


def test_outline_repair_patch_preserves_exact_replacement_whitespace() -> None:
    contract = _story_contract()
    candidate = {
        "stage": "generating_episode_outline",
        "content": "开头\n  原句  \n结尾",
        "episode_count": 1,
        "episodes": [{"episode_number": 1, "plan": "第一集计划"}],
        "story_contract": contract.model_dump(mode="json"),
    }
    patch = OutlineRepairPatch.model_validate(
        {
            "stage": "generating_episode_outline",
            "content_replacements": [{"old": "  原句  ", "new": "  新句  "}],
        }
    )

    repaired = _apply_outline_repair_patch(candidate, patch)

    assert repaired.content == "开头\n  新句  \n结尾"


def test_outline_repair_patch_supports_guarded_episode_structure_edits() -> None:
    contract = _story_contract(episode_count=2)
    candidate = {
        "stage": "generating_episode_outline",
        "content": "两集分集大纲",
        "episode_count": 1,
        "episodes": [{"episode_number": 1, "plan": "第一集计划"}],
        "story_contract": contract.model_dump(mode="json"),
    }
    patch = OutlineRepairPatch.model_validate(
        {
            "stage": "generating_episode_outline",
            "json_edits": [
                {
                    "op": "replace",
                    "path": "/episode_count",
                    "expected": 1,
                    "value": 2,
                },
                {
                    "op": "add",
                    "path": "/episodes/-",
                    "expected": 1,
                    "value": {"episode_number": 2, "plan": "第二集计划"},
                },
            ],
        }
    )

    repaired = _apply_outline_repair_patch(candidate, patch)

    assert repaired.episode_count == 2
    assert [episode.episode_number for episode in repaired.episodes] == [1, 2]


def test_outline_repair_patch_rejects_boolean_list_length_guard() -> None:
    contract = _story_contract()
    candidate = {
        "stage": "generating_episode_outline",
        "content": "分集大纲",
        "episode_count": 1,
        "episodes": [{"episode_number": 1, "plan": "第一集计划"}],
        "story_contract": contract.model_dump(mode="json"),
    }
    patch = OutlineRepairPatch.model_validate(
        {
            "stage": "generating_episode_outline",
            "json_edits": [
                {
                    "op": "add",
                    "path": "/episodes/-",
                    "expected": True,
                    "value": {"episode_number": 2, "plan": "第二集计划"},
                }
            ],
        }
    )

    with pytest.raises(ValueError, match="patch_target_mismatch"):
        _apply_outline_repair_patch(candidate, patch)


def test_outline_repair_patch_rejects_oversized_payload() -> None:
    replacements = [
        {"old": f"旧{index}" * 1_500, "new": f"新{index}" * 1_500} for index in range(3)
    ]

    with pytest.raises(ValidationError, match="16000"):
        OutlineRepairPatch.model_validate(
            {
                "stage": "generating_episode_outline",
                "content_replacements": replacements,
            }
        )


def test_outline_repair_patch_must_be_less_than_half_the_candidate() -> None:
    contract = _story_contract()
    candidate = {
        "stage": "generating_episode_outline",
        "content": "分集大纲",
        "episode_count": 1,
        "episodes": [{"episode_number": 1, "plan": "第一集计划"}],
        "story_contract": contract.model_dump(mode="json"),
    }
    patch = OutlineRepairPatch.model_validate(
        {
            "stage": "generating_episode_outline",
            "content_replacements": [{"old": "分集大纲", "new": "修" * 2_000}],
        }
    )

    with pytest.raises(ValueError, match="outline_repair_patch_not_minimal"):
        _apply_outline_repair_patch(candidate, patch)


def test_outline_repair_patch_is_atomic_when_a_later_edit_fails() -> None:
    contract = _story_contract()
    candidate = {
        "stage": "generating_episode_outline",
        "content": "分集大纲",
        "episode_count": 1,
        "episodes": [{"episode_number": 1, "plan": "第一集计划"}],
        "story_contract": contract.model_dump(mode="json"),
    }
    original = copy.deepcopy(candidate)
    patch = OutlineRepairPatch.model_validate(
        {
            "stage": "generating_episode_outline",
            "json_edits": [
                {
                    "op": "replace",
                    "path": "/episodes/0/plan",
                    "expected": "第一集计划",
                    "value": "修复后的第一集计划",
                },
                {
                    "op": "replace",
                    "path": "/story_contract/characters/0/role",
                    "expected": "错误旧值",
                    "value": "调查真相的主角",
                },
            ],
        }
    )

    with pytest.raises(ValueError, match="patch_target_mismatch"):
        _apply_outline_repair_patch(candidate, patch)

    assert candidate == original


def test_outline_repair_patch_rejects_ambiguous_text_replacement() -> None:
    contract = _story_contract()
    candidate = {
        "stage": "generating_episode_outline",
        "content": "重复句。重复句。",
        "episode_count": 1,
        "episodes": [{"episode_number": 1, "plan": "第一集计划"}],
        "story_contract": contract.model_dump(mode="json"),
    }
    patch = OutlineRepairPatch.model_validate(
        {
            "stage": "generating_episode_outline",
            "content_replacements": [{"old": "重复句", "new": "修复句"}],
        }
    )

    with pytest.raises(ValueError, match="ambiguous_content_replacement"):
        _apply_outline_repair_patch(candidate, patch)


def test_outline_repair_patch_rejects_invalid_full_episode_sequence() -> None:
    contract = _story_contract()
    candidate = {
        "stage": "generating_episode_outline",
        "content": "分集大纲",
        "episode_count": 1,
        "episodes": [{"episode_number": 1, "plan": "第一集计划"}],
        "story_contract": contract.model_dump(mode="json"),
    }
    patch = OutlineRepairPatch.model_validate(
        {
            "stage": "generating_episode_outline",
            "json_edits": [
                {
                    "op": "replace",
                    "path": "/episodes/0/episode_number",
                    "expected": 1,
                    "value": 2,
                }
            ],
        }
    )

    with pytest.raises(ValidationError, match="ordered and contiguous"):
        _apply_outline_repair_patch(candidate, patch)


def test_outline_repair_patch_rejects_wrong_output_language() -> None:
    contract = _story_contract()
    candidate = {
        "stage": "generating_episode_outline",
        "content": "分集大纲",
        "episode_count": 1,
        "episodes": [{"episode_number": 1, "plan": "第一集计划"}],
        "story_contract": contract.model_dump(mode="json"),
    }
    patch = OutlineRepairPatch.model_validate(
        {
            "stage": "generating_episode_outline",
            "content_replacements": [{"old": "分集大纲", "new": "English outline"}],
        }
    )

    with pytest.raises(AgentProtocolError, match="invalid structured output"):
        _apply_outline_repair_patch(
            candidate,
            patch,
            output_language=SIMPLIFIED_CHINESE,
        )


def test_workflow_completion_does_not_repeat_approved_content() -> None:
    schema = WorkflowCompletion.model_json_schema()

    assert set(schema["properties"]) == {"completed"}
    assert schema["properties"]["completed"]["const"] is True


def test_supervisor_preserves_persona_episode_baseline_when_request_omits_count() -> None:
    prompt = _supervisor_prompt(
        story="故事",
        requirements="按人格设定完成完整交付。",
        feedback=None,
        approved_json="{}",
    )

    normalized = " ".join(prompt.split())
    assert "active persona L4 baseline is authoritative" in normalized
    assert "Do not invent a different episode count" in normalized


def test_supervisor_carries_the_inferred_language_contract() -> None:
    contract = language_instruction(SIMPLIFIED_CHINESE)

    prompt = _supervisor_prompt(
        story="一个海边故事",
        requirements="十集",
        feedback=None,
        approved_json="{}",
        language_contract=contract,
    )

    assert contract in prompt
    assert "Output language contract" in prompt


def test_structured_output_retry_reports_safe_validation_details() -> None:
    try:
        StoryArchitectResult.model_validate(
            {
                "stage": "selecting_l0_variant",
                "selected_l0_variant": None,
                "selection_rationale": None,
            }
        )
    except ValidationError as error:
        message = _structured_output_retry_message(error)
    else:
        raise AssertionError("The invalid story result unexpectedly validated")

    assert "Correct these validation errors" in message
    assert "L0 selection requires only variant and rationale" in message
    assert "input_value" not in message


def test_structured_output_retry_reports_safe_temporal_format_errors() -> None:
    invalid = _story_contract().model_dump(mode="json")
    invalid["facts"][0].update(kind="date", value="第0天", unit=None)

    try:
        StoryContract.model_validate(invalid)
    except ValidationError as error:
        message = _structured_output_retry_message(error)
    else:
        raise AssertionError("The invalid temporal fact unexpectedly validated")

    assert "facts.0: Invalid date value" in message
    assert "第0天" not in message


def test_structured_output_retry_reports_duplicate_evidence_targets_safely() -> None:
    contract = _story_contract()
    invalid = _state_delta(contract, 1)
    invalid["evidence"].append(dict(invalid["evidence"][0]))

    try:
        EpisodeStateDelta.model_validate(invalid)
    except ValidationError as error:
        message = _structured_output_retry_message(error)
    else:
        raise AssertionError("The duplicate evidence targets unexpectedly validated")

    assert "Episode evidence target IDs must be unique" in message
    assert "事实1" not in message


def test_structured_output_retry_does_not_echo_model_input_or_custom_error_values() -> None:
    secret = "SECRET-API-KEY"

    class LeakyResult(BaseModel):
        token: str

        @field_validator("token")
        @classmethod
        def reject_token(cls, value: str) -> str:
            raise ValueError(f"forbidden value {value}")

    try:
        LeakyResult.model_validate({"token": secret})
    except ValidationError as source:
        error = StructuredOutputValidationError(
            "LeakyResult",
            source,
            AIMessage(content=f"relay https://secret.example/?key={secret}"),
        )
    else:
        raise AssertionError("The leaky result unexpectedly validated")

    message = _structured_output_retry_message(error)

    assert "field: value_error" in message
    assert secret not in message
    assert "secret.example" not in message


def test_generation_prompts_require_cross_artifact_consistency() -> None:
    assert "future dialogue counts" in _STORY_ARCHITECT_PROMPT
    assert "Never append an English translation" in _STORY_ARCHITECT_PROMPT
    assert "episode-specific action" in _EPISODE_PLANNER_PROMPT
    assert "dates, countdowns, amounts, counts, and arithmetic" in _EPISODE_PLANNER_PROMPT
    assert "/workspace/approved-checkpoints.json" in _EPISODE_PLANNER_PROMPT
    assert "new_information_fact_ids must exactly equal" in _EPISODE_PLANNER_PROMPT
    assert "free-form user request" in _EPISODE_PLANNER_PROMPT
    assert "Never ask the user" in _EPISODE_PLANNER_PROMPT
    assert "Leave genuinely unspecified details out" in _EPISODE_PLANNER_PROMPT
    assert "preserve only the explicitly locked or formally committed facts" in (
        _STORY_ARCHITECT_PROMPT
    )
    assert "Capture explicitly locked or formally committed aliases" in _EPISODE_PLANNER_PROMPT
    assert "Knowledge states are sparse cumulative snapshots" in _EPISODE_PLANNER_PROMPT
    assert "aliases, pronouns, ages, elapsed durations, call participants" in (
        _EPISODE_PLANNER_PROMPT
    )
    assert "exact dialogue-count claims" in _SCRIPT_WRITER_PROMPT
    assert "Every explicitly locked upstream commitment must appear" in _SCRIPT_WRITER_PROMPT
    assert "Unspecified creative details remain the writer's choice" in _SCRIPT_WRITER_PROMPT
    assert "explicitly locked or formally committed aliases" in _SCRIPT_WRITER_PROMPT
    assert "calculate_arithmetic" in _SCRIPT_WRITER_PROMPT
    assert "canonical contract names in every speaker label" in _SCRIPT_WRITER_PROMPT
    assert "call participants" in _SCRIPT_WRITER_PROMPT
    assert "complete non-null state_delta" in _SCRIPT_WRITER_PROMPT
    assert "grandfathered pre-contract run" not in _SCRIPT_WRITER_PROMPT


def test_specialist_skills_are_packaged_and_not_assigned_to_stage_owners() -> None:
    assert _SPECIALIST_SKILL_SOURCES == {
        "canon_reviewer": ["/skills/canon-review"],
        "episode_reviewer": ["/skills/episode-continuity-review"],
        "episode_repair": ["/skills/continuity-repair"],
        "story_repair": ["/skills/story-repair"],
    }
    assert set(load_agent_skill_files()) == {
        "/skills/canon-review/SKILL.md",
        "/skills/episode-continuity-review/SKILL.md",
        "/skills/continuity-repair/SKILL.md",
        "/skills/story-repair/SKILL.md",
    }
    assert not {"story_architect", "episode_planner", "script_writer", "quality_reviewer"} & set(
        _SPECIALIST_SKILL_SOURCES
    )
    skill_files = load_agent_skill_files()
    canon_skill = skill_files["/skills/canon-review/SKILL.md"]
    assert "smallest set" in canon_skill
    assert "explicit hard Canon" in canon_skill
    assert "Ordinary approved prose" in canon_skill
    assert (
        "writer is free to choose unspecified details"
        in skill_files["/skills/canon-review/SKILL.md"]
    )
    assert (
        "complete committed series prefix"
        in skill_files["/skills/episode-continuity-review/SKILL.md"]
    )
    episode_skill = skill_files["/skills/episode-continuity-review/SKILL.md"]
    assert "Ordinary prose in an approved" in episode_skill
    assert "leave unspecified" in episode_skill
    assert "free" in episode_skill


def test_review_prompts_do_not_turn_missing_creative_detail_into_failure() -> None:
    assert "missing upstream commitment" not in _STORY_ARCHITECT_PROMPT
    assert "missing upstream commitment" not in _EPISODE_PLANNER_PROMPT
    assert "explicitly locked" in _SCRIPT_WRITER_PROMPT


def test_calculate_arithmetic_preserves_exact_decimal_result() -> None:
    assert _calculate_arithmetic("190", "divide", "8") == "23.75"
    assert _calculate_arithmetic("12", "multiply", "16") == "192"
    assert _calculate_arithmetic("1", "divide", "3") == (
        "1/3 (non-terminating decimal; do not round without an explicit rule)"
    )


def test_calculate_arithmetic_returns_correction_for_clock_time_operands() -> None:
    result = _arithmetic_tool().invoke({"left": "22:50", "operation": "subtract", "right": "22:20"})

    assert "Operands must be decimal numbers" in result
    assert "convert clock times to elapsed minutes" in result


@pytest.mark.parametrize("operand", ["NaN", "Infinity", "1e1000000"])
def test_calculate_arithmetic_rejects_non_finite_or_unbounded_operands(
    operand: str,
) -> None:
    with pytest.raises(ValueError, match="finite bounded decimal"):
        _calculate_arithmetic(operand, "add", "1")


@pytest.mark.asyncio
async def test_workflow_routes_generation_and_review_roles_to_distinct_models(
    tmp_path: Path,
    monkeypatch,
) -> None:
    class RoutingCaptured(Exception):
        pass

    captured: dict[str, Any] = {}

    def capture_agent(**kwargs: Any) -> None:
        captured.update(kwargs)
        raise RoutingCaptured

    monkeypatch.setattr("pengine.agents.create_deep_agent", capture_agent)
    generation_model = ToolCallingFakeModel(responses=[])
    review_model = ToolCallingFakeModel(responses=[])
    database = tmp_path / "routing-checkpoints.sqlite3"

    async def before_stage(_: InternalStage) -> int:
        return 1

    async def approve_stage(_: InternalStage, __: dict[str, Any]) -> None:
        raise AssertionError("No stage should run during topology capture")

    async with AsyncSqliteSaver.from_conn_string(str(database)) as saver:
        await saver.setup()
        workflow = DeepAgentWorkflow(
            generation_model=generation_model,
            review_model=review_model,
            checkpointer=saver,
            generation_provider_profile_key="toolcallingfakemodel",
            review_provider_profile_key="toolcallingfakemodel",
        )
        with pytest.raises(RoutingCaptured):
            await workflow.execute(
                thread_id="routing-thread",
                story="一句创意。",
                requirements="生成短剧。",
                persona_files={"/persona/project.md": "规则"},
                before_stage=before_stage,
                approve_stage=approve_stage,
            )

    assert captured["model"] is generation_model
    subagent_models = {spec["name"]: spec["model"] for spec in captured["subagents"]}
    assert {name for name, model in subagent_models.items() if model is generation_model} == {
        "story_architect",
        "episode_planner",
        "script_writer",
        "episode_repair",
        "story_repair",
    }
    assert {name for name, model in subagent_models.items() if model is review_model} == {
        "quality_reviewer",
        "canon_reviewer",
        "episode_reviewer",
        "series_reviewer",
    }


@pytest.mark.asyncio
async def test_real_deepagents_topology_and_structured_flow(tmp_path: Path) -> None:
    database = tmp_path / "checkpoints.sqlite3"
    events: list[tuple[str, str]] = []

    async def before_stage(stage: InternalStage) -> int:
        events.append(("before", stage.value))
        return 1

    async def approve_stage(stage: InternalStage, _: dict[str, Any]) -> None:
        events.append(("approve", stage.value))

    async with AsyncSqliteSaver.from_conn_string(str(database)) as saver:
        await saver.setup()
        model = ToolCallingFakeModel(responses=_successful_responses())
        workflow = _fake_workflow(
            model=model,
            checkpointer=saver,
            recursion_limit=40,
            provider_profile_key="toolcallingfakemodel",
        )

        episode_hooks, episode_attempts = _episode_hook_kwargs()
        result = await workflow.execute(
            thread_id="initial-thread",
            story="一个人回乡面对旧事。",
            requirements="生成完整短剧。",
            persona_files={"/persona/project.md": "只读人格规则"},
            before_stage=before_stage,
            approve_stage=approve_stage,
            **episode_hooks,
        )

        assert result.content_package.episode_scripts == "第 1 集\n事实1\n钩子1"
        assert [stage for kind, stage in events if kind == "before"] == [
            "selecting_l0_variant",
            "generating_story_outline",
            "generating_character_relationships",
            "generating_episode_outline",
            "accepting_l0",
            "accepting_l4",
        ]
        assert [stage for kind, stage in events if kind == "approve"] == [
            "selecting_l0_variant",
            "generating_story_outline",
            "generating_character_relationships",
            "generating_episode_outline",
            "generating_episode_scripts",
            "accepting_l0",
            "accepting_l4",
        ]
        assert episode_attempts == [1]

        checkpoint = await saver.aget_tuple({"configurable": {"thread_id": "initial-thread"}})
        assert checkpoint is not None

        all_tool_names = {name for snapshot in model.bound_tool_names for name in snapshot}
        assert "execute" not in all_tool_names
        assert "task" in all_tool_names
        assert "calculate_arithmetic" in all_tool_names
        task_descriptions = [
            description
            for names, descriptions in zip(
                model.bound_tool_names,
                model.bound_tool_descriptions,
                strict=True,
            )
            for name, description in zip(names, descriptions, strict=True)
            if name == "task"
        ]
        assert task_descriptions
        for name in (
            "story_architect",
            "episode_planner",
            "script_writer",
            "quality_reviewer",
        ):
            assert any(name in description for description in task_descriptions)
        assert all("\n- general-purpose:" not in description for description in task_descriptions)


@pytest.mark.asyncio
async def test_story_artifact_is_reviewed_and_minimally_repaired_before_approval() -> None:
    approved_payloads: dict[InternalStage, Any] = {
        InternalStage.SELECTING_L0_VARIANT: {
            "stage": "selecting_l0_variant",
            "content": None,
            "selected_l0_variant": "主动选择",
            "selection_rationale": "契合创意",
        },
        InternalStage.GENERATING_STORY_OUTLINE: {
            "stage": "generating_story_outline",
            "content": "程远海难时二十二岁，程屿二十岁。",
            "character_biographies": None,
            "relationship_logic": None,
            "selected_l0_variant": None,
            "selection_rationale": None,
        },
    }
    frozen_upstream = copy.deepcopy(approved_payloads)
    approvals: list[tuple[InternalStage, dict[str, Any]]] = []
    review_calls = 0
    repair_calls = 0

    async def before_stage(stage: InternalStage) -> int:
        assert stage is InternalStage.GENERATING_CHARACTER_RELATIONSHIPS
        return 1

    async def approve_stage(stage: InternalStage, payload: dict[str, Any]) -> None:
        approvals.append((stage, payload))
        approved_payloads[stage] = payload

    middleware = StageGuardMiddleware(
        before_stage=before_stage,
        approve_stage=approve_stage,
        approved_stages=set(approved_payloads),
        approved_payloads=approved_payloads,
    )
    request = ToolCallRequest(
        tool_call={
            "name": "task",
            "args": {
                "description": (
                    "[stage=generating_character_relationships] write character + relationships"
                ),
                "subagent_type": "story_architect",
            },
            "id": "story-consistency",
            "type": "tool_call",
        },
        tool=None,
        state={"files": {}},
        runtime=None,
    )

    async def handler(candidate: ToolCallRequest) -> ToolMessage:
        nonlocal review_calls, repair_calls
        subagent_type = candidate.tool_call["args"]["subagent_type"]
        if subagent_type == "story_architect":
            payload = {
                "stage": "generating_character_relationships",
                "content": None,
                "character_biographies": "程远二十二岁，比程屿大两岁。",
                "relationship_logic": (
                    "程远二十四岁，比程屿大六岁。电话对象写成周砚。\n"
                    "通话记录确认电话对象写成周砚。\n"
                    "两人的年龄关系影响程屿对兄长的依赖和调查选择，并约束后续全部情节。\n"
                    "其余人物身份、秘密来源、证物链和时间线均保持已批准版本。"
                ),
                "selected_l0_variant": None,
                "selection_rationale": None,
            }
        elif subagent_type == "story_repair":
            # The c+r repair subagent returns the complete rewritten candidate
            # (not a patch) with both confirmed conflicts resolved jointly.
            repair_calls += 1
            current = candidate.state["files"]["/workspace/current_story_candidate.md"]["content"]
            assert "二十四岁" in current
            assert "电话对象写成周砚" in current
            payload = {
                "stage": "generating_character_relationships",
                "content": None,
                "character_biographies": "程远二十二岁，比程屿大两岁。",
                "relationship_logic": (
                    "程远二十二岁，比程屿大两岁。电话对象写成程屿。\n"
                    "通话记录确认电话对象写成程屿。\n"
                    "两人的年龄关系影响程屿对兄长的依赖和调查选择，并约束后续全部情节。\n"
                    "其余人物身份、秘密来源、证物链和时间线均保持已批准版本。"
                ),
                "selected_l0_variant": None,
                "selection_rationale": None,
            }
        else:
            assert subagent_type == "canon_reviewer"
            review_calls += 1
            rl = candidate.state["files"].get("/workspace/current_relationship_logic.md", {})
            current = rl.get("content", "")
            if review_calls <= 2:
                assert "二十四岁" in current
                if review_calls == 1:
                    payload = {
                        "passed": False,
                        "evidence": "关系稿年龄与人物小传冲突",
                        "issues": [
                            {
                                "code": "relative_age_conflict",
                                "message": "权威小传为二十二岁、相差两岁",
                                "script_excerpt": "程远二十四岁，比程屿大六岁。",
                            }
                        ],
                    }
                else:
                    payload = {
                        "passed": False,
                        "evidence": "通话对象与上游冲突",
                        "issues": [
                            {
                                "code": "call_participant_conflict",
                                "message": "权威通话对象为程屿",
                                "script_excerpt": "电话对象写成周砚。",
                            }
                        ],
                    }
            else:
                assert "二十二岁" in current
                assert "电话对象写成程屿" in current
                payload = {"passed": True, "evidence": "故事工件一致", "issues": []}
            payload["prior_issue_closures"] = _resolved_prior_story_closures(candidate)
        return ToolMessage(
            content=json.dumps(payload, ensure_ascii=False),
            tool_call_id=candidate.tool_call["id"],
            name="task",
        )

    await middleware.awrap_tool_call(request, handler)

    assert review_calls == 4
    assert repair_calls == 1
    assert (
        approved_payloads | {key: value for key, value in frozen_upstream.items()}
        == approved_payloads
    )
    assert all(approved_payloads[stage] == payload for stage, payload in frozen_upstream.items())
    assert len(approvals) == 1
    stage, repaired = approvals[0]
    assert stage is InternalStage.GENERATING_CHARACTER_RELATIONSHIPS
    assert repaired["consistency_review"]["passed"] is True
    assert repaired["consistency_repair_rounds"] == 1
    assert "二十二岁，比程屿大两岁" in repaired["relationship_logic"]
    assert "电话对象写成程屿" in repaired["relationship_logic"]


@pytest.mark.asyncio
async def test_story_consistency_converges_at_fourth_repair_round() -> None:
    approvals: list[tuple[InternalStage, dict[str, Any]]] = []
    approved_payloads: dict[InternalStage, Any] = {
        InternalStage.SELECTING_L0_VARIANT: {
            "stage": "selecting_l0_variant",
            "content": None,
            "selected_l0_variant": "主动选择",
            "selection_rationale": "契合创意",
        },
        InternalStage.GENERATING_STORY_OUTLINE: {
            "stage": "generating_story_outline",
            "content": "林夏追查父亲承担责任的原因，旧表与救援站构成既有证物链。",
            "character_biographies": None,
            "relationship_logic": None,
            "selected_l0_variant": None,
            "selection_rationale": None,
        },
    }
    review_calls = 0
    repair_calls = 0

    async def before_stage(stage: InternalStage) -> int:
        assert stage is InternalStage.GENERATING_CHARACTER_RELATIONSHIPS
        return 1

    async def approve_stage(stage: InternalStage, payload: dict[str, Any]) -> None:
        approvals.append((stage, payload))
        approved_payloads[stage] = payload

    middleware = StageGuardMiddleware(
        before_stage=before_stage,
        approve_stage=approve_stage,
        approved_stages=set(approved_payloads),
        approved_payloads=approved_payloads,
    )
    request = ToolCallRequest(
        tool_call={
            "name": "task",
            "args": {
                "description": (
                    "[stage=generating_character_relationships] write character + relationships"
                ),
                "subagent_type": "story_architect",
            },
            "id": "story-backstop",
            "type": "tool_call",
        },
        tool=None,
        state={"files": {}},
        runtime=None,
    )

    async def handler(candidate: ToolCallRequest) -> ToolMessage:
        nonlocal review_calls, repair_calls
        subagent_type = candidate.tool_call["args"]["subagent_type"]
        if subagent_type == "story_architect":
            payload = {
                "stage": "generating_character_relationships",
                "content": None,
                "character_biographies": "陈伯是岛上救援站值班员，长期保管周远的值班记录与手记。",
                "relationship_logic": (
                    "电话打给周砚后，陈伯决定在终局作证。\n"
                    "人物摘要仍称陈伯一直知情并准备在终局作证。\n"
                    "关系摘要仍称陈伯一直知情并准备在终局作证。\n"
                    "其余人物身份、时间线与结局选择保持上游版本。"
                ),
                "selected_l0_variant": None,
                "selection_rationale": None,
            }
        elif subagent_type == "story_repair":
            # The c+r repair subagent returns the complete rewritten candidate
            # (not a patch). Each round cumulatively fixes the confirmed issues,
            # mirroring what the old line-range patches produced.
            repair_calls += 1
            current = candidate.state["files"]["/workspace/current_story_candidate.md"]["content"]
            if repair_calls == 1:
                # Round 1 fixes only the call-participant conflict; the
                # knowledge-source gap remains, so the post-repair review still
                # fails and the stage keeps repairing.
                assert "电话打给周砚" in current
                relationship_logic = (
                    "电话改为打给程屿后，陈伯一直知情并准备在终局作证。\n"
                    "人物摘要仍称陈伯一直知情并准备在终局作证。\n"
                    "关系摘要仍称陈伯一直知情并准备在终局作证。\n"
                    "其余人物身份、时间线与结局选择保持上游版本。"
                )
            elif repair_calls == 2:
                # Round 2 closes the knowledge-source gap on the main line by
                # supplying 陈伯's既有记录来源; the two summaries still lack it.
                assert "陈伯一直知情并准备在终局作证" in current
                relationship_logic = (
                    "电话改为打给程屿；陈伯因保管周远的值班记录与手记而知情，并据此在终局作证。\n"
                    "人物摘要仍称陈伯一直知情并准备在终局作证。\n"
                    "关系摘要仍称陈伯一直知情并准备在终局作证。\n"
                    "其余人物身份、时间线与结局选择保持上游版本。"
                )
            elif repair_calls == 3:
                # Round 3 fixes 人物摘要 to include the source; 关系摘要 still lacks it.
                assert "值班记录与手记" in current
                assert "人物摘要仍称陈伯一直知情" in current
                relationship_logic = (
                    "电话改为打给程屿；陈伯因保管周远的值班记录与手记而知情，并据此在终局作证。\n"
                    "人物摘要同步说明陈伯因保管周远的值班记录与手记而知情。\n"
                    "关系摘要仍称陈伯一直知情并准备在终局作证。\n"
                    "其余人物身份、时间线与结局选择保持上游版本。"
                )
            else:
                # Round 4 (the cap) closes the last residual gap, so both review
                # lenses pass and the stage converges exactly at the maximum round.
                assert repair_calls == 4
                assert "关系摘要仍称陈伯一直知情" in current
                relationship_logic = (
                    "电话改为打给程屿；陈伯因保管周远的值班记录与手记而知情，并据此在终局作证。\n"
                    "人物摘要同步说明陈伯因保管周远的值班记录与手记而知情。\n"
                    "关系摘要同步说明陈伯因保管周远的值班记录与手记而知情。\n"
                    "其余人物身份、时间线与结局选择保持上游版本。"
                )
            payload = {
                "stage": "generating_character_relationships",
                "content": None,
                "character_biographies": "陈伯是岛上救援站值班员，长期保管周远的值班记录与手记。",
                "relationship_logic": relationship_logic,
                "selected_l0_variant": None,
                "selection_rationale": None,
            }
        else:
            assert subagent_type == "canon_reviewer"
            # c+r reviews now receive per-section files.
            rl = candidate.state["files"].get("/workspace/current_relationship_logic.md", {})
            current = rl.get("content", "")
            review_calls += 1
            if review_calls in {1, 2}:
                # repair_rounds == 0: both lenses flag the call-participant conflict.
                assert "电话打给周砚" in current
                payload = {
                    "passed": False,
                    "evidence": "通话对象与上游冲突",
                    "issues": [
                        {
                            "code": "call_participant_conflict",
                            "message": "权威通话对象为程屿。",
                            "script_excerpt": "电话打给周砚后，陈伯决定在终局作证。",
                        }
                    ],
                }
            elif review_calls in {3, 4}:
                # repair_rounds == 1: call conflict closed, but the knowledge gap remains.
                assert "电话改为打给程屿后，陈伯一直知情并准备在终局作证。" in current
                payload = {
                    "passed": False,
                    "evidence": "知情来源缺口",
                    "issues": [
                        {
                            "code": "knowledge_source_gap",
                            "message": "必须补出陈伯如何得知真相的既有来源。",
                            "script_excerpt": "陈伯一直知情并准备在终局作证。",
                        },
                    ],
                }
            elif review_calls in {5, 6}:
                # repair_rounds == 2: knowledge source closed on the main line, but the
                # 人物摘要 still lacks the source.
                assert "值班记录与手记" in current
                assert "人物摘要仍称陈伯一直知情" in current
                payload = {
                    "passed": False,
                    "evidence": "重复摘要仍缺少知情来源",
                    "issues": [
                        {
                            "code": "repeated_knowledge_source_gap",
                            "message": "第二行也必须同步写明陈伯知情的既有记录来源。",
                            "script_excerpt": "人物摘要仍称陈伯一直知情",
                        }
                    ],
                }
            elif review_calls in {7, 8}:
                # repair_rounds == 3: 人物摘要 closed, 关系摘要 still lacks it.
                assert "值班记录与手记" in current
                assert "关系摘要仍称陈伯一直知情" in current
                payload = {
                    "passed": False,
                    "evidence": "关系摘要仍缺少知情来源",
                    "issues": [
                        {
                            "code": "relationship_summary_knowledge_gap",
                            "message": "第三行也必须同步写明陈伯知情的既有记录来源。",
                            "script_excerpt": "关系摘要仍称陈伯一直知情",
                        }
                    ],
                }
            else:
                # repair_rounds == 4 (the cap): every gap is closed, both lenses pass
                # and the stage converges exactly at the maximum repair round.
                assert review_calls in {9, 10}
                assert "值班记录与手记" in current
                assert "人物摘要仍称陈伯一直知情" not in current
                assert "关系摘要仍称陈伯一直知情" not in current
                payload = {"passed": True, "evidence": "故事工件一致", "issues": []}
            payload["prior_issue_closures"] = _resolved_prior_story_closures(candidate)
        return ToolMessage(
            content=json.dumps(payload, ensure_ascii=False),
            tool_call_id=candidate.tool_call["id"],
            name="task",
        )

    await middleware.awrap_tool_call(request, handler)

    assert review_calls == 10
    assert repair_calls == 4
    assert len(approvals) == 1
    stage, repaired = approvals[0]
    assert stage is InternalStage.GENERATING_CHARACTER_RELATIONSHIPS
    assert repaired["consistency_review"]["passed"] is True
    assert repaired["consistency_repair_rounds"] == 4
    assert "值班记录与手记" in repaired["relationship_logic"]


@pytest.mark.asyncio
async def test_contract_review_repairs_once_before_outline_lock(tmp_path: Path) -> None:
    database = tmp_path / "checkpoints.sqlite3"
    responses = _successful_responses()
    outline_review_index = _index_of_tool_call(responses, "CanonReviewerResult", occurrence=4)
    responses[outline_review_index] = _tool_call(
        "CanonReviewerResult",
        {
            "passed": False,
            "evidence": "合同遗漏一项上游承诺",
            "issues": [
                {
                    "code": "missing_commitment",
                    "message": "必须补齐承诺",
                    "contract_refs": [],
                    "script_excerpt": None,
                }
            ],
        },
        outline_review_index,
    )
    responses.insert(
        outline_review_index + 1,
        _tool_call(
            "OutlineRepairPatch",
            {
                "stage": "generating_episode_outline",
                "content_replacements": [{"old": "分集大纲", "new": "修复后的分集大纲"}],
                "json_edits": [],
            },
            101,
        ),
    )
    responses.insert(
        outline_review_index + 2,
        _tool_call(
            "CanonReviewerResult",
            {"passed": True, "evidence": "修复后合同一致", "issues": []},
            102,
        ),
    )
    approved: dict[InternalStage, dict[str, Any]] = {}

    async def before_stage(_: InternalStage) -> int:
        return 1

    async def approve_stage(stage: InternalStage, payload: dict[str, Any]) -> None:
        approved[stage] = payload

    async with AsyncSqliteSaver.from_conn_string(str(database)) as saver:
        await saver.setup()
        workflow = _fake_workflow(
            model=ToolCallingFakeModel(responses=responses),
            checkpointer=saver,
            provider_profile_key="toolcallingfakemodel",
        )
        await workflow.execute(
            thread_id="contract-repair-thread",
            story="故事",
            requirements="要求",
            persona_files={"/persona/project.md": "规则"},
            before_stage=before_stage,
            approve_stage=approve_stage,
            **_episode_hook_kwargs()[0],
        )

    outline = approved[InternalStage.GENERATING_EPISODE_OUTLINE]
    assert outline["contract_repair_rounds"] == 1
    assert outline["contract_review"]["passed"] is True
    assert outline["content"] == "修复后的分集大纲"
    assert len(outline["story_contract_sha256"]) == 64


@pytest.mark.asyncio
async def test_contract_repair_stops_after_one_invalid_patch_correction(tmp_path: Path) -> None:
    database = tmp_path / "checkpoints.sqlite3"
    responses = _successful_responses()
    outline_review_index = _index_of_tool_call(responses, "CanonReviewerResult", occurrence=4)
    responses[outline_review_index] = _tool_call(
        "CanonReviewerResult",
        {
            "passed": False,
            "evidence": "合同遗漏一项上游承诺",
            "issues": [{"code": "missing_commitment", "message": "必须补齐承诺"}],
        },
        outline_review_index,
    )
    invalid_patch = {
        "stage": "generating_episode_outline",
        "content_replacements": [],
        "json_edits": [],
    }
    responses.insert(outline_review_index + 1, _tool_call("OutlineRepairPatch", invalid_patch, 101))
    responses.insert(outline_review_index + 2, _tool_call("OutlineRepairPatch", invalid_patch, 102))
    responses.insert(
        outline_review_index + 3,
        _tool_call(
            "OutlineRepairPatch",
            {
                "stage": "generating_episode_outline",
                "content_replacements": [{"old": "分集大纲", "new": "不应被消费"}],
            },
            103,
        ),
    )
    approved: set[InternalStage] = set()

    async def before_stage(_: InternalStage) -> int:
        return 1

    async def approve_stage(stage: InternalStage, _: dict[str, Any]) -> None:
        approved.add(stage)

    async with AsyncSqliteSaver.from_conn_string(str(database)) as saver:
        await saver.setup()
        workflow = _fake_workflow(
            model=ToolCallingFakeModel(responses=responses),
            checkpointer=saver,
            provider_profile_key="toolcallingfakemodel",
        )
        with pytest.raises(AgentProtocolError) as error:
            await workflow.execute(
                thread_id="contract-patch-correction-limit-thread",
                story="故事",
                requirements="要求",
                persona_files={"/persona/project.md": "规则"},
                before_stage=before_stage,
                approve_stage=approve_stage,
                **_episode_hook_kwargs()[0],
            )

    assert error.value.safe_message == "分集大纲修复补丁未通过结构化校验。"
    assert InternalStage.GENERATING_EPISODE_OUTLINE not in approved


@pytest.mark.asyncio
async def test_outline_canon_review_receives_structured_episode_plans() -> None:
    contract = _story_contract()
    planner_payload = {
        "stage": "generating_episode_outline",
        "content": "分集大纲",
        "episode_count": 1,
        "episodes": [{"episode_number": 1, "plan": "第一集计划"}],
        "story_contract": contract.model_dump(mode="json"),
    }
    reviewed_plans: list[list[dict[str, Any]]] = []

    async def before_stage(_: InternalStage) -> int:
        return 1

    async def approve_stage(_: InternalStage, __: dict[str, Any]) -> None:
        return None

    async def generate_patch(
        _: Mapping[str, Any],
        __: CanonReviewerResult,
        ___: int,
        correction: str | None,
    ) -> Any:
        assert correction is None
        return {
            "stage": "generating_episode_outline",
            "json_edits": [
                {
                    "op": "replace",
                    "path": "/episodes/0/plan",
                    "expected": "第一集计划",
                    "value": "修复后的第一集计划",
                }
            ],
        }

    middleware = StageGuardMiddleware(
        before_stage=before_stage,
        approve_stage=approve_stage,
        approved_stages=set(),
        output_language=SIMPLIFIED_CHINESE,
        generate_outline_patch=generate_patch,
    )
    request = ToolCallRequest(
        tool_call={
            "name": "task",
            "args": {
                "description": "[stage=generating_episode_outline] generate outline",
                "subagent_type": "episode_planner",
            },
            "id": "call-outline-review-files",
            "type": "tool_call",
        },
        tool=None,
        state={"files": {}},
        runtime=None,
    )

    async def handler(candidate_request: ToolCallRequest) -> ToolMessage:
        subagent_type = candidate_request.tool_call["args"]["subagent_type"]
        if subagent_type == "episode_planner":
            payload = planner_payload
        else:
            assert subagent_type == "canon_reviewer"
            reviewed_plans.append(
                json.loads(
                    candidate_request.state["files"]["/workspace/episode_plans.json"]["content"]
                )
            )
            payload = (
                {
                    "passed": False,
                    "evidence": "第一集计划需要修复",
                    "issues": [{"code": "episode_plan", "message": "修复第一集计划"}],
                }
                if len(reviewed_plans) == 1
                else {"passed": True, "evidence": "合同一致", "issues": []}
            )
        return ToolMessage(
            content=json.dumps(payload, ensure_ascii=False),
            tool_call_id="call-outline-review-files",
        )

    _, locked = await middleware._generate_locked_outline(
        request,
        handler,
        request.tool_call["args"],
    )

    assert reviewed_plans == [
        [{"episode_number": 1, "plan": "第一集计划"}],
        [{"episode_number": 1, "plan": "修复后的第一集计划"}],
    ]
    assert locked["contract_review"]["passed"] is True
    assert locked["episodes"][0]["plan"] == "修复后的第一集计划"


@pytest.mark.asyncio
async def test_episode_review_stops_after_two_repairs_without_commit(tmp_path: Path) -> None:
    database = tmp_path / "checkpoints.sqlite3"
    responses = _successful_responses()
    writer_index = _index_of_tool_call(responses, "ScriptWriterResult", occurrence=1)
    review_index = _index_of_tool_call(responses, "EpisodeReviewerResult", occurrence=1)
    writer_payload = responses[writer_index].tool_calls[0]["args"]
    failed_review = {
        "passed": False,
        "evidence": "人物身份与上游小传不一致",
        "issues": [
            {
                "code": "identity_drift",
                "message": "剧本把母亲姓名改成了合同外角色",
                "contract_refs": ["fact_ep1"],
                "script_excerpt": "事实1",
            }
        ],
    }
    responses[review_index] = _tool_call("EpisodeReviewerResult", failed_review, review_index)
    responses.insert(review_index + 1, _tool_call("ScriptWriterResult", writer_payload, 201))
    responses.insert(review_index + 2, _tool_call("EpisodeReviewerResult", failed_review, 202))
    responses.insert(review_index + 3, _tool_call("ScriptWriterResult", writer_payload, 203))
    responses.insert(review_index + 4, _tool_call("EpisodeReviewerResult", failed_review, 204))
    approved: list[InternalStage] = []
    episode_hooks, episode_attempts = _episode_hook_kwargs()

    async def before_stage(_: InternalStage) -> int:
        return 1

    async def approve_stage(stage: InternalStage, _: dict[str, Any]) -> None:
        approved.append(stage)

    async with AsyncSqliteSaver.from_conn_string(str(database)) as saver:
        await saver.setup()
        workflow = _fake_workflow(
            model=ToolCallingFakeModel(responses=responses),
            checkpointer=saver,
            provider_profile_key="toolcallingfakemodel",
        )
        with pytest.raises(ContentReviewRejectedError) as error:
            await workflow.execute(
                thread_id="episode-repair-limit-thread",
                story="故事",
                requirements="要求",
                persona_files={"/persona/project.md": "规则"},
                before_stage=before_stage,
                approve_stage=approve_stage,
                **episode_hooks,
            )

    assert error.value.episode_number == 1
    assert error.value.repair_rounds == 2
    assert episode_attempts == [1]
    assert InternalStage.GENERATING_EPISODE_SCRIPTS not in approved


@pytest.mark.asyncio
async def test_episode_repair_receives_deterministic_and_semantic_issues_together(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database = tmp_path / "checkpoints.sqlite3"
    responses = _successful_responses()
    writer_index = _index_of_tool_call(responses, "ScriptWriterResult", occurrence=1)
    review_index = _index_of_tool_call(responses, "EpisodeReviewerResult", occurrence=1)
    repaired_writer_payload = copy.deepcopy(responses[writer_index].tool_calls[0]["args"])
    invalid_writer_payload = copy.deepcopy(repaired_writer_payload)
    invalid_writer_payload["content"] = "钩子1"
    responses[writer_index] = _tool_call("ScriptWriterResult", invalid_writer_payload, writer_index)
    responses[review_index] = _tool_call(
        "EpisodeReviewerResult",
        {
            "passed": False,
            "evidence": "人物身份发生漂移",
            "issues": [
                {
                    "code": "identity_drift",
                    "message": "人物身份与上游不一致",
                    "script_excerpt": "钩子1",
                }
            ],
        },
        review_index,
    )
    responses.insert(
        review_index + 1,
        _tool_call("ScriptWriterResult", repaired_writer_payload, 201),
    )
    responses.insert(
        review_index + 2,
        _tool_call(
            "EpisodeReviewerResult",
            {"passed": True, "evidence": "修复后分集一致", "issues": []},
            202,
        ),
    )
    captured_files: list[Mapping[str, str]] = []
    original_repair = StageGuardMiddleware._invoke_repair_subagent

    async def capture_repair(
        middleware: StageGuardMiddleware,
        **kwargs: Any,
    ) -> tuple[Any, dict[str, Any]]:
        captured_files.append(kwargs["files"])
        return await original_repair(middleware, **kwargs)

    monkeypatch.setattr(StageGuardMiddleware, "_invoke_repair_subagent", capture_repair)
    approved: list[InternalStage] = []

    async def before_stage(_: InternalStage) -> int:
        return 1

    async def approve_stage(stage: InternalStage, _: dict[str, Any]) -> None:
        approved.append(stage)

    async with AsyncSqliteSaver.from_conn_string(str(database)) as saver:
        await saver.setup()
        workflow = _fake_workflow(
            model=ToolCallingFakeModel(responses=responses),
            checkpointer=saver,
            provider_profile_key="toolcallingfakemodel",
        )
        await workflow.execute(
            thread_id="merged-episode-review-thread",
            story="故事",
            requirements="要求",
            persona_files={"/persona/project.md": "规则"},
            before_stage=before_stage,
            approve_stage=approve_stage,
            **_episode_hook_kwargs()[0],
        )

    review = json.loads(captured_files[0]["/workspace/episode_review.json"])
    assert {issue["code"] for issue in review["issues"]} >= {
        "evidence_not_in_script",
        "identity_drift",
    }
    assert captured_files[0]["/workspace/current_episode_plan.md"] == "第一集计划"
    obligation = json.loads(captured_files[0]["/workspace/current_episode_obligation.json"])
    assert obligation["end_hook"] == "钩子1"
    assert InternalStage.GENERATING_EPISODE_SCRIPTS in approved


@pytest.mark.asyncio
async def test_last_episode_review_receives_complete_series_prefix_before_approval() -> None:
    contract = _story_contract(episode_count=2)
    contract_hash = story_contract_sha256(contract)
    approved_payloads: dict[InternalStage, dict[str, Any]] = {
        InternalStage.SELECTING_L0_VARIANT: {
            "stage": "selecting_l0_variant",
            "selected_l0_variant": "主动选择",
            "selection_rationale": "契合故事",
        },
        InternalStage.GENERATING_STORY_OUTLINE: {
            "stage": "generating_story_outline",
            "content": "故事大纲",
            "character_biographies": None,
            "relationship_logic": None,
        },
        InternalStage.GENERATING_CHARACTER_RELATIONSHIPS: {
            "stage": "generating_character_relationships",
            "content": None,
            "character_biographies": "人物小传",
            "relationship_logic": "关系逻辑",
        },
        InternalStage.GENERATING_EPISODE_OUTLINE: {
            "stage": "generating_episode_outline",
            "content": "两集分集大纲",
            "episode_count": 2,
            "episodes": [
                {"episode_number": 1, "plan": "第一集计划"},
                {"episode_number": 2, "plan": "第二集计划"},
            ],
            "story_contract": contract.model_dump(mode="json"),
            "story_contract_sha256": contract_hash,
            "story_contract_markdown": render_story_contract_markdown(contract, contract_hash),
            "contract_review": {"passed": True, "evidence": "合同一致", "issues": []},
            "contract_repair_rounds": 0,
        },
    }
    approved_stages = set(approved_payloads)
    approvals: dict[InternalStage, dict[str, Any]] = {}
    episode_hooks, attempts = _episode_hook_kwargs()

    async def before_stage(_: InternalStage) -> int:
        return 1

    async def approve_stage(stage: InternalStage, payload: dict[str, Any]) -> None:
        approvals[stage] = payload

    middleware = StageGuardMiddleware(
        before_stage=before_stage,
        approve_stage=approve_stage,
        approved_stages=approved_stages,
        approved_payloads=approved_payloads,
        **episode_hooks,
    )
    request = ToolCallRequest(
        tool_call={
            "name": "task",
            "args": {
                "description": "[stage=generating_episode_scripts] write scripts",
                "subagent_type": "script_writer",
            },
            "id": "call-series-prefix",
            "type": "tool_call",
        },
        tool=None,
        state={"files": {}},
        runtime=None,
    )
    reviewed_prefixes: list[set[str]] = []
    reviewed_prefix_contents: list[str] = []

    async def handler(subagent_request: ToolCallRequest) -> ToolMessage:
        description = subagent_request.tool_call["args"]["description"]
        subagent_type = subagent_request.tool_call["args"]["subagent_type"]
        episode_number = 2 if "episode=2" in description or "episode 2" in description else 1
        if subagent_type == "script_writer":
            payload = {
                "stage": "generating_episode_scripts",
                "episode_number": episode_number,
                "content": f"事实{episode_number}\n钩子{episode_number}",
                "state_delta": _state_delta(contract, episode_number),
            }
        else:
            assert subagent_type == "episode_reviewer"
            assert "complete committed series prefix" in description
            assert "trusted runtime metadata" in description
            assert "episodes[].content" in description
            for required_check in (
                "identities",
                "relationships",
                "pronouns",
                "call participants",
                "clue meanings",
            ):
                assert required_check in description
            reviewed_prefixes.append(set(subagent_request.state["files"]))
            prefix_file = subagent_request.state["files"]["/workspace/series_prefix.json"]
            reviewed_prefix_contents.append(prefix_file["content"])
            payload = {"passed": True, "evidence": "完整前缀一致", "issues": []}
        return ToolMessage(
            content=json.dumps(payload, ensure_ascii=False),
            tool_call_id="call-series-prefix",
        )

    await middleware.awrap_tool_call(request, handler)

    assert attempts == [1, 2]
    assert "/workspace/episodes/ep1.md" not in reviewed_prefixes[0]
    assert {
        "/workspace/episodes/ep1.md",
        "/workspace/candidate_episode.md",
        "/workspace/story_outline.md",
        "/workspace/character_biographies.md",
        "/workspace/relationship_logic.md",
        "/workspace/episode_outline.md",
        "/workspace/story_contract.json",
        "/workspace/series_prefix.json",
    } <= reviewed_prefixes[1]
    assert json.loads(reviewed_prefix_contents[1]) == {
        "episodes": [
            {"episode_number": 1, "content": "事实1\n钩子1"},
            {"episode_number": 2, "content": "事实2\n钩子2"},
        ]
    }
    assert "第 1 集" not in reviewed_prefix_contents[1]
    assert InternalStage.GENERATING_EPISODE_SCRIPTS in approvals


@pytest.mark.asyncio
async def test_episode_writer_receives_complete_active_design_and_verbatim_prefix() -> None:
    """FSW-A2: episode N input carries the full SeriesBible, verbatim scripts 1..N-1,
    the folded SeriesState, the exact plan/obligation, and bounded WriterNotes."""
    contract = _story_contract(episode_count=2)
    contract_hash = story_contract_sha256(contract)
    summary = project_series_bible(
        build_series_bible(
            run_id="run-1",
            run_kind="initial",
            l0_variant="主动选择",
            genre="general",
            story_outline="故事大纲",
            character_biographies="人物小传",
            relationship_logic="关系逻辑",
            episode_outline="两集分集大纲",
            story_contract_payload=contract.model_dump(mode="json"),
        ),
        is_active=True,
    )
    approved_payloads: dict[InternalStage, dict[str, Any]] = {
        InternalStage.SELECTING_L0_VARIANT: {
            "stage": "selecting_l0_variant",
            "selected_l0_variant": "主动选择",
            "selection_rationale": "契合故事",
        },
        InternalStage.GENERATING_STORY_OUTLINE: {
            "stage": "generating_story_outline",
            "content": "故事大纲",
            "character_biographies": None,
            "relationship_logic": None,
        },
        InternalStage.GENERATING_CHARACTER_RELATIONSHIPS: {
            "stage": "generating_character_relationships",
            "content": None,
            "character_biographies": "人物小传",
            "relationship_logic": "关系逻辑",
        },
        InternalStage.GENERATING_EPISODE_OUTLINE: {
            "stage": "generating_episode_outline",
            "content": "两集分集大纲",
            "episode_count": 2,
            "episodes": [
                {"episode_number": 1, "plan": "第一集计划"},
                {"episode_number": 2, "plan": "第二集计划"},
            ],
            "story_contract": contract.model_dump(mode="json"),
            "story_contract_sha256": contract_hash,
            "story_contract_markdown": render_story_contract_markdown(contract, contract_hash),
            "contract_review": {"passed": True, "evidence": "合同一致", "issues": []},
            "contract_repair_rounds": 0,
        },
    }
    approved_stages = set(approved_payloads)
    episode_hooks, attempts = _episode_hook_kwargs()
    registered_reviews: list[dict[str, Any]] = []

    async def register_series_review(**kwargs: Any) -> str:
        registered_reviews.append(kwargs)
        return f"review-{kwargs['episode_number']}"

    middleware = StageGuardMiddleware(
        before_stage=lambda _stage: _async_one(),
        approve_stage=lambda _stage, _payload: _async_none(),
        approved_stages=approved_stages,
        approved_payloads=approved_payloads,
        series_bible=summary,
        register_series_review=register_series_review,
        **episode_hooks,
    )
    request = ToolCallRequest(
        tool_call={
            "name": "task",
            "args": {
                "description": "[stage=generating_episode_scripts] write scripts",
                "subagent_type": "script_writer",
            },
            "id": "call-writer-input",
            "type": "tool_call",
        },
        tool=None,
        state={"files": {}},
        runtime=None,
    )
    writer_inputs: list[dict[str, Any]] = []
    writer_descriptions: list[str] = []
    series_review_inputs: list[dict[str, Any]] = []

    async def handler(subagent_request: ToolCallRequest) -> ToolMessage:
        description = subagent_request.tool_call["args"]["description"]
        subagent_type = subagent_request.tool_call["args"]["subagent_type"]
        episode_number = 2 if "episode=2" in description or "episode 2" in description else 1
        if subagent_type == "script_writer":
            writer_inputs.append(dict(subagent_request.state["files"]))
            writer_descriptions.append(description)
            payload = {
                "stage": "generating_episode_scripts",
                "episode_number": episode_number,
                "content": f"事实{episode_number}\n钩子{episode_number}",
                "state_delta": _state_delta(contract, episode_number),
                "writer_notes": f"第{episode_number}集备忘",
            }
        elif subagent_type == "series_reviewer":
            # The final milestone (episode 2) fires the bound final structural review.
            assert "trusted runtime metadata" in description
            assert "episodes[].content" in description
            series_review_inputs.append(dict(subagent_request.state["files"]))
            payload = {"passed": True, "category": "pass", "evidence": "全系列一致"}
        else:
            assert subagent_type == "episode_reviewer"
            payload = {"passed": True, "evidence": "完整前缀一致", "issues": []}
        return ToolMessage(
            content=json.dumps(payload, ensure_ascii=False),
            tool_call_id="call-writer-input",
        )

    await middleware.awrap_tool_call(request, handler)

    assert attempts == [1, 2]
    assert len(series_review_inputs) == 1
    series_prefix = json.loads(series_review_inputs[0]["/workspace/series_prefix.json"]["content"])
    assert series_prefix == {
        "episodes": [
            {"episode_number": 1, "content": "事实1\n钩子1"},
            {"episode_number": 2, "content": "事实2\n钩子2"},
        ]
    }
    first_input = writer_inputs[0]
    second_input = writer_inputs[1]

    # Every episode request carries the complete active SeriesBible projections.
    for key in (
        "/workspace/series_bible/story_outline.md",
        "/workspace/series_bible/character_biographies.md",
        "/workspace/series_bible/relationship_logic.md",
        "/workspace/series_bible/episode_outline.md",
    ):
        assert key in first_input and key in second_input
    assert second_input["/workspace/series_bible/story_outline.md"]["content"] == "故事大纲"
    assert second_input["/workspace/series_bible/episode_outline.md"]["content"] == "两集分集大纲"
    assert "/workspace/story_contract.json" in second_input
    assert "第二集计划" in writer_descriptions[1]

    # Episode 2 receives script 1 verbatim (never a summary).
    assert second_input["/workspace/episodes/ep1.md"]["content"] == "事实1\n钩子1"

    # Episode 2 receives the folded SeriesState after episode 1.
    series_state = json.loads(second_input["/workspace/series_state.json"]["content"])
    assert series_state["locked_through_episode"] == 1
    assert series_state["established_fact_ids"] == ["fact_ep1"]

    # Bounded advisory WriterNotes from episode 1 flow forward.
    assert second_input["/workspace/writer_notes.md"]["content"] == "第1集备忘"

    # The final episode fired the bound final structural review (RPR-A2).
    assert registered_reviews == [
        {
            "review_type": "final",
            "episode_number": 2,
            "passed": True,
            "category": "pass",
            "evidence": "全系列一致",
            "earliest_affected_episode": None,
        }
    ]


async def _async_one() -> int:
    return 1


async def _async_none() -> None:
    return None


@pytest.mark.asyncio
async def test_structured_output_validation_error_is_corrected_within_stage(
    tmp_path: Path,
) -> None:
    database = tmp_path / "checkpoints.sqlite3"
    responses = _successful_responses()
    responses.insert(
        1,
        _tool_call(
            "StoryArchitectResult",
            {
                "stage": "selecting_l0_variant",
                "content": "invalid content for the selection stage",
                "character_biographies": None,
                "relationship_logic": None,
                "selected_l0_variant": None,
                "selection_rationale": None,
            },
            99,
        ),
    )
    attempted: list[InternalStage] = []
    approved: list[InternalStage] = []

    async def before_stage(stage: InternalStage) -> int:
        attempted.append(stage)
        return 1

    async def approve_stage(stage: InternalStage, _: dict[str, Any]) -> None:
        approved.append(stage)

    async with AsyncSqliteSaver.from_conn_string(str(database)) as saver:
        await saver.setup()
        workflow = _fake_workflow(
            model=ToolCallingFakeModel(responses=responses),
            checkpointer=saver,
            recursion_limit=40,
            provider_profile_key="toolcallingfakemodel",
        )

        result = await workflow.execute(
            thread_id="structured-retry-thread",
            story="故事",
            requirements="按人格设定完成完整交付。",
            persona_files={"/persona/project.md": "规则"},
            before_stage=before_stage,
            approve_stage=approve_stage,
            **_episode_hook_kwargs()[0],
        )

    assert result.content_package.episode_scripts == "第 1 集\n事实1\n钩子1"
    assert attempted.count(InternalStage.SELECTING_L0_VARIANT) == 1
    assert approved[0] is InternalStage.SELECTING_L0_VARIANT


@pytest.mark.asyncio
async def test_contract_episode_missing_state_delta_is_corrected_within_stage(
    tmp_path: Path,
) -> None:
    database = tmp_path / "checkpoints.sqlite3"
    responses = _successful_responses()
    script_result_index = next(
        index
        for index, message in enumerate(responses)
        if message.tool_calls and message.tool_calls[0]["name"] == "ScriptWriterResult"
    )
    responses.insert(
        script_result_index,
        _tool_call(
            "ScriptWriterResult",
            {
                "stage": "generating_episode_scripts",
                "episode_number": 1,
                "content": "事实1\n钩子1",
                "state_delta": None,
            },
            99,
        ),
    )
    attempted: list[InternalStage] = []
    approved: list[InternalStage] = []

    async def before_stage(stage: InternalStage) -> int:
        attempted.append(stage)
        return 1

    async def approve_stage(stage: InternalStage, _: dict[str, Any]) -> None:
        approved.append(stage)

    episode_hooks, episode_attempts = _episode_hook_kwargs()
    async with AsyncSqliteSaver.from_conn_string(str(database)) as saver:
        await saver.setup()
        workflow = _fake_workflow(
            model=ToolCallingFakeModel(responses=responses),
            checkpointer=saver,
            recursion_limit=40,
            provider_profile_key="toolcallingfakemodel",
        )

        result = await workflow.execute(
            thread_id="missing-state-delta-retry-thread",
            story="故事",
            requirements="按人格设定完成完整交付。",
            persona_files={"/persona/project.md": "规则"},
            before_stage=before_stage,
            approve_stage=approve_stage,
            **episode_hooks,
        )

    assert result.content_package.episode_scripts == "第 1 集\n事实1\n钩子1"
    assert episode_attempts == [1]
    assert InternalStage.GENERATING_EPISODE_SCRIPTS in approved


@pytest.mark.asyncio
async def test_missing_structured_result_is_corrected_once_within_stage(
    tmp_path: Path,
) -> None:
    database = tmp_path / "checkpoints.sqlite3"
    responses = _successful_responses()
    responses.insert(1, AIMessage(content="The workspace artifact is complete."))
    attempted: list[InternalStage] = []
    approved: list[InternalStage] = []

    async def before_stage(stage: InternalStage) -> int:
        attempted.append(stage)
        return 1

    async def approve_stage(stage: InternalStage, _: dict[str, Any]) -> None:
        approved.append(stage)

    async with AsyncSqliteSaver.from_conn_string(str(database)) as saver:
        await saver.setup()
        workflow = _fake_workflow(
            model=ToolCallingFakeModel(responses=responses),
            checkpointer=saver,
            recursion_limit=40,
            provider_profile_key="toolcallingfakemodel",
        )

        result = await workflow.execute(
            thread_id="missing-structured-result-thread",
            story="故事",
            requirements="按人格设定完成完整交付。",
            persona_files={"/persona/project.md": "规则"},
            before_stage=before_stage,
            approve_stage=approve_stage,
            **_episode_hook_kwargs()[0],
        )

    assert result.content_package.story_outline == "故事大纲"
    assert attempted.count(InternalStage.SELECTING_L0_VARIANT) == 1
    assert approved[0] is InternalStage.SELECTING_L0_VARIANT


@pytest.mark.asyncio
async def test_missing_structured_result_fails_after_one_correction(
    tmp_path: Path,
) -> None:
    database = tmp_path / "checkpoints.sqlite3"
    responses = _successful_responses()
    responses[1:1] = [
        AIMessage(content="The workspace artifact is complete."),
        AIMessage(content="Still returning prose."),
    ]
    attempted: list[InternalStage] = []

    async def before_stage(stage: InternalStage) -> int:
        attempted.append(stage)
        return 1

    async def approve_stage(_: InternalStage, __: dict[str, Any]) -> None:
        raise AssertionError("The invalid stage must not be approved")

    async with AsyncSqliteSaver.from_conn_string(str(database)) as saver:
        await saver.setup()
        workflow = _fake_workflow(
            model=ToolCallingFakeModel(responses=responses),
            checkpointer=saver,
            recursion_limit=40,
            provider_profile_key="toolcallingfakemodel",
        )

        with pytest.raises(AgentProtocolError, match="invalid structured output"):
            await workflow.execute(
                thread_id="missing-structured-result-fail-thread",
                story="故事",
                requirements="按人格设定完成完整交付。",
                persona_files={"/persona/project.md": "规则"},
                before_stage=before_stage,
                approve_stage=approve_stage,
                **_episode_hook_kwargs()[0],
            )

    assert attempted == [InternalStage.SELECTING_L0_VARIANT]


@pytest.mark.asyncio
async def test_chinese_language_mismatch_is_repaired_before_checkpoint() -> None:
    attempted: list[InternalStage] = []
    approved: list[dict[str, Any]] = []
    descriptions: list[str] = []

    async def before_stage(stage: InternalStage) -> int:
        attempted.append(stage)
        return 1

    async def approve_stage(_: InternalStage, payload: dict[str, Any]) -> None:
        approved.append(payload)

    middleware = StageGuardMiddleware(
        before_stage=before_stage,
        approve_stage=approve_stage,
        approved_stages=set(),
        output_language=SIMPLIFIED_CHINESE,
    )
    request = ToolCallRequest(
        tool_call={
            "name": "task",
            "args": {
                "description": "[stage=selecting_l0_variant] choose the L0 variant",
                "subagent_type": "story_architect",
            },
            "id": "call-language-repair",
            "type": "tool_call",
        },
        tool=None,
        state={},
        runtime=None,
    )

    async def handler(subagent_request: ToolCallRequest) -> ToolMessage:
        descriptions.append(subagent_request.tool_call["args"]["description"])
        rationale = (
            "The daughter actively chooses to uncover the family secret."
            if len(descriptions) == 1
            else "女儿主动选择揭开家庭秘密，符合创作内核。"
        )
        return ToolMessage(
            content=json.dumps(
                {
                    "stage": "selecting_l0_variant",
                    "content": None,
                    "character_biographies": None,
                    "relationship_logic": None,
                    "selected_l0_variant": "L0-B",
                    "selection_rationale": rationale,
                },
                ensure_ascii=False,
            ),
            tool_call_id="call-language-repair",
        )

    await middleware.awrap_tool_call(request, handler)

    assert attempted == [InternalStage.SELECTING_L0_VARIANT]
    assert len(descriptions) == 2
    assert all("简体中文" in description for description in descriptions)
    assert "violated the output language contract" in descriptions[1]
    assert approved[0]["selection_rationale"].startswith("女儿主动")


def test_chinese_language_guard_covers_story_contract_narrative_fields() -> None:
    contract_payload = _story_contract().model_dump(mode="json")
    contract_payload["characters"][0]["role"] = "Investigator of a buried family secret"
    planner_result = EpisodePlannerResult.model_validate(
        {
            "stage": "generating_episode_outline",
            "content": "完整分集大纲。",
            "episode_count": 1,
            "episodes": [{"episode_number": 1, "plan": "调查者发现关键证据。"}],
            "story_contract": contract_payload,
        }
    )

    with pytest.raises(AgentProtocolError, match="invalid structured output"):
        _validate_result_language(
            planner_result,
            output_language=SIMPLIFIED_CHINESE,
            stage=InternalStage.GENERATING_EPISODE_OUTLINE,
        )


@pytest.mark.parametrize("unit", ["years", "kg", "km", "USD", "CNY", "m/s"])
def test_contract_language_guard_allows_character_name_subject_and_typed_unit(
    unit: str,
) -> None:
    contract_payload = _story_contract().model_dump(mode="json")
    contract_payload["characters"][0]["name"] = "Alice"
    contract_payload["facts"][0].update(
        {
            "subject": "Alice",
            "predicate": "年龄",
            "kind": "duration",
            "value": "10",
            "unit": unit,
        }
    )

    def planner_result() -> EpisodePlannerResult:
        return EpisodePlannerResult.model_validate(
            {
                "stage": "generating_episode_outline",
                "content": "完整分集大纲。",
                "episode_count": 1,
                "episodes": [{"episode_number": 1, "plan": "调查者发现关键证据。"}],
                "story_contract": contract_payload,
            }
        )

    _validate_result_language(
        planner_result(),
        output_language=SIMPLIFIED_CHINESE,
        stage=InternalStage.GENERATING_EPISODE_OUTLINE,
    )


def test_language_retry_translates_relative_timeline_but_locks_contract_version() -> None:
    contract_payload = _story_contract().model_dump(mode="json")
    contract_payload["timeline"][0]["when"] = "The next morning"

    def planner_result(contract: dict[str, Any]) -> EpisodePlannerResult:
        return EpisodePlannerResult.model_validate(
            {
                "stage": "generating_episode_outline",
                "content": "完整分集大纲。",
                "episode_count": 1,
                "episodes": [{"episode_number": 1, "plan": "调查者发现关键证据。"}],
                "story_contract": contract,
            }
        )

    original = planner_result(contract_payload)
    translated_payload = json.loads(json.dumps(contract_payload))
    translated_payload["timeline"][0]["when"] = "次日清晨"
    translated = planner_result(translated_payload)
    changed_version_payload = json.loads(json.dumps(translated_payload))
    changed_version_payload["version"] = 2
    changed_version = planner_result(changed_version_payload)

    with pytest.raises(AgentProtocolError, match="invalid structured output"):
        _validate_result_language(
            original,
            output_language=SIMPLIFIED_CHINESE,
            stage=InternalStage.GENERATING_EPISODE_OUTLINE,
        )

    original_fingerprint = _language_retry_fingerprint(original)
    assert _language_retry_matches(
        original_fingerprint,
        _language_retry_fingerprint(translated),
    )
    assert not _language_retry_matches(
        original_fingerprint,
        _language_retry_fingerprint(changed_version),
    )


def test_language_retry_allows_variant_label_translation_but_locks_machine_id() -> None:
    english_label = StoryArchitectResult(
        stage="selecting_l0_variant",
        selected_l0_variant="Active Choice",
        selection_rationale="The protagonist acts first.",
    )
    translated_label = StoryArchitectResult(
        stage="selecting_l0_variant",
        selected_l0_variant="主动选择",
        selection_rationale="主角率先行动。",
    )
    machine_variant = StoryArchitectResult(
        stage="selecting_l0_variant",
        selected_l0_variant="L0-B",
        selection_rationale="主角率先行动。",
    )
    changed_machine_variant = StoryArchitectResult(
        stage="selecting_l0_variant",
        selected_l0_variant="L0-C",
        selection_rationale="主角率先行动。",
    )
    bilingual_label = StoryArchitectResult(
        stage="selecting_l0_variant",
        selected_l0_variant="克制情感悬疑（Restrained Emotional Suspense）",
        selection_rationale="主角率先行动。",
    )
    stripped_label = StoryArchitectResult(
        stage="selecting_l0_variant",
        selected_l0_variant="克制情感悬疑",
        selection_rationale="主角率先行动。",
    )
    changed_choice = StoryArchitectResult(
        stage="selecting_l0_variant",
        selected_l0_variant="爆笑荒诞喜剧",
        selection_rationale="主角率先行动。",
    )

    assert _language_retry_matches(
        _language_retry_fingerprint(english_label),
        _language_retry_fingerprint(translated_label),
    )
    assert not _language_retry_matches(
        _language_retry_fingerprint(machine_variant),
        _language_retry_fingerprint(changed_machine_variant),
    )
    assert _language_retry_matches(
        _language_retry_fingerprint(bilingual_label),
        _language_retry_fingerprint(stripped_label),
    )
    assert not _language_retry_matches(
        _language_retry_fingerprint(bilingual_label),
        _language_retry_fingerprint(changed_choice),
    )


def test_episode_language_retry_locks_already_chinese_creative_fields() -> None:
    contract_payload = _story_contract().model_dump(mode="json")
    contract_payload["characters"][0]["role"] = "Investigator"

    def planner_result(contract: dict[str, Any], plan: str = "调查者发现证据。") -> Any:
        return EpisodePlannerResult.model_validate(
            {
                "stage": "generating_episode_outline",
                "content": "完整分集大纲。",
                "episode_count": 1,
                "episodes": [{"episode_number": 1, "plan": plan}],
                "story_contract": contract,
            }
        )

    translated_payload = json.loads(json.dumps(contract_payload))
    translated_payload["characters"][0]["role"] = "调查者"
    original = planner_result(contract_payload)
    translated = planner_result(translated_payload)

    assert _language_retry_matches(
        _language_retry_fingerprint(original),
        _language_retry_fingerprint(translated),
    )
    assert not _language_retry_matches(
        _language_retry_fingerprint(original),
        _language_retry_fingerprint(planner_result(translated_payload, plan="主角放弃调查。")),
    )

    changed_fact = json.loads(json.dumps(translated_payload))
    changed_fact["facts"][0]["predicate"] = "从未发生"
    changed_fact["facts"][0]["value"] = "改写后的事实"
    assert not _language_retry_matches(
        _language_retry_fingerprint(original),
        _language_retry_fingerprint(planner_result(changed_fact)),
    )


def test_chinese_language_guard_rejects_bilingual_l0_variant_title() -> None:
    result = StoryArchitectResult(
        stage="selecting_l0_variant",
        selected_l0_variant="克制情感悬疑（Restrained Emotional Suspense）",
        selection_rationale="以人物关系推动真相揭晓。",
    )

    with pytest.raises(AgentProtocolError) as error:
        _validate_result_language(
            result,
            output_language=SIMPLIFIED_CHINESE,
            stage=InternalStage.SELECTING_L0_VARIANT,
        )

    assert "remove every appended English translation" in (error.value.repair_instruction or "")


@pytest.mark.asyncio
async def test_l0_language_gloss_is_repaired_deterministically_without_model_call() -> None:
    descriptions: list[str] = []

    async def before_stage(_: InternalStage) -> int:
        return 1

    async def approve_stage(_: InternalStage, __: dict[str, Any]) -> None:
        raise AssertionError("The helper must not approve a stage")

    middleware = StageGuardMiddleware(
        before_stage=before_stage,
        approve_stage=approve_stage,
        approved_stages=set(),
        output_language=SIMPLIFIED_CHINESE,
    )
    request = ToolCallRequest(
        tool_call={
            "name": "task",
            "args": {
                "description": "[stage=selecting_l0_variant] select L0",
                "subagent_type": "story_architect",
            },
            "id": "call-l0-language",
            "type": "tool_call",
        },
        tool=None,
        state={},
        runtime=None,
    )

    async def handler(language_request: ToolCallRequest) -> ToolMessage:
        descriptions.append(language_request.tool_call["args"]["description"])
        payload = {
            "stage": "selecting_l0_variant",
            "content": None,
            "character_biographies": None,
            "relationship_logic": None,
            "selected_l0_variant": "克制情感悬疑（Restrained Emotional Suspense）",
            "selection_rationale": "用人物关系推动真相揭晓。",
        }
        return ToolMessage(
            content=json.dumps(payload, ensure_ascii=False),
            tool_call_id="call-l0-language",
        )

    _, payload = await middleware._call_structured_stage(
        InternalStage.SELECTING_L0_VARIANT,
        request,
        handler,
        request.tool_call["args"],
    )

    # The appended English gloss is stripped in code; no model retry happens.
    assert len(descriptions) == 1
    assert payload["selected_l0_variant"] == "克制情感悬疑"
    assert payload["selection_rationale"] == "用人物关系推动真相揭晓。"


@pytest.mark.asyncio
async def test_l0_language_retry_merges_translation_and_keeps_locked_fields() -> None:
    descriptions: list[str] = []
    captured_source: list[dict[str, Any]] = []

    async def before_stage(_: InternalStage) -> int:
        return 1

    async def approve_stage(_: InternalStage, __: dict[str, Any]) -> None:
        raise AssertionError("The helper must not approve a stage")

    middleware = StageGuardMiddleware(
        before_stage=before_stage,
        approve_stage=approve_stage,
        approved_stages=set(),
        output_language=SIMPLIFIED_CHINESE,
    )
    request = ToolCallRequest(
        tool_call={
            "name": "task",
            "args": {
                "description": "[stage=selecting_l0_variant] select L0",
                "subagent_type": "story_architect",
            },
            "id": "call-l0-language",
            "type": "tool_call",
        },
        tool=None,
        state={},
        runtime=None,
    )

    async def handler(language_request: ToolCallRequest) -> ToolMessage:
        descriptions.append(language_request.tool_call["args"]["description"])
        if len(descriptions) == 1:
            payload = {
                "stage": "selecting_l0_variant",
                "content": None,
                "character_biographies": None,
                "relationship_logic": None,
                "selected_l0_variant": "克制情感悬疑",
                "selection_rationale": (
                    "The relationships drive the reveal of the truth in every episode."
                ),
            }
        else:
            files = language_request.state["files"]
            source = json.loads(files["/workspace/result_to_translate.json"]["content"])
            captured_source.append(source)
            payload = {
                **source,
                # The model translates the flagged field but also tampers with
                # an already-Chinese locked field; the merge must ignore that.
                "selected_l0_variant": "全新的创作方向",
                "selection_rationale": "用人物关系推动真相揭晓。",
            }
        return ToolMessage(
            content=json.dumps(payload, ensure_ascii=False),
            tool_call_id="call-l0-language",
        )

    _, payload = await middleware._call_structured_stage(
        InternalStage.SELECTING_L0_VARIANT,
        request,
        handler,
        request.tool_call["args"],
    )

    assert len(descriptions) == 2
    assert "translate only" in descriptions[1]
    assert "do not make a new creative choice" in descriptions[1]
    assert captured_source[0]["selection_rationale"].startswith("The relationships")
    assert payload["selection_rationale"] == "用人物关系推动真相揭晓。"
    assert payload["selected_l0_variant"] == "克制情感悬疑"


def test_chinese_language_guard_covers_script_handoff() -> None:
    contract = _story_contract()
    state_delta = _state_delta(contract, 1)
    state_delta["handoff"] = "The next episode follows the hidden witness."
    script_result = ScriptWriterResult.model_validate(
        {
            "stage": "generating_episode_scripts",
            "episode_number": 1,
            "content": "测试人物：事实1。\n门后传来第二次敲击。",
            "state_delta": state_delta,
        }
    )

    with pytest.raises(AgentProtocolError, match="invalid structured output"):
        _validate_result_language(
            script_result,
            output_language=SIMPLIFIED_CHINESE,
            stage=InternalStage.GENERATING_EPISODE_SCRIPTS,
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("initial_decision", "repaired_decision"),
    [(False, True), (True, False)],
)
async def test_language_repair_cannot_change_quality_gate_decision(
    initial_decision: bool,
    repaired_decision: bool,
) -> None:
    descriptions: list[str] = []
    approved: list[dict[str, Any]] = []

    async def before_stage(_: InternalStage) -> int:
        return 1

    async def approve_stage(_: InternalStage, payload: dict[str, Any]) -> None:
        approved.append(payload)

    middleware = StageGuardMiddleware(
        before_stage=before_stage,
        approve_stage=approve_stage,
        approved_stages={
            InternalStage.SELECTING_L0_VARIANT,
            InternalStage.GENERATING_STORY_OUTLINE,
            InternalStage.GENERATING_CHARACTER_RELATIONSHIPS,
            InternalStage.GENERATING_EPISODE_OUTLINE,
            InternalStage.GENERATING_EPISODE_SCRIPTS,
        },
        output_language=SIMPLIFIED_CHINESE,
    )
    request = ToolCallRequest(
        tool_call={
            "name": "task",
            "args": {
                "description": "[stage=accepting_l0] review the L0 gate",
                "subagent_type": "quality_reviewer",
            },
            "id": "call-gate-language-repair",
            "type": "tool_call",
        },
        tool=None,
        state={},
        runtime=None,
    )

    async def handler(review_request: ToolCallRequest) -> ToolMessage:
        descriptions.append(review_request.tool_call["args"]["description"])
        is_repair = len(descriptions) == 2
        return ToolMessage(
            content=json.dumps(
                {
                    "stage": "accepting_l0",
                    "passed": repaired_decision if is_repair else initial_decision,
                    "evidence": "审核结论已翻译。" if is_repair else "The L0 review is complete.",
                    "feedback_handling": [],
                },
                ensure_ascii=False,
            ),
            tool_call_id="call-gate-language-repair",
        )

    # The merge keeps the original decision by construction: the model's
    # flipped verdict is ignored and only the translated evidence is spliced.
    if initial_decision:
        await middleware.awrap_tool_call(request, handler)
        assert approved and approved[0]["passed"] is True
        assert approved[0]["evidence"] == "审核结论已翻译。"
    else:
        with pytest.raises(QualityGateRejectedError):
            await middleware.awrap_tool_call(request, handler)
        assert not approved

    assert len(descriptions) == 2


@pytest.mark.asyncio
async def test_semantic_review_language_repair_preserves_failed_decision() -> None:
    descriptions: list[str] = []

    async def before_stage(_: InternalStage) -> int:
        return 1

    async def approve_stage(_: InternalStage, __: dict[str, Any]) -> None:
        raise AssertionError("The helper must not approve a stage")

    middleware = StageGuardMiddleware(
        before_stage=before_stage,
        approve_stage=approve_stage,
        approved_stages=set(),
        output_language=SIMPLIFIED_CHINESE,
    )
    request = ToolCallRequest(
        tool_call={
            "name": "task",
            "args": {
                "description": "[stage=generating_episode_outline] review contract",
                "subagent_type": "episode_planner",
            },
            "id": "call-review-language",
            "type": "tool_call",
        },
        tool=None,
        state={},
        runtime=None,
    )

    async def handler(review_request: ToolCallRequest) -> ToolMessage:
        descriptions.append(review_request.tool_call["args"]["description"])
        chinese = len(descriptions) == 2
        if chinese:
            source = review_request.state["files"]["/workspace/review_to_translate.json"]
            original_review = json.loads(source["content"])
            assert original_review["passed"] is False
            assert original_review["evidence"] == "The contract omits a fact."
        return ToolMessage(
            content=json.dumps(
                {
                    "passed": False,
                    "evidence": "合同遗漏了既定事实。" if chinese else "The contract omits a fact.",
                    "issues": [
                        {
                            "code": "missing_fact",
                            "message": "必须补齐事实。" if chinese else "The fact must be added.",
                            "contract_refs": [],
                            "script_excerpt": None,
                        }
                    ],
                },
                ensure_ascii=False,
            ),
            tool_call_id="call-review-language",
        )

    review = await middleware._invoke_semantic_reviewer(
        request=request,
        handler=handler,
        subagent_type="canon_reviewer",
        description="Review the contract.",
        files={},
        schema=CanonReviewerResult,
        stage=InternalStage.GENERATING_EPISODE_OUTLINE,
    )

    assert len(descriptions) == 2
    assert "without changing the review decision" in descriptions[1]
    assert "do not perform a new review" in descriptions[1]
    assert review.passed is False
    assert review.evidence == "合同遗漏了既定事实。"


@pytest.mark.asyncio
async def test_semantic_review_allows_chinese_issue_with_many_stable_ids() -> None:
    calls = 0

    async def before_stage(_: InternalStage) -> int:
        return 1

    async def approve_stage(_: InternalStage, __: dict[str, Any]) -> None:
        raise AssertionError("The helper must not approve a stage")

    middleware = StageGuardMiddleware(
        before_stage=before_stage,
        approve_stage=approve_stage,
        approved_stages=set(),
        output_language=SIMPLIFIED_CHINESE,
    )
    request = ToolCallRequest(
        tool_call={
            "name": "task",
            "args": {
                "description": "[stage=generating_episode_scripts] review episode",
                "subagent_type": "script_writer",
            },
            "id": "call-review-stable-ids",
            "type": "tool_call",
        },
        tool=None,
        state={},
        runtime=None,
    )

    async def handler(_: ToolCallRequest) -> ToolMessage:
        nonlocal calls
        calls += 1
        return ToolMessage(
            content=json.dumps(
                {
                    "passed": False,
                    "evidence": "本集仍有连续性问题。",
                    "issues": [
                        {
                            "code": "missing_locked_beats",
                            "message": (
                                "fact_keep_secret、fact_station_deadline、clue_old_watch、"
                                "clue_rescue_log、obligation_ep6、lin_xia、chen_station、E6 "
                                "均未按合同兑现。"
                            ),
                            "contract_refs": ["fact_keep_secret", "obligation_ep6"],
                            "script_excerpt": None,
                        }
                    ],
                },
                ensure_ascii=False,
            ),
            tool_call_id="call-review-stable-ids",
        )

    review = await middleware._invoke_semantic_reviewer(
        request=request,
        handler=handler,
        subagent_type="episode_reviewer",
        description="Review the episode.",
        files={},
        schema=EpisodeReviewerResult,
        stage=InternalStage.GENERATING_EPISODE_SCRIPTS,
    )

    assert calls == 1
    assert review.passed is False
    assert review.issues[0].code == "missing_locked_beats"


@pytest.mark.asyncio
async def test_semantic_language_repair_cannot_change_review_decision() -> None:
    calls = 0

    async def before_stage(_: InternalStage) -> int:
        return 1

    async def approve_stage(_: InternalStage, __: dict[str, Any]) -> None:
        raise AssertionError("The helper must not approve a stage")

    middleware = StageGuardMiddleware(
        before_stage=before_stage,
        approve_stage=approve_stage,
        approved_stages=set(),
        output_language=SIMPLIFIED_CHINESE,
    )
    request = ToolCallRequest(
        tool_call={
            "name": "task",
            "args": {
                "description": "[stage=generating_episode_outline] review contract",
                "subagent_type": "episode_planner",
            },
            "id": "call-review-decision-lock",
            "type": "tool_call",
        },
        tool=None,
        state={},
        runtime=None,
    )

    async def handler(_: ToolCallRequest) -> ToolMessage:
        nonlocal calls
        calls += 1
        payload = (
            {"passed": True, "evidence": "合同审核通过。", "issues": []}
            if calls == 2
            else {
                "passed": False,
                "evidence": "The contract omits a fact.",
                "issues": [
                    {
                        "code": "missing_fact",
                        "message": "The fact must be added.",
                        "contract_refs": [],
                        "script_excerpt": None,
                    }
                ],
            }
        )
        return ToolMessage(
            content=json.dumps(payload, ensure_ascii=False),
            tool_call_id="call-review-decision-lock",
        )

    with pytest.raises(AgentProtocolError, match="changed the semantic review decision"):
        await middleware._invoke_semantic_reviewer(
            request=request,
            handler=handler,
            subagent_type="canon_reviewer",
            description="Review the contract.",
            files={},
            schema=CanonReviewerResult,
            stage=InternalStage.GENERATING_EPISODE_OUTLINE,
        )

    assert calls == 2


@pytest.mark.asyncio
async def test_outline_repair_gets_one_bounded_structured_correction() -> None:
    corrections: list[str | None] = []

    async def before_stage(_: InternalStage) -> int:
        return 1

    async def approve_stage(_: InternalStage, __: dict[str, Any]) -> None:
        return None

    contract = _story_contract()
    candidate = {
        "stage": "generating_episode_outline",
        "content": "旧分集大纲",
        "episode_count": 1,
        "episodes": [{"episode_number": 1, "plan": "第一集计划"}],
        "story_contract": contract.model_dump(mode="json"),
    }
    review = CanonReviewerResult(
        passed=False,
        evidence="大纲文字需要修复。",
        issues=[{"code": "stale_text", "message": "替换旧文字。"}],
    )

    async def generate_patch(
        _: Mapping[str, Any],
        __: CanonReviewerResult,
        ___: int,
        correction: str | None,
    ) -> Any:
        corrections.append(correction)
        if len(corrections) == 1:
            return "I will analyze the repair first."
        return {
            "stage": "generating_episode_outline",
            "content_replacements": [{"old": "旧分集大纲", "new": "修复后的分集大纲"}],
            "json_edits": [],
        }

    middleware = StageGuardMiddleware(
        before_stage=before_stage,
        approve_stage=approve_stage,
        approved_stages=set(),
        output_language=SIMPLIFIED_CHINESE,
        generate_outline_patch=generate_patch,
    )

    repaired = await middleware._invoke_outline_repair(
        candidate=candidate,
        review=review,
        repair_round=1,
    )

    assert len(corrections) == 2
    assert corrections[0] is None
    assert corrections[1] is not None
    assert "exactly one corrected OutlineRepairPatch" in corrections[1]
    assert repaired["content"] == "修复后的分集大纲"


@pytest.mark.asyncio
async def test_outline_repair_fails_after_one_correction_with_safe_chinese_message() -> None:
    corrections: list[str | None] = []

    async def before_stage(_: InternalStage) -> int:
        return 1

    async def approve_stage(_: InternalStage, __: dict[str, Any]) -> None:
        return None

    async def generate_invalid_patch(
        _: Mapping[str, Any],
        __: CanonReviewerResult,
        ___: int,
        correction: str | None,
    ) -> Any:
        corrections.append(correction)
        return "not a structured patch"

    middleware = StageGuardMiddleware(
        before_stage=before_stage,
        approve_stage=approve_stage,
        approved_stages=set(),
        output_language=SIMPLIFIED_CHINESE,
        generate_outline_patch=generate_invalid_patch,
    )
    contract = _story_contract()
    candidate = {
        "stage": "generating_episode_outline",
        "content": "旧分集大纲",
        "episode_count": 1,
        "episodes": [{"episode_number": 1, "plan": "第一集计划"}],
        "story_contract": contract.model_dump(mode="json"),
    }
    review = CanonReviewerResult(
        passed=False,
        evidence="大纲文字需要修复。",
        issues=[{"code": "stale_text", "message": "替换旧文字。"}],
    )

    with pytest.raises(AgentProtocolError) as error:
        await middleware._invoke_outline_repair(
            candidate=candidate,
            review=review,
            repair_round=1,
        )

    assert len(corrections) == 2
    assert error.value.safe_message == "分集大纲修复补丁未通过结构化校验。"


@pytest.mark.asyncio
async def test_outline_repair_preserves_relay_errors_for_worker_recovery() -> None:
    async def before_stage(_: InternalStage) -> int:
        return 1

    async def approve_stage(_: InternalStage, __: dict[str, Any]) -> None:
        return None

    request = httpx.Request("POST", "https://relay.example/v1/chat/completions")

    async def unavailable_patch(
        _: Mapping[str, Any],
        __: CanonReviewerResult,
        ___: int,
        ____: str | None,
    ) -> Any:
        raise httpx.ReadTimeout("relay timed out", request=request)

    middleware = StageGuardMiddleware(
        before_stage=before_stage,
        approve_stage=approve_stage,
        approved_stages=set(),
        generate_outline_patch=unavailable_patch,
    )
    contract = _story_contract()
    candidate = {
        "stage": "generating_episode_outline",
        "content": "分集大纲",
        "episode_count": 1,
        "episodes": [{"episode_number": 1, "plan": "第一集计划"}],
        "story_contract": contract.model_dump(mode="json"),
    }
    review = CanonReviewerResult(
        passed=False,
        evidence="需要修复。",
        issues=[{"code": "stale_text", "message": "替换旧文字。"}],
    )

    with pytest.raises(httpx.ReadTimeout):
        await middleware._invoke_outline_repair(
            candidate=candidate,
            review=review,
            repair_round=1,
        )


@pytest.mark.asyncio
async def test_repair_subagent_gets_one_bounded_language_retry() -> None:
    descriptions: list[str] = []
    captured_source: list[dict[str, Any]] = []

    async def before_stage(_: InternalStage) -> int:
        return 1

    async def approve_stage(_: InternalStage, __: dict[str, Any]) -> None:
        raise AssertionError("The helper must not approve a stage")

    middleware = StageGuardMiddleware(
        before_stage=before_stage,
        approve_stage=approve_stage,
        approved_stages=set(),
        output_language=SIMPLIFIED_CHINESE,
    )
    request = ToolCallRequest(
        tool_call={
            "name": "task",
            "args": {
                "description": "[stage=generating_episode_outline] repair contract",
                "subagent_type": "episode_planner",
            },
            "id": "call-repair-language",
            "type": "tool_call",
        },
        tool=None,
        state={},
        runtime=None,
    )
    contract_payload = _story_contract().model_dump(mode="json")

    async def handler(repair_request: ToolCallRequest) -> ToolMessage:
        descriptions.append(repair_request.tool_call["args"]["description"])
        if len(descriptions) == 1:
            result_payload = {
                "stage": "generating_episode_outline",
                "content": "完整分集大纲。",
                "episode_count": 1,
                "episodes": [{"episode_number": 1, "plan": "调查者发现证据。"}],
                "story_contract": json.loads(json.dumps(contract_payload)),
            }
            result_payload["story_contract"]["characters"][0]["role"] = (
                "Investigator of a buried family secret"
            )
        else:
            files = repair_request.state["files"]
            result_payload = json.loads(files["/workspace/result_to_translate.json"]["content"])
            captured_source.append(json.loads(json.dumps(result_payload)))
            result_payload["story_contract"]["characters"][0]["role"] = "调查者"
        return ToolMessage(
            content=json.dumps(result_payload, ensure_ascii=False),
            tool_call_id="call-repair-language",
        )

    _, payload = await middleware._invoke_repair_subagent(
        request=request,
        handler=handler,
        subagent_type="canon_repair",
        description="Repair the contract.",
        files={},
        schema=EpisodePlannerResult,
        stage=InternalStage.GENERATING_EPISODE_OUTLINE,
    )

    assert len(descriptions) == 2
    assert "Read /workspace/result_to_translate.json" in descriptions[1]
    assert "Do not perform a new repair" in descriptions[1]
    assert captured_source[0]["story_contract"]["characters"][0]["role"].startswith("Investigator")
    assert payload["story_contract"]["characters"][0]["role"] == "调查者"


@pytest.mark.asyncio
async def test_episode_repair_corrects_missing_state_delta_once() -> None:
    descriptions: list[str] = []

    async def before_stage(_: InternalStage) -> int:
        return 1

    async def approve_stage(_: InternalStage, __: dict[str, Any]) -> None:
        raise AssertionError("The helper must not approve a stage")

    middleware = StageGuardMiddleware(
        before_stage=before_stage,
        approve_stage=approve_stage,
        approved_stages=set(),
        output_language=SIMPLIFIED_CHINESE,
    )
    request = ToolCallRequest(
        tool_call={
            "name": "task",
            "args": {
                "description": "[stage=generating_episode_scripts] repair episode 1",
                "subagent_type": "episode_repair",
            },
            "id": "call-repair-state",
            "type": "tool_call",
        },
        tool=None,
        state={},
        runtime=None,
    )
    contract = _story_contract()

    async def handler(repair_request: ToolCallRequest) -> ToolMessage:
        descriptions.append(repair_request.tool_call["args"]["description"])
        if len(descriptions) == 2:
            assert "state_delta: missing" in descriptions[-1]
        payload = {
            "stage": "generating_episode_scripts",
            "episode_number": 1,
            "content": "事实1\n钩子1",
        }
        if len(descriptions) == 2:
            payload["state_delta"] = _state_delta(contract, 1)
        return ToolMessage(
            content=json.dumps(payload, ensure_ascii=False),
            tool_call_id="call-repair-state",
        )

    _, payload = await middleware._invoke_repair_subagent(
        request=request,
        handler=handler,
        subagent_type="episode_repair",
        description="Repair episode 1.",
        files={},
        schema=ScriptWriterResult,
        stage=InternalStage.GENERATING_EPISODE_SCRIPTS,
        expected_episode_number=1,
    )

    assert len(descriptions) == 2
    assert payload["state_delta"] == _state_delta(contract, 1)


@pytest.mark.asyncio
async def test_episode_repair_fails_after_repeated_missing_state_delta() -> None:
    calls = 0

    async def before_stage(_: InternalStage) -> int:
        return 1

    async def approve_stage(_: InternalStage, __: dict[str, Any]) -> None:
        raise AssertionError("The helper must not approve a stage")

    middleware = StageGuardMiddleware(
        before_stage=before_stage,
        approve_stage=approve_stage,
        approved_stages=set(),
        output_language=SIMPLIFIED_CHINESE,
    )
    request = ToolCallRequest(
        tool_call={
            "name": "task",
            "args": {
                "description": "[stage=generating_episode_scripts] repair episode 1",
                "subagent_type": "episode_repair",
            },
            "id": "call-repair-state-fail",
            "type": "tool_call",
        },
        tool=None,
        state={},
        runtime=None,
    )

    async def handler(_: ToolCallRequest) -> ToolMessage:
        nonlocal calls
        calls += 1
        return ToolMessage(
            content=json.dumps(
                {
                    "stage": "generating_episode_scripts",
                    "episode_number": 1,
                    "content": "事实1\n钩子1",
                },
                ensure_ascii=False,
            ),
            tool_call_id="call-repair-state-fail",
        )

    with pytest.raises(AgentProtocolError, match="invalid structured output"):
        await middleware._invoke_repair_subagent(
            request=request,
            handler=handler,
            subagent_type="episode_repair",
            description="Repair episode 1.",
            files={},
            schema=ScriptWriterResult,
            stage=InternalStage.GENERATING_EPISODE_SCRIPTS,
            expected_episode_number=1,
        )

    assert calls == 2


@pytest.mark.asyncio
async def test_wrong_stage_result_is_not_corrected() -> None:
    attempted: list[InternalStage] = []
    handler_calls = 0

    async def before_stage(stage: InternalStage) -> int:
        attempted.append(stage)
        return 1

    async def approve_stage(_: InternalStage, __: dict[str, Any]) -> None:
        raise AssertionError("The wrong stage must not be approved")

    middleware = StageGuardMiddleware(
        before_stage=before_stage,
        approve_stage=approve_stage,
        approved_stages={
            InternalStage.SELECTING_L0_VARIANT,
            InternalStage.GENERATING_STORY_OUTLINE,
            InternalStage.GENERATING_CHARACTER_RELATIONSHIPS,
        },
    )
    request = ToolCallRequest(
        tool_call={
            "name": "task",
            "args": {
                "description": "[stage=generating_episode_outline] create the outline",
                "subagent_type": "episode_planner",
            },
            "id": "call-wrong-stage",
            "type": "tool_call",
        },
        tool=None,
        state={},
        runtime=None,
    )

    async def handler(_: ToolCallRequest) -> ToolMessage:
        nonlocal handler_calls
        handler_calls += 1
        return ToolMessage(
            content='{"stage":"generating_episode_scripts","content":"wrong stage"}',
            tool_call_id="call-wrong-stage",
        )

    with pytest.raises(AgentProtocolError, match="different stage"):
        await middleware.awrap_tool_call(request, handler)

    assert attempted == [InternalStage.GENERATING_EPISODE_OUTLINE]
    assert handler_calls == 1


@pytest.mark.asyncio
async def test_quality_review_drops_stale_script_when_canonical_payload_is_missing() -> None:
    approved_stages = {
        InternalStage.SELECTING_L0_VARIANT,
        InternalStage.GENERATING_STORY_OUTLINE,
        InternalStage.GENERATING_CHARACTER_RELATIONSHIPS,
        InternalStage.GENERATING_EPISODE_OUTLINE,
        InternalStage.GENERATING_EPISODE_SCRIPTS,
    }

    async def before_stage(_: InternalStage) -> int:
        return 1

    async def approve_stage(_: InternalStage, __: dict[str, Any]) -> None:
        return None

    middleware = StageGuardMiddleware(
        before_stage=before_stage,
        approve_stage=approve_stage,
        approved_stages=approved_stages,
        approved_payloads={
            InternalStage.GENERATING_EPISODE_SCRIPTS: {"stage": "generating_episode_scripts"}
        },
    )
    request = ToolCallRequest(
        tool_call={
            "name": "task",
            "args": {
                "description": "[stage=accepting_l0] review the approved artifacts",
                "subagent_type": "quality_reviewer",
            },
            "id": "call-missing-canonical-script",
            "type": "tool_call",
        },
        tool=None,
        state={
            "files": {
                "/workspace/episode_scripts.md": {
                    "content": "旧工作区剧本",
                    "encoding": "utf-8",
                }
            }
        },
        runtime=None,
    )

    async def handler(review_request: ToolCallRequest) -> ToolMessage:
        assert "/workspace/episode_scripts.md" not in review_request.state["files"]
        return ToolMessage(
            content=(
                '{"stage":"accepting_l0","passed":true,'
                '"evidence":"缺失稿件未被旧文件替代","feedback_handling":[]}'
            ),
            tool_call_id="call-missing-canonical-script",
        )

    await middleware.awrap_tool_call(request, handler)


@pytest.mark.asyncio
async def test_stage_token_is_required_before_any_attempt(tmp_path: Path) -> None:
    database = tmp_path / "checkpoints.sqlite3"
    model = ToolCallingFakeModel(
        responses=[
            _tool_call(
                "task",
                {
                    "description": "missing machine stage token",
                    "subagent_type": "story_architect",
                },
                0,
            )
        ]
    )
    attempted: list[InternalStage] = []

    async def before_stage(stage: InternalStage) -> int:
        attempted.append(stage)
        return 1

    async def approve_stage(_: InternalStage, __: dict[str, Any]) -> None:
        raise AssertionError("No checkpoint may be approved")

    async with AsyncSqliteSaver.from_conn_string(str(database)) as saver:
        await saver.setup()
        workflow = _fake_workflow(
            model=model,
            checkpointer=saver,
            provider_profile_key="toolcallingfakemodel",
        )

        with pytest.raises(AgentProtocolError, match="stage token"):
            await workflow.execute(
                thread_id="invalid-thread",
                story="故事",
                requirements="要求",
                persona_files={"/persona/project.md": "规则"},
                before_stage=before_stage,
                approve_stage=approve_stage,
                **_episode_hook_kwargs()[0],
            )

    assert attempted == []


@pytest.mark.asyncio
async def test_failed_quality_gate_is_not_approved(tmp_path: Path) -> None:
    database = tmp_path / "checkpoints.sqlite3"
    responses = _successful_responses()
    l0_gate_index = _index_of_tool_call(responses, "QualityReviewerResult", occurrence=1)
    responses[l0_gate_index] = _tool_call(
        "QualityReviewerResult",
        {
            "stage": "accepting_l0",
            "passed": False,
            "evidence": "成品没有通过 L0 闸门。",
            "feedback_handling": [],
        },
        l0_gate_index,
    )
    approved: list[InternalStage] = []

    async def before_stage(_: InternalStage) -> int:
        return 1

    async def approve_stage(stage: InternalStage, _: dict[str, Any]) -> None:
        approved.append(stage)

    async with AsyncSqliteSaver.from_conn_string(str(database)) as saver:
        await saver.setup()
        workflow = _fake_workflow(
            model=ToolCallingFakeModel(responses=responses),
            checkpointer=saver,
            provider_profile_key="toolcallingfakemodel",
        )

        with pytest.raises(QualityGateRejectedError, match="Quality gate did not pass") as error:
            await workflow.execute(
                thread_id="failed-gate-thread",
                story="故事",
                requirements="要求",
                persona_files={"/persona/project.md": "规则"},
                before_stage=before_stage,
                approve_stage=approve_stage,
                **_episode_hook_kwargs()[0],
            )

    assert error.value.stage is InternalStage.ACCEPTING_L0
    assert error.value.evidence == "成品没有通过 L0 闸门。"

    assert approved == [
        InternalStage.SELECTING_L0_VARIANT,
        InternalStage.GENERATING_STORY_OUTLINE,
        InternalStage.GENERATING_CHARACTER_RELATIONSHIPS,
        InternalStage.GENERATING_EPISODE_OUTLINE,
        InternalStage.GENERATING_EPISODE_SCRIPTS,
    ]


@pytest.mark.asyncio
async def test_quality_rejection_reuses_thread_and_only_retries_final_gates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database = tmp_path / "checkpoints.sqlite3"
    reviewer_reads: list[str] = []
    original_generate = ToolCallingFakeModel._generate

    def capture_reviewer_reads(
        model: ToolCallingFakeModel,
        messages: list[Any],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        reviewer_reads.extend(
            str(message.content)
            for message in messages
            if isinstance(message, ToolMessage) and message.name == "read_file"
        )
        return original_generate(model, messages, *args, **kwargs)

    monkeypatch.setattr(ToolCallingFakeModel, "_generate", capture_reviewer_reads)
    responses = _successful_responses()
    l4_gate_index = _index_of_tool_call(responses, "QualityReviewerResult", occurrence=2)
    responses[l4_gate_index] = _tool_call(
        "QualityReviewerResult",
        {
            "stage": "accepting_l4",
            "passed": False,
            "evidence": "成品没有通过 L4 闸门。",
            "feedback_handling": [],
        },
        l4_gate_index,
    )
    responses.insert(
        l4_gate_index,
        _tool_call(
            "read_file",
            {"file_path": "/workspace/episode_scripts.md"},
            100,
        ),
    )
    responses.insert(
        l4_gate_index + 1,
        _tool_call(
            "read_file",
            {"file_path": "/workspace/approved-checkpoints.json"},
            101,
        ),
    )
    approved: dict[InternalStage, dict[str, Any]] = {}
    first_attempts: list[InternalStage] = []

    async def before_first_stage(stage: InternalStage) -> int:
        first_attempts.append(stage)
        return 1

    async def approve_stage(stage: InternalStage, payload: dict[str, Any]) -> None:
        approved[stage] = payload

    async with AsyncSqliteSaver.from_conn_string(str(database)) as saver:
        await saver.setup()
        workflow = _fake_workflow(
            model=ToolCallingFakeModel(responses=responses),
            checkpointer=saver,
            provider_profile_key="toolcallingfakemodel",
        )
        with pytest.raises(QualityGateRejectedError):
            await workflow.execute(
                thread_id="quality-retry-thread",
                story="故事",
                requirements="要求",
                persona_files={
                    "/persona/project.md": "规则",
                    "/workspace/episode_scripts.md": "旧工作区剧本",
                    "/workspace/approved-checkpoints.json": "{}",
                },
                before_stage=before_first_stage,
                approve_stage=approve_stage,
                **_episode_hook_kwargs()[0],
            )

    assert not any("not found" in content for content in reviewer_reads)
    assert any("第 1 集" in content and "事实1" in content for content in reviewer_reads)
    assert any("generating_episode_scripts" in content for content in reviewer_reads)
    assert not any("旧工作区剧本" in content for content in reviewer_reads)
    reviewer_reads.clear()

    resumed_attempts: list[InternalStage] = []

    async def before_resumed_stage(stage: InternalStage) -> int:
        resumed_attempts.append(stage)
        return 1

    async with AsyncSqliteSaver.from_conn_string(str(database)) as saver:
        await saver.setup()
        fresh = _successful_responses()
        l4_result_index = _index_of_tool_call(fresh, "QualityReviewerResult", occurrence=2)
        resumed_responses = fresh[l4_result_index - 1 :]
        resumed_responses.insert(
            1,
            _tool_call(
                "read_file",
                {"file_path": "/workspace/episode_scripts.md"},
                102,
            ),
        )
        resumed_responses.insert(
            2,
            _tool_call(
                "read_file",
                {"file_path": "/workspace/approved-checkpoints.json"},
                103,
            ),
        )
        resumed_workflow = _fake_workflow(
            model=ToolCallingFakeModel(responses=resumed_responses),
            checkpointer=saver,
            provider_profile_key="toolcallingfakemodel",
        )
        resumed_episode_hooks, resumed_episode_attempts = _episode_hook_kwargs()
        result = await resumed_workflow.execute(
            thread_id="quality-retry-thread",
            story="故事",
            requirements="要求",
            persona_files={
                "/persona/project.md": "规则",
                "/workspace/episode_scripts.md": "旧工作区剧本",
                "/workspace/approved-checkpoints.json": "{}",
            },
            before_stage=before_resumed_stage,
            approve_stage=approve_stage,
            approved_checkpoints=approved,
            **resumed_episode_hooks,
        )

    assert first_attempts == [
        InternalStage.SELECTING_L0_VARIANT,
        InternalStage.GENERATING_STORY_OUTLINE,
        InternalStage.GENERATING_CHARACTER_RELATIONSHIPS,
        InternalStage.GENERATING_EPISODE_OUTLINE,
        InternalStage.ACCEPTING_L0,
        InternalStage.ACCEPTING_L4,
    ]
    assert resumed_attempts == [InternalStage.ACCEPTING_L4]
    assert resumed_episode_attempts == []
    assert not any("not found" in content for content in reviewer_reads)
    assert any("第 1 集" in content and "事实1" in content for content in reviewer_reads)
    assert any("generating_episode_scripts" in content for content in reviewer_reads)
    assert not any("旧工作区剧本" in content for content in reviewer_reads)
    assert result.content_package.episode_scripts == "第 1 集\n事实1\n钩子1"


@pytest.mark.asyncio
async def test_out_of_order_stage_is_rejected_without_attempt_and_can_recover(
    tmp_path: Path,
) -> None:
    database = tmp_path / "checkpoints.sqlite3"
    model = ToolCallingFakeModel(
        responses=[
            _tool_call(
                "task",
                {
                    "description": (
                        "[stage=generating_episode_outline] skip required story stages"
                    ),
                    "subagent_type": "episode_planner",
                },
                0,
            ),
            *_successful_responses(),
        ]
    )
    attempted: list[InternalStage] = []
    approved: list[InternalStage] = []

    async def before_stage(stage: InternalStage) -> int:
        attempted.append(stage)
        return 1

    async def approve_stage(stage: InternalStage, _: dict[str, Any]) -> None:
        approved.append(stage)

    async with AsyncSqliteSaver.from_conn_string(str(database)) as saver:
        await saver.setup()
        workflow = _fake_workflow(
            model=model,
            checkpointer=saver,
            provider_profile_key="toolcallingfakemodel",
        )

        result = await workflow.execute(
            thread_id="out-of-order-thread",
            story="故事",
            requirements="要求",
            persona_files={"/persona/project.md": "规则"},
            before_stage=before_stage,
            approve_stage=approve_stage,
            **_episode_hook_kwargs()[0],
        )

    expected = [
        InternalStage.SELECTING_L0_VARIANT,
        InternalStage.GENERATING_STORY_OUTLINE,
        InternalStage.GENERATING_CHARACTER_RELATIONSHIPS,
        InternalStage.GENERATING_EPISODE_OUTLINE,
        InternalStage.GENERATING_EPISODE_SCRIPTS,
        InternalStage.ACCEPTING_L0,
        InternalStage.ACCEPTING_L4,
    ]
    assert result.content_package.episode_scripts == "第 1 集\n事实1\n钩子1"
    assert attempted == [
        stage for stage in expected if stage is not InternalStage.GENERATING_EPISODE_SCRIPTS
    ]
    assert approved == expected


@pytest.mark.asyncio
async def test_restart_reuses_thread_checkpoint_and_skips_approved_stage(
    tmp_path: Path,
) -> None:
    database = tmp_path / "checkpoints.sqlite3"
    approved: dict[InternalStage, dict[str, Any]] = {}
    attempts: list[InternalStage] = []

    async def before_stage(stage: InternalStage) -> int:
        attempts.append(stage)
        return 1

    async def approve_stage(stage: InternalStage, payload: dict[str, Any]) -> None:
        approved[stage] = payload

    responses = _successful_responses()
    interrupted_model = ToolCallingFakeModel(
        responses=[
            *responses[:2],
            AIMessage(content="interrupted before a complete structured response"),
        ]
    )
    async with AsyncSqliteSaver.from_conn_string(str(database)) as saver:
        await saver.setup()
        workflow = _fake_workflow(
            model=interrupted_model,
            checkpointer=saver,
            provider_profile_key="toolcallingfakemodel",
        )
        with pytest.raises(
            AgentProtocolError,
            match="Supervisor did not return structured output",
        ):
            await workflow.execute(
                thread_id="restart-thread",
                story="故事",
                requirements="要求",
                persona_files={"/persona/project.md": "规则"},
                before_stage=before_stage,
                approve_stage=approve_stage,
                **_episode_hook_kwargs()[0],
            )

    first_stage = InternalStage.SELECTING_L0_VARIANT
    assert attempts == [first_stage]
    assert set(approved) == {first_stage}

    resumed_attempts: list[InternalStage] = []

    async def before_resumed_stage(stage: InternalStage) -> int:
        resumed_attempts.append(stage)
        return 1

    async with AsyncSqliteSaver.from_conn_string(str(database)) as saver:
        await saver.setup()
        resumed_workflow = _fake_workflow(
            model=ToolCallingFakeModel(responses=responses[2:]),
            checkpointer=saver,
            provider_profile_key="toolcallingfakemodel",
        )
        result = await resumed_workflow.execute(
            thread_id="restart-thread",
            story="故事",
            requirements="要求",
            persona_files={"/persona/project.md": "规则"},
            before_stage=before_resumed_stage,
            approve_stage=approve_stage,
            approved_checkpoints=approved,
            **_episode_hook_kwargs()[0],
        )

        checkpoint = await saver.aget_tuple({"configurable": {"thread_id": "restart-thread"}})

    assert checkpoint is not None
    assert result.content_package.episode_scripts == "第 1 集\n事实1\n钩子1"
    assert first_stage not in resumed_attempts
    assert resumed_attempts == [
        InternalStage.GENERATING_STORY_OUTLINE,
        InternalStage.GENERATING_CHARACTER_RELATIONSHIPS,
        InternalStage.GENERATING_EPISODE_OUTLINE,
        InternalStage.ACCEPTING_L0,
        InternalStage.ACCEPTING_L4,
    ]


@pytest.mark.asyncio
async def test_restarted_worker_resumes_same_run_and_thread(
    tmp_path: Path,
) -> None:
    persona_root = tmp_path / "personas"
    create_persona_package(persona_root / "active")
    settings = Settings(persona_root=persona_root, data_dir=tmp_path / "data")
    repository = Repository(settings.database_path)
    catalog = PersonaCatalog(persona_root, settings.snapshot_root)
    await repository.initialize()
    snapshot = catalog.create_snapshot("test-persona")
    stopped_at = datetime(2020, 1, 1, tzinfo=UTC)
    accepted = await repository.create_creation(
        "restart-create",
        CreateCreationRequest(
            persona_id="test-persona",
            story="一个人回乡。",
            requirements="生成完整短剧。",
        ),
        snapshot.summary,
        now=stopped_at,
    )
    lease = await repository.lease_next_job(
        "stopped-worker",
        lease_seconds=5,
        now=stopped_at,
    )
    assert lease is not None

    responses = _successful_responses_unified()

    async def before_stage(stage: InternalStage) -> int:
        return await repository.record_stage_attempt(lease.run_id, stage)

    async def approve_stage(stage: InternalStage, payload: dict[str, Any]) -> None:
        await repository.approve_checkpoint(lease.run_id, stage, payload)

    async with AsyncSqliteSaver.from_conn_string(str(settings.database_path)) as saver:
        await saver.setup()
        interrupted_workflow = _fake_workflow(
            model=ToolCallingFakeModel(
                responses=[
                    *responses[:2],
                    AIMessage(content="process stopped before completion"),
                ]
            ),
            checkpointer=saver,
            provider_profile_key="toolcallingfakemodel",
        )
        with pytest.raises(AgentProtocolError):
            await interrupted_workflow.execute(
                thread_id=lease.thread_id,
                story="一个人回乡。",
                requirements="生成完整短剧。",
                persona_files={"/persona/project.md": "规则"},
                before_stage=before_stage,
                approve_stage=approve_stage,
            )
        checkpoint_before_restart = await saver.aget_tuple(
            {"configurable": {"thread_id": lease.thread_id}}
        )

    first_stage = InternalStage.SELECTING_L0_VARIANT
    assert checkpoint_before_restart is not None
    assert await repository.get_stage_attempt_counts(lease.run_id) == {first_stage: 1}
    assert set(await repository.get_business_checkpoints(lease.run_id)) == {first_stage}
    assert await repository.requeue_expired_jobs(now=stopped_at + timedelta(seconds=6)) == 1

    async with AsyncSqliteSaver.from_conn_string(str(settings.database_path)) as saver:
        await saver.setup()
        resumed_workflow = _fake_workflow(
            model=ToolCallingFakeModel(responses=responses[2:]),
            checkpointer=saver,
            provider_profile_key="toolcallingfakemodel",
        )
        restarted_worker = Worker(
            settings=settings,
            repository=repository,
            catalog=catalog,
            workflow=resumed_workflow,
            worker_id="restarted-worker",
        )

        assert await restarted_worker.run_once() is True
        checkpoint_after_restart = await saver.aget_tuple(
            {"configurable": {"thread_id": lease.thread_id}}
        )

    resource = await repository.get_creation(accepted.creation_id)
    work_item = await repository.get_run_work_item(lease.run_id)
    attempts = await repository.get_stage_attempt_counts(lease.run_id)

    assert checkpoint_after_restart is not None
    assert (
        checkpoint_after_restart.config["configurable"]["checkpoint_id"]
        != checkpoint_before_restart.config["configurable"]["checkpoint_id"]
    )
    assert work_item.thread_id == lease.thread_id
    assert resource is not None
    assert resource.initial.state == "succeeded"
    assert attempts[first_stage] == 1
    assert all(count == 1 for count in attempts.values())


@pytest.mark.asyncio
async def test_recovered_run_fails_safely_when_thread_checkpoint_is_missing(
    tmp_path: Path,
) -> None:
    persona_root = tmp_path / "personas"
    create_persona_package(persona_root / "active")
    settings = Settings(persona_root=persona_root, data_dir=tmp_path / "data")
    repository = Repository(settings.database_path)
    catalog = PersonaCatalog(persona_root, settings.snapshot_root)
    await repository.initialize()
    snapshot = catalog.create_snapshot("test-persona")
    stopped_at = datetime(2020, 1, 1, tzinfo=UTC)
    accepted = await repository.create_creation(
        "missing-checkpoint-create",
        CreateCreationRequest(
            persona_id="test-persona",
            story="故事",
            requirements="要求",
        ),
        snapshot.summary,
        now=stopped_at,
    )
    lease = await repository.lease_next_job(
        "stopped-worker",
        lease_seconds=5,
        now=stopped_at,
    )
    assert lease is not None
    first_stage = InternalStage.SELECTING_L0_VARIANT
    await repository.record_stage_attempt(lease.run_id, first_stage)
    await repository.approve_checkpoint(
        lease.run_id,
        first_stage,
        {
            "stage": first_stage.value,
            "selected_l0_variant": "已批准变体",
            "selection_rationale": "已批准理由",
        },
    )
    assert await repository.requeue_expired_jobs(now=stopped_at + timedelta(seconds=6)) == 1

    model = ToolCallingFakeModel(responses=_successful_responses()[2:])
    async with AsyncSqliteSaver.from_conn_string(str(settings.database_path)) as saver:
        await saver.setup()
        worker = Worker(
            settings=settings,
            repository=repository,
            catalog=catalog,
            workflow=_fake_workflow(
                model=model,
                checkpointer=saver,
                provider_profile_key="toolcallingfakemodel",
            ),
            worker_id="restarted-worker",
        )

        assert await worker.run_once() is True

    resource = await repository.get_creation(accepted.creation_id)
    attempts = await repository.get_stage_attempt_counts(lease.run_id)

    assert resource is not None
    assert resource.initial.state == "failed"
    assert resource.initial.failure.code == "checkpoint_unavailable"
    assert attempts[first_stage] == 1
    assert model.bound_tool_names == []


def test_virtual_permissions_deny_persona_writes_and_unmatched_paths() -> None:
    rules = [
        (tuple(rule.operations), tuple(rule.paths), rule.mode) for rule in VIRTUAL_FILE_PERMISSIONS
    ]

    assert (("read",), ("/persona", "/persona/**"), "allow") in rules
    assert (("write",), ("/persona", "/persona/**"), "deny") in rules
    assert (
        ("read", "write"),
        ("/workspace", "/workspace/**"),
        "allow",
    ) in rules
    assert not any("/skills" in paths for _, paths, _ in rules)
    assert (("read", "write"), ("/**",), "deny") in rules

    skilled_rules = [
        (tuple(rule.operations), tuple(rule.paths), rule.mode) for rule in SKILLED_WRITE_PERMISSIONS
    ]
    assert (
        ("read",),
        ("/persona", "/persona/**", "/skills", "/skills/**"),
        "allow",
    ) in skilled_rules
    assert (
        ("write",),
        ("/persona", "/persona/**", "/skills", "/skills/**"),
        "deny",
    ) in skilled_rules
