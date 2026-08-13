import copy
import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
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
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import StructuredTool
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.types import Command
from persona_factory import create_persona_package
from pydantic import BaseModel, Field, ValidationError, field_validator

from pengine.agents import (
    _CANON_REVIEWER_PROMPT,
    _EPISODE_PLANNER_PROMPT,
    _EPISODE_REPAIR_PROMPT,
    _EPISODE_REVIEWER_PROMPT,
    _INTERNAL_RUNTIME_LEAK_POLICY,
    _PROJECT_CREATIVE_POLICY,
    _PROJECT_INLINE_MARKER,
    _PROJECT_REVIEW_BOUNDARY,
    _QUALITY_REVIEWER_PROMPT,
    _REPAIR_TOOL_ALLOWLIST,
    _SCRIPT_WRITER_PROMPT,
    _SERIES_REVIEWER_PROMPT,
    _SPECIALIST_SKILL_SOURCES,
    _STORY_ARCHITECT_PROMPT,
    _STORY_REPAIR_PROMPT,
    REVIEW_FILE_PERMISSIONS,
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
    ToolAllowlistMiddleware,
    WorkflowCompletion,
    _apply_outline_repair_patch,
    _apply_story_artifact_repair_patch,
    _arithmetic_tool,
    _bind_outline_contract_repairs,
    _calculate_arithmetic,
    _canon_issue_ledger,
    _compact_supervisor_messages,
    _drop_dangling_tool_call_messages,
    _evidence_contract,
    _language_retry_fingerprint,
    _language_retry_matches,
    _merge_story_canon_reviews,
    _outline_repair_context,
    _outline_repair_result,
    _request_with_canonical_workspace,
    _required_read_paths,
    _result_with_payload,
    _story_patch_correction,
    _story_repair_context,
    _structured_output_retry_message,
    _structured_result_validation_correction,
    _successful_required_reads,
    _suffix_rewrite_feedback_for_episode,
    _supervisor_prompt,
    _validate_outline_repair_patch_targets,
    _validate_result_language,
    _with_inline_project,
    _with_inline_soul,
    _with_l3_policy,
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
from pengine.model_calls import (
    ModelCallState,
    ModelCallStore,
    build_started_record,
    new_call_id,
)
from pengine.personas import PersonaCatalog
from pengine.repository import Repository
from pengine.schemas import CreateCreationRequest, EpisodeDraft, EpisodePlan, InternalStage
from pengine.series_bible import build_series_bible, project_series_bible
from pengine.skill_assets import load_agent_skill_files
from pengine.worker import Worker


class ToolCallingFakeModel(FakeMessagesListChatModel):
    bound_tool_names: list[list[str]] = Field(default_factory=list)
    bound_tool_descriptions: list[list[str]] = Field(default_factory=list)
    model_system_prompts: list[str] = Field(default_factory=list)

    def bind_tools(
        self,
        tools: Sequence[Any],
        **_: Any,
    ) -> "ToolCallingFakeModel":
        self.bound_tool_names.append([_tool_name(tool) for tool in tools])
        self.bound_tool_descriptions.append([getattr(tool, "description", "") for tool in tools])
        return self

    def _generate(
        self,
        messages: list[Any],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> Any:
        self.model_system_prompts.append(
            "\n\n".join(message.text for message in messages if isinstance(message, SystemMessage))
        )
        return super()._generate(
            messages,
            stop=stop,
            run_manager=run_manager,
            **kwargs,
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class _ProvenanceFakeWorkflow(DeepAgentWorkflow):
    """Make artifact-producing fake calls obey the physical ledger contract."""

    model_call_state: ModelCallState

    def _record_succeeded_call(self, role: str) -> None:
        state = self.model_call_state
        assert state.store is not None
        call_id = new_call_id()
        state.claim_physical_call_id(call_id)
        record = build_started_record(
            call_id=call_id,
            role=role,
            adapter="fake",
            provider="fake",
            model="toolcallingfakemodel",
            context=state.context,
            estimated_input_tokens=10,
            estimated_output_tokens=20,
            verified_limit_tokens=200_000,
        )
        record.status = "succeeded"
        record.outcome = "success"
        state.store.upsert(record)
        state.remember_succeeded(record)

    async def execute(self, **kwargs: Any):
        approve_stage = kwargs["approve_stage"]
        commit_episode = kwargs.get("commit_episode")
        register_series_review = kwargs.get("register_series_review")

        async def approve_with_provenance(stage, payload):
            if stage is InternalStage.GENERATING_EPISODE_OUTLINE and "story_contract" in payload:
                self._record_succeeded_call("review")
            await approve_stage(stage, payload)

        async def commit_with_provenance(*args, **commit_kwargs):
            assert commit_episode is not None
            self._record_succeeded_call("generation")
            return await commit_episode(*args, **commit_kwargs)

        async def review_with_provenance(**review_kwargs):
            assert register_series_review is not None
            self._record_succeeded_call("review")
            return await register_series_review(**review_kwargs)

        return await DeepAgentWorkflow.execute(
            self,
            **{
                **kwargs,
                "approve_stage": approve_with_provenance,
                "commit_episode": commit_with_provenance,
                "register_series_review": review_with_provenance,
            },
        )


def _fake_workflow(
    *,
    model: ToolCallingFakeModel,
    checkpointer: Any,
    recursion_limit: int = 80,
    provider_profile_key: str = "toolcallingfakemodel",
    model_call_state: ModelCallState | None = None,
) -> DeepAgentWorkflow:
    workflow_type = _ProvenanceFakeWorkflow if model_call_state is not None else DeepAgentWorkflow
    kwargs = dict(
        generation_model=model,
        review_model=model,
        checkpointer=checkpointer,
        recursion_limit=recursion_limit,
        generation_provider_profile_key=provider_profile_key,
        review_provider_profile_key=provider_profile_key,
    )
    if model_call_state is not None:
        return workflow_type(**kwargs, model_call_state=model_call_state)
    return workflow_type(**kwargs)


def _tool_name(tool: Any) -> str:
    if hasattr(tool, "name"):
        return tool.name
    if isinstance(tool, dict):
        if "name" in tool:
            return tool["name"]
        return tool.get("function", {}).get("name", "")
    return ""


def _prompt_mentions_tool(prompt: str, tool_name: str) -> bool:
    return (
        re.search(
            rf"(?<![A-Za-z0-9_]){re.escape(tool_name)}(?![A-Za-z0-9_])",
            prompt,
        )
        is not None
    )


def _assert_task_lists_exact_workspace_paths(request: ToolCallRequest) -> None:
    description = request.tool_call["args"]["description"]
    files = request.state.get("files", {})
    workspace_paths = sorted(path for path in files if path.startswith("/workspace/"))
    assert workspace_paths
    assert "Workspace inputs for this delegated task are exactly" in description
    assert "do not probe directories, roots, or alternate paths" in description
    for path in workspace_paths:
        assert path in description


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


def _story_contract(
    episode_count: int = 1,
    *,
    verbatim_episodes: set[int] | None = None,
) -> StoryContract:
    verbatim_episodes = verbatim_episodes or set()
    facts = [
        {
            "fact_id": f"fact_ep{episode}",
            "subject": "测试人物",
            "predicate": "确认事实",
            "kind": "text",
            "value": f"事实{episode}",
            "verbatim": episode in verbatim_episodes,
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


def _successful_responses(*, contract: StoryContract | None = None) -> list[AIMessage]:
    contract = contract or _story_contract()
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
                "evidence": (
                    "母题兑现：人物用行动回答母题。\n"
                    "选定侧面：创作方向贯穿全剧。\n"
                    "雷区：未发现解释性表达。\n"
                    "温度：情绪克制且峰后有收拍。"
                ),
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
                "evidence": (
                    "L4-A：人物与情感成立。\n"
                    "短剧硬规则：适用硬规则均有证据。\n"
                    "产品参数：采用 Pengine 默认参数。"
                ),
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
                    {
                        "passed": True,
                        "evidence": "L4硬规则：故事大纲适用硬规则一致。",
                        "issues": [],
                    },
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
                        {
                            "passed": True,
                            "evidence": "L4硬规则：人物关系适用硬规则一致。",
                            "issues": [],
                        },
                        index,
                    )
                )
                index += 1
        if stage == "generating_episode_outline":
            responses.append(
                _tool_call(
                    "CanonReviewerResult",
                    {
                        "passed": True,
                        "evidence": "L4硬规则：合同与分集大纲适用硬规则一致。",
                        "issues": [],
                    },
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
                    {
                        "passed": True,
                        "category": "pass",
                        "evidence": "L4硬规则：全系列适用硬规则一致。",
                    },
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
    assert "For [ID:D], return D, not [ID:D]" in properties["selected_l0_variant"]["description"]
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
                "contract_mutation_required": False,
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
                        "contract_mutation_required": False,
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
                "contract_mutation_required": False,
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
                "contract_mutation_required": False,
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
        issues=[
            {
                "code": "stale_text",
                "message": "替换旧文字。",
                "contract_mutation_required": False,
            }
        ],
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


def test_result_with_payload_preserves_command_state_and_tool_metadata() -> None:
    tool_message = ToolMessage(
        content='{"value":"stale"}',
        tool_call_id="task-call",
        name="task",
        artifact={"audit": "preserved"},
        additional_kwargs={"source": "subagent"},
        response_metadata={"trace": "preserved"},
    )
    original = Command(
        graph=Command.PARENT,
        update={
            "messages": [tool_message],
            "files": {"/workspace/scratch.md": {"content": "保留"}},
            "counter": 2,
        },
        resume={"approved": True},
        goto="supervisor",
    )

    result = _result_with_payload(original, {"value": "approved"})

    assert isinstance(result, Command)
    assert result.graph == original.graph
    assert result.resume == original.resume
    assert result.goto == original.goto
    assert result.update["files"] == original.update["files"]
    assert result.update["counter"] == 2
    message = result.update["messages"][0]
    assert json.loads(message.content) == {"value": "approved"}
    assert message.tool_call_id == tool_message.tool_call_id
    assert message.name == tool_message.name
    assert message.artifact == tool_message.artifact
    assert message.additional_kwargs == tool_message.additional_kwargs
    assert message.response_metadata == tool_message.response_metadata


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
            "consistency_review": {
                "passed": True,
                "evidence": "已淘汰的旧事实只用于审计",
                "issues": [],
            },
            "consistency_repair_rounds": 1,
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
    manifest = json.loads(files["/workspace/approved-checkpoints.json"]["content"])
    assert manifest["generating_story_outline"]["content"] == "已批准大纲"
    assert "consistency_review" not in manifest["generating_story_outline"]
    assert "consistency_repair_rounds" not in manifest["generating_story_outline"]
    assert "已淘汰的旧事实" not in files["/workspace/approved-checkpoints.json"]["content"]


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
                "contract_refs": ["fact_ep1"],
                "contract_mutation_required": True,
                "repair_targets": [
                    {
                        "target_id": f"replace_knowledge_{index}",
                        "collection": "knowledge_states",
                        "intent": "replace_existing",
                        "index": index,
                        "expected_value": state,
                        "value": {
                            **state,
                            "known_fact_ids": [
                                fact_id
                                for fact_id in state["known_fact_ids"]
                                if fact_id != "fact_ep1"
                            ],
                        },
                    }
                    for index, state in enumerate(contract["knowledge_states"])
                ],
            }
        ],
    )

    context = _outline_repair_context(candidate, review)
    serialized_context = json.dumps(context, ensure_ascii=False, separators=(",", ":"))
    serialized_candidate = json.dumps(candidate, ensure_ascii=False, separators=(",", ":"))
    mutation_by_path = {
        item["path"]: item for item in context["contract_mutations_applied_by_runtime"]
    }
    context_by_path = {item["path"]: item for item in context["contract_context"]}

    assert "frozen_upstream" not in context
    assert "candidate" not in context
    assert "DO_NOT_SEND" not in serialized_context
    assert len(serialized_context) * 2 < len(serialized_candidate)
    assert context["matched_contract_refs"] == ["fact_ep1"]
    assert context["matched_collection_scopes"] == ["knowledge_states"]
    assert context["unmatched_contract_refs"] == []
    assert context["readable_outline"]["value"] == candidate["content"]
    assert [item["value"] for item in context["episode_plans"]] == candidate["episodes"]
    assert set(mutation_by_path) == {
        "/story_contract/knowledge_states/0",
        "/story_contract/knowledge_states/1",
    }
    assert set(context_by_path) == {"/story_contract/facts/0"}
    assert mutation_by_path["/story_contract/knowledge_states/0"]["op"] == "replace"


def test_outline_repair_context_uses_explicit_contract_targets() -> None:
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
        evidence="事实一的谓词和时间线标签与上游冲突。",
        issues=[
            {
                "code": "fact_timeline_conflict",
                "message": "精确修正 fact_ep1 及其时间线标签。",
                "contract_refs": ["fact_ep1"],
                "contract_mutation_required": True,
                "repair_targets": [
                    {
                        "target_id": "replace_fact_ep1",
                        "collection": "facts",
                        "intent": "replace_existing",
                        "index": 0,
                        "expected_value": contract["facts"][0],
                        "value": {
                            **contract["facts"][0],
                            "predicate": "确认修复事实",
                        },
                    },
                    {
                        "target_id": "replace_timeline_ep1",
                        "collection": "timeline",
                        "intent": "replace_existing",
                        "index": 0,
                        "expected_value": contract["timeline"][0],
                        "value": {
                            **contract["timeline"][0],
                            "when": "第一幕",
                        },
                    },
                ],
            }
        ],
    )

    context = _outline_repair_context(candidate, review)
    mutation_by_path = {
        item["path"]: item for item in context["contract_mutations_applied_by_runtime"]
    }

    assert set(mutation_by_path) == {
        "/story_contract/facts/0",
        "/story_contract/timeline/0",
    }
    assert all(item["op"] == "replace" for item in mutation_by_path.values())


def test_outline_repair_context_exposes_explicit_collection_scope() -> None:
    contract = _story_contract().model_dump(mode="json")
    contract["prohibitions"] = ["保留正确约束", "删除冲突约束"]
    candidate = {
        "stage": "generating_episode_outline",
        "content": "第一集发现事实一，并遵守所有明确约束。",
        "episode_count": 1,
        "episodes": [{"episode_number": 1, "plan": "第一集计划"}],
        "story_contract": contract,
    }
    review = CanonReviewerResult(
        passed=False,
        evidence="合同的禁止项与正式事实冲突。",
        issues=[
            {
                "code": "conflicting_prohibition",
                "message": "删除与 fact_ep1 冲突的禁止项。",
                "contract_refs": ["fact_ep1"],
                "script_excerpt": "删除冲突约束",
                "contract_mutation_required": True,
                "repair_targets": [
                    {
                        "target_id": "remove_conflicting_prohibition",
                        "collection": "prohibitions",
                        "intent": "remove_existing",
                        "index": 1,
                        "expected_value": "删除冲突约束",
                    }
                ],
            }
        ],
    )

    context = _outline_repair_context(candidate, review)
    mutation_by_path = {
        item["path"]: item for item in context["contract_mutations_applied_by_runtime"]
    }
    context_by_path = {item["path"]: item for item in context["contract_context"]}
    with pytest.raises(ValidationError, match="episode plans"):
        OutlineRepairPatch.model_validate(
            {
                "stage": "generating_episode_outline",
                "json_edits": [
                    {
                        "op": "replace",
                        "path": "/story_contract/prohibitions/1",
                        "expected": "删除冲突约束",
                        "value": "替换冲突约束",
                    }
                ],
            }
        )

    assert set(context_by_path) == {"/story_contract/facts/0"}
    assert mutation_by_path["/story_contract/prohibitions/1"]["op"] == "remove"
    repaired_contract, _ = _bind_outline_contract_repairs(contract, review)
    assert repaired_contract.prohibitions == ["保留正确约束"]

    mismatched_review_payload = review.model_dump(mode="json")
    mismatched_review_payload["issues"][0]["repair_targets"][0]["expected_value"] = "错误旧值"
    mismatched_review = CanonReviewerResult.model_validate(mismatched_review_payload)
    with pytest.raises(ValueError, match="outline_repair_review_target_mismatch"):
        _outline_repair_context(candidate, mismatched_review)


def test_outline_repair_context_keeps_explicit_id_scope_narrow() -> None:
    contract = _story_contract(episode_count=2).model_dump(mode="json")
    candidate = {
        "stage": "generating_episode_outline",
        "content": "第一集和第二集各自建立一个事实。",
        "episode_count": 2,
        "episodes": [
            {"episode_number": 1, "plan": "第一集计划"},
            {"episode_number": 2, "plan": "第二集计划"},
        ],
        "story_contract": contract,
    }
    review = CanonReviewerResult(
        passed=False,
        evidence="只需修复事实一。",
        issues=[
            {
                "code": "fact_one_conflict",
                "message": "fact_ep1 的正式值冲突。",
                "contract_refs": ["fact_ep1"],
                "contract_mutation_required": True,
                "repair_targets": [
                    {
                        "target_id": "replace_fact_ep1",
                        "collection": "facts",
                        "intent": "replace_existing",
                        "index": 0,
                        "expected_value": contract["facts"][0],
                        "value": {
                            **contract["facts"][0],
                            "value": "修复事实1",
                        },
                    }
                ],
            }
        ],
    )

    context = _outline_repair_context(candidate, review)
    mutation_by_path = {
        item["path"]: item for item in context["contract_mutations_applied_by_runtime"]
    }

    assert mutation_by_path["/story_contract/facts/0"]["op"] == "replace"
    assert "/story_contract/facts/1" not in mutation_by_path
    with pytest.raises(ValidationError, match="episode plans"):
        OutlineRepairPatch.model_validate(
            {
                "stage": "generating_episode_outline",
                "json_edits": [
                    {
                        "op": "replace",
                        "path": "/story_contract/facts/-",
                        "expected": None,
                        "value": contract["facts"][1],
                    }
                ],
            }
        )


def test_outline_repair_context_allows_explicit_scalar_append_once() -> None:
    contract = _story_contract().model_dump(mode="json")
    candidate = {
        "stage": "generating_episode_outline",
        "content": "第一集尚未建立缺失事实。",
        "episode_count": 1,
        "episodes": [{"episode_number": 1, "plan": "第一集计划"}],
        "story_contract": contract,
    }
    review = CanonReviewerResult(
        passed=False,
        evidence="必须新增禁止项。",
        issues=[
            {
                "code": "missing_prohibition",
                "message": "补充不得省略关键事实的禁止项。",
                "contract_refs": [],
                "contract_mutation_required": True,
                "repair_targets": [
                    {
                        "target_id": "append_required_prohibition",
                        "collection": "prohibitions",
                        "intent": "append_missing",
                        "value": "  不得省略关键事实。  ",
                    }
                ],
            }
        ],
    )
    context = _outline_repair_context(candidate, review)
    with pytest.raises(ValidationError, match="episode plans"):
        OutlineRepairPatch.model_validate(
            {
                "stage": "generating_episode_outline",
                "json_edits": [
                    {
                        "op": "replace",
                        "path": "/story_contract/prohibitions/-",
                        "expected": len(contract["prohibitions"]),
                        "value": "不得省略关键事实。",
                    }
                ],
            }
        )

    assert context["contract_mutations_applied_by_runtime"] == [
        {
            "target_id": "append_required_prohibition",
            "op": "add",
            "path": "/story_contract/prohibitions/-",
            "expected_value": None,
            "value": "不得省略关键事实。",
        }
    ]
    repaired_contract, _ = _bind_outline_contract_repairs(contract, review)
    assert repaired_contract.prohibitions[-1] == "不得省略关键事实。"

    empty_patch = OutlineRepairPatch.model_validate(
        {
            "stage": "generating_episode_outline",
            "content_replacements": [],
            "json_edits": [],
        }
    )
    _validate_outline_repair_patch_targets(empty_patch, context)


def test_outline_repair_context_rejects_invalid_or_existing_append_targets() -> None:
    contract = _story_contract().model_dump(mode="json")
    candidate = {
        "stage": "generating_episode_outline",
        "content": "分集大纲",
        "episode_count": 1,
        "episodes": [{"episode_number": 1, "plan": "第一集计划"}],
        "story_contract": contract,
    }

    invalid_shape = CanonReviewerResult(
        passed=False,
        evidence="禁止项目标类型错误。",
        issues=[
            {
                "code": "missing_prohibition",
                "message": "补充禁止项。",
                "contract_refs": [],
                "contract_mutation_required": True,
                "repair_targets": [
                    {
                        "target_id": "append_invalid_prohibition",
                        "collection": "prohibitions",
                        "intent": "append_missing",
                        "value": {"text": "不得增加人物"},
                    }
                ],
            }
        ],
    )
    with pytest.raises(ValueError, match="outline_repair_review_target_value_invalid"):
        _outline_repair_context(candidate, invalid_shape)

    existing_value = CanonReviewerResult(
        passed=False,
        evidence="禁止项已存在。",
        issues=[
            {
                "code": "duplicate_prohibition",
                "message": "重复补充禁止项。",
                "contract_refs": [],
                "contract_mutation_required": True,
                "repair_targets": [
                    {
                        "target_id": "append_existing_prohibition",
                        "collection": "prohibitions",
                        "intent": "append_missing",
                        "value": "不得增加人物",
                    }
                ],
            }
        ],
    )
    with pytest.raises(ValueError, match="outline_repair_review_append_already_exists"):
        _outline_repair_context(candidate, existing_value)

    existing_identity = CanonReviewerResult(
        passed=False,
        evidence="角色 ID 已存在。",
        issues=[
            {
                "code": "duplicate_character",
                "message": "不得以追加方式覆盖现有角色。",
                "contract_refs": [],
                "contract_mutation_required": True,
                "repair_targets": [
                    {
                        "target_id": "append_existing_character",
                        "collection": "characters",
                        "intent": "append_missing",
                        "value": {
                            **contract["characters"][0],
                            "role": "错误的新身份",
                        },
                    }
                ],
            }
        ],
    )
    with pytest.raises(ValueError, match="outline_repair_review_proposed_contract_invalid"):
        _outline_repair_context(candidate, existing_identity)

    conflicting_pending_targets = CanonReviewerResult(
        passed=False,
        evidence="两个追加目标复用了同一角色 ID。",
        issues=[
            {
                "code": "conflicting_character_appends",
                "message": "追加目标彼此冲突。",
                "contract_refs": [],
                "contract_mutation_required": True,
                "repair_targets": [
                    {
                        "target_id": "append_new_character_as_witness",
                        "collection": "characters",
                        "intent": "append_missing",
                        "value": {
                            "character_id": "new_character",
                            "name": "新角色",
                            "role": "证人",
                            "initial_known_fact_ids": [],
                        },
                    },
                    {
                        "target_id": "append_new_character_as_editor",
                        "collection": "characters",
                        "intent": "append_missing",
                        "value": {
                            "character_id": "new_character",
                            "name": "新角色",
                            "role": "剪辑师",
                            "initial_known_fact_ids": [],
                        },
                    },
                ],
            }
        ],
    )
    with pytest.raises(ValueError, match="outline_repair_review_proposed_contract_invalid"):
        _outline_repair_context(candidate, conflicting_pending_targets)


def test_outline_repair_append_targets_bind_all_refs_and_sequential_lengths() -> None:
    contract = _story_contract().model_dump(mode="json")
    contract["characters"].extend(
        [
            {
                "character_id": "bob",
                "name": "鲍勃",
                "role": "同事",
                "initial_known_fact_ids": [],
            },
            {
                "character_id": "carol",
                "name": "卡萝尔",
                "role": "主管",
                "initial_known_fact_ids": [],
            },
        ]
    )
    contract["relationships"] = [
        {
            "source_character_id": "test_character",
            "target_character_id": "bob",
            "relation": "同事",
        }
    ]
    candidate = {
        "stage": "generating_episode_outline",
        "content": "三个人的关系需要补齐。",
        "episode_count": 1,
        "episodes": [{"episode_number": 1, "plan": "第一集计划"}],
        "story_contract": contract,
    }
    review = CanonReviewerResult(
        passed=False,
        evidence="缺少两条明确要求的关系。",
        issues=[
            {
                "code": "missing_relationships",
                "message": "新增 bob→carol 与 carol→test_character。",
                "contract_refs": [
                    "test_character",
                    "bob",
                    "carol",
                ],
                "contract_mutation_required": True,
                "repair_targets": [
                    {
                        "target_id": "append_bob_carol",
                        "collection": "relationships",
                        "intent": "append_missing",
                        "value": {
                            "source_character_id": "bob",
                            "target_character_id": "carol",
                            "relation": "下属与主管",
                        },
                    },
                    {
                        "target_id": "append_carol_editor",
                        "collection": "relationships",
                        "intent": "append_missing",
                        "value": {
                            "source_character_id": "carol",
                            "target_character_id": "test_character",
                            "relation": "主管与剪辑师",
                        },
                    },
                ],
            }
        ],
    )
    context = _outline_repair_context(candidate, review)
    assert [item["op"] for item in context["contract_mutations_applied_by_runtime"]] == [
        "add",
        "add",
    ]
    repaired_contract, _ = _bind_outline_contract_repairs(contract, review)
    assert len(repaired_contract.relationships) == 3
    assert repaired_contract.relationships[-2].source_character_id == "bob"
    assert repaired_contract.relationships[-1].source_character_id == "carol"

    normalized_duplicate = CanonReviewerResult(
        passed=False,
        evidence="重复追加既有关系。",
        issues=[
            {
                "code": "duplicate_relationship",
                "message": "带空格的值仍是同一关系。",
                "contract_refs": ["test_character", "bob"],
                "contract_mutation_required": True,
                "repair_targets": [
                    {
                        "target_id": "append_normalized_duplicate",
                        "collection": "relationships",
                        "intent": "append_missing",
                        "value": {
                            "source_character_id": "test_character",
                            "target_character_id": "bob",
                            "relation": " 同事 ",
                        },
                    }
                ],
            }
        ],
    )
    with pytest.raises(ValueError, match="outline_repair_review_append_already_exists"):
        _outline_repair_context(candidate, normalized_duplicate)

    replace_then_duplicate = CanonReviewerResult(
        passed=False,
        evidence="替换和追加会生成重复关系。",
        issues=[
            {
                "code": "conflicting_relationship_targets",
                "message": "原子操作的最终集合不得重复。",
                "contract_refs": ["test_character", "bob"],
                "contract_mutation_required": True,
                "repair_targets": [
                    {
                        "target_id": "replace_existing_relationship",
                        "collection": "relationships",
                        "intent": "replace_existing",
                        "index": 0,
                        "expected_value": contract["relationships"][0],
                        "value": {
                            **contract["relationships"][0],
                            "relation": "盟友",
                        },
                    },
                    {
                        "target_id": "append_replacement_duplicate",
                        "collection": "relationships",
                        "intent": "append_missing",
                        "value": {
                            **contract["relationships"][0],
                            "relation": " 盟友 ",
                        },
                    },
                ],
            }
        ],
    )
    with pytest.raises(ValueError, match="outline_repair_review_append_already_exists"):
        _outline_repair_context(candidate, replace_then_duplicate)


def test_outline_repair_append_target_requires_exact_value() -> None:
    contract = _story_contract().model_dump(mode="json")
    candidate = {
        "stage": "generating_episode_outline",
        "content": "需要补充一个已锁定角色。",
        "episode_count": 1,
        "episodes": [{"episode_number": 1, "plan": "第一集计划"}],
        "story_contract": contract,
    }
    review = CanonReviewerResult(
        passed=False,
        evidence="角色合同缺少已锁定角色。",
        issues=[
            {
                "code": "missing_character",
                "message": "补充 new_character。",
                "contract_refs": ["new_character"],
                "contract_mutation_required": True,
                "repair_targets": [
                    {
                        "target_id": "append_new_character",
                        "collection": "characters",
                        "intent": "append_missing",
                        "value": {
                            "character_id": "new_character",
                            "name": "新角色",
                            "role": "证人",
                            "initial_known_fact_ids": [],
                        },
                    },
                    {
                        "target_id": "append_new_character_relationship",
                        "collection": "relationships",
                        "intent": "append_missing",
                        "value": {
                            "source_character_id": "test_character",
                            "target_character_id": "new_character",
                            "relation": "保护证人",
                        },
                    },
                ],
            }
        ],
    )
    _outline_repair_context(candidate, review)
    with pytest.raises(ValidationError, match="episode plans"):
        OutlineRepairPatch.model_validate(
            {
                "stage": "generating_episode_outline",
                "json_edits": [
                    {
                        "op": "replace",
                        "path": "/story_contract/characters/-",
                        "expected": len(contract["characters"]),
                        "value": {
                            "character_id": "new_character",
                            "name": "新角色",
                            "role": "证人",
                            "initial_known_fact_ids": [],
                        },
                    }
                ],
            }
        )
    repaired_contract, _ = _bind_outline_contract_repairs(contract, review)
    assert repaired_contract.characters[-1].character_id == "new_character"
    assert repaired_contract.relationships[-1].target_character_id == "new_character"

    duplicate_name_payload = review.model_dump(mode="json")
    duplicate_name_payload["issues"][0]["repair_targets"][0]["value"]["name"] = "测试人物"
    duplicate_name = CanonReviewerResult.model_validate(duplicate_name_payload)
    with pytest.raises(ValueError, match="outline_repair_review_proposed_contract_invalid"):
        _outline_repair_context(candidate, duplicate_name)

    missing_endpoint_payload = review.model_dump(mode="json")
    missing_endpoint_payload["issues"][0]["repair_targets"][1]["value"]["target_character_id"] = (
        "missing_character"
    )
    missing_endpoint = CanonReviewerResult.model_validate(missing_endpoint_payload)
    with pytest.raises(ValueError, match="outline_repair_review_proposed_contract_invalid"):
        _outline_repair_context(candidate, missing_endpoint)


def test_canon_review_requires_explicit_contract_mutation_authority() -> None:
    with pytest.raises(ValidationError, match="contract_mutation_required"):
        CanonReviewerResult(
            passed=False,
            evidence="禁止项冲突。",
            issues=[
                {
                    "code": "conflicting_prohibition",
                    "message": "必须定位具体禁止项。",
                    "contract_refs": ["prohibitions"],
                }
            ],
        )

    with pytest.raises(ValidationError, match="Contract mutations require"):
        CanonReviewerResult(
            passed=False,
            evidence="合同需要修改。",
            issues=[
                {
                    "code": "missing_repair_target",
                    "message": "必须声明精确目标。",
                    "contract_refs": ["fact_ep1"],
                    "contract_mutation_required": True,
                }
            ],
        )

    with pytest.raises(ValidationError, match="Prose-only canon issues cannot grant"):
        CanonReviewerResult(
            passed=False,
            evidence="纯文本修复不得授予合同权限。",
            issues=[
                {
                    "code": "prose_only_issue",
                    "message": "只修改可读大纲。",
                    "contract_refs": ["fact_ep1"],
                    "contract_mutation_required": False,
                    "repair_targets": [
                        {
                            "target_id": "replace_fact_without_authority",
                            "collection": "prohibitions",
                            "intent": "remove_existing",
                            "index": 0,
                            "expected_value": "不得增加人物",
                        }
                    ],
                }
            ],
        )

    with pytest.raises(ValidationError, match="target IDs must be unique"):
        CanonReviewerResult(
            passed=False,
            evidence="同一缺失关系不得重复授权。",
            issues=[
                {
                    "code": "missing_relationship",
                    "message": "补充一条关系。",
                    "contract_refs": ["test_character", "bob"],
                    "contract_mutation_required": True,
                    "repair_targets": [
                        {
                            "target_id": "duplicate_relationship",
                            "collection": "relationships",
                            "intent": "append_missing",
                            "value": {
                                "source_character_id": "test_character",
                                "target_character_id": "bob",
                                "relation": "同事",
                            },
                        },
                        {
                            "target_id": "duplicate_relationship",
                            "collection": "relationships",
                            "intent": "append_missing",
                            "value": {
                                "source_character_id": "bob",
                                "target_character_id": "test_character",
                                "relation": "同事",
                            },
                        },
                    ],
                }
            ],
        )


def test_contract_ref_matching_collection_name_remains_an_entity_id() -> None:
    contract = _story_contract().model_dump(mode="json")
    contract["facts"][0]["fact_id"] = "facts"
    contract["timeline"][0]["fact_ids"] = ["facts"]
    contract["knowledge_states"][0]["known_fact_ids"] = ["facts"]
    contract["episode_obligations"][0]["new_information_fact_ids"] = ["facts"]
    candidate = {
        "stage": "generating_episode_outline",
        "content": "分集大纲中的事实措辞需要同步。",
        "episode_count": 1,
        "episodes": [{"episode_number": 1, "plan": "第一集计划"}],
        "story_contract": contract,
    }
    review = CanonReviewerResult(
        passed=False,
        evidence="只需同步可读大纲。",
        issues=[
            {
                "code": "prose_fact_wording",
                "message": "同步 facts 的可读措辞。",
                "contract_refs": ["facts"],
                "contract_mutation_required": False,
            }
        ],
    )

    context = _outline_repair_context(candidate, review)

    assert context["matched_contract_refs"] == ["facts"]
    assert context["matched_collection_scopes"] == []
    assert context["contract_mutations_applied_by_runtime"] == []
    assert context["contract_context"] == [
        {"path": "/story_contract/facts/0", "value": contract["facts"][0]}
    ]


def test_contract_ref_exposes_every_node_with_the_same_cross_domain_id() -> None:
    contract = _story_contract().model_dump(mode="json")
    contract["timeline"][0]["event_id"] = "test_character"
    candidate = {
        "stage": "generating_episode_outline",
        "content": "分集大纲中的同名 Canon 节点都需要作为只读上下文。",
        "episode_count": 1,
        "episodes": [{"episode_number": 1, "plan": "第一集计划"}],
        "story_contract": contract,
    }
    review = CanonReviewerResult(
        passed=False,
        evidence="同一稳定 ID 在不同 Canon 域均有相关上下文。",
        issues=[
            {
                "code": "cross_domain_context",
                "message": "同步引用 test_character 的可读措辞。",
                "contract_refs": ["test_character"],
                "contract_mutation_required": False,
            }
        ],
    )

    context = _outline_repair_context(candidate, review)

    assert context["matched_contract_refs"] == ["test_character"]
    assert context["contract_context"] == [
        {"path": "/story_contract/characters/0", "value": contract["characters"][0]},
        {"path": "/story_contract/timeline/0", "value": contract["timeline"][0]},
    ]


def test_outline_repair_patch_targets_only_exposed_editable_nodes() -> None:
    contract = _story_contract().model_dump(mode="json")
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
                "contract_refs": ["fact_ep1"],
                "contract_mutation_required": True,
                "repair_targets": [
                    {
                        "target_id": "replace_knowledge_state",
                        "collection": "knowledge_states",
                        "intent": "replace_existing",
                        "index": 0,
                        "expected_value": contract["knowledge_states"][0],
                        "value": {
                            **contract["knowledge_states"][0],
                            "known_fact_ids": [],
                        },
                    }
                ],
            }
        ],
    )
    context = _outline_repair_context(candidate, review)
    allowed = OutlineRepairPatch.model_validate(
        {
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
    )
    with pytest.raises(ValidationError, match="episode plans"):
        OutlineRepairPatch.model_validate(
            {
                "stage": "generating_episode_outline",
                "json_edits": [
                    {
                        "op": "replace",
                        "path": "/story_contract/facts/0",
                        "expected": contract["facts"][0],
                        "value": {**contract["facts"][0], "value": "不允许的改写"},
                    }
                ],
            }
        )

    _validate_outline_repair_patch_targets(allowed, context)
    repaired_contract, _ = _bind_outline_contract_repairs(contract, review)
    assert repaired_contract.knowledge_states[0].known_fact_ids == []


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
async def test_tool_allowlist_filters_model_request_and_keeps_result_tool() -> None:
    class Result(BaseModel):
        value: str

    model = ToolCallingFakeModel(
        responses=[
            _tool_call("read_file", {"file_path": "/workspace/candidate.md"}, 1),
            _tool_call("Result", {"value": "done"}, 2),
        ]
    )
    read_file = StructuredTool.from_function(
        lambda file_path: f"contents for {file_path}",
        name="read_file",
        description="Read one explicitly named workspace file.",
    )
    list_files = StructuredTool.from_function(
        lambda: "candidate.md",
        name="ls",
        description="List workspace files.",
    )
    agent = create_agent(
        model,
        tools=[read_file, list_files],
        middleware=[
            ToolAllowlistMiddleware(
                frozenset({"read_file"}),
                system_prompt="Read the explicitly named candidate and return the result.",
            )
        ],
        response_format=ToolStrategy(Result),
    )

    result = await agent.ainvoke({"messages": [HumanMessage(content="Read the candidate.")]})

    assert result["structured_response"] == Result(value="done")
    assert model.bound_tool_names == [["read_file", "Result"], ["read_file", "Result"]]
    assert len(model.model_system_prompts) == 2
    assert all(_prompt_mentions_tool(prompt, "read_file") for prompt in model.model_system_prompts)
    assert all(_prompt_mentions_tool(prompt, "Result") for prompt in model.model_system_prompts)
    assert all(not _prompt_mentions_tool(prompt, "ls") for prompt in model.model_system_prompts)


@pytest.mark.asyncio
async def test_repair_tool_allowlist_is_read_only_and_keeps_result_tool() -> None:
    class Result(BaseModel):
        value: str

    model = ToolCallingFakeModel(
        responses=[
            _tool_call(
                "read_file",
                {"file_path": "/workspace/candidate.md"},
                1,
            ),
            _tool_call("Result", {"value": "repaired"}, 2),
        ]
    )
    read_file = StructuredTool.from_function(
        lambda file_path: f"contents for {file_path}",
        name="read_file",
        description="Read one explicitly named workspace file.",
    )
    edit_file = StructuredTool.from_function(
        lambda file_path, old_string, new_string: f"edited {file_path}",
        name="edit_file",
        description="Edit one explicitly named workspace file.",
    )
    list_files = StructuredTool.from_function(
        lambda: "candidate.md",
        name="ls",
        description="List workspace files.",
    )
    agent = create_agent(
        model,
        tools=[read_file, _arithmetic_tool(), edit_file, list_files],
        middleware=[
            ToolAllowlistMiddleware(
                _REPAIR_TOOL_ALLOWLIST,
                system_prompt="Read the candidate and return the complete repaired result.",
            )
        ],
        response_format=ToolStrategy(Result),
    )

    result = await agent.ainvoke({"messages": [HumanMessage(content="Repair the candidate.")]})

    assert result["structured_response"] == Result(value="repaired")
    assert model.bound_tool_names == [
        ["read_file", "calculate_arithmetic", "Result"],
        ["read_file", "calculate_arithmetic", "Result"],
    ]
    assert all(
        not _prompt_mentions_tool(prompt, "edit_file") for prompt in model.model_system_prompts
    )
    assert all(not _prompt_mentions_tool(prompt, "ls") for prompt in model.model_system_prompts)


@pytest.mark.asyncio
async def test_tool_allowlist_prompt_tracks_result_only_correction_request() -> None:
    class Result(BaseModel):
        value: str

    model = ToolCallingFakeModel(
        responses=[
            AIMessage(content="The work is complete in prose."),
            _tool_call("Result", {"value": "done"}, 1),
        ]
    )
    read_file = StructuredTool.from_function(
        lambda file_path: f"contents for {file_path}",
        name="read_file",
        description="Read one explicitly named workspace file.",
    )
    list_files = StructuredTool.from_function(
        lambda: "candidate.md",
        name="ls",
        description="List workspace files.",
    )
    agent = create_agent(
        model,
        tools=[read_file, list_files],
        middleware=[
            StructuredResultMiddleware(),
            ToolAllowlistMiddleware(
                frozenset({"read_file"}),
                system_prompt="Read the candidate with read_file, then return the result.",
            ),
        ],
        response_format=ToolStrategy(Result),
    )

    result = await agent.ainvoke({"messages": [HumanMessage(content="Read the candidate.")]})

    assert result["structured_response"] == Result(value="done")
    assert model.bound_tool_names[-2:] == [["read_file", "Result"], ["Result"]]
    assert _prompt_mentions_tool(model.model_system_prompts[-2], "read_file")
    assert not _prompt_mentions_tool(model.model_system_prompts[-1], "read_file")
    assert _prompt_mentions_tool(model.model_system_prompts[-1], "Result")
    assert not _prompt_mentions_tool(model.model_system_prompts[-1], "ls")
    assert "Do not generate, revise, repair, expand, summarize" in (model.model_system_prompts[-1])
    assert "without changing its content" in model.model_system_prompts[-1]


@pytest.mark.asyncio
async def test_structured_result_requires_every_declared_file_read_before_result() -> None:
    class Result(BaseModel):
        value: str

    read_paths: list[str] = []
    model = ToolCallingFakeModel(
        responses=[
            _tool_call("Result", {"value": "done"}, 1),
        ],
    )
    read_file = StructuredTool.from_function(
        lambda file_path: read_paths.append(file_path) or f"contents for {file_path}",
        name="read_file",
        description="Read one explicitly named workspace file.",
    )
    description = (
        "Review both mounted inputs before returning a result.\n"
        "<pengine-required-read-paths>\n"
        "/workspace/a.md\n"
        "/workspace/b.md\n"
        "</pengine-required-read-paths>"
    )
    agent = create_agent(
        model,
        tools=[read_file],
        middleware=[
            StructuredResultMiddleware(),
            ToolAllowlistMiddleware(
                frozenset({"read_file"}),
                system_prompt="Read every required file, then return the result.",
            ),
        ],
        response_format=ToolStrategy(Result),
    )

    result = await agent.ainvoke({"messages": [HumanMessage(content=description)]})

    assert result["structured_response"] == Result(value="done")
    assert read_paths == ["/workspace/a.md", "/workspace/b.md"]
    assert len(model.bound_tool_names) == 1
    assert set(model.bound_tool_names[0]) == {"read_file", "Result"}


@pytest.mark.asyncio
async def test_engine_schedules_every_required_canon_read_before_first_model_call() -> None:
    paths = (
        "/workspace/approved-checkpoints.json",
        "/workspace/creation-request.md",
        "/workspace/current_character_biographies.md",
        "/workspace/current_relationship_logic.md",
        "/workspace/story_outline.md",
    )
    review = CanonReviewerResult(
        passed=True,
        evidence="已读取全部必需输入，未发现硬连续性矛盾。",
        issues=[],
    )
    request = ModelRequest(
        model=FakeMessagesListChatModel(responses=[]),
        messages=[
            HumanMessage(
                content=(
                    "Review every mounted input before returning a result.\n"
                    "<pengine-required-read-paths>\n"
                    + "\n".join((*paths, paths[2]))
                    + "\n</pengine-required-read-paths>"
                )
            )
        ],
        tools=[{"type": "function", "function": {"name": "read_file"}}],
        response_format=ToolStrategy(CanonReviewerResult),
    )
    handler_calls: list[ModelRequest] = []

    async def handler(candidate: ModelRequest) -> ModelResponse:
        handler_calls.append(candidate)
        return ModelResponse(
            result=[AIMessage(content="", tool_calls=[])],
            structured_response=review,
        )

    middleware = StructuredResultMiddleware()
    observed_paths: list[str] = []
    for expected_path in paths:
        response = await middleware.awrap_model_call(request, handler)

        assert handler_calls == []
        assert response.structured_response is None
        assert len(response.result) == 1
        message = response.result[0]
        assert isinstance(message, AIMessage)
        assert len(message.tool_calls) == 1
        call = message.tool_calls[0]
        assert call["name"] == "read_file"
        assert call["args"] == {"file_path": expected_path}
        observed_paths.append(expected_path)
        request = request.override(
            messages=[
                *request.messages,
                message,
                ToolMessage(
                    content=f"contents for {expected_path}",
                    name="read_file",
                    tool_call_id=call["id"],
                ),
            ]
        )

    response = await middleware.awrap_model_call(request, handler)

    assert response.structured_response == review
    assert len(handler_calls) == 1
    assert observed_paths == list(paths)
    assert _successful_required_reads(request.messages) == frozenset(paths)


@pytest.mark.asyncio
async def test_structured_result_rejects_result_after_failed_required_read() -> None:
    class Result(BaseModel):
        value: str

    request = ModelRequest(
        model=FakeMessagesListChatModel(responses=[]),
        messages=[
            HumanMessage(
                content=(
                    "Review the input.\n"
                    "<pengine-required-read-paths>\n"
                    "/workspace/missing.md\n"
                    "</pengine-required-read-paths>"
                )
            ),
            _tool_call("read_file", {"file_path": "/workspace/missing.md"}, 1),
            ToolMessage(
                content="Error: file not found",
                name="read_file",
                tool_call_id="call-1",
                status="error",
            ),
        ],
        tools=[{"type": "function", "function": {"name": "read_file"}}],
        response_format=ToolStrategy(Result),
    )
    calls = 0

    async def handler(_: ModelRequest) -> ModelResponse:
        nonlocal calls
        calls += 1
        return ModelResponse(
            result=[AIMessage(content="", tool_calls=[])],
            structured_response=Result(value="unreviewable"),
        )

    with pytest.raises(AgentProtocolError) as error:
        await StructuredResultMiddleware().awrap_model_call(request, handler)

    assert calls == 0
    assert error.value.safe_message == "审核代理无法读取必需输入。"


@pytest.mark.asyncio
async def test_structured_result_allows_failed_non_required_read() -> None:
    class Result(BaseModel):
        value: str

    request = ModelRequest(
        model=FakeMessagesListChatModel(responses=[]),
        messages=[
            HumanMessage(content="Read the input, then return the result."),
            _tool_call("read_file", {"file_path": "/workspace/one-line.json"}, 1),
            ToolMessage(
                content="Error: Line offset 1 exceeds file length (1 lines)",
                name="read_file",
                tool_call_id="call-1",
                status="error",
            ),
        ],
        tools=[{"type": "function", "function": {"name": "read_file"}}],
        response_format=ToolStrategy(Result),
    )
    expected = Result(value="done")

    async def handler(_: ModelRequest) -> ModelResponse:
        return ModelResponse(
            result=[AIMessage(content="", tool_calls=[])],
            structured_response=expected,
        )

    response = await StructuredResultMiddleware().awrap_model_call(request, handler)

    assert response.structured_response == expected


@pytest.mark.asyncio
async def test_failed_non_required_read_does_not_block_completed_required_reads() -> None:
    class Result(BaseModel):
        value: str

    request = ModelRequest(
        model=FakeMessagesListChatModel(responses=[]),
        messages=[
            HumanMessage(
                content=(
                    "Review the input.\n"
                    "<pengine-required-read-paths>\n"
                    "/workspace/required.md\n"
                    "</pengine-required-read-paths>"
                )
            ),
            _tool_call("read_file", {"file_path": "/workspace/required.md"}, 1),
            ToolMessage(
                content="required contents",
                name="read_file",
                tool_call_id="call-1",
            ),
            _tool_call("read_file", {"file_path": "/workspace/optional.md"}, 2),
            ToolMessage(
                content="Error: file not found",
                name="read_file",
                tool_call_id="call-2",
                status="error",
            ),
        ],
        tools=[{"type": "function", "function": {"name": "read_file"}}],
        response_format=ToolStrategy(Result),
    )
    expected = Result(value="reviewed")

    async def handler(_: ModelRequest) -> ModelResponse:
        return ModelResponse(
            result=[AIMessage(content="", tool_calls=[])],
            structured_response=expected,
        )

    response = await StructuredResultMiddleware().awrap_model_call(request, handler)

    assert response.structured_response == expected


@pytest.mark.asyncio
async def test_unread_canon_result_is_blocked_until_engine_schedules_required_reads() -> None:
    paths = (
        "/workspace/current_character_biographies.md",
        "/workspace/current_relationship_logic.md",
        "/workspace/previous_character_biographies.md",
        "/workspace/previous_relationship_logic.md",
        "/workspace/current_story_review.json",
        "/workspace/previous_story_review.json",
    )
    evidence = "审核代理声称未读取这些输入：" + "、".join(paths) + "。"
    premature_review = CanonReviewerResult(
        passed=False,
        evidence=evidence,
        issues=[
            {
                "code": "unreviewable_input",
                "message": "未获得必需输入的实际内容。",
                "contract_refs": [],
                "script_excerpt": None,
                "contract_mutation_required": False,
            }
        ],
    )
    request = ModelRequest(
        model=FakeMessagesListChatModel(responses=[]),
        messages=[
            HumanMessage(
                content=(
                    "Review every mounted input before returning a result.\n"
                    "<pengine-required-read-paths>\n"
                    + "\n".join(paths)
                    + "\n</pengine-required-read-paths>"
                )
            )
        ],
        tools=[{"type": "function", "function": {"name": "read_file"}}],
        response_format=ToolStrategy(CanonReviewerResult),
    )
    calls: list[ModelRequest] = []

    async def handler(candidate: ModelRequest) -> ModelResponse:
        calls.append(candidate)
        if len(calls) == 1:
            return ModelResponse(
                result=[AIMessage(content="I cannot review files that were not provided.")],
                structured_response=None,
            )
        return ModelResponse(
            result=[AIMessage(content="", tool_calls=[])],
            structured_response=premature_review,
        )

    response = await StructuredResultMiddleware().awrap_model_call(request, handler)

    assert calls == []
    assert response.structured_response is None
    assert len(response.result) == 1
    message = response.result[0]
    assert isinstance(message, AIMessage)
    assert message.tool_calls[0]["name"] == "read_file"
    assert message.tool_calls[0]["args"] == {"file_path": paths[0]}
    _validate_result_language(
        premature_review,
        output_language=SIMPLIFIED_CHINESE,
        stage=InternalStage.GENERATING_CHARACTER_RELATIONSHIPS,
    )


@pytest.mark.asyncio
async def test_required_read_fails_closed_when_read_file_tool_is_missing() -> None:
    class Result(BaseModel):
        value: str

    request = ModelRequest(
        model=FakeMessagesListChatModel(responses=[]),
        messages=[
            HumanMessage(
                content=(
                    "Review the input.\n"
                    "<pengine-required-read-paths>\n"
                    "/workspace/candidate.md\n"
                    "</pengine-required-read-paths>"
                )
            )
        ],
        tools=[],
        response_format=ToolStrategy(Result),
    )
    calls = 0

    async def handler(_: ModelRequest) -> ModelResponse:
        nonlocal calls
        calls += 1
        return ModelResponse(result=[AIMessage(content="")])

    with pytest.raises(AgentProtocolError) as error:
        await StructuredResultMiddleware().awrap_model_call(request, handler)

    assert calls == 0
    assert error.value.safe_message == "审核代理缺少必需的读取工具。"


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
async def test_structured_result_middleware_keeps_only_latest_large_failed_result() -> None:
    valid_args = {
        "stage": "generating_episode_outline",
        "content": "完整分集大纲" * 4_000,
        "episode_count": 2,
        "episodes": [
            {"episode_number": 1, "plan": "第一集计划"},
            {"episode_number": 2, "plan": "第二集计划"},
        ],
        "story_contract": _story_contract(episode_count=2).model_dump(mode="json"),
    }
    first_invalid = copy.deepcopy(valid_args)
    first_invalid["story_contract"]["clues"] = [
        {
            "clue_id": "clue_watch",
            "description": "旧表",
            "introduced_episode": 1,
            "explained_episode": 2,
            "callback_episode": 1,
            "introduction_is_visible_or_audible": True,
        }
    ]
    second_invalid = copy.deepcopy(valid_args)
    second_invalid["story_contract"]["knowledge_states"][1]["known_fact_ids"] = []
    first_result = _tool_call("EpisodePlannerResult", first_invalid, 201)
    second_result = _tool_call("EpisodePlannerResult", second_invalid, 202)
    prior_correction = HumanMessage(
        content=(
            "Correct these validation errors: story_contract.clues.0: "
            "A clue callback cannot precede its explanation."
        )
    )
    request = ModelRequest(
        model=FakeMessagesListChatModel(responses=[]),
        messages=[
            HumanMessage(content="Create the complete episode outline."),
            _tool_call("read_file", {"file_path": "/workspace/story_outline.md"}, 200),
            ToolMessage(
                content="source material" * 2_000,
                tool_call_id="call-200",
                name="read_file",
            ),
            first_result,
            ToolMessage(
                content="Return a valid structured result.",
                tool_call_id="call-201",
                name="EpisodePlannerResult",
            ),
            prior_correction,
            second_result,
            ToolMessage(
                content="Return a valid structured result.",
                tool_call_id="call-202",
                name="EpisodePlannerResult",
            ),
        ],
        tools=[{"type": "function", "function": {"name": "read_file"}}],
        response_format=ToolStrategy(EpisodePlannerResult),
    )

    async def handler(candidate: ModelRequest) -> ModelResponse:
        assert candidate.tools == []
        assert len(candidate.messages) == 5
        assert candidate.messages[0] == request.messages[0]
        assert prior_correction in candidate.messages
        assert second_result in candidate.messages
        assert first_result not in candidate.messages
        assert not any(
            isinstance(message, ToolMessage) and message.name == "read_file"
            for message in candidate.messages
        )
        assert isinstance(candidate.messages[-1], HumanMessage)
        assert (
            "Character knowledge cannot silently disappear between episodes"
            in candidate.messages[-1].content
        )
        return ModelResponse(
            result=[AIMessage(content="", tool_calls=[])],
            structured_response=EpisodePlannerResult.model_validate(valid_args),
        )

    response = await StructuredResultMiddleware().awrap_model_call(request, handler)

    assert isinstance(response.structured_response, EpisodePlannerResult)


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


def test_compact_supervisor_messages_keeps_latest_complete_exchange_and_feedback() -> None:
    initial_task = HumanMessage(content="Execute the bounded workflow.")
    old_call = _tool_call("task", {"description": "old task"}, 101)
    old_result = ToolMessage(
        content="old task result " * 4_000,
        tool_call_id="call-101",
        name="task",
    )
    old_feedback = HumanMessage(content="Old correction should be discarded.")
    latest_call = _tool_call("WorkflowCompletion", {"completed": True}, 102)
    latest_result = ToolMessage(
        content="latest completion result",
        tool_call_id="call-102",
        name="WorkflowCompletion",
    )
    correction = HumanMessage(content="Continue with the latest completed state.")
    messages = [
        initial_task,
        old_call,
        old_result,
        old_feedback,
        latest_call,
        latest_result,
        correction,
    ]

    compacted = _compact_supervisor_messages(messages)

    assert compacted == [initial_task, latest_call, latest_result, correction]
    assert old_result not in compacted
    assert old_feedback not in compacted
    assert not any(
        isinstance(message, ToolMessage) and message.tool_call_id == "call-101"
        for message in compacted
    )


def test_compact_supervisor_messages_leaves_incomplete_history_untouched() -> None:
    messages = [
        HumanMessage(content="Execute the bounded workflow."),
        _tool_call("task", {"description": "unfinished"}, 103),
        ToolMessage(content="orphan", tool_call_id="orphan-call", name="task"),
        HumanMessage(content="Retry safely."),
    ]

    assert _compact_supervisor_messages(messages) == messages


def test_supervisor_history_compaction_overrides_only_model_messages() -> None:
    initial_task = HumanMessage(content="Execute the bounded workflow.")
    old_call = _tool_call("task", {"description": "old task"}, 104)
    old_result = ToolMessage(
        content="old result " * 2_000,
        tool_call_id="call-104",
        name="task",
    )
    latest_call = _tool_call("WorkflowCompletion", {"completed": True}, 105)
    latest_result = ToolMessage(
        content="latest result",
        tool_call_id="call-105",
        name="WorkflowCompletion",
    )
    state = {"checkpoint_marker": "unchanged", "files": {"/workspace/task.md": "same"}}
    request = ModelRequest(
        model=FakeMessagesListChatModel(responses=[]),
        messages=[initial_task, old_call, old_result, latest_call, latest_result],
        tools=[{"type": "function", "function": {"name": "task"}}],
        state=state,
    )

    supervisor_middleware = ToolAllowlistMiddleware(
        frozenset({"task"}),
        system_prompt="Run the workflow.",
        compact_tool_history=True,
    )
    compacted_request = supervisor_middleware._filter_request(request)

    assert compacted_request.messages == [initial_task, latest_call, latest_result]
    assert compacted_request.state is request.state
    assert compacted_request.state == state

    specialist_middleware = ToolAllowlistMiddleware(
        frozenset({"task"}),
        system_prompt="Run the specialist task.",
    )
    specialist_request = specialist_middleware._filter_request(request)
    assert specialist_middleware.compact_tool_history is False
    assert specialist_request.messages == request.messages


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
                    "path": "/episodes/0/plan",
                    "expected": "第一集计划",
                    "value": "调查真相的第一集计划",
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
    assert repaired.episodes[0].plan == "调查真相的第一集计划"
    assert candidate["content"] == "旧分集大纲。其余内容保持。"
    assert candidate["episodes"][0]["plan"] == "第一集计划"


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
                    "path": "/episodes/0/plan",
                    "expected": "错误旧值",
                    "value": "调查真相的第一集计划",
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
        "episode_count": 2,
        "episodes": [
            {"episode_number": 1, "plan": "第一集计划"},
            {"episode_number": 2, "plan": "第二集计划"},
        ],
        "story_contract": contract.model_dump(mode="json"),
    }
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
                    "path": "/episodes/1/plan",
                    "expected": "第二集计划",
                    "value": "修复后的第二集计划",
                },
            ],
        }
    )

    repaired = _apply_outline_repair_patch(candidate, patch)

    assert [episode.plan for episode in repaired.episodes] == [
        "修复后的第一集计划",
        "修复后的第二集计划",
    ]


def test_outline_repair_patch_rejects_non_replace_operations() -> None:
    with pytest.raises(ValidationError, match="replace"):
        OutlineRepairPatch.model_validate(
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
    contract = _story_contract(episode_count=2)
    candidate = {
        "stage": "generating_episode_outline",
        "content": "分集大纲",
        "episode_count": 2,
        "episodes": [
            {"episode_number": 1, "plan": "第一集计划"},
            {"episode_number": 2, "plan": "第二集计划"},
        ],
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
                    "path": "/episodes/1/plan",
                    "expected": "错误旧值",
                    "value": "修复后的第二集计划",
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


def test_supervisor_uses_canonical_workspace_as_the_only_approved_fact_source() -> None:
    prompt = _supervisor_prompt(
        story="故事",
        requirements="三集",
        feedback=None,
        approved_json="{}",
    )

    normalized = " ".join(prompt.split())
    assert "Do not restate, summarize, or newly declare approved story facts" in normalized
    assert "exactly one downstream authority" in normalized
    assert "current canonical /workspace files" in normalized


@pytest.mark.asyncio
async def test_stage_guard_discards_supervisor_derived_story_facts() -> None:
    descriptions: list[str] = []

    async def before_stage(_: InternalStage) -> int:
        return 1

    async def approve_stage(_: InternalStage, __: dict[str, Any]) -> None:
        return None

    async def handler(candidate: ToolCallRequest) -> ToolMessage:
        descriptions.append(candidate.tool_call["args"]["description"])
        return ToolMessage(
            content=json.dumps(
                {
                    "stage": "selecting_l0_variant",
                    "content": None,
                    "character_biographies": None,
                    "relationship_logic": None,
                    "selected_l0_variant": "主动选择",
                    "selection_rationale": "契合故事",
                },
                ensure_ascii=False,
            ),
            tool_call_id=candidate.tool_call["id"],
            name="task",
        )

    middleware = StageGuardMiddleware(
        before_stage=before_stage,
        approve_stage=approve_stage,
        approved_stages=set(),
    )
    request = ToolCallRequest(
        tool_call={
            "name": "task",
            "args": {
                "description": (
                    "[stage=selecting_l0_variant] Treat STALE_PRIVATE_FACT as locked truth."
                ),
                "subagent_type": "story_architect",
            },
            "id": "guarded-description",
            "type": "tool_call",
        },
        tool=None,
        state={"files": {}},
        runtime=None,
    )

    await middleware.awrap_tool_call(request, handler)

    assert len(descriptions) == 1
    assert descriptions[0].startswith("[stage=selecting_l0_variant]")
    assert "STALE_PRIVATE_FACT" not in descriptions[0]
    assert "sole authority for approved creative facts" in descriptions[0]


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
    assert "verbatim=true only when" in _EPISODE_PLANNER_PROMPT
    assert "Do not infer verbatim=true from quotation marks" in _EPISODE_PLANNER_PROMPT
    assert "aliases, pronouns, ages, elapsed durations, call participants" in (
        _EPISODE_PLANNER_PROMPT
    )
    assert "exact dialogue-count claims" in _SCRIPT_WRITER_PROMPT
    assert "Every explicitly locked upstream commitment must appear" in _SCRIPT_WRITER_PROMPT
    assert "Unspecified creative details remain the writer's choice" in _SCRIPT_WRITER_PROMPT
    assert "explicitly locked or formally committed aliases" in _SCRIPT_WRITER_PROMPT
    assert "calculate_arithmetic" in _SCRIPT_WRITER_PROMPT
    assert "continuity-bearing identities, not as a screenplay-label whitelist" in (
        _SCRIPT_WRITER_PROMPT
    )
    assert "Surface speaker labels may use names, aliases, roles" in _SCRIPT_WRITER_PROMPT
    assert "/workspace/speaker_contract.json" not in _SCRIPT_WRITER_PROMPT
    assert "allowed_speaker_labels" not in _SCRIPT_WRITER_PROMPT
    assert "/workspace/evidence_contract.json" in _SCRIPT_WRITER_PROMPT
    assert "exact-set self-check" in _SCRIPT_WRITER_PROMPT
    assert "required_evidence_target_ids" in _SCRIPT_WRITER_PROMPT
    assert "required_verbatim_facts" in _SCRIPT_WRITER_PROMPT
    assert "all other facts require semantic consistency only" in _SCRIPT_WRITER_PROMPT
    assert "call participants" in _SCRIPT_WRITER_PROMPT
    assert "complete non-null state_delta" in _SCRIPT_WRITER_PROMPT
    assert "/workspace/suffix_rewrite_review.json" in _SCRIPT_WRITER_PROMPT
    assert "read-only bound" in _SCRIPT_WRITER_PROMPT
    assert "do not reproduce the named defect" in _SCRIPT_WRITER_PROMPT
    assert "grandfathered pre-contract run" not in _SCRIPT_WRITER_PROMPT
    assert "/workspace/speaker_contract.json" not in _EPISODE_REPAIR_PROMPT
    assert "/workspace/evidence_contract.json" in _EPISODE_REPAIR_PROMPT
    assert "evidence_coverage_mismatch" in _EPISODE_REPAIR_PROMPT
    assert "issue.contract_refs" in _EPISODE_REPAIR_PROMPT
    assert "no extras" in _EPISODE_REPAIR_PROMPT
    assert "no duplicates" in _EPISODE_REPAIR_PROMPT
    assert "unknown_speaker" in _EPISODE_REPAIR_PROMPT
    assert "contextually proven new continuity-bearing character" in _EPISODE_REPAIR_PROMPT
    assert "without normalizing screenplay notation" in _EPISODE_REPAIR_PROMPT
    assert "alias" in _EPISODE_REPAIR_PROMPT
    assert "occupational title" in _EPISODE_REPAIR_PROMPT
    assert "state_delta" in _EPISODE_REPAIR_PROMPT
    assert "/workspace/suffix_rewrite_review.json" in _EPISODE_REPAIR_PROMPT
    assert "locked story contract priority" in _EPISODE_REPAIR_PROMPT
    assert "verbatim_fact_missing" in _EPISODE_REPAIR_PROMPT
    assert "required_verbatim_facts" in _EPISODE_REPAIR_PROMPT


def test_specialist_prompts_apply_l0_by_stage_without_copying_source_content() -> None:
    assert "complete /persona/l0.md" in _STORY_ARCHITECT_PROMPT
    assert "exact declared [ID:<value>]" in _STORY_ARCHITECT_PROMPT
    assert "protagonist, central conflict, choice, cost, and ending" in (_STORY_ARCHITECT_PROMPT)
    assert "Do not invent a universal interpretation" in _STORY_ARCHITECT_PROMPT
    assert "selected L0 facet" in _EPISODE_PLANNER_PROMPT
    assert "this persona's L0" in _EPISODE_PLANNER_PROMPT
    assert "current persona's exact red" in _SCRIPT_WRITER_PROMPT
    for label in ("母题兑现：", "选定侧面：", "雷区：", "温度："):
        assert label in _QUALITY_REVIEWER_PROMPT
    assert "do not reselect or reopen L0" in _QUALITY_REVIEWER_PROMPT

    combined = "\n".join(
        (
            _STORY_ARCHITECT_PROMPT,
            _EPISODE_PLANNER_PROMPT,
            _SCRIPT_WRITER_PROMPT,
            _QUALITY_REVIEWER_PROMPT,
        )
    )
    assert "初心 vs 现实的落差" not in combined
    assert "warmth before harm" not in combined
    assert "explanatory voiceover" not in combined


def test_all_model_stage_prompts_read_full_advisory_soul() -> None:
    prompts = (
        _STORY_ARCHITECT_PROMPT,
        _EPISODE_PLANNER_PROMPT,
        _SCRIPT_WRITER_PROMPT,
        _EPISODE_REPAIR_PROMPT,
        _QUALITY_REVIEWER_PROMPT,
        _EPISODE_REVIEWER_PROMPT,
        _SERIES_REVIEWER_PROMPT,
        _CANON_REVIEWER_PROMPT,
        _STORY_REPAIR_PROMPT,
    )

    for prompt in prompts:
        assert "When /persona/soul.md is present, read its complete text" in prompt
        assert "Never summarize, retrieve, slice, or silently truncate Soul" in prompt
        assert "must never pass or reject work" in prompt
        assert "hard Canon" in prompt
        assert "StoryContract" in prompt


def test_all_model_stage_prompts_enforce_l3_method_and_authority_boundaries() -> None:
    prompts = (
        _STORY_ARCHITECT_PROMPT,
        _EPISODE_PLANNER_PROMPT,
        _SCRIPT_WRITER_PROMPT,
        _EPISODE_REPAIR_PROMPT,
        _QUALITY_REVIEWER_PROMPT,
        _EPISODE_REVIEWER_PROMPT,
        _SERIES_REVIEWER_PROMPT,
        _CANON_REVIEWER_PROMPT,
        _STORY_REPAIR_PROMPT,
    )

    for prompt in prompts:
        assert "When /persona/l3.md is present, read its complete text" in prompt
        assert "never use L3 to add, rename, reweight, or reselect an L0 variant" in prompt
        assert "do not reopen the approved direction" in prompt
        assert "never use L3 to rediffuse" in prompt
        assert "L3 is not Gate evidence" in prompt
        assert "SeriesState" in prompt


def test_all_model_stage_prompts_enforce_l4_authority_boundaries() -> None:
    prompts = (
        _STORY_ARCHITECT_PROMPT,
        _EPISODE_PLANNER_PROMPT,
        _SCRIPT_WRITER_PROMPT,
        _EPISODE_REPAIR_PROMPT,
        _QUALITY_REVIEWER_PROMPT,
        _EPISODE_REVIEWER_PROMPT,
        _SERIES_REVIEWER_PROMPT,
        _CANON_REVIEWER_PROMPT,
        _STORY_REPAIR_PROMPT,
    )

    for prompt in prompts:
        assert "Only rules explicitly labeled as creator-confirmed hard rules" in prompt
        assert "confirmed creative advice is advisory and non-blocking" in prompt
        assert "Pengine owns episode count, duration, and scene-count defaults" in prompt
        assert "explicit user requirements and locked production parameters override" in prompt

    assert "L4-A：" in _QUALITY_REVIEWER_PROMPT
    assert "短剧硬规则：" in _QUALITY_REVIEWER_PROMPT
    assert "产品参数：" in _QUALITY_REVIEWER_PROMPT
    assert "L4硬规则：" in _CANON_REVIEWER_PROMPT
    assert "L4硬规则：" in _SERIES_REVIEWER_PROMPT


def test_direct_patch_prompt_inlines_project_and_soul_but_not_l3_body() -> None:
    project = "# Project\n\nPROJECT-UNIQUE-RUNTIME-CONSTITUTION"
    soul = "# Soul\n\nfirst line\nlast line"
    l3 = "# L3\n\nprivate complete method"
    persona_files = {
        "/persona/project.md": project,
        "/persona/soul.md": soul,
        "/persona/l3.md": l3,
    }
    prompt = _with_inline_project(
        _with_inline_soul(_with_l3_policy("Repair the candidate."), persona_files),
        persona_files,
    )

    assert prompt.count(project) == 1
    assert prompt.count(soul) == 1
    assert l3 not in prompt
    assert _PROJECT_CREATIVE_POLICY in prompt
    assert _PROJECT_INLINE_MARKER in prompt
    assert "Complete text of /persona/soul.md:" in prompt
    assert "Never summarize, retrieve, slice, or silently truncate Soul" in prompt
    assert _with_inline_soul("Repair the candidate.", {}) == "Repair the candidate."

    assert _with_inline_project(prompt, persona_files) == prompt
    for missing in ({}, {"/persona/project.md": "   "}):
        with pytest.raises(AgentProtocolError) as error:
            _with_inline_project("Generate content.", missing)
        assert error.value.safe_message == "当前人格缺少可用的 Project 宪章。"


def test_reviewer_prompts_keep_project_as_a_boundary_not_a_style_gate() -> None:
    for prompt in (
        _QUALITY_REVIEWER_PROMPT,
        _CANON_REVIEWER_PROMPT,
        _EPISODE_REVIEWER_PROMPT,
        _SERIES_REVIEWER_PROMPT,
    ):
        assert _PROJECT_REVIEW_BOUNDARY in prompt
        assert "Project adds no separate Gate" in prompt
        assert _PROJECT_INLINE_MARKER not in prompt


def test_l4_reviewer_prompt_only_locks_explicit_verbatim_facts() -> None:
    assert "/workspace/story_contract.json" in _QUALITY_REVIEWER_PROMPT
    assert "only when that fact has verbatim=true" in _QUALITY_REVIEWER_PROMPT
    assert "kind=text" in _QUALITY_REVIEWER_PROMPT
    assert "quotation marks" in _QUALITY_REVIEWER_PROMPT
    assert "subject or predicate" in _QUALITY_REVIEWER_PROMPT
    assert "semantic consistency only" in _QUALITY_REVIEWER_PROMPT


def test_internal_runtime_leak_policy_requires_unambiguous_source_evidence() -> None:
    policy = " ".join(_INTERNAL_RUNTIME_LEAK_POLICY.split())

    for internal_marker in (
        "exact canonical /workspace path",
        "fact/clue/obligation stable ID",
        "tool-call or model-message envelope",
        "validation or retry status text",
        "raw contract serialization",
    ):
        assert internal_marker in policy
    assert "concrete runtime-only token or record" in policy
    assert "not established as story-world content" in policy
    assert "quote the exact screenplay excerpt" in policy
    assert "name the matching private source" in policy
    assert "runtime provenance is unambiguous" in policy
    assert "If provenance is ambiguous, pass this dimension" in policy
    assert "Story-world facts encoded by a contract are content" in policy


def test_internal_runtime_leak_policy_accepts_diverse_screenplay_forms() -> None:
    policy = " ".join(_INTERNAL_RUNTIME_LEAK_POLICY.split())

    for legitimate_content in (
        "Episode, chapter, act, and scene headings",
        "title cards",
        "recap labels",
        "end markers such as 本集终",
        "screenplay directions",
        "arithmetic, equations, mental calculation, checking, and other reasoning",
        "JSON, code, paths, tools, models, AI, or validation",
    ):
        assert legitimate_content in policy
    assert "Never reject or rewrite content merely because" in policy
    assert "episode_number is not provenance evidence" in policy
    assert "the screenplay may show operands, equations, mental calculation" in (
        _SCRIPT_WRITER_PROMPT
    )
    assert "Never copy a tool-call envelope or private validation log" in _SCRIPT_WRITER_PROMPT


def test_internal_runtime_leak_review_prompts_apply_proof_standard() -> None:
    assert "At accepting_l4" in _QUALITY_REVIEWER_PROMPT
    assert "applicable explicit persona gate rule" in _QUALITY_REVIEWER_PROMPT
    assert "matters of taste are not sufficient reasons" in _QUALITY_REVIEWER_PROMPT
    assert "provenance and evidence standard is a blocking leakage defect" in (
        _QUALITY_REVIEWER_PROMPT
    )
    assert "script_defect" in _SERIES_REVIEWER_PROMPT
    assert "current prefix contains the blocker" in _SERIES_REVIEWER_PROMPT
    assert "earliest affected episode N" in _SERIES_REVIEWER_PROMPT
    assert "Ordinary SeriesBible prose, screenplay format or style" in _SERIES_REVIEWER_PROMPT
    assert "never defects on their own" in _EPISODE_REVIEWER_PROMPT
    for prompt in (_EPISODE_REVIEWER_PROMPT, _SERIES_REVIEWER_PROMPT):
        assert "/workspace/series_prefix.json" in prompt
        assert "trusted runtime metadata" in prompt
        assert "episodes[].content" in prompt


def test_evidence_contract_exposes_episode_verbatim_facts_and_rejected_issue() -> None:
    contract = _story_contract(verbatim_episodes={1})
    issue = EpisodeReviewerResult.model_validate(
        {
            "passed": False,
            "evidence": "事实未逐字出现",
            "issues": [
                {
                    "code": "verbatim_fact_missing",
                    "message": "事实值未出现",
                    "contract_refs": ["fact_ep1"],
                }
            ],
        }
    ).issues[0]

    evidence_contract = _evidence_contract(
        contract,
        1,
        rejected_issues=[issue],
        phase="episode_repair",
    )

    assert evidence_contract["required_verbatim_facts"] == [
        {"fact_id": "fact_ep1", "value": "事实1"}
    ]
    assert evidence_contract["rejected_issues"] == [
        {
            "code": "verbatim_fact_missing",
            "contract_refs": ["fact_ep1"],
            "message": "事实值未出现",
            "script_excerpt": None,
        }
    ]


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

    project = "# Project\n\nPROJECT-UNIQUE-RUNTIME-CONSTITUTION"
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
                persona_files={"/persona/project.md": project},
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
    subagents_by_name = {spec["name"]: spec for spec in captured["subagents"]}
    for name in (
        "story_architect",
        "episode_planner",
        "script_writer",
        "episode_repair",
        "story_repair",
    ):
        system_prompt = subagents_by_name[name]["system_prompt"]
        assert system_prompt.count(project) == 1
        assert _PROJECT_CREATIVE_POLICY in system_prompt
    for name in (
        "quality_reviewer",
        "canon_reviewer",
        "episode_reviewer",
        "series_reviewer",
    ):
        system_prompt = subagents_by_name[name]["system_prompt"]
        assert project not in system_prompt
        assert _PROJECT_REVIEW_BOUNDARY in system_prompt
    assert project not in captured["system_prompt"]
    assert _PROJECT_REVIEW_BOUNDARY not in captured["system_prompt"]
    for name in (
        "script_writer",
        "quality_reviewer",
        "episode_reviewer",
        "series_reviewer",
    ):
        system_prompt = subagents_by_name[name]["system_prompt"]
        assert system_prompt.count(_INTERNAL_RUNTIME_LEAK_POLICY) == 1
        assert f"\n\n{_INTERNAL_RUNTIME_LEAK_POLICY}" in system_prompt
    assert subagents_by_name["episode_repair"]["permissions"] == REVIEW_FILE_PERMISSIONS
    assert subagents_by_name["story_repair"]["permissions"] == REVIEW_FILE_PERMISSIONS
    supervisor_allowlists = [
        middleware
        for middleware in captured["middleware"]
        if isinstance(middleware, ToolAllowlistMiddleware)
    ]
    assert len(supervisor_allowlists) == 1
    assert supervisor_allowlists[0].compact_tool_history is True
    for spec in captured["subagents"]:
        for middleware in spec["middleware"]:
            if isinstance(middleware, ToolAllowlistMiddleware):
                assert middleware.compact_tool_history is False


@pytest.mark.asyncio
async def test_episode_writer_treats_speaker_labels_as_format_not_a_whitelist(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database = tmp_path / "format-agnostic-speaker-checkpoints.sqlite3"
    contract = _story_contract(verbatim_episodes={1})
    captured_requests: list[ToolCallRequest] = []
    original_call = StageGuardMiddleware._call_structured_stage

    async def capture_call(
        middleware: StageGuardMiddleware,
        stage: InternalStage,
        request: ToolCallRequest,
        handler: Any,
        args: Mapping[str, Any],
        *,
        expected_episode_number: int | None = None,
    ) -> tuple[Any, Mapping[str, Any]]:
        if stage is InternalStage.GENERATING_EPISODE_SCRIPTS:
            captured_requests.append(request)
        return await original_call(
            middleware,
            stage,
            request,
            handler,
            args,
            expected_episode_number=expected_episode_number,
        )

    monkeypatch.setattr(StageGuardMiddleware, "_call_structured_stage", capture_call)

    async def before_stage(_: InternalStage) -> int:
        return 1

    async def approve_stage(_: InternalStage, __: dict[str, Any]) -> None:
        return None

    async with AsyncSqliteSaver.from_conn_string(str(database)) as saver:
        await saver.setup()
        workflow = _fake_workflow(
            model=ToolCallingFakeModel(responses=_successful_responses(contract=contract)),
            checkpointer=saver,
            provider_profile_key="toolcallingfakemodel",
        )
        await workflow.execute(
            thread_id="format-agnostic-speaker-writer-thread",
            story="故事",
            requirements="要求",
            persona_files={"/persona/project.md": "规则"},
            before_stage=before_stage,
            approve_stage=approve_stage,
            **_episode_hook_kwargs()[0],
        )

    assert len(captured_requests) == 1
    request = captured_requests[0]
    files = request.state["files"]
    assert "/workspace/speaker_contract.json" not in files
    evidence_contract = json.loads(files["/workspace/evidence_contract.json"]["content"])
    assert evidence_contract["episode_number"] == 1
    assert evidence_contract["required_evidence_target_ids"] == [
        "fact_ep1",
        "obligation_ep1",
    ]
    assert evidence_contract["required_verbatim_facts"] == [
        {"fact_id": "fact_ep1", "value": "事实1"}
    ]
    assert evidence_contract["rejected_issues"] == []
    assert "/workspace/suffix_rewrite_review.json" not in files
    description = request.tool_call["args"]["description"]
    assert "Screenplay labels and dialogue notation are format choices" in description
    assert "without normalizing aliases, roles, generic labels" in description
    assert "Allowed exact speaker labels" not in description
    assert 'required_evidence_target_ids=["fact_ep1", "obligation_ep1"]' in description
    assert 'required_verbatim_facts=[{"fact_id": "fact_ep1", "value": "事实1"}]' in description
    assert "exact-set self-check" in description


@pytest.mark.asyncio
async def test_suffix_rewrite_feedback_is_injected_only_from_effective_episode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database = tmp_path / "suffix-rewrite-checkpoints.sqlite3"
    captured_requests: list[ToolCallRequest] = []
    original_call = StageGuardMiddleware._call_structured_stage

    async def capture_call(
        middleware: StageGuardMiddleware,
        stage: InternalStage,
        request: ToolCallRequest,
        handler: Any,
        args: Mapping[str, Any],
        *,
        expected_episode_number: int | None = None,
    ) -> tuple[Any, Mapping[str, Any]]:
        if stage is InternalStage.GENERATING_EPISODE_SCRIPTS:
            captured_requests.append(request)
        return await original_call(
            middleware,
            stage,
            request,
            handler,
            args,
            expected_episode_number=expected_episode_number,
        )

    monkeypatch.setattr(StageGuardMiddleware, "_call_structured_stage", capture_call)
    feedback = {
        "version": 1,
        "effective_earliest_affected_episode": 1,
        "reviews": [
            {
                "review_id": "review-2",
                "category": "script_defect",
                "evidence": "第1集校牌事实错误。",
                "earliest_affected_episode": 1,
                "binding": {
                    "review_epoch": 2,
                    "design_candidate_id": "design-1",
                    "batch_id": "batch-1",
                },
            },
            {
                "review_id": "review-3",
                "category": "script_defect",
                "evidence": "第2集后续动机错误。",
                "earliest_affected_episode": 2,
                "binding": {
                    "review_epoch": 3,
                    "design_candidate_id": "design-1",
                    "batch_id": "batch-1",
                },
            },
        ],
    }

    async with AsyncSqliteSaver.from_conn_string(str(database)) as saver:
        await saver.setup()
        workflow = _fake_workflow(
            model=ToolCallingFakeModel(responses=_successful_responses()),
            checkpointer=saver,
            provider_profile_key="toolcallingfakemodel",
        )
        await workflow.execute(
            thread_id="suffix-rewrite-writer-thread",
            story="故事",
            requirements="要求",
            persona_files={"/persona/project.md": "规则"},
            before_stage=lambda _stage: _async_one(),
            approve_stage=lambda _stage, _payload: _async_none(),
            suffix_rewrite_feedback=feedback,
            **_episode_hook_kwargs()[0],
        )

    assert len(captured_requests) == 1
    request = captured_requests[0]
    injected_feedback = json.loads(
        request.state["files"]["/workspace/suffix_rewrite_review.json"]["content"]
    )
    assert injected_feedback == _suffix_rewrite_feedback_for_episode(feedback, 1)
    assert [item["review_id"] for item in injected_feedback["reviews"]] == ["review-2"]
    description = request.tool_call["args"]["description"]
    assert "suffix rewrite" in description
    assert "fix every named conflict" in description
    assert "locked story contract has priority" in description
    assert "do not reproduce the named defect" in description
    assert _suffix_rewrite_feedback_for_episode(feedback, 0) is None
    assert _suffix_rewrite_feedback_for_episode(feedback, 1)["reviews"] == [feedback["reviews"][0]]
    assert _suffix_rewrite_feedback_for_episode(feedback, 2) == feedback


def test_suffix_rewrite_feedback_filters_future_reviews_per_episode() -> None:
    feedback = {
        "version": 1,
        "effective_earliest_affected_episode": 8,
        "reviews": [
            {"review_id": "review-2", "earliest_affected_episode": 8},
            {"review_id": "review-3", "earliest_affected_episode": 9},
        ],
    }

    assert _suffix_rewrite_feedback_for_episode(feedback, 7) is None
    episode_eight = _suffix_rewrite_feedback_for_episode(feedback, 8)
    assert episode_eight is not None
    assert episode_eight["effective_earliest_affected_episode"] == 8
    assert [review["review_id"] for review in episode_eight["reviews"]] == ["review-2"]
    episode_nine = _suffix_rewrite_feedback_for_episode(feedback, 9)
    assert episode_nine is not None
    assert episode_nine["effective_earliest_affected_episode"] == 8
    assert [review["review_id"] for review in episode_nine["reviews"]] == [
        "review-2",
        "review-3",
    ]


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
        project = "# Project\n\nPROJECT-UNIQUE-RUNTIME-CONSTITUTION"
        result = await workflow.execute(
            thread_id="initial-thread",
            story="一个人回乡面对旧事。",
            requirements="生成完整短剧。",
            persona_files={"/persona/project.md": project},
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
        assert not all_tool_names & {
            "write_todos",
            "ls",
            "write_file",
            "edit_file",
            "glob",
            "grep",
        }
        assert len(model.bound_tool_names) == len(model.model_system_prompts)
        creative_result_tools = {
            "StoryArchitectResult",
            "EpisodePlannerResult",
            "ScriptWriterResult",
        }
        reviewer_result_tools = {
            "QualityReviewerResult",
            "CanonReviewerResult",
            "EpisodeReviewerResult",
        }
        for offered_tools, system_prompt in zip(
            model.bound_tool_names,
            model.model_system_prompts,
            strict=True,
        ):
            tool_names = set(offered_tools)
            if tool_names & creative_result_tools:
                assert system_prompt.count(project) == 1
            if tool_names & reviewer_result_tools:
                assert project not in system_prompt
                assert _PROJECT_REVIEW_BOUNDARY in system_prompt
            if "WorkflowCompletion" in tool_names:
                assert project not in system_prompt
        known_tool_names = {
            "task",
            "read_file",
            "calculate_arithmetic",
            "retrieve_persona_references",
            "write_todos",
            "ls",
            "write_file",
            "edit_file",
            "glob",
            "grep",
            "execute",
        }
        for offered_tools, system_prompt in zip(
            model.bound_tool_names,
            model.model_system_prompts,
            strict=True,
        ):
            assert system_prompt
            for tool_name in known_tool_names:
                assert _prompt_mentions_tool(system_prompt, tool_name) is (
                    tool_name in offered_tools
                ), (tool_name, offered_tools, system_prompt)
            for tool_name in offered_tools:
                assert _prompt_mentions_tool(system_prompt, tool_name), (
                    tool_name,
                    offered_tools,
                    system_prompt,
                )

        def bindings_for(result_tool: str) -> list[set[str]]:
            return [set(names) for names in model.bound_tool_names if result_tool in names]

        workflow_bindings = bindings_for("WorkflowCompletion")
        assert workflow_bindings
        assert all(names <= {"task", "WorkflowCompletion"} for names in workflow_bindings)
        assert {"task", "WorkflowCompletion"} in workflow_bindings

        for result_tool in (
            "StoryArchitectResult",
            "EpisodePlannerResult",
            "ScriptWriterResult",
            "QualityReviewerResult",
            "CanonReviewerResult",
            "EpisodeReviewerResult",
        ):
            result_bindings = bindings_for(result_tool)
            assert result_bindings, result_tool
            assert all("read_file" in names for names in result_bindings), result_tool
            assert all(
                result_tool in names
                and names
                <= {
                    result_tool,
                    "read_file",
                    "calculate_arithmetic",
                    "retrieve_persona_references",
                }
                for names in result_bindings
            ), result_tool

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
            _assert_task_lists_exact_workspace_paths(candidate)
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
            _assert_task_lists_exact_workspace_paths(candidate)
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
                                "contract_mutation_required": False,
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
                                "contract_mutation_required": False,
                            }
                        ],
                    }
            else:
                assert "二十二岁" in current
                assert "电话对象写成程屿" in current
                payload = {
                    "passed": True,
                    "evidence": "L4硬规则：故事工件一致",
                    "issues": [],
                }
            payload["prior_issue_closures"] = _resolved_prior_story_closures(candidate)
        return ToolMessage(
            content=json.dumps(payload, ensure_ascii=False),
            tool_call_id=candidate.tool_call["id"],
            name="task",
        )

    returned = await middleware.awrap_tool_call(request, handler)

    assert review_calls == 4
    assert repair_calls == 1
    assert isinstance(returned, ToolMessage)
    returned_payload = json.loads(returned.content)
    assert "二十二岁，比程屿大两岁" in returned_payload["relationship_logic"]
    assert "电话对象写成程屿" in returned_payload["relationship_logic"]
    assert "二十四岁" not in returned_payload["relationship_logic"]
    assert "电话对象写成周砚" not in returned_payload["relationship_logic"]
    assert "consistency_review" not in returned_payload
    assert "consistency_repair_rounds" not in returned_payload
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
                            "contract_mutation_required": False,
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
                            "contract_mutation_required": False,
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
                            "contract_mutation_required": False,
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
                            "contract_mutation_required": False,
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
                payload = {
                    "passed": True,
                    "evidence": "L4硬规则：故事工件一致",
                    "issues": [],
                }
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
async def test_contract_review_repairs_once_before_outline_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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
                    "contract_mutation_required": False,
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
            {
                "passed": True,
                "evidence": "L4硬规则：修复后合同一致",
                "issues": [],
            },
            102,
        ),
    )
    approved: dict[InternalStage, dict[str, Any]] = {}
    inlined_project_prompts: list[str] = []
    original_inline_project = _with_inline_project

    def capture_inline_project(prompt: str, persona_files: Mapping[str, str]) -> str:
        inlined = original_inline_project(prompt, persona_files)
        inlined_project_prompts.append(inlined)
        return inlined

    monkeypatch.setattr("pengine.agents._with_inline_project", capture_inline_project)

    async def before_stage(_: InternalStage) -> int:
        return 1

    async def approve_stage(stage: InternalStage, payload: dict[str, Any]) -> None:
        approved[stage] = payload

    async with AsyncSqliteSaver.from_conn_string(str(database)) as saver:
        await saver.setup()
        model = ToolCallingFakeModel(responses=responses)
        project = "# Project\n\nPROJECT-DIRECT-OUTLINE-PATCH"
        workflow = _fake_workflow(
            model=model,
            checkpointer=saver,
            provider_profile_key="toolcallingfakemodel",
        )
        await workflow.execute(
            thread_id="contract-repair-thread",
            story="故事",
            requirements="要求",
            persona_files={"/persona/project.md": project},
            before_stage=before_stage,
            approve_stage=approve_stage,
            **_episode_hook_kwargs()[0],
        )

    outline = approved[InternalStage.GENERATING_EPISODE_OUTLINE]
    assert outline["contract_repair_rounds"] == 1
    assert outline["contract_review"]["passed"] is True
    assert outline["content"] == "修复后的分集大纲"
    assert len(outline["story_contract_sha256"]) == 64
    outline_patch_prompts = [
        prompt
        for prompt in inlined_project_prompts
        if "Repair only the confirmed canon-review issues" in prompt
    ]
    assert len(outline_patch_prompts) == 1
    assert outline_patch_prompts[0].count(project) == 1


@pytest.mark.asyncio
async def test_story_patch_direct_call_inlines_project_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database = tmp_path / "story-project-patch-checkpoints.sqlite3"
    responses = _successful_responses()
    story_result_index = _index_of_tool_call(
        responses,
        "StoryArchitectResult",
        occurrence=2,
    )
    story_payload = copy.deepcopy(responses[story_result_index].tool_calls[0]["args"])
    story_payload["content"] = "第一行保留。\n旧冲突。\n第三行保留。\n第四行保留。"
    responses[story_result_index] = _tool_call(
        "StoryArchitectResult",
        story_payload,
        story_result_index,
    )
    story_review_index = _index_of_tool_call(
        responses,
        "CanonReviewerResult",
        occurrence=1,
    )
    failed_review_payload = {
        "passed": False,
        "evidence": "故事大纲含有已确认冲突",
        "issues": [
            {
                "code": "stale_story_fact",
                "message": "把旧冲突改为已修复事实。",
                "script_excerpt": "旧冲突。",
                "contract_mutation_required": False,
            }
        ],
    }
    responses[story_review_index] = _tool_call(
        "CanonReviewerResult",
        failed_review_payload,
        story_review_index,
    )
    prior_issue_id = _canon_issue_ledger(CanonReviewerResult.model_validate(failed_review_payload))[
        0
    ]["issue_id"]
    responses.insert(
        story_review_index + 1,
        _tool_call(
            "StoryArtifactRepairPatch",
            {
                "stage": "generating_story_outline",
                "line_replacements": [
                    {
                        "start_line": 2,
                        "end_line": 2,
                        "replacement": "已修复事实。",
                    }
                ],
            },
            201,
        ),
    )
    responses.insert(
        story_review_index + 2,
        _tool_call(
            "CanonReviewerResult",
            {
                "passed": True,
                "evidence": "L4硬规则：修复后故事大纲一致。",
                "issues": [],
                "prior_issue_closures": [
                    {
                        "issue_id": prior_issue_id,
                        "status": "resolved",
                        "evidence": "当前第二行已改为已修复事实，旧冲突不再存在。",
                    }
                ],
            },
            202,
        ),
    )
    approved: dict[InternalStage, dict[str, Any]] = {}
    inlined_project_prompts: list[str] = []
    original_inline_project = _with_inline_project

    def capture_inline_project(prompt: str, persona_files: Mapping[str, str]) -> str:
        inlined = original_inline_project(prompt, persona_files)
        inlined_project_prompts.append(inlined)
        return inlined

    monkeypatch.setattr("pengine.agents._with_inline_project", capture_inline_project)

    async def before_stage(_: InternalStage) -> int:
        return 1

    async def approve_stage(stage: InternalStage, payload: dict[str, Any]) -> None:
        approved[stage] = payload

    async with AsyncSqliteSaver.from_conn_string(str(database)) as saver:
        await saver.setup()
        model = ToolCallingFakeModel(responses=responses)
        project = "# Project\n\nPROJECT-DIRECT-STORY-PATCH"
        workflow = _fake_workflow(
            model=model,
            checkpointer=saver,
            provider_profile_key="toolcallingfakemodel",
        )
        await workflow.execute(
            thread_id="story-project-patch-thread",
            story="故事",
            requirements="要求",
            persona_files={"/persona/project.md": project},
            before_stage=before_stage,
            approve_stage=approve_stage,
            **_episode_hook_kwargs()[0],
        )

    assert approved[InternalStage.GENERATING_STORY_OUTLINE]["content"] == (
        "第一行保留。\n已修复事实。\n第三行保留。\n第四行保留。"
    )
    story_patch_prompts = [
        prompt
        for prompt in inlined_project_prompts
        if "Repair only the unlocked generating_story_outline candidate" in prompt
    ]
    assert len(story_patch_prompts) == 1
    assert story_patch_prompts[0].count(project) == 1


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
            "issues": [
                {
                    "code": "missing_commitment",
                    "message": "必须补齐承诺",
                    "contract_mutation_required": False,
                }
            ],
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
                    "issues": [
                        {
                            "code": "episode_plan",
                            "message": "修复第一集计划",
                            "contract_mutation_required": False,
                        }
                    ],
                }
                if len(reviewed_plans) == 1
                else {
                    "passed": True,
                    "evidence": "L4硬规则：合同一致",
                    "issues": [],
                }
            )
        return ToolMessage(
            content=json.dumps(payload, ensure_ascii=False),
            tool_call_id="call-outline-review-files",
        )

    returned, locked = await middleware._generate_locked_outline(
        request,
        handler,
        request.tool_call["args"],
    )

    assert isinstance(returned, ToolMessage)
    returned_payload = json.loads(returned.content)
    assert returned_payload["episodes"] == [{"episode_number": 1, "plan": "修复后的第一集计划"}]
    assert "contract_review" not in returned_payload
    assert "contract_repair_rounds" not in returned_payload
    assert reviewed_plans == [
        [{"episode_number": 1, "plan": "第一集计划"}],
        [{"episode_number": 1, "plan": "修复后的第一集计划"}],
    ]
    assert locked["contract_review"]["passed"] is True
    assert locked["episodes"][0]["plan"] == "修复后的第一集计划"


@pytest.mark.asyncio
async def test_outline_review_target_mismatch_gets_one_fresh_review_before_repair() -> None:
    contract = _story_contract()
    planner_payload = {
        "stage": "generating_episode_outline",
        "content": "分集大纲",
        "episode_count": 1,
        "episodes": [{"episode_number": 1, "plan": "第一集计划"}],
        "story_contract": contract.model_dump(mode="json"),
    }
    review_calls = 0
    patch_calls = 0

    async def before_stage(_: InternalStage) -> int:
        return 1

    async def approve_stage(_: InternalStage, __: dict[str, Any]) -> None:
        return None

    async def generate_patch(
        _: Mapping[str, Any],
        review: CanonReviewerResult,
        __: int,
        correction: str | None,
    ) -> Any:
        nonlocal patch_calls
        patch_calls += 1
        assert correction is None
        assert review.issues[0].repair_targets[0].expected_value == "不得增加人物"
        return {
            "stage": "generating_episode_outline",
            "content_replacements": [],
            "json_edits": [],
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
            "id": "call-outline-review-target-correction",
            "type": "tool_call",
        },
        tool=None,
        state={"files": {}},
        runtime=None,
    )

    async def handler(candidate_request: ToolCallRequest) -> ToolMessage:
        nonlocal review_calls
        subagent_type = candidate_request.tool_call["args"]["subagent_type"]
        if subagent_type == "episode_planner":
            payload = planner_payload
        else:
            assert subagent_type == "canon_reviewer"
            review_calls += 1
            if review_calls == 1:
                expected_value = "错误旧值"
            elif review_calls == 2:
                expected_value = "不得增加人物"
                assert "/workspace/invalid_contract_review.json" in candidate_request.state["files"]
                assert "could not bind" in candidate_request.tool_call["args"]["description"]
            else:
                payload = {
                    "passed": True,
                    "evidence": "L4硬规则：修复后合同一致",
                    "issues": [],
                }
                return ToolMessage(
                    content=json.dumps(payload, ensure_ascii=False),
                    tool_call_id="call-outline-review-target-correction",
                )
            payload = {
                "passed": False,
                "evidence": "禁止项与正式事实冲突。",
                "issues": [
                    {
                        "code": "conflicting_prohibition",
                        "message": "删除冲突禁止项。",
                        "contract_refs": [],
                        "contract_mutation_required": True,
                        "repair_targets": [
                            {
                                "target_id": "remove_conflicting_prohibition",
                                "collection": "prohibitions",
                                "intent": "remove_existing",
                                "index": 0,
                                "expected_value": expected_value,
                            }
                        ],
                    }
                ],
            }
        return ToolMessage(
            content=json.dumps(payload, ensure_ascii=False),
            tool_call_id="call-outline-review-target-correction",
        )

    returned, locked = await middleware._generate_locked_outline(
        request,
        handler,
        request.tool_call["args"],
    )

    assert review_calls == 3
    assert patch_calls == 1
    assert isinstance(returned, ToolMessage)
    assert json.loads(returned.content)["story_contract"]["prohibitions"] == []
    assert locked["contract_review"]["passed"] is True
    assert locked["contract_repair_rounds"] == 1


@pytest.mark.asyncio
async def test_outline_review_target_mismatch_fails_closed_after_fresh_review() -> None:
    contract = _story_contract()
    planner_payload = {
        "stage": "generating_episode_outline",
        "content": "分集大纲",
        "episode_count": 1,
        "episodes": [{"episode_number": 1, "plan": "第一集计划"}],
        "story_contract": contract.model_dump(mode="json"),
    }
    review_calls = 0
    patch_calls = 0

    async def before_stage(_: InternalStage) -> int:
        return 1

    async def approve_stage(_: InternalStage, __: dict[str, Any]) -> None:
        return None

    async def generate_patch(
        _: Mapping[str, Any],
        __: CanonReviewerResult,
        ___: int,
        ____: str | None,
    ) -> Any:
        nonlocal patch_calls
        patch_calls += 1
        raise AssertionError("repair must not run for an unbound review target")

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
            "id": "call-outline-review-target-fail-closed",
            "type": "tool_call",
        },
        tool=None,
        state={"files": {}},
        runtime=None,
    )

    async def handler(candidate_request: ToolCallRequest) -> ToolMessage:
        nonlocal review_calls
        subagent_type = candidate_request.tool_call["args"]["subagent_type"]
        if subagent_type == "episode_planner":
            payload = planner_payload
        else:
            assert subagent_type == "canon_reviewer"
            review_calls += 1
            if review_calls == 2:
                assert "/workspace/invalid_contract_review.json" in candidate_request.state["files"]
            payload = {
                "passed": False,
                "evidence": "禁止项与正式事实冲突。",
                "issues": [
                    {
                        "code": "conflicting_prohibition",
                        "message": "删除冲突禁止项。",
                        "contract_refs": [],
                        "contract_mutation_required": True,
                        "repair_targets": [
                            {
                                "target_id": "remove_conflicting_prohibition",
                                "collection": "prohibitions",
                                "intent": "remove_existing",
                                "index": 0,
                                "expected_value": f"错误旧值{review_calls}",
                            }
                        ],
                    }
                ],
            }
        return ToolMessage(
            content=json.dumps(payload, ensure_ascii=False),
            tool_call_id="call-outline-review-target-fail-closed",
        )

    with pytest.raises(AgentProtocolError) as error:
        await middleware._generate_locked_outline(
            request,
            handler,
            request.tool_call["args"],
        )

    assert error.value.safe_message == "分集大纲审查目标未能绑定当前合同。"
    assert review_calls == 2
    assert patch_calls == 0


@pytest.mark.asyncio
async def test_episode_review_stops_after_two_repairs_without_commit(tmp_path: Path) -> None:
    database = tmp_path / "checkpoints.sqlite3"
    responses = _successful_responses()
    writer_index = _index_of_tool_call(responses, "ScriptWriterResult", occurrence=1)
    review_index = _index_of_tool_call(responses, "EpisodeReviewerResult", occurrence=1)
    writer_payload = copy.deepcopy(responses[writer_index].tool_calls[0]["args"])
    writer_payload["state_delta"]["evidence"] = [
        item
        for item in writer_payload["state_delta"]["evidence"]
        if item["target_id"] != "fact_ep1"
    ]
    responses[writer_index] = _tool_call("ScriptWriterResult", writer_payload, writer_index)
    failed_review = {
        "passed": False,
        "evidence": "人物身份与上游小传不一致",
        "issues": [
            {
                "code": "identity_drift",
                "message": "剧本把母亲姓名改成了合同外角色",
                "contract_refs": ["semantic_target"],
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
        model = ToolCallingFakeModel(responses=responses)
        workflow = _fake_workflow(
            model=model,
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
    assert "missing_evidence_targets" in error.value.evidence
    assert "目标：fact_ep1" in error.value.evidence
    assert "审查目标：fact_ep1, semantic_target" in error.value.evidence
    assert episode_attempts == [1]
    assert InternalStage.GENERATING_EPISODE_SCRIPTS not in approved
    repair_requests = [
        (set(tool_names), system_prompt)
        for tool_names, system_prompt in zip(
            model.bound_tool_names,
            model.model_system_prompts,
            strict=True,
        )
        if "/skills/continuity-repair/SKILL.md" in system_prompt
    ]
    assert len(repair_requests) == 2
    for tool_names, system_prompt in repair_requests:
        assert tool_names == {"read_file", "calculate_arithmetic", "ScriptWriterResult"}
        for hidden_tool in ("ls", "glob", "grep", "write_todos", "write_file", "edit_file"):
            assert not _prompt_mentions_tool(system_prompt, hidden_tool)


@pytest.mark.asyncio
async def test_episode_repair_receives_deterministic_and_semantic_issues_together(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database = tmp_path / "checkpoints.sqlite3"
    contract = _story_contract(verbatim_episodes={1})
    responses = _successful_responses(contract=contract)
    writer_index = _index_of_tool_call(responses, "ScriptWriterResult", occurrence=1)
    review_index = _index_of_tool_call(responses, "EpisodeReviewerResult", occurrence=1)
    repaired_writer_payload = copy.deepcopy(responses[writer_index].tool_calls[0]["args"])
    invalid_writer_payload = copy.deepcopy(repaired_writer_payload)
    invalid_writer_payload["content"] = "钩子1"
    invalid_writer_payload["state_delta"]["evidence"] = [
        item
        for item in invalid_writer_payload["state_delta"]["evidence"]
        if item["target_id"] != "fact_ep1"
    ]
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
                },
                {
                    "code": "unknown_speaker",
                    "message": "剧本引入了锁定角色表之外的说话人 年轻协办",
                    "script_excerpt": "年轻协办：我来开车。",
                },
                {
                    "code": "evidence_coverage_mismatch",
                    "message": "剧本证据未覆盖全部必需事实、线索和分集义务",
                    "contract_refs": ["fact_ep1", "obligation_ep1"],
                },
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
    captured_descriptions: list[str] = []
    original_repair = StageGuardMiddleware._invoke_repair_subagent

    async def capture_repair(
        middleware: StageGuardMiddleware,
        **kwargs: Any,
    ) -> tuple[Any, dict[str, Any]]:
        captured_files.append(kwargs["files"])
        captured_descriptions.append(kwargs["description"])
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
            suffix_rewrite_feedback={
                "version": 1,
                "effective_earliest_affected_episode": 1,
                "reviews": [
                    {
                        "review_id": "suffix-review",
                        "category": "script_defect",
                        "evidence": "重写原因：校牌事实冲突。",
                        "earliest_affected_episode": 1,
                        "binding": {"review_epoch": 1, "batch_id": "batch-1"},
                    }
                ],
            },
            **_episode_hook_kwargs()[0],
        )

    review = json.loads(captured_files[0]["/workspace/episode_review.json"])
    assert {issue["code"] for issue in review["issues"]} >= {
        "missing_evidence_targets",
        "identity_drift",
    }
    missing = next(
        issue for issue in review["issues"] if issue["code"] == "missing_evidence_targets"
    )
    assert missing["contract_refs"] == ["fact_ep1"]
    assert "exactly one state_delta.evidence entry" in captured_descriptions[0]
    assert "remove every unexpected target" in captured_descriptions[0]
    assert "verbatim" in captured_descriptions[0]
    assert captured_files[0]["/workspace/current_episode_plan.md"] == "第一集计划"
    obligation = json.loads(captured_files[0]["/workspace/current_episode_obligation.json"])
    assert obligation["end_hook"] == "钩子1"
    assert "/workspace/speaker_contract.json" not in captured_files[0]
    evidence_contract = json.loads(captured_files[0]["/workspace/evidence_contract.json"])
    assert evidence_contract["episode_number"] == 1
    assert evidence_contract["phase"] == "episode_repair"
    assert evidence_contract["required_evidence_target_ids"] == [
        "fact_ep1",
        "obligation_ep1",
    ]
    assert evidence_contract["required_verbatim_facts"] == [
        {"fact_id": "fact_ep1", "value": "事实1"}
    ]
    evidence_issues = evidence_contract["rejected_issues"]
    assert {issue["code"] for issue in evidence_issues} == {
        "evidence_coverage_mismatch",
        "verbatim_fact_missing",
    }
    coverage_issue = next(
        issue for issue in evidence_issues if issue["code"] == "evidence_coverage_mismatch"
    )
    assert coverage_issue["contract_refs"] == ["fact_ep1", "obligation_ep1"]
    assert coverage_issue["message"] == "剧本证据未覆盖全部必需事实、线索和分集义务"
    verbatim_issue = next(
        issue for issue in evidence_issues if issue["code"] == "verbatim_fact_missing"
    )
    assert verbatim_issue["contract_refs"] == ["fact_ep1"]
    assert "事实1" in verbatim_issue["message"]
    suffix_review = json.loads(captured_files[0]["/workspace/suffix_rewrite_review.json"])
    assert suffix_review["reviews"][0]["evidence"] == "重写原因：校牌事实冲突。"
    assert len(captured_descriptions) == 1
    repair_description = captured_descriptions[0]
    assert "suffix_rewrite_review.json" in repair_description
    assert "fix every named conflict" in repair_description
    assert "unknown_speaker" in repair_description
    assert "genuinely new continuity-bearing character" in repair_description
    assert "preserving surface notation" in repair_description
    assert "Do not rewrite a label merely because it is an alias" in repair_description
    assert "generic or descriptive label" in repair_description
    assert "speaker_contract.json" not in repair_description
    assert 'issue.contract_refs: ["fact_ep1", "obligation_ep1"]' in repair_description
    assert "exact set" in repair_description
    assert "no extras" in repair_description
    assert "no duplicates" in repair_description
    assert "every required target exactly once" in repair_description
    assert "Verbatim fact repair is mandatory" in repair_description
    assert "fact.value" in repair_description
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
            _assert_task_lists_exact_workspace_paths(subagent_request)
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
            _assert_task_lists_exact_workspace_paths(subagent_request)
            assert "trusted runtime metadata" in description
            assert "episodes[].content" in description
            series_review_inputs.append(dict(subagent_request.state["files"]))
            payload = {
                "passed": True,
                "category": "pass",
                "evidence": "L4硬规则：全系列一致",
            }
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
            "evidence": "L4硬规则：全系列一致",
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


def test_review_language_retry_cannot_change_contract_repair_authority() -> None:
    original = CanonReviewerResult(
        passed=False,
        evidence="The prohibition conflicts with locked canon.",
        issues=[
            {
                "code": "conflicting_prohibition",
                "message": "Remove the stale prohibition.",
                "contract_refs": [],
                "script_excerpt": "旧禁止项",
                "contract_mutation_required": True,
                "repair_targets": [
                    {
                        "target_id": "remove_stale_prohibition",
                        "collection": "prohibitions",
                        "intent": "remove_existing",
                        "index": 1,
                        "expected_value": "旧禁止项",
                    }
                ],
            }
        ],
    )
    translated_payload = original.model_dump(mode="json")
    translated_payload["evidence"] = "该禁止项与锁定事实冲突。"
    translated_payload["issues"][0]["message"] = "删除陈旧禁止项。"
    translated = CanonReviewerResult.model_validate(translated_payload)
    changed_payload = translated.model_dump(mode="json")
    changed_payload["issues"][0]["repair_targets"][0]["index"] = 0
    changed = CanonReviewerResult.model_validate(changed_payload)

    original_fingerprint = _language_retry_fingerprint(original)
    assert _language_retry_matches(original_fingerprint, _language_retry_fingerprint(translated))
    assert not _language_retry_matches(
        original_fingerprint,
        _language_retry_fingerprint(changed),
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
        _assert_task_lists_exact_workspace_paths(review_request)
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
        state={
            "files": {
                "/workspace/creation-request.md": {
                    "content": "故事要求",
                    "encoding": "utf-8",
                }
            }
        },
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
                            "contract_mutation_required": False,
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
        files={"/workspace/candidate.md": "候选内容"},
        schema=CanonReviewerResult,
        stage=InternalStage.GENERATING_EPISODE_OUTLINE,
    )

    assert len(descriptions) == 2
    assert _required_read_paths([HumanMessage(content=descriptions[0])]) == (
        "/workspace/approved-checkpoints.json",
        "/workspace/candidate.md",
        "/workspace/creation-request.md",
    )
    assert _required_read_paths([HumanMessage(content=descriptions[1])]) == (
        "/workspace/review_to_translate.json",
    )
    assert "without changing the review decision" in descriptions[1]
    assert "do not perform a new review" in descriptions[1]
    assert review.passed is False
    assert review.evidence == "合同遗漏了既定事实。"


@pytest.mark.asyncio
async def test_semantic_review_allows_chinese_issue_with_virtual_paths_and_stable_ids() -> None:
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
                    "evidence": (
                        "已审阅 /workspace/current_character_biographies.md、"
                        "/workspace/current_relationship_logic.md、"
                        "/workspace/previous_character_biographies.md、"
                        "/workspace/previous_relationship_logic.md、"
                        "/workspace/current_story_review.json 和 "
                        "/workspace/previous_story_review.json；本集仍有连续性问题。"
                    ),
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
                        "contract_mutation_required": False,
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
        issues=[
            {
                "code": "stale_text",
                "message": "替换旧文字。",
                "contract_mutation_required": False,
            }
        ],
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
async def test_outline_repair_correction_reports_a_safe_target_error() -> None:
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
        evidence="只需修复可读大纲。",
        issues=[
            {
                "code": "stale_text",
                "message": "替换旧文字。",
                "contract_mutation_required": False,
            }
        ],
    )

    async def generate_patch(
        _: Mapping[str, Any],
        __: CanonReviewerResult,
        ___: int,
        correction: str | None,
    ) -> Any:
        corrections.append(correction)
        if correction is None:
            return {
                "stage": "generating_episode_outline",
                "json_edits": [
                    {
                        "op": "replace",
                        "path": "/story_contract/prohibitions/0",
                        "expected": "不得增加人物",
                        "value": "替换禁止项",
                    }
                ],
            }
        return {
            "stage": "generating_episode_outline",
            "content_replacements": [{"old": "旧分集大纲", "new": "修复后的分集大纲"}],
        }

    middleware = StageGuardMiddleware(
        before_stage=before_stage,
        approve_stage=approve_stage,
        approved_stages=set(),
        generate_outline_patch=generate_patch,
    )

    repaired = await middleware._invoke_outline_repair(
        candidate=candidate,
        review=review,
        repair_round=1,
    )

    assert len(corrections) == 2
    assert "Outline repair paths must target episode plans" in (corrections[1] or "")
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
        issues=[
            {
                "code": "stale_text",
                "message": "替换旧文字。",
                "contract_mutation_required": False,
            }
        ],
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
        issues=[
            {
                "code": "stale_text",
                "message": "替换旧文字。",
                "contract_mutation_required": False,
            }
        ],
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
        _assert_task_lists_exact_workspace_paths(repair_request)
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
        files={"/workspace/story_contract.json": "{}"},
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
        _assert_task_lists_exact_workspace_paths(repair_request)
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
        files={"/workspace/candidate_episode.md": "事实1\n钩子1"},
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

    model_call_store = ModelCallStore(settings.database_path)
    model_call_state = ModelCallState(store=model_call_store)
    async with AsyncSqliteSaver.from_conn_string(str(settings.database_path)) as saver:
        await saver.setup()
        resumed_workflow = _fake_workflow(
            model=ToolCallingFakeModel(responses=responses[2:]),
            checkpointer=saver,
            provider_profile_key="toolcallingfakemodel",
            model_call_state=model_call_state,
        )
        restarted_worker = Worker(
            settings=settings,
            repository=repository,
            catalog=catalog,
            workflow=resumed_workflow,
            worker_id="restarted-worker",
        )
        restarted_worker._model_call_state = model_call_state

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
    model_call_store.close()


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
