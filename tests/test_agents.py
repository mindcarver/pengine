import copy
import hashlib
import json
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from langchain.agents.middleware.types import ToolCallRequest
from langchain.agents.structured_output import StructuredOutputValidationError
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, ToolMessage
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
    OutlineRepairPatch,
    QualityGateRejectedError,
    QualityReviewerResult,
    ScriptWriterResult,
    StageGuardMiddleware,
    StoryArchitectResult,
    WorkflowCompletion,
    _apply_outline_repair_patch,
    _arithmetic_tool,
    _calculate_arithmetic,
    _language_retry_fingerprint,
    _language_retry_matches,
    _structured_output_retry_message,
    _supervisor_prompt,
    _validate_result_language,
)
from pengine.config import Settings
from pengine.continuity import (
    StoryContract,
    render_story_contract_markdown,
    story_contract_sha256,
)
from pengine.language import SIMPLIFIED_CHINESE, language_instruction
from pengine.personas import PersonaCatalog
from pengine.repository import Repository
from pengine.schemas import CreateCreationRequest, EpisodeDraft, EpisodePlan, InternalStage
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
                "selected_l0_variant": "主动选择",
                "selection_rationale": "契合故事",
                "content": None,
            },
        ),
        (
            "generating_story_outline",
            "story_architect",
            "StoryArchitectResult",
            {
                "stage": "generating_story_outline",
                "content": "故事大纲",
                "selected_l0_variant": None,
                "selection_rationale": None,
            },
        ),
        (
            "generating_character_biographies",
            "story_architect",
            "StoryArchitectResult",
            {
                "stage": "generating_character_biographies",
                "content": "人物小传",
                "selected_l0_variant": None,
                "selection_rationale": None,
            },
        ),
        (
            "generating_relationship_logic",
            "story_architect",
            "StoryArchitectResult",
            {
                "stage": "generating_relationship_logic",
                "content": "关系逻辑",
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

    assert "selecting_l0_variant" in properties["content"]["description"]
    assert "Must be null" in properties["content"]["description"]
    assert "selecting_l0_variant" in properties["selected_l0_variant"]["description"]
    assert "Must be null" in properties["selected_l0_variant"]["description"]
    assert "selecting_l0_variant" in properties["selection_rationale"]["description"]
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
                "stage": "generating_story_outline",
                "content": None,
            }
        )
    except ValidationError as error:
        message = _structured_output_retry_message(error)
    else:
        raise AssertionError("The invalid story result unexpectedly validated")

    assert "Correct these validation errors" in message
    assert "Story artifact stages require only content" in message
    assert "input_value" not in message


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
    assert "episode-specific action" in _EPISODE_PLANNER_PROMPT
    assert "dates, countdowns, amounts, counts, and arithmetic" in _EPISODE_PLANNER_PROMPT
    assert "/workspace/approved-checkpoints.json" in _EPISODE_PLANNER_PROMPT
    assert "new_information_fact_ids must exactly equal" in _EPISODE_PLANNER_PROMPT
    assert "free-form user request" in _EPISODE_PLANNER_PROMPT
    assert "Never ask the user" in _EPISODE_PLANNER_PROMPT
    assert "Leave genuinely unspecified details out" in _EPISODE_PLANNER_PROMPT
    assert "Knowledge states are sparse cumulative snapshots" in _EPISODE_PLANNER_PROMPT
    assert "aliases, pronouns, ages, elapsed durations, call participants" in (
        _EPISODE_PLANNER_PROMPT
    )
    assert "exact dialogue-count claims" in _SCRIPT_WRITER_PROMPT
    assert "Every upstream commitment must appear" in _SCRIPT_WRITER_PROMPT
    assert "calculate_arithmetic" in _SCRIPT_WRITER_PROMPT
    assert "canonical contract names in every speaker label" in _SCRIPT_WRITER_PROMPT
    assert "call participants" in _SCRIPT_WRITER_PROMPT
    assert "grandfathered pre-contract run" in _SCRIPT_WRITER_PROMPT
    assert "return null state_delta" in _SCRIPT_WRITER_PROMPT


def test_specialist_skills_are_packaged_and_not_assigned_to_stage_owners() -> None:
    assert _SPECIALIST_SKILL_SOURCES == {
        "canon_reviewer": ["/skills/canon-review"],
        "episode_reviewer": ["/skills/episode-continuity-review"],
        "episode_repair": ["/skills/continuity-repair"],
    }
    assert set(load_agent_skill_files()) == {
        "/skills/canon-review/SKILL.md",
        "/skills/episode-continuity-review/SKILL.md",
        "/skills/continuity-repair/SKILL.md",
    }
    assert not {"story_architect", "episode_planner", "script_writer", "quality_reviewer"} & set(
        _SPECIALIST_SKILL_SOURCES
    )
    skill_files = load_agent_skill_files()
    assert "minimum continuity ledger" in skill_files["/skills/canon-review/SKILL.md"]
    assert (
        "complete committed series prefix"
        in skill_files["/skills/episode-continuity-review/SKILL.md"]
    )


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
        workflow = DeepAgentWorkflow(
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
            "generating_character_biographies",
            "generating_relationship_logic",
            "generating_episode_outline",
            "accepting_l0",
            "accepting_l4",
        ]
        assert [stage for kind, stage in events if kind == "approve"] == [
            "selecting_l0_variant",
            "generating_story_outline",
            "generating_character_biographies",
            "generating_relationship_logic",
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
async def test_contract_review_repairs_once_before_outline_lock(tmp_path: Path) -> None:
    database = tmp_path / "checkpoints.sqlite3"
    responses = _successful_responses()
    responses[10] = _tool_call(
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
        10,
    )
    responses.insert(
        11,
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
        12,
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
        workflow = DeepAgentWorkflow(
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
    responses[10] = _tool_call(
        "CanonReviewerResult",
        {
            "passed": False,
            "evidence": "合同遗漏一项上游承诺",
            "issues": [{"code": "missing_commitment", "message": "必须补齐承诺"}],
        },
        10,
    )
    invalid_patch = {
        "stage": "generating_episode_outline",
        "content_replacements": [],
        "json_edits": [],
    }
    responses.insert(11, _tool_call("OutlineRepairPatch", invalid_patch, 101))
    responses.insert(12, _tool_call("OutlineRepairPatch", invalid_patch, 102))
    responses.insert(
        13,
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
        workflow = DeepAgentWorkflow(
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
    writer_payload = responses[12].tool_calls[0]["args"]
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
    responses[13] = _tool_call("EpisodeReviewerResult", failed_review, 13)
    responses.insert(14, _tool_call("ScriptWriterResult", writer_payload, 201))
    responses.insert(15, _tool_call("EpisodeReviewerResult", failed_review, 202))
    responses.insert(16, _tool_call("ScriptWriterResult", writer_payload, 203))
    responses.insert(17, _tool_call("EpisodeReviewerResult", failed_review, 204))
    approved: list[InternalStage] = []
    episode_hooks, episode_attempts = _episode_hook_kwargs()

    async def before_stage(_: InternalStage) -> int:
        return 1

    async def approve_stage(stage: InternalStage, _: dict[str, Any]) -> None:
        approved.append(stage)

    async with AsyncSqliteSaver.from_conn_string(str(database)) as saver:
        await saver.setup()
        workflow = DeepAgentWorkflow(
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
        },
        InternalStage.GENERATING_CHARACTER_BIOGRAPHIES: {
            "stage": "generating_character_biographies",
            "content": "人物小传",
        },
        InternalStage.GENERATING_RELATIONSHIP_LOGIC: {
            "stage": "generating_relationship_logic",
            "content": "关系逻辑",
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
            for required_check in (
                "identities",
                "relationships",
                "pronouns",
                "call participants",
                "clue meanings",
            ):
                assert required_check in description
            reviewed_prefixes.append(set(subagent_request.state["files"]))
            prefix_file = subagent_request.state["files"]["/workspace/series_prefix.md"]
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
        "/workspace/series_prefix.md",
    } <= reviewed_prefixes[1]
    assert "第 1 集\n事实1" in reviewed_prefix_contents[1]
    assert "第 2 集\n事实2" in reviewed_prefix_contents[1]
    assert InternalStage.GENERATING_EPISODE_SCRIPTS in approvals


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
                "content": "invalid for the selection stage",
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
        workflow = DeepAgentWorkflow(
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
        workflow = DeepAgentWorkflow(
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
        workflow = DeepAgentWorkflow(
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


def test_language_retry_fingerprint_locks_timeline_and_contract_version() -> None:
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

    original_fingerprint = _language_retry_fingerprint(original)
    assert not _language_retry_matches(
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

    assert _language_retry_matches(
        _language_retry_fingerprint(english_label),
        _language_retry_fingerprint(translated_label),
    )
    assert not _language_retry_matches(
        _language_retry_fingerprint(machine_variant),
        _language_retry_fingerprint(changed_machine_variant),
    )


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

    async def before_stage(_: InternalStage) -> int:
        return 1

    async def approve_stage(_: InternalStage, __: dict[str, Any]) -> None:
        raise AssertionError("A changed gate decision must not be approved")

    middleware = StageGuardMiddleware(
        before_stage=before_stage,
        approve_stage=approve_stage,
        approved_stages={
            InternalStage.SELECTING_L0_VARIANT,
            InternalStage.GENERATING_STORY_OUTLINE,
            InternalStage.GENERATING_CHARACTER_BIOGRAPHIES,
            InternalStage.GENERATING_RELATIONSHIP_LOGIC,
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

    with pytest.raises(AgentProtocolError, match="changed non-language fields"):
        await middleware.awrap_tool_call(request, handler)

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
    assert review.passed is False
    assert review.evidence == "合同遗漏了既定事实。"


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
async def test_repair_subagent_gets_one_bounded_language_retry() -> None:
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
        payload = json.loads(json.dumps(contract_payload))
        payload["characters"][0]["role"] = (
            "调查者" if len(descriptions) == 2 else "Investigator of a buried family secret"
        )
        return ToolMessage(
            content=json.dumps(
                {
                    "stage": "generating_episode_outline",
                    "content": "完整分集大纲。",
                    "episode_count": 1,
                    "episodes": [{"episode_number": 1, "plan": "调查者发现证据。"}],
                    "story_contract": payload,
                },
                ensure_ascii=False,
            ),
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
    assert payload["story_contract"]["characters"][0]["role"] == "调查者"


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
            InternalStage.GENERATING_CHARACTER_BIOGRAPHIES,
            InternalStage.GENERATING_RELATIONSHIP_LOGIC,
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
async def test_legacy_episode_recovery_skips_committed_draft_without_requiring_new_contract() -> (
    None
):
    first_content = "第一集不可替换剧本"
    first = EpisodeDraft(
        episode_number=1,
        content=first_content,
        content_sha256=hashlib.sha256(first_content.encode()).hexdigest(),
        completed_at=datetime(2026, 8, 1, tzinfo=UTC),
    )
    approved_payloads = {
        InternalStage.GENERATING_EPISODE_OUTLINE: {
            "stage": "generating_episode_outline",
            "content": "两集旧版分集大纲",
            "episode_count": 2,
            "episodes": [
                {"episode_number": 1, "plan": "第一集计划"},
                {"episode_number": 2, "plan": "第二集计划"},
            ],
        }
    }
    attempts: list[int] = []
    commits: list[tuple[int, str, object | None]] = []
    approved: dict[InternalStage, dict[str, Any]] = {}

    async def before_stage(_: InternalStage) -> int:
        return 1

    async def approve_stage(stage: InternalStage, payload: dict[str, Any]) -> None:
        approved[stage] = payload

    async def before_episode(plan: EpisodePlan) -> int:
        attempts.append(plan.episode_number)
        return 2

    async def commit_episode(
        episode_number: int,
        content: str,
        episode_lock=None,
    ) -> EpisodeDraft:
        commits.append((episode_number, content, episode_lock))
        return EpisodeDraft(
            episode_number=episode_number,
            content=content,
            content_sha256=hashlib.sha256(content.encode()).hexdigest(),
            completed_at=datetime(2026, 8, 1, tzinfo=UTC),
        )

    async def assemble_episode_scripts() -> str:
        return "第 1 集\n第一集不可替换剧本\n\n---\n\n第 2 集\n第二集恢复剧本"

    middleware = StageGuardMiddleware(
        before_stage=before_stage,
        approve_stage=approve_stage,
        approved_stages={
            InternalStage.SELECTING_L0_VARIANT,
            InternalStage.GENERATING_STORY_OUTLINE,
            InternalStage.GENERATING_CHARACTER_BIOGRAPHIES,
            InternalStage.GENERATING_RELATIONSHIP_LOGIC,
            InternalStage.GENERATING_EPISODE_OUTLINE,
        },
        approved_payloads=approved_payloads,
        episode_drafts=[first],
        before_episode=before_episode,
        commit_episode=commit_episode,
        assemble_episode_scripts=assemble_episode_scripts,
    )
    request = ToolCallRequest(
        tool_call={
            "name": "task",
            "args": {
                "description": "[stage=generating_episode_scripts] write scripts",
                "subagent_type": "script_writer",
            },
            "id": "call-legacy-episode",
            "type": "tool_call",
        },
        tool=None,
        state={"files": {}},
        runtime=None,
    )

    async def handler(episode_request: ToolCallRequest) -> ToolMessage:
        assert "[episode=2]" in episode_request.tool_call["args"]["description"]
        assert "/workspace/episodes/ep1.md" in episode_request.state["files"]
        assert "/workspace/story_contract.json" not in episode_request.state["files"]
        return ToolMessage(
            content=(
                '{"stage":"generating_episode_scripts","episode_number":2,'
                '"content":"第二集恢复剧本"}'
            ),
            tool_call_id="call-legacy-episode",
        )

    await middleware.awrap_tool_call(request, handler)

    assert attempts == [2]
    assert commits == [(2, "第二集恢复剧本", None)]
    assert approved[InternalStage.GENERATING_EPISODE_SCRIPTS] == {
        "stage": "generating_episode_scripts",
        "content": "第 1 集\n第一集不可替换剧本\n\n---\n\n第 2 集\n第二集恢复剧本",
    }


@pytest.mark.asyncio
async def test_quality_review_drops_stale_script_when_canonical_payload_is_missing() -> None:
    approved_stages = {
        InternalStage.SELECTING_L0_VARIANT,
        InternalStage.GENERATING_STORY_OUTLINE,
        InternalStage.GENERATING_CHARACTER_BIOGRAPHIES,
        InternalStage.GENERATING_RELATIONSHIP_LOGIC,
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
        workflow = DeepAgentWorkflow(
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
    responses[15] = _tool_call(
        "QualityReviewerResult",
        {
            "stage": "accepting_l0",
            "passed": False,
            "evidence": "成品没有通过 L0 闸门。",
            "feedback_handling": [],
        },
        15,
    )
    approved: list[InternalStage] = []

    async def before_stage(_: InternalStage) -> int:
        return 1

    async def approve_stage(stage: InternalStage, _: dict[str, Any]) -> None:
        approved.append(stage)

    async with AsyncSqliteSaver.from_conn_string(str(database)) as saver:
        await saver.setup()
        workflow = DeepAgentWorkflow(
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
        InternalStage.GENERATING_CHARACTER_BIOGRAPHIES,
        InternalStage.GENERATING_RELATIONSHIP_LOGIC,
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
    responses[17] = _tool_call(
        "QualityReviewerResult",
        {
            "stage": "accepting_l4",
            "passed": False,
            "evidence": "成品没有通过 L4 闸门。",
            "feedback_handling": [],
        },
        17,
    )
    responses.insert(
        17,
        _tool_call(
            "read_file",
            {"file_path": "/workspace/episode_scripts.md"},
            100,
        ),
    )
    responses.insert(
        18,
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
        workflow = DeepAgentWorkflow(
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
        resumed_responses = _successful_responses()[16:]
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
        resumed_workflow = DeepAgentWorkflow(
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
        InternalStage.GENERATING_CHARACTER_BIOGRAPHIES,
        InternalStage.GENERATING_RELATIONSHIP_LOGIC,
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
        workflow = DeepAgentWorkflow(
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
        InternalStage.GENERATING_CHARACTER_BIOGRAPHIES,
        InternalStage.GENERATING_RELATIONSHIP_LOGIC,
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
        workflow = DeepAgentWorkflow(
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
        resumed_workflow = DeepAgentWorkflow(
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
        InternalStage.GENERATING_CHARACTER_BIOGRAPHIES,
        InternalStage.GENERATING_RELATIONSHIP_LOGIC,
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

    responses = _successful_responses()

    async def before_stage(stage: InternalStage) -> int:
        return await repository.record_stage_attempt(lease.run_id, stage)

    async def approve_stage(stage: InternalStage, payload: dict[str, Any]) -> None:
        await repository.approve_checkpoint(lease.run_id, stage, payload)

    async with AsyncSqliteSaver.from_conn_string(str(settings.database_path)) as saver:
        await saver.setup()
        interrupted_workflow = DeepAgentWorkflow(
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
        resumed_workflow = DeepAgentWorkflow(
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
            workflow=DeepAgentWorkflow(
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
