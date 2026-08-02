import asyncio
import json
import re
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, replace
from decimal import Decimal, DecimalException
from fractions import Fraction
from typing import Any, Literal

from deepagents import (
    GeneralPurposeSubagentProfile,
    HarnessProfile,
    create_deep_agent,
    register_harness_profile,
)
from deepagents.backends import StateBackend
from deepagents.middleware.filesystem import FilesystemPermission
from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import ToolCallRequest
from langchain.agents.structured_output import ToolStrategy
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import ToolMessage
from langchain_core.tools import StructuredTool, ToolException
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.types import Command
from pydantic import Field, model_validator

from pengine.continuity import (
    ContinuityViolation,
    EpisodeLock,
    EpisodeStateDelta,
    SemanticReview,
    StoryContract,
    build_episode_lock,
    canonical_model_hash,
    initial_series_state,
    render_story_contract_markdown,
    story_contract_sha256,
    validate_episode_candidate,
)
from pengine.schemas import (
    EpisodeDraft,
    EpisodePlan,
    FeedbackHandlingItem,
    InternalStage,
    NonEmptyText,
    StrictModel,
    WorkflowResult,
)
from pengine.skill_assets import load_agent_skill_files

StageHook = Callable[[InternalStage], Awaitable[int]]
CheckpointHook = Callable[[InternalStage, Mapping[str, Any]], Awaitable[None]]
ReferenceRetriever = Callable[[str], Awaitable[str]]
EpisodeAttemptHook = Callable[[EpisodePlan], Awaitable[int]]
EpisodeCommitHook = Callable[[int, str, EpisodeLock | None], Awaitable[EpisodeDraft]]
EpisodeAssemblyHook = Callable[[], Awaitable[str]]
EpisodeDeadlineReset = Callable[[], Awaitable[None]]

_STAGE_TOKEN = re.compile(r"^\[stage=([a-z0-9_]+)\](?:\[episode=\d+\])?(?:\s|$)")
_REGISTERED_PROFILE_KEYS: set[str] = set()
_SPECIALIST_SKILL_SOURCES = {
    "canon_reviewer": ["/skills/canon-review"],
    "canon_repair": ["/skills/continuity-repair"],
    "episode_reviewer": ["/skills/episode-continuity-review"],
    "episode_repair": ["/skills/continuity-repair"],
}

_STORY_STAGES = (
    InternalStage.SELECTING_L0_VARIANT,
    InternalStage.GENERATING_STORY_OUTLINE,
    InternalStage.GENERATING_CHARACTER_BIOGRAPHIES,
    InternalStage.GENERATING_RELATIONSHIP_LOGIC,
)
_TASK_OWNER = {
    InternalStage.SELECTING_L0_VARIANT: "story_architect",
    InternalStage.GENERATING_STORY_OUTLINE: "story_architect",
    InternalStage.GENERATING_CHARACTER_BIOGRAPHIES: "story_architect",
    InternalStage.GENERATING_RELATIONSHIP_LOGIC: "story_architect",
    InternalStage.GENERATING_EPISODE_OUTLINE: "episode_planner",
    InternalStage.GENERATING_EPISODE_SCRIPTS: "script_writer",
    InternalStage.ACCEPTING_L0: "quality_reviewer",
    InternalStage.ACCEPTING_L4: "quality_reviewer",
}
_ORDERED_SPECIALIST_STAGES = tuple(_TASK_OWNER)
_RESULT_TOOL = {
    InternalStage.SELECTING_L0_VARIANT: "StoryArchitectResult",
    InternalStage.GENERATING_STORY_OUTLINE: "StoryArchitectResult",
    InternalStage.GENERATING_CHARACTER_BIOGRAPHIES: "StoryArchitectResult",
    InternalStage.GENERATING_RELATIONSHIP_LOGIC: "StoryArchitectResult",
    InternalStage.GENERATING_EPISODE_OUTLINE: "EpisodePlannerResult",
    InternalStage.GENERATING_EPISODE_SCRIPTS: "ScriptWriterResult",
    InternalStage.ACCEPTING_L0: "QualityReviewerResult",
    InternalStage.ACCEPTING_L4: "QualityReviewerResult",
}
_WORKSPACE_ARTIFACT_PATHS = {
    InternalStage.GENERATING_STORY_OUTLINE: "/workspace/story_outline.md",
    InternalStage.GENERATING_CHARACTER_BIOGRAPHIES: "/workspace/character_biographies.md",
    InternalStage.GENERATING_RELATIONSHIP_LOGIC: "/workspace/relationship_logic.md",
    InternalStage.GENERATING_EPISODE_OUTLINE: "/workspace/episode_outline.md",
    InternalStage.GENERATING_EPISODE_SCRIPTS: "/workspace/episode_scripts.md",
}

_STORY_ARCHITECT_PROMPT = (
    "Read the relevant /persona context. Return only the structured result for "
    "the stage named in the task. For selecting_l0_variant, set "
    "selected_l0_variant and selection_rationale, and leave content null. For "
    "each generation stage, set content and leave both L0 selection fields null. "
    "Treat every prior approved artifact as binding. Reconcile dates, amounts, "
    "counts, and episode-specific actions before returning. Avoid unnecessary "
    "exact claims about future dialogue counts or scene placement; when such a "
    "claim is required, make it an explicit downstream commitment. Use "
    "calculate_arithmetic for every derived numeric claim and copy its exact result."
)

_EPISODE_PLANNER_PROMPT = (
    "Read /workspace/creation-request.md, /workspace/approved-checkpoints.json, "
    "/workspace/story_outline.md, /workspace/character_biographies.md, and "
    "/workspace/relationship_logic.md, then apply the persona rules. Return only the "
    "structured episode-outline result. Preserve every explicit numeric "
    "constraint from the script requirements. When the requirements do not "
    "specify an episode count, read the stage-specific persona L4 file and use "
    "its baseline; never invent a different count. Before returning, verify "
    "that every episode-specific action promised by the character biographies "
    "or relationship logic appears in the matching episode, and that dates, "
    "countdowns, amounts, counts, and arithmetic agree across artifacts. Use "
    "calculate_arithmetic for every derived numeric claim. Never round a "
    "non-integral division unless the story states the rounding rule. Include a "
    "contiguous episode list beginning at 1, with one concrete plan for every "
    "episode, while preserving the readable full outline in content. The free-form user request "
    "is the only required creative input: automatically compile the minimum continuity ledger "
    "from facts established by the approved artifacts. Never ask the user for character sheets, "
    "timelines, or evidence tables. Leave genuinely unspecified details out instead of inventing "
    "them only for validation. Capture established aliases, pronouns, ages, elapsed durations, "
    "call participants, identity and relationship facts, and canonical clue meanings as typed "
    "facts or existing structured contract fields. Compile the same approved facts into "
    "story_contract. Use unique lowercase snake_case IDs "
    "and a closed cast; every relationship, timeline participant, and knowledge entry "
    "must reference that cast. Use ISO dates/times. Numeric facts require exact decimal "
    "values and explicit units, and the same numeric value cannot mean different kinds "
    "or units. Timeline order must be contiguous from 1. Include exactly one knowledge "
    "state for every character in every episode, and never remove known facts. Every "
    "clue must first be visible or audible, with introduction no later than explanation "
    "or callback. Include exactly one obligation and hook per episode; its "
    "new_information_fact_ids must exactly equal all facts whose first_revealed_episode "
    "is that episode."
)

_SCRIPT_WRITER_PROMPT = (
    "If the task explicitly identifies a grandfathered pre-contract run, the contract "
    "and series-state files are absent by design: do not read or invent them, preserve "
    "every approved upstream artifact and committed earlier episode, and return null "
    "state_delta. Otherwise, read /workspace/story_contract.json and "
    "/workspace/series_state.json and return a state_delta bound to the supplied contract "
    "hash. Use the requested single episode plan and persona rules without changing any "
    "locked episode count, cast, facts, units, timeline, knowledge states, or clue plan. Before "
    "returning, reread every approved upstream artifact and audit this episode "
    "against them. Use canonical contract names in every speaker label, including when a "
    "parenthetical delivery direction follows the name. Treat established aliases, pronouns, "
    "ages, elapsed durations, call participants, identity and relationship facts, and clue "
    "meanings as binding. "
    "Correct contradictions in dates or countdowns, amounts or arithmetic, "
    "exact dialogue-count claims, and episode-specific promised actions. Every "
    "upstream commitment must appear in the scripts. Use calculate_arithmetic "
    "for every derived numeric claim and copy its exact result. The tool accepts "
    "decimal operands only: convert clock times to elapsed minutes before subtracting "
    "them (for example, 22:50 becomes 1370 and 22:20 becomes 1340). Never round a "
    "non-integral division unless the script states the rounding rule. Every required "
    "fact, clue event, and episode obligation must cite a verbatim "
    "excerpt that exists in the script. Return only the structured episode-script "
    "result for the requested episode number."
)

VIRTUAL_FILE_PERMISSIONS = [
    FilesystemPermission(operations=["read"], paths=["/persona", "/persona/**"]),
    FilesystemPermission(
        operations=["write"],
        paths=["/persona", "/persona/**"],
        mode="deny",
    ),
    FilesystemPermission(
        operations=["read", "write"],
        paths=["/workspace", "/workspace/**"],
    ),
    FilesystemPermission(operations=["read", "write"], paths=["/**"], mode="deny"),
]

SKILLED_WRITE_PERMISSIONS = [
    FilesystemPermission(
        operations=["read"],
        paths=["/persona", "/persona/**", "/skills", "/skills/**"],
    ),
    FilesystemPermission(
        operations=["write"],
        paths=["/persona", "/persona/**", "/skills", "/skills/**"],
        mode="deny",
    ),
    FilesystemPermission(
        operations=["read", "write"],
        paths=["/workspace", "/workspace/**"],
    ),
    FilesystemPermission(operations=["read", "write"], paths=["/**"], mode="deny"),
]

REVIEW_FILE_PERMISSIONS = [
    FilesystemPermission(
        operations=["read"],
        paths=["/persona", "/persona/**", "/skills", "/skills/**", "/workspace", "/workspace/**"],
    ),
    FilesystemPermission(operations=["write"], paths=["/**"], mode="deny"),
]


class StoryArchitectResult(StrictModel):
    stage: Literal[
        "selecting_l0_variant",
        "generating_story_outline",
        "generating_character_biographies",
        "generating_relationship_logic",
    ] = Field(description="The exact stage named in the delegated task.")
    content: NonEmptyText | None = Field(
        default=None,
        description=(
            "Required for generating_story_outline, "
            "generating_character_biographies, and generating_relationship_logic. "
            "Must be null for selecting_l0_variant."
        ),
    )
    selected_l0_variant: NonEmptyText | None = Field(
        default=None,
        description=(
            "Required only for selecting_l0_variant. Must be null for every generation stage."
        ),
    )
    selection_rationale: NonEmptyText | None = Field(
        default=None,
        description=(
            "Required only for selecting_l0_variant. Must be null for every generation stage."
        ),
    )

    @model_validator(mode="after")
    def validate_stage_payload(self) -> "StoryArchitectResult":
        if self.stage == InternalStage.SELECTING_L0_VARIANT:
            if not self.selected_l0_variant or not self.selection_rationale or self.content:
                raise ValueError("L0 selection requires only variant and rationale")
        elif not self.content or self.selected_l0_variant or self.selection_rationale:
            raise ValueError("Story artifact stages require only content")
        return self


class EpisodePlannerResult(StrictModel):
    stage: Literal["generating_episode_outline"]
    content: NonEmptyText
    episode_count: int = Field(ge=1)
    episodes: list[EpisodePlan] = Field(min_length=1)
    story_contract: StoryContract

    @model_validator(mode="after")
    def validate_episode_sequence(self) -> "EpisodePlannerResult":
        expected = list(range(1, self.episode_count + 1))
        numbers = [episode.episode_number for episode in self.episodes]
        if numbers != expected:
            raise ValueError("Episode plans must be ordered and contiguous from 1")
        if self.story_contract.episode_count != self.episode_count:
            raise ValueError("Story contract episode count must match the episode plan")
        return self


class LegacyEpisodePlannerResult(StrictModel):
    """Pre-contract outline retained only so an already-started run can finish."""

    stage: Literal["generating_episode_outline"]
    content: NonEmptyText
    episode_count: int = Field(ge=1)
    episodes: list[EpisodePlan] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_episode_sequence(self) -> "LegacyEpisodePlannerResult":
        expected = list(range(1, self.episode_count + 1))
        if [episode.episode_number for episode in self.episodes] != expected:
            raise ValueError("Episode plans must be ordered and contiguous from 1")
        return self


class ScriptWriterResult(StrictModel):
    stage: Literal["generating_episode_scripts"]
    episode_number: int = Field(ge=1)
    content: NonEmptyText
    state_delta: EpisodeStateDelta | None = None

    @model_validator(mode="after")
    def validate_delta_episode(self) -> "ScriptWriterResult":
        if self.state_delta is not None and self.state_delta.episode_number != self.episode_number:
            raise ValueError("Episode state delta must match the script episode")
        return self


class CanonReviewerResult(SemanticReview):
    pass


class EpisodeReviewerResult(SemanticReview):
    pass


class QualityReviewerResult(StrictModel):
    stage: Literal["accepting_l0", "accepting_l4"] = Field(
        description="The exact gate stage named in the delegated task."
    )
    passed: bool = Field(
        description=(
            "Whether the approved artifacts satisfy the named gate. "
            "Never report true without concrete evidence."
        )
    )
    evidence: NonEmptyText = Field(description="Concrete evidence supporting the gate decision.")
    feedback_handling: list[FeedbackHandlingItem] = Field(
        default_factory=list,
        description=(
            "Empty for accepting_l0 and for an initial run. "
            "For a revision's accepting_l4 gate, itemize every frozen feedback item."
        ),
    )


class WorkflowCompletion(StrictModel):
    completed: Literal[True] = Field(
        description="Confirms that every required specialist stage and gate completed."
    )


class AgentProtocolError(RuntimeError):
    def __init__(self, message: str, *, stage: InternalStage | None = None) -> None:
        super().__init__(message)
        self.stage = stage


class QualityGateRejectedError(RuntimeError):
    def __init__(self, *, stage: InternalStage, evidence: str | None = None) -> None:
        super().__init__("Quality gate did not pass")
        self.stage = stage
        self.evidence = evidence


class ContentReviewRejectedError(RuntimeError):
    def __init__(
        self,
        *,
        stage: InternalStage,
        evidence: str,
        episode_number: int | None = None,
        repair_rounds: int = 2,
    ) -> None:
        super().__init__("Content review did not converge")
        self.stage = stage
        self.evidence = evidence
        self.episode_number = episode_number
        self.repair_rounds = repair_rounds


class CheckpointUnavailableError(RuntimeError):
    """The durable thread state required for a resumed run is missing."""


class EpisodeTimeoutError(TimeoutError):
    def __init__(self, episode_number: int) -> None:
        super().__init__("Episode script generation timed out")
        self.stage = InternalStage.GENERATING_EPISODE_SCRIPTS
        self.episode_number = episode_number


def _calculate_arithmetic(
    left: str,
    operation: Literal["add", "subtract", "multiply", "divide"],
    right: str,
) -> str:
    lhs = _bounded_decimal(left)
    rhs = _bounded_decimal(right)
    left_fraction = Fraction(lhs)
    right_fraction = Fraction(rhs)
    if operation == "add":
        result = left_fraction + right_fraction
    elif operation == "subtract":
        result = left_fraction - right_fraction
    elif operation == "multiply":
        result = left_fraction * right_fraction
    else:
        if right_fraction == 0:
            raise ValueError("Cannot divide by zero")
        result = left_fraction / right_fraction
    decimal_result = _exact_decimal(result)
    if decimal_result is not None:
        return decimal_result
    return (
        f"{result.numerator}/{result.denominator} "
        "(non-terminating decimal; do not round without an explicit rule)"
    )


def _calculate_arithmetic_tool(
    left: str,
    operation: Literal["add", "subtract", "multiply", "divide"],
    right: str,
) -> str:
    try:
        return _calculate_arithmetic(left, operation, right)
    except ValueError as exc:
        raise ToolException(
            f"Invalid arithmetic input: {exc}. Use decimal operands only; "
            "convert clock times to elapsed minutes before calculating."
        ) from exc


def _arithmetic_tool() -> StructuredTool:
    return StructuredTool.from_function(
        func=_calculate_arithmetic_tool,
        name="calculate_arithmetic",
        description=(
            "Calculate one exact decimal add, subtract, multiply, or divide operation. "
            "Operands must be decimal numbers; convert clock times to elapsed minutes first. "
            "Use it before writing any derived numeric claim."
        ),
        handle_tool_error=True,
    )


def _bounded_decimal(value: str) -> Decimal:
    stripped = value.strip()
    if not stripped or len(stripped) > 64:
        raise ValueError("Operand is empty or too long")
    try:
        parsed = Decimal(stripped)
    except DecimalException as exc:
        raise ValueError("Operands must be decimal numbers") from exc
    sign, digits, exponent = parsed.as_tuple()
    del sign
    if not parsed.is_finite() or len(digits) > 64 or abs(exponent) > 100:
        raise ValueError("Operand must be a finite bounded decimal")
    return parsed


def _exact_decimal(value: Fraction) -> str | None:
    denominator = value.denominator
    twos = 0
    fives = 0
    while denominator % 2 == 0:
        denominator //= 2
        twos += 1
    while denominator % 5 == 0:
        denominator //= 5
        fives += 1
    if denominator != 1:
        return None
    scale = max(twos, fives)
    scaled = value.numerator * (2 ** (scale - twos)) * (5 ** (scale - fives))
    if scale == 0:
        return str(scaled)
    sign = "-" if scaled < 0 else ""
    digits = str(abs(scaled)).zfill(scale + 1)
    whole = digits[:-scale]
    fraction = digits[-scale:].rstrip("0")
    return f"{sign}{whole}.{fraction}" if fraction else f"{sign}{whole}"


def register_pengine_harness_profile(provider_key: str = "anthropic") -> None:
    if provider_key in _REGISTERED_PROFILE_KEYS:
        return
    register_harness_profile(
        provider_key,
        HarnessProfile(
            excluded_tools=frozenset({"execute"}),
            general_purpose_subagent=GeneralPurposeSubagentProfile(enabled=False),
        ),
    )
    _REGISTERED_PROFILE_KEYS.add(provider_key)


def _tool_message(result: ToolMessage | Command[Any]) -> ToolMessage:
    if isinstance(result, ToolMessage):
        return result
    update = result.update
    if not isinstance(update, Mapping):
        raise AgentProtocolError("Subagent command did not contain an update")
    messages = update.get("messages")
    if not isinstance(messages, list) or len(messages) != 1:
        raise AgentProtocolError("Subagent command did not contain one tool result")
    message = messages[0]
    if not isinstance(message, ToolMessage):
        raise AgentProtocolError("Subagent command result was not a tool message")
    return message


def _request_state_after_result(
    request: ToolCallRequest,
    result: ToolMessage | Command[Any],
) -> Any:
    if not isinstance(result, Command):
        return request.state
    if not isinstance(request.state, Mapping) or not isinstance(result.update, Mapping):
        return request.state
    return {**request.state, **result.update}


def _request_with_files(
    request: ToolCallRequest,
    files: Mapping[str, str],
) -> ToolCallRequest:
    if not isinstance(request.state, Mapping):
        raise AgentProtocolError("Task state is unavailable")
    existing_files = request.state.get("files")
    state = {
        **request.state,
        "files": {
            **(dict(existing_files) if isinstance(existing_files, Mapping) else {}),
            **{path: {"content": content, "encoding": "utf-8"} for path, content in files.items()},
        },
    }
    return request.override(
        state=state,
        runtime=(replace(request.runtime, state=state) if request.runtime is not None else None),
    )


def _subagent_request(
    request: ToolCallRequest,
    *,
    subagent_type: str,
    description: str,
    files: Mapping[str, str],
) -> ToolCallRequest:
    with_files = _request_with_files(request, files)
    return with_files.override(
        tool_call={
            **request.tool_call,
            "args": {
                "description": description,
                "subagent_type": subagent_type,
            },
        }
    )


def _validated_stage_payload(
    stage: InternalStage,
    content: str,
    *,
    expected_episode_number: int | None = None,
) -> Mapping[str, Any]:
    try:
        raw = json.loads(content)
        if (
            isinstance(raw, Mapping)
            and isinstance(raw.get("stage"), str)
            and raw["stage"] != stage.value
        ):
            raise AgentProtocolError("Subagent returned a different stage", stage=stage)
        if stage in _STORY_STAGES:
            parsed = StoryArchitectResult.model_validate(raw)
        elif stage is InternalStage.GENERATING_EPISODE_OUTLINE:
            parsed = EpisodePlannerResult.model_validate(raw)
        elif stage is InternalStage.GENERATING_EPISODE_SCRIPTS:
            parsed = ScriptWriterResult.model_validate(raw)
        elif stage in (InternalStage.ACCEPTING_L0, InternalStage.ACCEPTING_L4):
            parsed = QualityReviewerResult.model_validate(raw)
        else:
            raise AgentProtocolError("Task tool declared a non-specialist stage", stage=stage)
    except AgentProtocolError:
        raise
    except Exception as exc:
        raise AgentProtocolError(
            "Subagent returned invalid structured output",
            stage=stage,
        ) from exc
    if parsed.stage != stage.value:
        raise AgentProtocolError("Subagent returned a different stage", stage=stage)
    if isinstance(parsed, ScriptWriterResult) and (
        expected_episode_number is None or parsed.episode_number != expected_episode_number
    ):
        raise AgentProtocolError("Subagent returned a different episode", stage=stage)
    if isinstance(parsed, QualityReviewerResult) and not parsed.passed:
        raise QualityGateRejectedError(stage=stage, evidence=parsed.evidence)
    return parsed.model_dump(mode="json")


def _workflow_result_from_checkpoints(
    approved: Mapping[InternalStage, Any],
) -> WorkflowResult:
    missing = [stage for stage in _ORDERED_SPECIALIST_STAGES if stage not in approved]
    if missing:
        raise AgentProtocolError(
            "Supervisor finished before every specialist stage was approved",
            stage=missing[0],
        )
    try:
        l0_selection = approved[InternalStage.SELECTING_L0_VARIANT]
        l0_gate = approved[InternalStage.ACCEPTING_L0]
        l4_gate = approved[InternalStage.ACCEPTING_L4]
        return WorkflowResult.model_validate(
            {
                "content_package": {
                    "story_outline": approved[InternalStage.GENERATING_STORY_OUTLINE]["content"],
                    "character_biographies": approved[
                        InternalStage.GENERATING_CHARACTER_BIOGRAPHIES
                    ]["content"],
                    "relationship_logic": approved[InternalStage.GENERATING_RELATIONSHIP_LOGIC][
                        "content"
                    ],
                    "episode_outline": approved[InternalStage.GENERATING_EPISODE_OUTLINE][
                        "content"
                    ],
                    "episode_scripts": approved[InternalStage.GENERATING_EPISODE_SCRIPTS][
                        "content"
                    ],
                },
                "selected_l0_variant": l0_selection["selected_l0_variant"],
                "selection_rationale": l0_selection["selection_rationale"],
                "l0_gate": {
                    "passed": l0_gate["passed"],
                    "evidence": l0_gate["evidence"],
                },
                "l4_gate": {
                    "passed": l4_gate["passed"],
                    "evidence": l4_gate["evidence"],
                },
                "feedback_handling": l4_gate.get("feedback_handling", []),
            }
        )
    except AgentProtocolError:
        raise
    except Exception as exc:
        raise AgentProtocolError(
            "Approved specialist checkpoints are invalid",
            stage=InternalStage.ASSEMBLING_DELIVERY,
        ) from exc


def _review_workspace_files(
    approved: Mapping[InternalStage, Any],
) -> dict[str, dict[str, str]]:
    files: dict[str, dict[str, str]] = {}
    for stage, path in _WORKSPACE_ARTIFACT_PATHS.items():
        payload = approved.get(stage)
        if not isinstance(payload, Mapping):
            continue
        content = payload.get("content")
        if isinstance(content, str) and content:
            files[path] = {"content": content, "encoding": "utf-8"}
        if stage is InternalStage.GENERATING_EPISODE_OUTLINE:
            contract = payload.get("story_contract")
            contract_markdown = payload.get("story_contract_markdown")
            if isinstance(contract, Mapping):
                files["/workspace/story_contract.json"] = {
                    "content": json.dumps(contract, ensure_ascii=False, sort_keys=True),
                    "encoding": "utf-8",
                }
            if isinstance(contract_markdown, str) and contract_markdown:
                files["/workspace/story_contract.md"] = {
                    "content": contract_markdown,
                    "encoding": "utf-8",
                }
    manifest = json.dumps(
        {stage.value: payload for stage, payload in approved.items()},
        ensure_ascii=False,
        sort_keys=True,
    )
    files["/workspace/approved-checkpoints.json"] = {
        "content": manifest,
        "encoding": "utf-8",
    }
    return files


class StageGuardMiddleware(AgentMiddleware):
    def __init__(
        self,
        before_stage: StageHook,
        approve_stage: CheckpointHook,
        approved_stages: set[InternalStage],
        *,
        approved_payloads: dict[InternalStage, Any] | None = None,
        episode_drafts: list[EpisodeDraft] | None = None,
        before_episode: EpisodeAttemptHook | None = None,
        commit_episode: EpisodeCommitHook | None = None,
        assemble_episode_scripts: EpisodeAssemblyHook | None = None,
        episode_timeout_seconds: float | None = None,
        reset_episode_deadline: EpisodeDeadlineReset | None = None,
    ) -> None:
        self.before_stage = before_stage
        self.approve_stage = approve_stage
        self.approved_stages = approved_stages
        self.approved_payloads = approved_payloads if approved_payloads is not None else {}
        self.episode_drafts = {draft.episode_number: draft for draft in episode_drafts or []}
        self.before_episode = before_episode
        self.commit_episode = commit_episode
        self.assemble_episode_scripts = assemble_episode_scripts
        self.episode_timeout_seconds = episode_timeout_seconds
        self.reset_episode_deadline = reset_episode_deadline

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command[Any]]],
    ) -> ToolMessage | Command[Any]:
        if request.tool_call["name"] != "task":
            return await handler(request)

        args = request.tool_call.get("args", {})
        description = args.get("description")
        subagent_type = args.get("subagent_type")
        if not isinstance(description, str):
            raise AgentProtocolError("Subagent task omitted its stage token")
        match = _STAGE_TOKEN.match(description)
        if match is None:
            raise AgentProtocolError("Subagent task omitted its stage token")
        try:
            stage = InternalStage(match.group(1))
        except ValueError as exc:
            raise AgentProtocolError("Subagent task declared an unknown stage") from exc
        if _TASK_OWNER.get(stage) != subagent_type:
            raise AgentProtocolError("Stage was delegated to the wrong subagent", stage=stage)
        expected_stage = next(
            (
                candidate
                for candidate in _ORDERED_SPECIALIST_STAGES
                if candidate not in self.approved_stages
            ),
            None,
        )
        if stage != expected_stage:
            return ToolMessage(
                content=json.dumps(
                    {
                        "error": "stage_out_of_order",
                        "expected_stage": expected_stage.value if expected_stage else None,
                        "instruction": (
                            "Delegate exactly one missing specialist stage per turn and "
                            "wait for its tool result before delegating the next stage."
                        ),
                    },
                    separators=(",", ":"),
                ),
                tool_call_id=request.tool_call["id"],
                name="task",
                status="error",
            )

        if stage in (
            InternalStage.ACCEPTING_L0,
            InternalStage.ACCEPTING_L4,
        ):
            review_files = _review_workspace_files(self.approved_payloads)
            if not isinstance(request.state, Mapping):
                raise AgentProtocolError("Task state is unavailable", stage=stage)
            existing_files = request.state.get("files")
            canonical_paths = {
                *_WORKSPACE_ARTIFACT_PATHS.values(),
                "/workspace/approved-checkpoints.json",
            }
            noncanonical_files = (
                {
                    path: file_data
                    for path, file_data in existing_files.items()
                    if path not in canonical_paths
                }
                if isinstance(existing_files, Mapping)
                else {}
            )
            review_state = {
                **request.state,
                "files": {
                    **noncanonical_files,
                    **review_files,
                },
            }
            request = request.override(
                state=review_state,
                runtime=(
                    replace(request.runtime, state=review_state)
                    if request.runtime is not None
                    else None
                ),
            )

        if stage is InternalStage.GENERATING_EPISODE_SCRIPTS:
            return await self._write_episodes(request, handler, args)

        await self.before_stage(stage)
        if stage is InternalStage.GENERATING_EPISODE_OUTLINE:
            request = _request_with_files(
                request,
                {
                    path: data["content"]
                    for path, data in _review_workspace_files(self.approved_payloads).items()
                },
            )
            result, payload = await self._generate_locked_outline(
                request,
                handler,
                args,
            )
            await self.approve_stage(stage, payload)
            self.approved_payloads[stage] = dict(payload)
            self.approved_stages.add(stage)
            return result
        result, payload = await self._call_structured_stage(
            stage,
            request,
            handler,
            args,
        )
        await self.approve_stage(stage, payload)
        self.approved_stages.add(stage)
        return result

    async def _generate_locked_outline(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command[Any]]],
        args: Mapping[str, Any],
    ) -> tuple[ToolMessage | Command[Any], Mapping[str, Any]]:
        result, payload = await self._call_structured_stage(
            InternalStage.GENERATING_EPISODE_OUTLINE,
            request,
            handler,
            args,
        )
        repair_rounds = 0
        while True:
            contract = StoryContract.model_validate(payload["story_contract"])
            contract_hash = story_contract_sha256(contract)
            contract_markdown = render_story_contract_markdown(contract, contract_hash)
            candidate = {
                **payload,
                "story_contract_sha256": contract_hash,
                "story_contract_markdown": contract_markdown,
            }
            review = await self._invoke_semantic_reviewer(
                request=request,
                handler=handler,
                subagent_type="canon_reviewer",
                description=(
                    "Review the proposed minimum continuity ledger against every approved "
                    "upstream artifact. Fail on contradictions, missing established identity, "
                    "relationship, alias, pronoun, age, duration, call-participant, clue or causal "
                    "facts, ambiguous typed numbers, unfair knowledge withholding, or incomplete "
                    "clue lifecycle. Do not require facts that the upstream artifacts leave "
                    "genuinely unspecified."
                ),
                files={
                    "/workspace/story_contract.json": json.dumps(
                        contract.model_dump(mode="json"),
                        ensure_ascii=False,
                        sort_keys=True,
                    ),
                    "/workspace/story_contract.md": contract_markdown,
                    "/workspace/episode_outline.md": payload["content"],
                },
                schema=CanonReviewerResult,
            )
            if review.passed:
                return result, {
                    **candidate,
                    "contract_review": review.model_dump(mode="json"),
                    "contract_repair_rounds": repair_rounds,
                }
            if repair_rounds >= 2:
                raise ContentReviewRejectedError(
                    stage=InternalStage.GENERATING_EPISODE_OUTLINE,
                    evidence=review.evidence,
                    repair_rounds=repair_rounds,
                )
            repair_rounds += 1
            result, payload = await self._invoke_repair_subagent(
                request=request,
                handler=handler,
                subagent_type="canon_repair",
                description=(
                    "Repair the episode outline and story contract using the frozen upstream "
                    f"artifacts. This is repair round {repair_rounds} of 2. Address every "
                    "review issue without changing approved story intent."
                ),
                files={
                    "/workspace/story_contract.json": json.dumps(
                        contract.model_dump(mode="json"),
                        ensure_ascii=False,
                        sort_keys=True,
                    ),
                    "/workspace/story_contract_review.json": review.model_dump_json(),
                    "/workspace/episode_outline.md": payload["content"],
                },
                schema=EpisodePlannerResult,
                stage=InternalStage.GENERATING_EPISODE_OUTLINE,
            )

    async def _call_structured_stage(
        self,
        stage: InternalStage,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command[Any]]],
        args: Mapping[str, Any],
        *,
        expected_episode_number: int | None = None,
    ) -> tuple[ToolMessage | Command[Any], Mapping[str, Any]]:
        description = args.get("description")
        if not isinstance(description, str):
            raise AgentProtocolError("Subagent task omitted its stage token", stage=stage)
        result = await handler(request)
        message = _tool_message(result)
        if not isinstance(message.content, str):
            raise AgentProtocolError("Subagent result was not JSON text", stage=stage)
        try:
            payload = _validated_stage_payload(
                stage,
                message.content,
                expected_episode_number=expected_episode_number,
            )
        except AgentProtocolError as exc:
            if str(exc) != "Subagent returned invalid structured output":
                raise
            retry_args = {
                **args,
                "description": (
                    f"{description}\n"
                    "The previous attempt ended without the required structured "
                    f"result. Reuse its completed workspace artifacts and return "
                    f"exactly one {_RESULT_TOOL[stage]} tool call now. Do not return "
                    "a prose summary."
                ),
            }
            retry_request = request.override(
                tool_call={**request.tool_call, "args": retry_args},
                state=_request_state_after_result(request, result),
            )
            result = await handler(retry_request)
            message = _tool_message(result)
            if not isinstance(message.content, str):
                raise AgentProtocolError(
                    "Subagent result was not JSON text",
                    stage=stage,
                ) from exc
            payload = _validated_stage_payload(
                stage,
                message.content,
                expected_episode_number=expected_episode_number,
            )
        return result, payload

    async def _invoke_semantic_reviewer(
        self,
        *,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command[Any]]],
        subagent_type: str,
        description: str,
        files: Mapping[str, str],
        schema: type[SemanticReview],
    ) -> SemanticReview:
        review_request = _subagent_request(
            request,
            subagent_type=subagent_type,
            description=description,
            files={
                **{
                    path: data["content"]
                    for path, data in _review_workspace_files(self.approved_payloads).items()
                },
                **files,
            },
        )
        result = await handler(review_request)
        message = _tool_message(result)
        if not isinstance(message.content, str):
            raise AgentProtocolError("Semantic reviewer result was not JSON text")
        try:
            return schema.model_validate_json(message.content)
        except Exception as exc:
            raise AgentProtocolError(
                "Semantic reviewer returned invalid structured output"
            ) from exc

    async def _invoke_repair_subagent(
        self,
        *,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command[Any]]],
        subagent_type: str,
        description: str,
        files: Mapping[str, str],
        schema: type[EpisodePlannerResult] | type[ScriptWriterResult],
        stage: InternalStage,
        expected_episode_number: int | None = None,
    ) -> tuple[ToolMessage | Command[Any], Mapping[str, Any]]:
        repair_request = _subagent_request(
            request,
            subagent_type=subagent_type,
            description=description,
            files={
                **{
                    path: data["content"]
                    for path, data in _review_workspace_files(self.approved_payloads).items()
                },
                **files,
            },
        )
        result = await handler(repair_request)
        message = _tool_message(result)
        if not isinstance(message.content, str):
            raise AgentProtocolError("Repair result was not JSON text", stage=stage)
        try:
            parsed = schema.model_validate_json(message.content)
        except Exception as exc:
            raise AgentProtocolError(
                "Repair subagent returned invalid structured output",
                stage=stage,
            ) from exc
        if parsed.stage != stage.value:
            raise AgentProtocolError("Repair subagent returned a different stage", stage=stage)
        if (
            isinstance(parsed, ScriptWriterResult)
            and parsed.episode_number != expected_episode_number
        ):
            raise AgentProtocolError("Repair subagent returned a different episode", stage=stage)
        return result, parsed.model_dump(mode="json")

    async def _write_episodes(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command[Any]]],
        args: Mapping[str, Any],
    ) -> ToolMessage | Command[Any]:
        if (
            self.before_episode is None
            or self.commit_episode is None
            or self.assemble_episode_scripts is None
        ):
            raise AgentProtocolError(
                "Episode generation hooks are required",
                stage=InternalStage.GENERATING_EPISODE_SCRIPTS,
            )
        outline = self.approved_payloads.get(InternalStage.GENERATING_EPISODE_OUTLINE)
        legacy_recovery = isinstance(outline, Mapping) and "story_contract" not in outline
        try:
            if legacy_recovery:
                legacy_outline = LegacyEpisodePlannerResult.model_validate(outline)
                plans = legacy_outline.episodes
                contract = None
                contract_hash = None
                contract_json = None
                prior_state = None
                if any(
                    draft.contract_sha256 is not None
                    or draft.series_state is not None
                    or draft.series_state_sha256 is not None
                    for draft in self.episode_drafts.values()
                ):
                    raise ValueError("Legacy episode drafts contain unexpected lock data")
            else:
                parsed_outline = EpisodePlannerResult.model_validate(
                    {field: outline[field] for field in EpisodePlannerResult.model_fields}
                )
                plans = parsed_outline.episodes
                contract = parsed_outline.story_contract
                contract_hash = story_contract_sha256(contract)
                if outline["story_contract_sha256"] != contract_hash:
                    raise ValueError("Story contract hash does not match its content")
                contract_review = CanonReviewerResult.model_validate(outline["contract_review"])
                if not contract_review.passed:
                    raise ValueError("Story contract was not independently approved")
                contract_json = json.dumps(
                    contract.model_dump(mode="json"),
                    ensure_ascii=False,
                    sort_keys=True,
                )
                prior_state = initial_series_state(contract, contract_hash)
                for episode_number, draft in sorted(self.episode_drafts.items()):
                    if (
                        draft.contract_sha256 != contract_hash
                        or draft.series_state is None
                        or draft.series_state_sha256 is None
                        or canonical_model_hash(draft.series_state) != draft.series_state_sha256
                        or episode_number != prior_state.locked_through_episode + 1
                    ):
                        raise CheckpointUnavailableError(
                            "Committed episode continuity state is unavailable or mismatched."
                        )
                    prior_state = draft.series_state
        except CheckpointUnavailableError:
            raise
        except Exception as exc:
            message = (
                "Legacy episode recovery requires a valid approved episode outline"
                if legacy_recovery
                else "Episode scripts require an approved locked story contract"
            )
            raise AgentProtocolError(
                message,
                stage=InternalStage.GENERATING_EPISODE_SCRIPTS,
            ) from exc

        last_result: ToolMessage | Command[Any] | None = None
        for plan in plans:
            if plan.episode_number in self.episode_drafts:
                continue
            if self.reset_episode_deadline is not None:
                await self.reset_episode_deadline()
            await self.before_episode(plan)
            episode_args = {
                **args,
                "description": (
                    f"[stage=generating_episode_scripts][episode={plan.episode_number}] "
                    f"Write only episode {plan.episode_number}.\n"
                    f"Approved episode plan:\n{plan.plan}"
                    + (
                        "\nThis is a grandfathered pre-contract run. Preserve every approved "
                        "upstream artifact and committed earlier episode; return content for this "
                        "episode without inventing a replacement story contract."
                        if legacy_recovery
                        else f"\nLocked contract SHA-256: {contract_hash}"
                    )
                ),
            }
            episode_files = {
                f"/workspace/episodes/ep{number}.md": draft.content
                for number, draft in sorted(self.episode_drafts.items())
            }
            if not legacy_recovery:
                assert contract_json is not None and prior_state is not None
                episode_files.update(
                    {
                        "/workspace/story_contract.json": contract_json,
                        "/workspace/story_contract.md": outline["story_contract_markdown"],
                        "/workspace/series_state.json": prior_state.model_dump_json(),
                        "/workspace/previous_episode_handoff.md": prior_state.handoff or "None",
                    }
                )
            episode_request = _request_with_files(
                request.override(
                    tool_call={**request.tool_call, "args": episode_args},
                ),
                episode_files,
            )
            try:
                if self.episode_timeout_seconds is None:
                    result, payload = await self._call_structured_stage(
                        InternalStage.GENERATING_EPISODE_SCRIPTS,
                        episode_request,
                        handler,
                        episode_args,
                        expected_episode_number=plan.episode_number,
                    )
                else:
                    async with asyncio.timeout(self.episode_timeout_seconds):
                        result, payload = await self._call_structured_stage(
                            InternalStage.GENERATING_EPISODE_SCRIPTS,
                            episode_request,
                            handler,
                            episode_args,
                            expected_episode_number=plan.episode_number,
                        )
            except TimeoutError as exc:
                raise EpisodeTimeoutError(plan.episode_number) from exc

            parsed = ScriptWriterResult.model_validate(payload)
            if legacy_recovery:
                committed = await self.commit_episode(
                    plan.episode_number,
                    parsed.content,
                    None,
                )
                self.episode_drafts[plan.episode_number] = committed
                last_result = result
                continue
            if parsed.state_delta is None:
                raise AgentProtocolError(
                    "Contract-bound episode omitted its continuity state delta",
                    stage=InternalStage.GENERATING_EPISODE_SCRIPTS,
                )
            assert contract is not None and contract_hash is not None and prior_state is not None
            repair_rounds = 0
            while True:
                deterministic_issues = validate_episode_candidate(
                    contract=contract,
                    contract_sha256=contract_hash,
                    prior_state=prior_state,
                    content=parsed.content,
                    delta=parsed.state_delta,
                )
                if deterministic_issues:
                    review = EpisodeReviewerResult(
                        passed=False,
                        evidence="; ".join(
                            f"{issue.code}: {issue.message}" for issue in deterministic_issues
                        ),
                        issues=deterministic_issues,
                    )
                else:
                    review = await self._invoke_semantic_reviewer(
                        request=episode_request,
                        handler=handler,
                        subagent_type="episode_reviewer",
                        description=(
                            f"Review episode {plan.episode_number} and the complete committed "
                            "series prefix against the locked contract and every approved upstream "
                            "artifact. Compare identities, relationships, aliases, pronouns, ages, "
                            "durations, call participants, clue meanings, causal facts, viewpoint "
                            "knowledge, cast, and episode obligation across all prior scripts and "
                            "the current candidate. On the final episode this is the whole-series "
                            "consistency review before script-stage approval. Return structured "
                            "evidence only."
                        ),
                        files={
                            "/workspace/story_contract.json": contract_json,
                            "/workspace/series_state.json": prior_state.model_dump_json(),
                            "/workspace/candidate_episode.md": parsed.content,
                            "/workspace/series_prefix.md": "\n\n---\n\n".join(
                                [
                                    *(
                                        f"第 {episode_number} 集\n{draft.content}"
                                        for episode_number, draft in sorted(
                                            self.episode_drafts.items()
                                        )
                                    ),
                                    f"第 {plan.episode_number} 集\n{parsed.content}",
                                ]
                            ),
                            "/workspace/candidate_state_delta.json": (
                                parsed.state_delta.model_dump_json()
                            ),
                        },
                        schema=EpisodeReviewerResult,
                    )
                if review.passed:
                    try:
                        episode_lock = build_episode_lock(
                            contract=contract,
                            contract_sha256=contract_hash,
                            prior_state=prior_state,
                            content=parsed.content,
                            delta=parsed.state_delta,
                            semantic_review=review,
                            repair_rounds=repair_rounds,
                        )
                    except ContinuityViolation as exc:
                        raise AgentProtocolError(
                            exc.evidence,
                            stage=InternalStage.GENERATING_EPISODE_SCRIPTS,
                        ) from exc
                    committed = await self.commit_episode(
                        plan.episode_number,
                        parsed.content,
                        episode_lock,
                    )
                    self.episode_drafts[plan.episode_number] = committed
                    prior_state = episode_lock.series_state
                    last_result = result
                    break
                if repair_rounds >= 2:
                    raise ContentReviewRejectedError(
                        stage=InternalStage.GENERATING_EPISODE_SCRIPTS,
                        evidence=review.evidence,
                        episode_number=plan.episode_number,
                        repair_rounds=repair_rounds,
                    )
                repair_rounds += 1
                result, payload = await self._invoke_repair_subagent(
                    request=episode_request,
                    handler=handler,
                    subagent_type="episode_repair",
                    description=(
                        f"Repair episode {plan.episode_number}; round {repair_rounds} of 2. "
                        "Change only the unlocked candidate and state delta. Keep the locked "
                        "contract and earlier episodes unchanged, and address every review issue."
                    ),
                    files={
                        "/workspace/story_contract.json": contract_json,
                        "/workspace/series_state.json": prior_state.model_dump_json(),
                        "/workspace/candidate_episode.md": parsed.content,
                        "/workspace/candidate_state_delta.json": (
                            parsed.state_delta.model_dump_json()
                        ),
                        "/workspace/episode_review.json": review.model_dump_json(),
                    },
                    schema=ScriptWriterResult,
                    stage=InternalStage.GENERATING_EPISODE_SCRIPTS,
                    expected_episode_number=plan.episode_number,
                )
                parsed = ScriptWriterResult.model_validate(payload)
                if parsed.state_delta is None:
                    raise AgentProtocolError(
                        "Contract-bound episode repair omitted its continuity state delta",
                        stage=InternalStage.GENERATING_EPISODE_SCRIPTS,
                    )

        aggregate = await self.assemble_episode_scripts()
        payload = {"stage": InternalStage.GENERATING_EPISODE_SCRIPTS.value, "content": aggregate}
        if not legacy_recovery:
            assert contract_hash is not None and prior_state is not None
            payload.update(
                {
                    "contract_sha256": contract_hash,
                    "episode_hashes": [
                        {
                            "episode_number": episode_number,
                            "content_sha256": draft.content_sha256,
                            "series_state_sha256": draft.series_state_sha256,
                        }
                        for episode_number, draft in sorted(self.episode_drafts.items())
                    ],
                    "series_state_sha256": canonical_model_hash(prior_state),
                }
            )
        await self.approve_stage(InternalStage.GENERATING_EPISODE_SCRIPTS, payload)
        self.approved_payloads[InternalStage.GENERATING_EPISODE_SCRIPTS] = payload
        self.approved_stages.add(InternalStage.GENERATING_EPISODE_SCRIPTS)
        if last_result is not None:
            return last_result
        return ToolMessage(
            content=json.dumps(payload, ensure_ascii=False),
            tool_call_id=request.tool_call["id"],
            name="task",
        )


@dataclass(frozen=True, slots=True)
class DeepAgentWorkflow:
    model: BaseChatModel
    checkpointer: BaseCheckpointSaver
    recursion_limit: int = 80
    provider_profile_key: str = "anthropic"

    def __post_init__(self) -> None:
        register_pengine_harness_profile(self.provider_profile_key)

    async def has_checkpoint(self, thread_id: str) -> bool:
        checkpoint = await self.checkpointer.aget_tuple({"configurable": {"thread_id": thread_id}})
        return checkpoint is not None

    async def execute(
        self,
        *,
        thread_id: str,
        story: str,
        requirements: str,
        persona_files: Mapping[str, str],
        before_stage: StageHook,
        approve_stage: CheckpointHook,
        approved_checkpoints: Mapping[InternalStage, Any] | None = None,
        episode_drafts: list[EpisodeDraft] | None = None,
        before_episode: EpisodeAttemptHook | None = None,
        commit_episode: EpisodeCommitHook | None = None,
        assemble_episode_scripts: EpisodeAssemblyHook | None = None,
        episode_timeout_seconds: float | None = None,
        reset_episode_deadline: EpisodeDeadlineReset | None = None,
        feedback: str | None = None,
        retrieve_references: ReferenceRetriever | None = None,
    ) -> WorkflowResult:
        approved_payloads: dict[InternalStage, Any] = {
            stage: payload for stage, payload in (approved_checkpoints or {}).items()
        }

        async def approve_and_capture(
            stage: InternalStage,
            payload: Mapping[str, Any],
        ) -> None:
            await approve_stage(stage, payload)
            approved_payloads[stage] = dict(payload)

        files = {
            path: {"content": content, "encoding": "utf-8"}
            for path, content in {**persona_files, **load_agent_skill_files()}.items()
        }
        files["/workspace/creation-request.md"] = {
            "content": (
                f"# Story\n\n{story}\n\n# Script requirements\n\n{requirements}\n\n"
                f"# Frozen revision feedback\n\n{feedback or 'None; this is the initial run.'}\n"
            ),
            "encoding": "utf-8",
        }
        approved_json = json.dumps(
            {stage.value: payload for stage, payload in (approved_checkpoints or {}).items()},
            ensure_ascii=False,
            sort_keys=True,
        )
        files["/workspace/approved-checkpoints.json"] = {
            "content": approved_json,
            "encoding": "utf-8",
        }

        tools = [_arithmetic_tool()]
        if retrieve_references is not None:

            async def retrieve_persona_references(query: str) -> str:
                """Return bounded read-only L5/L6 persona references for a focused query."""
                return await retrieve_references(query)

            tools.append(
                StructuredTool.from_function(
                    coroutine=retrieve_persona_references,
                    name="retrieve_persona_references",
                    description=(
                        "Retrieve a bounded set of read-only L5/L6 references. "
                        "Use only when a stage needs a focused style or craft example."
                    ),
                )
            )

        structured_output_retry = (
            "Return exactly one valid structured result tool call for the requested "
            "stage. Do not return the result as prose."
        )
        subagents = [
            {
                "name": "story_architect",
                "description": (
                    "Selects L0 and creates story outline, character biographies, "
                    "and relationship logic as separate structured tasks."
                ),
                "system_prompt": _STORY_ARCHITECT_PROMPT,
                "model": self.model,
                "tools": tools,
                "permissions": VIRTUAL_FILE_PERMISSIONS,
                "response_format": ToolStrategy(
                    schema=StoryArchitectResult,
                    handle_errors=structured_output_retry,
                ),
            },
            {
                "name": "episode_planner",
                "description": "Creates the complete episode outline.",
                "system_prompt": _EPISODE_PLANNER_PROMPT,
                "model": self.model,
                "tools": tools,
                "permissions": VIRTUAL_FILE_PERMISSIONS,
                "response_format": ToolStrategy(
                    schema=EpisodePlannerResult,
                    handle_errors=structured_output_retry,
                ),
            },
            {
                "name": "script_writer",
                "description": "Creates the complete episode scripts.",
                "system_prompt": _SCRIPT_WRITER_PROMPT,
                "model": self.model,
                "tools": tools,
                "permissions": VIRTUAL_FILE_PERMISSIONS,
                "response_format": ToolStrategy(
                    schema=ScriptWriterResult,
                    handle_errors=structured_output_retry,
                ),
            },
            {
                "name": "quality_reviewer",
                "description": (
                    "Reviews the L0 and L4 gates and itemizes revision-feedback coverage."
                ),
                "system_prompt": (
                    "Read the relevant /persona context and review only the named gate "
                    "against the approved artifacts supplied in the task. Always return "
                    "the structured stage, passed decision, and concrete evidence; never "
                    "return prose instead. Keep feedback_handling empty for accepting_l0 "
                    "and for an initial run. For a revision's accepting_l4 gate, itemize "
                    "every frozen feedback item."
                ),
                "model": self.model,
                "tools": tools,
                "permissions": VIRTUAL_FILE_PERMISSIONS,
                "response_format": ToolStrategy(
                    schema=QualityReviewerResult,
                    handle_errors=structured_output_retry,
                ),
            },
            {
                "name": "canon_reviewer",
                "description": "Independently reviews a proposed structured story contract.",
                "system_prompt": (
                    "Use the canon-review skill. Treat upstream artifacts as frozen. Read the "
                    "proposed JSON contract and return only structured review evidence. Never "
                    "repair or rewrite the candidate."
                ),
                "model": self.model,
                "tools": tools,
                "permissions": REVIEW_FILE_PERMISSIONS,
                "skills": _SPECIALIST_SKILL_SOURCES["canon_reviewer"],
                "response_format": ToolStrategy(
                    schema=CanonReviewerResult,
                    handle_errors=structured_output_retry,
                ),
            },
            {
                "name": "canon_repair",
                "description": "Repairs an unlocked episode outline and contract candidate.",
                "system_prompt": (
                    "Use the continuity-repair skill. Address every review issue while "
                    "preserving frozen upstream intent. Return the full structured episode "
                    "planner result only."
                ),
                "model": self.model,
                "tools": tools,
                "permissions": SKILLED_WRITE_PERMISSIONS,
                "skills": _SPECIALIST_SKILL_SOURCES["canon_repair"],
                "response_format": ToolStrategy(
                    schema=EpisodePlannerResult,
                    handle_errors=structured_output_retry,
                ),
            },
            {
                "name": "episode_reviewer",
                "description": "Independently reviews one episode against locked continuity.",
                "system_prompt": (
                    "Use the episode-continuity-review skill. The contract and prior state are "
                    "immutable. Return only structured review evidence and never repair content."
                ),
                "model": self.model,
                "tools": tools,
                "permissions": REVIEW_FILE_PERMISSIONS,
                "skills": _SPECIALIST_SKILL_SOURCES["episode_reviewer"],
                "response_format": ToolStrategy(
                    schema=EpisodeReviewerResult,
                    handle_errors=structured_output_retry,
                ),
            },
            {
                "name": "episode_repair",
                "description": "Repairs only the current unlocked episode candidate.",
                "system_prompt": (
                    "Use the continuity-repair skill. Keep the locked contract and earlier "
                    "episodes unchanged. Return the complete structured script result only."
                ),
                "model": self.model,
                "tools": tools,
                "permissions": SKILLED_WRITE_PERMISSIONS,
                "skills": _SPECIALIST_SKILL_SOURCES["episode_repair"],
                "response_format": ToolStrategy(
                    schema=ScriptWriterResult,
                    handle_errors=structured_output_retry,
                ),
            },
        ]

        supervisor = create_deep_agent(
            model=self.model,
            name="workflow_supervisor",
            system_prompt=_supervisor_prompt(
                story=story,
                requirements=requirements,
                feedback=feedback,
                approved_json=approved_json,
            ),
            tools=tools,
            middleware=[
                StageGuardMiddleware(
                    before_stage,
                    approve_and_capture,
                    set(approved_checkpoints or {}),
                    approved_payloads=approved_payloads,
                    episode_drafts=episode_drafts,
                    before_episode=before_episode,
                    commit_episode=commit_episode,
                    assemble_episode_scripts=assemble_episode_scripts,
                    episode_timeout_seconds=episode_timeout_seconds,
                    reset_episode_deadline=reset_episode_deadline,
                )
            ],
            subagents=subagents,
            permissions=VIRTUAL_FILE_PERMISSIONS,
            backend=StateBackend(),
            response_format=ToolStrategy(
                schema=WorkflowCompletion,
                handle_errors=structured_output_retry,
            ),
            checkpointer=self.checkpointer,
            store=None,
        )
        result = await supervisor.ainvoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": (
                            "Execute the bounded short-drama workflow now. "
                            "Return the complete structured result only after all gates pass."
                        ),
                    }
                ],
                "files": files,
            },
            {
                "configurable": {"thread_id": thread_id},
                "recursion_limit": self.recursion_limit,
            },
        )
        structured = result.get("structured_response")
        if structured is None:
            raise AgentProtocolError("Supervisor did not return structured output")
        try:
            WorkflowCompletion.model_validate(structured)
        except Exception as exc:
            raise AgentProtocolError("Supervisor returned invalid completion output") from exc
        return _workflow_result_from_checkpoints(approved_payloads)


def _supervisor_prompt(
    *,
    story: str,
    requirements: str,
    feedback: str | None,
    approved_json: str,
) -> str:
    revision = feedback if feedback is not None else "None; this is the initial run."
    return f"""\
You are the persona-bound workflow_supervisor for one short-drama creation.

Story:
{story}

Script requirements:
{requirements}

Frozen revision feedback:
{revision}

Already approved business checkpoints:
{approved_json}

Delegate every missing specialist stage exactly once, in this order:
1. selecting_l0_variant -> story_architect
2. generating_story_outline -> story_architect
3. generating_character_biographies -> story_architect
4. generating_relationship_logic -> story_architect
5. generating_episode_outline -> episode_planner
6. generating_episode_scripts -> script_writer
7. accepting_l0 -> quality_reviewer
8. accepting_l4 -> quality_reviewer

Every task description MUST begin with the exact token
`[stage=<stage_name>]`. Issue exactly one task tool call per model turn and wait
for its tool result before delegating the next stage. Do not delegate an already
approved stage. Direct stage tasks only to the four owners listed above; the
guarded runtime invokes contract review, episode review, and bounded repair
specialists automatically. Treat /persona as read-only and /workspace as temporary
thread scratch. Never claim a
gate passed without the quality_reviewer evidence. After all stages are
complete, return WorkflowCompletion only. Do not repeat the approved artifacts
or return partial content.

Preserve explicit numeric constraints from Script requirements. When Script
requirements do not specify an episode count, the active persona L4 baseline is
authoritative. Do not invent a different episode count or override any persona
numeric constraint in a delegated task.
"""
