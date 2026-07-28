import json
import re
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
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
from langchain_core.tools import StructuredTool
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.types import Command
from pydantic import Field, model_validator

from pengine.schemas import (
    FeedbackHandlingItem,
    InternalStage,
    NonEmptyText,
    StrictModel,
    WorkflowResult,
)

StageHook = Callable[[InternalStage], Awaitable[int]]
CheckpointHook = Callable[[InternalStage, Mapping[str, Any]], Awaitable[None]]
ReferenceRetriever = Callable[[str], Awaitable[str]]

_STAGE_TOKEN = re.compile(r"^\[stage=([a-z0-9_]+)\](?:\s|$)")
_REGISTERED_PROFILE_KEYS: set[str] = set()

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


class StoryArchitectResult(StrictModel):
    stage: Literal[
        "selecting_l0_variant",
        "generating_story_outline",
        "generating_character_biographies",
        "generating_relationship_logic",
    ]
    content: NonEmptyText | None = None
    selected_l0_variant: NonEmptyText | None = None
    selection_rationale: NonEmptyText | None = None

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


class ScriptWriterResult(StrictModel):
    stage: Literal["generating_episode_scripts"]
    content: NonEmptyText


class QualityReviewerResult(StrictModel):
    stage: Literal["accepting_l0", "accepting_l4"]
    passed: Literal[True]
    evidence: NonEmptyText
    feedback_handling: list[FeedbackHandlingItem] = Field(default_factory=list)


class AgentProtocolError(RuntimeError):
    def __init__(self, message: str, *, stage: InternalStage | None = None) -> None:
        super().__init__(message)
        self.stage = stage


class CheckpointUnavailableError(RuntimeError):
    """The durable thread state required for a resumed run is missing."""


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


def _validated_stage_payload(
    stage: InternalStage,
    content: str,
) -> Mapping[str, Any]:
    try:
        raw = json.loads(content)
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
    return parsed.model_dump(mode="json")


class StageGuardMiddleware(AgentMiddleware):
    def __init__(self, before_stage: StageHook, approve_stage: CheckpointHook) -> None:
        self.before_stage = before_stage
        self.approve_stage = approve_stage

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

        await self.before_stage(stage)
        result = await handler(request)
        message = _tool_message(result)
        if not isinstance(message.content, str):
            raise AgentProtocolError("Subagent result was not JSON text", stage=stage)
        payload = _validated_stage_payload(stage, message.content)
        await self.approve_stage(stage, payload)
        return result


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
        feedback: str | None = None,
        retrieve_references: ReferenceRetriever | None = None,
    ) -> WorkflowResult:
        files = {
            path: {"content": content, "encoding": "utf-8"}
            for path, content in persona_files.items()
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

        tools = []
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

        no_retry = {"handle_errors": False}
        subagents = [
            {
                "name": "story_architect",
                "description": (
                    "Selects L0 and creates story outline, character biographies, "
                    "and relationship logic as separate structured tasks."
                ),
                "system_prompt": (
                    "Read the relevant /persona context. Return only the structured "
                    "result for the stage named in the task."
                ),
                "model": self.model,
                "tools": tools,
                "permissions": VIRTUAL_FILE_PERMISSIONS,
                "response_format": ToolStrategy(
                    schema=StoryArchitectResult,
                    **no_retry,
                ),
            },
            {
                "name": "episode_planner",
                "description": "Creates the complete episode outline.",
                "system_prompt": (
                    "Use approved upstream artifacts and persona rules. Return only "
                    "the structured episode-outline result."
                ),
                "model": self.model,
                "tools": tools,
                "permissions": VIRTUAL_FILE_PERMISSIONS,
                "response_format": ToolStrategy(
                    schema=EpisodePlannerResult,
                    **no_retry,
                ),
            },
            {
                "name": "script_writer",
                "description": "Creates the complete episode scripts.",
                "system_prompt": (
                    "Use the approved episode outline and persona rules. Return only "
                    "the structured episode-script result."
                ),
                "model": self.model,
                "tools": tools,
                "permissions": VIRTUAL_FILE_PERMISSIONS,
                "response_format": ToolStrategy(
                    schema=ScriptWriterResult,
                    **no_retry,
                ),
            },
            {
                "name": "quality_reviewer",
                "description": (
                    "Reviews the L0 and L4 gates and itemizes revision-feedback coverage."
                ),
                "system_prompt": (
                    "Review only the named gate. A passing result requires concrete "
                    "evidence. For a revision, itemize feedback handling during L4 review."
                ),
                "model": self.model,
                "tools": tools,
                "permissions": VIRTUAL_FILE_PERMISSIONS,
                "response_format": ToolStrategy(
                    schema=QualityReviewerResult,
                    **no_retry,
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
            middleware=[StageGuardMiddleware(before_stage, approve_stage)],
            subagents=subagents,
            permissions=VIRTUAL_FILE_PERMISSIONS,
            backend=StateBackend(),
            response_format=ToolStrategy(schema=WorkflowResult, handle_errors=False),
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
        return WorkflowResult.model_validate(structured)


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
`[stage=<stage_name>]`. Do not delegate an already approved stage. Do not use
any subagent other than the four listed above. Treat /persona as read-only and
/workspace as temporary thread scratch. Never claim a gate passed without the
quality_reviewer evidence. After all stages are complete, return WorkflowResult
using the approved artifacts. Do not return partial content.
"""
