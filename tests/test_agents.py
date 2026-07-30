from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from persona_factory import create_persona_package
from pydantic import Field

from pengine.agents import (
    VIRTUAL_FILE_PERMISSIONS,
    AgentProtocolError,
    DeepAgentWorkflow,
    QualityReviewerResult,
    StoryArchitectResult,
    WorkflowCompletion,
    _supervisor_prompt,
)
from pengine.config import Settings
from pengine.personas import PersonaCatalog
from pengine.repository import Repository
from pengine.schemas import CreateCreationRequest, InternalStage
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


def _successful_responses() -> list[AIMessage]:
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
            {"stage": "generating_episode_outline", "content": "分集大纲"},
        ),
        (
            "generating_episode_scripts",
            "script_writer",
            "ScriptWriterResult",
            {"stage": "generating_episode_scripts", "content": "分集剧本"},
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
    responses.append(
        _tool_call(
            "WorkflowCompletion",
            {"completed": True},
            index,
        )
    )
    return responses


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

        result = await workflow.execute(
            thread_id="initial-thread",
            story="一个人回乡面对旧事。",
            requirements="生成完整短剧。",
            persona_files={"/persona/project.md": "只读人格规则"},
            before_stage=before_stage,
            approve_stage=approve_stage,
        )

        assert result.content_package.episode_scripts == "分集剧本"
        assert [event[0] for event in events] == ["before", "approve"] * 8
        assert [event[1] for event in events[::2]] == [
            "selecting_l0_variant",
            "generating_story_outline",
            "generating_character_biographies",
            "generating_relationship_logic",
            "generating_episode_outline",
            "generating_episode_scripts",
            "accepting_l0",
            "accepting_l4",
        ]

        checkpoint = await saver.aget_tuple({"configurable": {"thread_id": "initial-thread"}})
        assert checkpoint is not None

        all_tool_names = {name for snapshot in model.bound_tool_names for name in snapshot}
        assert "execute" not in all_tool_names
        assert "task" in all_tool_names
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
        )

    assert result.content_package.episode_scripts == "分集剧本"
    assert attempted.count(InternalStage.SELECTING_L0_VARIANT) == 1
    assert approved[0] is InternalStage.SELECTING_L0_VARIANT


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
            )

    assert attempted == []


@pytest.mark.asyncio
async def test_failed_quality_gate_is_not_approved(tmp_path: Path) -> None:
    database = tmp_path / "checkpoints.sqlite3"
    responses = _successful_responses()
    responses[13] = _tool_call(
        "QualityReviewerResult",
        {
            "stage": "accepting_l0",
            "passed": False,
            "evidence": "成品没有通过 L0 闸门。",
            "feedback_handling": [],
        },
        13,
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

        with pytest.raises(AgentProtocolError, match="Quality gate did not pass"):
            await workflow.execute(
                thread_id="failed-gate-thread",
                story="故事",
                requirements="要求",
                persona_files={"/persona/project.md": "规则"},
                before_stage=before_stage,
                approve_stage=approve_stage,
            )

    assert approved == [
        InternalStage.SELECTING_L0_VARIANT,
        InternalStage.GENERATING_STORY_OUTLINE,
        InternalStage.GENERATING_CHARACTER_BIOGRAPHIES,
        InternalStage.GENERATING_RELATIONSHIP_LOGIC,
        InternalStage.GENERATING_EPISODE_OUTLINE,
        InternalStage.GENERATING_EPISODE_SCRIPTS,
    ]


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
    assert result.content_package.episode_scripts == "分集剧本"
    assert attempted == expected
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
        )

        checkpoint = await saver.aget_tuple({"configurable": {"thread_id": "restart-thread"}})

    assert checkpoint is not None
    assert result.content_package.episode_scripts == "分集剧本"
    assert first_stage not in resumed_attempts
    assert resumed_attempts == [
        InternalStage.GENERATING_STORY_OUTLINE,
        InternalStage.GENERATING_CHARACTER_BIOGRAPHIES,
        InternalStage.GENERATING_RELATIONSHIP_LOGIC,
        InternalStage.GENERATING_EPISODE_OUTLINE,
        InternalStage.GENERATING_EPISODE_SCRIPTS,
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
    assert (("read", "write"), ("/**",), "deny") in rules
