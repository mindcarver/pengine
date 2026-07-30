import asyncio
import logging
import sqlite3
from pathlib import Path
from typing import Any
from uuid import UUID

import httpx
import pytest
from langgraph.errors import GraphRecursionError
from persona_factory import create_persona_package

from pengine.agents import QualityGateRejectedError
from pengine.config import Settings
from pengine.personas import PersonaCatalog
from pengine.repository import Repository
from pengine.schemas import (
    ContentPackage,
    CreateCreationRequest,
    FeedbackHandlingItem,
    GateResult,
    InternalStage,
    RevisionRequest,
    WorkflowResult,
)
from pengine.worker import Worker


class DeterministicWorkflow:
    async def execute(
        self,
        *,
        thread_id: str,
        story: str,
        requirements: str,
        persona_files,
        before_stage,
        approve_stage,
        approved_checkpoints=None,
        feedback=None,
        retrieve_references=None,
    ) -> WorkflowResult:
        del thread_id, story, requirements, persona_files, retrieve_references
        approved = dict(approved_checkpoints or {})
        handling = (
            [
                FeedbackHandlingItem(
                    feedback_item=feedback,
                    handling="Applied during the full rerun.",
                    result="The final scripts reflect the feedback.",
                )
            ]
            if feedback
            else []
        )
        payloads: dict[InternalStage, dict[str, Any]] = {
            InternalStage.SELECTING_L0_VARIANT: {
                "stage": "selecting_l0_variant",
                "selected_l0_variant": "主动选择",
                "selection_rationale": "符合测试故事",
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
                "content": "分集大纲",
            },
            InternalStage.GENERATING_EPISODE_SCRIPTS: {
                "stage": "generating_episode_scripts",
                "content": "分集剧本",
            },
            InternalStage.ACCEPTING_L0: {
                "stage": "accepting_l0",
                "passed": True,
                "evidence": "L0 evidence",
            },
            InternalStage.ACCEPTING_L4: {
                "stage": "accepting_l4",
                "passed": True,
                "evidence": "L4 evidence",
                "feedback_handling": [item.model_dump(mode="json") for item in handling],
            },
        }
        for stage, payload in payloads.items():
            if stage not in approved:
                await before_stage(stage)
                await approve_stage(stage, payload)
        return WorkflowResult(
            content_package=ContentPackage(
                story_outline="故事大纲",
                character_biographies="人物小传",
                relationship_logic="关系逻辑",
                episode_outline="分集大纲",
                episode_scripts="分集剧本",
            ),
            selected_l0_variant="主动选择",
            selection_rationale="符合测试故事",
            l0_gate=GateResult(passed=True, evidence="L0 evidence"),
            l4_gate=GateResult(passed=True, evidence="L4 evidence"),
            feedback_handling=handling,
        )


class ProviderFailureWorkflow:
    async def execute(self, **_: Any) -> WorkflowResult:
        raise httpx.ReadTimeout(
            "vendor-body SECRET-API-KEY SECRET-STORY-CONTENT SECRET-GENERATED-CONTENT"
        )


class BypassWorkflow:
    async def execute(self, **_: Any) -> WorkflowResult:
        return WorkflowResult(
            content_package=ContentPackage(
                story_outline="未批准故事大纲",
                character_biographies="未批准人物小传",
                relationship_logic="未批准关系逻辑",
                episode_outline="未批准分集大纲",
                episode_scripts="未批准分集剧本",
            ),
            selected_l0_variant="未批准变体",
            selection_rationale="未批准理由",
            l0_gate=GateResult(passed=True, evidence="未批准 L0 证据"),
            l4_gate=GateResult(passed=True, evidence="未批准 L4 证据"),
            feedback_handling=[],
        )


class CheckpointMismatchWorkflow(DeterministicWorkflow):
    async def execute(self, **kwargs: Any) -> WorkflowResult:
        result = await super().execute(**kwargs)
        return result.model_copy(
            update={
                "content_package": result.content_package.model_copy(
                    update={"story_outline": "与批准检查点不同"}
                )
            }
        )


class TimeoutOnceWorkflow(DeterministicWorkflow):
    def __init__(self) -> None:
        self.calls = 0

    async def execute(self, **kwargs: Any) -> WorkflowResult:
        self.calls += 1
        if self.calls > 1:
            return await super().execute(**kwargs)

        approved = dict(kwargs["approved_checkpoints"] or {})
        if InternalStage.SELECTING_L0_VARIANT not in approved:
            await kwargs["before_stage"](InternalStage.SELECTING_L0_VARIANT)
            await kwargs["approve_stage"](
                InternalStage.SELECTING_L0_VARIANT,
                {
                    "stage": "selecting_l0_variant",
                    "selected_l0_variant": "主动选择",
                    "selection_rationale": "符合测试故事",
                },
            )
        await kwargs["before_stage"](InternalStage.GENERATING_STORY_OUTLINE)
        await asyncio.sleep(0.1)
        raise AssertionError("the workflow timeout did not cancel execution")


class RaisingWorkflow:
    def __init__(self, error: Exception) -> None:
        self.error = error

    async def execute(self, **_: Any) -> WorkflowResult:
        raise self.error


async def _services(tmp_path: Path):
    persona_root = tmp_path / "personas"
    create_persona_package(persona_root / "active")
    settings = Settings(persona_root=persona_root, data_dir=tmp_path / "data")
    catalog = PersonaCatalog(persona_root, settings.snapshot_root)
    repository = Repository(settings.database_path)
    await repository.initialize()
    snapshot = catalog.create_snapshot("test-persona")
    return settings, catalog, repository, snapshot


@pytest.mark.asyncio
async def test_worker_completes_initial_and_one_revision(tmp_path: Path) -> None:
    settings, catalog, repository, snapshot = await _services(tmp_path)
    accepted = await repository.create_creation(
        "initial-key",
        CreateCreationRequest(
            persona_id="test-persona",
            story="一个人回乡。",
            requirements="生成完整短剧。",
        ),
        snapshot.summary,
    )
    worker = Worker(
        settings=settings,
        repository=repository,
        catalog=catalog,
        workflow=DeterministicWorkflow(),
        worker_id="test-worker",
    )

    assert await worker.run_once() is True
    initial_resource = await repository.get_creation(accepted.creation_id)
    assert initial_resource is not None
    assert initial_resource.initial.state == "succeeded"
    assert initial_resource.initial.result.content_package.story_outline == "故事大纲"
    assert initial_resource.revision.state == "available"

    await repository.create_or_retry_revision(
        creation_id=accepted.creation_id,
        idempotency_key="revision-key",
        request=RevisionRequest(feedback="让人物付出更明确的代价。"),
    )
    assert await worker.run_once() is True

    revised_resource = await repository.get_creation(accepted.creation_id)
    assert revised_resource is not None
    assert revised_resource.initial == initial_resource.initial
    assert revised_resource.revision.state == "succeeded"
    assert revised_resource.revision.result.delivery_report.feedback_handling[0].feedback_item == (
        "让人物付出更明确的代价。"
    )
    assert revised_resource.persona.snapshot_sha256 == initial_resource.persona.snapshot_sha256
    assert await worker.run_once() is False


@pytest.mark.asyncio
async def test_worker_auto_resumes_first_wall_clock_timeout_from_approved_checkpoint(
    tmp_path: Path,
) -> None:
    settings, catalog, repository, snapshot = await _services(tmp_path)
    settings = settings.model_copy(update={"run_timeout_seconds": 0.02})
    accepted = await repository.create_creation(
        "timeout-recovery",
        CreateCreationRequest(
            persona_id="test-persona",
            story="一个人回乡。",
            requirements="生成完整短剧。",
        ),
        snapshot.summary,
    )
    workflow = TimeoutOnceWorkflow()
    worker = Worker(
        settings=settings,
        repository=repository,
        catalog=catalog,
        workflow=workflow,
        worker_id="timeout-recovery-worker",
    )

    assert await worker.run_once() is True
    recovering = await repository.get_creation(accepted.creation_id)
    assert recovering is not None
    assert recovering.initial.state == "auto_resuming"
    assert recovering.initial.progress.current_stage == "generating_story_outline"
    assert recovering.initial.progress.completed_stages == ["determining_direction"]
    assert not hasattr(recovering.initial, "result")

    worker.settings = settings.model_copy(update={"run_timeout_seconds": 1.0})
    assert await worker.run_once() is True
    succeeded = await repository.get_creation(accepted.creation_id)
    assert succeeded is not None
    assert succeeded.initial.state == "succeeded"
    assert succeeded.initial.progress.completed_stages == [
        "determining_direction",
        "generating_story_outline",
        "generating_character_biographies",
        "generating_relationships",
        "generating_episode_outline",
        "generating_episode_scripts",
        "final_review",
    ]
    async with repository._connection() as connection:
        row = await (
            await connection.execute(
                "SELECT id FROM runs WHERE creation_id = ? AND kind = 'initial'",
                (str(accepted.creation_id),),
            )
        ).fetchone()
    assert row is not None
    counts = await repository.get_stage_attempt_counts(UUID(row["id"]))
    assert counts[InternalStage.SELECTING_L0_VARIANT] == 1
    assert counts[InternalStage.GENERATING_STORY_OUTLINE] == 2


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("error", "expected_code"),
    [
        (GraphRecursionError("recursion exhausted"), "graph_recursion_limit"),
        (
            QualityGateRejectedError(stage=InternalStage.ACCEPTING_L0),
            "quality_gate_rejected",
        ),
    ],
)
async def test_worker_reports_graph_and_quality_failures_separately(
    tmp_path: Path,
    error: Exception,
    expected_code: str,
) -> None:
    settings, catalog, repository, snapshot = await _services(tmp_path)
    accepted = await repository.create_creation(
        f"failure-{expected_code}",
        CreateCreationRequest(
            persona_id="test-persona",
            story="一个人回乡。",
            requirements="生成完整短剧。",
        ),
        snapshot.summary,
    )
    worker = Worker(
        settings=settings,
        repository=repository,
        catalog=catalog,
        workflow=RaisingWorkflow(error),
        worker_id="failure-classification-worker",
    )

    assert await worker.run_once() is True
    resource = await repository.get_creation(accepted.creation_id)
    assert resource is not None
    assert resource.initial.state == "failed"
    assert resource.initial.failure.code == expected_code


@pytest.mark.asyncio
async def test_background_worker_recovers_after_transient_iteration_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    settings, catalog, repository, snapshot = await _services(tmp_path)
    settings = settings.model_copy(update={"worker_poll_seconds": 0.01})
    accepted = await repository.create_creation(
        "transient-failure",
        CreateCreationRequest(
            persona_id="test-persona",
            story="一个人回乡。",
            requirements="生成完整短剧。",
        ),
        snapshot.summary,
    )
    requeue_expired_jobs = repository.requeue_expired_jobs
    calls = 0

    async def no_startup_reconciliation():
        return []

    async def fail_once_then_requeue(*, now=None):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise sqlite3.OperationalError("SECRET-ITERATION-DETAIL")
        return await requeue_expired_jobs(now=now)

    monkeypatch.setattr(repository, "reconcile_startup", no_startup_reconciliation)
    monkeypatch.setattr(repository, "requeue_expired_jobs", fail_once_then_requeue)
    caplog.set_level(logging.ERROR, logger="pengine.worker")
    worker = Worker(
        settings=settings,
        repository=repository,
        catalog=catalog,
        workflow=DeterministicWorkflow(),
        worker_id="resilient-worker",
    )

    await worker.start()
    deadline = asyncio.get_running_loop().time() + 2.0
    try:
        while asyncio.get_running_loop().time() < deadline:
            resource = await repository.get_creation(accepted.creation_id)
            assert resource is not None
            if resource.initial.state == "succeeded":
                break
            await asyncio.sleep(0.01)
        else:
            raise AssertionError("worker did not recover after transient failure")
    finally:
        await worker.stop()

    assert calls >= 2
    assert "worker iteration failed worker_id=resilient-worker" in caplog.text
    assert "OperationalError" in caplog.text
    assert "SECRET-ITERATION-DETAIL" not in caplog.text


@pytest.mark.asyncio
async def test_missing_relay_fails_safely_without_leaking_content(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.DEBUG, logger="pengine")
    settings, catalog, repository, snapshot = await _services(tmp_path)
    accepted = await repository.create_creation(
        "missing-relay",
        CreateCreationRequest(
            persona_id="test-persona",
            story="SECRET-STORY-CONTENT",
            requirements="SECRET-REQUIREMENTS-CONTENT",
        ),
        snapshot.summary,
    )
    worker = Worker(
        settings=settings,
        repository=repository,
        catalog=catalog,
        worker_id="test-worker",
    )

    assert await worker.run_once() is True
    resource = await repository.get_creation(accepted.creation_id)
    assert resource is not None
    assert resource.initial.state == "failed"
    assert resource.initial.failure.code == "relay_unavailable"
    assert resource.initial.failure.failed_stage is InternalStage.SELECTING_L0_VARIANT
    assert "SECRET-STORY-CONTENT" not in caplog.text
    assert "SECRET-REQUIREMENTS-CONTENT" not in caplog.text


@pytest.mark.asyncio
async def test_worker_uses_domain_database_for_langgraph_checkpoints(
    tmp_path: Path,
) -> None:
    settings, catalog, repository, _ = await _services(tmp_path)
    worker = Worker(
        settings=settings,
        repository=repository,
        catalog=catalog,
        worker_id="checkpoint-table-test-worker",
    )

    await worker.start()
    await worker.stop()

    with sqlite3.connect(settings.database_path) as connection:
        rows = connection.execute(
            """
            SELECT name
            FROM sqlite_master
            WHERE type = 'table'
              AND name IN ('creations', 'checkpoints', 'writes')
            """
        ).fetchall()

    assert {row[0] for row in rows} == {
        "creations",
        "checkpoints",
        "writes",
    }


@pytest.mark.asyncio
async def test_provider_failure_is_safe_and_never_logged(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.DEBUG, logger="pengine")
    settings, catalog, repository, snapshot = await _services(tmp_path)
    accepted = await repository.create_creation(
        "provider-failure",
        CreateCreationRequest(
            persona_id="test-persona",
            story="SECRET-STORY-CONTENT",
            requirements="SECRET-REQUIREMENTS-CONTENT",
        ),
        snapshot.summary,
    )
    worker = Worker(
        settings=settings,
        repository=repository,
        catalog=catalog,
        workflow=ProviderFailureWorkflow(),
        worker_id="provider-failure-test-worker",
    )

    assert await worker.run_once() is True
    resource = await repository.get_creation(accepted.creation_id)

    assert resource is not None
    assert resource.initial.state == "failed"
    assert resource.initial.failure.code == "relay_unavailable"
    assert resource.initial.failure.message == "The model relay request failed."
    for sensitive_value in (
        "SECRET-API-KEY",
        "vendor-body",
        "SECRET-STORY-CONTENT",
        "SECRET-REQUIREMENTS-CONTENT",
        "SECRET-GENERATED-CONTENT",
    ):
        assert sensitive_value not in caplog.text


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "workflow",
    [BypassWorkflow(), CheckpointMismatchWorkflow()],
)
async def test_worker_rejects_results_not_derived_from_approved_checkpoints(
    tmp_path: Path,
    workflow: Any,
) -> None:
    settings, catalog, repository, snapshot = await _services(tmp_path)
    accepted = await repository.create_creation(
        "unapproved-result",
        CreateCreationRequest(
            persona_id="test-persona",
            story="故事",
            requirements="要求",
        ),
        snapshot.summary,
    )
    worker = Worker(
        settings=settings,
        repository=repository,
        catalog=catalog,
        workflow=workflow,
        worker_id="checkpoint-truth-test-worker",
    )

    assert await worker.run_once() is True
    resource = await repository.get_creation(accepted.creation_id)

    assert resource is not None
    assert resource.initial.state == "failed"
    assert resource.initial.failure.code == "structured_output_invalid"
