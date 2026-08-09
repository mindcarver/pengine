import asyncio
import logging
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from uuid import UUID, uuid4

import anthropic
import httpx
import openai
import pytest
from langgraph.errors import GraphRecursionError
from persona_factory import create_persona_package

from pengine.agents import AgentProtocolError, EpisodeTimeoutError, QualityGateRejectedError
from pengine.config import Settings
from pengine.errors import DomainError
from pengine.personas import PersonaCatalog
from pengine.relay import PreflightBlockedError, RelayError
from pengine.repository import Repository
from pengine.schemas import (
    ContentPackage,
    CreateCreationRequest,
    EpisodePlan,
    FeedbackHandlingItem,
    GateResult,
    InternalStage,
    RevisionRequest,
    WorkflowResult,
)
from pengine.worker import Worker


class DeterministicWorkflow:
    def __init__(self, episode_count: int = 2) -> None:
        self.episode_count = episode_count

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
        episode_drafts=None,
        before_episode=None,
        commit_episode=None,
        assemble_episode_scripts=None,
        episode_timeout_seconds=None,
        reset_episode_deadline=None,
        output_language=None,
        feedback=None,
        retrieve_references=None,
        series_bible=None,
        register_series_review=None,
        get_series_bible=None,
    ) -> WorkflowResult:
        del (
            thread_id,
            story,
            requirements,
            persona_files,
            episode_timeout_seconds,
            output_language,
            retrieve_references,
            series_bible,
            register_series_review,
            get_series_bible,
        )
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
            InternalStage.GENERATING_CHARACTER_RELATIONSHIPS: {
                "stage": "generating_character_relationships",
                "character_biographies": "人物小传",
                "relationship_logic": "关系逻辑",
            },
            InternalStage.GENERATING_EPISODE_OUTLINE: {
                "stage": "generating_episode_outline",
                "content": "分集大纲",
                "episode_count": self.episode_count,
                "episodes": [
                    {
                        "episode_number": episode_number,
                        "plan": f"第{episode_number}集计划",
                    }
                    for episode_number in range(1, self.episode_count + 1)
                ],
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
            if stage is InternalStage.GENERATING_EPISODE_SCRIPTS:
                continue
            if stage not in approved:
                await before_stage(stage)
                await approve_stage(stage, payload)
                approved[stage] = payload
            if stage is InternalStage.GENERATING_EPISODE_OUTLINE:
                break

        if InternalStage.GENERATING_EPISODE_SCRIPTS not in approved:
            assert before_episode is not None
            assert commit_episode is not None
            assert assemble_episode_scripts is not None
            committed = {draft.episode_number for draft in episode_drafts or []}
            for episode_number in range(1, self.episode_count + 1):
                if episode_number in committed:
                    continue
                if reset_episode_deadline is not None:
                    await reset_episode_deadline()
                await before_episode(EpisodePlan(episode_number=episode_number, plan="测试计划"))
                await commit_episode(episode_number, f"第{episode_number}集剧本")
            aggregate = await assemble_episode_scripts()
            payload = {
                "stage": "generating_episode_scripts",
                "content": aggregate,
            }
            await approve_stage(InternalStage.GENERATING_EPISODE_SCRIPTS, payload)
            approved[InternalStage.GENERATING_EPISODE_SCRIPTS] = payload

        for stage, payload in payloads.items():
            if stage in approved or stage in {
                InternalStage.GENERATING_EPISODE_OUTLINE,
                InternalStage.GENERATING_EPISODE_SCRIPTS,
            }:
                continue
            await before_stage(stage)
            await approve_stage(stage, payload)
            approved[stage] = payload
        aggregate = (
            await assemble_episode_scripts() if assemble_episode_scripts is not None else "分集剧本"
        )
        return WorkflowResult(
            content_package=ContentPackage(
                story_outline="故事大纲",
                character_biographies="人物小传",
                relationship_logic="关系逻辑",
                episode_outline="分集大纲",
                episode_scripts=aggregate,
            ),
            selected_l0_variant="主动选择",
            selection_rationale="符合测试故事",
            l0_gate=GateResult(passed=True, evidence="L0 evidence"),
            l4_gate=GateResult(passed=True, evidence="L4 evidence"),
            feedback_handling=handling,
        )


class ProviderFailureWorkflow(DeterministicWorkflow):
    def __init__(self, failure_kind: str = "httpx") -> None:
        super().__init__()
        self.calls = 0
        self.failure_kind = failure_kind

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
        request = httpx.Request(
            "POST",
            "https://secret-relay.example/chat/completions?api_key=SECRET-API-KEY",
        )
        detail = "vendor-body SECRET-STORY-CONTENT SECRET-GENERATED-CONTENT"
        if self.failure_kind in {"openai_connection", "openai_remote_protocol"}:
            try:
                if self.failure_kind == "openai_remote_protocol":
                    raise httpx.RemoteProtocolError(detail, request=request)
                raise httpx.ReadTimeout(detail, request=request)
            except (httpx.ReadTimeout, httpx.RemoteProtocolError) as cause:
                raise openai.APIConnectionError(message=detail, request=request) from cause
        if self.failure_kind == "openai_status":
            response = httpx.Response(
                503,
                headers={"retry-after": "0"},
                request=request,
            )
            raise openai.APIStatusError(
                detail,
                response=response,
                body={"detail": detail},
            )
        raise httpx.ReadTimeout(detail, request=request)


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
        super().__init__()
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


class TimeoutAfterFirstEpisodeWorkflow(DeterministicWorkflow):
    def __init__(
        self,
        *,
        episode_count: int = 2,
        timeout_after_episode: int = 1,
        relay_interruption: bool = False,
    ) -> None:
        super().__init__(episode_count)
        if timeout_after_episode >= episode_count:
            raise ValueError("A timeout must leave one episode unfinished")
        self.calls = 0
        self.timeout_after_episode = timeout_after_episode
        self.relay_interruption = relay_interruption
        self.writer_commits: list[int] = []
        self.retry_drafts: list[int] = []
        self.events: list[str] = []

    async def execute(self, **kwargs: Any) -> WorkflowResult:
        self.calls += 1
        if self.calls > 1:
            self.retry_drafts = [draft.episode_number for draft in kwargs["episode_drafts"]]
        commit_episode = kwargs["commit_episode"]
        approve_stage = kwargs["approve_stage"]
        assert commit_episode is not None

        async def commit_then_timeout(episode_number: int, content: str, *args, **kwargs):
            draft = await commit_episode(episode_number, content, *args, **kwargs)
            self.writer_commits.append(episode_number)
            self.events.append(f"commit:{episode_number}")
            if self.calls == 1 and episode_number == self.timeout_after_episode:
                if self.relay_interruption:
                    raise httpx.ReadTimeout("relay read timed out")
                raise EpisodeTimeoutError(episode_number + 1)
            return draft

        async def capture_approval(stage: InternalStage, payload: dict[str, Any]) -> None:
            self.events.append(f"approve:{stage.value}")
            await approve_stage(stage, payload)

        return await super().execute(
            **{
                **kwargs,
                "approve_stage": capture_approval,
                "commit_episode": commit_then_timeout,
            },
        )


class EpisodeArithmeticErrorOnceWorkflow(DeterministicWorkflow):
    def __init__(self) -> None:
        super().__init__(episode_count=2)
        self.calls = 0
        self.retry_drafts: list[int] = []

    async def execute(self, **kwargs: Any) -> WorkflowResult:
        self.calls += 1
        if self.calls > 1:
            self.retry_drafts = [draft.episode_number for draft in kwargs["episode_drafts"]]
            return await super().execute(**kwargs)

        before_episode = kwargs["before_episode"]
        assert before_episode is not None

        async def fail_on_second_episode(plan: EpisodePlan) -> int:
            attempt = await before_episode(plan)
            if plan.episode_number == 2:
                raise ValueError("Operands must be decimal numbers")
            return attempt

        return await super().execute(
            **{
                **kwargs,
                "before_episode": fail_on_second_episode,
            }
        )


class EpisodeConnectionErrorWorkflow(DeterministicWorkflow):
    def __init__(self, provider: str = "anthropic") -> None:
        super().__init__()
        self.provider = provider

    async def execute(self, **kwargs: Any) -> WorkflowResult:
        before_episode = kwargs["before_episode"]
        assert before_episode is not None

        async def fail_on_second_episode(plan: EpisodePlan) -> int:
            attempt = await before_episode(plan)
            if plan.episode_number == 2:
                request = httpx.Request(
                    "POST",
                    "https://secret-relay.example/messages?api_key=SECRET-KEY",
                )
                try:
                    raise httpx.RemoteProtocolError("SECRET-UPSTREAM-DETAIL", request=request)
                except httpx.RemoteProtocolError as cause:
                    if self.provider == "openai":
                        raise openai.APIConnectionError(request=request) from cause
                    raise anthropic.APIConnectionError(request=request) from cause
            return attempt

        return await super().execute(
            **{
                **kwargs,
                "before_episode": fail_on_second_episode,
            }
        )


class PreAttemptEpisodeProtocolErrorWorkflow(DeterministicWorkflow):
    async def execute(self, **kwargs: Any) -> WorkflowResult:
        async def fail_before_attempt(_: EpisodePlan) -> int:
            raise AgentProtocolError(
                "Episode outline cannot start writing",
                stage=InternalStage.GENERATING_EPISODE_SCRIPTS,
            )

        return await super().execute(
            **{
                **kwargs,
                "before_episode": fail_before_attempt,
            }
        )


class QualityRejectedThenPassedWorkflow(DeterministicWorkflow):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0
        self.retry_approved: set[InternalStage] = set()
        self.retry_stages: list[InternalStage] = []

    async def execute(self, **kwargs: Any) -> WorkflowResult:
        self.calls += 1
        if self.calls == 1:
            approve_stage = kwargs["approve_stage"]

            async def reject_l0(stage: InternalStage, payload: dict[str, Any]) -> None:
                if stage is InternalStage.ACCEPTING_L0:
                    raise QualityGateRejectedError(
                        stage=stage,
                        evidence="L0 审核发现核心冲突。",
                    )
                await approve_stage(stage, payload)

            return await super().execute(**{**kwargs, "approve_stage": reject_l0})

        self.retry_approved = set(kwargs["approved_checkpoints"] or {})
        before_stage = kwargs["before_stage"]

        async def capture_retry_stage(stage: InternalStage) -> int:
            self.retry_stages.append(stage)
            return await before_stage(stage)

        return await super().execute(**{**kwargs, "before_stage": capture_retry_stage})


class PreflightBlockedWorkflow:
    def __init__(self) -> None:
        self.calls = 0

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
        episode_drafts=None,
        before_episode=None,
        commit_episode=None,
        assemble_episode_scripts=None,
        episode_timeout_seconds=None,
        reset_episode_deadline=None,
        output_language=None,
        feedback=None,
        retrieve_references=None,
        series_bible=None,
        register_series_review=None,
        get_series_bible=None,
    ) -> WorkflowResult:
        del (
            thread_id,
            story,
            requirements,
            persona_files,
            episode_drafts,
            before_episode,
            commit_episode,
            assemble_episode_scripts,
            episode_timeout_seconds,
            reset_episode_deadline,
            output_language,
            feedback,
            retrieve_references,
            series_bible,
            register_series_review,
            get_series_bible,
        )
        self.calls += 1
        approved = dict(approved_checkpoints or {})
        if InternalStage.SELECTING_L0_VARIANT not in approved:
            await before_stage(InternalStage.SELECTING_L0_VARIANT)
            await approve_stage(
                InternalStage.SELECTING_L0_VARIANT,
                {
                    "stage": "selecting_l0_variant",
                    "selected_l0_variant": "主动选择",
                    "selection_rationale": "符合测试故事",
                },
            )
        await before_stage(InternalStage.GENERATING_STORY_OUTLINE)
        raise PreflightBlockedError(
            role="generation",
            model_id="claude-opus-5",
            stage="generating_story_outline",
            episode_number=None,
            required_tokens=2_000_000,
            verified_limit_tokens=200_000,
        )


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
async def test_worker_start_injects_distinct_generation_and_review_routes(
    tmp_path: Path,
    monkeypatch,
) -> None:
    settings, catalog, repository, _ = await _services(tmp_path)
    settings = Settings(
        _env_file=None,
        persona_root=settings.persona_root,
        data_dir=settings.data_dir,
        relay_base_url="https://relay.example/v1",
        relay_api_key="secret-value",
        generation_model_id="claude-opus-5",
        review_model_id="deepseek-v4-flash",
    )
    generation_model = object()
    review_model = object()
    captured: dict[str, Any] = {}

    def fake_build_relay_routes(_: Settings) -> SimpleNamespace:
        return SimpleNamespace(
            generation=SimpleNamespace(
                model=generation_model,
                provider_profile_key="anthropic-generation",
            ),
            review=SimpleNamespace(
                model=review_model,
                provider_profile_key="deepseek-review",
            ),
        )

    class FakeDeepAgentWorkflow:
        def __init__(self, **kwargs: Any) -> None:
            captured.update(kwargs)

    monkeypatch.setattr("pengine.worker.build_relay_routes", fake_build_relay_routes)
    monkeypatch.setattr("pengine.worker.DeepAgentWorkflow", FakeDeepAgentWorkflow)

    worker = Worker(
        settings=settings,
        repository=repository,
        catalog=catalog,
        worker_id="route-injection-worker",
    )

    try:
        await worker.start()
        assert captured["generation_model"] is generation_model
        assert captured["review_model"] is review_model
        assert captured["generation_model"] is not captured["review_model"]
        assert captured["generation_provider_profile_key"] == "anthropic-generation"
        assert captured["review_provider_profile_key"] == "deepseek-review"
    finally:
        await worker.stop()


@pytest.mark.asyncio
@pytest.mark.parametrize("missing_route", ["generation", "review"])
async def test_worker_start_does_not_build_a_workflow_when_a_relay_route_is_missing(
    tmp_path: Path,
    monkeypatch,
    missing_route: str,
) -> None:
    settings, catalog, repository, _ = await _services(tmp_path)
    settings = Settings(
        _env_file=None,
        persona_root=settings.persona_root,
        data_dir=settings.data_dir,
        relay_base_url="https://relay.example/v1",
        relay_api_key="secret-value",
        generation_model_id=None if missing_route == "generation" else "claude-opus-5",
        review_model_id=None if missing_route == "review" else "deepseek-v4-flash",
    )
    routes_built = False
    workflow_built = False

    def fake_build_relay_routes(_: Settings) -> None:
        nonlocal routes_built
        routes_built = True

    class FakeDeepAgentWorkflow:
        def __init__(self, **_: Any) -> None:
            nonlocal workflow_built
            workflow_built = True

    monkeypatch.setattr("pengine.worker.build_relay_routes", fake_build_relay_routes)
    monkeypatch.setattr("pengine.worker.DeepAgentWorkflow", FakeDeepAgentWorkflow)

    worker = Worker(
        settings=settings,
        repository=repository,
        catalog=catalog,
        worker_id="route-missing-worker",
    )

    try:
        await worker.start()
        assert routes_built is False
        assert workflow_built is False
        assert worker.workflow is None
    finally:
        await worker.stop()


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
    assert (
        initial_resource.initial.result.delivery_report.ownership_statement
        == "最终创作所有权与判断由内部操作人员保留。"
    )
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
        "generating_character_relationships",
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
async def test_worker_resumes_the_first_unfinished_episode_without_rewriting_prior_drafts(
    tmp_path: Path,
) -> None:
    settings, catalog, repository, snapshot = await _services(tmp_path)
    accepted = await repository.create_creation(
        "episode-timeout-recovery",
        CreateCreationRequest(
            persona_id="test-persona",
            story="一个人回乡。",
            requirements="生成完整短剧。",
        ),
        snapshot.summary,
    )
    workflow = TimeoutAfterFirstEpisodeWorkflow()
    worker = Worker(
        settings=settings,
        repository=repository,
        catalog=catalog,
        workflow=workflow,
        worker_id="episode-timeout-recovery-worker",
    )

    assert await worker.run_once() is True
    recovering = await repository.get_creation(accepted.creation_id)
    assert recovering is not None
    assert recovering.initial.state == "auto_resuming"
    assert recovering.initial.progress.episodes.model_dump() == {
        "total": 2,
        "completed": 1,
        "current": 2,
    }
    recovered_episodes = [
        (draft.episode_number, draft.content) for draft in recovering.initial.drafts.episodes
    ]
    assert recovered_episodes == [(1, "第1集剧本")]
    assert workflow.writer_commits == [1]

    assert await worker.run_once() is True
    completed = await repository.get_creation(accepted.creation_id)
    assert completed is not None
    assert completed.initial.state == "succeeded"
    assert workflow.retry_drafts == [1]
    assert workflow.writer_commits == [1, 2]
    async with repository._connection() as connection:
        row = await (
            await connection.execute(
                "SELECT id FROM runs WHERE creation_id = ? AND kind = 'initial'",
                (str(accepted.creation_id),),
            )
        ).fetchone()
    assert row is not None
    assert await repository.get_episode_attempt_counts(UUID(row["id"])) == {1: 1, 2: 1}


@pytest.mark.asyncio
async def test_worker_resumes_the_first_unfinished_episode_after_a_relay_interruption(
    tmp_path: Path,
) -> None:
    settings, catalog, repository, snapshot = await _services(tmp_path)
    accepted = await repository.create_creation(
        "episode-relay-recovery",
        CreateCreationRequest(
            persona_id="test-persona",
            story="一个人回乡。",
            requirements="生成完整短剧。",
        ),
        snapshot.summary,
    )
    workflow = TimeoutAfterFirstEpisodeWorkflow(relay_interruption=True)
    worker = Worker(
        settings=settings,
        repository=repository,
        catalog=catalog,
        workflow=workflow,
        worker_id="episode-relay-recovery-worker",
    )

    assert await worker.run_once() is True
    recovering = await repository.get_creation(accepted.creation_id)
    assert recovering is not None
    assert recovering.initial.state == "auto_resuming"
    assert recovering.initial.progress.recovery_reason == "relay_interruption"
    assert [draft.episode_number for draft in recovering.initial.drafts.episodes] == [1]

    resumed = await repository.lease_next_job(
        "episode-relay-recovery-worker-2",
        30,
        now=datetime.now(UTC) + timedelta(seconds=11),
    )
    assert resumed is not None
    await worker._process_job(resumed)
    completed = await repository.get_creation(accepted.creation_id)
    assert completed is not None
    assert completed.initial.state == "succeeded"
    assert workflow.retry_drafts == [1]
    assert workflow.writer_commits == [1, 2]


@pytest.mark.asyncio
async def test_worker_pauses_arithmetic_error_and_resumes_only_failed_episode(
    tmp_path: Path,
) -> None:
    settings, catalog, repository, snapshot = await _services(tmp_path)
    accepted = await repository.create_creation(
        "episode-arithmetic-recovery",
        CreateCreationRequest(
            persona_id="test-persona",
            story="一个人回乡。",
            requirements="生成完整短剧。",
        ),
        snapshot.summary,
    )
    workflow = EpisodeArithmeticErrorOnceWorkflow()
    worker = Worker(
        settings=settings,
        repository=repository,
        catalog=catalog,
        workflow=workflow,
        worker_id="episode-arithmetic-recovery-worker",
    )

    assert await worker.run_once() is True
    paused = await repository.get_creation(accepted.creation_id)
    assert paused is not None
    assert paused.initial.state == "paused"
    assert paused.initial.pause.code == "episode_error"
    assert paused.initial.pause.episode_number == 2
    assert "非十进制参数" in paused.initial.pause.message
    assert [draft.episode_number for draft in paused.initial.drafts.episodes] == [1]

    await repository.continue_run(
        creation_id=accepted.creation_id,
        run_kind="initial",
        idempotency_key="continue-arithmetic-episode",
    )
    assert await worker.run_once() is True
    completed = await repository.get_creation(accepted.creation_id)
    assert completed is not None
    assert completed.initial.state == "succeeded"
    assert workflow.retry_drafts == [1]
    async with repository._connection() as connection:
        row = await (
            await connection.execute(
                "SELECT id FROM runs WHERE creation_id = ? AND kind = 'initial'",
                (str(accepted.creation_id),),
            )
        ).fetchone()
    assert row is not None
    assert await repository.get_episode_attempt_counts(UUID(row["id"])) == {1: 1, 2: 2}


@pytest.mark.asyncio
async def test_worker_fails_closed_when_episode_error_precedes_writer_attempt(
    tmp_path: Path,
) -> None:
    settings, catalog, repository, snapshot = await _services(tmp_path)
    accepted = await repository.create_creation(
        "pre-attempt-episode-error",
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
        workflow=PreAttemptEpisodeProtocolErrorWorkflow(),
        worker_id="pre-attempt-episode-error-worker",
    )

    assert await worker.run_once() is True
    failed = await repository.get_creation(accepted.creation_id)

    assert failed is not None
    assert failed.initial.state == "failed"
    assert failed.initial.failure.code == "structured_output_invalid"
    assert failed.initial.failure.message == "模型未返回有效的结构化结果。"
    assert failed.initial.progress.can_continue is False
    assert failed.initial.progress.can_end is False


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", ["anthropic", "openai"])
async def test_episode_connection_error_auto_resumes_once_then_pauses_safely(
    tmp_path: Path,
    provider: str,
) -> None:
    settings, catalog, repository, snapshot = await _services(tmp_path)
    accepted = await repository.create_creation(
        "episode-connection-error",
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
        workflow=EpisodeConnectionErrorWorkflow(provider),
        worker_id="episode-connection-error-worker",
    )

    assert await worker.run_once() is True
    recovering = await repository.get_creation(accepted.creation_id)

    assert recovering is not None
    assert recovering.initial.state == "auto_resuming"
    assert recovering.initial.progress.recovery_reason == "relay_interruption"
    assert [draft.episode_number for draft in recovering.initial.drafts.episodes] == [1]

    async with repository._connection() as connection:
        run = await (
            await connection.execute(
                "SELECT id FROM runs WHERE creation_id = ? AND kind = 'initial'",
                (str(accepted.creation_id),),
            )
        ).fetchone()
    assert run is not None
    resumed = await repository.lease_next_job(
        "episode-connection-recovery-worker",
        30,
        now=datetime.now(UTC) + timedelta(seconds=11),
    )
    assert resumed is not None
    assert str(resumed.run_id) == run["id"]
    await worker._process_job(resumed)
    paused = await repository.get_creation(accepted.creation_id)

    assert paused is not None
    assert paused.initial.state == "paused"
    assert paused.initial.pause.code == "relay_interruption"
    assert paused.initial.pause.episode_number == 2
    assert "interrupted twice" in paused.initial.pause.message
    assert "SECRET" not in paused.initial.pause.message
    assert "secret-relay.example" not in paused.initial.pause.message
    assert [draft.episode_number for draft in paused.initial.drafts.episodes] == [1]


@pytest.mark.asyncio
async def test_ten_episode_run_resumes_without_a_writer_call_for_committed_drafts(
    tmp_path: Path,
) -> None:
    settings, catalog, repository, snapshot = await _services(tmp_path)
    accepted = await repository.create_creation(
        "ten-episode-recovery",
        CreateCreationRequest(
            persona_id="test-persona",
            story="一个人回乡。",
            requirements="生成完整十集短剧。",
        ),
        snapshot.summary,
    )
    workflow = TimeoutAfterFirstEpisodeWorkflow(
        episode_count=10,
        timeout_after_episode=5,
    )
    worker = Worker(
        settings=settings,
        repository=repository,
        catalog=catalog,
        workflow=workflow,
        worker_id="ten-episode-recovery-worker",
    )

    assert await worker.run_once() is True
    recovering = await repository.get_creation(accepted.creation_id)
    assert recovering is not None
    assert recovering.initial.state == "auto_resuming"
    assert recovering.initial.progress.episodes.model_dump() == {
        "total": 10,
        "completed": 5,
        "current": 6,
    }
    assert workflow.writer_commits == [1, 2, 3, 4, 5]

    assert await worker.run_once() is True
    completed = await repository.get_creation(accepted.creation_id)
    assert completed is not None
    assert completed.initial.state == "succeeded"
    assert workflow.retry_drafts == [1, 2, 3, 4, 5]
    assert workflow.writer_commits == list(range(1, 11))
    assert workflow.events.index("approve:accepting_l0") > workflow.events.index("commit:10")
    assert workflow.events.index("approve:accepting_l4") > workflow.events.index("commit:10")

    async with repository._connection() as connection:
        row = await (
            await connection.execute(
                "SELECT id FROM runs WHERE creation_id = ? AND kind = 'initial'",
                (str(accepted.creation_id),),
            )
        ).fetchone()
    assert row is not None
    run_id = UUID(row["id"])
    assert [draft.episode_number for draft in await repository.get_episode_drafts(run_id)] == list(
        range(1, 11)
    )
    assert await repository.get_episode_attempt_counts(run_id) == {
        episode_number: 1 for episode_number in range(1, 11)
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("error", "expected_code", "expected_state"),
    [
        (GraphRecursionError("recursion exhausted"), "graph_recursion_limit", "failed"),
        (
            httpx.RemoteProtocolError("relay protocol mismatch"),
            "relay_interruption",
            "auto_resuming",
        ),
        (
            RelayError(
                code="relay_incompatible",
                safe_message="The model relay does not support the required tool protocol.",
            ),
            "relay_incompatible",
            "failed",
        ),
        (
            QualityGateRejectedError(
                stage=InternalStage.ACCEPTING_L0,
                evidence="L0 与已批准稿件的核心冲突。",
            ),
            "quality_gate_rejected",
            "quality_rejected",
        ),
    ],
)
async def test_worker_reports_graph_and_quality_failures_separately(
    tmp_path: Path,
    error: Exception,
    expected_code: str,
    expected_state: str,
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
    assert resource.initial.state == expected_state
    if expected_state == "failed":
        assert resource.initial.failure.code == expected_code
        assert resource.initial.progress.can_continue is False
        assert resource.initial.progress.can_end is False
    elif expected_state == "auto_resuming":
        assert resource.initial.progress.recovery_reason == expected_code
    else:
        assert resource.initial.quality_rejection.code == expected_code
        assert resource.initial.quality_rejection.evidence == "L0 与已批准稿件的核心冲突。"


@pytest.mark.asyncio
async def test_quality_rejection_retries_only_missing_final_gates_on_the_same_run(
    tmp_path: Path,
) -> None:
    settings, catalog, repository, snapshot = await _services(tmp_path)
    accepted = await repository.create_creation(
        "quality-retry-create",
        CreateCreationRequest(
            persona_id="test-persona",
            story="一个人回乡。",
            requirements="生成完整短剧。",
        ),
        snapshot.summary,
    )
    workflow = QualityRejectedThenPassedWorkflow()
    worker = Worker(
        settings=settings,
        repository=repository,
        catalog=catalog,
        workflow=workflow,
        worker_id="quality-retry-worker",
    )

    assert await worker.run_once() is True
    rejected = await repository.get_creation(accepted.creation_id)
    assert rejected is not None
    assert rejected.initial.state == "quality_rejected"
    assert rejected.initial.quality_rejection.model_dump() == {
        "code": "quality_gate_rejected",
        "stage": "accepting_l0",
        "evidence": "L0 审核发现核心冲突。",
        "attempt_count": 1,
        "can_retry": True,
    }
    first_retry = await repository.retry_final_review(
        creation_id=accepted.creation_id,
        run_kind="initial",
        idempotency_key="quality-retry-review",
    )
    replay_retry = await repository.retry_final_review(
        creation_id=accepted.creation_id,
        run_kind="initial",
        idempotency_key="quality-retry-review",
    )
    with pytest.raises(DomainError) as duplicate_retry:
        await repository.retry_final_review(
            creation_id=accepted.creation_id,
            run_kind="initial",
            idempotency_key="quality-retry-review-again",
        )
    assert first_retry == replay_retry
    assert first_retry.run_state == "queued"
    assert duplicate_retry.value.code == "run_not_controllable"

    assert await worker.run_once() is True
    completed = await repository.get_creation(accepted.creation_id)
    assert completed is not None
    assert completed.initial.state == "succeeded"
    assert workflow.retry_approved == {
        InternalStage.LOADING_PERSONA,
        InternalStage.SELECTING_L0_VARIANT,
        InternalStage.GENERATING_STORY_OUTLINE,
        InternalStage.GENERATING_CHARACTER_RELATIONSHIPS,
        InternalStage.GENERATING_EPISODE_OUTLINE,
        InternalStage.GENERATING_EPISODE_SCRIPTS,
    }
    assert workflow.retry_stages == [InternalStage.ACCEPTING_L0, InternalStage.ACCEPTING_L4]


@pytest.mark.asyncio
async def test_worker_pauses_on_context_preflight_block_without_losing_prior_work(
    tmp_path: Path,
) -> None:
    settings, catalog, repository, snapshot = await _services(tmp_path)
    accepted = await repository.create_creation(
        "context-budget-pause",
        CreateCreationRequest(
            persona_id="test-persona",
            story="一个离乡的人回家处理旧屋。",
            requirements="创作一部当代短剧。",
        ),
        snapshot.summary,
    )
    workflow = PreflightBlockedWorkflow()
    worker = Worker(
        settings=settings,
        repository=repository,
        catalog=catalog,
        workflow=workflow,
        worker_id="context-budget-pause-worker",
    )

    assert await worker.run_once() is True
    resource = await repository.get_creation(accepted.creation_id)

    assert resource is not None
    assert resource.initial.state == "paused"
    assert resource.initial.pause.code == "context_budget"
    assert "verified context limit" in resource.initial.pause.message
    assert resource.initial.pause.stage.value == "generating_story_outline"
    assert resource.initial.progress.recovery_reason == "context_budget"
    # Prior approved work is unchanged and the run can be continued.
    assert resource.initial.drafts.artifacts[0].selected_l0_variant == "主动选择"
    assert resource.initial.progress.can_continue is True
    assert workflow.calls == 1


@pytest.mark.asyncio
async def test_paused_resource_exposes_blocked_model_call_lineage(tmp_path: Path) -> None:
    """The paused resource and workbench show required tokens, verified limit,
    route/model, and affected stage from the durable model-call envelope."""
    settings, catalog, repository, snapshot = await _services(tmp_path)
    accepted = await repository.create_creation(
        "context-budget-resource",
        CreateCreationRequest(
            persona_id="test-persona",
            story="一个离乡的人回家处理旧屋。",
            requirements="创作一部当代短剧。",
        ),
        snapshot.summary,
    )
    lease = await repository.lease_next_job("cb-worker", 30)
    assert lease is not None
    await repository.mark_run_running(lease.run_id)

    # The audit handler would have written this durable record before raising.
    from pengine.model_calls import (
        ModelCallContext,
        ModelCallStore,
        build_started_record,
    )

    store = ModelCallStore(settings.database_path)
    context = ModelCallContext(
        run_id=str(lease.run_id),
        creation_id=str(accepted.creation_id),
        thread_id=lease.thread_id,
        run_kind="initial",
        stage="generating_story_outline",
    )
    record = build_started_record(
        role="generation",
        adapter="anthropic",
        provider="anthropic",
        model="claude-opus-5",
        context=context,
        estimated_input_tokens=2_000_000,
        estimated_output_tokens=128_000,
        verified_limit_tokens=200_000,
    )
    record.status = "preflight_blocked"
    record.outcome = "blocked"
    record.preflight = "blocked"
    store.upsert(record)
    store.close()

    await repository.pause_context_budget(
        lease.run_id,
        stage=InternalStage.GENERATING_STORY_OUTLINE,
        safe_message=(
            "The generation model request needs about 2128000 tokens (input plus "
            "reserved output), but the verified context limit for claude-opus-5 is "
            "200000. No request was sent; the current run paused without changing "
            "approved work."
        ),
    )
    resource = await repository.get_creation(accepted.creation_id)
    assert resource is not None
    assert resource.initial.state == "paused"
    assert resource.initial.pause.code == "context_budget"
    assert resource.initial.pause.stage.value == "generating_story_outline"

    calls = resource.initial.progress.model_calls
    assert len(calls) == 1
    blocked = calls[0]
    assert blocked.status == "preflight_blocked"
    assert blocked.preflight == "blocked"
    assert blocked.model == "claude-opus-5"
    assert blocked.stage == "generating_story_outline"
    assert blocked.estimated_total_tokens == 2_128_000
    assert blocked.verified_limit_tokens == 200_000
    assert blocked.usage.status == "unavailable"
    assert blocked.usage.input_tokens is None


@pytest.mark.asyncio
async def test_worker_requires_exact_successful_physical_provenance(
    tmp_path: Path,
) -> None:
    """A different operation's latest call must never be guessed as the producer."""
    settings, catalog, repository, _snapshot = await _services(tmp_path)
    from pengine.model_calls import ModelCallContext, ModelCallStore, build_started_record

    run_id = uuid4()
    store = ModelCallStore(settings.database_path)
    record = build_started_record(
        call_id="physical-generation-1",
        role="generation",
        adapter="anthropic",
        provider="anthropic",
        model="claude-opus-5",
        context=ModelCallContext(
            run_id=str(run_id),
            stage="generating_episode_scripts",
            episode_number=3,
            operation_id="episode-3-operation",
        ),
        estimated_input_tokens=10,
        estimated_output_tokens=20,
        verified_limit_tokens=200_000,
    )
    record.status = "succeeded"
    record.outcome = "success"
    store.upsert(record)
    store.close()
    worker = Worker(
        settings=settings,
        repository=repository,
        catalog=catalog,
        workflow=DeterministicWorkflow(),
    )

    assert (
        await worker._require_physical_call_id(
            run_id=run_id,
            role="generation",
            stage=InternalStage.GENERATING_EPISODE_SCRIPTS,
            episode_number=3,
            operation_id="episode-3-operation",
        )
        == "physical-generation-1"
    )
    with pytest.raises(AgentProtocolError, match="successful physical generation call"):
        await worker._require_physical_call_id(
            run_id=run_id,
            role="generation",
            stage=InternalStage.GENERATING_EPISODE_SCRIPTS,
            episode_number=3,
            operation_id="different-operation",
        )


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
@pytest.mark.parametrize(
    "failure_kind",
    ["httpx", "openai_connection", "openai_remote_protocol", "openai_status"],
)
async def test_provider_failure_is_safe_and_never_logged(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    failure_kind: str,
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
    workflow = ProviderFailureWorkflow(failure_kind)
    worker = Worker(
        settings=settings,
        repository=repository,
        catalog=catalog,
        workflow=workflow,
        worker_id="provider-failure-test-worker",
    )

    assert await worker.run_once() is True
    resource = await repository.get_creation(accepted.creation_id)

    assert resource is not None
    assert resource.initial.state == "auto_resuming"
    assert resource.initial.progress.recovery_reason == "relay_interruption"
    assert resource.initial.progress.current_stage == "generating_story_outline"
    assert resource.initial.drafts.artifacts[0].selected_l0_variant == "主动选择"
    async with repository._connection() as connection:
        run = await (
            await connection.execute(
                "SELECT id, thread_id FROM runs WHERE creation_id = ? AND kind = 'initial'",
                (str(accepted.creation_id),),
            )
        ).fetchone()
        assert run is not None
        job = await (
            await connection.execute(
                "SELECT available_at FROM jobs WHERE run_id = ?",
                (run["id"],),
            )
        ).fetchone()
    assert job is not None
    assert datetime.fromisoformat(job["available_at"]) > datetime.now(UTC)
    resumed = await repository.lease_next_job(
        "provider-recovery-worker",
        30,
        now=datetime.now(UTC) + timedelta(seconds=11),
    )
    assert resumed is not None
    assert str(resumed.run_id) == run["id"]
    assert resumed.thread_id == run["thread_id"]
    await worker._process_job(resumed)
    completed = await repository.get_creation(accepted.creation_id)
    assert completed is not None
    assert completed.initial.state == "succeeded"
    assert workflow.calls == 2
    for sensitive_value in (
        "SECRET-API-KEY",
        "vendor-body",
        "SECRET-STORY-CONTENT",
        "SECRET-REQUIREMENTS-CONTENT",
        "SECRET-GENERATED-CONTENT",
        "secret-relay.example",
    ):
        assert sensitive_value not in caplog.text
        assert sensitive_value not in resource.model_dump_json()


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
