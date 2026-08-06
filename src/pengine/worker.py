import asyncio
import json
import logging
import sqlite3
from collections.abc import Awaitable, Callable, Mapping
from contextlib import AbstractAsyncContextManager, asynccontextmanager, suppress
from dataclasses import asdict
from typing import Any, Protocol
from uuid import UUID, uuid4

import aiosqlite
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.errors import GraphRecursionError

from pengine.agents import (
    AgentProtocolError,
    CheckpointUnavailableError,
    ContentReviewRejectedError,
    DeepAgentWorkflow,
    EpisodeTimeoutError,
    MilestoneRejectedError,
    QualityGateRejectedError,
)
from pengine.config import Settings
from pengine.continuity import EpisodeLock
from pengine.errors import DomainError
from pengine.language import OutputLanguage
from pengine.model_calls import ModelCallState, ModelCallStore, new_call_id
from pengine.personas import PersonaCatalog, PersonaPackageError
from pengine.relay import (
    PreflightBlockedError,
    RelayError,
    build_relay_routes,
    classify_relay_exception,
    drain_audit_writes,
    is_relay_connection_error,
    is_relay_exception,
    retryable_relay_interruption,
    submit_store_write,
)
from pengine.repository import LeasedJob, Repository, RunWorkItem
from pengine.schemas import (
    Delivery,
    DeliveryReport,
    EpisodeDraft,
    EpisodePlan,
    InternalStage,
    RunFailure,
    WorkflowResult,
)
from pengine.series_bible import (
    GlobalDesignReview,
    SeriesBible,
    SeriesBibleSummary,
    bind_global_design_review,
    build_series_bible,
    detect_genre,
    validate_series_bible,
)
from pengine.series_review import active_prefix_hash

logger = logging.getLogger(__name__)

_SPECIALIST_STAGES = (
    InternalStage.SELECTING_L0_VARIANT,
    InternalStage.GENERATING_STORY_OUTLINE,
    InternalStage.GENERATING_CHARACTER_RELATIONSHIPS,
    InternalStage.GENERATING_EPISODE_OUTLINE,
    InternalStage.GENERATING_EPISODE_SCRIPTS,
    InternalStage.ACCEPTING_L0,
    InternalStage.ACCEPTING_L4,
)
_ALL_STAGES = (
    InternalStage.LOADING_PERSONA,
    *_SPECIALIST_STAGES,
    InternalStage.ASSEMBLING_DELIVERY,
)


def _exception_type_chain(exc: BaseException) -> tuple[str, ...]:
    types: list[str] = []
    candidate: BaseException | None = exc
    seen: set[int] = set()
    while candidate is not None and id(candidate) not in seen:
        seen.add(id(candidate))
        types.append(f"{type(candidate).__module__}.{type(candidate).__name__}")
        candidate = candidate.__cause__ or candidate.__context__
    return tuple(types)


StageHook = Callable[[InternalStage], Awaitable[int]]
CheckpointHook = Callable[[InternalStage, Mapping[str, Any]], Awaitable[None]]
ReferenceRetriever = Callable[[str], Awaitable[str]]
EpisodeAttemptHook = Callable[[EpisodePlan], Awaitable[int]]
EpisodeCommitHook = Callable[[int, str, EpisodeLock | None], Awaitable[EpisodeDraft]]
EpisodeAssemblyHook = Callable[[], Awaitable[str]]
EpisodeDeadlineReset = Callable[[], Awaitable[None]]
SeriesReviewRegistration = Callable[..., Awaitable[str]]


class WorkflowExecutor(Protocol):
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
        output_language: OutputLanguage | None = None,
        feedback: str | None = None,
        retrieve_references: ReferenceRetriever | None = None,
        series_bible: SeriesBibleSummary | None = None,
        register_series_review: SeriesReviewRegistration | None = None,
        get_series_bible: Callable[[], Awaitable[SeriesBibleSummary | None]] | None = None,
    ) -> WorkflowResult: ...


class Worker:
    def __init__(
        self,
        *,
        settings: Settings,
        repository: Repository,
        catalog: PersonaCatalog,
        workflow: WorkflowExecutor | None = None,
        worker_id: str | None = None,
    ) -> None:
        self.settings = settings
        self.repository = repository
        self.catalog = catalog
        self.workflow = workflow
        self.worker_id = worker_id or f"pengine-{uuid4()}"
        self._stop_event = asyncio.Event()
        self._task: asyncio.Task[None] | None = None
        self._saver_context: AbstractAsyncContextManager[AsyncSqliteSaver] | None = None
        self._saver: AsyncSqliteSaver | None = None
        self._model_call_state: ModelCallState | None = None
        self._model_call_store: ModelCallStore | None = None

    async def start(self) -> None:
        if self._task is not None:
            return
        await self.repository.reconcile_startup()
        if self.workflow is None:

            @asynccontextmanager
            async def _saver_conn() -> AbstractAsyncContextManager[AsyncSqliteSaver]:
                connection = await aiosqlite.connect(str(self.settings.database_path), timeout=30)
                await connection.execute("PRAGMA busy_timeout = 30000")
                try:
                    yield AsyncSqliteSaver(connection)
                finally:
                    await connection.close()

            self._saver_context = _saver_conn()
            self._saver = await self._saver_context.__aenter__()
            await self._saver.setup()
            if self.settings.relay_configured:
                routes = build_relay_routes(self.settings)
                state = getattr(routes, "model_call_state", None)
                if state is not None:
                    store = ModelCallStore(self.settings.database_path)
                    state.store = store
                    state.context.reset()
                    self._model_call_state = state
                    self._model_call_store = store
                self.workflow = DeepAgentWorkflow(
                    generation_model=routes.generation.model,
                    review_model=routes.review.model,
                    checkpointer=self._saver,
                    recursion_limit=self.settings.agent_recursion_limit,
                    generation_provider_profile_key=routes.generation.provider_profile_key,
                    review_provider_profile_key=routes.review.provider_profile_key,
                )
        self._stop_event.clear()
        self._task = asyncio.create_task(self._run_loop(), name="pengine-worker")

    async def stop(self) -> None:
        self._stop_event.set()
        if self._task is not None:
            self._task.cancel()
            with suppress(asyncio.CancelledError):
                await self._task
            self._task = None
        await asyncio.to_thread(drain_audit_writes)
        if self._saver_context is not None:
            await self._saver_context.__aexit__(None, None, None)
            self._saver_context = None
            self._saver = None
        if self._model_call_store is not None:
            self._model_call_store.close()
            self._model_call_store = None
        self._model_call_state = None

    async def _run_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                worked = await self.run_once()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.error(
                    "worker iteration failed worker_id=%s error_type=%s",
                    self.worker_id,
                    type(exc).__name__,
                )
                worked = False
            if worked:
                continue
            with suppress(TimeoutError):
                await asyncio.wait_for(
                    self._stop_event.wait(),
                    timeout=self.settings.worker_poll_seconds,
                )

    async def run_once(self) -> bool:
        await self.repository.requeue_expired_jobs()
        job = await self.repository.lease_next_job(
            self.worker_id,
            self.settings.lease_seconds,
        )
        if job is None:
            return False
        await self._process_job(job)
        return True

    async def _process_job(self, job: LeasedJob) -> None:
        await self.repository.mark_run_running(job.run_id)
        work = await self.repository.get_run_work_item(job.run_id)
        approved: dict[InternalStage, Any] = dict(work.business_checkpoints)
        if (
            work.episode_plans
            and len(work.episode_drafts) == len(work.episode_plans)
            and InternalStage.GENERATING_EPISODE_SCRIPTS not in approved
        ):
            aggregate = await self.repository.episode_aggregate_checkpoint_payload(work.run_id)
            payload = {
                "stage": InternalStage.GENERATING_EPISODE_SCRIPTS.value,
                **aggregate,
            }
            await self.repository.approve_checkpoint(
                work.run_id,
                InternalStage.GENERATING_EPISODE_SCRIPTS,
                payload,
            )
            approved[InternalStage.GENERATING_EPISODE_SCRIPTS] = payload
        current_stage = InternalStage.LOADING_PERSONA
        run_timeout_scope: asyncio.Timeout | None = None
        model_call_state = self._model_call_state
        if model_call_state is not None:
            model_call_state.context.reset()
            model_call_state.context.run_id = str(work.run_id)
            model_call_state.context.creation_id = str(work.creation_id)
            model_call_state.context.thread_id = work.thread_id
            model_call_state.context.run_kind = work.run_kind
            model_call_state.context.stage = current_stage.value
        logger.info(
            "workflow run started run_id=%s creation_id=%s kind=%s",
            work.run_id,
            work.creation_id,
            work.run_kind,
        )

        try:
            if InternalStage.LOADING_PERSONA not in approved:
                await self.repository.record_stage_attempt(
                    work.run_id,
                    InternalStage.LOADING_PERSONA,
                )
            snapshot = self.catalog.resolve_snapshot(work.persona.snapshot_sha256)
            if InternalStage.LOADING_PERSONA not in approved:
                loading_payload = snapshot.summary.model_dump(mode="json")
                await self.repository.approve_checkpoint(
                    work.run_id,
                    InternalStage.LOADING_PERSONA,
                    loading_payload,
                )
                approved[InternalStage.LOADING_PERSONA] = loading_payload

            persona_files = self._persona_files(work)

            if InternalStage.GENERATING_EPISODE_OUTLINE in approved and not work.episode_plans:
                current_stage = InternalStage.GENERATING_EPISODE_SCRIPTS
                raise CheckpointUnavailableError(
                    "The approved legacy episode outline has no durable episode plan."
                )

            if InternalStage.GENERATING_EPISODE_OUTLINE in approved:
                # Restart recovery: the outline checkpoint is durable and the
                # approve hook that normally syncs the design only fires during
                # live delegation. If the process died after the outline commit
                # but before candidate promotion, re-run the idempotent sync so
                # the run never proceeds without an active design (SDP-A8).
                await self._sync_series_bible(work, approved)
                if model_call_state is not None:
                    active = await self.repository.get_run_series_bible(work.run_id)
                    model_call_state.context.candidate = (
                        active.candidate_id if active is not None else None
                    )

            if self.workflow is None:
                current_stage = self._next_unapproved(approved)
                if current_stage is InternalStage.GENERATING_EPISODE_SCRIPTS:
                    refreshed = await self.repository.get_run_work_item(work.run_id)
                    await self.repository.record_episode_attempt(
                        work.run_id,
                        len(refreshed.episode_drafts) + 1,
                    )
                else:
                    await self.repository.record_stage_attempt(work.run_id, current_stage)
                raise RelayError(
                    code="relay_unavailable",
                    safe_message="The model relay is not configured.",
                )
            if (
                isinstance(self.workflow, DeepAgentWorkflow)
                and self._requires_langgraph_checkpoint(work)
                and not await self.workflow.has_checkpoint(work.thread_id)
            ):
                raise CheckpointUnavailableError("The durable workflow checkpoint is missing.")

            async def before_stage(stage: InternalStage) -> int:
                nonlocal current_stage
                current_stage = stage
                if model_call_state is not None:
                    model_call_state.context.stage = stage.value
                    model_call_state.context.episode_number = None
                return await self.repository.record_stage_attempt(work.run_id, stage)

            async def approve_stage(
                stage: InternalStage,
                payload: Mapping[str, Any],
            ) -> None:
                await self.repository.approve_checkpoint(work.run_id, stage, payload)
                approved[stage] = dict(payload)
                if stage is InternalStage.GENERATING_EPISODE_OUTLINE:
                    await self._sync_series_bible(work, approved)
                    if model_call_state is not None:
                        active = await self.repository.get_run_series_bible(work.run_id)
                        model_call_state.context.candidate = (
                            active.candidate_id if active is not None else None
                        )

            async def before_episode(plan: EpisodePlan) -> int:
                nonlocal current_stage
                current_stage = InternalStage.GENERATING_EPISODE_SCRIPTS
                if model_call_state is not None:
                    model_call_state.context.stage = InternalStage.GENERATING_EPISODE_SCRIPTS.value
                    model_call_state.context.episode_number = plan.episode_number
                    model_call_state.context.call_id = new_call_id()
                return await self.repository.record_episode_attempt(
                    work.run_id,
                    plan.episode_number,
                )

            async def commit_episode(
                episode_number: int,
                content: str,
                episode_lock: EpisodeLock | None = None,
                *,
                call_id: str | None = None,
                writer_notes: str = "",
            ) -> EpisodeDraft:
                if episode_lock is None:
                    # Legacy outline path without a locked story contract keeps the
                    # immutable per-episode draft behavior unchanged.
                    return await self.repository.commit_episode_draft(
                        work.run_id,
                        episode_number,
                        content,
                        episode_lock=episode_lock,
                    )
                active = await self.repository.get_run_series_bible(work.run_id)
                if active is None:
                    raise AgentProtocolError(
                        "Episode generation requires an active SeriesBible design",
                        stage=InternalStage.GENERATING_EPISODE_SCRIPTS,
                    )
                await self.repository.create_script_batch(
                    work.run_id,
                    design_candidate_id=active.candidate_id,
                    design_content_hash=active.content_hash,
                    design_epoch=active.design_epoch,
                )
                resolved_call_id = (
                    call_id
                    or (model_call_state.context.call_id if model_call_state is not None else None)
                    or f"{work.run_id}-episode-{episode_number}"
                )
                candidate = await self.repository.commit_episode_candidate(
                    work.run_id,
                    episode_number=episode_number,
                    content=content,
                    episode_lock=episode_lock,
                    call_id=resolved_call_id,
                    writer_notes=writer_notes,
                )
                return EpisodeDraft(
                    episode_number=candidate.episode_number,
                    content=candidate.content,
                    content_sha256=candidate.content_sha256,
                    completed_at=candidate.created_at,
                    contract_sha256=candidate.state_delta.contract_sha256,
                    state_delta=candidate.state_delta,
                    series_state=candidate.series_state,
                    series_state_sha256=candidate.series_state_sha256,
                    semantic_review=candidate.semantic_review,
                    repair_rounds=candidate.repair_rounds,
                )

            async def retrieve_references(query: str) -> str:
                hits = self.catalog.retrieve_references(
                    work.persona.snapshot_sha256,
                    query,
                    limit=self.settings.retrieval_limit,
                )
                return json.dumps(
                    [asdict(hit) for hit in hits],
                    ensure_ascii=False,
                    separators=(",", ":"),
                )

            async def register_series_review(
                *,
                review_type: str,
                episode_number: int,
                passed: bool,
                category: str,
                evidence: str,
                earliest_affected_episode: int | None,
            ) -> str:
                """Bind and persist one structural review to the exact active lineage."""
                active = await self.repository.get_run_series_bible(work.run_id)
                batch = await self.repository.get_script_batch_lineage(work.run_id)
                if active is None or batch is None:
                    raise AgentProtocolError(
                        "A structural review requires an active SeriesBible and script batch",
                        stage=InternalStage.GENERATING_EPISODE_SCRIPTS,
                    )
                prefix = await self.repository.get_active_episode_candidates(work.run_id)
                prefix_hash = active_prefix_hash(
                    [
                        {
                            "episode_number": candidate.episode_number,
                            "content_sha256": candidate.content_sha256,
                        }
                        for candidate in prefix
                    ]
                )
                call_id = await self.repository.latest_review_call_id(
                    work.run_id,
                    stage=InternalStage.GENERATING_EPISODE_SCRIPTS,
                )
                bound = await self.repository.register_series_review(
                    work.run_id,
                    review_type=review_type,
                    episode_number=episode_number,
                    design_candidate_id=active.candidate_id,
                    design_content_hash=active.content_hash,
                    design_epoch=active.design_epoch,
                    batch_id=batch.batch_id,
                    batch_epoch=batch.batch_epoch,
                    prefix_hash=prefix_hash,
                    call_id=call_id or f"{work.run_id}-review-{episode_number}",
                    passed=passed,
                    category=category,
                    evidence=evidence,
                    earliest_affected_episode=earliest_affected_episode,
                )
                return bound.review_id

            run_timeout_scope = asyncio.timeout(self.settings.run_timeout_seconds)
            async with run_timeout_scope:

                async def reset_episode_deadline() -> None:
                    run_timeout_scope.reschedule(
                        asyncio.get_running_loop().time() + self.settings.run_timeout_seconds
                    )

                result = await self.workflow.execute(
                    thread_id=work.thread_id,
                    story=work.story,
                    requirements=work.requirements,
                    persona_files=persona_files,
                    before_stage=before_stage,
                    approve_stage=approve_stage,
                    approved_checkpoints=approved,
                    episode_drafts=work.episode_drafts,
                    before_episode=before_episode,
                    commit_episode=commit_episode,
                    assemble_episode_scripts=lambda: self.repository.assemble_episode_scripts(
                        work.run_id
                    ),
                    episode_timeout_seconds=self.settings.run_timeout_seconds,
                    reset_episode_deadline=reset_episode_deadline,
                    output_language=work.output_language,
                    feedback=work.frozen_feedback,
                    retrieve_references=retrieve_references,
                    series_bible=await self.repository.get_run_series_bible(work.run_id),
                    register_series_review=register_series_review,
                    get_series_bible=lambda: self.repository.get_run_series_bible(work.run_id),
                )
            if work.run_kind == "revision" and not result.feedback_handling:
                raise AgentProtocolError(
                    "Revision result omitted feedback handling",
                    stage=InternalStage.ACCEPTING_L4,
                )
            result = await self._validated_checkpoint_result(work, result, approved)

            current_stage = InternalStage.ASSEMBLING_DELIVERY
            if current_stage not in approved:
                await self.repository.record_stage_attempt(work.run_id, current_stage)
            delivery = self._assemble_delivery(work, result)
            if current_stage not in approved:
                assembly_payload = delivery.model_dump(mode="json")
                await self.repository.approve_checkpoint(
                    work.run_id,
                    current_stage,
                    assembly_payload,
                )
                approved[current_stage] = assembly_payload
            final_review_id = await self._resolve_final_review_id(work.run_id)
            await self.repository.succeed_run(
                work.run_id,
                delivery,
                final_review_id=final_review_id,
            )
            logger.info(
                "workflow run succeeded run_id=%s creation_id=%s",
                work.run_id,
                work.creation_id,
            )
        except asyncio.CancelledError:
            raise
        except MilestoneRejectedError as exc:
            await self._handle_milestone_rejection(work, exc)
            return
        except EpisodeTimeoutError as exc:
            recovery_state = await self.repository.handle_episode_timeout(
                work.run_id,
                exc.episode_number,
            )
            logger.warning(
                "episode script timed out run_id=%s creation_id=%s episode=%s state=%s",
                work.run_id,
                work.creation_id,
                exc.episode_number,
                recovery_state,
            )
            return
        except ContentReviewRejectedError as exc:
            await self.repository.pause_content_rejection(
                work.run_id,
                stage=exc.stage,
                evidence=exc.evidence,
                repair_rounds=exc.repair_rounds,
                episode_number=exc.episode_number,
            )
            logger.info(
                "content review paused run_id=%s creation_id=%s stage=%s episode=%s",
                work.run_id,
                work.creation_id,
                exc.stage.value,
                exc.episode_number,
            )
            return
        except QualityGateRejectedError as exc:
            rejection = await self.repository.reject_quality_gate(
                work.run_id,
                stage=exc.stage,
                evidence=exc.evidence,
            )
            logger.info(
                "quality gate rejected run_id=%s creation_id=%s stage=%s attempt=%s",
                work.run_id,
                work.creation_id,
                rejection.stage,
                rejection.attempt_count,
            )
            return
        except PreflightBlockedError as exc:
            await self.repository.pause_context_budget(
                work.run_id,
                stage=(InternalStage(exc.stage) if exc.stage else current_stage),
                safe_message=exc.safe_message,
                episode_number=exc.episode_number,
            )
            logger.warning(
                "context preflight blocked run_id=%s creation_id=%s stage=%s episode=%s "
                "required_tokens=%s verified_limit_tokens=%s",
                work.run_id,
                work.creation_id,
                exc.stage,
                exc.episode_number,
                exc.required_tokens,
                exc.verified_limit_tokens,
            )
            return
        except TimeoutError as exc:
            if run_timeout_scope is not None and run_timeout_scope.expired():
                timeout_stage = self._failure_stage(exc, current_stage, approved)
                if model_call_state is not None and model_call_state.store is not None:
                    submit_store_write(
                        model_call_state.store.mark_timed_out,
                        run_id=str(work.run_id),
                    )
                if timeout_stage is InternalStage.GENERATING_EPISODE_SCRIPTS:
                    refreshed = await self.repository.get_run_work_item(work.run_id)
                    episode_number = len(refreshed.episode_drafts) + 1
                    if episode_number <= len(refreshed.episode_plans):
                        recovery_state = await self.repository.handle_episode_timeout(
                            work.run_id,
                            episode_number,
                        )
                        logger.warning(
                            "episode script timed out run_id=%s creation_id=%s episode=%s state=%s",
                            work.run_id,
                            work.creation_id,
                            episode_number,
                            recovery_state,
                        )
                        return
                recovery_state = await self.repository.handle_run_timeout(
                    work.run_id,
                    timeout_stage,
                )
                logger.warning(
                    "workflow run timed out run_id=%s creation_id=%s stage=%s state=%s",
                    work.run_id,
                    work.creation_id,
                    timeout_stage.value,
                    recovery_state,
                )
                return
            failure_stage = self._failure_stage(exc, current_stage, approved)
            failure = await self._safe_failure(work.run_id, failure_stage, exc)
            await self.repository.fail_run(work.run_id, failure)
            logger.warning(
                "workflow run failed run_id=%s creation_id=%s stage=%s code=%s",
                work.run_id,
                work.creation_id,
                failure.failed_stage.value,
                failure.code,
            )
        except Exception as exc:
            failure_stage = self._failure_stage(exc, current_stage, approved)
            interruption = retryable_relay_interruption(exc)
            if interruption is not None:
                recovery_state, episode_number = await self._recover_relay_interruption(
                    work,
                    failure_stage,
                    interruption.retry_delay_seconds,
                )
                logger.warning(
                    "relay interruption recovered run_id=%s creation_id=%s stage=%s episode=%s "
                    "state=%s retry_delay_seconds=%s error_types=%s",
                    work.run_id,
                    work.creation_id,
                    failure_stage.value,
                    episode_number,
                    recovery_state,
                    interruption.retry_delay_seconds,
                    _exception_type_chain(exc),
                )
                return
            if await self._pause_recoverable_episode_error(work, failure_stage, exc):
                return
            failure = await self._safe_failure(work.run_id, failure_stage, exc)
            await self.repository.fail_run(work.run_id, failure)
            logger.warning(
                "workflow run failed run_id=%s creation_id=%s stage=%s code=%s error_types=%s",
                work.run_id,
                work.creation_id,
                failure.failed_stage.value,
                failure.code,
                _exception_type_chain(exc),
            )
        finally:
            # The audit handler dispatches SQLite writes to a writer thread so it
            # never deadlocks with the LangGraph checkpointer on the loop. Drain
            # before this job's run state is observed so the durable model-call
            # records (usage, stale, failed, blocked) are present for the API, UI,
            # and evidence (Delivery #58 INT-A2/A6/A9).
            await asyncio.to_thread(drain_audit_writes)

    async def _pause_recoverable_episode_error(
        self,
        work: RunWorkItem,
        stage: InternalStage,
        exc: Exception,
    ) -> bool:
        if stage is not InternalStage.GENERATING_EPISODE_SCRIPTS or isinstance(
            exc,
            (
                CheckpointUnavailableError,
                PersonaPackageError,
                DomainError,
                GraphRecursionError,
                RelayError,
                sqlite3.Error,
            ),
        ):
            return False
        refreshed = await self.repository.get_run_work_item(work.run_id)
        episode_number = len(refreshed.episode_drafts) + 1
        if episode_number > len(refreshed.episode_plans):
            return False
        attempts = await self.repository.get_episode_attempt_counts(work.run_id)
        attempt_count = attempts.get(episode_number, 0)
        if attempt_count == 0 or attempt_count >= 3:
            return False
        safe_message = _episode_error_message(exc)
        await self.repository.pause_episode_error(
            work.run_id,
            episode_number=episode_number,
            safe_message=safe_message,
        )
        logger.warning(
            "episode execution paused run_id=%s creation_id=%s episode=%s error_type=%s",
            work.run_id,
            work.creation_id,
            episode_number,
            type(exc).__name__,
        )
        return True

    async def _recover_relay_interruption(
        self,
        work: RunWorkItem,
        stage: InternalStage,
        retry_delay_seconds: int,
    ) -> tuple[str, int | None]:
        if stage is InternalStage.GENERATING_EPISODE_SCRIPTS:
            refreshed = await self.repository.get_run_work_item(work.run_id)
            episode_number = len(refreshed.episode_drafts) + 1
            if episode_number <= len(refreshed.episode_plans):
                return (
                    await self.repository.handle_episode_relay_interruption(
                        work.run_id,
                        episode_number,
                        retry_delay_seconds=retry_delay_seconds,
                    ),
                    episode_number,
                )
        return (
            await self.repository.handle_run_relay_interruption(
                work.run_id,
                stage,
                retry_delay_seconds=retry_delay_seconds,
            ),
            None,
        )

    def _persona_files(self, work: RunWorkItem) -> dict[str, str]:
        files: dict[str, str] = {}
        for stage in _SPECIALIST_STAGES:
            context = self.catalog.load_stage_context(
                work.persona.snapshot_sha256,
                stage,
            )
            for path, content in context.files.items():
                if path == "/persona/l4.md":
                    files[f"/persona/l4/{stage.value}.md"] = content
                else:
                    files.setdefault(path, content)
        return files

    async def _sync_series_bible(
        self,
        work: RunWorkItem,
        approved: Mapping[InternalStage, Any],
    ) -> None:
        """Assemble, validate, bind the review, and atomically promote one design candidate.

        Only unified runs whose approved episode outline carries a story contract
        produce a SeriesBible. The candidate is immutable; a confirmed design
        defect may trigger one complete automatic rebuild per run lineage, and a
        late candidate can never promote the active pointer (SDP-A5/A6/A7).
        """
        outline = approved.get(InternalStage.GENERATING_EPISODE_OUTLINE)
        if outline is None or "story_contract" not in outline:
            return
        story_outline = approved.get(InternalStage.GENERATING_STORY_OUTLINE)
        character_relationships = approved.get(InternalStage.GENERATING_CHARACTER_RELATIONSHIPS)
        l0_selection = approved.get(InternalStage.SELECTING_L0_VARIANT)
        if not story_outline or not character_relationships or not l0_selection:
            raise AgentProtocolError(
                "SeriesBible assembly requires every approved design projection",
                stage=InternalStage.GENERATING_EPISODE_OUTLINE,
            )
        active = await self.repository.get_run_series_bible(work.run_id)
        base = dict(
            run_id=str(work.run_id),
            run_kind=work.run_kind,
            l0_variant=l0_selection["selected_l0_variant"],
            genre=detect_genre(work.story, work.requirements),
            story_outline=story_outline["content"],
            character_biographies=character_relationships["character_biographies"],
            relationship_logic=character_relationships["relationship_logic"],
            episode_outline=outline["content"],
            story_contract_payload=outline["story_contract"],
            review_milestones=(
                outline.get("review_milestones") if isinstance(outline, Mapping) else None
            ),
        )
        candidate = build_series_bible(**base)
        if active is not None and active.content_hash == candidate.content_hash:
            return
        is_rebuild = active is not None
        authorized_rebuild = False
        persisted_rebuild_id = None
        if is_rebuild:
            lineage = await self.repository.get_series_bible_lineage(work.run_id)
            if lineage is not None and int(lineage["rebuild_count"]) >= 1:
                # An explicit one-cycle repair authorization (RPR-A9) may rebuild
                # exactly once more after the automatic budget is consumed. The
                # authorization is bound to the active design it was granted for.
                auth = await self.repository.get_repair_authorization(work.run_id)
                if (
                    auth is None
                    or auth["kind"] != "design_rebuild"
                    or auth["consumed_at"] is None
                    or auth["design_content_hash"] != active.content_hash
                ):
                    raise AgentProtocolError(
                        "This run lineage may rebuild the design automatically at most once",
                        stage=InternalStage.GENERATING_EPISODE_OUTLINE,
                    )
                authorized_rebuild = True
                persisted_rebuild_id = auth["rebuild_candidate_id"]
            candidate = build_series_bible(
                **base,
                parent_candidate_id=active.candidate_id,
                rebuild_count=1,
                design_epoch=active.design_epoch + 1,
                candidate_id=persisted_rebuild_id,
            )
        evidence = validate_series_bible(candidate)
        if is_rebuild:
            # Returns the persisted candidate for this epoch: after a crash
            # between INSERT and promotion, the same authorized one-cycle rebuild
            # resumes the identical candidate instead of building a duplicate.
            candidate = await self.repository.rebuild_series_bible(
                str(work.creation_id),
                work.run_id,
                candidate,
                evidence,
                authorized=authorized_rebuild,
            )
        else:
            await self.repository.register_series_bible_candidate(
                str(work.creation_id),
                work.run_id,
                candidate,
                evidence,
            )
        if not evidence.passed:
            # A deterministically invalid candidate is retained as immutable
            # evidence but is never reviewed or promoted, so no active pointer
            # can change (SDP-A3). The already-approved outline continues the run.
            logger.warning(
                "series bible candidate rejected run_id=%s candidate=%s issues=%s",
                work.run_id,
                candidate.candidate_id,
                sorted(issue.code for issue in evidence.issues),
            )
            return
        review = await self._bind_global_design_review(work, candidate, outline)
        await self.repository.record_series_bible_review(
            work.run_id,
            candidate.candidate_id,
            review,
        )
        await self.repository.promote_series_bible(work.run_id, candidate.candidate_id)
        await self.repository.mark_series_bible_stale(
            work.run_id,
            active_candidate_id=candidate.candidate_id,
        )

    async def _bind_global_design_review(
        self,
        work: RunWorkItem,
        candidate: SeriesBible,
        outline: Mapping[str, Any],
    ) -> GlobalDesignReview:
        """Bind the DeepSeek review evidence to this exact candidate and model call."""
        contract_review = outline.get("contract_review")
        if not isinstance(contract_review, Mapping):
            raise AgentProtocolError(
                "A unified outline requires bound global design review evidence",
                stage=InternalStage.GENERATING_EPISODE_OUTLINE,
            )
        call_id = await self.repository.latest_review_call_id(
            work.run_id,
            stage=InternalStage.GENERATING_EPISODE_OUTLINE,
        )
        if call_id is None:
            call_id = f"{work.run_id}-outline-review"

        return bind_global_design_review(
            candidate,
            review_call_id=call_id,
            review_model_id=self._review_model_id(),
            passed=bool(contract_review.get("passed")),
            evidence=str(contract_review.get("evidence", "")),
            issues=contract_review.get("issues") or [],
        )

    def _review_model_id(self) -> str:
        return self.settings.review_model_id or "gpt-5.5"

    @staticmethod
    def _assemble_delivery(work: RunWorkItem, result: WorkflowResult) -> Delivery:
        return Delivery(
            content_package=result.content_package,
            delivery_report=DeliveryReport(
                persona_id=work.persona.persona_id,
                persona_version=work.persona.version,
                persona_snapshot_sha256=work.persona.snapshot_sha256,
                selected_l0_variant=result.selected_l0_variant,
                selection_rationale=result.selection_rationale,
                l0_gate=result.l0_gate,
                l4_gate=result.l4_gate,
                ownership_statement=(
                    "最终创作所有权与判断由内部操作人员保留。"
                    if work.output_language == "zh-CN"
                    else "The internal operator retains final ownership and judgment."
                ),
                feedback_handling=result.feedback_handling,
            ),
        )

    @staticmethod
    def _next_unapproved(approved: Mapping[InternalStage, Any]) -> InternalStage:
        return next(
            (stage for stage in _ALL_STAGES if stage not in approved),
            InternalStage.ASSEMBLING_DELIVERY,
        )

    async def _resolve_final_review_id(self, run_id: UUID) -> str | None:
        """The bound passing final review that authorizes formal delivery.

        Legacy runs without an active SeriesBible keep the previous behavior
        (``None``). A unified run must have a passing bound final whole-series
        review; otherwise the run never reaches delivery (RPR-A2/A13).
        """
        active = await self.repository.get_run_series_bible(run_id)
        if active is None:
            return None
        batch = await self.repository.get_script_batch_lineage(run_id)
        if batch is None:
            raise AgentProtocolError(
                "A unified run requires an active script batch before delivery.",
                stage=InternalStage.ASSEMBLING_DELIVERY,
            )
        prefix = await self.repository.get_active_episode_candidates(run_id)
        prefix_hash = active_prefix_hash(
            [
                {
                    "episode_number": candidate.episode_number,
                    "content_sha256": candidate.content_sha256,
                }
                for candidate in prefix
            ]
        )
        review = await self.repository.get_latest_passing_final_review(
            run_id,
            design_content_hash=active.content_hash,
            batch_id=batch.batch_id,
            prefix_hash=prefix_hash,
        )
        if review is None:
            raise AgentProtocolError(
                "Formal delivery requires a passing bound final whole-series review.",
                stage=InternalStage.ASSEMBLING_DELIVERY,
            )
        return review.review_id

    async def _handle_milestone_rejection(
        self,
        work: RunWorkItem,
        exc: MilestoneRejectedError,
    ) -> None:
        """Classify and bound a rejected structural review.

        A script defect consumes the single automatic suffix-rewrite budget shared
        by all milestone and final reviews, preserves the retained prefix, and
        requeues the run to rewrite N..end. A design defect triggers the one
        automatic complete design regeneration per lineage. When an automatic
        budget is exhausted, the run pauses for an exact one-cycle authorization
        (RPR-A4/A5/A6/A8/A9).
        """
        if exc.category == "script_defect":
            batch = await self.repository.get_script_batch_lineage(work.run_id)
            if (
                batch is not None
                and exc.earliest_affected_episode is not None
                and await self.repository.has_automatic_suffix_budget(
                    work.run_id,
                    batch.batch_id,
                )
            ):
                await self.repository.consume_automatic_suffix_budget(
                    work.run_id,
                    batch.batch_id,
                )
                await self.repository.rewrite_episode_suffix(
                    work.run_id,
                    exc.earliest_affected_episode,
                )
                await self.repository.requeue_run_job(work.run_id)
                logger.info(
                    "automatic suffix rewrite run_id=%s creation_id=%s from_episode=%s",
                    work.run_id,
                    work.creation_id,
                    exc.earliest_affected_episode,
                )
                return
            await self.repository.pause_repair_authorization(
                work.run_id,
                kind="suffix_rewrite",
                design_candidate_id=(batch.design_candidate_id if batch is not None else ""),
                design_content_hash=(batch.design_content_hash if batch is not None else ""),
                design_epoch=(batch.design_epoch if batch is not None else 1),
                batch_id=(batch.batch_id if batch is not None else ""),
                batch_epoch=(batch.batch_epoch if batch is not None else 1),
                earliest_affected_episode=exc.earliest_affected_episode,
                range_episodes=(
                    len(await self.repository.get_active_episode_candidates(work.run_id))
                    - (exc.earliest_affected_episode or 1)
                    + 1
                ),
                estimated_tokens=await self._repair_token_estimate(
                    work.run_id,
                    from_episode=exc.earliest_affected_episode or 1,
                ),
                evidence=exc.evidence,
                review_id=exc.review_id or "",
            )
            logger.info(
                "suffix-rewrite authorization required run_id=%s creation_id=%s",
                work.run_id,
                work.creation_id,
            )
            return
        if exc.category == "design_defect":
            if await self.repository.design_rebuild_budget_available(work.run_id):
                await self.repository.trigger_design_rebuild(
                    work.run_id,
                    evidence=exc.evidence,
                )
                logger.info(
                    "automatic design rebuild run_id=%s creation_id=%s",
                    work.run_id,
                    work.creation_id,
                )
                return
            active = await self.repository.get_run_series_bible(work.run_id)
            batch = await self.repository.get_script_batch_lineage(work.run_id)
            await self.repository.pause_repair_authorization(
                work.run_id,
                kind="design_rebuild",
                design_candidate_id=(active.candidate_id if active else ""),
                design_content_hash=(active.content_hash if active else ""),
                design_epoch=(active.design_epoch if active else 1),
                batch_id=(batch.batch_id if batch else ""),
                batch_epoch=(batch.batch_epoch if batch else 1),
                earliest_affected_episode=None,
                range_episodes=None,
                estimated_tokens=await self._repair_token_estimate(
                    work.run_id,
                    from_episode=1,
                ),
                evidence=exc.evidence,
                review_id=exc.review_id or "",
            )
            logger.info(
                "design-rebuild authorization required run_id=%s creation_id=%s",
                work.run_id,
                work.creation_id,
            )
            return
        logger.warning(
            "unexpected structural review category run_id=%s category=%s",
            work.run_id,
            exc.category,
        )

    async def _repair_token_estimate(self, run_id: UUID, *, from_episode: int) -> int:
        """A deterministic token estimate for the one authorized generation+review cycle.

        The estimate covers the retained active prefix scripts plus the active design
        projections that the writer must carry into the regenerated range (RPR-A8).
        """
        from pengine.model_calls import estimate_text_tokens

        active = await self.repository.get_run_series_bible(run_id)
        prefix = await self.repository.get_active_episode_candidates(run_id)
        parts: list[str] = []
        if active is not None:
            projections = active.projections
            parts.extend(
                [
                    projections.story_outline,
                    projections.character_biographies,
                    projections.relationship_logic,
                    projections.episode_outline,
                    projections.story_contract_markdown,
                ]
            )
        for candidate in prefix:
            if candidate.episode_number < from_episode:
                parts.append(candidate.content)
        return estimate_text_tokens("\n".join(parts))

    @staticmethod
    def _requires_langgraph_checkpoint(work: RunWorkItem) -> bool:
        durable_stages = set(work.stage_attempts) | set(work.business_checkpoints)
        return (
            bool(work.episode_plans)
            or bool(work.episode_drafts)
            or any(stage in durable_stages for stage in _SPECIALIST_STAGES)
        )

    async def _validated_checkpoint_result(
        self,
        work: RunWorkItem,
        result: WorkflowResult,
        approved: Mapping[InternalStage, Any],
    ) -> WorkflowResult:
        missing = [stage for stage in _SPECIALIST_STAGES if stage not in approved]
        if missing:
            raise AgentProtocolError(
                "Supervisor finished before every specialist stage was approved",
                stage=missing[0],
            )
        try:
            aggregate_payload = dict(
                await self.repository.episode_aggregate_checkpoint_payload(work.run_id)
            )
            aggregate_episode_scripts = aggregate_payload["content"]
            l0_selection = approved[InternalStage.SELECTING_L0_VARIANT]
            l0_gate = approved[InternalStage.ACCEPTING_L0]
            l4_gate = approved[InternalStage.ACCEPTING_L4]
            story_outline = approved[InternalStage.GENERATING_STORY_OUTLINE]
            character_relationships = approved[InternalStage.GENERATING_CHARACTER_RELATIONSHIPS]
            expected = WorkflowResult.model_validate(
                {
                    "content_package": {
                        "story_outline": story_outline["content"],
                        "character_biographies": character_relationships["character_biographies"],
                        "relationship_logic": character_relationships["relationship_logic"],
                        "episode_outline": approved[InternalStage.GENERATING_EPISODE_OUTLINE][
                            "content"
                        ],
                        "episode_scripts": aggregate_episode_scripts,
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
        except Exception as exc:
            raise AgentProtocolError(
                "Approved specialist checkpoints are invalid",
                stage=InternalStage.ASSEMBLING_DELIVERY,
            ) from exc
        approved_episode_scripts = dict(approved[InternalStage.GENERATING_EPISODE_SCRIPTS])
        expected_episode_scripts = {
            **(
                {"stage": InternalStage.GENERATING_EPISODE_SCRIPTS.value}
                if "stage" in approved_episode_scripts
                else {}
            ),
            **aggregate_payload,
        }
        if approved_episode_scripts != expected_episode_scripts:
            raise AgentProtocolError(
                "Approved episode scripts or lock hashes differ from committed episodes",
                stage=InternalStage.ASSEMBLING_DELIVERY,
            )
        if result != expected:
            raise AgentProtocolError(
                "Supervisor result differs from approved specialist checkpoints",
                stage=InternalStage.ASSEMBLING_DELIVERY,
            )
        return expected

    def _failure_stage(
        self,
        exc: Exception,
        current_stage: InternalStage,
        approved: Mapping[InternalStage, Any],
    ) -> InternalStage:
        declared = getattr(exc, "stage", None)
        if isinstance(declared, InternalStage):
            return declared
        if current_stage not in approved:
            return current_stage
        return self._next_unapproved(approved)

    async def _safe_failure(
        self,
        run_id: UUID,
        stage: InternalStage,
        exc: Exception,
    ) -> RunFailure:
        if stage is InternalStage.GENERATING_EPISODE_SCRIPTS:
            work = await self.repository.get_run_work_item(run_id)
            episode_number = len(work.episode_drafts) + 1
            counts = await self.repository.get_episode_attempt_counts(run_id)
            attempt_count = counts.get(episode_number, 0)
            if attempt_count == 0 and episode_number <= len(work.episode_plans):
                try:
                    attempt_count = await self.repository.record_episode_attempt(
                        run_id,
                        episode_number,
                    )
                except DomainError:
                    attempt_count = 1
            code, message = _classify_failure(exc)
            return RunFailure(
                code=code,
                message=message,
                failed_stage=stage,
                attempt_count=max(1, min(3, attempt_count)),
            )
        counts = await self.repository.get_stage_attempt_counts(run_id)
        attempt_count = counts.get(stage, 0)
        if attempt_count == 0:
            try:
                attempt_count = await self.repository.record_stage_attempt(run_id, stage)
            except DomainError:
                attempt_count = 1

        code, message = _classify_failure(exc)
        return RunFailure(
            code=code,
            message=message,
            failed_stage=stage,
            attempt_count=max(1, min(3, attempt_count)),
        )


def _classify_failure(exc: Exception) -> tuple[str, str]:
    if isinstance(exc, RelayError):
        return exc.code, exc.safe_message
    if isinstance(exc, AgentProtocolError):
        return "structured_output_invalid", exc.safe_message
    if isinstance(exc, QualityGateRejectedError):
        return "quality_gate_rejected", "The final quality gate rejected the generated work."
    if isinstance(exc, ContentReviewRejectedError):
        return "content_review_rejected", "The bounded content review did not converge."
    if isinstance(exc, PersonaPackageError):
        return "persona_package_invalid", "The persona snapshot could not be loaded."
    if isinstance(exc, CheckpointUnavailableError):
        return "checkpoint_unavailable", "The workflow checkpoint is unavailable."
    if isinstance(exc, GraphRecursionError):
        return "graph_recursion_limit", "The workflow graph recursion limit was exhausted."
    if isinstance(exc, TimeoutError):
        return "relay_unavailable", "The model relay timed out."
    if isinstance(exc, DomainError) and exc.code == "attempts_exhausted":
        return "attempts_exhausted", "The stage attempt limit was exhausted."
    if isinstance(exc, sqlite3.Error) or type(exc).__module__.startswith("langgraph.checkpoint"):
        return "checkpoint_unavailable", "The workflow checkpoint is unavailable."
    if "structuredoutput" in type(exc).__name__.lower():
        return "structured_output_invalid", "模型未返回有效的结构化结果。"
    if is_relay_exception(exc):
        relay_error = classify_relay_exception(exc)
        return relay_error.code, relay_error.safe_message
    return "internal_error", "The workflow failed safely."


def _episode_error_message(exc: Exception) -> str:
    if isinstance(exc, ValueError) and "decimal numbers" in str(exc):
        return (
            "算术工具收到非十进制参数。需要先把时刻换算为当日经过分钟数后再计算；"
            "已完成分集不受影响。"
        )
    if is_relay_connection_error(exc):
        return "当前集与模型 Relay / 网络连接失败。已完成分集不受影响；继续时只会重试当前集。"
    if isinstance(exc, AgentProtocolError):
        return "当前集代理返回了无效的结构化结果。已完成分集不受影响；继续时只会重试当前集。"
    return "当前集遇到可恢复的执行错误。已完成分集不受影响；继续时只会重试当前集。"
