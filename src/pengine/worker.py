import asyncio
import json
import logging
import sqlite3
from collections.abc import Awaitable, Callable, Mapping
from contextlib import AbstractAsyncContextManager, asynccontextmanager, contextmanager, suppress
from dataclasses import asdict
from typing import Any, Literal, Protocol
from uuid import UUID, uuid4

import aiosqlite
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.errors import GraphRecursionError

from pengine.agents import (
    L0_GATE_EVIDENCE_LABELS,
    L4_GATE_EVIDENCE_LABELS,
    L4_STAGE_EVIDENCE_LABEL,
    AgentExecutionLimitError,
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
from pengine.model_calls import (
    ModelCallState,
    ModelCallStore,
    StageCallBudgetExceeded,
    new_operation_id,
)
from pengine.observability import content_fingerprint, record_langfuse_event
from pengine.personas import (
    PERSONA_SCHEMA_V2,
    PERSONA_SCHEMA_V3,
    PersonaCatalog,
    PersonaPackageError,
    extract_l0_variant_ids,
)
from pengine.relay import (
    PreflightBlockedError,
    RelayError,
    RelayIdentityError,
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
from pengine.series_review import (
    BoundStructuralReview,
    active_prefix_hash,
    aggregate_script_defect_evidence,
)

logger = logging.getLogger(__name__)


def _publish_langfuse_env(settings: Settings) -> None:
    """Seed LANGFUSE_* process env vars so the SDK auto-configures itself.

    pengine keeps its own PENGINE_LANGFUSE_* settings (prefix discipline), but the
    Langfuse SDK reads LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY / LANGFUSE_BASE_URL
    from the environment. We mirror them here, once at worker startup and before
    the relay models are built, so the callback handler picks them up.
    """
    if not settings.langfuse_configured:
        return
    assert settings.langfuse_public_key is not None  # noqa: S101 - narrowed by property
    assert settings.langfuse_secret_key is not None  # noqa: S101 - narrowed by property
    assert settings.langfuse_host is not None  # noqa: S101 - narrowed by property
    import os

    os.environ["LANGFUSE_PUBLIC_KEY"] = settings.langfuse_public_key.get_secret_value()
    os.environ["LANGFUSE_SECRET_KEY"] = settings.langfuse_secret_key.get_secret_value()
    os.environ["LANGFUSE_BASE_URL"] = settings.langfuse_host


def _flush_langfuse(settings: Settings) -> None:
    """Flush pending Langfuse traces at worker shutdown so nothing is lost."""
    if not settings.langfuse_configured:
        return
    try:
        from langfuse import get_client

        get_client().flush()
    except Exception as exc:  # pragma: no cover - tracing must never block shutdown
        logger.warning("langfuse flush failed error_type=%s", type(exc).__name__)


@contextmanager
def _safe_langfuse_context(context_manager: Any):
    """Suppress harmless OTel context-detach errors during async teardown."""
    try:
        with context_manager as value:
            yield value
    except ValueError as exc:
        if "different Context" not in str(exc):
            raise
        logger.debug("langfuse context detach skipped after async teardown")


def _langfuse_trace_context(settings: Settings, work: RunWorkItem):
    """Return a context manager that scopes a run's traces in Langfuse.

    When tracing is disabled this is a no-op :class:`contextlib.nullcontext`, so the
    worker's control flow is unchanged. When enabled it does two things:

    1. Opens a root ``agent`` observation named after the run, so every LangChain
       callback the relay models fire (LLM generations, tool calls) nests under it
       instead of becoming isolated flat traces. deepagents propagates the parent
       OTel context to sub-agents automatically, so the supervisor + 9 specialists
       form a single tree.
    2. Stamps the trace with ``session_id`` = run id, ``user_id`` = creation id,
       tags, and run lineage metadata. A run is the unit of workflow evidence;
       creation-level grouping can be done through ``user_id``.

    The worker processes one run at a time, so the ambient OTel context is
    unambiguous across runs.
    """
    if not settings.langfuse_configured:
        from contextlib import nullcontext

        return nullcontext()
    try:
        from langfuse import get_client, propagate_attributes
    except ImportError:  # pragma: no cover - langfuse optional
        from contextlib import nullcontext

        return nullcontext()

    @contextmanager
    def _scoped():
        # propagate_attributes sets trace-level attributes (session/user/tags) on
        # whatever trace is current inside the block; the root observation opened
        # next becomes that trace, and nested callback observations inherit it.
        # Both are OTel agnostic context managers: synchronous enter/exit (context
        # attach is sync), so the worker can use a plain ``with`` to scope a run.
        with (
            _safe_langfuse_context(
                propagate_attributes(
                    session_id=str(work.run_id),
                    user_id=str(work.creation_id),
                    metadata={
                        "run_id": str(work.run_id),
                        "run_kind": work.run_kind,
                        "thread_id": work.thread_id,
                        "trace_version": "pengine-1",
                    },
                    tags=["pengine", work.run_kind],
                )
            ),
            _safe_langfuse_context(
                get_client().start_as_current_observation(
                    name=f"pengine.run:{work.run_kind}",
                    as_type="agent",
                    input={
                        "run_id": str(work.run_id),
                        "creation_id": str(work.creation_id),
                        "run_kind": work.run_kind,
                        "story_chars": len(work.story),
                        "requirements_chars": len(work.requirements),
                    },
                    metadata={
                        "thread_id": work.thread_id,
                        "trace_version": "pengine-1",
                    },
                )
            ),
        ):
            yield

    return _scoped()


_SPECIALIST_STAGES = (
    InternalStage.SELECTING_L0_VARIANT,
    InternalStage.GENERATING_STORY_OUTLINE,
    InternalStage.GENERATING_CHARACTER_RELATIONSHIPS,
    InternalStage.GENERATING_EPISODE_OUTLINE,
    InternalStage.GENERATING_EPISODE_SCRIPTS,
)
_ALL_STAGES = (
    InternalStage.LOADING_PERSONA,
    *_SPECIALIST_STAGES,
    InternalStage.ASSEMBLING_DELIVERY,
)


def _validate_persona_checkpoint_payload(
    stage: InternalStage,
    payload: Mapping[str, Any],
    explicit_variant_ids: tuple[str, ...],
) -> dict[str, Any]:
    validated_payload = dict(payload)
    if stage is InternalStage.SELECTING_L0_VARIANT and explicit_variant_ids:
        selected = validated_payload.get("selected_l0_variant")
        if isinstance(selected, str):
            for variant_id in explicit_variant_ids:
                if selected == f"[ID:{variant_id}]":
                    validated_payload["selected_l0_variant"] = variant_id
                    break
    if (
        stage is InternalStage.SELECTING_L0_VARIANT
        and explicit_variant_ids
        and validated_payload.get("selected_l0_variant") not in explicit_variant_ids
    ):
        raise AgentProtocolError(
            "The selected L0 variant is not one of the persona's declared IDs",
            stage=stage,
        )
    if stage is InternalStage.ACCEPTING_L0 and validated_payload.get("passed") is True:
        evidence = validated_payload.get("evidence")
        if not isinstance(evidence, str) or any(
            label not in evidence for label in L0_GATE_EVIDENCE_LABELS
        ):
            raise AgentProtocolError(
                "Passing L0 evidence is missing one or more required sections",
                stage=stage,
            )
    if stage is InternalStage.ACCEPTING_L4 and validated_payload.get("passed") is True:
        evidence = validated_payload.get("evidence")
        if not isinstance(evidence, str) or any(
            label not in evidence for label in L4_GATE_EVIDENCE_LABELS
        ):
            raise AgentProtocolError(
                "Passing L4 evidence is missing one or more required authority sections",
                stage=stage,
            )
    review_field = {
        InternalStage.GENERATING_STORY_OUTLINE: "consistency_review",
        InternalStage.GENERATING_CHARACTER_RELATIONSHIPS: "consistency_review",
        InternalStage.GENERATING_EPISODE_OUTLINE: "contract_review",
    }.get(stage)
    review = validated_payload.get(review_field) if review_field is not None else None
    if (
        isinstance(review, Mapping)
        and review.get("passed") is True
        and L4_STAGE_EVIDENCE_LABEL not in str(review.get("evidence", ""))
    ):
        raise AgentProtocolError(
            "Passing stage review evidence is missing the required L4 hard-rule section",
            stage=stage,
        )
    return validated_payload


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


def _suffix_rewrite_feedback_payload(
    reviews: list[BoundStructuralReview],
    effective_earliest_episode: int,
) -> dict[str, Any]:
    """Serialize unresolved defects bound to the latest reviewed prefix."""
    return {
        "version": 1,
        "effective_earliest_affected_episode": effective_earliest_episode,
        "reviews": [
            {
                "review_id": review.review_id,
                "category": review.category,
                "evidence": review.evidence,
                "earliest_affected_episode": review.earliest_affected_episode,
                "binding": {
                    "run_id": review.run_id,
                    "review_epoch": review.review_epoch,
                    "review_type": review.review_type,
                    "episode_number": review.episode_number,
                    "design_candidate_id": review.design_candidate_id,
                    "design_content_hash": review.design_content_hash,
                    "design_epoch": review.design_epoch,
                    "batch_id": review.batch_id,
                    "batch_epoch": review.batch_epoch,
                    "prefix_hash": review.prefix_hash,
                    "call_id": review.call_id,
                },
            }
            for review in reviews
        ],
    }


class StageValidationError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        stage: InternalStage,
        safe_message: str,
    ) -> None:
        super().__init__(message)
        self.stage = stage
        self.safe_message = safe_message


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
        get_series_review_boundary: Callable[[int], Awaitable[BoundStructuralReview | None]]
        | None = None,
        suffix_rewrite_feedback: Mapping[str, Any] | None = None,
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
        self._flake_requeues: dict[UUID, int] = {}
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
                _publish_langfuse_env(self.settings)
                routes = build_relay_routes(self.settings)
                state = getattr(routes, "model_call_state", None)
                if state is not None:
                    store = ModelCallStore(self.settings.database_path)
                    state.store = store
                    state.reset()
                    self._model_call_state = state
                    self._model_call_store = store
                self.workflow = DeepAgentWorkflow(
                    generation_model=routes.generation.model,
                    review_model=routes.review.model,
                    checkpointer=self._saver,
                    recursion_limit=self.settings.agent_recursion_limit,
                    generation_provider_profile_key=routes.generation.provider_profile_key,
                    review_provider_profile_key=routes.review.provider_profile_key,
                    model_call_state=state,
                    generation_max_output_tokens=self.settings.generation_max_output_tokens,
                    review_max_output_tokens=self.settings.review_max_output_tokens,
                    review_context_limit_tokens=self.settings.review_context_limit_tokens,
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
        _flush_langfuse(self.settings)
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
        heartbeat = asyncio.create_task(
            self._renew_job_lease(job),
            name=f"pengine-lease-heartbeat-{job.run_id}",
        )
        try:
            await self._process_job_body(job)
        finally:
            heartbeat.cancel()
            with suppress(asyncio.CancelledError):
                await heartbeat

    async def _renew_job_lease(self, job: LeasedJob) -> None:
        interval = max(1.0, self.settings.lease_seconds / 3)
        while True:
            try:
                await asyncio.sleep(interval)
                await self.repository.renew_job_lease(
                    job_id=job.job_id,
                    worker_id=job.lease_owner,
                    lease_seconds=self.settings.lease_seconds,
                )
                logger.debug(
                    "job lease renewed worker_id=%s run_id=%s lease_seconds=%s",
                    self.worker_id,
                    job.run_id,
                    self.settings.lease_seconds,
                )
            except asyncio.CancelledError:
                raise
            except DomainError as exc:
                logger.error(
                    "job lease renewal stopped worker_id=%s run_id=%s error_code=%s",
                    self.worker_id,
                    job.run_id,
                    exc.code,
                )
                return
            except Exception as exc:
                logger.warning(
                    "job lease renewal failed worker_id=%s run_id=%s error_type=%s",
                    self.worker_id,
                    job.run_id,
                    type(exc).__name__,
                )

    async def _unresolved_suffix_reviews(
        self,
        run_id: UUID,
    ) -> list[BoundStructuralReview]:
        return await self.repository.get_unresolved_script_defect_reviews(run_id)

    async def _suffix_rewrite_feedback_for_writer(
        self,
        run_id: UUID,
    ) -> Mapping[str, Any] | None:
        reviews = await self._unresolved_suffix_reviews(run_id)
        if not reviews:
            return None
        affected_episodes = [
            review.earliest_affected_episode
            for review in reviews
            if review.earliest_affected_episode is not None
        ]
        if not affected_episodes:
            return None
        effective_earliest = min(affected_episodes)
        unfinished_episode = await self.repository.first_unfinished_episode(run_id)
        if unfinished_episode is None or unfinished_episode < effective_earliest:
            return None
        return _suffix_rewrite_feedback_payload(reviews, effective_earliest)

    async def _process_job_body(self, job: LeasedJob) -> None:
        await self.repository.mark_run_running(job.run_id)
        work = await self.repository.get_run_work_item(job.run_id)
        approved: dict[InternalStage, Any] = dict(work.business_checkpoints)
        active_series_bible = await self.repository.get_run_series_bible(work.run_id)
        final_review_ready = active_series_bible is None
        if active_series_bible is not None:
            try:
                await self._resolve_final_review_id(work.run_id)
            except AgentProtocolError:
                final_review_ready = False
            else:
                final_review_ready = True
        if (
            work.episode_plans
            and len(work.episode_drafts) == len(work.episode_plans)
            and InternalStage.GENERATING_EPISODE_SCRIPTS not in approved
            and final_review_ready
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
            model_call_state.reset()
            model_call_state.context.run_id = str(work.run_id)
            model_call_state.context.creation_id = str(work.creation_id)
            model_call_state.context.thread_id = work.thread_id
            model_call_state.context.run_kind = work.run_kind
            model_call_state.context.stage = current_stage.value
            model_call_state.context.operation_id = new_operation_id()
        logger.info(
            "workflow run started run_id=%s creation_id=%s kind=%s",
            work.run_id,
            work.creation_id,
            work.run_kind,
        )

        trace_cm = _langfuse_trace_context(self.settings, work)
        stage_cm: Any = None
        stage_observation: Any = None
        persona_trace_metadata: dict[str, Any] = {}

        def close_stage_observation(*, status: str, error: str | None = None) -> None:
            nonlocal stage_cm, stage_observation
            if stage_observation is None or stage_cm is None:
                return
            try:
                stage_observation.update(
                    output={"status": status, **({"error": error} if error else {})}
                )
            except Exception as exc:  # pragma: no cover - observability must not fail runs
                logger.debug("langfuse stage update failed error_type=%s", type(exc).__name__)
            try:
                stage_cm.__exit__(None, None, None)
            except Exception as exc:  # pragma: no cover - observability must not fail runs
                logger.debug("langfuse stage close failed error_type=%s", type(exc).__name__)
            stage_cm = None
            stage_observation = None

        def open_stage_observation(stage: InternalStage, attempt: int) -> None:
            nonlocal stage_cm, stage_observation
            if not self.settings.langfuse_configured:
                return
            close_stage_observation(status="superseded")
            try:
                from langfuse import get_client

                stage_cm = _safe_langfuse_context(
                    get_client().start_as_current_observation(
                        name=f"pengine.stage:{stage.value}",
                        as_type="span",
                        input={"stage": stage.value, "attempt": attempt},
                        metadata={
                            "run_id": str(work.run_id),
                            "stage": stage.value,
                            "attempt": attempt,
                            "trace_version": "pengine-1",
                            **persona_trace_metadata,
                        },
                    )
                )
                stage_observation = stage_cm.__enter__()
            except Exception as exc:  # pragma: no cover - observability must not fail runs
                logger.debug("langfuse stage open failed error_type=%s", type(exc).__name__)
                stage_cm = None
                stage_observation = None

        with trace_cm:
            try:
                if InternalStage.LOADING_PERSONA not in approved:
                    await self.repository.record_stage_attempt(
                        work.run_id,
                        InternalStage.LOADING_PERSONA,
                    )
                snapshot = self.catalog.resolve_snapshot(work.persona.snapshot_sha256)
                schema_version = str(snapshot.manifest["schema_version"])
                soul_text = (
                    snapshot.text("soul")
                    if schema_version in {PERSONA_SCHEMA_V2, PERSONA_SCHEMA_V3}
                    else None
                )
                l3_text = snapshot.text("l3") if schema_version == PERSONA_SCHEMA_V3 else None
                persona_trace_metadata.update(
                    {
                        "persona_schema_version": schema_version,
                        "persona_id": snapshot.summary.persona_id,
                        "persona_version": snapshot.summary.version,
                        "persona_snapshot_sha256": snapshot.summary.snapshot_sha256,
                        "soul_sha256": (
                            snapshot.manifest["files"]["soul"]["sha256"]
                            if soul_text is not None
                            else None
                        ),
                        "soul_char_count": len(soul_text) if soul_text is not None else None,
                        "soul_mount_path": "/persona/soul.md" if soul_text is not None else None,
                        "soul_full_text_loaded": soul_text is not None,
                        "l3_sha256": (
                            snapshot.manifest["files"]["l3"]["sha256"]
                            if l3_text is not None
                            else None
                        ),
                        "l3_char_count": len(l3_text) if l3_text is not None else None,
                        "l3_mount_path": "/persona/l3.md" if l3_text is not None else None,
                        "l3_full_text_mounted": l3_text is not None,
                    }
                )
                if model_call_state is not None:
                    for key, value in persona_trace_metadata.items():
                        setattr(model_call_state.context, key, value)
                if InternalStage.LOADING_PERSONA not in approved:
                    loading_payload = snapshot.summary.model_dump(mode="json")
                    await self.repository.approve_checkpoint(
                        work.run_id,
                        InternalStage.LOADING_PERSONA,
                        loading_payload,
                    )
                    approved[InternalStage.LOADING_PERSONA] = loading_payload

                persona_files = self._persona_files(work)
                explicit_l0_variant_ids = extract_l0_variant_ids(persona_files["/persona/l0.md"])

                if await self._handle_queued_quality_repair(
                    work,
                    approved=approved,
                    persona_files=persona_files,
                    model_call_state=model_call_state,
                ):
                    return

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
                workflow_thread_id = work.thread_id
                if isinstance(self.workflow, DeepAgentWorkflow):
                    outline_approved = InternalStage.GENERATING_EPISODE_OUTLINE in approved
                    if outline_approved:
                        active_series_bible = await self.repository.get_run_series_bible(
                            work.run_id
                        )
                        if active_series_bible is None:
                            raise CheckpointUnavailableError(
                                "The active SeriesBible checkpoint is missing."
                            )
                        workflow_thread_id = self.workflow.episode_script_thread_id(
                            work.thread_id,
                            active_series_bible.candidate_id,
                        )
                    elif self._requires_langgraph_checkpoint(
                        work
                    ) and not await self.workflow.has_checkpoint(work.thread_id):
                        raise CheckpointUnavailableError(
                            "The durable workflow checkpoint is missing."
                        )

                async def before_stage(stage: InternalStage) -> int:
                    nonlocal current_stage
                    current_stage = stage
                    if model_call_state is not None:
                        model_call_state.context.stage = stage.value
                        model_call_state.context.episode_number = None
                        model_call_state.context.operation_id = new_operation_id()
                    attempt = await self.repository.record_stage_attempt(work.run_id, stage)
                    open_stage_observation(stage, attempt)
                    return attempt

                async def approve_stage(
                    stage: InternalStage,
                    payload: Mapping[str, Any],
                ) -> None:
                    approved_payload = _validate_persona_checkpoint_payload(
                        stage,
                        payload,
                        explicit_l0_variant_ids,
                    )
                    review_call_id = None
                    if (
                        stage is InternalStage.GENERATING_EPISODE_OUTLINE
                        and "story_contract" in approved_payload
                    ):
                        operation_id = (
                            model_call_state.context.operation_id
                            if model_call_state is not None
                            else None
                        )
                        review_call_id = await self._require_physical_call_id(
                            run_id=work.run_id,
                            role="review",
                            stage=stage,
                            episode_number=None,
                            operation_id=operation_id,
                        )
                    await self.repository.approve_checkpoint(
                        work.run_id,
                        stage,
                        approved_payload,
                        review_call_id=review_call_id,
                    )
                    approved[stage] = approved_payload
                    close_stage_observation(status="approved")
                    if stage is InternalStage.GENERATING_EPISODE_OUTLINE:
                        await self._sync_series_bible(work, approved)
                        if model_call_state is not None:
                            active = await self.repository.get_run_series_bible(work.run_id)
                            model_call_state.context.candidate = (
                                active.candidate_id if active is not None else None
                            )

                async def before_episode(
                    plan: EpisodePlan,
                    *,
                    new_operation: bool = True,
                ) -> int:
                    nonlocal current_stage
                    current_stage = InternalStage.GENERATING_EPISODE_SCRIPTS
                    if model_call_state is not None:
                        model_call_state.context.stage = (
                            InternalStage.GENERATING_EPISODE_SCRIPTS.value
                        )
                        model_call_state.context.episode_number = plan.episode_number
                        if new_operation:
                            model_call_state.context.operation_id = new_operation_id()
                    return await self.repository.record_episode_attempt(
                        work.run_id,
                        plan.episode_number,
                    )

                async def load_outline_season_map() -> Mapping[str, Any] | None:
                    return await self.repository.get_outline_season_map(work.run_id)

                async def commit_outline_season_map(payload: Mapping[str, Any]) -> None:
                    operation_id = (
                        model_call_state.context.operation_id
                        if model_call_state is not None
                        else None
                    )
                    call_id = await self._require_physical_call_id(
                        run_id=work.run_id,
                        role="generation",
                        stage=InternalStage.GENERATING_EPISODE_OUTLINE,
                        episode_number=None,
                        operation_id=operation_id,
                    )
                    await self.repository.commit_outline_season_map(
                        work.run_id,
                        payload,
                        call_id=call_id,
                    )

                async def load_outline_groups() -> list[Mapping[str, Any]]:
                    return await self.repository.get_committed_outline_groups(work.run_id)

                outline_group_context: dict[str, tuple[str, int]] = {}

                async def begin_outline_group(
                    *,
                    group_id: str,
                    position: int,
                    start_episode: int,
                    end_episode: int,
                ) -> str:
                    operation_id = new_operation_id()
                    if model_call_state is not None:
                        model_call_state.context.stage = (
                            InternalStage.GENERATING_EPISODE_OUTLINE.value
                        )
                        model_call_state.context.episode_number = start_episode
                        model_call_state.context.operation_id = operation_id
                        model_call_state.context.batch = group_id
                    await self.repository.begin_outline_group(
                        work.run_id,
                        group_id=group_id,
                        position=position,
                        start_episode=start_episode,
                        end_episode=end_episode,
                        operation_id=operation_id,
                    )
                    outline_group_context[group_id] = (operation_id, start_episode)
                    return operation_id

                async def load_outline_group_body(
                    group_id: str,
                ) -> Mapping[str, Any] | None:
                    return await self.repository.get_outline_group_body(
                        work.run_id,
                        group_id=group_id,
                    )

                async def persist_outline_group_body(
                    group_id: str,
                    *,
                    operation_id: str,
                    outline_markdown: str,
                    outline_markdown_sha256: str,
                    expected_outline_markdown_sha256: str | None = None,
                ) -> str:
                    expected_operation, start_episode = outline_group_context[group_id]
                    if expected_operation != operation_id:
                        raise AgentProtocolError(
                            "Outline body operation no longer matches its active attempt",
                            stage=InternalStage.GENERATING_EPISODE_OUTLINE,
                        )
                    body_call_id = await self._require_physical_call_id(
                        run_id=work.run_id,
                        role="generation",
                        stage=InternalStage.GENERATING_EPISODE_OUTLINE,
                        episode_number=start_episode,
                        operation_id=operation_id,
                    )
                    if expected_outline_markdown_sha256 is None:
                        await self.repository.save_outline_group_body(
                            work.run_id,
                            group_id=group_id,
                            operation_id=operation_id,
                            outline_markdown=outline_markdown,
                            outline_markdown_sha256=outline_markdown_sha256,
                            body_call_id=body_call_id,
                        )
                    else:
                        await self.repository.replace_outline_group_body(
                            work.run_id,
                            group_id=group_id,
                            operation_id=operation_id,
                            expected_outline_markdown_sha256=(expected_outline_markdown_sha256),
                            outline_markdown=outline_markdown,
                            outline_markdown_sha256=outline_markdown_sha256,
                            body_call_id=body_call_id,
                        )
                    return body_call_id

                async def complete_outline_group(
                    *,
                    group_id: str,
                    operation_id: str,
                    payload: Mapping[str, Any],
                ) -> None:
                    expected_operation, start_episode = outline_group_context[group_id]
                    if expected_operation != operation_id:
                        raise AgentProtocolError(
                            "Outline group operation no longer matches its active attempt",
                            stage=InternalStage.GENERATING_EPISODE_OUTLINE,
                        )
                    sidecar_call_id = await self._require_physical_call_id(
                        run_id=work.run_id,
                        role="generation",
                        stage=InternalStage.GENERATING_EPISODE_OUTLINE,
                        episode_number=start_episode,
                        operation_id=operation_id,
                    )
                    await self.repository.complete_outline_group(
                        work.run_id,
                        group_id=group_id,
                        operation_id=operation_id,
                        payload=payload,
                        sidecar_call_id=sidecar_call_id,
                    )

                async def fail_outline_group(
                    *,
                    group_id: str,
                    operation_id: str,
                ) -> None:
                    await self.repository.fail_outline_group(
                        work.run_id,
                        group_id=group_id,
                        operation_id=operation_id,
                    )

                async def load_outline_group_rejection(
                    group_id: str,
                ) -> Mapping[str, Any] | None:
                    return await self.repository.get_outline_group_rejection(
                        work.run_id,
                        group_id=group_id,
                    )

                async def record_outline_markdown_failure(
                    *,
                    group_id: str,
                    operation_id: str,
                    attempt_index: int,
                    raw_text: str,
                    normalized_text: str,
                    parse_error: str,
                ) -> None:
                    await self.repository.record_outline_markdown_failure(
                        work.run_id,
                        group_id=group_id,
                        operation_id=operation_id,
                        attempt_index=attempt_index,
                        raw_text=raw_text,
                        normalized_text=normalized_text,
                        parse_error=parse_error,
                    )

                generation_window_context: dict[str, tuple[str, int]] = {}

                async def begin_generation_group(
                    *,
                    group_id: str,
                    start_episode: int,
                    end_episode: int,
                ) -> str:
                    active = await self.repository.get_run_series_bible(work.run_id)
                    if active is None:
                        raise AgentProtocolError(
                            "Episode generation requires an active SeriesBible design",
                            stage=InternalStage.GENERATING_EPISODE_SCRIPTS,
                        )
                    operation_id = (
                        model_call_state.context.operation_id
                        if model_call_state is not None
                        else None
                    )
                    if operation_id is None or operation_id in {
                        stored_operation_id
                        for stored_operation_id, _ in generation_window_context.values()
                    }:
                        operation_id = new_operation_id()
                    if model_call_state is not None:
                        model_call_state.context.stage = (
                            InternalStage.GENERATING_EPISODE_SCRIPTS.value
                        )
                        model_call_state.context.episode_number = start_episode
                        model_call_state.context.operation_id = operation_id
                    window_id = await self.repository.begin_episode_generation_window(
                        work.run_id,
                        design_candidate_id=active.candidate_id,
                        design_content_hash=active.content_hash,
                        design_epoch=active.design_epoch,
                        group_id=group_id,
                        start_episode=start_episode,
                        end_episode=end_episode,
                        operation_id=operation_id,
                    )
                    stored_text = await self.repository.get_episode_generation_text(
                        work.run_id,
                        window_id,
                    )
                    if stored_text is not None:
                        operation_id = str(stored_text["operation_id"])
                        if model_call_state is not None:
                            model_call_state.context.operation_id = operation_id
                    generation_window_context[window_id] = (operation_id, start_episode)
                    return window_id

                async def load_generation_group_text(
                    window_id: str,
                ) -> Mapping[str, Any] | None:
                    return await self.repository.get_episode_generation_text(
                        work.run_id,
                        window_id,
                    )

                async def persist_generation_group_text(
                    window_id: str,
                    *,
                    nonce: str,
                    raw_text: str,
                    manifest: list[Mapping[str, Any]],
                ) -> str:
                    operation_id, start_episode = generation_window_context[window_id]
                    content_call_id = await self._require_physical_call_id(
                        run_id=work.run_id,
                        role="generation",
                        stage=InternalStage.GENERATING_EPISODE_SCRIPTS,
                        episode_number=start_episode,
                        operation_id=operation_id,
                    )
                    await self.repository.save_episode_generation_text(
                        work.run_id,
                        window_id,
                        screenplay_text=raw_text,
                        nonce=nonce,
                        manifest=manifest,
                        content_call_id=content_call_id,
                        context_bundle_sha256=(
                            model_call_state.context.context_bundle_sha256
                            if model_call_state is not None
                            else None
                        ),
                    )
                    return content_call_id

                async def complete_generation_group(
                    window_id: str,
                    *,
                    provenance_episode_number: int,
                ) -> str:
                    operation_id, start_episode = generation_window_context[window_id]
                    if start_episode != provenance_episode_number:
                        raise AgentProtocolError(
                            "Generation window provenance episode does not match its range",
                            stage=InternalStage.GENERATING_EPISODE_SCRIPTS,
                        )
                    call_id = await self._require_physical_call_id(
                        run_id=work.run_id,
                        role="generation",
                        stage=InternalStage.GENERATING_EPISODE_SCRIPTS,
                        episode_number=start_episode,
                        operation_id=operation_id,
                    )
                    stored_text = await self.repository.get_episode_generation_text(
                        work.run_id,
                        window_id,
                    )
                    content_call_id = (
                        str(stored_text["content_call_id"]) if stored_text is not None else call_id
                    )
                    await self.repository.bind_episode_generation_window_call(
                        work.run_id,
                        window_id,
                        call_id=content_call_id,
                        sidecar_call_id=(call_id if stored_text is not None else None),
                    )
                    return content_call_id

                async def fail_generation_group(
                    window_id: str,
                    *,
                    preserve_text: bool = False,
                ) -> None:
                    await self.repository.fail_episode_generation_window(
                        work.run_id,
                        window_id,
                        preserve_text=preserve_text,
                    )

                async def commit_episode(
                    episode_number: int,
                    content: str,
                    episode_lock: EpisodeLock | None = None,
                    *,
                    call_id: str | None = None,
                    writer_notes: str = "",
                    provenance_episode_number: int | None = None,
                    generation_window_id: str | None = None,
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
                    if generation_window_id is not None and call_id is not None:
                        stored_text = await self.repository.get_episode_generation_text(
                            work.run_id,
                            generation_window_id,
                        )
                        if (
                            stored_text is not None
                            and stored_text.get("content_call_id") != call_id
                        ):
                            raise AgentProtocolError(
                                "Episode candidate call_id does not match its plaintext call",
                                stage=InternalStage.GENERATING_EPISODE_SCRIPTS,
                            )
                        resolved_call_id = call_id
                    else:
                        operation_id = (
                            model_call_state.context.operation_id
                            if model_call_state is not None
                            else None
                        )
                        resolved_call_id = await self._require_physical_call_id(
                            run_id=work.run_id,
                            role="generation",
                            stage=InternalStage.GENERATING_EPISODE_SCRIPTS,
                            episode_number=(provenance_episode_number or episode_number),
                            operation_id=operation_id,
                        )
                    if call_id is not None and call_id != resolved_call_id:
                        raise AgentProtocolError(
                            "Episode candidate call_id does not match its successful physical "
                            "generation call",
                            stage=InternalStage.GENERATING_EPISODE_SCRIPTS,
                        )
                    candidate = await self.repository.commit_episode_candidate(
                        work.run_id,
                        episode_number=episode_number,
                        content=content,
                        episode_lock=episode_lock,
                        call_id=resolved_call_id,
                        writer_notes=writer_notes,
                        generation_window_id=generation_window_id,
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
                    operation_id = (
                        model_call_state.context.operation_id
                        if model_call_state is not None
                        else None
                    )
                    call_id = await self._require_physical_call_id(
                        run_id=work.run_id,
                        role="review",
                        stage=InternalStage.GENERATING_EPISODE_SCRIPTS,
                        episode_number=episode_number,
                        operation_id=operation_id,
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
                        call_id=call_id,
                        passed=passed,
                        category=category,
                        evidence=evidence,
                        earliest_affected_episode=earliest_affected_episode,
                    )
                    record_langfuse_event(
                        "pengine.review.result",
                        input={
                            "run_id": str(work.run_id),
                            "stage": current_stage.value,
                            "review_type": review_type,
                            "episode_number": episode_number,
                            "passed": passed,
                            "category": category,
                            "evidence_chars": len(evidence),
                        },
                        metadata={
                            "review_id": bound.review_id,
                            "evidence_sha256": content_fingerprint(evidence),
                            "trace_version": "pengine-1",
                        },
                    )
                    return bound.review_id

                async def get_series_review_boundary(
                    episode_number: int,
                ) -> BoundStructuralReview | None:
                    """Return the latest passing receipt in the exact active lineage."""
                    active = await self.repository.get_run_series_bible(work.run_id)
                    batch = await self.repository.get_script_batch_lineage(work.run_id)
                    if active is None or batch is None:
                        return None
                    eligible = [
                        review
                        for review in await self.repository.get_series_reviews(work.run_id)
                        if review.status == "active"
                        and review.passed
                        and review.category == "pass"
                        and review.episode_number < episode_number
                        and review.design_candidate_id == active.candidate_id
                        and review.design_content_hash == active.content_hash
                        and review.design_epoch == active.design_epoch
                        and review.batch_id == batch.batch_id
                        and review.batch_epoch == batch.batch_epoch
                    ]
                    return max(eligible, key=lambda review: review.episode_number, default=None)

                run_timeout_scope = asyncio.timeout(self.settings.run_timeout_seconds)
                async with run_timeout_scope:

                    async def reset_episode_deadline() -> None:
                        run_timeout_scope.reschedule(
                            asyncio.get_running_loop().time() + self.settings.run_timeout_seconds
                        )

                    workflow_kwargs: dict[str, Any] = {
                        "thread_id": workflow_thread_id,
                        "story": work.story,
                        "requirements": work.requirements,
                        "persona_files": persona_files,
                        "before_stage": before_stage,
                        "approve_stage": approve_stage,
                        "approved_checkpoints": approved,
                        "episode_drafts": work.episode_drafts,
                        "before_episode": before_episode,
                        "commit_episode": commit_episode,
                        "assemble_episode_scripts": (
                            lambda: self.repository.assemble_episode_scripts(work.run_id)
                        ),
                        "episode_timeout_seconds": self.settings.run_timeout_seconds,
                        "reset_episode_deadline": reset_episode_deadline,
                        "output_language": work.output_language,
                        "feedback": work.frozen_feedback,
                        "retrieve_references": retrieve_references,
                        "series_bible": await self.repository.get_run_series_bible(work.run_id),
                        "register_series_review": register_series_review,
                        "get_series_bible": lambda: self.repository.get_run_series_bible(
                            work.run_id
                        ),
                    }
                    if isinstance(self.workflow, DeepAgentWorkflow):
                        workflow_kwargs.update(
                            {
                                "get_series_review_boundary": get_series_review_boundary,
                                "begin_generation_group": begin_generation_group,
                                "complete_generation_group": complete_generation_group,
                                "fail_generation_group": fail_generation_group,
                                "load_generation_group_text": load_generation_group_text,
                                "persist_generation_group_text": persist_generation_group_text,
                                "load_outline_season_map": load_outline_season_map,
                                "commit_outline_season_map": commit_outline_season_map,
                                "load_outline_groups": load_outline_groups,
                                "begin_outline_group": begin_outline_group,
                                "load_outline_group_body": load_outline_group_body,
                                "persist_outline_group_body": persist_outline_group_body,
                                "complete_outline_group": complete_outline_group,
                                "fail_outline_group": fail_outline_group,
                                "record_outline_markdown_failure": record_outline_markdown_failure,
                                "load_outline_group_rejection": load_outline_group_rejection,
                            }
                        )
                    suffix_rewrite_feedback = await self._suffix_rewrite_feedback_for_writer(
                        work.run_id
                    )
                    if suffix_rewrite_feedback is not None:
                        workflow_kwargs["suffix_rewrite_feedback"] = suffix_rewrite_feedback
                    result = await self.workflow.execute(**workflow_kwargs)
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
                # Publish succeeded state only after every physical model-call row is
                # durable, so the first terminal API projection cannot omit ledger rows.
                await asyncio.to_thread(drain_audit_writes)
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
            except (AgentProtocolError, StageValidationError) as exc:
                # Structured-output flakes are stochastic model behavior, not
                # terminal defects: consume the existing stage attempt budget
                # (already recorded by before_stage) and auto-retry the stage
                # from its approved checkpoints before ever failing the run.
                failure_stage = (
                    exc.stage
                    if isinstance(getattr(exc, "stage", None), InternalStage)
                    else InternalStage(work.current_stage or "loading_persona")
                )
                # Only retry stages that a resume will actually re-enter (not yet
                # approved, so before_stage re-fires and the budget advances);
                # approved-stage sync errors are deterministic and stay terminal.
                # The script stage budgets per episode instead of stage_attempts
                # and keeps its existing recoverable-episode pause path.
                retryable_flake = (
                    failure_stage is not InternalStage.GENERATING_EPISODE_SCRIPTS
                    and failure_stage not in approved
                )
                requeues = self._flake_requeues.get(work.run_id, 0)
                if retryable_flake and requeues < 2:
                    self._flake_requeues[work.run_id] = requeues + 1
                    await self.repository.requeue_stage_flake_retry(
                        work.run_id,
                        stage=failure_stage,
                    )
                    logger.warning(
                        "structured flake retried run_id=%s creation_id=%s stage=%s "
                        "requeue=%d error=%s",
                        work.run_id,
                        work.creation_id,
                        failure_stage.value,
                        requeues + 1,
                        str(exc)[:200],
                    )
                    return
                # Preserve the pre-existing recoverable-episode pause path.
                if (
                    failure_stage is InternalStage.GENERATING_EPISODE_SCRIPTS
                    and await self._pause_recoverable_episode_error(work, failure_stage, exc)
                ):
                    return
                failure = await self._safe_failure(work.run_id, failure_stage, exc)
                await self.repository.fail_run(work.run_id, failure)
                logger.warning(
                    "workflow run failed run_id=%s creation_id=%s stage=%s code=%s "
                    "(stage flake budget exhausted or terminal sync error)",
                    work.run_id,
                    work.creation_id,
                    failure_stage.value,
                    failure.code,
                )
                return
            except ContentReviewRejectedError as exc:
                await self.repository.pause_content_rejection(
                    work.run_id,
                    stage=exc.stage,
                    evidence=exc.evidence,
                    repair_rounds=exc.repair_rounds,
                    episode_number=exc.episode_number,
                    outline_group_id=exc.outline_group_id,
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
                    repair_plan=exc.repair_plan,
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
            except RelayIdentityError as exc:
                try:
                    identity_stage = InternalStage(exc.stage) if exc.stage else current_stage
                except ValueError:
                    identity_stage = current_stage
                episode_number = exc.episode_number
                if (
                    identity_stage is InternalStage.GENERATING_EPISODE_SCRIPTS
                    and episode_number is None
                ):
                    refreshed = await self.repository.get_run_work_item(work.run_id)
                    next_episode = len(refreshed.episode_drafts) + 1
                    if next_episode <= len(refreshed.episode_plans):
                        episode_number = next_episode
                await self.repository.pause_relay_identity_mismatch(
                    work.run_id,
                    stage=identity_stage,
                    safe_message=exc.safe_message,
                    episode_number=episode_number,
                )
                logger.warning(
                    "relay identity mismatch paused run_id=%s creation_id=%s stage=%s "
                    "episode=%s requested_model_id=%s response_model_ids=%s",
                    work.run_id,
                    work.creation_id,
                    identity_stage.value,
                    episode_number,
                    exc.requested_model_id,
                    list(exc.response_model_ids),
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
                                "episode script timed out run_id=%s creation_id=%s "
                                "episode=%s state=%s",
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
                close_stage_observation(status="failed")
                # The audit handler dispatches SQLite writes to a writer thread so it
                # never deadlocks with the LangGraph checkpointer on the loop. Drain
                # before this job's run state is observed so the durable model-call
                # records (usage, stale, failed, blocked) are present for the API, UI,
                # and evidence (Delivery #58 INT-A2/A6/A9).
                await asyncio.to_thread(drain_audit_writes)

    async def _require_physical_call_id(
        self,
        *,
        run_id: UUID,
        role: Literal["generation", "review"],
        stage: InternalStage,
        episode_number: int | None,
        operation_id: str | None,
    ) -> str:
        """Resolve one exact successful provider call, never synthetic provenance."""
        if operation_id is None:
            raise AgentProtocolError(
                f"Missing successful physical {role} call for artifact provenance",
                stage=stage,
            )
        expected_call_id = None
        if self._model_call_state is not None:
            expected_call_id = self._model_call_state.latest_succeeded_call_id(
                role=role,
                run_id=str(run_id),
                stage=stage.value,
                episode_number=episode_number,
                operation_id=operation_id,
            )
        # The callback updates in-memory completion synchronously, while SQLite writes
        # land on the audit thread. Drain before binding an immutable artifact so the
        # referenced physical call is already durable if the process stops afterward.
        await asyncio.to_thread(drain_audit_writes)
        durable_call_id = await self.repository.latest_successful_model_call_id(
            run_id,
            role=role,
            stage=stage,
            episode_number=episode_number,
            operation_id=operation_id,
        )
        if durable_call_id is None or (
            expected_call_id is not None and expected_call_id != durable_call_id
        ):
            raise AgentProtocolError(
                f"Missing successful physical {role} call for artifact provenance",
                stage=stage,
            )
        return durable_call_id

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
                AgentExecutionLimitError,
                StageCallBudgetExceeded,
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
            "episode execution paused run_id=%s creation_id=%s episode=%s "
            "error_type=%s error=%s safe_message=%s",
            work.run_id,
            work.creation_id,
            episode_number,
            type(exc).__name__,
            str(exc),
            safe_message,
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
        approved: dict[InternalStage, Any],
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
            script_generation_groups=(
                outline.get("script_generation_groups") if isinstance(outline, Mapping) else None
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

        if evidence.passed:
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
                candidate = await self.repository.register_series_bible_candidate(
                    str(work.creation_id),
                    work.run_id,
                    candidate,
                    evidence,
                )
        else:
            if is_rebuild:
                candidate = await self.repository.rebuild_series_bible(
                    str(work.creation_id),
                    work.run_id,
                    candidate,
                    evidence,
                    authorized=authorized_rebuild,
                )
            else:
                candidate = await self.repository.register_series_bible_candidate(
                    str(work.creation_id),
                    work.run_id,
                    candidate,
                    evidence,
                )
            # A deterministically invalid candidate is retained as immutable
            # evidence but is never reviewed or promoted, so no active pointer
            # can change (SDP-A3). Episode writing requires an active design, so
            # stop at the design boundary instead of failing later during commit.
            issue_codes = sorted(issue.code for issue in evidence.issues)
            logger.warning(
                "series bible candidate rejected run_id=%s candidate=%s issues=%s",
                work.run_id,
                candidate.candidate_id,
                issue_codes,
            )
            raise StageValidationError(
                "SeriesBible candidate failed deterministic validation: " + ", ".join(issue_codes),
                stage=InternalStage.GENERATING_EPISODE_OUTLINE,
                safe_message="系列设计未通过确定性校验，未进入分集剧本生成。",
            )
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
        """Bind review evidence to this exact candidate and model call."""
        contract_review = outline.get("contract_review")
        if not isinstance(contract_review, Mapping):
            raise AgentProtocolError(
                "A unified outline requires bound global design review evidence",
                stage=InternalStage.GENERATING_EPISODE_OUTLINE,
            )
        call_id = await self.repository.get_checkpoint_review_call_id(
            work.run_id,
            InternalStage.GENERATING_EPISODE_OUTLINE,
        )
        if call_id is None:
            raise AgentProtocolError(
                "Global design review lacks a successful physical review call",
                stage=InternalStage.GENERATING_EPISODE_OUTLINE,
            )
        if not await self.repository.is_successful_model_call(
            work.run_id,
            call_id=call_id,
            role="review",
            stage=InternalStage.GENERATING_EPISODE_OUTLINE,
            episode_number=None,
        ):
            raise AgentProtocolError(
                "Global design review lacks a successful physical review call",
                stage=InternalStage.GENERATING_EPISODE_OUTLINE,
            )

        return bind_global_design_review(
            candidate,
            review_call_id=call_id,
            review_model_id=self._review_model_id(),
            passed=bool(contract_review.get("passed")),
            evidence=str(contract_review.get("evidence", "")),
            issues=contract_review.get("issues") or [],
        )

    def _review_model_id(self) -> str:
        return self.settings.review_model_id or "claude-opus-5"

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

    async def _handle_queued_quality_repair(
        self,
        work: RunWorkItem,
        *,
        approved: dict[InternalStage, Any],
        persona_files: Mapping[str, str],
        model_call_state: ModelCallState | None,
    ) -> bool:
        request = await self.repository.get_queued_quality_repair(work.run_id)
        if request is None:
            return False
        try:
            return await self._execute_queued_quality_repair(
                work,
                request=request,
                approved=approved,
                persona_files=persona_files,
                model_call_state=model_call_state,
            )
        except (AgentProtocolError, DomainError, ValueError) as exc:
            await self.repository.block_quality_repair(
                work.run_id,
                stage=InternalStage(request["stage"]),
                rejection_attempt=request["rejection_attempt"],
                evidence=f"受限修复未能安全提交：{exc}",
            )
            return True

    async def _execute_queued_quality_repair(
        self,
        work: RunWorkItem,
        *,
        request: Mapping[str, Any],
        approved: dict[InternalStage, Any],
        persona_files: Mapping[str, str],
        model_call_state: ModelCallState | None,
    ) -> bool:
        if not isinstance(self.workflow, DeepAgentWorkflow):
            await self.repository.block_quality_repair(
                work.run_id,
                stage=InternalStage(request["stage"]),
                rejection_attempt=request["rejection_attempt"],
                evidence="当前运行时没有配置证据修复执行器。",
            )
            return True
        stage = InternalStage(request["stage"])
        outline = approved.get(InternalStage.GENERATING_EPISODE_OUTLINE)
        active_design = await self.repository.get_run_series_bible(work.run_id)
        batch = await self.repository.get_script_batch_lineage(work.run_id)
        candidates = await self.repository.get_active_episode_candidates(work.run_id)
        if (
            not isinstance(outline, Mapping)
            or not isinstance(outline.get("story_contract"), Mapping)
            or active_design is None
            or batch is None
            or not candidates
            or batch.batch_id != request["original_batch_id"]
        ):
            await self.repository.block_quality_repair(
                work.run_id,
                stage=stage,
                rejection_attempt=request["rejection_attempt"],
                evidence="审核证据无法绑定当前 StoryContract、设计或完整剧本批次。",
            )
            return True
        episodes = {candidate.episode_number: candidate.content for candidate in candidates}
        plan = request["plan"]
        if plan is None:
            if model_call_state is not None:
                model_call_state.context.stage = stage.value
                model_call_state.context.episode_number = None
                model_call_state.context.operation_id = new_operation_id()
            plan = await self.workflow.plan_quality_repair(
                stage=stage,
                evidence=request["evidence"] or "",
                episodes=episodes,
                persona_files=persona_files,
                story_contract=outline["story_contract"],
                output_language=work.output_language,
            )
            await self.repository.set_quality_repair_plan(
                work.run_id,
                stage=stage,
                rejection_attempt=request["rejection_attempt"],
                plan=plan,
            )
            if plan.scope != "episode_content":
                return True
        if plan.scope != "episode_content":
            await self.repository.block_quality_repair(
                work.run_id,
                stage=stage,
                rejection_attempt=request["rejection_attempt"],
                evidence=plan.rationale,
            )
            return True

        patched_contents: dict[int, tuple[str, str]] = {}
        for episode_number in sorted({issue.episode_number for issue in plan.issues}):
            if model_call_state is not None:
                model_call_state.context.stage = stage.value
                model_call_state.context.episode_number = episode_number
                model_call_state.context.operation_id = new_operation_id()
            repaired, _patch = await self.workflow.generate_quality_episode_patch(
                stage=stage,
                episode_number=episode_number,
                content=episodes[episode_number],
                plan=plan,
                persona_files=persona_files,
                story_contract=outline["story_contract"],
                output_language=work.output_language,
            )
            operation_id = (
                model_call_state.context.operation_id if model_call_state is not None else None
            )
            call_id = await self._require_physical_call_id(
                run_id=work.run_id,
                role="generation",
                stage=stage,
                episode_number=episode_number,
                operation_id=operation_id,
            )
            patched_contents[episode_number] = (repaired, call_id)

        preview = {
            number: patched_contents.get(number, (content, ""))[0]
            for number, content in episodes.items()
        }
        final_episode = max(preview)
        if model_call_state is not None:
            model_call_state.context.stage = InternalStage.GENERATING_EPISODE_SCRIPTS.value
            model_call_state.context.episode_number = final_episode
            model_call_state.context.operation_id = new_operation_id()
        structural = await self.workflow.review_quality_repaired_series(
            original_episodes=episodes,
            repaired_episodes=preview,
            repair_plan=plan,
            story_contract=outline["story_contract"],
            series_bible=active_design.model_dump(mode="json"),
            output_language=work.output_language,
        )
        structural_operation = (
            model_call_state.context.operation_id if model_call_state is not None else None
        )
        structural_call_id = await self._require_physical_call_id(
            run_id=work.run_id,
            role="review",
            stage=InternalStage.GENERATING_EPISODE_SCRIPTS,
            episode_number=final_episode,
            operation_id=structural_operation,
        )
        if not structural.passed:
            await self.repository.block_quality_repair(
                work.run_id,
                stage=stage,
                rejection_attempt=request["rejection_attempt"],
                evidence=structural.evidence,
            )
            return True

        new_batch = await self.repository.apply_quality_episode_patches(
            work.run_id,
            expected_batch_id=batch.batch_id,
            patched_contents=patched_contents,
            stage=stage,
            rejection_attempt=request["rejection_attempt"],
        )
        active_after = await self.repository.get_active_episode_candidates(work.run_id)
        prefix_hash = active_prefix_hash(
            [
                {
                    "episode_number": candidate.episode_number,
                    "content_sha256": candidate.content_sha256,
                }
                for candidate in active_after
            ]
        )
        await self.repository.register_series_review(
            work.run_id,
            review_type="final",
            episode_number=final_episode,
            design_candidate_id=active_design.candidate_id,
            design_content_hash=active_design.content_hash,
            design_epoch=active_design.design_epoch,
            batch_id=new_batch.batch_id,
            batch_epoch=new_batch.batch_epoch,
            prefix_hash=prefix_hash,
            call_id=structural_call_id,
            passed=True,
            category="pass",
            evidence=structural.evidence,
            earliest_affected_episode=None,
        )

        refreshed = await self.repository.get_run_work_item(work.run_id)
        repaired_approved = dict(refreshed.business_checkpoints)
        gates = [InternalStage.ACCEPTING_L0]
        if stage is InternalStage.ACCEPTING_L4:
            gates.append(InternalStage.ACCEPTING_L4)
        for gate in gates:
            await self.repository.record_stage_attempt(work.run_id, gate)
            if model_call_state is not None:
                model_call_state.context.stage = gate.value
                model_call_state.context.episode_number = None
                model_call_state.context.operation_id = new_operation_id()
            result = await self.workflow.review_quality_gate(
                stage=gate,
                approved_artifacts={
                    checkpoint.value: payload for checkpoint, payload in repaired_approved.items()
                },
                persona_files=persona_files,
                output_language=work.output_language,
            )
            operation_id = (
                model_call_state.context.operation_id if model_call_state is not None else None
            )
            review_call_id = await self._require_physical_call_id(
                run_id=work.run_id,
                role="review",
                stage=gate,
                episode_number=None,
                operation_id=operation_id,
            )
            if not result.passed:
                await self.repository.reject_quality_gate(
                    work.run_id,
                    stage=gate,
                    evidence=result.evidence,
                    repair_plan=result.repair_plan,
                )
                return True
            payload = _validate_persona_checkpoint_payload(
                gate,
                result.model_dump(mode="json"),
                (),
            )
            await self.repository.approve_checkpoint(
                work.run_id,
                gate,
                payload,
                review_call_id=review_call_id,
            )
            repaired_approved[gate] = payload
        await self.repository.requeue_run_job(work.run_id)
        return True

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
            unresolved_reviews = await self._unresolved_suffix_reviews(work.run_id)
            affected_episodes = [
                review.earliest_affected_episode
                for review in unresolved_reviews
                if review.earliest_affected_episode is not None
            ]
            effective_earliest = (
                min(affected_episodes) if affected_episodes else exc.earliest_affected_episode
            )
            aggregated_evidence = (
                aggregate_script_defect_evidence(unresolved_reviews)
                if unresolved_reviews
                else exc.evidence
            )
            latest_review_id = (
                unresolved_reviews[-1].review_id if unresolved_reviews else exc.review_id
            )
            if (
                batch is not None
                and effective_earliest is not None
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
                    effective_earliest,
                )
                await self.repository.requeue_run_job(work.run_id)
                logger.info(
                    "automatic suffix rewrite run_id=%s creation_id=%s from_episode=%s",
                    work.run_id,
                    work.creation_id,
                    effective_earliest,
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
                earliest_affected_episode=effective_earliest,
                range_episodes=(len(work.episode_plans) - (effective_earliest or 1) + 1),
                estimated_tokens=await self._repair_reference_context_tokens(
                    work.run_id,
                    from_episode=effective_earliest or 1,
                ),
                evidence=aggregated_evidence,
                review_id=latest_review_id or "",
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
                estimated_tokens=await self._repair_reference_context_tokens(
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

    async def _repair_reference_context_tokens(self, run_id: UUID, *, from_episode: int) -> int:
        """Return the reference context amount shown at authorization.

        It counts the retained active prefix plus the active design projections at the pause.
        Those texts are not guaranteed to be the exact input of a design rebuild or every
        suffix call, so this is neither a lower bound nor a total cycle forecast (RPR-A8).
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
                    "feedback_handling": result.feedback_handling,
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
        normalized_result = result.model_copy(update={"l0_gate": None, "l4_gate": None})
        if normalized_result != expected:
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
    if isinstance(exc, (StageCallBudgetExceeded, AgentExecutionLimitError)):
        return "agent_execution_limit", str(exc)
    if isinstance(exc, RelayError):
        return exc.code, exc.safe_message
    if isinstance(exc, StageValidationError):
        return "stage_validation_failed", exc.safe_message
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
