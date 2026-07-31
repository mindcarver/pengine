import asyncio
import json
import logging
import sqlite3
from collections.abc import Awaitable, Callable, Mapping
from contextlib import AbstractAsyncContextManager, suppress
from dataclasses import asdict
from typing import Any, Protocol
from uuid import UUID, uuid4

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.errors import GraphRecursionError

from pengine.agents import (
    AgentProtocolError,
    CheckpointUnavailableError,
    DeepAgentWorkflow,
    EpisodeTimeoutError,
    QualityGateRejectedError,
)
from pengine.config import Settings
from pengine.errors import DomainError
from pengine.personas import PersonaCatalog, PersonaPackageError
from pengine.relay import RelayError, build_chat_model, classify_relay_exception
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

logger = logging.getLogger(__name__)

_SPECIALIST_STAGES = (
    InternalStage.SELECTING_L0_VARIANT,
    InternalStage.GENERATING_STORY_OUTLINE,
    InternalStage.GENERATING_CHARACTER_BIOGRAPHIES,
    InternalStage.GENERATING_RELATIONSHIP_LOGIC,
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

StageHook = Callable[[InternalStage], Awaitable[int]]
CheckpointHook = Callable[[InternalStage, Mapping[str, Any]], Awaitable[None]]
ReferenceRetriever = Callable[[str], Awaitable[str]]
EpisodeAttemptHook = Callable[[EpisodePlan], Awaitable[int]]
EpisodeCommitHook = Callable[[int, str], Awaitable[EpisodeDraft]]
EpisodeAssemblyHook = Callable[[], Awaitable[str]]
EpisodeDeadlineReset = Callable[[], Awaitable[None]]


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
        feedback: str | None = None,
        retrieve_references: ReferenceRetriever | None = None,
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

    async def start(self) -> None:
        if self._task is not None:
            return
        await self.repository.reconcile_startup()
        if self.workflow is None:
            self._saver_context = AsyncSqliteSaver.from_conn_string(
                str(self.settings.database_path)
            )
            self._saver = await self._saver_context.__aenter__()
            await self._saver.setup()
            if self.settings.relay_configured:
                self.workflow = DeepAgentWorkflow(
                    model=build_chat_model(self.settings),
                    checkpointer=self._saver,
                    recursion_limit=self.settings.agent_recursion_limit,
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
        if self._saver_context is not None:
            await self._saver_context.__aexit__(None, None, None)
            self._saver_context = None
            self._saver = None

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
            aggregate = await self.repository.assemble_episode_scripts(work.run_id)
            payload = {
                "stage": InternalStage.GENERATING_EPISODE_SCRIPTS.value,
                "content": aggregate,
            }
            await self.repository.approve_checkpoint(
                work.run_id,
                InternalStage.GENERATING_EPISODE_SCRIPTS,
                payload,
            )
            approved[InternalStage.GENERATING_EPISODE_SCRIPTS] = payload
        current_stage = InternalStage.LOADING_PERSONA
        run_timeout_scope: asyncio.Timeout | None = None
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
                return await self.repository.record_stage_attempt(work.run_id, stage)

            async def approve_stage(
                stage: InternalStage,
                payload: Mapping[str, Any],
            ) -> None:
                await self.repository.approve_checkpoint(work.run_id, stage, payload)
                approved[stage] = dict(payload)

            async def before_episode(plan: EpisodePlan) -> int:
                nonlocal current_stage
                current_stage = InternalStage.GENERATING_EPISODE_SCRIPTS
                return await self.repository.record_episode_attempt(
                    work.run_id,
                    plan.episode_number,
                )

            async def commit_episode(episode_number: int, content: str) -> EpisodeDraft:
                return await self.repository.commit_episode_draft(
                    work.run_id,
                    episode_number,
                    content,
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
                    feedback=work.frozen_feedback,
                    retrieve_references=retrieve_references,
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
            await self.repository.succeed_run(work.run_id, delivery)
            logger.info(
                "workflow run succeeded run_id=%s creation_id=%s",
                work.run_id,
                work.creation_id,
            )
        except asyncio.CancelledError:
            raise
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
        except TimeoutError as exc:
            if run_timeout_scope is not None and run_timeout_scope.expired():
                timeout_stage = self._failure_stage(exc, current_stage, approved)
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
            failure = await self._safe_failure(work.run_id, failure_stage, exc)
            await self.repository.fail_run(work.run_id, failure)
            logger.warning(
                "workflow run failed run_id=%s creation_id=%s stage=%s code=%s",
                work.run_id,
                work.creation_id,
                failure.failed_stage.value,
                failure.code,
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
                ownership_statement=("The internal operator retains final ownership and judgment."),
                feedback_handling=result.feedback_handling,
            ),
        )

    @staticmethod
    def _next_unapproved(approved: Mapping[InternalStage, Any]) -> InternalStage:
        return next(
            (stage for stage in _ALL_STAGES if stage not in approved),
            InternalStage.ASSEMBLING_DELIVERY,
        )

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
            aggregate_episode_scripts = await self.repository.assemble_episode_scripts(work.run_id)
            l0_selection = approved[InternalStage.SELECTING_L0_VARIANT]
            l0_gate = approved[InternalStage.ACCEPTING_L0]
            l4_gate = approved[InternalStage.ACCEPTING_L4]
            expected = WorkflowResult.model_validate(
                {
                    "content_package": {
                        "story_outline": approved[InternalStage.GENERATING_STORY_OUTLINE][
                            "content"
                        ],
                        "character_biographies": approved[
                            InternalStage.GENERATING_CHARACTER_BIOGRAPHIES
                        ]["content"],
                        "relationship_logic": approved[InternalStage.GENERATING_RELATIONSHIP_LOGIC][
                            "content"
                        ],
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
        if (
            approved[InternalStage.GENERATING_EPISODE_SCRIPTS]["content"]
            != aggregate_episode_scripts
        ):
            raise AgentProtocolError(
                "Approved episode scripts differ from committed episode drafts",
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
        return "structured_output_invalid", "The agent returned invalid structured output."
    if isinstance(exc, QualityGateRejectedError):
        return "quality_gate_rejected", "The final quality gate rejected the generated work."
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
        return "structured_output_invalid", "The agent returned invalid structured output."
    if type(exc).__module__.startswith(("anthropic", "httpx", "httpcore")):
        relay_error = classify_relay_exception(exc)
        return relay_error.code, relay_error.safe_message
    return "internal_error", "The workflow failed safely."
