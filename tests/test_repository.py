import sqlite3
from datetime import UTC, datetime, timedelta

import aiosqlite
import pytest

from pengine.errors import DomainError
from pengine.repository import Repository
from pengine.schemas import (
    ContentPackage,
    CreateCreationRequest,
    Delivery,
    DeliveryReport,
    FeedbackHandlingItem,
    GateResult,
    InternalStage,
    PersonaSnapshot,
    RevisionRequest,
    RunFailure,
)

NOW = datetime(2026, 7, 28, 12, 0, tzinfo=UTC)
SNAPSHOT_HASH = "a" * 64


@pytest.fixture
async def repository(tmp_path):
    value = Repository(tmp_path / "pengine.sqlite3")
    await value.initialize()
    return value


@pytest.fixture
def persona() -> PersonaSnapshot:
    return PersonaSnapshot(
        persona_id="fixture-writer",
        display_name="非生产测试人格",
        version="fixture-1",
        snapshot_sha256=SNAPSHOT_HASH,
    )


@pytest.fixture
def creation_request() -> CreateCreationRequest:
    return CreateCreationRequest(
        persona_id="fixture-writer",
        story="一个离乡的人回家处理旧屋。",
        requirements="创作一部当代短剧。",
    )


def make_delivery(*, revised: bool = False) -> Delivery:
    handling = (
        [
            FeedbackHandlingItem(
                feedback_item="加强结尾行动。",
                handling="重写终场。",
                result="主角以行动完成选择。",
            )
        ]
        if revised
        else []
    )
    return Delivery(
        content_package=ContentPackage(
            story_outline="故事梗概",
            character_biographies="人物小传",
            relationship_logic="人物关系逻辑",
            episode_outline="分集大纲",
            episode_scripts="分集剧本",
        ),
        delivery_report=DeliveryReport(
            persona_id="fixture-writer",
            persona_version="fixture-1",
            persona_snapshot_sha256=SNAPSHOT_HASH,
            selected_l0_variant="归返",
            selection_rationale="与故事输入一致。",
            l0_gate=GateResult(passed=True, evidence="L0 证据"),
            l4_gate=GateResult(passed=True, evidence="L4 证据"),
            ownership_statement="本次创作归当前任务所有。",
            feedback_handling=handling,
        ),
    )


def make_failure() -> RunFailure:
    return RunFailure(
        code="attempts_exhausted",
        message="The stage attempt limit was exhausted.",
        failed_stage=InternalStage.GENERATING_STORY_OUTLINE,
        attempt_count=3,
    )


async def create_and_lease_initial(
    repository: Repository,
    persona: PersonaSnapshot,
    request: CreateCreationRequest,
):
    accepted = await repository.create_creation(
        idempotency_key="create-1",
        request=request,
        persona_snapshot=persona,
        now=NOW,
    )
    lease = await repository.lease_next_job(
        worker_id="worker-1",
        lease_seconds=30,
        now=NOW,
    )
    assert lease is not None
    return accepted, lease


async def test_initialize_enables_wal_foreign_keys_and_domain_tables(repository) -> None:
    async with repository._connection() as connection:
        journal = await (await connection.execute("PRAGMA journal_mode")).fetchone()
        foreign_keys = await (await connection.execute("PRAGMA foreign_keys")).fetchone()
        rows = await (
            await connection.execute(
                """
                SELECT name FROM sqlite_master
                WHERE type = 'table' AND name NOT LIKE 'sqlite_%'
                """
            )
        ).fetchall()

    assert journal[0] == "wal"
    assert foreign_keys[0] == 1
    timestamp = NOW.isoformat()
    with pytest.raises(sqlite3.IntegrityError):
        async with repository._connection() as connection:
            await connection.execute(
                """
                INSERT INTO jobs(id, run_id, state, available_at, created_at, updated_at)
                VALUES ('dangling-job', 'missing-run', 'queued', ?, ?, ?)
                """,
                (timestamp, timestamp, timestamp),
            )
    assert {
        "creations",
        "runs",
        "jobs",
        "stage_attempts",
        "business_checkpoints",
        "deliveries",
        "frozen_revisions",
        "idempotency_records",
        "run_progress",
    } <= {row[0] for row in rows}


async def test_creation_is_idempotent_and_payload_conflicts(
    repository,
    persona,
    creation_request,
) -> None:
    first = await repository.create_creation(
        idempotency_key="same-key",
        request=creation_request,
        persona_snapshot=persona,
        payload_hash="payload-a",
        now=NOW,
    )
    replay = await repository.create_creation(
        idempotency_key="same-key",
        request=creation_request,
        persona_snapshot=persona,
        payload_hash="payload-a",
        now=NOW + timedelta(seconds=1),
    )

    assert replay == first

    with pytest.raises(DomainError, match="Idempotency") as error:
        await repository.create_creation(
            idempotency_key="same-key",
            request=creation_request,
            persona_snapshot=persona,
            payload_hash="payload-b",
            now=NOW + timedelta(seconds=2),
        )
    assert error.value.code == "idempotency_conflict"

    async with aiosqlite.connect(repository.database_path) as connection:
        count = await (await connection.execute("SELECT COUNT(*) FROM creations")).fetchone()
    assert count == (1,)


async def test_attempt_is_recorded_before_invocation_and_fourth_is_rejected(
    repository,
    persona,
    creation_request,
) -> None:
    _, lease = await create_and_lease_initial(repository, persona, creation_request)
    stage = InternalStage.GENERATING_STORY_OUTLINE

    assert await repository.record_stage_attempt(lease.run_id, stage) == 1
    assert await repository.record_stage_attempt(lease.run_id, stage) == 2
    assert await repository.record_stage_attempt(lease.run_id, stage) == 3

    with pytest.raises(DomainError) as error:
        await repository.record_stage_attempt(lease.run_id, stage)
    assert error.value.code == "attempts_exhausted"
    assert await repository.get_stage_attempt_counts(lease.run_id) == {stage: 3}


async def test_approved_checkpoint_is_immutable_and_blocks_regeneration(
    repository,
    persona,
    creation_request,
) -> None:
    _, lease = await create_and_lease_initial(repository, persona, creation_request)
    stage = InternalStage.SELECTING_L0_VARIANT
    await repository.record_stage_attempt(lease.run_id, stage)

    payload = {"selected_l0_variant": "归返", "selection_rationale": "匹配母题"}
    assert await repository.approve_business_checkpoint(lease.run_id, stage, payload) == payload
    assert await repository.approve_business_checkpoint(lease.run_id, stage, payload) == payload

    with pytest.raises(DomainError) as conflict:
        await repository.approve_business_checkpoint(
            lease.run_id,
            stage,
            {"selected_l0_variant": "异化"},
        )
    assert conflict.value.code == "checkpoint_conflict"

    with pytest.raises(DomainError) as approved:
        await repository.record_stage_attempt(lease.run_id, stage)
    assert approved.value.code == "stage_already_approved"


async def test_failed_revision_retry_uses_frozen_feedback_and_new_thread(
    repository,
    persona,
    creation_request,
) -> None:
    accepted, initial_lease = await create_and_lease_initial(
        repository,
        persona,
        creation_request,
    )
    await repository.succeed_run(initial_lease.run_id, make_delivery(), now=NOW)

    revision_request = RevisionRequest(feedback="  加强结尾行动。\n")
    assert revision_request.feedback == "  加强结尾行动。\n"
    await repository.create_or_retry_revision(
        creation_id=accepted.creation_id,
        idempotency_key="revision-1",
        request=revision_request,
        now=NOW + timedelta(minutes=1),
    )
    first_revision = await repository.lease_next_job(
        worker_id="worker-1",
        lease_seconds=30,
        now=NOW + timedelta(minutes=1),
    )
    assert first_revision is not None
    assert first_revision.thread_id != initial_lease.thread_id
    await repository.fail_run(first_revision.run_id, make_failure())

    with pytest.raises(DomainError) as locked:
        await repository.create_or_retry_revision(
            creation_id=accepted.creation_id,
            idempotency_key="revision-changed",
            request=RevisionRequest(feedback="加强结尾行动。"),
        )
    assert locked.value.code == "revision_feedback_locked"

    await repository.create_or_retry_revision(
        creation_id=accepted.creation_id,
        idempotency_key="revision-retry",
        request=revision_request,
    )
    retry = await repository.lease_next_job(
        worker_id="worker-1",
        lease_seconds=30,
    )
    assert retry is not None
    assert retry.run_sequence == 2
    assert retry.thread_id not in {initial_lease.thread_id, first_revision.thread_id}


async def test_revision_rules_and_initial_delivery_remains_visible(
    repository,
    persona,
    creation_request,
) -> None:
    accepted, initial_lease = await create_and_lease_initial(
        repository,
        persona,
        creation_request,
    )
    before_success = await repository.get_creation(accepted.creation_id)
    assert before_success.initial.state == "running"
    assert before_success.revision.state == "unavailable"

    initial_delivery = make_delivery()
    await repository.succeed_run(initial_lease.run_id, initial_delivery)
    after_success = await repository.get_creation(accepted.creation_id)
    assert after_success.initial.result == initial_delivery
    assert after_success.revision.state == "available"

    request = RevisionRequest(feedback="加强结尾行动。")
    await repository.create_or_retry_revision(
        creation_id=accepted.creation_id,
        idempotency_key="revision-1",
        request=request,
    )
    queued = await repository.get_creation(accepted.creation_id)
    assert queued.initial.result == initial_delivery
    assert queued.revision.state == "queued"

    with pytest.raises(DomainError) as duplicate:
        await repository.create_or_retry_revision(
            creation_id=accepted.creation_id,
            idempotency_key="revision-duplicate",
            request=request,
        )
    assert duplicate.value.code == "revision_not_allowed"

    revision_lease = await repository.lease_next_job(
        worker_id="worker-1",
        lease_seconds=30,
    )
    assert revision_lease is not None
    await repository.fail_run(revision_lease.run_id, make_failure())
    failed = await repository.get_creation(accepted.creation_id)
    assert failed.initial.result == initial_delivery
    assert failed.revision.state == "failed"


async def test_revision_becomes_available_from_authoritative_initial_status(
    repository,
    persona,
    creation_request,
    monkeypatch,
) -> None:
    accepted, initial_lease = await create_and_lease_initial(
        repository,
        persona,
        creation_request,
    )
    async with repository._connection() as connection:
        stale_initial = await repository._fetch_initial_run(
            connection,
            accepted.creation_id,
        )
    assert stale_initial is not None
    assert stale_initial["state"] == "running"

    await repository.succeed_run(initial_lease.run_id, make_delivery())

    async def fetch_stale_initial(connection, creation_id):
        return stale_initial

    monkeypatch.setattr(repository, "_fetch_initial_run", fetch_stale_initial)
    aggregate = await repository.get_creation(accepted.creation_id)

    assert aggregate.initial.state == "succeeded"
    assert aggregate.revision.state == "available"


async def test_successful_revision_consumes_entitlement(
    repository,
    persona,
    creation_request,
) -> None:
    accepted, initial_lease = await create_and_lease_initial(
        repository,
        persona,
        creation_request,
    )
    await repository.succeed_run(initial_lease.run_id, make_delivery())
    request = RevisionRequest(feedback="加强结尾行动。")
    await repository.queue_revision(
        accepted.creation_id,
        "revision-success",
        request,
    )
    revision_lease = await repository.lease_next_job("worker-1", 30)
    assert revision_lease is not None
    revised_delivery = make_delivery(revised=True)
    await repository.succeed_run(revision_lease.run_id, revised_delivery)

    aggregate = await repository.get_creation(accepted.creation_id)
    assert aggregate is not None
    assert aggregate.initial.result == make_delivery()
    assert aggregate.revision.state == "succeeded"
    assert aggregate.revision.result == revised_delivery

    with pytest.raises(DomainError) as consumed:
        await repository.queue_revision(
            accepted.creation_id,
            "revision-after-success",
            request,
        )
    assert consumed.value.code == "revision_not_allowed"


async def test_expired_lease_requeues_same_thread_with_durable_business_state(
    repository,
    persona,
    creation_request,
) -> None:
    _, lease = await create_and_lease_initial(repository, persona, creation_request)
    stage = InternalStage.GENERATING_STORY_OUTLINE
    await repository.record_stage_attempt(lease.run_id, stage, now=NOW)
    await repository.approve_business_checkpoint(
        lease.run_id,
        stage,
        {"content": "已批准的故事梗概"},
        now=NOW,
    )

    recoverable = await repository.reconcile_startup(
        now=NOW + timedelta(seconds=31),
    )
    assert len(recoverable) == 1
    assert recoverable[0].thread_id == lease.thread_id
    assert recoverable[0].stage_attempts == {stage: 1}
    assert recoverable[0].business_checkpoints == {stage: {"content": "已批准的故事梗概"}}

    resumed = await repository.lease_next_job(
        worker_id="worker-after-restart",
        lease_seconds=30,
        now=NOW + timedelta(seconds=31),
    )
    assert resumed is not None
    assert resumed.run_id == lease.run_id
    assert resumed.thread_id == lease.thread_id


async def test_progress_timeout_pause_continue_and_end_are_durable_and_idempotent(
    repository,
    persona,
    creation_request,
) -> None:
    accepted, lease = await create_and_lease_initial(
        repository,
        persona,
        creation_request,
    )
    await repository.record_stage_attempt(
        lease.run_id,
        InternalStage.SELECTING_L0_VARIANT,
        now=NOW + timedelta(seconds=1),
    )
    await repository.approve_business_checkpoint(
        lease.run_id,
        InternalStage.SELECTING_L0_VARIANT,
        {"selected_l0_variant": "归返", "selection_rationale": "匹配母题"},
        now=NOW + timedelta(seconds=2),
    )
    await repository.record_stage_attempt(
        lease.run_id,
        InternalStage.GENERATING_STORY_OUTLINE,
        now=NOW + timedelta(seconds=3),
    )

    running = await repository.get_creation(
        accepted.creation_id,
        now=NOW + timedelta(seconds=20),
    )
    assert running is not None
    assert running.initial.state == "running"
    assert running.initial.progress.current_stage == "generating_story_outline"
    assert running.initial.progress.completed_stages == ["determining_direction"]
    assert running.initial.progress.elapsed_seconds == 20
    assert not hasattr(running.initial, "result")
    assert running.initial.drafts.model_dump() == {
        "artifacts": [
            {
                "stage": "determining_direction",
                "selected_l0_variant": "归返",
                "selection_rationale": "匹配母题",
            }
        ],
        "review_status": {"l0": "pending", "l4": "pending"},
    }

    first_timeout = await repository.handle_run_timeout(
        lease.run_id,
        InternalStage.GENERATING_STORY_OUTLINE,
        now=NOW + timedelta(seconds=30),
    )
    assert first_timeout == "auto_resuming"
    recovering = await repository.get_creation(
        accepted.creation_id,
        now=NOW + timedelta(seconds=40),
    )
    assert recovering is not None
    assert recovering.initial.state == "auto_resuming"
    assert recovering.initial.progress.elapsed_seconds == 30
    assert recovering.initial.progress.recovery_state == "auto_resuming"
    assert recovering.initial.drafts == running.initial.drafts

    resumed = await repository.lease_next_job(
        "worker-2",
        30,
        now=NOW + timedelta(seconds=40),
    )
    assert resumed is not None
    await repository.record_stage_attempt(
        resumed.run_id,
        InternalStage.GENERATING_STORY_OUTLINE,
        now=NOW + timedelta(seconds=41),
    )
    second_timeout = await repository.handle_run_timeout(
        resumed.run_id,
        InternalStage.GENERATING_STORY_OUTLINE,
        now=NOW + timedelta(seconds=55),
    )
    assert second_timeout == "paused"

    paused = await repository.get_creation(accepted.creation_id)
    assert paused is not None
    assert paused.initial.state == "paused"
    assert paused.initial.pause.code == "run_timeout"
    assert paused.initial.pause.timeout_count == 2
    assert paused.initial.progress.can_continue is True
    assert paused.initial.progress.can_end is True
    assert paused.initial.drafts == running.initial.drafts

    first_continue = await repository.continue_run(
        creation_id=accepted.creation_id,
        run_kind="initial",
        idempotency_key="continue-paused",
        now=NOW + timedelta(seconds=60),
    )
    replay_continue = await repository.continue_run(
        creation_id=accepted.creation_id,
        run_kind="initial",
        idempotency_key="continue-paused",
        now=NOW + timedelta(seconds=61),
    )
    duplicate_continue = await repository.continue_run(
        creation_id=accepted.creation_id,
        run_kind="initial",
        idempotency_key="continue-paused-again",
        now=NOW + timedelta(seconds=61),
    )
    assert replay_continue == first_continue == duplicate_continue
    assert first_continue.run_state == "queued"
    continued = await repository.get_creation(accepted.creation_id)
    assert continued is not None
    assert continued.initial.state == "queued"
    assert continued.initial.drafts == running.initial.drafts

    resumed_again = await repository.lease_next_job(
        "worker-3",
        30,
        now=NOW + timedelta(seconds=62),
    )
    assert resumed_again is not None
    await repository.record_stage_attempt(
        resumed_again.run_id,
        InternalStage.GENERATING_STORY_OUTLINE,
        now=NOW + timedelta(seconds=63),
    )
    assert (
        await repository.handle_run_timeout(
            resumed_again.run_id,
            InternalStage.GENERATING_STORY_OUTLINE,
            now=NOW + timedelta(seconds=70),
        )
        == "paused"
    )

    first_end = await repository.end_run(
        creation_id=accepted.creation_id,
        run_kind="initial",
        idempotency_key="end-paused",
        now=NOW + timedelta(seconds=71),
    )
    replay_end = await repository.end_run(
        creation_id=accepted.creation_id,
        run_kind="initial",
        idempotency_key="end-paused",
        now=NOW + timedelta(seconds=72),
    )
    duplicate_end = await repository.end_run(
        creation_id=accepted.creation_id,
        run_kind="initial",
        idempotency_key="end-paused-again",
        now=NOW + timedelta(seconds=72),
    )
    assert replay_end == first_end == duplicate_end
    assert first_end.run_state == "ended"

    ended = await repository.get_creation(accepted.creation_id)
    assert ended is not None
    assert ended.initial.state == "ended"
    assert await repository.reconcile_startup(now=NOW + timedelta(minutes=5)) == []
    with pytest.raises(DomainError) as cannot_continue:
        await repository.continue_run(
            creation_id=accepted.creation_id,
            run_kind="initial",
            idempotency_key="continue-ended",
        )
    assert cannot_continue.value.code == "run_not_controllable"


async def test_drafts_are_ordered_from_valid_checkpoints_and_isolate_a_revision(
    repository,
    persona,
    creation_request,
) -> None:
    accepted, initial_lease = await create_and_lease_initial(
        repository,
        persona,
        creation_request,
    )
    await repository.succeed_run(initial_lease.run_id, make_delivery())
    await repository.create_or_retry_revision(
        creation_id=accepted.creation_id,
        idempotency_key="revision-drafts",
        request=RevisionRequest(feedback="加强人物选择。"),
    )
    revision_lease = await repository.lease_next_job("worker-2", 30)
    assert revision_lease is not None
    checkpoints = [
        (
            InternalStage.SELECTING_L0_VARIANT,
            {"selected_l0_variant": "新方向", "selection_rationale": "响应反馈"},
        ),
        (InternalStage.GENERATING_STORY_OUTLINE, {"content": "修订故事大纲"}),
        (InternalStage.GENERATING_CHARACTER_BIOGRAPHIES, {"content": "   "}),
        (InternalStage.GENERATING_RELATIONSHIP_LOGIC, {"content": "修订人物关系"}),
        (InternalStage.GENERATING_EPISODE_OUTLINE, {"content": "修订分集大纲"}),
        (InternalStage.GENERATING_EPISODE_SCRIPTS, {"content": "不应显示的分集剧本"}),
    ]
    for stage, payload in checkpoints:
        await repository.approve_business_checkpoint(revision_lease.run_id, stage, payload)

    resource = await repository.get_creation(accepted.creation_id)
    assert resource is not None
    assert resource.initial.state == "succeeded"
    assert resource.initial.result == make_delivery()
    assert resource.revision.state == "running"
    assert resource.revision.drafts.model_dump() == {
        "artifacts": [
            {
                "stage": "determining_direction",
                "selected_l0_variant": "新方向",
                "selection_rationale": "响应反馈",
            },
            {"stage": "generating_story_outline", "content": "修订故事大纲"},
            {"stage": "generating_relationships", "content": "修订人物关系"},
            {"stage": "generating_episode_outline", "content": "修订分集大纲"},
        ],
        "review_status": {"l0": "running", "l4": "pending"},
    }


async def test_schema_v1_database_is_backfilled_without_losing_creation(
    repository,
    persona,
    creation_request,
) -> None:
    accepted = await repository.create_creation(
        idempotency_key="migration-create",
        request=creation_request,
        persona_snapshot=persona,
        now=NOW,
    )
    async with repository._connection() as connection:
        await connection.execute("DROP TABLE run_progress")
        await connection.execute("DELETE FROM pengine_schema WHERE version = 2")
        await connection.commit()

    await repository.initialize()

    migrated = await repository.get_creation(accepted.creation_id, now=NOW)
    assert migrated is not None
    assert migrated.initial.state == "queued"
    assert migrated.initial.progress.current_stage == "determining_direction"
    async with repository._connection() as connection:
        version = await (
            await connection.execute("SELECT MAX(version) FROM pengine_schema")
        ).fetchone()
    assert version[0] == 2
