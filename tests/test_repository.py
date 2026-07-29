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
