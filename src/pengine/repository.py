from __future__ import annotations

import hashlib
import json
from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Literal, TypeVar
from uuid import UUID, uuid4

import aiosqlite
from pydantic import BaseModel

from pengine.errors import DomainError
from pengine.schemas import (
    CreateCreationRequest,
    CreationAccepted,
    CreationResource,
    Delivery,
    FailedRun,
    InternalStage,
    PersonaSnapshot,
    QueuedRun,
    RevisionAccepted,
    RevisionAvailable,
    RevisionFailed,
    RevisionQueued,
    RevisionRequest,
    RevisionRunning,
    RevisionSucceeded,
    RevisionUnavailable,
    RunFailure,
    RunningRun,
    SucceededRun,
)

SCHEMA_VERSION = 1
MAX_STAGE_ATTEMPTS = 3

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS pengine_schema (
    version INTEGER PRIMARY KEY
);

INSERT OR IGNORE INTO pengine_schema(version) VALUES (1);

CREATE TABLE IF NOT EXISTS creations (
    id TEXT PRIMARY KEY,
    persona_id TEXT NOT NULL,
    persona_display_name TEXT NOT NULL,
    persona_version TEXT NOT NULL,
    persona_snapshot_sha256 TEXT NOT NULL,
    story TEXT NOT NULL,
    requirements TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS runs (
    id TEXT PRIMARY KEY,
    creation_id TEXT NOT NULL REFERENCES creations(id) ON DELETE CASCADE,
    kind TEXT NOT NULL CHECK (kind IN ('initial', 'revision')),
    sequence INTEGER NOT NULL,
    thread_id TEXT NOT NULL UNIQUE,
    state TEXT NOT NULL CHECK (state IN ('queued', 'running', 'succeeded', 'failed')),
    failure_code TEXT,
    failure_message TEXT,
    failed_stage TEXT,
    failure_attempt_count INTEGER,
    created_at TEXT NOT NULL,
    started_at TEXT,
    completed_at TEXT,
    updated_at TEXT NOT NULL,
    UNIQUE (creation_id, kind, sequence),
    CHECK (
        (kind = 'initial' AND sequence = 0)
        OR (kind = 'revision' AND sequence >= 1)
    ),
    CHECK (
        (state = 'failed'
         AND failure_code IS NOT NULL
         AND failure_message IS NOT NULL
         AND failed_stage IS NOT NULL
         AND failure_attempt_count IS NOT NULL)
        OR (state <> 'failed'
            AND failure_code IS NULL
            AND failure_message IS NULL
            AND failed_stage IS NULL
            AND failure_attempt_count IS NULL)
    )
);

CREATE UNIQUE INDEX IF NOT EXISTS runs_one_initial
ON runs(creation_id)
WHERE kind = 'initial';

CREATE UNIQUE INDEX IF NOT EXISTS runs_one_successful_revision
ON runs(creation_id)
WHERE kind = 'revision' AND state = 'succeeded';

CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL UNIQUE REFERENCES runs(id) ON DELETE CASCADE,
    state TEXT NOT NULL CHECK (state IN ('queued', 'leased', 'succeeded', 'failed')),
    available_at TEXT NOT NULL,
    lease_owner TEXT,
    lease_expires_at TEXT,
    lease_count INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    CHECK (
        (state = 'leased' AND lease_owner IS NOT NULL AND lease_expires_at IS NOT NULL)
        OR (state <> 'leased' AND lease_owner IS NULL AND lease_expires_at IS NULL)
    )
);

CREATE INDEX IF NOT EXISTS jobs_queue_order
ON jobs(state, available_at, created_at);

CREATE TABLE IF NOT EXISTS stage_attempts (
    run_id TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    stage TEXT NOT NULL,
    attempt_number INTEGER NOT NULL CHECK (
        attempt_number >= 1 AND attempt_number <= 3
    ),
    recorded_at TEXT NOT NULL,
    PRIMARY KEY (run_id, stage, attempt_number)
);

CREATE TABLE IF NOT EXISTS business_checkpoints (
    run_id TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    stage TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    payload_sha256 TEXT NOT NULL,
    approved_at TEXT NOT NULL,
    PRIMARY KEY (run_id, stage)
);

CREATE TABLE IF NOT EXISTS deliveries (
    run_id TEXT PRIMARY KEY REFERENCES runs(id) ON DELETE CASCADE,
    content_package_json TEXT NOT NULL,
    delivery_report_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS frozen_revisions (
    creation_id TEXT PRIMARY KEY REFERENCES creations(id) ON DELETE CASCADE,
    feedback TEXT NOT NULL,
    feedback_sha256 TEXT NOT NULL,
    succeeded_run_id TEXT UNIQUE REFERENCES runs(id),
    frozen_at TEXT NOT NULL,
    consumed_at TEXT
);

CREATE TABLE IF NOT EXISTS idempotency_records (
    idempotency_key TEXT PRIMARY KEY,
    command_scope TEXT NOT NULL,
    payload_sha256 TEXT NOT NULL,
    response_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);
"""

ModelT = TypeVar("ModelT", bound=BaseModel)


@dataclass(frozen=True, slots=True)
class LeasedJob:
    job_id: UUID
    run_id: UUID
    creation_id: UUID
    run_kind: Literal["initial", "revision"]
    run_sequence: int
    thread_id: str
    lease_owner: str
    lease_expires_at: datetime


@dataclass(frozen=True, slots=True)
class RunRecovery:
    run_id: UUID
    creation_id: UUID
    thread_id: str
    state: Literal["queued", "running"]
    stage_attempts: Mapping[InternalStage, int]
    business_checkpoints: Mapping[InternalStage, Any]


@dataclass(frozen=True, slots=True)
class RunWorkItem:
    run_id: UUID
    creation_id: UUID
    run_kind: Literal["initial", "revision"]
    run_sequence: int
    thread_id: str
    state: Literal["queued", "running", "succeeded", "failed"]
    persona: PersonaSnapshot
    story: str
    requirements: str
    frozen_feedback: str | None
    stage_attempts: Mapping[InternalStage, int]
    business_checkpoints: Mapping[InternalStage, Any]


Job = LeasedJob


def canonical_payload_hash(payload: BaseModel | Mapping[str, Any]) -> str:
    value = payload.model_dump(mode="json") if isinstance(payload, BaseModel) else dict(payload)
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _text_hash(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _json(value: Any) -> str:
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json")
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _timestamp(value: datetime) -> str:
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value.astimezone(UTC).isoformat()


def _datetime(value: str) -> datetime:
    return datetime.fromisoformat(value)


class Repository:
    def __init__(self, database_path: str | Path) -> None:
        self.database_path = Path(database_path)

    @asynccontextmanager
    async def _connection(self) -> AsyncIterator[aiosqlite.Connection]:
        connection = await aiosqlite.connect(self.database_path)
        connection.row_factory = aiosqlite.Row
        await connection.execute("PRAGMA foreign_keys = ON")
        await connection.execute("PRAGMA busy_timeout = 5000")
        try:
            yield connection
        finally:
            await connection.close()

    @asynccontextmanager
    async def _transaction(self) -> AsyncIterator[aiosqlite.Connection]:
        async with self._connection() as connection:
            await connection.execute("BEGIN IMMEDIATE")
            try:
                yield connection
            except BaseException:
                await connection.rollback()
                raise
            else:
                await connection.commit()

    async def initialize(self) -> None:
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        async with self._connection() as connection:
            await connection.execute("PRAGMA journal_mode = WAL")
            await connection.execute("PRAGMA synchronous = NORMAL")
            await connection.executescript(_SCHEMA_SQL)
            await connection.commit()

            cursor = await connection.execute("SELECT MAX(version) FROM pengine_schema")
            row = await cursor.fetchone()
            if row is None or row[0] != SCHEMA_VERSION:
                raise RuntimeError("Unsupported pengine SQLite schema version")

    async def setup(self) -> None:
        await self.initialize()

    async def replay_create_creation(
        self,
        idempotency_key: str,
        request: CreateCreationRequest,
    ) -> CreationAccepted | None:
        async with self._connection() as connection:
            return await self._idempotency_replay(
                connection,
                idempotency_key,
                "create",
                canonical_payload_hash(request),
                CreationAccepted,
            )

    async def create_creation(
        self,
        idempotency_key: str,
        request: CreateCreationRequest,
        persona_snapshot: PersonaSnapshot,
        *,
        payload_hash: str | None = None,
        creation_id: UUID | None = None,
        thread_id: str | None = None,
        now: datetime | None = None,
    ) -> CreationAccepted:
        if request.persona_id != persona_snapshot.persona_id:
            raise DomainError(
                "persona_not_found",
                "The selected persona does not match the resolved snapshot.",
                404,
            )
        request_hash = payload_hash or canonical_payload_hash(request)
        timestamp = _timestamp(now or _utc_now())
        scope = "create"

        async with self._transaction() as connection:
            replay = await self._idempotency_replay(
                connection,
                idempotency_key,
                scope,
                request_hash,
                CreationAccepted,
            )
            if replay is not None:
                return replay

            new_creation_id = creation_id or uuid4()
            run_id = uuid4()
            new_thread_id = thread_id or str(uuid4())
            job_id = uuid4()
            response = CreationAccepted(
                creation_id=new_creation_id,
                resource_url=f"/creations/{new_creation_id}",
            )

            await connection.execute(
                """
                INSERT INTO creations(
                    id, persona_id, persona_display_name, persona_version,
                    persona_snapshot_sha256, story, requirements, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(new_creation_id),
                    persona_snapshot.persona_id,
                    persona_snapshot.display_name,
                    persona_snapshot.version,
                    persona_snapshot.snapshot_sha256,
                    request.story,
                    request.requirements,
                    timestamp,
                    timestamp,
                ),
            )
            await connection.execute(
                """
                INSERT INTO runs(
                    id, creation_id, kind, sequence, thread_id, state,
                    created_at, updated_at
                ) VALUES (?, ?, 'initial', 0, ?, 'queued', ?, ?)
                """,
                (str(run_id), str(new_creation_id), new_thread_id, timestamp, timestamp),
            )
            await connection.execute(
                """
                INSERT INTO jobs(
                    id, run_id, state, available_at, created_at, updated_at
                ) VALUES (?, ?, 'queued', ?, ?, ?)
                """,
                (str(job_id), str(run_id), timestamp, timestamp, timestamp),
            )
            await self._store_idempotency(
                connection,
                idempotency_key,
                scope,
                request_hash,
                response,
                timestamp,
            )
            return response

    async def queue_revision(
        self,
        creation_id: UUID,
        idempotency_key: str,
        feedback: RevisionRequest | str,
        *,
        payload_hash: str | None = None,
        thread_id: str | None = None,
        now: datetime | None = None,
    ) -> RevisionAccepted:
        request = (
            feedback
            if isinstance(feedback, RevisionRequest)
            else RevisionRequest(feedback=feedback)
        )
        return await self.create_or_retry_revision(
            creation_id=creation_id,
            idempotency_key=idempotency_key,
            request=request,
            payload_hash=payload_hash,
            thread_id=thread_id,
            now=now,
        )

    async def create_or_retry_revision(
        self,
        *,
        creation_id: UUID,
        idempotency_key: str,
        request: RevisionRequest,
        payload_hash: str | None = None,
        thread_id: str | None = None,
        now: datetime | None = None,
    ) -> RevisionAccepted:
        request_hash = payload_hash or canonical_payload_hash(request)
        timestamp = _timestamp(now or _utc_now())
        scope = f"revision:{creation_id}"

        async with self._transaction() as connection:
            replay = await self._idempotency_replay(
                connection,
                idempotency_key,
                scope,
                request_hash,
                RevisionAccepted,
            )
            if replay is not None:
                return replay

            initial = await self._fetch_initial_run(connection, creation_id)
            if initial is None:
                raise DomainError("creation_not_found", "Creation not found.", 404)
            if initial["state"] != "succeeded":
                raise DomainError(
                    "revision_not_allowed",
                    "Revision is unavailable until the initial run succeeds.",
                    409,
                )

            frozen = await self._fetchone(
                connection,
                "SELECT * FROM frozen_revisions WHERE creation_id = ?",
                (str(creation_id),),
            )
            latest = await self._fetch_latest_revision_run(connection, creation_id)

            if frozen is None:
                await connection.execute(
                    """
                    INSERT INTO frozen_revisions(
                        creation_id, feedback, feedback_sha256, frozen_at
                    ) VALUES (?, ?, ?, ?)
                    """,
                    (str(creation_id), request.feedback, _text_hash(request.feedback), timestamp),
                )
                sequence = 1
            else:
                if frozen["succeeded_run_id"] is not None or (
                    latest is not None and latest["state"] == "succeeded"
                ):
                    raise DomainError(
                        "revision_not_allowed",
                        "The revision entitlement has already been consumed.",
                        409,
                    )
                if frozen["feedback"] != request.feedback or frozen[
                    "feedback_sha256"
                ] != _text_hash(request.feedback):
                    raise DomainError(
                        "revision_feedback_locked",
                        "Revision feedback differs from the frozen feedback.",
                        409,
                    )
                if latest is None or latest["state"] != "failed":
                    raise DomainError(
                        "revision_not_allowed",
                        "A revision is already queued or running.",
                        409,
                    )
                sequence = int(latest["sequence"]) + 1

            run_id = uuid4()
            job_id = uuid4()
            new_thread_id = thread_id or str(uuid4())
            response = RevisionAccepted(
                creation_id=creation_id,
                resource_url=f"/creations/{creation_id}",
            )
            await connection.execute(
                """
                INSERT INTO runs(
                    id, creation_id, kind, sequence, thread_id, state,
                    created_at, updated_at
                ) VALUES (?, ?, 'revision', ?, ?, 'queued', ?, ?)
                """,
                (
                    str(run_id),
                    str(creation_id),
                    sequence,
                    new_thread_id,
                    timestamp,
                    timestamp,
                ),
            )
            await connection.execute(
                """
                INSERT INTO jobs(
                    id, run_id, state, available_at, created_at, updated_at
                ) VALUES (?, ?, 'queued', ?, ?, ?)
                """,
                (str(job_id), str(run_id), timestamp, timestamp, timestamp),
            )
            await connection.execute(
                "UPDATE creations SET updated_at = ? WHERE id = ?",
                (timestamp, str(creation_id)),
            )
            await self._store_idempotency(
                connection,
                idempotency_key,
                scope,
                request_hash,
                response,
                timestamp,
            )
            return response

    async def lease_next_job(
        self,
        worker_id: str,
        lease_seconds: int,
        *,
        now: datetime | None = None,
    ) -> LeasedJob | None:
        current = now or _utc_now()
        timestamp = _timestamp(current)
        expires_at = current + timedelta(seconds=lease_seconds)
        expires_timestamp = _timestamp(expires_at)

        async with self._transaction() as connection:
            row = await self._fetchone(
                connection,
                """
                SELECT
                    jobs.id AS job_id,
                    jobs.run_id,
                    runs.creation_id,
                    runs.kind,
                    runs.sequence,
                    runs.thread_id
                FROM jobs
                JOIN runs ON runs.id = jobs.run_id
                WHERE jobs.state = 'queued'
                  AND jobs.available_at <= ?
                  AND runs.state IN ('queued', 'running')
                ORDER BY jobs.created_at, jobs.id
                LIMIT 1
                """,
                (timestamp,),
            )
            if row is None:
                return None

            await connection.execute(
                """
                UPDATE jobs
                SET state = 'leased',
                    lease_owner = ?,
                    lease_expires_at = ?,
                    lease_count = lease_count + 1,
                    updated_at = ?
                WHERE id = ? AND state = 'queued'
                """,
                (worker_id, expires_timestamp, timestamp, row["job_id"]),
            )
            await connection.execute(
                """
                UPDATE runs
                SET state = 'running',
                    started_at = COALESCE(started_at, ?),
                    updated_at = ?
                WHERE id = ? AND state = 'queued'
                """,
                (timestamp, timestamp, row["run_id"]),
            )
            await connection.execute(
                "UPDATE creations SET updated_at = ? WHERE id = ?",
                (timestamp, row["creation_id"]),
            )
            return LeasedJob(
                job_id=UUID(row["job_id"]),
                run_id=UUID(row["run_id"]),
                creation_id=UUID(row["creation_id"]),
                run_kind=row["kind"],
                run_sequence=int(row["sequence"]),
                thread_id=row["thread_id"],
                lease_owner=worker_id,
                lease_expires_at=expires_at,
            )

    async def mark_run_running(
        self,
        run_id: UUID,
        *,
        now: datetime | None = None,
    ) -> None:
        timestamp = _timestamp(now or _utc_now())
        async with self._transaction() as connection:
            run = await self._fetchone(
                connection,
                "SELECT state, creation_id FROM runs WHERE id = ?",
                (str(run_id),),
            )
            if run is None:
                raise DomainError("run_not_found", "Workflow run not found.", 404)
            if run["state"] == "running":
                return
            if run["state"] != "queued":
                raise DomainError("run_already_completed", "Workflow run is already terminal.", 409)
            await connection.execute(
                """
                UPDATE runs
                SET state = 'running',
                    started_at = COALESCE(started_at, ?),
                    updated_at = ?
                WHERE id = ?
                """,
                (timestamp, timestamp, str(run_id)),
            )
            await connection.execute(
                "UPDATE creations SET updated_at = ? WHERE id = ?",
                (timestamp, run["creation_id"]),
            )

    async def renew_job_lease(
        self,
        *,
        job_id: UUID,
        worker_id: str,
        lease_seconds: int,
        now: datetime | None = None,
    ) -> datetime:
        current = now or _utc_now()
        expires_at = current + timedelta(seconds=lease_seconds)
        async with self._transaction() as connection:
            cursor = await connection.execute(
                """
                UPDATE jobs
                SET lease_expires_at = ?, updated_at = ?
                WHERE id = ? AND state = 'leased' AND lease_owner = ?
                """,
                (
                    _timestamp(expires_at),
                    _timestamp(current),
                    str(job_id),
                    worker_id,
                ),
            )
            if cursor.rowcount != 1:
                raise DomainError("job_lease_lost", "The job lease is no longer owned.", 409)
        return expires_at

    async def requeue_expired_jobs(self, *, now: datetime | None = None) -> int:
        timestamp = _timestamp(now or _utc_now())
        async with self._transaction() as connection:
            cursor = await connection.execute(
                """
                UPDATE jobs
                SET state = 'queued',
                    available_at = ?,
                    lease_owner = NULL,
                    lease_expires_at = NULL,
                    updated_at = ?
                WHERE state = 'leased' AND lease_expires_at <= ?
                """,
                (timestamp, timestamp, timestamp),
            )
            return cursor.rowcount

    async def record_stage_attempt(
        self,
        run_id: UUID,
        stage: InternalStage,
        *,
        now: datetime | None = None,
    ) -> int:
        timestamp = _timestamp(now or _utc_now())
        async with self._transaction() as connection:
            run = await self._fetchone(
                connection,
                "SELECT state FROM runs WHERE id = ?",
                (str(run_id),),
            )
            if run is None:
                raise DomainError("run_not_found", "Workflow run not found.", 404)
            if run["state"] != "running":
                raise DomainError("run_not_running", "Workflow run is not running.", 409)
            checkpoint = await self._fetchone(
                connection,
                """
                SELECT 1 FROM business_checkpoints
                WHERE run_id = ? AND stage = ?
                """,
                (str(run_id), stage.value),
            )
            if checkpoint is not None:
                raise DomainError(
                    "stage_already_approved",
                    "The business stage is already approved.",
                    409,
                )

            cursor = await connection.execute(
                """
                SELECT COALESCE(MAX(attempt_number), 0)
                FROM stage_attempts
                WHERE run_id = ? AND stage = ?
                """,
                (str(run_id), stage.value),
            )
            row = await cursor.fetchone()
            current_count = int(row[0]) if row is not None else 0
            if current_count >= MAX_STAGE_ATTEMPTS:
                raise DomainError(
                    "attempts_exhausted",
                    "The stage attempt limit has been exhausted.",
                    409,
                )

            attempt_number = current_count + 1
            await connection.execute(
                """
                INSERT INTO stage_attempts(run_id, stage, attempt_number, recorded_at)
                VALUES (?, ?, ?, ?)
                """,
                (str(run_id), stage.value, attempt_number, timestamp),
            )
            return attempt_number

    async def approve_business_checkpoint(
        self,
        run_id: UUID,
        stage: InternalStage,
        payload: BaseModel | Mapping[str, Any],
        *,
        now: datetime | None = None,
    ) -> Any:
        payload_json = _json(payload)
        payload_hash = hashlib.sha256(payload_json.encode()).hexdigest()
        timestamp = _timestamp(now or _utc_now())
        async with self._transaction() as connection:
            run = await self._fetchone(
                connection,
                "SELECT state FROM runs WHERE id = ?",
                (str(run_id),),
            )
            if run is None:
                raise DomainError("run_not_found", "Workflow run not found.", 404)
            if run["state"] != "running":
                raise DomainError("run_not_running", "Workflow run is not running.", 409)

            existing = await self._fetchone(
                connection,
                """
                SELECT payload_json, payload_sha256
                FROM business_checkpoints
                WHERE run_id = ? AND stage = ?
                """,
                (str(run_id), stage.value),
            )
            if existing is not None:
                if existing["payload_sha256"] != payload_hash:
                    raise DomainError(
                        "checkpoint_conflict",
                        "An approved checkpoint cannot be replaced.",
                        409,
                    )
                return json.loads(existing["payload_json"])

            await connection.execute(
                """
                INSERT INTO business_checkpoints(
                    run_id, stage, payload_json, payload_sha256, approved_at
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (str(run_id), stage.value, payload_json, payload_hash, timestamp),
            )
            return json.loads(payload_json)

    async def approve_checkpoint(
        self,
        run_id: UUID,
        stage: InternalStage,
        payload: BaseModel | Mapping[str, Any],
        *,
        now: datetime | None = None,
    ) -> Any:
        return await self.approve_business_checkpoint(run_id, stage, payload, now=now)

    async def get_business_checkpoints(
        self,
        run_id: UUID,
    ) -> dict[InternalStage, Any]:
        async with self._connection() as connection:
            cursor = await connection.execute(
                """
                SELECT stage, payload_json
                FROM business_checkpoints
                WHERE run_id = ?
                ORDER BY approved_at, stage
                """,
                (str(run_id),),
            )
            rows = await cursor.fetchall()
        return {InternalStage(row["stage"]): json.loads(row["payload_json"]) for row in rows}

    async def get_stage_attempt_counts(
        self,
        run_id: UUID,
    ) -> dict[InternalStage, int]:
        async with self._connection() as connection:
            cursor = await connection.execute(
                """
                SELECT stage, COUNT(*) AS attempt_count
                FROM stage_attempts
                WHERE run_id = ?
                GROUP BY stage
                """,
                (str(run_id),),
            )
            rows = await cursor.fetchall()
        return {InternalStage(row["stage"]): int(row["attempt_count"]) for row in rows}

    async def succeed_run(
        self,
        run_id: UUID,
        delivery: Delivery,
        *,
        now: datetime | None = None,
    ) -> None:
        timestamp = _timestamp(now or _utc_now())
        async with self._transaction() as connection:
            run = await self._fetchone(
                connection,
                """
                SELECT runs.*, creations.persona_id, creations.persona_version,
                       creations.persona_snapshot_sha256
                FROM runs
                JOIN creations ON creations.id = runs.creation_id
                WHERE runs.id = ?
                """,
                (str(run_id),),
            )
            if run is None:
                raise DomainError("run_not_found", "Workflow run not found.", 404)

            self._validate_delivery_identity(run, delivery)
            if run["kind"] == "initial" and delivery.delivery_report.feedback_handling:
                raise DomainError(
                    "invalid_delivery",
                    "An initial delivery cannot contain revision feedback handling.",
                    409,
                )
            if run["kind"] == "revision" and not delivery.delivery_report.feedback_handling:
                raise DomainError(
                    "invalid_delivery",
                    "A revised delivery requires itemized feedback handling.",
                    409,
                )

            if run["state"] == "succeeded":
                existing = await self._load_delivery(connection, run_id)
                if existing == delivery:
                    return
                raise DomainError(
                    "run_already_completed",
                    "A successful run cannot be replaced.",
                    409,
                )
            if run["state"] != "running":
                raise DomainError("run_not_running", "Workflow run is not running.", 409)

            await connection.execute(
                """
                INSERT INTO deliveries(
                    run_id, content_package_json, delivery_report_json, created_at
                ) VALUES (?, ?, ?, ?)
                """,
                (
                    str(run_id),
                    _json(delivery.content_package),
                    _json(delivery.delivery_report),
                    timestamp,
                ),
            )
            await connection.execute(
                """
                UPDATE runs
                SET state = 'succeeded', completed_at = ?, updated_at = ?
                WHERE id = ?
                """,
                (timestamp, timestamp, str(run_id)),
            )
            await connection.execute(
                """
                UPDATE jobs
                SET state = 'succeeded',
                    lease_owner = NULL,
                    lease_expires_at = NULL,
                    updated_at = ?
                WHERE run_id = ?
                """,
                (timestamp, str(run_id)),
            )
            if run["kind"] == "revision":
                await connection.execute(
                    """
                    UPDATE frozen_revisions
                    SET succeeded_run_id = ?, consumed_at = ?
                    WHERE creation_id = ? AND succeeded_run_id IS NULL
                    """,
                    (str(run_id), timestamp, run["creation_id"]),
                )
            await connection.execute(
                "UPDATE creations SET updated_at = ? WHERE id = ?",
                (timestamp, run["creation_id"]),
            )

    async def fail_run(
        self,
        run_id: UUID,
        failure: RunFailure,
        *,
        now: datetime | None = None,
    ) -> None:
        timestamp = _timestamp(now or _utc_now())
        async with self._transaction() as connection:
            run = await self._fetchone(
                connection,
                "SELECT * FROM runs WHERE id = ?",
                (str(run_id),),
            )
            if run is None:
                raise DomainError("run_not_found", "Workflow run not found.", 404)
            if run["state"] == "failed":
                if self._failure_from_row(run) == failure:
                    return
                raise DomainError(
                    "run_already_completed",
                    "A failed run cannot be replaced.",
                    409,
                )
            if run["state"] not in {"queued", "running"}:
                raise DomainError(
                    "run_already_completed",
                    "A successful run cannot be failed.",
                    409,
                )

            await connection.execute(
                """
                UPDATE runs
                SET state = 'failed',
                    failure_code = ?,
                    failure_message = ?,
                    failed_stage = ?,
                    failure_attempt_count = ?,
                    completed_at = ?,
                    updated_at = ?
                WHERE id = ?
                """,
                (
                    failure.code,
                    failure.message,
                    failure.failed_stage.value,
                    failure.attempt_count,
                    timestamp,
                    timestamp,
                    str(run_id),
                ),
            )
            await connection.execute(
                """
                UPDATE jobs
                SET state = 'failed',
                    lease_owner = NULL,
                    lease_expires_at = NULL,
                    updated_at = ?
                WHERE run_id = ?
                """,
                (timestamp, str(run_id)),
            )
            await connection.execute(
                "UPDATE creations SET updated_at = ? WHERE id = ?",
                (timestamp, run["creation_id"]),
            )

    async def get_creation(self, creation_id: UUID) -> CreationResource | None:
        async with self._connection() as connection:
            creation = await self._fetchone(
                connection,
                "SELECT * FROM creations WHERE id = ?",
                (str(creation_id),),
            )
            if creation is None:
                return None

            initial = await self._fetch_initial_run(connection, creation_id)
            if initial is None:
                raise RuntimeError("Creation is missing its initial run")
            initial_status = await self._run_status(connection, initial)

            if initial["state"] != "succeeded":
                revision_status = RevisionUnavailable()
            else:
                frozen = await self._fetchone(
                    connection,
                    "SELECT * FROM frozen_revisions WHERE creation_id = ?",
                    (str(creation_id),),
                )
                if frozen is None:
                    revision_status = RevisionAvailable()
                else:
                    revision_run = await self._fetch_latest_revision_run(
                        connection,
                        creation_id,
                    )
                    if revision_run is None:
                        raise RuntimeError("Frozen revision is missing its workflow run")
                    revision_status = await self._revision_status(connection, revision_run)

            return CreationResource(
                creation_id=creation_id,
                persona=PersonaSnapshot(
                    persona_id=creation["persona_id"],
                    display_name=creation["persona_display_name"],
                    version=creation["persona_version"],
                    snapshot_sha256=creation["persona_snapshot_sha256"],
                ),
                initial=initial_status,
                revision=revision_status,
                created_at=_datetime(creation["created_at"]),
                updated_at=_datetime(creation["updated_at"]),
            )

    async def get_run_work_item(self, run_id: UUID) -> RunWorkItem:
        async with self._connection() as connection:
            run = await self._fetchone(
                connection,
                """
                SELECT
                    runs.*,
                    creations.persona_id,
                    creations.persona_display_name,
                    creations.persona_version,
                    creations.persona_snapshot_sha256,
                    creations.story,
                    creations.requirements,
                    frozen_revisions.feedback AS frozen_feedback
                FROM runs
                JOIN creations ON creations.id = runs.creation_id
                LEFT JOIN frozen_revisions
                    ON frozen_revisions.creation_id = runs.creation_id
                   AND runs.kind = 'revision'
                WHERE runs.id = ?
                """,
                (str(run_id),),
            )
            if run is None:
                raise DomainError("run_not_found", "Workflow run not found.", 404)
        return RunWorkItem(
            run_id=run_id,
            creation_id=UUID(run["creation_id"]),
            run_kind=run["kind"],
            run_sequence=int(run["sequence"]),
            thread_id=run["thread_id"],
            state=run["state"],
            persona=PersonaSnapshot(
                persona_id=run["persona_id"],
                display_name=run["persona_display_name"],
                version=run["persona_version"],
                snapshot_sha256=run["persona_snapshot_sha256"],
            ),
            story=run["story"],
            requirements=run["requirements"],
            frozen_feedback=run["frozen_feedback"],
            stage_attempts=await self.get_stage_attempt_counts(run_id),
            business_checkpoints=await self.get_business_checkpoints(run_id),
        )

    async def get_run_recovery(self, run_id: UUID) -> RunRecovery:
        async with self._connection() as connection:
            run = await self._fetchone(
                connection,
                """
                SELECT id, creation_id, thread_id, state
                FROM runs
                WHERE id = ? AND state IN ('queued', 'running')
                """,
                (str(run_id),),
            )
            if run is None:
                raise DomainError(
                    "run_not_recoverable",
                    "Workflow run is not recoverable.",
                    409,
                )
        return RunRecovery(
            run_id=run_id,
            creation_id=UUID(run["creation_id"]),
            thread_id=run["thread_id"],
            state=run["state"],
            stage_attempts=await self.get_stage_attempt_counts(run_id),
            business_checkpoints=await self.get_business_checkpoints(run_id),
        )

    async def reconcile_startup(
        self,
        *,
        now: datetime | None = None,
    ) -> list[RunRecovery]:
        await self.requeue_expired_jobs(now=now)
        timestamp = _timestamp(now or _utc_now())
        async with self._connection() as connection:
            cursor = await connection.execute(
                """
                SELECT runs.id
                FROM jobs
                JOIN runs ON runs.id = jobs.run_id
                WHERE jobs.state = 'queued'
                  AND jobs.available_at <= ?
                  AND runs.state IN ('queued', 'running')
                ORDER BY jobs.created_at, jobs.id
                """,
                (timestamp,),
            )
            rows = await cursor.fetchall()
        return [await self.get_run_recovery(UUID(row["id"])) for row in rows]

    async def _idempotency_replay(
        self,
        connection: aiosqlite.Connection,
        idempotency_key: str,
        command_scope: str,
        payload_hash: str,
        model_type: type[ModelT],
    ) -> ModelT | None:
        row = await self._fetchone(
            connection,
            """
            SELECT command_scope, payload_sha256, response_json
            FROM idempotency_records
            WHERE idempotency_key = ?
            """,
            (idempotency_key,),
        )
        if row is None:
            return None
        if row["command_scope"] != command_scope or row["payload_sha256"] != payload_hash:
            raise DomainError(
                "idempotency_conflict",
                "Idempotency key was reused with a different command or payload.",
                409,
            )
        return model_type.model_validate_json(row["response_json"])

    async def _store_idempotency(
        self,
        connection: aiosqlite.Connection,
        key: str,
        scope: str,
        payload_hash: str,
        response: BaseModel,
        timestamp: str,
    ) -> None:
        await connection.execute(
            """
            INSERT INTO idempotency_records(
                idempotency_key, command_scope, payload_sha256,
                response_json, created_at
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (key, scope, payload_hash, response.model_dump_json(), timestamp),
        )

    async def _run_status(
        self,
        connection: aiosqlite.Connection,
        run: aiosqlite.Row,
    ) -> QueuedRun | RunningRun | SucceededRun | FailedRun:
        match run["state"]:
            case "queued":
                return QueuedRun()
            case "running":
                return RunningRun()
            case "succeeded":
                delivery = await self._load_delivery(connection, UUID(run["id"]))
                if delivery is None:
                    raise RuntimeError("Successful run is missing its delivery")
                return SucceededRun(result=delivery)
            case "failed":
                return FailedRun(failure=self._failure_from_row(run))
            case _:
                raise RuntimeError("Unknown workflow run state")

    async def _revision_status(
        self,
        connection: aiosqlite.Connection,
        run: aiosqlite.Row,
    ) -> RevisionQueued | RevisionRunning | RevisionSucceeded | RevisionFailed:
        match run["state"]:
            case "queued":
                return RevisionQueued()
            case "running":
                return RevisionRunning()
            case "succeeded":
                delivery = await self._load_delivery(connection, UUID(run["id"]))
                if delivery is None:
                    raise RuntimeError("Successful revision is missing its delivery")
                return RevisionSucceeded(result=delivery)
            case "failed":
                return RevisionFailed(failure=self._failure_from_row(run))
            case _:
                raise RuntimeError("Unknown revision run state")

    async def _load_delivery(
        self,
        connection: aiosqlite.Connection,
        run_id: UUID,
    ) -> Delivery | None:
        row = await self._fetchone(
            connection,
            """
            SELECT content_package_json, delivery_report_json
            FROM deliveries
            WHERE run_id = ?
            """,
            (str(run_id),),
        )
        if row is None:
            return None
        return Delivery.model_validate(
            {
                "content_package": json.loads(row["content_package_json"]),
                "delivery_report": json.loads(row["delivery_report_json"]),
            }
        )

    @staticmethod
    def _validate_delivery_identity(run: aiosqlite.Row, delivery: Delivery) -> None:
        report = delivery.delivery_report
        if (
            report.persona_id != run["persona_id"]
            or report.persona_version != run["persona_version"]
            or report.persona_snapshot_sha256 != run["persona_snapshot_sha256"]
        ):
            raise DomainError(
                "invalid_delivery",
                "Delivery persona identity does not match the creation snapshot.",
                409,
            )

    @staticmethod
    def _failure_from_row(run: aiosqlite.Row) -> RunFailure:
        return RunFailure(
            code=run["failure_code"],
            message=run["failure_message"],
            failed_stage=run["failed_stage"],
            attempt_count=run["failure_attempt_count"],
        )

    @staticmethod
    async def _fetchone(
        connection: aiosqlite.Connection,
        query: str,
        parameters: tuple[Any, ...],
    ) -> aiosqlite.Row | None:
        cursor = await connection.execute(query, parameters)
        return await cursor.fetchone()

    async def _fetch_initial_run(
        self,
        connection: aiosqlite.Connection,
        creation_id: UUID,
    ) -> aiosqlite.Row | None:
        return await self._fetchone(
            connection,
            """
            SELECT * FROM runs
            WHERE creation_id = ? AND kind = 'initial'
            """,
            (str(creation_id),),
        )

    async def _fetch_latest_revision_run(
        self,
        connection: aiosqlite.Connection,
        creation_id: UUID,
    ) -> aiosqlite.Row | None:
        return await self._fetchone(
            connection,
            """
            SELECT * FROM runs
            WHERE creation_id = ? AND kind = 'revision'
            ORDER BY sequence DESC
            LIMIT 1
            """,
            (str(creation_id),),
        )
