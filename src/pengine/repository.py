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

from pengine.continuity import (
    EpisodeLock,
    SemanticReview,
    SeriesState,
    StoryContract,
    build_episode_lock,
    canonical_model_hash,
    initial_series_state,
    render_story_contract_markdown,
    story_contract_sha256,
)
from pengine.errors import DomainError
from pengine.language import OutputLanguage, infer_output_language
from pengine.schemas import (
    AutoResumingRun,
    CreateCreationRequest,
    CreationAccepted,
    CreationResource,
    CreativeDirectionDraft,
    CreativeTextDraft,
    Delivery,
    EndedRun,
    EpisodeDraft,
    EpisodePlan,
    EpisodeProgress,
    FailedRun,
    FinalReviewProgress,
    InternalStage,
    PausedRun,
    PersonaSnapshot,
    QualityGateRejection,
    QualityRejectedRun,
    QueuedRun,
    RevisionAccepted,
    RevisionAutoResuming,
    RevisionAvailable,
    RevisionEnded,
    RevisionFailed,
    RevisionPaused,
    RevisionQualityRejected,
    RevisionQueued,
    RevisionRequest,
    RevisionRunning,
    RevisionSucceeded,
    RevisionUnavailable,
    RunControlAccepted,
    RunDraftSnapshot,
    RunFailure,
    RunningRun,
    RunPause,
    RunProgress,
    SucceededRun,
    UserStage,
)

SCHEMA_VERSION = 8
MAX_STAGE_ATTEMPTS = 3
MAX_EPISODE_ATTEMPTS = 3
_MIN_RELAY_RETRY_DELAY_SECONDS = 10

RecoveryReason = Literal[
    "run_timeout",
    "relay_interruption",
    "content_rejected",
    "episode_error",
]
RecoveryState = Literal["auto_resuming", "paused", "failed"]

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

_SCHEMA_V2_SQL = """
CREATE TABLE IF NOT EXISTS run_progress (
    run_id TEXT PRIMARY KEY REFERENCES runs(id) ON DELETE CASCADE,
    current_stage TEXT NOT NULL,
    execution_state TEXT NOT NULL CHECK (
        execution_state IN (
            'queued', 'running', 'auto_resuming', 'paused',
            'ended', 'succeeded', 'failed'
        )
    ),
    elapsed_seconds REAL NOT NULL DEFAULT 0 CHECK (elapsed_seconds >= 0),
    active_started_at TEXT,
    timeout_stage TEXT,
    timeout_count INTEGER NOT NULL DEFAULT 0 CHECK (timeout_count >= 0),
    updated_at TEXT NOT NULL
);

INSERT OR IGNORE INTO run_progress(
    run_id, current_stage, execution_state, elapsed_seconds,
    active_started_at, timeout_stage, timeout_count, updated_at
)
SELECT
    runs.id,
    COALESCE(
        runs.failed_stage,
        (
            SELECT stage_attempts.stage
            FROM stage_attempts
            WHERE stage_attempts.run_id = runs.id
            ORDER BY stage_attempts.recorded_at DESC, stage_attempts.attempt_number DESC
            LIMIT 1
        ),
        'loading_persona'
    ),
    runs.state,
    CASE
        WHEN runs.state IN ('succeeded', 'failed') AND runs.started_at IS NOT NULL
        THEN MAX(
            0,
            (julianday(COALESCE(runs.completed_at, runs.updated_at))
             - julianday(runs.started_at)) * 86400
        )
        ELSE 0
    END,
    CASE
        WHEN runs.state = 'running' THEN COALESCE(runs.started_at, runs.updated_at)
        ELSE NULL
    END,
    NULL,
    0,
    runs.updated_at
FROM runs;

INSERT OR IGNORE INTO pengine_schema(version) VALUES (2);
"""

_SCHEMA_V3_SQL = """
ALTER TABLE run_progress ADD COLUMN current_episode INTEGER
    CHECK (current_episode IS NULL OR current_episode >= 1);

CREATE TABLE IF NOT EXISTS episode_plans (
    run_id TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    episode_number INTEGER NOT NULL CHECK (episode_number >= 1),
    plan TEXT NOT NULL,
    plan_sha256 TEXT NOT NULL,
    PRIMARY KEY (run_id, episode_number)
);

CREATE TABLE IF NOT EXISTS episode_drafts (
    run_id TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    episode_number INTEGER NOT NULL CHECK (episode_number >= 1),
    content TEXT NOT NULL,
    content_sha256 TEXT NOT NULL,
    completed_at TEXT NOT NULL,
    PRIMARY KEY (run_id, episode_number)
);

CREATE TABLE IF NOT EXISTS episode_attempts (
    run_id TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    episode_number INTEGER NOT NULL CHECK (episode_number >= 1),
    attempt_number INTEGER NOT NULL CHECK (
        attempt_number >= 1 AND attempt_number <= 3
    ),
    recorded_at TEXT NOT NULL,
    PRIMARY KEY (run_id, episode_number, attempt_number)
);

CREATE TABLE IF NOT EXISTS episode_timeouts (
    run_id TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    episode_number INTEGER NOT NULL CHECK (episode_number >= 1),
    timeout_count INTEGER NOT NULL DEFAULT 0 CHECK (timeout_count >= 0),
    updated_at TEXT NOT NULL,
    PRIMARY KEY (run_id, episode_number)
);

INSERT OR IGNORE INTO pengine_schema(version) VALUES (3);
"""

_SCHEMA_V4_SQL = """
CREATE TABLE run_progress_v4 (
    run_id TEXT PRIMARY KEY REFERENCES runs(id) ON DELETE CASCADE,
    current_stage TEXT NOT NULL,
    execution_state TEXT NOT NULL CHECK (
        execution_state IN (
            'queued', 'running', 'auto_resuming', 'paused',
            'quality_rejected', 'ended', 'succeeded', 'failed'
        )
    ),
    elapsed_seconds REAL NOT NULL DEFAULT 0 CHECK (elapsed_seconds >= 0),
    active_started_at TEXT,
    timeout_stage TEXT,
    timeout_count INTEGER NOT NULL DEFAULT 0 CHECK (timeout_count >= 0),
    updated_at TEXT NOT NULL,
    current_episode INTEGER CHECK (current_episode IS NULL OR current_episode >= 1)
);

INSERT INTO run_progress_v4(
    run_id, current_stage, execution_state, elapsed_seconds,
    active_started_at, timeout_stage, timeout_count, updated_at, current_episode
)
SELECT
    run_id, current_stage, execution_state, elapsed_seconds,
    active_started_at, timeout_stage, timeout_count, updated_at, current_episode
FROM run_progress;

DROP TABLE run_progress;
ALTER TABLE run_progress_v4 RENAME TO run_progress;

CREATE TABLE quality_gate_rejections (
    run_id TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    stage TEXT NOT NULL CHECK (stage IN ('accepting_l0', 'accepting_l4')),
    attempt_number INTEGER NOT NULL CHECK (
        attempt_number >= 1 AND attempt_number <= 3
    ),
    evidence TEXT,
    rejected_at TEXT NOT NULL,
    PRIMARY KEY (run_id, stage, attempt_number)
);

INSERT OR IGNORE INTO quality_gate_rejections(
    run_id, stage, attempt_number, evidence, rejected_at
)
SELECT
    runs.id,
    runs.failed_stage,
    CASE
        WHEN runs.failure_attempt_count BETWEEN 1 AND 3 THEN runs.failure_attempt_count
        ELSE 3
    END,
    NULL,
    COALESCE(runs.completed_at, runs.updated_at)
FROM runs
WHERE runs.state = 'failed'
  AND runs.failure_code = 'quality_gate_rejected'
  AND runs.failed_stage IN ('accepting_l0', 'accepting_l4');

UPDATE runs
SET state = 'running',
    failure_code = NULL,
    failure_message = NULL,
    failed_stage = NULL,
    failure_attempt_count = NULL,
    completed_at = NULL
WHERE state = 'failed'
  AND failure_code = 'quality_gate_rejected'
  AND failed_stage IN ('accepting_l0', 'accepting_l4')
  AND EXISTS (
      SELECT 1
      FROM quality_gate_rejections
      WHERE quality_gate_rejections.run_id = runs.id
        AND quality_gate_rejections.stage = runs.failed_stage
        AND quality_gate_rejections.attempt_number = CASE
            WHEN runs.failure_attempt_count BETWEEN 1 AND 3 THEN runs.failure_attempt_count
            ELSE 3
        END
  );

UPDATE run_progress
SET execution_state = 'quality_rejected',
    active_started_at = NULL
WHERE run_id IN (
    SELECT run_id FROM quality_gate_rejections
)
  AND execution_state = 'failed';

INSERT OR IGNORE INTO pengine_schema(version) VALUES (4);
"""

_SCHEMA_V5_SQL = """
ALTER TABLE run_progress ADD COLUMN recovery_reason TEXT NOT NULL DEFAULT 'none'
    CHECK (recovery_reason IN ('none', 'run_timeout', 'relay_interruption'));

UPDATE run_progress
SET recovery_reason = 'run_timeout'
WHERE execution_state IN ('auto_resuming', 'paused');

INSERT OR IGNORE INTO pengine_schema(version) VALUES (5);
"""

_SCHEMA_V6_SQL = """
CREATE TABLE run_progress_v6 (
    run_id TEXT PRIMARY KEY REFERENCES runs(id) ON DELETE CASCADE,
    current_stage TEXT NOT NULL,
    execution_state TEXT NOT NULL CHECK (
        execution_state IN (
            'queued', 'running', 'auto_resuming', 'paused',
            'quality_rejected', 'ended', 'succeeded', 'failed'
        )
    ),
    elapsed_seconds REAL NOT NULL DEFAULT 0 CHECK (elapsed_seconds >= 0),
    active_started_at TEXT,
    timeout_stage TEXT,
    timeout_count INTEGER NOT NULL DEFAULT 0 CHECK (timeout_count >= 0),
    updated_at TEXT NOT NULL,
    current_episode INTEGER CHECK (current_episode IS NULL OR current_episode >= 1),
    recovery_reason TEXT NOT NULL DEFAULT 'none' CHECK (
        recovery_reason IN (
            'none', 'run_timeout', 'relay_interruption', 'content_rejected'
        )
    ),
    content_repair_count INTEGER CHECK (
        content_repair_count IS NULL
        OR (content_repair_count >= 2 AND content_repair_count <= 2)
    ),
    pause_message TEXT
);

INSERT INTO run_progress_v6(
    run_id, current_stage, execution_state, elapsed_seconds,
    active_started_at, timeout_stage, timeout_count, updated_at,
    current_episode, recovery_reason, content_repair_count, pause_message
)
SELECT
    run_id, current_stage, execution_state, elapsed_seconds,
    active_started_at, timeout_stage, timeout_count, updated_at,
    current_episode, recovery_reason, NULL, NULL
FROM run_progress;

DROP TABLE run_progress;
ALTER TABLE run_progress_v6 RENAME TO run_progress;

ALTER TABLE episode_drafts ADD COLUMN contract_sha256 TEXT;
ALTER TABLE episode_drafts ADD COLUMN state_delta_json TEXT;
ALTER TABLE episode_drafts ADD COLUMN series_state_json TEXT;
ALTER TABLE episode_drafts ADD COLUMN series_state_sha256 TEXT;
ALTER TABLE episode_drafts ADD COLUMN semantic_review_json TEXT;
ALTER TABLE episode_drafts ADD COLUMN repair_rounds INTEGER CHECK (
    repair_rounds IS NULL OR (repair_rounds >= 0 AND repair_rounds <= 2)
);

CREATE TABLE IF NOT EXISTS content_rejections (
    run_id TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    stage TEXT NOT NULL CHECK (
        stage IN ('generating_episode_outline', 'generating_episode_scripts')
    ),
    episode_number INTEGER CHECK (episode_number IS NULL OR episode_number >= 1),
    repair_rounds INTEGER NOT NULL CHECK (repair_rounds = 2),
    evidence TEXT NOT NULL,
    rejected_at TEXT NOT NULL,
    PRIMARY KEY (run_id, stage, episode_number, rejected_at)
);

INSERT OR IGNORE INTO pengine_schema(version) VALUES (6);
"""

_SCHEMA_V7_SQL = """
CREATE TABLE run_progress_v7 (
    run_id TEXT PRIMARY KEY REFERENCES runs(id) ON DELETE CASCADE,
    current_stage TEXT NOT NULL,
    execution_state TEXT NOT NULL CHECK (
        execution_state IN (
            'queued', 'running', 'auto_resuming', 'paused',
            'quality_rejected', 'ended', 'succeeded', 'failed'
        )
    ),
    elapsed_seconds REAL NOT NULL DEFAULT 0 CHECK (elapsed_seconds >= 0),
    active_started_at TEXT,
    timeout_stage TEXT,
    timeout_count INTEGER NOT NULL DEFAULT 0 CHECK (timeout_count >= 0),
    updated_at TEXT NOT NULL,
    current_episode INTEGER CHECK (current_episode IS NULL OR current_episode >= 1),
    recovery_reason TEXT NOT NULL DEFAULT 'none' CHECK (
        recovery_reason IN (
            'none', 'run_timeout', 'relay_interruption', 'content_rejected',
            'episode_error'
        )
    ),
    content_repair_count INTEGER CHECK (
        content_repair_count IS NULL
        OR (content_repair_count >= 2 AND content_repair_count <= 2)
    ),
    pause_message TEXT
);

INSERT INTO run_progress_v7(
    run_id, current_stage, execution_state, elapsed_seconds,
    active_started_at, timeout_stage, timeout_count, updated_at,
    current_episode, recovery_reason, content_repair_count, pause_message
)
SELECT
    run_id, current_stage, execution_state, elapsed_seconds,
    active_started_at, timeout_stage, timeout_count, updated_at,
    current_episode, recovery_reason, content_repair_count, pause_message
FROM run_progress;

DROP TABLE run_progress;
ALTER TABLE run_progress_v7 RENAME TO run_progress;

UPDATE run_progress
SET execution_state = 'paused',
    active_started_at = NULL,
    timeout_stage = 'generating_episode_scripts',
    timeout_count = 0,
    recovery_reason = 'episode_error',
    content_repair_count = NULL,
    pause_message = (
        '旧版本未保存可安全展示的详细原因；已完成分集仍完好，'
        || '可从当前集继续。'
    )
WHERE execution_state = 'failed'
  AND current_stage = 'generating_episode_scripts'
  AND current_episode IS NOT NULL
  AND EXISTS (
      SELECT 1
      FROM runs
      WHERE runs.id = run_progress.run_id
        AND runs.state = 'failed'
        AND runs.failure_code = 'internal_error'
        AND runs.failed_stage = 'generating_episode_scripts'
  )
  AND EXISTS (
      SELECT 1
      FROM episode_drafts
      WHERE episode_drafts.run_id = run_progress.run_id
        AND episode_drafts.episode_number < run_progress.current_episode
  )
  AND EXISTS (
      SELECT 1
      FROM episode_plans
      WHERE episode_plans.run_id = run_progress.run_id
        AND episode_plans.episode_number = run_progress.current_episode
  )
  AND (
      SELECT COUNT(*)
      FROM episode_attempts
      WHERE episode_attempts.run_id = run_progress.run_id
        AND episode_attempts.episode_number = run_progress.current_episode
  ) BETWEEN 1 AND 2;

UPDATE runs
SET state = 'running',
    failure_code = NULL,
    failure_message = NULL,
    failed_stage = NULL,
    failure_attempt_count = NULL,
    completed_at = NULL
WHERE state = 'failed'
  AND failure_code = 'internal_error'
  AND failed_stage = 'generating_episode_scripts'
  AND EXISTS (
      SELECT 1
      FROM run_progress
      WHERE run_progress.run_id = runs.id
        AND run_progress.execution_state = 'paused'
        AND run_progress.recovery_reason = 'episode_error'
  );

INSERT OR IGNORE INTO pengine_schema(version) VALUES (7);
"""

ModelT = TypeVar("ModelT", bound=BaseModel)

RunKind = Literal["initial", "revision"]
ControlState = Literal["queued", "running", "auto_resuming", "ended"]

_USER_STAGE_BY_INTERNAL = {
    InternalStage.LOADING_PERSONA: UserStage.DETERMINING_DIRECTION,
    InternalStage.SELECTING_L0_VARIANT: UserStage.DETERMINING_DIRECTION,
    InternalStage.GENERATING_STORY_OUTLINE: UserStage.GENERATING_STORY_OUTLINE,
    InternalStage.GENERATING_CHARACTER_BIOGRAPHIES: (UserStage.GENERATING_CHARACTER_BIOGRAPHIES),
    InternalStage.GENERATING_RELATIONSHIP_LOGIC: UserStage.GENERATING_RELATIONSHIPS,
    InternalStage.GENERATING_EPISODE_OUTLINE: UserStage.GENERATING_EPISODE_OUTLINE,
    InternalStage.GENERATING_EPISODE_SCRIPTS: UserStage.GENERATING_EPISODE_SCRIPTS,
    InternalStage.ACCEPTING_L0: UserStage.FINAL_REVIEW,
    InternalStage.ACCEPTING_L4: UserStage.FINAL_REVIEW,
    InternalStage.ASSEMBLING_DELIVERY: UserStage.FINAL_REVIEW,
}

_COMPLETED_STAGE_CHECKPOINTS = (
    (UserStage.DETERMINING_DIRECTION, InternalStage.SELECTING_L0_VARIANT),
    (UserStage.GENERATING_STORY_OUTLINE, InternalStage.GENERATING_STORY_OUTLINE),
    (
        UserStage.GENERATING_CHARACTER_BIOGRAPHIES,
        InternalStage.GENERATING_CHARACTER_BIOGRAPHIES,
    ),
    (UserStage.GENERATING_RELATIONSHIPS, InternalStage.GENERATING_RELATIONSHIP_LOGIC),
    (UserStage.GENERATING_EPISODE_OUTLINE, InternalStage.GENERATING_EPISODE_OUTLINE),
    (UserStage.GENERATING_EPISODE_SCRIPTS, InternalStage.GENERATING_EPISODE_SCRIPTS),
)

_DRAFT_STAGE_CHECKPOINTS = (
    (InternalStage.SELECTING_L0_VARIANT, UserStage.DETERMINING_DIRECTION),
    (InternalStage.GENERATING_STORY_OUTLINE, UserStage.GENERATING_STORY_OUTLINE),
    (
        InternalStage.GENERATING_CHARACTER_BIOGRAPHIES,
        UserStage.GENERATING_CHARACTER_BIOGRAPHIES,
    ),
    (InternalStage.GENERATING_RELATIONSHIP_LOGIC, UserStage.GENERATING_RELATIONSHIPS),
    (InternalStage.GENERATING_EPISODE_OUTLINE, UserStage.GENERATING_EPISODE_OUTLINE),
)

_INTERNAL_STAGE_ORDER = (
    InternalStage.LOADING_PERSONA,
    InternalStage.SELECTING_L0_VARIANT,
    InternalStage.GENERATING_STORY_OUTLINE,
    InternalStage.GENERATING_CHARACTER_BIOGRAPHIES,
    InternalStage.GENERATING_RELATIONSHIP_LOGIC,
    InternalStage.GENERATING_EPISODE_OUTLINE,
    InternalStage.GENERATING_EPISODE_SCRIPTS,
    InternalStage.ACCEPTING_L0,
    InternalStage.ACCEPTING_L4,
    InternalStage.ASSEMBLING_DELIVERY,
)


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
    episode_plans: list[EpisodePlan]
    episode_drafts: list[EpisodeDraft]


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
    output_language: OutputLanguage | None
    frozen_feedback: str | None
    stage_attempts: Mapping[InternalStage, int]
    business_checkpoints: Mapping[InternalStage, Any]
    episode_plans: list[EpisodePlan]
    episode_drafts: list[EpisodeDraft]


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


def _episode_plans_from_payload(payload: BaseModel | Mapping[str, Any]) -> list[EpisodePlan]:
    value = payload.model_dump(mode="json") if isinstance(payload, BaseModel) else dict(payload)
    try:
        episode_count = value["episode_count"]
        plans = [EpisodePlan.model_validate(item) for item in value["episodes"]]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Episode outline is missing its structured episode plans") from exc
    expected = list(range(1, int(episode_count) + 1))
    if [plan.episode_number for plan in plans] != expected:
        raise ValueError("Episode plans must be ordered and contiguous from 1")
    return plans


def _aggregate_episode_scripts(
    plans: list[EpisodePlan],
    drafts: list[EpisodeDraft],
) -> str:
    expected = [plan.episode_number for plan in plans]
    if not expected or [draft.episode_number for draft in drafts] != expected:
        raise ValueError("Every planned episode must have one committed draft")
    return "\n\n---\n\n".join(f"第 {draft.episode_number} 集\n{draft.content}" for draft in drafts)


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
            if row is None:
                raise RuntimeError("Unsupported pengine SQLite schema version")
            schema_version = int(row[0])
            if schema_version not in set(range(1, SCHEMA_VERSION + 1)):
                raise RuntimeError("Unsupported pengine SQLite schema version")
            if schema_version == 1:
                await connection.executescript(_SCHEMA_V2_SQL)
                await connection.commit()
                schema_version = 2
            if schema_version == 2:
                await connection.executescript(_SCHEMA_V3_SQL)
                await connection.commit()
                schema_version = 3
            if schema_version == 3:
                await connection.executescript(_SCHEMA_V4_SQL)
                await connection.commit()
                schema_version = 4
            if schema_version == 4:
                await connection.executescript(_SCHEMA_V5_SQL)
                await connection.commit()
                schema_version = 5
            if schema_version == 5:
                await connection.executescript(_SCHEMA_V6_SQL)
                await connection.commit()
                schema_version = 6
            if schema_version == 6:
                await connection.executescript(_SCHEMA_V7_SQL)
                await connection.commit()
                schema_version = 7
            if schema_version == 7:
                await connection.execute("BEGIN IMMEDIATE")
                try:
                    columns = await (
                        await connection.execute("PRAGMA table_info(creations)")
                    ).fetchall()
                    if "output_language" not in {column[1] for column in columns}:
                        await connection.execute(
                            """
                            ALTER TABLE creations ADD COLUMN output_language TEXT
                                CHECK (
                                    output_language IS NULL OR output_language = 'zh-CN'
                                )
                            """
                        )
                    rows = await (
                        await connection.execute(
                            """
                            SELECT id, story, requirements
                            FROM creations
                            WHERE output_language IS NULL
                            """
                        )
                    ).fetchall()
                    await connection.executemany(
                        "UPDATE creations SET output_language = ? WHERE id = ?",
                        [
                            (infer_output_language(row["story"], row["requirements"]), row["id"])
                            for row in rows
                        ],
                    )
                    await connection.execute(
                        "INSERT OR IGNORE INTO pengine_schema(version) VALUES (8)"
                    )
                except BaseException:
                    await connection.rollback()
                    raise
                else:
                    await connection.commit()
                    schema_version = 8

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
        output_language = infer_output_language(request.story, request.requirements)
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
                    persona_snapshot_sha256, story, requirements, output_language,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(new_creation_id),
                    persona_snapshot.persona_id,
                    persona_snapshot.display_name,
                    persona_snapshot.version,
                    persona_snapshot.snapshot_sha256,
                    request.story,
                    request.requirements,
                    output_language,
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
                INSERT INTO run_progress(
                    run_id, current_stage, execution_state, updated_at
                ) VALUES (?, ?, 'queued', ?)
                """,
                (str(run_id), InternalStage.LOADING_PERSONA.value, timestamp),
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
            latest_progress = (
                await self._fetchone(
                    connection,
                    "SELECT execution_state FROM run_progress WHERE run_id = ?",
                    (latest["id"],),
                )
                if latest is not None
                else None
            )

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
                if latest_progress is not None and latest_progress["execution_state"] == "ended":
                    raise DomainError(
                        "revision_not_allowed",
                        "The ended revision cannot be restarted.",
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
                INSERT INTO run_progress(
                    run_id, current_stage, execution_state, updated_at
                ) VALUES (?, ?, 'queued', ?)
                """,
                (str(run_id), InternalStage.LOADING_PERSONA.value, timestamp),
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
                JOIN run_progress ON run_progress.run_id = runs.id
                WHERE jobs.state = 'queued'
                  AND jobs.available_at <= ?
                  AND runs.state IN ('queued', 'running')
                  AND run_progress.execution_state IN (
                      'queued', 'running', 'auto_resuming'
                  )
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
                """
                UPDATE run_progress
                SET execution_state = 'running',
                    active_started_at = COALESCE(active_started_at, ?),
                    recovery_reason = 'none',
                    updated_at = ?
                WHERE run_id = ?
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
                await connection.execute(
                    """
                    UPDATE run_progress
                    SET execution_state = 'running',
                        active_started_at = COALESCE(active_started_at, ?),
                        recovery_reason = 'none',
                        updated_at = ?
                    WHERE run_id = ?
                      AND execution_state IN ('queued', 'running', 'auto_resuming')
                    """,
                    (timestamp, timestamp, str(run_id)),
                )
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
                """
                UPDATE run_progress
                SET execution_state = 'running',
                    active_started_at = COALESCE(active_started_at, ?),
                    recovery_reason = 'none',
                    updated_at = ?
                WHERE run_id = ?
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
                  AND EXISTS (
                      SELECT 1 FROM run_progress
                      WHERE run_progress.run_id = jobs.run_id
                        AND run_progress.execution_state IN (
                            'queued', 'running', 'auto_resuming'
                        )
                  )
                """,
                (timestamp, timestamp, timestamp),
            )
            return cursor.rowcount

    async def record_episode_attempt(
        self,
        run_id: UUID,
        episode_number: int,
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
            plans = await self._episode_plans(connection, run_id)
            if episode_number not in {plan.episode_number for plan in plans}:
                raise DomainError(
                    "episode_not_planned",
                    "The episode is not in the approved outline.",
                    409,
                )
            drafts = await self._episode_drafts(connection, run_id)
            committed_numbers = {draft.episode_number for draft in drafts}
            if episode_number in committed_numbers:
                raise DomainError(
                    "episode_already_committed",
                    "The episode draft is already committed.",
                    409,
                )
            first_unfinished = next(
                (
                    plan.episode_number
                    for plan in plans
                    if plan.episode_number not in committed_numbers
                ),
                None,
            )
            if episode_number != first_unfinished:
                raise DomainError(
                    "episode_out_of_order",
                    "Only the first unfinished episode can be generated.",
                    409,
                )
            cursor = await connection.execute(
                """
                SELECT COALESCE(MAX(attempt_number), 0)
                FROM episode_attempts
                WHERE run_id = ? AND episode_number = ?
                """,
                (str(run_id), episode_number),
            )
            row = await cursor.fetchone()
            current_count = int(row[0]) if row is not None else 0
            if current_count >= MAX_EPISODE_ATTEMPTS:
                raise DomainError(
                    "attempts_exhausted",
                    "The episode attempt limit has been exhausted.",
                    409,
                )
            attempt_number = current_count + 1
            await connection.execute(
                """
                INSERT INTO episode_attempts(run_id, episode_number, attempt_number, recorded_at)
                VALUES (?, ?, ?, ?)
                """,
                (str(run_id), episode_number, attempt_number, timestamp),
            )
            await connection.execute(
                """
                UPDATE run_progress
                SET current_stage = ?,
                    current_episode = ?,
                    execution_state = 'running',
                    recovery_reason = 'none',
                    updated_at = ?
                WHERE run_id = ?
                """,
                (
                    InternalStage.GENERATING_EPISODE_SCRIPTS.value,
                    episode_number,
                    timestamp,
                    str(run_id),
                ),
            )
            return attempt_number

    async def commit_episode_draft(
        self,
        run_id: UUID,
        episode_number: int,
        content: str,
        *,
        episode_lock: EpisodeLock | None = None,
        now: datetime | None = None,
    ) -> EpisodeDraft:
        if not content.strip():
            raise DomainError(
                "invalid_episode_draft",
                "An episode draft must contain content.",
                409,
            )
        timestamp = _timestamp(now or _utc_now())
        content_hash = _text_hash(content)
        if episode_lock is not None and (
            episode_lock.episode_number != episode_number
            or episode_lock.content != content
            or episode_lock.content_sha256 != content_hash
        ):
            raise DomainError(
                "invalid_episode_lock",
                "The episode lock must match the committed script exactly.",
                409,
            )
        async with self._transaction() as connection:
            run = await self._fetchone(
                connection,
                "SELECT state, creation_id FROM runs WHERE id = ?",
                (str(run_id),),
            )
            if run is None:
                raise DomainError("run_not_found", "Workflow run not found.", 404)
            if run["state"] != "running":
                raise DomainError("run_not_running", "Workflow run is not running.", 409)
            plans = await self._episode_plans(connection, run_id)
            if episode_number not in {plan.episode_number for plan in plans}:
                raise DomainError(
                    "episode_not_planned",
                    "The episode is not in the approved outline.",
                    409,
                )
            outline = await self._fetchone(
                connection,
                """
                SELECT payload_json FROM business_checkpoints
                WHERE run_id = ? AND stage = ?
                """,
                (str(run_id), InternalStage.GENERATING_EPISODE_OUTLINE.value),
            )
            if outline is None:
                raise DomainError(
                    "invalid_episode_plan",
                    "A locked episode outline is required before script commit.",
                    409,
                )
            outline_payload = json.loads(outline["payload_json"])
            contract_payload = outline_payload.get("story_contract")
            existing = await self._fetchone(
                connection,
                """
                SELECT *
                FROM episode_drafts
                WHERE run_id = ? AND episode_number = ?
                """,
                (str(run_id), episode_number),
            )
            if existing is not None:
                if existing["content_sha256"] != content_hash:
                    raise DomainError(
                        "episode_conflict",
                        "A committed episode draft cannot be replaced.",
                        409,
                    )
                existing_draft = self._episode_draft_from_row(existing)
                if contract_payload is not None and episode_lock is None:
                    raise DomainError(
                        "episode_lock_required",
                        "A contract-bound episode requires validated lock evidence.",
                        409,
                    )
                if contract_payload is None and episode_lock is not None:
                    raise DomainError(
                        "unexpected_episode_lock",
                        "A legacy outline cannot accept unrelated contract lock data.",
                        409,
                    )
                if episode_lock is not None and (
                    existing_draft.contract_sha256 != episode_lock.contract_sha256
                    or existing_draft.state_delta != episode_lock.state_delta
                    or existing_draft.series_state != episode_lock.series_state
                    or existing_draft.series_state_sha256 != episode_lock.series_state_sha256
                    or existing_draft.semantic_review != episode_lock.semantic_review
                    or existing_draft.repair_rounds != episode_lock.repair_rounds
                ):
                    raise DomainError(
                        "episode_conflict",
                        "A committed episode lock cannot be replaced.",
                        409,
                    )
                return existing_draft
            if contract_payload is not None:
                if episode_lock is None:
                    raise DomainError(
                        "episode_lock_required",
                        "A contract-bound episode requires validated lock evidence.",
                        409,
                    )
                try:
                    contract = StoryContract.model_validate(contract_payload)
                    contract_hash = story_contract_sha256(contract)
                    if outline_payload["story_contract_sha256"] != contract_hash:
                        raise ValueError("Contract hash mismatch")
                    prior_drafts = await self._episode_drafts(connection, run_id)
                    prior_state = (
                        prior_drafts[-1].series_state
                        if prior_drafts
                        else initial_series_state(contract, contract_hash)
                    )
                    if prior_state is None:
                        raise ValueError("Prior series state is missing")
                    rebuilt = build_episode_lock(
                        contract=contract,
                        contract_sha256=contract_hash,
                        prior_state=prior_state,
                        content=content,
                        delta=episode_lock.state_delta,
                        semantic_review=episode_lock.semantic_review,
                        repair_rounds=episode_lock.repair_rounds,
                    )
                    if rebuilt != episode_lock:
                        raise ValueError("Episode lock is not deterministic")
                except Exception as exc:
                    raise DomainError(
                        "invalid_episode_lock",
                        "The episode lock conflicts with the approved story contract.",
                        409,
                    ) from exc
            elif episode_lock is not None:
                raise DomainError(
                    "unexpected_episode_lock",
                    "A legacy outline cannot accept unrelated contract lock data.",
                    409,
                )
            attempt = await self._fetchone(
                connection,
                """
                SELECT 1 FROM episode_attempts
                WHERE run_id = ? AND episode_number = ?
                LIMIT 1
                """,
                (str(run_id), episode_number),
            )
            if attempt is None:
                raise DomainError(
                    "episode_attempt_required",
                    "An episode draft requires a recorded writer attempt.",
                    409,
                )
            drafts = await self._episode_drafts(connection, run_id)
            expected = list(range(1, episode_number))
            if [draft.episode_number for draft in drafts] != expected:
                raise DomainError(
                    "episode_out_of_order",
                    "Episode drafts must commit in outline order.",
                    409,
                )
            await connection.execute(
                """
                INSERT INTO episode_drafts(
                    run_id, episode_number, content, content_sha256, completed_at,
                    contract_sha256, state_delta_json, series_state_json,
                    series_state_sha256, semantic_review_json, repair_rounds
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(run_id),
                    episode_number,
                    content,
                    content_hash,
                    timestamp,
                    episode_lock.contract_sha256 if episode_lock else None,
                    _json(episode_lock.state_delta) if episode_lock else None,
                    _json(episode_lock.series_state) if episode_lock else None,
                    episode_lock.series_state_sha256 if episode_lock else None,
                    _json(episode_lock.semantic_review) if episode_lock else None,
                    episode_lock.repair_rounds if episode_lock else None,
                ),
            )
            next_episode = episode_number + 1 if episode_number < len(plans) else None
            await connection.execute(
                """
                UPDATE run_progress
                SET current_stage = ?,
                    current_episode = ?,
                    timeout_stage = NULL,
                    timeout_count = 0,
                    recovery_reason = 'none',
                    content_repair_count = NULL,
                    pause_message = NULL,
                    updated_at = ?
                WHERE run_id = ?
                """,
                (
                    InternalStage.GENERATING_EPISODE_SCRIPTS.value,
                    next_episode,
                    timestamp,
                    str(run_id),
                ),
            )
            await connection.execute(
                "UPDATE creations SET updated_at = ? WHERE id = ?",
                (timestamp, run["creation_id"]),
            )
            return EpisodeDraft(
                episode_number=episode_number,
                content=content,
                content_sha256=content_hash,
                completed_at=_datetime(timestamp),
                contract_sha256=episode_lock.contract_sha256 if episode_lock else None,
                state_delta=episode_lock.state_delta if episode_lock else None,
                series_state=episode_lock.series_state if episode_lock else None,
                series_state_sha256=(episode_lock.series_state_sha256 if episode_lock else None),
                semantic_review=episode_lock.semantic_review if episode_lock else None,
                repair_rounds=episode_lock.repair_rounds if episode_lock else None,
            )

    async def handle_episode_timeout(
        self,
        run_id: UUID,
        episode_number: int,
        *,
        now: datetime | None = None,
    ) -> RecoveryState:
        return await self.handle_episode_interruption(
            run_id,
            episode_number,
            recovery_reason="run_timeout",
            now=now,
        )

    async def pause_content_rejection(
        self,
        run_id: UUID,
        *,
        stage: InternalStage,
        evidence: str,
        repair_rounds: int,
        episode_number: int | None = None,
        now: datetime | None = None,
    ) -> None:
        if stage not in {
            InternalStage.GENERATING_EPISODE_OUTLINE,
            InternalStage.GENERATING_EPISODE_SCRIPTS,
        }:
            raise ValueError("Only contract or episode generation can pause for content review")
        if repair_rounds != 2 or not evidence.strip():
            raise ValueError("Content rejection requires two repair rounds and evidence")
        if (stage is InternalStage.GENERATING_EPISODE_SCRIPTS) != (episode_number is not None):
            raise ValueError("Episode content rejection requires its episode number")
        current = now or _utc_now()
        timestamp = _timestamp(current)
        async with self._transaction() as connection:
            progress = await self._fetchone(
                connection,
                """
                SELECT run_progress.*, runs.state AS run_state, runs.creation_id
                FROM run_progress
                JOIN runs ON runs.id = run_progress.run_id
                WHERE run_progress.run_id = ?
                """,
                (str(run_id),),
            )
            if progress is None:
                raise DomainError("run_not_found", "Workflow run not found.", 404)
            if progress["run_state"] != "running" or progress["execution_state"] != "running":
                raise DomainError(
                    "run_not_controllable",
                    "Workflow run is not actively running.",
                    409,
                )
            await connection.execute(
                """
                INSERT INTO content_rejections(
                    run_id, stage, episode_number, repair_rounds, evidence, rejected_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    str(run_id),
                    stage.value,
                    episode_number,
                    repair_rounds,
                    evidence,
                    timestamp,
                ),
            )
            user_stage = _USER_STAGE_BY_INTERNAL[stage]
            await connection.execute(
                """
                UPDATE run_progress
                SET current_stage = ?,
                    current_episode = ?,
                    execution_state = 'paused',
                    elapsed_seconds = ?,
                    active_started_at = NULL,
                    timeout_stage = ?,
                    timeout_count = 0,
                    recovery_reason = 'content_rejected',
                    content_repair_count = ?,
                    pause_message = ?,
                    updated_at = ?
                WHERE run_id = ?
                """,
                (
                    stage.value,
                    episode_number,
                    self._elapsed_seconds(progress, current),
                    user_stage.value,
                    repair_rounds,
                    evidence,
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
                (timestamp, progress["creation_id"]),
            )

    async def pause_episode_error(
        self,
        run_id: UUID,
        *,
        episode_number: int,
        safe_message: str,
        now: datetime | None = None,
    ) -> None:
        if episode_number < 1 or not safe_message.strip():
            raise ValueError("Episode errors require an episode number and safe message")
        current = now or _utc_now()
        timestamp = _timestamp(current)
        async with self._transaction() as connection:
            progress = await self._fetchone(
                connection,
                """
                SELECT run_progress.*, runs.state AS run_state, runs.creation_id
                FROM run_progress
                JOIN runs ON runs.id = run_progress.run_id
                WHERE run_progress.run_id = ?
                """,
                (str(run_id),),
            )
            if progress is None:
                raise DomainError("run_not_found", "Workflow run not found.", 404)
            if progress["run_state"] != "running" or progress["execution_state"] != "running":
                raise DomainError(
                    "run_not_controllable",
                    "Workflow run is not actively running.",
                    409,
                )
            planned = await self._fetchone(
                connection,
                """
                SELECT 1 FROM episode_plans
                WHERE run_id = ? AND episode_number = ?
                """,
                (str(run_id), episode_number),
            )
            if planned is None:
                raise DomainError(
                    "episode_not_planned",
                    "The episode is not in the approved outline.",
                    409,
                )
            attempt = await self._fetchone(
                connection,
                """
                SELECT COUNT(*) AS attempt_count
                FROM episode_attempts
                WHERE run_id = ? AND episode_number = ?
                """,
                (str(run_id), episode_number),
            )
            attempt_count = int(attempt["attempt_count"]) if attempt is not None else 0
            if attempt_count == 0:
                raise DomainError(
                    "episode_attempt_required",
                    "An episode error requires a recorded writer attempt.",
                    409,
                )
            if attempt_count >= MAX_EPISODE_ATTEMPTS:
                raise DomainError(
                    "attempts_exhausted",
                    "The episode attempt limit has been exhausted.",
                    409,
                )
            await connection.execute(
                """
                UPDATE run_progress
                SET current_stage = ?,
                    current_episode = ?,
                    execution_state = 'paused',
                    elapsed_seconds = ?,
                    active_started_at = NULL,
                    timeout_stage = ?,
                    timeout_count = 0,
                    recovery_reason = 'episode_error',
                    content_repair_count = NULL,
                    pause_message = ?,
                    updated_at = ?
                WHERE run_id = ?
                """,
                (
                    InternalStage.GENERATING_EPISODE_SCRIPTS.value,
                    episode_number,
                    self._elapsed_seconds(progress, current),
                    UserStage.GENERATING_EPISODE_SCRIPTS.value,
                    safe_message.strip(),
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
                (timestamp, progress["creation_id"]),
            )

    async def handle_episode_relay_interruption(
        self,
        run_id: UUID,
        episode_number: int,
        *,
        retry_delay_seconds: int,
        now: datetime | None = None,
    ) -> RecoveryState:
        return await self.handle_episode_interruption(
            run_id,
            episode_number,
            recovery_reason="relay_interruption",
            retry_delay_seconds=retry_delay_seconds,
            now=now,
        )

    async def handle_episode_interruption(
        self,
        run_id: UUID,
        episode_number: int,
        *,
        recovery_reason: RecoveryReason,
        retry_delay_seconds: int = 0,
        now: datetime | None = None,
    ) -> RecoveryState:
        if retry_delay_seconds < 0:
            raise ValueError("Retry delay cannot be negative")
        current = now or _utc_now()
        timestamp = _timestamp(current)
        async with self._transaction() as connection:
            progress = await self._fetchone(
                connection,
                """
                SELECT run_progress.*, runs.state AS run_state, runs.creation_id
                FROM run_progress
                JOIN runs ON runs.id = run_progress.run_id
                WHERE run_progress.run_id = ?
                """,
                (str(run_id),),
            )
            if progress is None:
                raise DomainError("run_not_found", "Workflow run not found.", 404)
            if progress["run_state"] != "running" or progress["execution_state"] != "running":
                raise DomainError(
                    "run_not_controllable",
                    "Workflow run is not actively running.",
                    409,
                )
            plans = await self._episode_plans(connection, run_id)
            if episode_number not in {plan.episode_number for plan in plans}:
                raise DomainError(
                    "episode_not_planned",
                    "The episode is not in the approved outline.",
                    409,
                )
            drafts = await self._episode_drafts(connection, run_id)
            committed_numbers = {draft.episode_number for draft in drafts}
            if episode_number in committed_numbers:
                raise DomainError(
                    "episode_already_committed",
                    "A committed episode cannot time out.",
                    409,
                )
            first_unfinished = next(
                (
                    plan.episode_number
                    for plan in plans
                    if plan.episode_number not in committed_numbers
                ),
                None,
            )
            if episode_number != first_unfinished:
                raise DomainError(
                    "episode_out_of_order",
                    "Only the first unfinished episode can recover.",
                    409,
                )
            timeout = await self._fetchone(
                connection,
                """
                SELECT timeout_count FROM episode_timeouts
                WHERE run_id = ? AND episode_number = ?
                """,
                (str(run_id), episode_number),
            )
            timeout_count = (int(timeout["timeout_count"]) if timeout is not None else 0) + 1
            attempt = await self._fetchone(
                connection,
                """
                SELECT COUNT(*) AS attempt_count
                FROM episode_attempts
                WHERE run_id = ? AND episode_number = ?
                """,
                (str(run_id), episode_number),
            )
            if attempt is None:
                raise RuntimeError("Episode attempt count is unavailable")
            attempt_count = int(attempt["attempt_count"])
            next_state: RecoveryState = (
                "failed"
                if attempt_count >= MAX_EPISODE_ATTEMPTS
                else "auto_resuming"
                if timeout_count == 1
                else "paused"
            )
            await connection.execute(
                """
                INSERT INTO episode_timeouts(run_id, episode_number, timeout_count, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(run_id, episode_number) DO UPDATE SET
                    timeout_count = excluded.timeout_count,
                    updated_at = excluded.updated_at
                """,
                (str(run_id), episode_number, timeout_count, timestamp),
            )
            elapsed_seconds = self._elapsed_seconds(progress, current)
            if next_state == "failed":
                await self._mark_attempts_exhausted(
                    connection,
                    run_id=run_id,
                    creation_id=progress["creation_id"],
                    stage=InternalStage.GENERATING_EPISODE_SCRIPTS,
                    attempt_count=attempt_count,
                    elapsed_seconds=elapsed_seconds,
                    timestamp=timestamp,
                    episode=True,
                )
                return next_state
            await connection.execute(
                """
                UPDATE run_progress
                SET current_stage = ?,
                    current_episode = ?,
                    execution_state = ?,
                    elapsed_seconds = ?,
                    active_started_at = NULL,
                    timeout_stage = ?,
                    timeout_count = ?,
                    recovery_reason = ?,
                    updated_at = ?
                WHERE run_id = ?
                """,
                (
                    InternalStage.GENERATING_EPISODE_SCRIPTS.value,
                    episode_number,
                    next_state,
                    elapsed_seconds,
                    UserStage.GENERATING_EPISODE_SCRIPTS.value,
                    timeout_count,
                    recovery_reason,
                    timestamp,
                    str(run_id),
                ),
            )
            await connection.execute(
                """
                UPDATE jobs
                SET state = ?,
                    available_at = ?,
                    lease_owner = NULL,
                    lease_expires_at = NULL,
                    updated_at = ?
                WHERE run_id = ?
                """,
                (
                    "queued" if next_state == "auto_resuming" else "failed",
                    self._interruption_available_at(
                        current,
                        recovery_reason,
                        retry_delay_seconds,
                    ),
                    timestamp,
                    str(run_id),
                ),
            )
            await connection.execute(
                "UPDATE creations SET updated_at = ? WHERE id = ?",
                (timestamp, progress["creation_id"]),
            )
            return next_state

    async def get_episode_plans(self, run_id: UUID) -> list[EpisodePlan]:
        async with self._connection() as connection:
            return await self._episode_plans(connection, run_id)

    async def get_episode_drafts(self, run_id: UUID) -> list[EpisodeDraft]:
        async with self._connection() as connection:
            return await self._episode_drafts(connection, run_id)

    async def get_episode_attempt_counts(self, run_id: UUID) -> dict[int, int]:
        async with self._connection() as connection:
            cursor = await connection.execute(
                """
                SELECT episode_number, COUNT(*) AS attempt_count
                FROM episode_attempts
                WHERE run_id = ?
                GROUP BY episode_number
                """,
                (str(run_id),),
            )
            rows = await cursor.fetchall()
        return {int(row["episode_number"]): int(row["attempt_count"]) for row in rows}

    async def assemble_episode_scripts(self, run_id: UUID) -> str:
        async with self._connection() as connection:
            plans = await self._episode_plans(connection, run_id)
            drafts = await self._episode_drafts(connection, run_id)
        try:
            return _aggregate_episode_scripts(plans, drafts)
        except ValueError as exc:
            raise DomainError(
                "episode_sequence_incomplete",
                "Every planned episode must be committed before assembly.",
                409,
            ) from exc

    async def episode_aggregate_checkpoint_payload(
        self,
        run_id: UUID,
    ) -> Mapping[str, Any]:
        async with self._connection() as connection:
            return await self._episode_aggregate_payload(connection, run_id)

    async def _episode_aggregate_payload(
        self,
        connection: aiosqlite.Connection,
        run_id: UUID,
    ) -> Mapping[str, Any]:
        plans = await self._episode_plans(connection, run_id)
        drafts = await self._episode_drafts(connection, run_id)
        try:
            content = _aggregate_episode_scripts(plans, drafts)
        except ValueError as exc:
            raise DomainError(
                "episode_sequence_incomplete",
                "Every planned episode must be committed before assembly.",
                409,
            ) from exc
        outline = await self._fetchone(
            connection,
            """
            SELECT payload_json FROM business_checkpoints
            WHERE run_id = ? AND stage = ?
            """,
            (str(run_id), InternalStage.GENERATING_EPISODE_OUTLINE.value),
        )
        if outline is None:
            raise DomainError(
                "invalid_episode_plan",
                "The approved episode outline is unavailable.",
                409,
            )
        outline_payload = json.loads(outline["payload_json"])
        if "story_contract" not in outline_payload:
            return {"content": content}
        try:
            contract = StoryContract.model_validate(outline_payload["story_contract"])
            contract_hash = story_contract_sha256(contract)
            if outline_payload["story_contract_sha256"] != contract_hash:
                raise ValueError("Contract hash mismatch")
            review = SemanticReview.model_validate(outline_payload["contract_review"])
            if not review.passed:
                raise ValueError("Contract review did not pass")
            state = initial_series_state(contract, contract_hash)
            episode_hashes = []
            for draft in drafts:
                if (
                    draft.contract_sha256 != contract_hash
                    or draft.state_delta is None
                    or draft.series_state is None
                    or draft.series_state_sha256 is None
                    or draft.semantic_review is None
                    or draft.repair_rounds is None
                ):
                    raise ValueError("Episode lock data is incomplete")
                rebuilt = build_episode_lock(
                    contract=contract,
                    contract_sha256=contract_hash,
                    prior_state=state,
                    content=draft.content,
                    delta=draft.state_delta,
                    semantic_review=draft.semantic_review,
                    repair_rounds=draft.repair_rounds,
                )
                if (
                    rebuilt.series_state != draft.series_state
                    or rebuilt.series_state_sha256 != draft.series_state_sha256
                    or rebuilt.content_sha256 != draft.content_sha256
                ):
                    raise ValueError("Episode lock data is not deterministic")
                state = draft.series_state
                episode_hashes.append(
                    {
                        "episode_number": draft.episode_number,
                        "content_sha256": draft.content_sha256,
                        "series_state_sha256": draft.series_state_sha256,
                    }
                )
        except Exception as exc:
            raise DomainError(
                "episode_lock_invalid",
                "Every episode lock must match the approved story contract before review.",
                409,
            ) from exc
        return {
            "content": content,
            "contract_sha256": contract_hash,
            "episode_hashes": episode_hashes,
            "series_state_sha256": canonical_model_hash(state),
        }

    async def record_stage_attempt(
        self,
        run_id: UUID,
        stage: InternalStage,
        *,
        now: datetime | None = None,
    ) -> int:
        if stage is InternalStage.GENERATING_EPISODE_SCRIPTS:
            raise DomainError(
                "episode_attempt_required",
                "Episode scripts use per-episode attempts.",
                409,
            )
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
            await connection.execute(
                """
                UPDATE run_progress
                SET current_stage = ?,
                    execution_state = 'running',
                    recovery_reason = 'none',
                    updated_at = ?
                WHERE run_id = ?
                """,
                (stage.value, timestamp, str(run_id)),
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

            plans: list[EpisodePlan] | None = None
            if stage is InternalStage.GENERATING_EPISODE_OUTLINE:
                try:
                    plans = _episode_plans_from_payload(payload)
                except ValueError as exc:
                    raise DomainError(
                        "invalid_episode_plan",
                        "The episode outline must contain an ordered episode plan.",
                        409,
                    ) from exc
                supplied_outline = json.loads(payload_json)
                if "story_contract" in supplied_outline:
                    try:
                        contract = StoryContract.model_validate(supplied_outline["story_contract"])
                        contract_hash = story_contract_sha256(contract)
                        review = SemanticReview.model_validate(supplied_outline["contract_review"])
                        if (
                            supplied_outline["story_contract_sha256"] != contract_hash
                            or supplied_outline["story_contract_markdown"]
                            != render_story_contract_markdown(contract, contract_hash)
                            or not review.passed
                            or not 0 <= int(supplied_outline["contract_repair_rounds"]) <= 2
                        ):
                            raise ValueError("Story contract lock metadata is invalid")
                    except Exception as exc:
                        raise DomainError(
                            "invalid_story_contract",
                            "The episode outline story contract is not lockable.",
                            409,
                        ) from exc
            if stage is InternalStage.GENERATING_EPISODE_SCRIPTS:
                try:
                    supplied_scripts = json.loads(payload_json)
                    expected_scripts = dict(
                        await self._episode_aggregate_payload(connection, run_id)
                    )
                    if "stage" in supplied_scripts:
                        expected_scripts = {
                            "stage": InternalStage.GENERATING_EPISODE_SCRIPTS.value,
                            **expected_scripts,
                        }
                except (KeyError, TypeError, ValueError, DomainError) as exc:
                    raise DomainError(
                        "episode_sequence_incomplete",
                        "Every planned episode must be committed before assembly.",
                        409,
                    ) from exc
                if supplied_scripts != expected_scripts:
                    raise DomainError(
                        "episode_aggregate_conflict",
                        "The aggregate script and lock hashes must match committed episodes.",
                        409,
                    )
            if stage in {InternalStage.ACCEPTING_L0, InternalStage.ACCEPTING_L4}:
                existing_episode_plans = await self._episode_plans(connection, run_id)
                if existing_episode_plans:
                    scripts = await self._fetchone(
                        connection,
                        """
                        SELECT 1 FROM business_checkpoints
                        WHERE run_id = ? AND stage = ?
                        """,
                        (str(run_id), InternalStage.GENERATING_EPISODE_SCRIPTS.value),
                    )
                    if scripts is None:
                        raise DomainError(
                            "episode_sequence_incomplete",
                            "Every planned episode must be assembled before review.",
                            409,
                        )

            await connection.execute(
                """
                INSERT INTO business_checkpoints(
                    run_id, stage, payload_json, payload_sha256, approved_at
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (str(run_id), stage.value, payload_json, payload_hash, timestamp),
            )
            if plans is not None:
                await connection.executemany(
                    """
                    INSERT INTO episode_plans(run_id, episode_number, plan, plan_sha256)
                    VALUES (?, ?, ?, ?)
                    """,
                    [
                        (
                            str(run_id),
                            plan.episode_number,
                            plan.plan,
                            _text_hash(plan.plan),
                        )
                        for plan in plans
                    ],
                )
            stage_index = _INTERNAL_STAGE_ORDER.index(stage)
            next_stage = _INTERNAL_STAGE_ORDER[min(stage_index + 1, len(_INTERNAL_STAGE_ORDER) - 1)]
            await connection.execute(
                """
                UPDATE run_progress
                SET current_stage = ?, updated_at = ?
                WHERE run_id = ?
                """,
                (next_stage.value, timestamp, str(run_id)),
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

    async def handle_run_timeout(
        self,
        run_id: UUID,
        stage: InternalStage,
        *,
        now: datetime | None = None,
    ) -> RecoveryState:
        return await self.handle_run_interruption(
            run_id,
            stage,
            recovery_reason="run_timeout",
            now=now,
        )

    async def handle_run_relay_interruption(
        self,
        run_id: UUID,
        stage: InternalStage,
        *,
        retry_delay_seconds: int,
        now: datetime | None = None,
    ) -> RecoveryState:
        return await self.handle_run_interruption(
            run_id,
            stage,
            recovery_reason="relay_interruption",
            retry_delay_seconds=retry_delay_seconds,
            now=now,
        )

    async def handle_run_interruption(
        self,
        run_id: UUID,
        stage: InternalStage,
        *,
        recovery_reason: RecoveryReason,
        retry_delay_seconds: int = 0,
        now: datetime | None = None,
    ) -> RecoveryState:
        if retry_delay_seconds < 0:
            raise ValueError("Retry delay cannot be negative")
        current = now or _utc_now()
        timestamp = _timestamp(current)
        user_stage = _USER_STAGE_BY_INTERNAL[stage]

        async with self._transaction() as connection:
            progress = await self._fetchone(
                connection,
                """
                SELECT run_progress.*, runs.state AS run_state, runs.creation_id
                FROM run_progress
                JOIN runs ON runs.id = run_progress.run_id
                WHERE run_progress.run_id = ?
                """,
                (str(run_id),),
            )
            if progress is None:
                raise DomainError("run_not_found", "Workflow run not found.", 404)
            if progress["run_state"] != "running" or progress["execution_state"] != "running":
                raise DomainError(
                    "run_not_controllable",
                    "Workflow run is not actively running.",
                    409,
                )

            timeout_count = (
                int(progress["timeout_count"]) + 1
                if progress["timeout_stage"] == user_stage.value
                else 1
            )
            attempt = await self._fetchone(
                connection,
                """
                SELECT COUNT(*) AS attempt_count
                FROM stage_attempts
                WHERE run_id = ? AND stage = ?
                """,
                (str(run_id), stage.value),
            )
            if attempt is None:
                raise RuntimeError("Stage attempt count is unavailable")
            attempt_count = int(attempt["attempt_count"])
            next_state: RecoveryState = (
                "failed"
                if attempt_count >= MAX_STAGE_ATTEMPTS
                else "auto_resuming"
                if timeout_count == 1
                else "paused"
            )
            elapsed_seconds = self._elapsed_seconds(progress, current)
            if next_state == "failed":
                await self._mark_attempts_exhausted(
                    connection,
                    run_id=run_id,
                    creation_id=progress["creation_id"],
                    stage=stage,
                    attempt_count=attempt_count,
                    elapsed_seconds=elapsed_seconds,
                    timestamp=timestamp,
                )
                return next_state
            await connection.execute(
                """
                UPDATE run_progress
                SET current_stage = ?,
                    execution_state = ?,
                    elapsed_seconds = ?,
                    active_started_at = NULL,
                    timeout_stage = ?,
                    timeout_count = ?,
                    recovery_reason = ?,
                    updated_at = ?
                WHERE run_id = ?
                """,
                (
                    stage.value,
                    next_state,
                    elapsed_seconds,
                    user_stage.value,
                    timeout_count,
                    recovery_reason,
                    timestamp,
                    str(run_id),
                ),
            )
            await connection.execute(
                """
                UPDATE jobs
                SET state = ?,
                    available_at = ?,
                    lease_owner = NULL,
                    lease_expires_at = NULL,
                    updated_at = ?
                WHERE run_id = ?
                """,
                (
                    "queued" if next_state == "auto_resuming" else "failed",
                    self._interruption_available_at(
                        current,
                        recovery_reason,
                        retry_delay_seconds,
                    ),
                    timestamp,
                    str(run_id),
                ),
            )
            await connection.execute(
                "UPDATE creations SET updated_at = ? WHERE id = ?",
                (timestamp, progress["creation_id"]),
            )
            return next_state

    async def reject_quality_gate(
        self,
        run_id: UUID,
        *,
        stage: InternalStage,
        evidence: str | None,
        now: datetime | None = None,
    ) -> QualityGateRejection:
        if stage not in {InternalStage.ACCEPTING_L0, InternalStage.ACCEPTING_L4}:
            raise ValueError("Only final quality gates can be rejected")
        current = now or _utc_now()
        timestamp = _timestamp(current)

        async with self._transaction() as connection:
            progress = await self._fetchone(
                connection,
                """
                SELECT run_progress.*, runs.state AS run_state, runs.creation_id
                FROM run_progress
                JOIN runs ON runs.id = run_progress.run_id
                WHERE run_progress.run_id = ?
                """,
                (str(run_id),),
            )
            if progress is None:
                raise DomainError("run_not_found", "Workflow run not found.", 404)
            if progress["run_state"] != "running" or progress["execution_state"] != "running":
                raise DomainError(
                    "run_not_controllable",
                    "Workflow run is not actively running.",
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
            attempt_row = await cursor.fetchone()
            attempt_count = int(attempt_row[0]) if attempt_row is not None else 0
            if attempt_count == 0:
                attempt_count = 1
                await connection.execute(
                    """
                    INSERT INTO stage_attempts(run_id, stage, attempt_number, recorded_at)
                    VALUES (?, ?, ?, ?)
                    """,
                    (str(run_id), stage.value, attempt_count, timestamp),
                )

            existing = await self._fetchone(
                connection,
                """
                SELECT evidence FROM quality_gate_rejections
                WHERE run_id = ? AND stage = ? AND attempt_number = ?
                """,
                (str(run_id), stage.value, attempt_count),
            )
            if existing is not None:
                if existing["evidence"] != evidence:
                    raise DomainError(
                        "run_not_controllable",
                        "A quality-gate rejection cannot be replaced.",
                        409,
                    )
            else:
                await connection.execute(
                    """
                    INSERT INTO quality_gate_rejections(
                        run_id, stage, attempt_number, evidence, rejected_at
                    ) VALUES (?, ?, ?, ?, ?)
                    """,
                    (str(run_id), stage.value, attempt_count, evidence, timestamp),
                )

            elapsed_seconds = self._elapsed_seconds(progress, current)
            await connection.execute(
                """
                UPDATE run_progress
                SET current_stage = ?,
                    execution_state = 'quality_rejected',
                    elapsed_seconds = ?,
                    active_started_at = NULL,
                    updated_at = ?
                WHERE run_id = ?
                """,
                (stage.value, elapsed_seconds, timestamp, str(run_id)),
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
                (timestamp, progress["creation_id"]),
            )
            return QualityGateRejection(
                stage=stage.value,
                evidence=evidence,
                attempt_count=attempt_count,
                can_retry=attempt_count < MAX_STAGE_ATTEMPTS,
            )

    async def retry_final_review(
        self,
        *,
        creation_id: UUID,
        run_kind: RunKind,
        idempotency_key: str,
        now: datetime | None = None,
    ) -> RunControlAccepted:
        timestamp = _timestamp(now or _utc_now())
        scope = f"run-control:{creation_id}:{run_kind}:retry-final-review"
        payload_hash = canonical_payload_hash({"action": "retry-final-review"})

        async with self._transaction() as connection:
            replay = await self._idempotency_replay(
                connection,
                idempotency_key,
                scope,
                payload_hash,
                RunControlAccepted,
            )
            if replay is not None:
                return replay

            run = await self._fetch_control_run(connection, creation_id, run_kind)
            state = run["execution_state"]
            if state != "quality_rejected":
                raise DomainError(
                    "run_not_controllable",
                    "Only a quality-rejected workflow run can retry final review.",
                    409,
                )
            cursor = await connection.execute(
                """
                SELECT COALESCE(MAX(attempt_number), 0)
                FROM quality_gate_rejections
                WHERE run_id = ? AND stage = ?
                """,
                (run["id"], run["current_stage"]),
            )
            attempt_row = await cursor.fetchone()
            if attempt_row is None or int(attempt_row[0]) >= MAX_STAGE_ATTEMPTS:
                raise DomainError(
                    "run_not_controllable",
                    "The quality gate has reached its retry limit.",
                    409,
                )
            await connection.execute(
                """
                UPDATE run_progress
                SET execution_state = 'queued', updated_at = ?
                WHERE run_id = ?
                """,
                (timestamp, run["id"]),
            )
            await connection.execute(
                """
                UPDATE jobs
                SET state = 'queued',
                    available_at = ?,
                    lease_owner = NULL,
                    lease_expires_at = NULL,
                    updated_at = ?
                WHERE run_id = ?
                """,
                (timestamp, timestamp, run["id"]),
            )
            state = "queued"

            response = RunControlAccepted(
                creation_id=creation_id,
                run_kind=run_kind,
                run_state=state,
                resource_url=f"/creations/{creation_id}",
            )
            await connection.execute(
                "UPDATE creations SET updated_at = ? WHERE id = ?",
                (timestamp, str(creation_id)),
            )
            await self._store_idempotency(
                connection,
                idempotency_key,
                scope,
                payload_hash,
                response,
                timestamp,
            )
            return response

    async def continue_run(
        self,
        *,
        creation_id: UUID,
        run_kind: RunKind,
        idempotency_key: str,
        now: datetime | None = None,
    ) -> RunControlAccepted:
        timestamp = _timestamp(now or _utc_now())
        scope = f"run-control:{creation_id}:{run_kind}:continue"
        payload_hash = canonical_payload_hash({"action": "continue"})

        async with self._transaction() as connection:
            replay = await self._idempotency_replay(
                connection,
                idempotency_key,
                scope,
                payload_hash,
                RunControlAccepted,
            )
            if replay is not None:
                return replay

            run = await self._fetch_control_run(connection, creation_id, run_kind)
            state = run["execution_state"]
            if state == "paused":
                if not await self._has_remaining_attempts(
                    connection,
                    run_id=run["id"],
                    current_stage=run["current_stage"],
                    current_episode=run["current_episode"],
                ):
                    raise DomainError(
                        "run_not_controllable",
                        "The stage attempt limit has been exhausted.",
                        409,
                    )
                await connection.execute(
                    """
                    UPDATE run_progress
                    SET execution_state = 'queued',
                        recovery_reason = 'none',
                        content_repair_count = NULL,
                        pause_message = NULL,
                        updated_at = ?
                    WHERE run_id = ?
                    """,
                    (timestamp, run["id"]),
                )
                await connection.execute(
                    """
                    UPDATE jobs
                    SET state = 'queued',
                        available_at = ?,
                        lease_owner = NULL,
                        lease_expires_at = NULL,
                        updated_at = ?
                    WHERE run_id = ?
                    """,
                    (timestamp, timestamp, run["id"]),
                )
                state = "queued"
            elif state not in {"queued", "running", "auto_resuming"}:
                raise DomainError(
                    "run_not_controllable",
                    "Only a paused workflow run can continue.",
                    409,
                )

            response = RunControlAccepted(
                creation_id=creation_id,
                run_kind=run_kind,
                run_state=state,
                resource_url=f"/creations/{creation_id}",
            )
            await connection.execute(
                "UPDATE creations SET updated_at = ? WHERE id = ?",
                (timestamp, str(creation_id)),
            )
            await self._store_idempotency(
                connection,
                idempotency_key,
                scope,
                payload_hash,
                response,
                timestamp,
            )
            return response

    async def end_run(
        self,
        *,
        creation_id: UUID,
        run_kind: RunKind,
        idempotency_key: str,
        now: datetime | None = None,
    ) -> RunControlAccepted:
        timestamp = _timestamp(now or _utc_now())
        scope = f"run-control:{creation_id}:{run_kind}:end"
        payload_hash = canonical_payload_hash({"action": "end"})

        async with self._transaction() as connection:
            replay = await self._idempotency_replay(
                connection,
                idempotency_key,
                scope,
                payload_hash,
                RunControlAccepted,
            )
            if replay is not None:
                return replay

            run = await self._fetch_control_run(connection, creation_id, run_kind)
            state = run["execution_state"]
            if state == "ended":
                response = RunControlAccepted(
                    creation_id=creation_id,
                    run_kind=run_kind,
                    run_state="ended",
                    resource_url=f"/creations/{creation_id}",
                )
            elif state in {"paused", "quality_rejected"}:
                current_stage = InternalStage(run["current_stage"])
                ended_message = (
                    "The operator ended the quality-rejected workflow run."
                    if state == "quality_rejected"
                    else "The operator ended the paused workflow run."
                )
                if (
                    current_stage is InternalStage.GENERATING_EPISODE_SCRIPTS
                    and run["current_episode"] is not None
                ):
                    cursor = await connection.execute(
                        """
                        SELECT COUNT(*) FROM episode_attempts
                        WHERE run_id = ? AND episode_number = ?
                        """,
                        (run["id"], run["current_episode"]),
                    )
                else:
                    cursor = await connection.execute(
                        """
                        SELECT COUNT(*) FROM stage_attempts
                        WHERE run_id = ? AND stage = ?
                        """,
                        (run["id"], current_stage.value),
                    )
                attempt_row = await cursor.fetchone()
                attempt_count = max(1, min(MAX_STAGE_ATTEMPTS, int(attempt_row[0])))
                await connection.execute(
                    """
                    UPDATE runs
                    SET state = 'failed',
                        failure_code = 'ended_by_user',
                        failure_message = ?,
                        failed_stage = ?,
                        failure_attempt_count = ?,
                        completed_at = ?,
                        updated_at = ?
                    WHERE id = ? AND state = 'running'
                    """,
                    (
                        ended_message,
                        current_stage.value,
                        attempt_count,
                        timestamp,
                        timestamp,
                        run["id"],
                    ),
                )
                await connection.execute(
                    """
                    UPDATE run_progress
                    SET execution_state = 'ended',
                        recovery_reason = 'none',
                        updated_at = ?
                    WHERE run_id = ?
                    """,
                    (timestamp, run["id"]),
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
                    (timestamp, run["id"]),
                )
                response = RunControlAccepted(
                    creation_id=creation_id,
                    run_kind=run_kind,
                    run_state="ended",
                    resource_url=f"/creations/{creation_id}",
                )
            else:
                raise DomainError(
                    "run_not_controllable",
                    "Only a paused or quality-rejected workflow run can end.",
                    409,
                )

            await connection.execute(
                "UPDATE creations SET updated_at = ? WHERE id = ?",
                (timestamp, str(creation_id)),
            )
            await self._store_idempotency(
                connection,
                idempotency_key,
                scope,
                payload_hash,
                response,
                timestamp,
            )
            return response

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

            plans = await self._episode_plans(connection, run_id)
            if plans:
                try:
                    expected_payload = dict(
                        await self._episode_aggregate_payload(connection, run_id)
                    )
                    expected_scripts = expected_payload["content"]
                except (ValueError, DomainError) as exc:
                    raise DomainError(
                        "episode_sequence_incomplete",
                        "Every planned episode must be committed before delivery.",
                        409,
                    ) from exc
                scripts = await self._fetchone(
                    connection,
                    """
                    SELECT payload_json FROM business_checkpoints
                    WHERE run_id = ? AND stage = ?
                    """,
                    (str(run_id), InternalStage.GENERATING_EPISODE_SCRIPTS.value),
                )
                if scripts is None:
                    raise DomainError(
                        "episode_sequence_incomplete",
                        "Every planned episode must be assembled before delivery.",
                        409,
                    )
                try:
                    approved_payload = json.loads(scripts["payload_json"])
                    expected_checkpoint = {
                        **(
                            {"stage": InternalStage.GENERATING_EPISODE_SCRIPTS.value}
                            if "stage" in approved_payload
                            else {}
                        ),
                        **expected_payload,
                    }
                except (KeyError, TypeError, ValueError) as exc:
                    raise DomainError(
                        "episode_aggregate_conflict",
                        "The approved aggregate script is invalid.",
                        409,
                    ) from exc
                if (
                    approved_payload != expected_checkpoint
                    or delivery.content_package.episode_scripts != expected_scripts
                ):
                    raise DomainError(
                        "episode_aggregate_conflict",
                        "The delivery must use the committed episode aggregate.",
                        409,
                    )

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
            progress = await self._fetchone(
                connection,
                "SELECT * FROM run_progress WHERE run_id = ?",
                (str(run_id),),
            )
            if progress is None:
                raise RuntimeError("Workflow run is missing progress state")
            await connection.execute(
                """
                UPDATE run_progress
                SET execution_state = 'succeeded',
                    elapsed_seconds = ?,
                    active_started_at = NULL,
                    recovery_reason = 'none',
                    updated_at = ?
                WHERE run_id = ?
                """,
                (self._elapsed_seconds(progress, _datetime(timestamp)), timestamp, str(run_id)),
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
            progress = await self._fetchone(
                connection,
                "SELECT * FROM run_progress WHERE run_id = ?",
                (str(run_id),),
            )
            if progress is None:
                raise RuntimeError("Workflow run is missing progress state")
            await connection.execute(
                """
                UPDATE run_progress
                SET current_stage = ?,
                    execution_state = 'failed',
                    elapsed_seconds = ?,
                    active_started_at = NULL,
                    recovery_reason = 'none',
                    updated_at = ?
                WHERE run_id = ?
                """,
                (
                    failure.failed_stage.value,
                    self._elapsed_seconds(progress, _datetime(timestamp)),
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

    async def get_creation(
        self,
        creation_id: UUID,
        *,
        now: datetime | None = None,
    ) -> CreationResource | None:
        current = now or _utc_now()
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
            initial_status = await self._run_status(connection, initial, current)

            if initial_status.state != "succeeded":
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
                    revision_status = await self._revision_status(
                        connection,
                        revision_run,
                        current,
                    )

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
                    creations.output_language,
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
            output_language=run["output_language"],
            frozen_feedback=run["frozen_feedback"],
            stage_attempts=await self.get_stage_attempt_counts(run_id),
            business_checkpoints=await self.get_business_checkpoints(run_id),
            episode_plans=await self.get_episode_plans(run_id),
            episode_drafts=await self.get_episode_drafts(run_id),
        )

    async def get_run_recovery(self, run_id: UUID) -> RunRecovery:
        async with self._connection() as connection:
            run = await self._fetchone(
                connection,
                """
                SELECT id, creation_id, thread_id, state
                FROM runs
                WHERE id = ?
                  AND state IN ('queued', 'running')
                  AND EXISTS (
                      SELECT 1 FROM run_progress
                      WHERE run_progress.run_id = runs.id
                        AND run_progress.execution_state IN (
                            'queued', 'running', 'auto_resuming'
                        )
                  )
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
            episode_plans=await self.get_episode_plans(run_id),
            episode_drafts=await self.get_episode_drafts(run_id),
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
                JOIN run_progress ON run_progress.run_id = runs.id
                WHERE jobs.state = 'queued'
                  AND jobs.available_at <= ?
                  AND runs.state IN ('queued', 'running')
                  AND run_progress.execution_state IN (
                      'queued', 'running', 'auto_resuming'
                  )
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
        now: datetime,
    ) -> (
        QueuedRun
        | RunningRun
        | AutoResumingRun
        | PausedRun
        | EndedRun
        | SucceededRun
        | QualityRejectedRun
        | FailedRun
    ):
        progress_row, progress = await self._run_progress(connection, run, now)
        match progress_row["execution_state"]:
            case "queued":
                return QueuedRun(
                    progress=progress,
                    drafts=await self._draft_snapshot(connection, UUID(run["id"]), progress),
                )
            case "running":
                return RunningRun(
                    progress=progress,
                    drafts=await self._draft_snapshot(connection, UUID(run["id"]), progress),
                )
            case "auto_resuming":
                return AutoResumingRun(
                    progress=progress,
                    drafts=await self._draft_snapshot(connection, UUID(run["id"]), progress),
                )
            case "paused":
                return PausedRun(
                    progress=progress,
                    pause=self._pause_from_progress(progress_row),
                    drafts=await self._draft_snapshot(connection, UUID(run["id"]), progress),
                )
            case "ended":
                return EndedRun(
                    progress=progress,
                    drafts=await self._draft_snapshot(connection, UUID(run["id"]), progress),
                )
            case "succeeded":
                delivery = await self._load_delivery(connection, UUID(run["id"]))
                if delivery is None:
                    raise RuntimeError("Successful run is missing its delivery")
                return SucceededRun(progress=progress, result=delivery)
            case "quality_rejected":
                return QualityRejectedRun(
                    progress=progress,
                    quality_rejection=await self._quality_rejection_from_progress(
                        connection,
                        UUID(run["id"]),
                        progress_row,
                    ),
                    drafts=await self._draft_snapshot(connection, UUID(run["id"]), progress),
                )
            case "failed":
                return FailedRun(
                    progress=progress,
                    failure=self._failure_from_row(run),
                    drafts=await self._draft_snapshot(connection, UUID(run["id"]), progress),
                )
            case _:
                raise RuntimeError("Unknown workflow progress state")

    async def _revision_status(
        self,
        connection: aiosqlite.Connection,
        run: aiosqlite.Row,
        now: datetime,
    ) -> (
        RevisionQueued
        | RevisionRunning
        | RevisionAutoResuming
        | RevisionPaused
        | RevisionEnded
        | RevisionSucceeded
        | RevisionQualityRejected
        | RevisionFailed
    ):
        progress_row, progress = await self._run_progress(connection, run, now)
        match progress_row["execution_state"]:
            case "queued":
                return RevisionQueued(
                    progress=progress,
                    drafts=await self._draft_snapshot(connection, UUID(run["id"]), progress),
                )
            case "running":
                return RevisionRunning(
                    progress=progress,
                    drafts=await self._draft_snapshot(connection, UUID(run["id"]), progress),
                )
            case "auto_resuming":
                return RevisionAutoResuming(
                    progress=progress,
                    drafts=await self._draft_snapshot(connection, UUID(run["id"]), progress),
                )
            case "paused":
                return RevisionPaused(
                    progress=progress,
                    pause=self._pause_from_progress(progress_row),
                    drafts=await self._draft_snapshot(connection, UUID(run["id"]), progress),
                )
            case "ended":
                return RevisionEnded(
                    progress=progress,
                    drafts=await self._draft_snapshot(connection, UUID(run["id"]), progress),
                )
            case "succeeded":
                delivery = await self._load_delivery(connection, UUID(run["id"]))
                if delivery is None:
                    raise RuntimeError("Successful revision is missing its delivery")
                return RevisionSucceeded(progress=progress, result=delivery)
            case "quality_rejected":
                return RevisionQualityRejected(
                    progress=progress,
                    quality_rejection=await self._quality_rejection_from_progress(
                        connection,
                        UUID(run["id"]),
                        progress_row,
                    ),
                    drafts=await self._draft_snapshot(connection, UUID(run["id"]), progress),
                )
            case "failed":
                return RevisionFailed(
                    progress=progress,
                    failure=self._failure_from_row(run),
                    drafts=await self._draft_snapshot(connection, UUID(run["id"]), progress),
                )
            case _:
                raise RuntimeError("Unknown revision progress state")

    async def _run_progress(
        self,
        connection: aiosqlite.Connection,
        run: aiosqlite.Row,
        now: datetime,
    ) -> tuple[aiosqlite.Row, RunProgress]:
        progress = await self._fetchone(
            connection,
            "SELECT * FROM run_progress WHERE run_id = ?",
            (run["id"],),
        )
        if progress is None:
            raise RuntimeError("Workflow run is missing progress state")
        cursor = await connection.execute(
            "SELECT stage FROM business_checkpoints WHERE run_id = ?",
            (run["id"],),
        )
        checkpoints = {InternalStage(row["stage"]) for row in await cursor.fetchall()}
        current_internal = InternalStage(progress["current_stage"])
        completed = [
            user_stage
            for user_stage, checkpoint in _COMPLETED_STAGE_CHECKPOINTS
            if checkpoint in checkpoints
        ]
        if progress["execution_state"] == "succeeded":
            completed.append(UserStage.FINAL_REVIEW)

        execution_state = progress["execution_state"]
        episodes = await self._episode_progress(
            connection,
            UUID(run["id"]),
            progress["current_episode"],
        )
        return progress, RunProgress(
            current_stage=_USER_STAGE_BY_INTERNAL[current_internal],
            completed_stages=completed,
            elapsed_seconds=int(self._elapsed_seconds(progress, now)),
            recovery_state=(
                execution_state if execution_state in {"auto_resuming", "paused"} else "none"
            ),
            recovery_reason=(
                progress["recovery_reason"]
                if execution_state in {"auto_resuming", "paused"}
                else "none"
            ),
            final_review=FinalReviewProgress(
                l0=self._gate_progress(
                    InternalStage.ACCEPTING_L0,
                    current_internal,
                    checkpoints,
                    execution_state,
                ),
                l4=self._gate_progress(
                    InternalStage.ACCEPTING_L4,
                    current_internal,
                    checkpoints,
                    execution_state,
                ),
            ),
            episodes=episodes,
            can_continue=(
                execution_state == "paused"
                and await self._has_remaining_attempts(
                    connection,
                    run_id=progress["run_id"],
                    current_stage=progress["current_stage"],
                    current_episode=progress["current_episode"],
                )
            ),
            can_end=execution_state == "paused",
        )

    async def _draft_snapshot(
        self,
        connection: aiosqlite.Connection,
        run_id: UUID,
        progress: RunProgress,
    ) -> RunDraftSnapshot:
        cursor = await connection.execute(
            """
            SELECT stage, payload_json
            FROM business_checkpoints
            WHERE run_id = ?
            """,
            (str(run_id),),
        )
        payloads = {row["stage"]: row["payload_json"] for row in await cursor.fetchall()}
        artifacts = []
        for checkpoint_stage, user_stage in _DRAFT_STAGE_CHECKPOINTS:
            artifact = self._draft_artifact_from_payload(
                user_stage,
                payloads.get(checkpoint_stage.value),
            )
            if artifact is not None:
                artifacts.append(artifact)
        return RunDraftSnapshot(
            artifacts=artifacts,
            episodes=await self._episode_drafts(connection, run_id),
            review_status=progress.final_review,
        )

    async def _episode_progress(
        self,
        connection: aiosqlite.Connection,
        run_id: UUID,
        current_episode: int | None,
    ) -> EpisodeProgress | None:
        plans = await self._episode_plans(connection, run_id)
        if not plans:
            return None
        drafts = await self._episode_drafts(connection, run_id)
        current = int(current_episode) if current_episode is not None else None
        if current is not None and current not in {plan.episode_number for plan in plans}:
            raise RuntimeError("Workflow progress references an unplanned episode")
        return EpisodeProgress(
            total=len(plans),
            completed=len(drafts),
            current=current,
        )

    @staticmethod
    def _draft_artifact_from_payload(
        stage: UserStage,
        payload_json: str | None,
    ) -> CreativeDirectionDraft | CreativeTextDraft | None:
        if payload_json is None:
            return None
        try:
            payload = json.loads(payload_json)
        except (TypeError, ValueError):
            return None
        if not isinstance(payload, Mapping):
            return None
        try:
            if stage is UserStage.DETERMINING_DIRECTION:
                return CreativeDirectionDraft(
                    selected_l0_variant=payload["selected_l0_variant"],
                    selection_rationale=payload["selection_rationale"],
                )
            return CreativeTextDraft(stage=stage.value, content=payload["content"])
        except (KeyError, TypeError, ValueError):
            return None

    @staticmethod
    def _gate_progress(
        gate: InternalStage,
        current_stage: InternalStage,
        checkpoints: set[InternalStage],
        execution_state: str,
    ) -> Literal["pending", "running", "passed", "paused", "failed"]:
        if gate in checkpoints:
            return "passed"
        if current_stage is not gate:
            return "pending"
        if execution_state == "paused":
            return "paused"
        if execution_state in {"quality_rejected", "failed", "ended"}:
            return "failed"
        return "running"

    async def _quality_rejection_from_progress(
        self,
        connection: aiosqlite.Connection,
        run_id: UUID,
        progress: aiosqlite.Row,
    ) -> QualityGateRejection:
        stage = InternalStage(progress["current_stage"])
        if stage not in {InternalStage.ACCEPTING_L0, InternalStage.ACCEPTING_L4}:
            raise RuntimeError("Quality-rejected run is not at a final review gate")
        rejection = await self._fetchone(
            connection,
            """
            SELECT stage, evidence, attempt_number
            FROM quality_gate_rejections
            WHERE run_id = ? AND stage = ?
            ORDER BY attempt_number DESC
            LIMIT 1
            """,
            (str(run_id), stage.value),
        )
        if rejection is None:
            raise RuntimeError("Quality-rejected run is missing rejection evidence")
        return QualityGateRejection(
            stage=rejection["stage"],
            evidence=rejection["evidence"],
            attempt_count=int(rejection["attempt_number"]),
            can_retry=int(rejection["attempt_number"]) < MAX_STAGE_ATTEMPTS,
        )

    @staticmethod
    def _pause_from_progress(progress: aiosqlite.Row) -> RunPause:
        timeout_stage = progress["timeout_stage"]
        if timeout_stage is None:
            raise RuntimeError("Paused workflow run is missing its timeout stage")
        stage = UserStage(timeout_stage)
        recovery_reason = progress["recovery_reason"]
        if recovery_reason not in {
            "run_timeout",
            "relay_interruption",
            "content_rejected",
            "episode_error",
        }:
            raise RuntimeError("Paused workflow run is missing its recovery reason")
        if recovery_reason == "content_rejected":
            if progress["content_repair_count"] is None or not progress["pause_message"]:
                raise RuntimeError("Content-rejected run is missing review evidence")
            return RunPause(
                message=progress["pause_message"],
                code="content_rejected",
                stage=stage,
                content_repair_count=int(progress["content_repair_count"]),
                episode_number=(
                    int(progress["current_episode"])
                    if progress["current_episode"] is not None
                    else None
                ),
            )
        if recovery_reason == "episode_error":
            if not progress["pause_message"] or progress["current_episode"] is None:
                raise RuntimeError("Episode-error pause is missing safe recovery evidence")
            return RunPause(
                message=progress["pause_message"],
                code="episode_error",
                stage=stage,
                episode_number=int(progress["current_episode"]),
            )
        return RunPause(
            message=(
                "The relay or network connection was interrupted twice "
                "while generating this episode."
                if recovery_reason == "relay_interruption"
                and stage is UserStage.GENERATING_EPISODE_SCRIPTS
                else "The relay or network connection was interrupted twice in this stage."
                if recovery_reason == "relay_interruption"
                else "The episode exceeded its generation limit twice."
                if stage is UserStage.GENERATING_EPISODE_SCRIPTS
                else "The workflow exceeded its wall-clock limit twice in this stage."
            ),
            code=recovery_reason,
            stage=stage,
            timeout_count=int(progress["timeout_count"]),
            episode_number=(
                int(progress["current_episode"])
                if stage is UserStage.GENERATING_EPISODE_SCRIPTS
                and progress["current_episode"] is not None
                else None
            ),
        )

    @staticmethod
    def _interruption_available_at(
        current: datetime,
        recovery_reason: RecoveryReason,
        retry_delay_seconds: int,
    ) -> str:
        delay_seconds = (
            max(_MIN_RELAY_RETRY_DELAY_SECONDS, retry_delay_seconds)
            if recovery_reason == "relay_interruption"
            else retry_delay_seconds
        )
        return _timestamp(current + timedelta(seconds=delay_seconds))

    @staticmethod
    async def _mark_attempts_exhausted(
        connection: aiosqlite.Connection,
        *,
        run_id: UUID,
        creation_id: str,
        stage: InternalStage,
        attempt_count: int,
        elapsed_seconds: float,
        timestamp: str,
        episode: bool = False,
    ) -> None:
        message = (
            "The episode attempt limit has been exhausted."
            if episode
            else "The stage attempt limit has been exhausted."
        )
        await connection.execute(
            """
            UPDATE runs
            SET state = 'failed',
                failure_code = 'attempts_exhausted',
                failure_message = ?,
                failed_stage = ?,
                failure_attempt_count = ?,
                completed_at = ?,
                updated_at = ?
            WHERE id = ? AND state = 'running'
            """,
            (
                message,
                stage.value,
                max(
                    1,
                    min(
                        MAX_EPISODE_ATTEMPTS if episode else MAX_STAGE_ATTEMPTS,
                        attempt_count,
                    ),
                ),
                timestamp,
                timestamp,
                str(run_id),
            ),
        )
        await connection.execute(
            """
            UPDATE run_progress
            SET current_stage = ?,
                execution_state = 'failed',
                elapsed_seconds = ?,
                active_started_at = NULL,
                recovery_reason = 'none',
                updated_at = ?
            WHERE run_id = ?
            """,
            (stage.value, elapsed_seconds, timestamp, str(run_id)),
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
            (timestamp, creation_id),
        )

    @staticmethod
    async def _has_remaining_attempts(
        connection: aiosqlite.Connection,
        *,
        run_id: str,
        current_stage: str,
        current_episode: int | None,
    ) -> bool:
        stage = InternalStage(current_stage)
        if stage is InternalStage.GENERATING_EPISODE_SCRIPTS and current_episode is not None:
            cursor = await connection.execute(
                """
                SELECT COUNT(*) FROM episode_attempts
                WHERE run_id = ? AND episode_number = ?
                """,
                (run_id, current_episode),
            )
            row = await cursor.fetchone()
            return int(row[0]) < MAX_EPISODE_ATTEMPTS
        cursor = await connection.execute(
            "SELECT COUNT(*) FROM stage_attempts WHERE run_id = ? AND stage = ?",
            (run_id, stage.value),
        )
        row = await cursor.fetchone()
        return int(row[0]) < MAX_STAGE_ATTEMPTS

    @staticmethod
    def _elapsed_seconds(progress: aiosqlite.Row, now: datetime) -> float:
        elapsed = float(progress["elapsed_seconds"])
        active_started_at = progress["active_started_at"]
        if active_started_at is None:
            return max(0, elapsed)
        active_since = _datetime(active_started_at)
        if active_since.tzinfo is None:
            active_since = active_since.replace(tzinfo=UTC)
        current = now if now.tzinfo is not None else now.replace(tzinfo=UTC)
        return max(0, elapsed + (current.astimezone(UTC) - active_since).total_seconds())

    @staticmethod
    async def _episode_plans(
        connection: aiosqlite.Connection,
        run_id: UUID,
    ) -> list[EpisodePlan]:
        cursor = await connection.execute(
            """
            SELECT episode_number, plan
            FROM episode_plans
            WHERE run_id = ?
            ORDER BY episode_number
            """,
            (str(run_id),),
        )
        return [
            EpisodePlan(episode_number=int(row["episode_number"]), plan=row["plan"])
            for row in await cursor.fetchall()
        ]

    @staticmethod
    async def _episode_drafts(
        connection: aiosqlite.Connection,
        run_id: UUID,
    ) -> list[EpisodeDraft]:
        cursor = await connection.execute(
            """
            SELECT *
            FROM episode_drafts
            WHERE run_id = ?
            ORDER BY episode_number
            """,
            (str(run_id),),
        )
        return [Repository._episode_draft_from_row(row) for row in await cursor.fetchall()]

    @staticmethod
    def _episode_draft_from_row(row: aiosqlite.Row) -> EpisodeDraft:
        return EpisodeDraft(
            episode_number=int(row["episode_number"]),
            content=row["content"],
            content_sha256=row["content_sha256"],
            completed_at=_datetime(row["completed_at"]),
            contract_sha256=row["contract_sha256"],
            state_delta=(
                json.loads(row["state_delta_json"]) if row["state_delta_json"] is not None else None
            ),
            series_state=(
                SeriesState.model_validate_json(row["series_state_json"])
                if row["series_state_json"] is not None
                else None
            ),
            series_state_sha256=row["series_state_sha256"],
            semantic_review=(
                SemanticReview.model_validate_json(row["semantic_review_json"])
                if row["semantic_review_json"] is not None
                else None
            ),
            repair_rounds=(int(row["repair_rounds"]) if row["repair_rounds"] is not None else None),
        )

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

    async def _fetch_control_run(
        self,
        connection: aiosqlite.Connection,
        creation_id: UUID,
        run_kind: RunKind,
    ) -> aiosqlite.Row:
        creation = await self._fetchone(
            connection,
            "SELECT 1 FROM creations WHERE id = ?",
            (str(creation_id),),
        )
        if creation is None:
            raise DomainError("creation_not_found", "Creation not found.", 404)
        order = "ASC" if run_kind == "initial" else "DESC"
        run = await self._fetchone(
            connection,
            f"""
            SELECT
                runs.id,
                runs.state AS run_state,
                run_progress.current_stage,
                run_progress.current_episode,
                run_progress.execution_state
            FROM runs
            JOIN run_progress ON run_progress.run_id = runs.id
            WHERE runs.creation_id = ? AND runs.kind = ?
            ORDER BY runs.sequence {order}
            LIMIT 1
            """,
            (str(creation_id), run_kind),
        )
        if run is None:
            raise DomainError(
                "run_not_controllable",
                "The requested workflow run does not exist.",
                409,
            )
        return run

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
