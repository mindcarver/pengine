from __future__ import annotations

import hashlib
import json
import re
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
    ContinuityViolation,
    EpisodeLock,
    EpisodeStateDelta,
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
from pengine.presentation import compile_delivery_presentation, recover_delivery_presentation
from pengine.schemas import (
    AutoResumingRun,
    ContentPackage,
    CreateCreationRequest,
    CreationAccepted,
    CreationResource,
    CreativeDirectionDraft,
    CreativeTextDraft,
    Delivery,
    DeliveryPresentation,
    EndedRun,
    EpisodeDraft,
    EpisodePlan,
    EpisodeProgress,
    FailedRun,
    FinalReviewProgress,
    InternalStage,
    ModelCallSummary,
    ModelCallUsage,
    OutlineGroupProgress,
    PausedRun,
    PersonaSnapshot,
    QualityGateRejection,
    QualityRejectedRun,
    QualityRepairPlan,
    QueuedRun,
    RepairAuthorization,
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
from pengine.script_batch import (
    EpisodeCandidate,
    ScriptBatchLineage,
    build_episode_candidate,
    new_batch_id,
    new_candidate_id,
)
from pengine.series_bible import (
    GlobalDesignReview,
    SeriesBible,
    SeriesBibleContent,
    SeriesBibleSummary,
    ValidationEvidence,
    project_series_bible,
)
from pengine.series_review import (
    BoundStructuralReview,
    aggregate_script_defect_evidence,
    new_review_id,
)

SCHEMA_VERSION = 32
MAX_STAGE_ATTEMPTS = 3
MAX_EPISODE_ATTEMPTS = 3
# Failure codes an operator can resolve before reviving a terminally failed
# initial run: relay quota/availability, or a deterministic stage failure whose
# cause was fixed (the run re-enters through its approved checkpoints).
RETRYABLE_FAILURE_CODES = frozenset({"relay_unavailable", "stage_validation_failed"})
_MIN_RELAY_RETRY_DELAY_SECONDS = 10

_EXTENDED_REPAIR_STAGES = frozenset(
    {
        InternalStage.GENERATING_STORY_OUTLINE,
        InternalStage.GENERATING_CHARACTER_RELATIONSHIPS,
    }
)

# Mirrors the flattened candidate shape used by the agent review/repair loop so
# the character+relationships payload renders consistently in draft snapshots.
_CR_DRAFT_SECTIONS = (
    ("character_biographies", "人物小传 / Character Biographies"),
    ("relationship_logic", "人物关系 / Relationship Logic"),
)


def _flatten_cr_draft(
    *,
    character_biographies: str,
    relationship_logic: str,
) -> str:
    parts = {
        "character_biographies": character_biographies,
        "relationship_logic": relationship_logic,
    }
    return "\n\n".join(
        f"# {header}\n{parts[field].strip()}" for field, header in _CR_DRAFT_SECTIONS
    )


RecoveryReason = Literal[
    "run_timeout",
    "relay_interruption",
    "content_rejected",
    "episode_error",
    "context_budget",
    "relay_identity_mismatch",
    "repair_authorization",
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
    review_call_id TEXT,
    approved_at TEXT NOT NULL,
    PRIMARY KEY (run_id, stage)
);

CREATE TABLE IF NOT EXISTS deliveries (
    run_id TEXT PRIMARY KEY REFERENCES runs(id) ON DELETE CASCADE,
    content_package_json TEXT NOT NULL,
    delivery_report_json TEXT NOT NULL,
    presentation_manifest_json TEXT,
    presentation_manifest_sha256 TEXT,
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

_SCHEMA_V9_CONTENT_REJECTIONS_SQL = """
CREATE TABLE content_rejections_v9 (
    run_id TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    stage TEXT NOT NULL CHECK (
        stage IN (
            'generating_story_outline',
            'generating_character_biographies',
            'generating_relationship_logic',
            'generating_episode_outline',
            'generating_episode_scripts'
        )
    ),
    episode_number INTEGER,
    repair_rounds INTEGER NOT NULL CHECK (repair_rounds = 2),
    evidence TEXT NOT NULL,
    rejected_at TEXT NOT NULL,
    PRIMARY KEY (run_id, stage, episode_number, rejected_at),
    CHECK (
        (
            stage = 'generating_episode_scripts'
            AND episode_number IS NOT NULL
            AND episode_number >= 1
        )
        OR (
            stage != 'generating_episode_scripts'
            AND episode_number IS NULL
        )
    )
)
"""

_SCHEMA_V10_RUN_PROGRESS_SQL = """
CREATE TABLE run_progress_v10 (
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
        OR content_repair_count BETWEEN 2 AND 6
    ),
    pause_message TEXT
)
"""

_SCHEMA_V10_CONTENT_REJECTIONS_SQL = """
CREATE TABLE content_rejections_v10 (
    run_id TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    stage TEXT NOT NULL CHECK (
        stage IN (
            'generating_story_outline',
            'generating_character_biographies',
            'generating_relationship_logic',
            'generating_story_design',
            'generating_episode_outline',
            'generating_episode_scripts'
        )
    ),
    episode_number INTEGER,
    repair_rounds INTEGER NOT NULL,
    evidence TEXT NOT NULL,
    rejected_at TEXT NOT NULL,
    PRIMARY KEY (run_id, stage, episode_number, rejected_at),
    CHECK (
        (
            stage = 'generating_story_design'
            AND repair_rounds BETWEEN 2 AND 4
        )
        OR (
            stage IN (
                'generating_story_outline',
                'generating_character_biographies',
                'generating_relationship_logic'
            )
            AND repair_rounds BETWEEN 2 AND 6
        )
        OR (
            stage IN ('generating_episode_outline', 'generating_episode_scripts')
            AND repair_rounds = 2
        )
    ),
    CHECK (
        (
            stage = 'generating_episode_scripts'
            AND episode_number IS NOT NULL
            AND episode_number >= 1
        )
        OR (
            stage != 'generating_episode_scripts'
            AND episode_number IS NULL
        )
    )
)
"""

# v16 rebuilds content_rejections to accept the merged generating_story_design
# stage (2..4 repair rounds) while still tolerating legacy stage names carried
# over from rows written before the merge.
_SCHEMA_V16_CONTENT_REJECTIONS_SQL = """
CREATE TABLE content_rejections_v16 (
    run_id TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    stage TEXT NOT NULL CHECK (
        stage IN (
            'generating_story_outline',
            'generating_character_biographies',
            'generating_relationship_logic',
            'generating_story_design',
            'generating_episode_outline',
            'generating_episode_scripts'
        )
    ),
    episode_number INTEGER,
    repair_rounds INTEGER NOT NULL,
    evidence TEXT NOT NULL,
    rejected_at TEXT NOT NULL,
    PRIMARY KEY (run_id, stage, episode_number, rejected_at),
    CHECK (
        (
            stage = 'generating_story_design'
            AND repair_rounds BETWEEN 2 AND 4
        )
        OR (
            stage IN (
                'generating_story_outline',
                'generating_character_biographies',
                'generating_relationship_logic'
            )
            AND repair_rounds BETWEEN 2 AND 6
        )
        OR (
            stage IN ('generating_episode_outline', 'generating_episode_scripts')
            AND repair_rounds = 2
        )
    ),
    CHECK (
        (
            stage = 'generating_episode_scripts'
            AND episode_number IS NOT NULL
            AND episode_number >= 1
        )
        OR (
            stage != 'generating_episode_scripts'
            AND episode_number IS NULL
        )
    )
)
"""

# v17 rebuilds content_rejections to accept the split generating_story_outline
# and merged generating_character_relationships stages (2..4 repair rounds) while
# still tolerating every legacy stage name carried over from prior schema versions.
_SCHEMA_V17_CONTENT_REJECTIONS_SQL = """
CREATE TABLE content_rejections_v17 (
    run_id TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    stage TEXT NOT NULL CHECK (
        stage IN (
            'generating_story_outline',
            'generating_character_biographies',
            'generating_relationship_logic',
            'generating_story_design',
            'generating_character_relationships',
            'generating_episode_outline',
            'generating_episode_scripts'
        )
    ),
    episode_number INTEGER,
    repair_rounds INTEGER NOT NULL,
    evidence TEXT NOT NULL,
    rejected_at TEXT NOT NULL,
    PRIMARY KEY (run_id, stage, episode_number, rejected_at),
    CHECK (
        (
            stage IN (
                'generating_story_outline',
                'generating_character_relationships',
                'generating_story_design'
            )
            AND repair_rounds BETWEEN 2 AND 4
        )
        OR (
            stage IN (
                'generating_character_biographies',
                'generating_relationship_logic'
            )
            AND repair_rounds BETWEEN 2 AND 6
        )
        OR (
            stage IN ('generating_episode_outline', 'generating_episode_scripts')
            AND repair_rounds = 2
        )
    ),
    CHECK (
        (
            stage = 'generating_episode_scripts'
            AND episode_number IS NOT NULL
            AND episode_number >= 1
        )
        OR (
            stage != 'generating_episode_scripts'
            AND episode_number IS NULL
        )
    )
)
"""

_SCHEMA_V11_RUN_PROGRESS_SQL = """
CREATE TABLE run_progress_v11 (
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
            'episode_error', 'context_budget'
        )
    ),
    content_repair_count INTEGER CHECK (
        content_repair_count IS NULL
        OR content_repair_count BETWEEN 2 AND 6
    ),
    pause_message TEXT
)
"""

_SCHEMA_V11_MODEL_CALLS_SQL = """
CREATE TABLE IF NOT EXISTS model_calls (
    call_id TEXT PRIMARY KEY,
    operation_id TEXT,
    run_id TEXT,
    creation_id TEXT,
    thread_id TEXT,
    run_kind TEXT,
    role TEXT NOT NULL,
    adapter TEXT NOT NULL,
    provider TEXT NOT NULL,
    model TEXT NOT NULL,
    stage TEXT,
    episode_number INTEGER,
    candidate TEXT,
    batch TEXT,
    requested_at TEXT NOT NULL,
    finished_at TEXT,
    duration_seconds REAL,
    estimated_input_tokens INTEGER NOT NULL CHECK (estimated_input_tokens >= 0),
    estimated_output_tokens INTEGER NOT NULL CHECK (estimated_output_tokens >= 0),
    estimated_total_tokens INTEGER NOT NULL CHECK (estimated_total_tokens >= 0),
    verified_limit_tokens INTEGER,
    preflight TEXT NOT NULL CHECK (preflight IN ('ok', 'blocked')),
    status TEXT NOT NULL,
    usage_status TEXT NOT NULL CHECK (usage_status IN ('reported', 'partial', 'unavailable')),
    actual_input_tokens INTEGER,
    actual_output_tokens INTEGER,
    cache_read_tokens INTEGER,
    cache_creation_tokens INTEGER,
    finish_reason TEXT,
    outcome TEXT NOT NULL,
    error_code TEXT,
    error_type TEXT,
    safe_message TEXT,
    supersedes_call_id TEXT
);
"""

_SCHEMA_V20_RUN_PROGRESS_SQL = """
CREATE TABLE run_progress_v20 (
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
            'episode_error', 'context_budget', 'relay_identity_mismatch',
            'repair_authorization'
        )
    ),
    content_repair_count INTEGER CHECK (
        content_repair_count IS NULL
        OR content_repair_count BETWEEN 2 AND 6
    ),
    pause_message TEXT
)
"""

_SCHEMA_V11_MODEL_CALLS_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS model_calls_run_id
ON model_calls(run_id);
"""

_SCHEMA_V18_MODEL_CALLS_OPERATION_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS model_calls_operation_id
ON model_calls(run_id, operation_id);
"""

_SCHEMA_V23_QUALITY_GATE_REPAIR_SQL = """
CREATE TABLE IF NOT EXISTS quality_gate_repairs (
    run_id TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    stage TEXT NOT NULL CHECK (stage IN ('accepting_l0', 'accepting_l4')),
    rejection_attempt INTEGER NOT NULL CHECK (
        rejection_attempt >= 1 AND rejection_attempt <= 3
    ),
    plan_json TEXT,
    status TEXT NOT NULL CHECK (
        status IN ('available', 'queued', 'repairing', 'applied', 'blocked')
    ),
    original_batch_id TEXT,
    repaired_batch_id TEXT,
    requested_at TEXT,
    completed_at TEXT,
    result_evidence TEXT,
    PRIMARY KEY (run_id, stage, rejection_attempt),
    FOREIGN KEY (run_id, stage, rejection_attempt)
        REFERENCES quality_gate_rejections(run_id, stage, attempt_number)
        ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS quality_gate_repairs_status
ON quality_gate_repairs(status, requested_at);
"""

_SCHEMA_V24_SERIES_BIBLE_PROJECTION_REPAIR_SQL = """
CREATE TABLE IF NOT EXISTS series_bible_projection_repairs (
    run_id TEXT PRIMARY KEY REFERENCES runs(id) ON DELETE CASCADE,
    original_candidate_id TEXT NOT NULL REFERENCES series_bible_candidates(candidate_id),
    status TEXT NOT NULL CHECK (status IN ('reserved', 'applied', 'failed')),
    target_character_ids_json TEXT NOT NULL,
    validation_json TEXT NOT NULL,
    original_checkpoint_sha256 TEXT NOT NULL,
    repaired_checkpoint_sha256 TEXT,
    repaired_content_hash TEXT,
    generation_call_id TEXT,
    review_call_id TEXT,
    failure_message TEXT,
    created_at TEXT NOT NULL,
    completed_at TEXT
);
CREATE INDEX IF NOT EXISTS series_bible_projection_repairs_status
ON series_bible_projection_repairs(status, created_at);
INSERT OR IGNORE INTO pengine_schema(version) VALUES (24);
"""

_SCHEMA_V25_EPISODE_GENERATION_WINDOWS_SQL = """
CREATE TABLE IF NOT EXISTS episode_generation_windows (
    window_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    design_candidate_id TEXT NOT NULL,
    design_content_hash TEXT NOT NULL,
    design_epoch INTEGER NOT NULL CHECK (design_epoch >= 1),
    group_id TEXT NOT NULL,
    start_episode INTEGER NOT NULL CHECK (start_episode >= 1),
    end_episode INTEGER NOT NULL CHECK (end_episode >= start_episode),
    operation_id TEXT NOT NULL,
    call_id TEXT,
    status TEXT NOT NULL CHECK (
        status IN ('generating', 'generated', 'partially_committed', 'committed', 'failed', 'stale')
    ),
    committed_through_episode INTEGER,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    UNIQUE (run_id, operation_id)
);
CREATE INDEX IF NOT EXISTS episode_generation_windows_run
ON episode_generation_windows(run_id, created_at);

INSERT OR IGNORE INTO pengine_schema(version) VALUES (25);
"""

_SCHEMA_V26_SCRIPT_CONTEXT_AUDIT_SQL = """
INSERT OR IGNORE INTO pengine_schema(version) VALUES (26);
"""

_SCHEMA_V28_SCRIPT_TEXT_SIDECAR_SQL = """
INSERT OR IGNORE INTO pengine_schema(version) VALUES (28);
"""

_SCHEMA_V29_OUTLINE_MARKDOWN_SIDECAR_SQL = """
CREATE TABLE outline_generation_groups_v29 (
    run_id TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    group_id TEXT NOT NULL,
    position INTEGER NOT NULL CHECK (position >= 1),
    start_episode INTEGER NOT NULL CHECK (start_episode >= 1),
    end_episode INTEGER NOT NULL CHECK (end_episode >= start_episode),
    operation_id TEXT NOT NULL,
    call_id TEXT,
    status TEXT NOT NULL CHECK (
        status IN ('generating', 'body_generated', 'committed', 'failed')
    ),
    attempt_count INTEGER NOT NULL CHECK (attempt_count >= 1),
    outline_markdown TEXT,
    outline_markdown_sha256 TEXT,
    body_call_id TEXT,
    sidecar_call_id TEXT,
    content_json TEXT,
    content_sha256 TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (run_id, group_id),
    UNIQUE (run_id, position),
    CHECK (
        (
            status = 'committed'
            AND call_id IS NOT NULL
            AND content_json IS NOT NULL
            AND content_sha256 IS NOT NULL
            AND (
                (
                    outline_markdown IS NULL
                    AND outline_markdown_sha256 IS NULL
                    AND body_call_id IS NULL
                    AND sidecar_call_id IS NULL
                )
                OR (
                    outline_markdown IS NOT NULL
                    AND outline_markdown_sha256 IS NOT NULL
                    AND body_call_id IS NOT NULL
                    AND sidecar_call_id IS NOT NULL
                )
            )
        )
        OR (
            status = 'body_generated'
            AND call_id IS NULL
            AND outline_markdown IS NOT NULL
            AND outline_markdown_sha256 IS NOT NULL
            AND body_call_id IS NOT NULL
            AND sidecar_call_id IS NULL
            AND content_json IS NULL
            AND content_sha256 IS NULL
        )
        OR (
            status IN ('generating', 'failed')
            AND call_id IS NULL
            AND outline_markdown IS NULL
            AND outline_markdown_sha256 IS NULL
            AND body_call_id IS NULL
            AND sidecar_call_id IS NULL
            AND content_json IS NULL
            AND content_sha256 IS NULL
        )
    )
);

INSERT INTO outline_generation_groups_v29(
    run_id, group_id, position, start_episode, end_episode, operation_id,
    call_id, status, attempt_count, content_json, content_sha256, created_at, updated_at
)
SELECT
    run_id, group_id, position, start_episode, end_episode, operation_id,
    call_id, status, attempt_count, content_json, content_sha256, created_at, updated_at
FROM outline_generation_groups;

DROP TABLE outline_generation_groups;
ALTER TABLE outline_generation_groups_v29 RENAME TO outline_generation_groups;
CREATE INDEX outline_generation_groups_run
ON outline_generation_groups(run_id, position);

INSERT OR IGNORE INTO pengine_schema(version) VALUES (29);
"""

_SCHEMA_V30_OUTLINE_MARKDOWN_FAILURES_SQL = """
CREATE TABLE IF NOT EXISTS outline_markdown_failures (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    group_id TEXT NOT NULL,
    operation_id TEXT NOT NULL,
    attempt_index INTEGER NOT NULL CHECK (attempt_index >= 1),
    raw_text TEXT NOT NULL,
    raw_text_sha256 TEXT NOT NULL,
    normalized_text TEXT NOT NULL,
    normalized_text_sha256 TEXT NOT NULL,
    parse_error TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS outline_markdown_failures_run
ON outline_markdown_failures(run_id, group_id, attempt_index);

INSERT OR IGNORE INTO pengine_schema(version) VALUES (30);
"""

_SCHEMA_V31_OUTLINE_GROUP_REJECTIONS_SQL = """
CREATE TABLE IF NOT EXISTS outline_group_rejections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    group_id TEXT NOT NULL,
    evidence TEXT NOT NULL,
    repair_rounds INTEGER NOT NULL CHECK (repair_rounds >= 1),
    rejected_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS outline_group_rejections_run
ON outline_group_rejections(run_id, group_id, rejected_at);

INSERT OR IGNORE INTO pengine_schema(version) VALUES (31);
"""

_SCHEMA_V32_PROJECTION_REPAIR_ATTEMPTS_SQL = """
CREATE TABLE IF NOT EXISTS series_bible_projection_repairs_v32 (
    run_id TEXT NOT NULL,
    original_candidate_id TEXT NOT NULL,
    status TEXT NOT NULL,
    target_character_ids_json TEXT NOT NULL,
    validation_json TEXT NOT NULL,
    original_checkpoint_sha256 TEXT,
    repaired_checkpoint_sha256 TEXT,
    repaired_content_hash TEXT,
    generation_call_id TEXT,
    review_call_id TEXT,
    failure_message TEXT,
    created_at TEXT NOT NULL,
    completed_at TEXT,
    attempt_count INTEGER NOT NULL DEFAULT 1,
    PRIMARY KEY (run_id)
);

INSERT INTO series_bible_projection_repairs_v32(
    run_id, original_candidate_id, status, target_character_ids_json,
    validation_json, original_checkpoint_sha256, repaired_checkpoint_sha256,
    repaired_content_hash, generation_call_id, review_call_id,
    failure_message, created_at, completed_at
)
SELECT
    run_id, original_candidate_id, status, target_character_ids_json,
    validation_json, original_checkpoint_sha256, repaired_checkpoint_sha256,
    repaired_content_hash, generation_call_id, review_call_id,
    failure_message, created_at, completed_at
FROM series_bible_projection_repairs;

DROP TABLE series_bible_projection_repairs;
ALTER TABLE series_bible_projection_repairs_v32 RENAME TO series_bible_projection_repairs;

INSERT OR IGNORE INTO pengine_schema(version) VALUES (32);
"""

_SCHEMA_V27_GROUPED_OUTLINE_SQL = """
CREATE TABLE IF NOT EXISTS outline_season_maps (
    run_id TEXT PRIMARY KEY REFERENCES runs(id) ON DELETE CASCADE,
    content_json TEXT NOT NULL,
    content_sha256 TEXT NOT NULL,
    call_id TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS outline_generation_groups (
    run_id TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    group_id TEXT NOT NULL,
    position INTEGER NOT NULL CHECK (position >= 1),
    start_episode INTEGER NOT NULL CHECK (start_episode >= 1),
    end_episode INTEGER NOT NULL CHECK (end_episode >= start_episode),
    operation_id TEXT NOT NULL,
    call_id TEXT,
    status TEXT NOT NULL CHECK (status IN ('generating', 'committed', 'failed')),
    attempt_count INTEGER NOT NULL CHECK (attempt_count >= 1),
    content_json TEXT,
    content_sha256 TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (run_id, group_id),
    UNIQUE (run_id, position),
    CHECK (
        (
            status = 'committed'
            AND call_id IS NOT NULL
            AND content_json IS NOT NULL
            AND content_sha256 IS NOT NULL
        )
        OR (
            status != 'committed'
            AND call_id IS NULL
            AND content_json IS NULL
            AND content_sha256 IS NULL
        )
    )
);
CREATE INDEX IF NOT EXISTS outline_generation_groups_run
ON outline_generation_groups(run_id, position);

INSERT OR IGNORE INTO pengine_schema(version) VALUES (27);
"""

_SCHEMA_V12_SERIES_BIBLE_SQL = """
CREATE TABLE IF NOT EXISTS series_bible_candidates (
    candidate_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    creation_id TEXT,
    version INTEGER NOT NULL CHECK (version >= 1),
    design_epoch INTEGER NOT NULL CHECK (design_epoch >= 1),
    content_hash TEXT NOT NULL,
    status TEXT NOT NULL CHECK (
        status IN ('unvalidated', 'validated', 'active', 'superseded', 'stale')
    ),
    l0_variant TEXT NOT NULL,
    genre TEXT NOT NULL CHECK (genre IN ('mystery', 'general')),
    lineage_json TEXT NOT NULL,
    content_json TEXT NOT NULL,
    validation_json TEXT,
    global_review_json TEXT,
    created_at TEXT NOT NULL,
    activated_at TEXT,
    superseded_at TEXT
);
CREATE INDEX IF NOT EXISTS series_bible_candidates_run
ON series_bible_candidates(run_id);

CREATE TABLE IF NOT EXISTS series_bible_lineage (
    run_id TEXT PRIMARY KEY REFERENCES runs(id) ON DELETE CASCADE,
    creation_id TEXT NOT NULL,
    active_candidate_id TEXT,
    active_design_epoch INTEGER NOT NULL DEFAULT 0 CHECK (active_design_epoch >= 0),
    active_content_hash TEXT,
    rebuild_count INTEGER NOT NULL DEFAULT 0 CHECK (rebuild_count IN (0, 1)),
    updated_at TEXT NOT NULL
);

INSERT OR IGNORE INTO pengine_schema(version) VALUES (12);
"""

_SCHEMA_V13_SCRIPT_BATCH_SQL = """
CREATE TABLE IF NOT EXISTS script_batches (
    batch_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    creation_id TEXT,
    run_kind TEXT NOT NULL CHECK (run_kind IN ('initial', 'revision')),
    batch_epoch INTEGER NOT NULL CHECK (batch_epoch >= 1),
    status TEXT NOT NULL CHECK (status IN ('active', 'superseded')),
    design_candidate_id TEXT NOT NULL,
    design_content_hash TEXT NOT NULL,
    design_epoch INTEGER NOT NULL CHECK (design_epoch >= 1),
    active_pointers_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    superseded_at TEXT
);
CREATE UNIQUE INDEX IF NOT EXISTS script_batches_one_active
ON script_batches(run_id) WHERE status = 'active';

CREATE TABLE IF NOT EXISTS episode_candidates (
    candidate_id TEXT PRIMARY KEY,
    batch_id TEXT NOT NULL REFERENCES script_batches(batch_id) ON DELETE CASCADE,
    batch_epoch INTEGER NOT NULL CHECK (batch_epoch >= 1),
    run_id TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    design_candidate_id TEXT NOT NULL,
    design_content_hash TEXT NOT NULL,
    design_epoch INTEGER NOT NULL CHECK (design_epoch >= 1),
    episode_number INTEGER NOT NULL CHECK (episode_number >= 1),
    version INTEGER NOT NULL CHECK (version >= 1),
    content TEXT NOT NULL,
    content_sha256 TEXT NOT NULL,
    predecessor_candidate_id TEXT,
    predecessor_sha256 TEXT,
    call_id TEXT NOT NULL,
    writer_notes TEXT NOT NULL DEFAULT '',
    state_delta_json TEXT NOT NULL,
    series_state_json TEXT NOT NULL,
    series_state_sha256 TEXT NOT NULL,
    semantic_review_json TEXT,
    repair_rounds INTEGER NOT NULL DEFAULT 0,
    status TEXT NOT NULL CHECK (
        status IN ('unvalidated', 'validated', 'active', 'superseded', 'stale')
    ),
    created_at TEXT NOT NULL,
    activated_at TEXT,
    superseded_at TEXT,
    UNIQUE (batch_id, episode_number, version)
);
CREATE INDEX IF NOT EXISTS episode_candidates_batch
ON episode_candidates(batch_id);
CREATE INDEX IF NOT EXISTS episode_candidates_run
ON episode_candidates(run_id);

INSERT OR IGNORE INTO pengine_schema(version) VALUES (13);
"""

_SCHEMA_V14_REVIEW_SQL = """
CREATE TABLE IF NOT EXISTS series_reviews (
    review_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    review_epoch INTEGER NOT NULL CHECK (review_epoch >= 1),
    review_type TEXT NOT NULL CHECK (review_type IN ('milestone', 'final')),
    episode_number INTEGER NOT NULL CHECK (episode_number >= 1),
    design_candidate_id TEXT NOT NULL,
    design_content_hash TEXT NOT NULL,
    design_epoch INTEGER NOT NULL CHECK (design_epoch >= 1),
    batch_id TEXT NOT NULL,
    batch_epoch INTEGER NOT NULL CHECK (batch_epoch >= 1),
    prefix_hash TEXT NOT NULL,
    call_id TEXT NOT NULL,
    passed INTEGER NOT NULL CHECK (passed IN (0, 1)),
    category TEXT NOT NULL CHECK (
        category IN (
            'pass', 'design_defect', 'script_defect',
            'protocol_failure', 'transient_failure', 'stale'
        )
    ),
    evidence TEXT NOT NULL,
    earliest_affected_episode INTEGER CHECK (
        earliest_affected_episode IS NULL OR earliest_affected_episode >= 1
    ),
    status TEXT NOT NULL CHECK (status IN ('active', 'superseded', 'stale')),
    created_at TEXT NOT NULL,
    consumed_at TEXT
);
CREATE INDEX IF NOT EXISTS series_reviews_run
ON series_reviews(run_id);

CREATE TABLE IF NOT EXISTS repair_authorizations (
    run_id TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    authorization_epoch INTEGER NOT NULL CHECK (authorization_epoch >= 1),
    kind TEXT NOT NULL CHECK (kind IN ('design_rebuild', 'suffix_rewrite')),
    design_candidate_id TEXT NOT NULL,
    design_content_hash TEXT NOT NULL,
    design_epoch INTEGER NOT NULL CHECK (design_epoch >= 1),
    batch_id TEXT NOT NULL,
    batch_epoch INTEGER NOT NULL CHECK (batch_epoch >= 1),
    earliest_affected_episode INTEGER,
    range_episodes INTEGER CHECK (range_episodes IS NULL OR range_episodes >= 1),
    estimated_tokens INTEGER CHECK (estimated_tokens IS NULL OR estimated_tokens >= 0),
    evidence TEXT NOT NULL,
    review_id TEXT NOT NULL,
    granted_at TEXT,
    consumed_at TEXT,
    PRIMARY KEY (run_id, authorization_epoch)
);

DROP TABLE IF EXISTS run_progress_v14;
CREATE TABLE run_progress_v14 (
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
            'episode_error', 'context_budget', 'repair_authorization'
        )
    ),
    content_repair_count INTEGER CHECK (
        content_repair_count IS NULL
        OR content_repair_count BETWEEN 2 AND 6
    ),
    pause_message TEXT
);

INSERT INTO run_progress_v14(
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
ALTER TABLE run_progress_v14 RENAME TO run_progress;

INSERT OR IGNORE INTO pengine_schema(version) VALUES (14);
"""


ModelT = TypeVar("ModelT", bound=BaseModel)

RunKind = Literal["initial", "revision"]
ControlState = Literal["queued", "running", "auto_resuming", "ended"]

_USER_STAGE_BY_INTERNAL = {
    InternalStage.LOADING_PERSONA: UserStage.DETERMINING_DIRECTION,
    InternalStage.SELECTING_L0_VARIANT: UserStage.DETERMINING_DIRECTION,
    InternalStage.GENERATING_STORY_OUTLINE: UserStage.GENERATING_STORY_OUTLINE,
    InternalStage.GENERATING_CHARACTER_RELATIONSHIPS: UserStage.GENERATING_CHARACTER_RELATIONSHIPS,
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
        UserStage.GENERATING_CHARACTER_RELATIONSHIPS,
        InternalStage.GENERATING_CHARACTER_RELATIONSHIPS,
    ),
    (UserStage.GENERATING_EPISODE_OUTLINE, InternalStage.GENERATING_EPISODE_OUTLINE),
    (UserStage.GENERATING_EPISODE_SCRIPTS, InternalStage.GENERATING_EPISODE_SCRIPTS),
)

_DRAFT_STAGE_CHECKPOINTS = (
    (InternalStage.SELECTING_L0_VARIANT, UserStage.DETERMINING_DIRECTION),
    (InternalStage.GENERATING_STORY_OUTLINE, UserStage.GENERATING_STORY_OUTLINE),
    (
        InternalStage.GENERATING_CHARACTER_RELATIONSHIPS,
        UserStage.GENERATING_CHARACTER_RELATIONSHIPS,
    ),
    (InternalStage.GENERATING_EPISODE_OUTLINE, UserStage.GENERATING_EPISODE_OUTLINE),
)

_INTERNAL_STAGE_ORDER = (
    InternalStage.LOADING_PERSONA,
    InternalStage.SELECTING_L0_VARIANT,
    InternalStage.GENERATING_STORY_OUTLINE,
    InternalStage.GENERATING_CHARACTER_RELATIONSHIPS,
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


def _is_bound_quality_patch(original: str, repaired: str, excerpts: list[str]) -> bool:
    if repaired == original or not excerpts:
        return False
    ordered = sorted(excerpts, key=original.find)
    if any(not excerpt or original.count(excerpt) != 1 for excerpt in ordered):
        return False
    cursor = 0
    unchanged: list[str] = []
    for excerpt in ordered:
        start = original.find(excerpt, cursor)
        if start < cursor:
            return False
        unchanged.append(original[cursor:start])
        cursor = start + len(excerpt)
    unchanged.append(original[cursor:])
    pattern = "^" + ".*?".join(re.escape(part) for part in unchanged) + "$"
    return re.fullmatch(pattern, repaired, flags=re.DOTALL) is not None


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
        connection = await aiosqlite.connect(self.database_path, timeout=30)
        connection.row_factory = aiosqlite.Row
        await connection.execute("PRAGMA foreign_keys = ON")
        await connection.execute("PRAGMA busy_timeout = 30000")
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
            if schema_version == 8:
                await connection.execute("BEGIN IMMEDIATE")
                try:
                    await connection.execute(_SCHEMA_V9_CONTENT_REJECTIONS_SQL)
                    await connection.execute(
                        """
                        INSERT INTO content_rejections_v9(
                            run_id, stage, episode_number, repair_rounds,
                            evidence, rejected_at
                        )
                        SELECT
                            run_id, stage, episode_number, repair_rounds,
                            evidence, rejected_at
                        FROM content_rejections
                        """
                    )
                    await connection.execute("DROP TABLE content_rejections")
                    await connection.execute(
                        "ALTER TABLE content_rejections_v9 RENAME TO content_rejections"
                    )
                    await connection.execute(
                        "INSERT OR IGNORE INTO pengine_schema(version) VALUES (9)"
                    )
                except BaseException:
                    await connection.rollback()
                    raise
                else:
                    await connection.commit()
                    schema_version = 9
            if schema_version == 9:
                await connection.execute("BEGIN IMMEDIATE")
                try:
                    await connection.execute("DROP TABLE IF EXISTS run_progress_v10")
                    await connection.execute(_SCHEMA_V10_RUN_PROGRESS_SQL)
                    await connection.execute(
                        """
                        INSERT INTO run_progress_v10(
                            run_id, current_stage, execution_state, elapsed_seconds,
                            active_started_at, timeout_stage, timeout_count, updated_at,
                            current_episode, recovery_reason, content_repair_count, pause_message
                        )
                        SELECT
                            run_id, current_stage, execution_state, elapsed_seconds,
                            active_started_at, timeout_stage, timeout_count, updated_at,
                            current_episode, recovery_reason, content_repair_count, pause_message
                        FROM run_progress
                        """
                    )
                    await connection.execute("DROP TABLE run_progress")
                    await connection.execute("ALTER TABLE run_progress_v10 RENAME TO run_progress")

                    await connection.execute("DROP TABLE IF EXISTS content_rejections_v10")
                    await connection.execute(_SCHEMA_V10_CONTENT_REJECTIONS_SQL)
                    await connection.execute(
                        """
                        INSERT INTO content_rejections_v10(
                            run_id, stage, episode_number, repair_rounds,
                            evidence, rejected_at
                        )
                        SELECT
                            run_id, stage, episode_number, repair_rounds,
                            evidence, rejected_at
                        FROM content_rejections
                        """
                    )
                    await connection.execute("DROP TABLE content_rejections")
                    await connection.execute(
                        "ALTER TABLE content_rejections_v10 RENAME TO content_rejections"
                    )
                    await connection.execute(
                        "INSERT OR IGNORE INTO pengine_schema(version) VALUES (10)"
                    )
                except BaseException:
                    await connection.rollback()
                    raise
                else:
                    await connection.commit()
                    schema_version = 10
            if schema_version == 10:
                await connection.execute("BEGIN IMMEDIATE")
                try:
                    await connection.execute("DROP TABLE IF EXISTS run_progress_v11")
                    await connection.execute(_SCHEMA_V11_RUN_PROGRESS_SQL)
                    await connection.execute(
                        """
                        INSERT INTO run_progress_v11(
                            run_id, current_stage, execution_state, elapsed_seconds,
                            active_started_at, timeout_stage, timeout_count, updated_at,
                            current_episode, recovery_reason, content_repair_count, pause_message
                        )
                        SELECT
                            run_id, current_stage, execution_state, elapsed_seconds,
                            active_started_at, timeout_stage, timeout_count, updated_at,
                            current_episode, recovery_reason, content_repair_count, pause_message
                        FROM run_progress
                        """
                    )
                    await connection.execute("DROP TABLE run_progress")
                    await connection.execute("ALTER TABLE run_progress_v11 RENAME TO run_progress")

                    await connection.execute(_SCHEMA_V11_MODEL_CALLS_SQL)
                    await connection.execute(_SCHEMA_V11_MODEL_CALLS_INDEX_SQL)
                    await connection.execute(
                        "INSERT OR IGNORE INTO pengine_schema(version) VALUES (11)"
                    )
                except BaseException:
                    await connection.rollback()
                    raise
                else:
                    await connection.commit()
                    schema_version = 11
            if schema_version == 11:
                await connection.executescript(_SCHEMA_V12_SERIES_BIBLE_SQL)
                await connection.commit()
                schema_version = 12
            if schema_version == 12:
                await connection.executescript(_SCHEMA_V13_SCRIPT_BATCH_SQL)
                await connection.commit()
                schema_version = 13
            if schema_version == 13:
                await connection.execute("BEGIN IMMEDIATE")
                try:
                    batch_columns = await (
                        await connection.execute("PRAGMA table_info(script_batches)")
                    ).fetchall()
                    if "suffix_rewrite_count" not in {column[1] for column in batch_columns}:
                        await connection.execute(
                            """
                            ALTER TABLE script_batches
                            ADD COLUMN suffix_rewrite_count INTEGER NOT NULL DEFAULT 0
                                CHECK (suffix_rewrite_count IN (0, 1))
                            """
                        )
                    await connection.executescript(_SCHEMA_V14_REVIEW_SQL)
                    await connection.execute(
                        "INSERT OR IGNORE INTO pengine_schema(version) VALUES (14)"
                    )
                except BaseException:
                    await connection.rollback()
                    raise
                else:
                    await connection.commit()
                    schema_version = 14
            if schema_version == 14:
                await connection.execute("BEGIN IMMEDIATE")
                try:
                    auth_columns = await (
                        await connection.execute("PRAGMA table_info(repair_authorizations)")
                    ).fetchall()
                    if "rebuild_candidate_id" not in {column[1] for column in auth_columns}:
                        await connection.execute(
                            "ALTER TABLE repair_authorizations ADD COLUMN rebuild_candidate_id TEXT"
                        )
                    await connection.execute(
                        "INSERT OR IGNORE INTO pengine_schema(version) VALUES (15)"
                    )
                except BaseException:
                    await connection.rollback()
                    raise
                else:
                    await connection.commit()
                    schema_version = 15
            if schema_version == 15:
                await connection.execute("BEGIN IMMEDIATE")
                try:
                    # The merged story-design stage (generating_story_design) replaces the
                    # three former story-artifact stages and caps repair rounds at four.
                    # Rebuild content_rejections so the new stage name and the 2..4 bound are
                    # accepted, while preserving every existing rejection row.
                    await connection.execute("DROP TABLE IF EXISTS content_rejections_v16")
                    await connection.execute(_SCHEMA_V16_CONTENT_REJECTIONS_SQL)
                    await connection.execute(
                        """
                        INSERT INTO content_rejections_v16(
                            run_id, stage, episode_number, repair_rounds,
                            evidence, rejected_at
                        )
                        SELECT
                            run_id, stage, episode_number, repair_rounds,
                            evidence, rejected_at
                        FROM content_rejections
                        """
                    )
                    await connection.execute("DROP TABLE content_rejections")
                    await connection.execute(
                        "ALTER TABLE content_rejections_v16 RENAME TO content_rejections"
                    )
                    await connection.execute(
                        "INSERT OR IGNORE INTO pengine_schema(version) VALUES (16)"
                    )
                except BaseException:
                    await connection.rollback()
                    raise
                else:
                    await connection.commit()
                    schema_version = 16
            if schema_version == 16:
                await connection.execute("BEGIN IMMEDIATE")
                try:
                    # Split the story-design stage into a light generating_story_outline
                    # gate plus a merged generating_character_relationships stage.
                    # Rebuild content_rejections so the new stage names are accepted while
                    # preserving every existing rejection row (all legacy names tolerated).
                    await connection.execute("DROP TABLE IF EXISTS content_rejections_v17")
                    await connection.execute(_SCHEMA_V17_CONTENT_REJECTIONS_SQL)
                    await connection.execute(
                        """
                        INSERT INTO content_rejections_v17(
                            run_id, stage, episode_number, repair_rounds,
                            evidence, rejected_at
                        )
                        SELECT
                            run_id, stage, episode_number, repair_rounds,
                            evidence, rejected_at
                        FROM content_rejections
                        """
                    )
                    await connection.execute("DROP TABLE content_rejections")
                    await connection.execute(
                        "ALTER TABLE content_rejections_v17 RENAME TO content_rejections"
                    )
                    await connection.execute(
                        "INSERT OR IGNORE INTO pengine_schema(version) VALUES (17)"
                    )
                except BaseException:
                    await connection.rollback()
                    raise
                else:
                    await connection.commit()
                    schema_version = 17
            if schema_version == 17:
                await connection.execute("BEGIN IMMEDIATE")
                try:
                    await connection.execute("DROP TABLE IF EXISTS episode_attempts_v18")
                    await connection.execute(
                        """
                        CREATE TABLE episode_attempts_v18 (
                            run_id TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
                            episode_number INTEGER NOT NULL CHECK (episode_number >= 1),
                            attempt_cycle INTEGER NOT NULL CHECK (attempt_cycle >= 0),
                            attempt_number INTEGER NOT NULL CHECK (
                                attempt_number >= 1 AND attempt_number <= 3
                            ),
                            recorded_at TEXT NOT NULL,
                            PRIMARY KEY (run_id, episode_number, attempt_cycle, attempt_number)
                        )
                        """
                    )
                    await connection.execute(
                        """
                        INSERT INTO episode_attempts_v18(
                            run_id, episode_number, attempt_cycle, attempt_number, recorded_at
                        )
                        SELECT run_id, episode_number, 0, attempt_number, recorded_at
                        FROM episode_attempts
                        """
                    )
                    await connection.execute("DROP TABLE episode_attempts")
                    await connection.execute(
                        "ALTER TABLE episode_attempts_v18 RENAME TO episode_attempts"
                    )
                    await connection.execute(
                        """
                        CREATE INDEX IF NOT EXISTS episode_attempts_current
                        ON episode_attempts(run_id, episode_number, attempt_cycle)
                        """
                    )
                    await connection.execute(
                        """
                        CREATE TABLE IF NOT EXISTS episode_attempt_cycles (
                            run_id TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
                            attempt_cycle INTEGER NOT NULL CHECK (attempt_cycle >= 0),
                            from_episode INTEGER NOT NULL CHECK (from_episode >= 1),
                            to_episode INTEGER NOT NULL CHECK (to_episode >= from_episode),
                            started_at TEXT NOT NULL,
                            PRIMARY KEY (run_id, attempt_cycle)
                        )
                        """
                    )
                    await connection.execute(
                        """
                        CREATE TABLE IF NOT EXISTS episode_attempt_current (
                            run_id TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
                            episode_number INTEGER NOT NULL CHECK (episode_number >= 1),
                            attempt_cycle INTEGER NOT NULL CHECK (attempt_cycle >= 0),
                            PRIMARY KEY (run_id, episode_number),
                            FOREIGN KEY (run_id, attempt_cycle)
                                REFERENCES episode_attempt_cycles(run_id, attempt_cycle)
                        )
                        """
                    )
                    await connection.execute(
                        """
                        INSERT OR IGNORE INTO episode_attempt_cycles(
                            run_id, attempt_cycle, from_episode, to_episode, started_at
                        )
                        SELECT
                            plans.run_id,
                            0,
                            1,
                            MAX(plans.episode_number),
                            runs.created_at
                        FROM episode_plans AS plans
                        JOIN runs ON runs.id = plans.run_id
                        GROUP BY plans.run_id
                        """
                    )
                    await connection.execute(
                        """
                        INSERT OR IGNORE INTO episode_attempt_cycles(
                            run_id, attempt_cycle, from_episode, to_episode, started_at
                        )
                        SELECT
                            attempts.run_id,
                            0,
                            MIN(attempts.episode_number),
                            MAX(attempts.episode_number),
                            MIN(attempts.recorded_at)
                        FROM episode_attempts AS attempts
                        WHERE NOT EXISTS (
                            SELECT 1
                            FROM episode_attempt_cycles AS cycles
                            WHERE cycles.run_id = attempts.run_id
                              AND cycles.attempt_cycle = 0
                        )
                        GROUP BY attempts.run_id
                        """
                    )
                    await connection.execute(
                        """
                        INSERT OR IGNORE INTO episode_attempt_current(
                            run_id, episode_number, attempt_cycle
                        )
                        SELECT run_id, episode_number, 0
                        FROM episode_plans
                        WHERE EXISTS (
                            SELECT 1
                            FROM episode_attempt_cycles
                            WHERE episode_attempt_cycles.run_id = episode_plans.run_id
                              AND episode_attempt_cycles.attempt_cycle = 0
                        )
                        """
                    )
                    await connection.execute(
                        """
                        INSERT OR IGNORE INTO episode_attempt_current(
                            run_id, episode_number, attempt_cycle
                        )
                        SELECT DISTINCT run_id, episode_number, 0
                        FROM episode_attempts
                        WHERE EXISTS (
                            SELECT 1
                            FROM episode_attempt_cycles
                            WHERE episode_attempt_cycles.run_id = episode_attempts.run_id
                              AND episode_attempt_cycles.attempt_cycle = 0
                        )
                        """
                    )
                    model_call_columns = await (
                        await connection.execute("PRAGMA table_info(model_calls)")
                    ).fetchall()
                    if "operation_id" not in {column[1] for column in model_call_columns}:
                        await connection.execute(
                            "ALTER TABLE model_calls ADD COLUMN operation_id TEXT"
                        )
                    checkpoint_columns = await (
                        await connection.execute("PRAGMA table_info(business_checkpoints)")
                    ).fetchall()
                    if "review_call_id" not in {column[1] for column in checkpoint_columns}:
                        await connection.execute(
                            "ALTER TABLE business_checkpoints ADD COLUMN review_call_id TEXT"
                        )
                    await connection.execute(_SCHEMA_V18_MODEL_CALLS_OPERATION_INDEX_SQL)
                    await connection.execute(
                        "INSERT OR IGNORE INTO pengine_schema(version) VALUES (18)"
                    )
                except BaseException:
                    await connection.rollback()
                    raise
                else:
                    await connection.commit()
                    schema_version = 18
            if schema_version == 18:
                await connection.execute("BEGIN IMMEDIATE")
                try:
                    episode_attempt_columns = await (
                        await connection.execute("PRAGMA table_info(episode_attempts)")
                    ).fetchall()
                    if "attempt_cycle" not in {column[1] for column in episode_attempt_columns}:
                        await connection.execute("DROP TABLE IF EXISTS episode_attempt_current")
                        await connection.execute("DROP TABLE IF EXISTS episode_attempt_cycles")
                        await connection.execute("DROP TABLE IF EXISTS episode_attempts_v19")
                        await connection.execute(
                            """
                            CREATE TABLE episode_attempts_v19 (
                                run_id TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
                                episode_number INTEGER NOT NULL CHECK (episode_number >= 1),
                                attempt_cycle INTEGER NOT NULL CHECK (attempt_cycle >= 0),
                                attempt_number INTEGER NOT NULL CHECK (
                                    attempt_number >= 1 AND attempt_number <= 3
                                ),
                                recorded_at TEXT NOT NULL,
                                PRIMARY KEY (
                                    run_id, episode_number, attempt_cycle, attempt_number
                                )
                            )
                            """
                        )
                        await connection.execute(
                            """
                            INSERT INTO episode_attempts_v19(
                                run_id, episode_number, attempt_cycle,
                                attempt_number, recorded_at
                            )
                            SELECT run_id, episode_number, 0, attempt_number, recorded_at
                            FROM episode_attempts
                            """
                        )
                        await connection.execute("DROP TABLE episode_attempts")
                        await connection.execute(
                            "ALTER TABLE episode_attempts_v19 RENAME TO episode_attempts"
                        )
                    await connection.execute(
                        """
                        CREATE INDEX IF NOT EXISTS episode_attempts_current
                        ON episode_attempts(run_id, episode_number, attempt_cycle)
                        """
                    )
                    await connection.execute(
                        """
                        CREATE TABLE IF NOT EXISTS episode_attempt_cycles (
                            run_id TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
                            attempt_cycle INTEGER NOT NULL CHECK (attempt_cycle >= 0),
                            from_episode INTEGER NOT NULL CHECK (from_episode >= 1),
                            to_episode INTEGER NOT NULL CHECK (to_episode >= from_episode),
                            started_at TEXT NOT NULL,
                            PRIMARY KEY (run_id, attempt_cycle)
                        )
                        """
                    )
                    await connection.execute(
                        """
                        CREATE TABLE IF NOT EXISTS episode_attempt_current (
                            run_id TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
                            episode_number INTEGER NOT NULL CHECK (episode_number >= 1),
                            attempt_cycle INTEGER NOT NULL CHECK (attempt_cycle >= 0),
                            PRIMARY KEY (run_id, episode_number),
                            FOREIGN KEY (run_id, attempt_cycle)
                                REFERENCES episode_attempt_cycles(run_id, attempt_cycle)
                        )
                        """
                    )
                    await connection.execute(
                        """
                        INSERT OR IGNORE INTO episode_attempt_cycles(
                            run_id, attempt_cycle, from_episode, to_episode, started_at
                        )
                        SELECT
                            plans.run_id,
                            0,
                            1,
                            MAX(plans.episode_number),
                            runs.created_at
                        FROM episode_plans AS plans
                        JOIN runs ON runs.id = plans.run_id
                        GROUP BY plans.run_id
                        """
                    )
                    await connection.execute(
                        """
                        INSERT OR IGNORE INTO episode_attempt_cycles(
                            run_id, attempt_cycle, from_episode, to_episode, started_at
                        )
                        SELECT
                            attempts.run_id,
                            0,
                            MIN(attempts.episode_number),
                            MAX(attempts.episode_number),
                            MIN(attempts.recorded_at)
                        FROM episode_attempts AS attempts
                        WHERE NOT EXISTS (
                            SELECT 1
                            FROM episode_attempt_cycles AS cycles
                            WHERE cycles.run_id = attempts.run_id
                              AND cycles.attempt_cycle = 0
                        )
                        GROUP BY attempts.run_id
                        """
                    )
                    await connection.execute(
                        """
                        INSERT OR IGNORE INTO episode_attempt_current(
                            run_id, episode_number, attempt_cycle
                        )
                        SELECT run_id, episode_number, 0
                        FROM episode_plans
                        WHERE EXISTS (
                            SELECT 1
                            FROM episode_attempt_cycles
                            WHERE episode_attempt_cycles.run_id = episode_plans.run_id
                              AND episode_attempt_cycles.attempt_cycle = 0
                        )
                        """
                    )
                    await connection.execute(
                        """
                        INSERT OR IGNORE INTO episode_attempt_current(
                            run_id, episode_number, attempt_cycle
                        )
                        SELECT DISTINCT run_id, episode_number, 0
                        FROM episode_attempts
                        WHERE EXISTS (
                            SELECT 1
                            FROM episode_attempt_cycles
                            WHERE episode_attempt_cycles.run_id = episode_attempts.run_id
                              AND episode_attempt_cycles.attempt_cycle = 0
                        )
                        """
                    )
                    model_call_columns = await (
                        await connection.execute("PRAGMA table_info(model_calls)")
                    ).fetchall()
                    if "operation_id" not in {column[1] for column in model_call_columns}:
                        await connection.execute(
                            "ALTER TABLE model_calls ADD COLUMN operation_id TEXT"
                        )
                    checkpoint_columns = await (
                        await connection.execute("PRAGMA table_info(business_checkpoints)")
                    ).fetchall()
                    if "review_call_id" not in {column[1] for column in checkpoint_columns}:
                        await connection.execute(
                            "ALTER TABLE business_checkpoints ADD COLUMN review_call_id TEXT"
                        )
                    await connection.execute(_SCHEMA_V18_MODEL_CALLS_OPERATION_INDEX_SQL)
                    await connection.execute(
                        "INSERT OR IGNORE INTO pengine_schema(version) VALUES (19)"
                    )
                except BaseException:
                    await connection.rollback()
                    raise
                else:
                    await connection.commit()
                    schema_version = 19
            if schema_version == 19:
                await connection.execute("BEGIN IMMEDIATE")
                try:
                    await connection.execute("DROP TABLE IF EXISTS run_progress_v20")
                    await connection.execute(_SCHEMA_V20_RUN_PROGRESS_SQL)
                    await connection.execute(
                        """
                        INSERT INTO run_progress_v20(
                            run_id, current_stage, execution_state, elapsed_seconds,
                            active_started_at, timeout_stage, timeout_count, updated_at,
                            current_episode, recovery_reason, content_repair_count, pause_message
                        )
                        SELECT
                            run_id, current_stage, execution_state, elapsed_seconds,
                            active_started_at, timeout_stage, timeout_count, updated_at,
                            current_episode, recovery_reason, content_repair_count, pause_message
                        FROM run_progress
                        """
                    )
                    await connection.execute("DROP TABLE run_progress")
                    await connection.execute("ALTER TABLE run_progress_v20 RENAME TO run_progress")
                    model_call_columns = await (
                        await connection.execute("PRAGMA table_info(model_calls)")
                    ).fetchall()
                    if "response_model_ids_json" not in {
                        column[1] for column in model_call_columns
                    }:
                        await connection.execute(
                            "ALTER TABLE model_calls ADD COLUMN response_model_ids_json TEXT"
                        )
                    await connection.execute(
                        "INSERT OR IGNORE INTO pengine_schema(version) VALUES (20)"
                    )
                except BaseException:
                    await connection.rollback()
                    raise
                else:
                    await connection.commit()
                    schema_version = 20
            if schema_version == 20:
                await connection.execute("BEGIN IMMEDIATE")
                try:
                    model_call_columns = await (
                        await connection.execute("PRAGMA table_info(model_calls)")
                    ).fetchall()
                    existing_columns = {column[1] for column in model_call_columns}
                    additions = (
                        ("l3_sha256", "ALTER TABLE model_calls ADD COLUMN l3_sha256 TEXT"),
                        (
                            "l3_char_count",
                            "ALTER TABLE model_calls ADD COLUMN l3_char_count INTEGER",
                        ),
                        (
                            "l3_mount_path",
                            "ALTER TABLE model_calls ADD COLUMN l3_mount_path TEXT",
                        ),
                        (
                            "l3_full_text_mounted",
                            "ALTER TABLE model_calls ADD COLUMN l3_full_text_mounted "
                            "INTEGER NOT NULL DEFAULT 0",
                        ),
                    )
                    for column_name, statement in additions:
                        if column_name not in existing_columns:
                            await connection.execute(statement)
                    await connection.execute(
                        "INSERT OR IGNORE INTO pengine_schema(version) VALUES (21)"
                    )
                except BaseException:
                    await connection.rollback()
                    raise
                else:
                    await connection.commit()
                    schema_version = 21
            if schema_version == 21:
                await connection.execute("BEGIN IMMEDIATE")
                try:
                    delivery_columns = await (
                        await connection.execute("PRAGMA table_info(deliveries)")
                    ).fetchall()
                    existing_columns = {column[1] for column in delivery_columns}
                    additions = (
                        (
                            "presentation_manifest_json",
                            "ALTER TABLE deliveries ADD COLUMN presentation_manifest_json TEXT",
                        ),
                        (
                            "presentation_manifest_sha256",
                            "ALTER TABLE deliveries ADD COLUMN presentation_manifest_sha256 TEXT",
                        ),
                    )
                    for column_name, statement in additions:
                        if column_name not in existing_columns:
                            await connection.execute(statement)
                    await connection.execute(
                        "INSERT OR IGNORE INTO pengine_schema(version) VALUES (22)"
                    )
                except BaseException:
                    await connection.rollback()
                    raise
                else:
                    await connection.commit()
                    schema_version = 22
            if schema_version == 22:
                await connection.execute("BEGIN IMMEDIATE")
                try:
                    await connection.executescript(_SCHEMA_V23_QUALITY_GATE_REPAIR_SQL)
                    await connection.execute(
                        "INSERT OR IGNORE INTO pengine_schema(version) VALUES (23)"
                    )
                except BaseException:
                    await connection.rollback()
                    raise
                else:
                    await connection.commit()
                    schema_version = 23
            if schema_version == 23:
                await connection.execute("BEGIN IMMEDIATE")
                try:
                    await connection.executescript(_SCHEMA_V24_SERIES_BIBLE_PROJECTION_REPAIR_SQL)
                except BaseException:
                    await connection.rollback()
                    raise
                else:
                    await connection.commit()
                    schema_version = 24
            if schema_version == 24:
                await connection.execute("BEGIN IMMEDIATE")
                try:
                    candidate_columns = await (
                        await connection.execute("PRAGMA table_info(episode_candidates)")
                    ).fetchall()
                    if "generation_window_id" not in {column[1] for column in candidate_columns}:
                        await connection.execute(
                            "ALTER TABLE episode_candidates ADD COLUMN generation_window_id TEXT"
                        )
                    await connection.executescript(_SCHEMA_V25_EPISODE_GENERATION_WINDOWS_SQL)
                except BaseException:
                    await connection.rollback()
                    raise
                else:
                    await connection.commit()
                    schema_version = 25
            if schema_version == 25:
                await connection.execute("BEGIN IMMEDIATE")
                try:
                    model_call_columns = await (
                        await connection.execute("PRAGMA table_info(model_calls)")
                    ).fetchall()
                    existing_model_call_columns = {column[1] for column in model_call_columns}
                    if "context_bundle_sha256" not in existing_model_call_columns:
                        await connection.execute(
                            "ALTER TABLE model_calls ADD COLUMN context_bundle_sha256 TEXT"
                        )
                    if "context_manifest_json" not in existing_model_call_columns:
                        await connection.execute(
                            "ALTER TABLE model_calls ADD COLUMN context_manifest_json TEXT"
                        )
                    await connection.executescript(_SCHEMA_V26_SCRIPT_CONTEXT_AUDIT_SQL)
                except BaseException:
                    await connection.rollback()
                    raise
                else:
                    await connection.commit()
                    schema_version = 26
            if schema_version == 26:
                await connection.execute("BEGIN IMMEDIATE")
                try:
                    await connection.executescript(_SCHEMA_V27_GROUPED_OUTLINE_SQL)
                except BaseException:
                    await connection.rollback()
                    raise
                else:
                    await connection.commit()
                    schema_version = 27
            if schema_version == 27:
                await connection.execute("BEGIN IMMEDIATE")
                try:
                    window_columns = await (
                        await connection.execute("PRAGMA table_info(episode_generation_windows)")
                    ).fetchall()
                    existing_window_columns = {column[1] for column in window_columns}
                    additions = {
                        "screenplay_text": "TEXT",
                        "screenplay_nonce": "TEXT",
                        "screenplay_manifest_json": "TEXT",
                        "content_call_id": "TEXT",
                        "sidecar_call_id": "TEXT",
                        "context_bundle_sha256": "TEXT",
                    }
                    for name, sql_type in additions.items():
                        if name not in existing_window_columns:
                            await connection.execute(
                                f"ALTER TABLE episode_generation_windows "
                                f"ADD COLUMN {name} {sql_type}"
                            )
                    await connection.executescript(_SCHEMA_V28_SCRIPT_TEXT_SIDECAR_SQL)
                except BaseException:
                    await connection.rollback()
                    raise
                else:
                    await connection.commit()
                    schema_version = 28
            if schema_version == 28:
                await connection.execute("BEGIN IMMEDIATE")
                try:
                    await connection.executescript(_SCHEMA_V29_OUTLINE_MARKDOWN_SIDECAR_SQL)
                except BaseException:
                    await connection.rollback()
                    raise
                else:
                    await connection.commit()
                    schema_version = 29
            if schema_version == 29:
                await connection.execute("BEGIN IMMEDIATE")
                try:
                    await connection.executescript(_SCHEMA_V30_OUTLINE_MARKDOWN_FAILURES_SQL)
                except BaseException:
                    await connection.rollback()
                    raise
                else:
                    await connection.commit()
                    schema_version = 30
            if schema_version == 30:
                await connection.execute("BEGIN IMMEDIATE")
                try:
                    await connection.executescript(_SCHEMA_V31_OUTLINE_GROUP_REJECTIONS_SQL)
                except BaseException:
                    await connection.rollback()
                    raise
                else:
                    await connection.commit()
                    schema_version = 31
            if schema_version == 31:
                await connection.execute("BEGIN IMMEDIATE")
                try:
                    await connection.executescript(_SCHEMA_V32_PROJECTION_REPAIR_ATTEMPTS_SQL)
                except BaseException:
                    await connection.rollback()
                    raise
                else:
                    await connection.commit()
                    schema_version = 32

    async def setup(self) -> None:
        await self.initialize()

    async def get_outline_season_map(self, run_id: UUID) -> dict[str, Any] | None:
        async with self._connection() as connection:
            row = await self._fetchone(
                connection,
                """
                SELECT content_json, content_sha256, call_id
                FROM outline_season_maps
                WHERE run_id = ?
                """,
                (str(run_id),),
            )
        if row is None:
            return None
        return {
            "content": json.loads(row["content_json"]),
            "content_sha256": row["content_sha256"],
            "call_id": row["call_id"],
        }

    async def commit_outline_season_map(
        self,
        run_id: UUID,
        payload: Mapping[str, Any],
        *,
        call_id: str,
    ) -> None:
        content_json = _json(payload)
        content_sha256 = _text_hash(content_json)
        timestamp = _timestamp(_utc_now())
        async with self._transaction() as connection:
            existing = await self._fetchone(
                connection,
                """
                SELECT content_sha256, call_id
                FROM outline_season_maps
                WHERE run_id = ?
                """,
                (str(run_id),),
            )
            if existing is not None:
                if existing["content_sha256"] != content_sha256 or existing["call_id"] != call_id:
                    raise DomainError(
                        "outline_season_map_conflict",
                        "The committed outline season map cannot be replaced.",
                        409,
                    )
                return
            await connection.execute(
                """
                INSERT INTO outline_season_maps(
                    run_id, content_json, content_sha256, call_id, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    str(run_id),
                    content_json,
                    content_sha256,
                    call_id,
                    timestamp,
                    timestamp,
                ),
            )

    async def get_committed_outline_groups(self, run_id: UUID) -> list[dict[str, Any]]:
        async with self._connection() as connection:
            rows = await (
                await connection.execute(
                    """
                SELECT group_id, position, start_episode, end_episode, content_json,
                       content_sha256, call_id, attempt_count, outline_markdown,
                       outline_markdown_sha256, body_call_id, sidecar_call_id
                FROM outline_generation_groups
                WHERE run_id = ? AND status = 'committed'
                ORDER BY position
                """,
                    (str(run_id),),
                )
            ).fetchall()
        return [
            {
                "group_id": row["group_id"],
                "position": int(row["position"]),
                "start_episode": int(row["start_episode"]),
                "end_episode": int(row["end_episode"]),
                "content": json.loads(row["content_json"]),
                "content_sha256": row["content_sha256"],
                "call_id": row["call_id"],
                "attempt_count": int(row["attempt_count"]),
                "outline_markdown": row["outline_markdown"],
                "outline_markdown_sha256": row["outline_markdown_sha256"],
                "body_call_id": row["body_call_id"],
                "sidecar_call_id": row["sidecar_call_id"],
            }
            for row in rows
        ]

    async def begin_outline_group(
        self,
        run_id: UUID,
        *,
        group_id: str,
        position: int,
        start_episode: int,
        end_episode: int,
        operation_id: str,
    ) -> int:
        timestamp = _timestamp(_utc_now())
        async with self._transaction() as connection:
            existing = await self._fetchone(
                connection,
                """
                SELECT position, start_episode, end_episode, status, attempt_count
                FROM outline_generation_groups
                WHERE run_id = ? AND group_id = ?
                """,
                (str(run_id), group_id),
            )
            if existing is None:
                attempt_count = 1
                await connection.execute(
                    """
                    INSERT INTO outline_generation_groups(
                        run_id, group_id, position, start_episode, end_episode,
                        operation_id, status, attempt_count, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, 'generating', ?, ?, ?)
                    """,
                    (
                        str(run_id),
                        group_id,
                        position,
                        start_episode,
                        end_episode,
                        operation_id,
                        attempt_count,
                        timestamp,
                        timestamp,
                    ),
                )
                return attempt_count
            if (
                int(existing["position"]) != position
                or int(existing["start_episode"]) != start_episode
                or int(existing["end_episode"]) != end_episode
            ):
                raise DomainError(
                    "outline_group_binding_conflict",
                    "The outline group no longer matches the committed season map.",
                    409,
                )
            if existing["status"] == "committed":
                raise DomainError(
                    "outline_group_already_committed",
                    "The outline group is already committed.",
                    409,
                )
            if existing["status"] == "body_generated":
                await connection.execute(
                    """
                    UPDATE outline_generation_groups
                    SET operation_id = ?, updated_at = ?
                    WHERE run_id = ? AND group_id = ? AND status = 'body_generated'
                    """,
                    (operation_id, timestamp, str(run_id), group_id),
                )
                return int(existing["attempt_count"])
            attempt_count = int(existing["attempt_count"]) + 1
            await connection.execute(
                """
                UPDATE outline_generation_groups
                SET operation_id = ?, call_id = NULL, status = 'generating',
                    attempt_count = ?, outline_markdown = NULL,
                    outline_markdown_sha256 = NULL, body_call_id = NULL,
                    sidecar_call_id = NULL, content_json = NULL,
                    content_sha256 = NULL, updated_at = ?
                WHERE run_id = ? AND group_id = ?
                """,
                (operation_id, attempt_count, timestamp, str(run_id), group_id),
            )
            return attempt_count

    async def save_outline_group_body(
        self,
        run_id: UUID,
        *,
        group_id: str,
        operation_id: str,
        outline_markdown: str,
        outline_markdown_sha256: str,
        body_call_id: str,
    ) -> None:
        if not outline_markdown.strip() or _text_hash(outline_markdown) != outline_markdown_sha256:
            raise DomainError(
                "invalid_outline_markdown",
                "The outline Markdown artifact is empty or has an invalid hash.",
                409,
            )
        timestamp = _timestamp(_utc_now())
        async with self._transaction() as connection:
            cursor = await connection.execute(
                """
                UPDATE outline_generation_groups
                SET status = 'body_generated', outline_markdown = ?,
                    outline_markdown_sha256 = ?, body_call_id = ?, updated_at = ?
                WHERE run_id = ? AND group_id = ? AND operation_id = ?
                  AND status = 'generating' AND outline_markdown IS NULL
                """,
                (
                    outline_markdown,
                    outline_markdown_sha256,
                    body_call_id,
                    timestamp,
                    str(run_id),
                    group_id,
                    operation_id,
                ),
            )
            if cursor.rowcount != 1:
                existing = await self._fetchone(
                    connection,
                    """
                    SELECT outline_markdown, outline_markdown_sha256, body_call_id
                    FROM outline_generation_groups
                    WHERE run_id = ? AND group_id = ? AND operation_id = ?
                      AND status = 'body_generated'
                    """,
                    (str(run_id), group_id, operation_id),
                )
                expected = (outline_markdown, outline_markdown_sha256, body_call_id)
                if existing is None or tuple(existing) != expected:
                    raise DomainError(
                        "outline_group_body_conflict",
                        "The outline Markdown no longer matches its active group.",
                        409,
                    )

    async def get_outline_group_body(
        self,
        run_id: UUID,
        *,
        group_id: str,
    ) -> dict[str, Any] | None:
        async with self._connection() as connection:
            row = await self._fetchone(
                connection,
                """
                SELECT outline_markdown, outline_markdown_sha256, body_call_id, operation_id
                FROM outline_generation_groups
                WHERE run_id = ? AND group_id = ?
                  AND status IN ('body_generated', 'committed')
                  AND outline_markdown IS NOT NULL
                """,
                (str(run_id), group_id),
            )
        if row is None:
            return None
        return dict(row)

    async def replace_outline_group_body(
        self,
        run_id: UUID,
        *,
        group_id: str,
        operation_id: str,
        expected_outline_markdown_sha256: str,
        outline_markdown: str,
        outline_markdown_sha256: str,
        body_call_id: str,
    ) -> None:
        if not outline_markdown.strip() or _text_hash(outline_markdown) != outline_markdown_sha256:
            raise DomainError(
                "invalid_outline_markdown",
                "The outline Markdown artifact is empty or has an invalid hash.",
                409,
            )
        timestamp = _timestamp(_utc_now())
        async with self._transaction() as connection:
            cursor = await connection.execute(
                """
                UPDATE outline_generation_groups
                SET outline_markdown = ?, outline_markdown_sha256 = ?, body_call_id = ?,
                    sidecar_call_id = NULL, content_json = NULL, content_sha256 = NULL,
                    updated_at = ?
                WHERE run_id = ? AND group_id = ? AND operation_id = ?
                  AND status = 'body_generated' AND outline_markdown_sha256 = ?
                """,
                (
                    outline_markdown,
                    outline_markdown_sha256,
                    body_call_id,
                    timestamp,
                    str(run_id),
                    group_id,
                    operation_id,
                    expected_outline_markdown_sha256,
                ),
            )
            if cursor.rowcount == 1:
                return
            existing = await self._fetchone(
                connection,
                """
                SELECT outline_markdown, outline_markdown_sha256, body_call_id
                FROM outline_generation_groups
                WHERE run_id = ? AND group_id = ? AND operation_id = ?
                  AND status = 'body_generated'
                """,
                (str(run_id), group_id, operation_id),
            )
            expected = (outline_markdown, outline_markdown_sha256, body_call_id)
            if existing is None or tuple(existing) != expected:
                raise DomainError(
                    "outline_group_body_conflict",
                    "The outline Markdown no longer matches its active group.",
                    409,
                )

    async def complete_outline_group(
        self,
        run_id: UUID,
        *,
        group_id: str,
        operation_id: str,
        payload: Mapping[str, Any],
        sidecar_call_id: str,
    ) -> None:
        content_json = _json(payload)
        content_sha256 = _text_hash(content_json)
        timestamp = _timestamp(_utc_now())
        async with self._transaction() as connection:
            cursor = await connection.execute(
                """
                UPDATE outline_generation_groups
                SET call_id = body_call_id, sidecar_call_id = ?, status = 'committed',
                    content_json = ?, content_sha256 = ?, updated_at = ?
                WHERE run_id = ? AND group_id = ? AND operation_id = ?
                  AND status = 'body_generated'
                """,
                (
                    sidecar_call_id,
                    content_json,
                    content_sha256,
                    timestamp,
                    str(run_id),
                    group_id,
                    operation_id,
                ),
            )
            if cursor.rowcount != 1:
                raise DomainError(
                    "outline_group_body_not_generated",
                    "The outline group Markdown artifact is not active.",
                    409,
                )

    async def fail_outline_group(
        self,
        run_id: UUID,
        *,
        group_id: str,
        operation_id: str,
    ) -> None:
        async with self._transaction() as connection:
            await connection.execute(
                """
                UPDATE outline_generation_groups
                SET status = 'failed', updated_at = ?
                WHERE run_id = ? AND group_id = ? AND operation_id = ?
                  AND status = 'generating'
                """,
                (_timestamp(_utc_now()), str(run_id), group_id, operation_id),
            )

    async def get_outline_group_rejection(
        self,
        run_id: UUID,
        *,
        group_id: str,
    ) -> dict[str, Any] | None:
        """Return the latest persisted semantic rejection for one outline group."""
        async with self._connection() as connection:
            row = await self._fetchone(
                connection,
                """
                SELECT group_id, evidence, repair_rounds, rejected_at
                FROM outline_group_rejections
                WHERE run_id = ? AND group_id = ?
                ORDER BY rejected_at DESC, id DESC
                LIMIT 1
                """,
                (str(run_id), group_id),
            )
        if row is None:
            return None
        return {
            "group_id": row["group_id"],
            "evidence": row["evidence"],
            "repair_rounds": int(row["repair_rounds"]),
            "rejected_at": row["rejected_at"],
        }

    async def record_outline_markdown_failure(
        self,
        run_id: UUID,
        *,
        group_id: str,
        operation_id: str,
        attempt_index: int,
        raw_text: str,
        normalized_text: str,
        parse_error: str,
        now: datetime | None = None,
    ) -> None:
        """Persist immutable raw-model-text evidence when outline parsing exhausts retries."""
        timestamp = _timestamp(now or _utc_now())
        async with self._transaction() as connection:
            await connection.execute(
                """
                INSERT INTO outline_markdown_failures(
                    run_id, group_id, operation_id, attempt_index,
                    raw_text, raw_text_sha256,
                    normalized_text, normalized_text_sha256,
                    parse_error, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(run_id),
                    group_id,
                    operation_id,
                    attempt_index,
                    raw_text,
                    hashlib.sha256(raw_text.encode()).hexdigest(),
                    normalized_text,
                    hashlib.sha256(normalized_text.encode()).hexdigest(),
                    parse_error,
                    timestamp,
                ),
            )

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

    async def requeue_run_job(
        self,
        run_id: UUID,
        *,
        now: datetime | None = None,
    ) -> None:
        """Requeue the run's job immediately so the worker re-processes it.

        Used after a suffix rewrite or an authorized repair so the run resumes
        deterministically instead of waiting for lease expiry (RPR-A5/A9).
        """
        timestamp = _timestamp(now or _utc_now())
        async with self._transaction() as connection:
            run = await self._fetchone(
                connection,
                "SELECT id, creation_id FROM runs WHERE id = ?",
                (str(run_id),),
            )
            if run is None:
                raise DomainError("run_not_found", "Workflow run not found.", 404)
            await connection.execute(
                """
                UPDATE run_progress
                SET execution_state = 'running',
                    recovery_reason = 'none',
                    content_repair_count = NULL,
                    pause_message = NULL,
                    updated_at = ?
                WHERE run_id = ?
                """,
                (timestamp, str(run_id)),
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
                (timestamp, timestamp, str(run_id)),
            )
            await connection.execute(
                "UPDATE creations SET updated_at = ? WHERE id = ?",
                (timestamp, str(run["creation_id"])),
            )

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
            attempt_cycle = await self._ensure_episode_attempt_current(
                connection,
                run_id,
                episode_number,
                timestamp=timestamp,
            )
            cursor = await connection.execute(
                """
                SELECT COALESCE(MAX(attempt_number), 0)
                FROM episode_attempts
                WHERE run_id = ? AND episode_number = ? AND attempt_cycle = ?
                """,
                (str(run_id), episode_number, attempt_cycle),
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
                INSERT INTO episode_attempts(
                    run_id, episode_number, attempt_cycle, attempt_number, recorded_at
                )
                VALUES (?, ?, ?, ?, ?)
                """,
                (str(run_id), episode_number, attempt_cycle, attempt_number, timestamp),
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
                        repair_constraints=episode_lock.series_state.repair_constraints,
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
            attempt_cycle = await self._ensure_episode_attempt_current(
                connection,
                run_id,
                episode_number,
                timestamp=timestamp,
            )
            attempt = await self._fetchone(
                connection,
                """
                SELECT 1 FROM episode_attempts
                WHERE run_id = ? AND episode_number = ? AND attempt_cycle = ?
                LIMIT 1
                """,
                (str(run_id), episode_number, attempt_cycle),
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
        outline_group_id: str | None = None,
        now: datetime | None = None,
    ) -> None:
        if stage not in {
            InternalStage.GENERATING_STORY_OUTLINE,
            InternalStage.GENERATING_CHARACTER_RELATIONSHIPS,
            InternalStage.GENERATING_EPISODE_OUTLINE,
            InternalStage.GENERATING_EPISODE_SCRIPTS,
        }:
            raise ValueError("Only story text or episode generation can pause for content review")
        allowed_repair_rounds = set(range(2, 5)) if stage in _EXTENDED_REPAIR_STAGES else {2}
        if repair_rounds not in allowed_repair_rounds or not evidence.strip():
            raise ValueError(
                "Story content rejection requires two to four repair rounds and evidence"
                if stage in _EXTENDED_REPAIR_STAGES
                else "Episode content rejection requires two repair rounds and evidence"
            )
        if (stage is InternalStage.GENERATING_EPISODE_SCRIPTS) != (episode_number is not None):
            raise ValueError("Episode number is required only for episode script rejection")
        if outline_group_id is not None and stage is not InternalStage.GENERATING_EPISODE_OUTLINE:
            raise ValueError("Outline group id is required only for outline group rejection")
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
            if outline_group_id is not None:
                await connection.execute(
                    """
                    INSERT INTO outline_group_rejections(
                        run_id, group_id, evidence, repair_rounds, rejected_at
                    ) VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        str(run_id),
                        outline_group_id,
                        evidence,
                        repair_rounds,
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
            attempt_cycle = await self._ensure_episode_attempt_current(
                connection,
                run_id,
                episode_number,
                timestamp=timestamp,
            )
            attempt = await self._fetchone(
                connection,
                """
                SELECT COUNT(*) AS attempt_count
                FROM episode_attempts
                WHERE run_id = ? AND episode_number = ? AND attempt_cycle = ?
                """,
                (str(run_id), episode_number, attempt_cycle),
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

    async def pause_context_budget(
        self,
        run_id: UUID,
        *,
        stage: InternalStage,
        safe_message: str,
        episode_number: int | None = None,
        now: datetime | None = None,
    ) -> None:
        """Pause a running workflow after a fail-closed context preflight block."""
        if not safe_message.strip():
            raise ValueError("Context-budget pauses require a safe message")
        if episode_number is not None and episode_number < 1:
            raise ValueError("Context-budget pauses require a valid episode number")
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
                UPDATE run_progress
                SET current_stage = ?,
                    current_episode = ?,
                    execution_state = 'paused',
                    elapsed_seconds = ?,
                    active_started_at = NULL,
                    timeout_stage = ?,
                    timeout_count = 0,
                    recovery_reason = 'context_budget',
                    content_repair_count = NULL,
                    pause_message = ?,
                    updated_at = ?
                WHERE run_id = ?
                """,
                (
                    stage.value,
                    episode_number,
                    self._elapsed_seconds(progress, current),
                    _USER_STAGE_BY_INTERNAL[stage].value,
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

    async def pause_relay_identity_mismatch(
        self,
        run_id: UUID,
        *,
        stage: InternalStage,
        safe_message: str,
        episode_number: int | None = None,
        now: datetime | None = None,
    ) -> None:
        """Discard an unverified response and pause until an operator checks the relay."""
        if not safe_message.strip():
            raise ValueError("Relay-identity pauses require a safe message")
        if episode_number is not None and episode_number < 1:
            raise ValueError("Relay-identity pauses require a valid episode number")
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
                UPDATE run_progress
                SET current_stage = ?,
                    current_episode = ?,
                    execution_state = 'paused',
                    elapsed_seconds = ?,
                    active_started_at = NULL,
                    timeout_stage = ?,
                    timeout_count = 0,
                    recovery_reason = 'relay_identity_mismatch',
                    content_repair_count = NULL,
                    pause_message = ?,
                    updated_at = ?
                WHERE run_id = ?
                """,
                (
                    stage.value,
                    episode_number,
                    self._elapsed_seconds(progress, current),
                    _USER_STAGE_BY_INTERNAL[stage].value,
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
            attempt_cycle = await self._current_episode_attempt_cycle(
                connection,
                run_id,
                episode_number,
            )
            attempt = await self._fetchone(
                connection,
                """
                SELECT COUNT(*) AS attempt_count
                FROM episode_attempts
                WHERE run_id = ? AND episode_number = ? AND attempt_cycle = ?
                """,
                (str(run_id), episode_number, attempt_cycle),
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
                  AND attempt_cycle = COALESCE(
                      (
                          SELECT attempt_state.attempt_cycle
                          FROM episode_attempt_current AS attempt_state
                          WHERE attempt_state.run_id = episode_attempts.run_id
                            AND attempt_state.episode_number = episode_attempts.episode_number
                      ),
                      0
                  )
                GROUP BY episode_number
                """,
                (str(run_id),),
            )
            rows = await cursor.fetchall()
        return {int(row["episode_number"]): int(row["attempt_count"]) for row in rows}

    async def get_episode_attempt_cycles(self, run_id: UUID) -> dict[int, int]:
        async with self._connection() as connection:
            cursor = await connection.execute(
                """
                SELECT episode_number, attempt_cycle
                FROM episode_attempt_current
                WHERE run_id = ?
                ORDER BY episode_number
                """,
                (str(run_id),),
            )
            rows = await cursor.fetchall()
        return {int(row["episode_number"]): int(row["attempt_cycle"]) for row in rows}

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
                    repair_constraints=draft.series_state.repair_constraints,
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
            active_batch = await self._fetch_script_batch_lineage(connection, run_id)
            applied_repair = (
                await self._fetchone(
                    connection,
                    """
                    SELECT repaired_batch_id
                    FROM quality_gate_repairs
                    WHERE run_id = ? AND status = 'applied'
                    ORDER BY rejection_attempt DESC
                    LIMIT 1
                    """,
                    (str(run_id),),
                )
                if active_batch is not None
                else None
            )
            if (
                active_batch is None
                or applied_repair is None
                or applied_repair["repaired_batch_id"] != active_batch["batch_id"]
                or any(
                    draft.series_state_sha256 is None or draft.contract_sha256 != contract_hash
                    for draft in drafts
                )
            ):
                raise DomainError(
                    "episode_lock_invalid",
                    "Every episode lock must match the approved story contract before review.",
                    409,
                ) from exc
            episode_hashes = [
                {
                    "episode_number": draft.episode_number,
                    "content_sha256": draft.content_sha256,
                    "series_state_sha256": draft.series_state_sha256,
                }
                for draft in drafts
            ]
            return {
                "content": content,
                "contract_sha256": contract_hash,
                "episode_hashes": episode_hashes,
                "series_state_sha256": drafts[-1].series_state_sha256,
            }
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
            grouped_outline_resume = False
            if stage is InternalStage.GENERATING_EPISODE_OUTLINE:
                season_map = await self._fetchone(
                    connection,
                    "SELECT 1 FROM outline_season_maps WHERE run_id = ?",
                    (str(run_id),),
                )
                grouped_outline_resume = season_map is not None and current_count > 0
            if current_count >= MAX_STAGE_ATTEMPTS and not grouped_outline_resume:
                raise DomainError(
                    "attempts_exhausted",
                    "The stage attempt limit has been exhausted.",
                    409,
                )

            attempt_number = current_count if grouped_outline_resume else current_count + 1
            if not grouped_outline_resume:
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
        review_call_id: str | None = None,
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
                SELECT payload_json, payload_sha256, review_call_id
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
                if review_call_id is not None and existing["review_call_id"] != review_call_id:
                    raise DomainError(
                        "checkpoint_conflict",
                        "An approved checkpoint cannot replace its model-call provenance.",
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
                    if not isinstance(review_call_id, str) or not review_call_id:
                        raise DomainError(
                            "invalid_story_contract",
                            "A lockable episode outline requires physical review provenance.",
                            409,
                        )
                    producing_review = await self._fetchone(
                        connection,
                        """
                        SELECT 1
                        FROM model_calls
                        WHERE call_id = ? AND run_id = ? AND role = 'review'
                              AND stage = ? AND episode_number IS NULL
                              AND operation_id IS NOT NULL AND status = 'succeeded'
                        """,
                        (
                            review_call_id,
                            str(run_id),
                            InternalStage.GENERATING_EPISODE_OUTLINE.value,
                        ),
                    )
                    if producing_review is None:
                        raise DomainError(
                            "invalid_story_contract",
                            "The episode outline review provenance is not a successful "
                            "physical review call.",
                            409,
                        )
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
                    run_id, stage, payload_json, payload_sha256, review_call_id, approved_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    str(run_id),
                    stage.value,
                    payload_json,
                    payload_hash,
                    review_call_id,
                    timestamp,
                ),
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
        review_call_id: str | None = None,
        now: datetime | None = None,
    ) -> Any:
        return await self.approve_business_checkpoint(
            run_id,
            stage,
            payload,
            review_call_id=review_call_id,
            now=now,
        )

    async def get_checkpoint_review_call_id(
        self,
        run_id: UUID,
        stage: InternalStage,
    ) -> str | None:
        """Return hidden physical review provenance without exposing it in payloads."""
        async with self._connection() as connection:
            row = await self._fetchone(
                connection,
                """
                SELECT review_call_id
                FROM business_checkpoints
                WHERE run_id = ? AND stage = ?
                """,
                (str(run_id), stage.value),
            )
            return row["review_call_id"] if row is not None else None

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

    # ------------------------------------------------------------------
    # SeriesBible design-package aggregate
    # ------------------------------------------------------------------

    async def register_series_bible_candidate(
        self,
        creation_id: str,
        run_id: UUID,
        candidate: SeriesBible,
        validation: ValidationEvidence,
        *,
        now: datetime | None = None,
    ) -> SeriesBible:
        """Persist one immutable design candidate with its deterministic evidence."""
        timestamp = _timestamp(now or _utc_now())
        candidate = candidate.model_copy(
            update={
                "status": "validated" if validation.passed else "unvalidated",
                "validation": validation,
            }
        )
        async with self._transaction() as connection:
            existing = await self._fetchone(
                connection,
                "SELECT candidate_id FROM series_bible_candidates WHERE candidate_id = ?",
                (candidate.candidate_id,),
            )
            if existing is not None:
                return self._series_bible_from_row(
                    await self._fetch_series_bible_candidate(
                        connection,
                        run_id,
                        candidate.candidate_id,
                    )
                )
            await connection.execute(
                """
                INSERT INTO series_bible_candidates(
                    candidate_id, run_id, creation_id, version, design_epoch,
                    content_hash, status, l0_variant, genre, lineage_json,
                    content_json, validation_json, global_review_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    candidate.candidate_id,
                    str(run_id),
                    creation_id,
                    candidate.version,
                    candidate.design_epoch,
                    candidate.content_hash,
                    candidate.status,
                    candidate.l0_variant,
                    candidate.genre,
                    _json(candidate.lineage),
                    _json(candidate.content),
                    _json(candidate.validation),
                    _json(candidate.global_review) if candidate.global_review is not None else None,
                    timestamp,
                ),
            )
            await connection.execute(
                """
                INSERT OR IGNORE INTO series_bible_lineage(
                    run_id, creation_id, active_design_epoch, rebuild_count, updated_at
                ) VALUES (?, ?, 0, 0, ?)
                """,
                (str(run_id), creation_id, timestamp),
            )
        return candidate

    async def record_series_bible_review(
        self,
        run_id: UUID,
        candidate_id: str,
        review: GlobalDesignReview,
        *,
        now: datetime | None = None,
    ) -> SeriesBible:
        """Bind one global design review to exactly one candidate."""
        async with self._transaction() as connection:
            candidate = await self._fetch_series_bible_candidate(
                connection,
                run_id,
                candidate_id,
            )
            if candidate is None:
                raise DomainError(
                    "series_bible_candidate_not_found",
                    "The design candidate is not registered for this run.",
                    409,
                )
            if (
                review.candidate_id != candidate_id
                or review.candidate_hash != candidate["content_hash"]
            ):
                raise DomainError(
                    "series_bible_review_mismatch",
                    "The review does not bind the candidate it records.",
                    409,
                )
            await connection.execute(
                """
                UPDATE series_bible_candidates
                SET global_review_json = ?,
                    status = CASE
                        WHEN validation_json IS NOT NULL
                             AND json_extract(validation_json, '$.passed') = 1
                        THEN 'validated'
                        ELSE 'unvalidated'
                    END
                WHERE candidate_id = ?
                """,
                (_json(review), candidate_id),
            )
            stored = await self._fetch_series_bible_candidate(connection, run_id, candidate_id)
        return self._series_bible_from_row(stored)

    async def promote_series_bible(
        self,
        run_id: UUID,
        candidate_id: str,
        *,
        now: datetime | None = None,
    ) -> SeriesBible:
        """Transactional/CAS promotion of exactly one active design candidate.

        Only the active lineage may move the active pointer. The promoted candidate
        must carry passing deterministic validation and a passing bound global
        design review whose candidate id and content hash match this candidate. A
        late candidate whose design epoch is not the next active epoch is retained
        but cannot promote.
        """
        timestamp = _timestamp(now or _utc_now())
        async with self._transaction() as connection:
            candidate = await self._fetch_series_bible_candidate(
                connection,
                run_id,
                candidate_id,
            )
            if candidate is None:
                raise DomainError(
                    "series_bible_candidate_not_found",
                    "The design candidate is not registered for this run.",
                    409,
                )
            if candidate["status"] == "active":
                return self._series_bible_from_row(candidate)
            validation = (
                json.loads(candidate["validation_json"]) if candidate["validation_json"] else None
            )
            if validation is None or not validation.get("passed"):
                raise DomainError(
                    "series_bible_unvalidated",
                    "A design candidate requires passing deterministic validation "
                    "before promotion.",
                    409,
                )
            review = (
                json.loads(candidate["global_review_json"])
                if candidate["global_review_json"]
                else None
            )
            if review is None or not review.get("passed"):
                raise DomainError(
                    "series_bible_review_required",
                    "A design candidate requires a passing bound global design review.",
                    409,
                )
            if (
                review["candidate_id"] != candidate_id
                or review["candidate_hash"] != candidate["content_hash"]
            ):
                raise DomainError(
                    "series_bible_review_mismatch",
                    "Another candidate's review cannot approve this candidate.",
                    409,
                )
            lineage = await self._fetch_series_bible_lineage(connection, run_id)
            if lineage is None:
                raise DomainError(
                    "series_bible_lineage_missing",
                    "The design lineage for this run is missing.",
                    409,
                )
            active_epoch = lineage["active_design_epoch"]
            candidate_epoch = candidate["design_epoch"]
            if (
                candidate_epoch != active_epoch + 1
                and candidate["content_hash"] != lineage["active_content_hash"]
            ):
                raise DomainError(
                    "series_bible_stale_promotion",
                    "Only the active lineage may move the active pointer.",
                    409,
                )
            cursor = await connection.execute(
                """
                UPDATE series_bible_lineage
                SET active_candidate_id = ?, active_design_epoch = ?,
                    active_content_hash = ?, updated_at = ?
                WHERE run_id = ? AND active_design_epoch = ?
                """,
                (
                    candidate_id,
                    candidate_epoch,
                    candidate["content_hash"],
                    timestamp,
                    str(run_id),
                    active_epoch,
                ),
            )
            if cursor.rowcount != 1:
                raise DomainError(
                    "series_bible_stale_promotion",
                    "A newer design epoch became active concurrently.",
                    409,
                )
            if lineage["active_candidate_id"] is not None:
                await connection.execute(
                    """
                    UPDATE series_bible_candidates
                    SET status = 'superseded', superseded_at = ?
                    WHERE candidate_id = ? AND status = 'active'
                    """,
                    (timestamp, lineage["active_candidate_id"]),
                )
            await connection.execute(
                """
                UPDATE series_bible_candidates
                SET status = 'active', activated_at = ?
                WHERE candidate_id = ?
                """,
                (timestamp, candidate_id),
            )
            stored = await self._fetch_series_bible_candidate(connection, run_id, candidate_id)
        return self._series_bible_from_row(stored)

    async def mark_series_bible_stale(
        self,
        run_id: UUID,
        *,
        active_candidate_id: str,
        now: datetime | None = None,
    ) -> None:
        """Retain every older non-active candidate as stale evidence."""
        timestamp = _timestamp(now or _utc_now())
        async with self._transaction() as connection:
            await connection.execute(
                """
                UPDATE series_bible_candidates
                SET status = 'stale', superseded_at = COALESCE(superseded_at, ?)
                WHERE run_id = ? AND candidate_id <> ? AND status NOT IN ('active', 'superseded')
                """,
                (timestamp, str(run_id), active_candidate_id),
            )

    async def rebuild_series_bible(
        self,
        creation_id: str,
        run_id: UUID,
        candidate: SeriesBible,
        validation: ValidationEvidence,
        *,
        now: datetime | None = None,
        authorized: bool = False,
    ) -> SeriesBible:
        """CAS-create one complete new candidate for a confirmed design defect.

        The same run lineage may automatically rebuild the complete design at most
        once (SDP-A6). A second automatic rebuild is rejected with a stable error.
        ``authorized`` allows exactly one further rebuild for an explicit one-cycle
        repair authorization (RPR-A9) while keeping the automatic budget consumed.
        The rebuilt candidate is a complete, fresh bundle (never a partial patch),
        and the superseded candidate remains immutable evidence.
        """
        timestamp = _timestamp(now or _utc_now())
        candidate = candidate.model_copy(
            update={
                "status": "validated" if validation.passed else "unvalidated",
                "validation": validation,
            }
        )
        async with self._transaction() as connection:
            lineage = await self._fetch_series_bible_lineage(connection, run_id)
            if lineage is None:
                raise DomainError(
                    "series_bible_lineage_missing",
                    "The design lineage for this run is missing.",
                    409,
                )
            if int(lineage["rebuild_count"]) >= 1 and not authorized:
                raise DomainError(
                    "series_bible_rebuild_exhausted",
                    "This run lineage may rebuild the design automatically at most once.",
                    409,
                )
            if lineage["active_candidate_id"] is None:
                raise DomainError(
                    "series_bible_no_active",
                    "A complete design rebuild requires an active candidate.",
                    409,
                )
            expected_epoch = lineage["active_design_epoch"] + 1
            if candidate.design_epoch != expected_epoch:
                raise DomainError(
                    "series_bible_stale_rebuild",
                    "The rebuild epoch must follow the active design epoch.",
                    409,
                )
            cursor = await connection.execute(
                """
                UPDATE series_bible_lineage
                SET rebuild_count = 1, updated_at = ?
                WHERE run_id = ? AND rebuild_count = ? AND active_design_epoch = ?
                """,
                (
                    timestamp,
                    str(run_id),
                    1 if authorized else 0,
                    lineage["active_design_epoch"],
                ),
            )
            if cursor.rowcount != 1:
                raise DomainError(
                    "series_bible_rebuild_exhausted",
                    "A concurrent rebuild already consumed the lineage budget.",
                    409,
                )
            cursor = await connection.execute(
                """
                INSERT OR IGNORE INTO series_bible_candidates(
                    candidate_id, run_id, creation_id, version, design_epoch,
                    content_hash, status, l0_variant, genre, lineage_json,
                    content_json, validation_json, global_review_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    candidate.candidate_id,
                    str(run_id),
                    creation_id,
                    candidate.version,
                    candidate.design_epoch,
                    candidate.content_hash,
                    candidate.status,
                    candidate.l0_variant,
                    candidate.genre,
                    _json(candidate.lineage),
                    _json(candidate.content),
                    _json(candidate.validation),
                    _json(candidate.global_review) if candidate.global_review is not None else None,
                    timestamp,
                ),
            )
            if cursor.rowcount == 0:
                # A candidate with this id already exists: a crash between INSERT
                # and promotion of the same one-cycle rebuild resumes the same
                # candidate instead of inserting a duplicate.
                existing = await self._fetch_series_bible_candidate(
                    connection,
                    run_id,
                    candidate.candidate_id,
                )
                if existing is None:
                    raise DomainError(
                        "series_bible_rebuild_candidate_missing",
                        "The rebuild candidate was lost between insert and read.",
                        409,
                    )
                return self._series_bible_from_row(existing)
            if authorized:
                # Bind this rebuild candidate to the granted design_rebuild
                # authorization so the worker resumes the exact same candidate
                # (and not an orphan) if it crashes before promotion.
                auth_row = await self._fetchone(
                    connection,
                    """
                    SELECT authorization_epoch
                    FROM repair_authorizations
                    WHERE run_id = ? AND kind = 'design_rebuild'
                    ORDER BY authorization_epoch DESC LIMIT 1
                    """,
                    (str(run_id),),
                )
                if auth_row is not None:
                    await connection.execute(
                        """
                        UPDATE repair_authorizations
                        SET rebuild_candidate_id = ?
                        WHERE run_id = ? AND authorization_epoch = ?
                          AND rebuild_candidate_id IS NULL
                        """,
                        (
                            candidate.candidate_id,
                            str(run_id),
                            int(auth_row["authorization_epoch"]),
                        ),
                    )
        return candidate

    async def get_run_series_bible(self, run_id: UUID) -> SeriesBibleSummary | None:
        """The durable active design candidate projection for one run."""
        async with self._connection() as connection:
            lineage = await self._fetch_series_bible_lineage(connection, run_id)
            if lineage is None or lineage["active_candidate_id"] is None:
                return None
            candidate = await self._fetch_series_bible_candidate(
                connection,
                run_id,
                lineage["active_candidate_id"],
            )
            if candidate is None:
                return None
        bible = self._series_bible_from_row(candidate)
        return project_series_bible(bible, is_active=True)

    async def get_run_series_bible_candidates(self, run_id: UUID) -> list[SeriesBibleSummary]:
        """Every candidate registered for one run, newest first (immutable evidence)."""
        async with self._connection() as connection:
            lineage = await self._fetch_series_bible_lineage(connection, run_id)
            active_id = lineage["active_candidate_id"] if lineage is not None else None
            cursor = await connection.execute(
                """
                SELECT *
                FROM series_bible_candidates
                WHERE run_id = ?
                ORDER BY design_epoch DESC, created_at DESC
                """,
                (str(run_id),),
            )
            rows = await cursor.fetchall()
        return [
            project_series_bible(
                self._series_bible_from_row(row),
                is_active=(row["candidate_id"] == active_id),
            )
            for row in rows
        ]

    async def get_series_bible_lineage(self, run_id: UUID) -> dict[str, Any] | None:
        async with self._connection() as connection:
            return await self._fetch_series_bible_lineage(connection, run_id)

    async def latest_successful_model_call_id(
        self,
        run_id: UUID,
        *,
        role: Literal["generation", "review"],
        stage: InternalStage,
        episode_number: int | None,
        operation_id: str,
    ) -> str | None:
        """Return the exact operation-scoped successful physical model call."""
        async with self._connection() as connection:
            row = await self._fetchone(
                connection,
                """
                SELECT call_id
                FROM model_calls
                WHERE run_id = ? AND role = ? AND stage = ?
                      AND episode_number IS ? AND operation_id = ?
                      AND status = 'succeeded'
                ORDER BY requested_at DESC, call_id DESC
                LIMIT 1
                """,
                (str(run_id), role, stage.value, episode_number, operation_id),
            )
            return row["call_id"] if row is not None else None

    async def is_successful_model_call(
        self,
        run_id: UUID,
        *,
        call_id: str,
        role: Literal["generation", "review"],
        stage: InternalStage,
        episode_number: int | None,
    ) -> bool:
        """Validate persisted physical provenance without selecting a nearby call."""
        async with self._connection() as connection:
            row = await self._fetchone(
                connection,
                """
                SELECT 1
                FROM model_calls
                WHERE call_id = ? AND run_id = ? AND role = ? AND stage = ?
                      AND episode_number IS ? AND status = 'succeeded'
                """,
                (call_id, str(run_id), role, stage.value, episode_number),
            )
            return row is not None

    async def assert_episode_batch_current(self, run_id: UUID) -> str | None:
        """Return the active design content hash, or ``None`` when no design is active.

        When a design hash/epoch change supersedes the active candidate, every prior
        script batch becomes ineligible for active generation or delivery (SDP-A9).
        The dependent writer Issue completes the full invalidation behavior.
        """
        lineage = await self.get_series_bible_lineage(run_id)
        if lineage is None or lineage["active_content_hash"] is None:
            return None
        return lineage["active_content_hash"]

    # ------------------------------------------------------------------
    # Versioned episode candidates and design-bound script batches
    # ------------------------------------------------------------------

    async def get_script_batch_lineage(self, run_id: UUID) -> ScriptBatchLineage | None:
        async with self._connection() as connection:
            row = await self._fetch_script_batch_lineage(connection, run_id)
        return self._script_batch_lineage_from_row(row) if row is not None else None

    async def get_active_episode_candidates(self, run_id: UUID) -> list[EpisodeCandidate]:
        """The active candidates in episode order (the active pointer lineage)."""
        async with self._connection() as connection:
            batch = await self._fetch_script_batch_lineage(connection, run_id)
            if batch is None or batch["status"] != "active":
                return []
            rows: list[aiosqlite.Row] = []
            for _episode, candidate_id in sorted(
                json.loads(batch["active_pointers_json"]).items(),
                key=lambda item: int(item[0]),
            ):
                row = await self._fetch_episode_candidate(connection, run_id, candidate_id)
                if row is not None:
                    rows.append(row)
        return [self._episode_candidate_from_row(row) for row in rows]

    async def get_episode_candidates(self, run_id: UUID) -> list[EpisodeCandidate]:
        """Every immutable candidate for one run, newest-first evidence."""
        async with self._connection() as connection:
            cursor = await connection.execute(
                """
                SELECT *
                FROM episode_candidates
                WHERE run_id = ?
                ORDER BY batch_epoch DESC, episode_number DESC, version DESC
                """,
                (str(run_id),),
            )
            rows = await cursor.fetchall()
        return [self._episode_candidate_from_row(row) for row in rows]

    async def first_unfinished_episode(self, run_id: UUID) -> int | None:
        """The next episode to write in the active batch, or ``None`` when complete."""
        async with self._connection() as connection:
            plans = await self._episode_plans(connection, run_id)
            batch = await self._fetch_script_batch_lineage(connection, run_id)
            if batch is None or batch["status"] != "active":
                return plans[0].episode_number if plans else None
            committed = {int(key) for key in json.loads(batch["active_pointers_json"])}
        return next(
            (plan.episode_number for plan in plans if plan.episode_number not in committed),
            None,
        )

    async def begin_episode_generation_window(
        self,
        run_id: UUID,
        *,
        design_candidate_id: str,
        design_content_hash: str,
        design_epoch: int,
        group_id: str,
        start_episode: int,
        end_episode: int,
        operation_id: str,
    ) -> str:
        """Persist one exact model-generation attempt before provider dispatch."""
        window_id = f"episode_generation_window_{uuid4().hex}"
        now = _utc_now().isoformat()
        async with self._transaction() as connection:
            reusable = await self._fetchone(
                connection,
                """
                SELECT window_id FROM episode_generation_windows
                WHERE run_id = ?
                  AND design_candidate_id = ?
                  AND design_content_hash = ?
                  AND design_epoch = ?
                  AND group_id = ?
                  AND start_episode = ?
                  AND end_episode = ?
                  AND status = 'generating'
                  AND screenplay_text IS NOT NULL
                  AND screenplay_nonce IS NOT NULL
                  AND screenplay_manifest_json IS NOT NULL
                  AND content_call_id IS NOT NULL
                ORDER BY created_at DESC
                LIMIT 1
                """,
                (
                    str(run_id),
                    design_candidate_id,
                    design_content_hash,
                    design_epoch,
                    group_id,
                    start_episode,
                    end_episode,
                ),
            )
            if reusable is not None:
                return str(reusable["window_id"])
            await connection.execute(
                """
                UPDATE episode_generation_windows
                SET status = 'stale', updated_at = ?
                WHERE run_id = ? AND status IN ('generating', 'generated', 'partially_committed')
                      AND start_episode >= ?
                """,
                (now, str(run_id), start_episode),
            )
            await connection.execute(
                """
                INSERT INTO episode_generation_windows(
                    window_id, run_id, design_candidate_id, design_content_hash,
                    design_epoch, group_id, start_episode, end_episode, operation_id,
                    call_id, status, committed_through_episode, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, 'generating', NULL, ?, ?)
                """,
                (
                    window_id,
                    str(run_id),
                    design_candidate_id,
                    design_content_hash,
                    design_epoch,
                    group_id,
                    start_episode,
                    end_episode,
                    operation_id,
                    now,
                    now,
                ),
            )
        return window_id

    async def save_episode_generation_text(
        self,
        run_id: UUID,
        window_id: str,
        *,
        screenplay_text: str,
        nonce: str,
        manifest: list[Mapping[str, Any]],
        content_call_id: str,
        context_bundle_sha256: str | None,
    ) -> None:
        """Persist immutable plaintext before the compact state-sidecar call."""
        if not screenplay_text.strip() or not nonce:
            raise DomainError(
                "invalid_screenplay_text",
                "A generated screenplay text artifact must be non-empty.",
                409,
            )
        now = _utc_now().isoformat()
        manifest_json = json.dumps(
            manifest,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        async with self._transaction() as connection:
            cursor = await connection.execute(
                """
                UPDATE episode_generation_windows
                SET screenplay_text = ?, screenplay_nonce = ?,
                    screenplay_manifest_json = ?, content_call_id = ?,
                    context_bundle_sha256 = ?, updated_at = ?
                WHERE window_id = ? AND run_id = ? AND status = 'generating'
                  AND screenplay_text IS NULL
                """,
                (
                    screenplay_text,
                    nonce,
                    manifest_json,
                    content_call_id,
                    context_bundle_sha256,
                    now,
                    window_id,
                    str(run_id),
                ),
            )
            if cursor.rowcount != 1:
                existing = await self._fetchone(
                    connection,
                    """
                    SELECT screenplay_text, screenplay_nonce, screenplay_manifest_json,
                           content_call_id, context_bundle_sha256
                    FROM episode_generation_windows
                    WHERE window_id = ? AND run_id = ? AND status = 'generating'
                    """,
                    (window_id, str(run_id)),
                )
                expected = (
                    screenplay_text,
                    nonce,
                    manifest_json,
                    content_call_id,
                    context_bundle_sha256,
                )
                actual = tuple(existing) if existing is not None else None
                if actual != expected:
                    raise DomainError(
                        "generation_window_conflict",
                        "The screenplay text artifact no longer matches its generation window.",
                        409,
                    )

    async def get_episode_generation_text(
        self,
        run_id: UUID,
        window_id: str,
    ) -> dict[str, Any] | None:
        async with self._connection() as connection:
            row = await self._fetchone(
                connection,
                """
                SELECT screenplay_text, screenplay_nonce, screenplay_manifest_json,
                       content_call_id, context_bundle_sha256, operation_id
                FROM episode_generation_windows
                WHERE window_id = ? AND run_id = ?
                  AND status IN ('generating', 'generated', 'partially_committed', 'committed')
                  AND screenplay_text IS NOT NULL
                """,
                (window_id, str(run_id)),
            )
        if row is None:
            return None
        return {
            "raw_text": row["screenplay_text"],
            "nonce": row["screenplay_nonce"],
            "manifest": json.loads(row["screenplay_manifest_json"]),
            "content_call_id": row["content_call_id"],
            "context_bundle_sha256": row["context_bundle_sha256"],
            "operation_id": row["operation_id"],
        }

    async def bind_episode_generation_window_call(
        self,
        run_id: UUID,
        window_id: str,
        *,
        call_id: str,
        sidecar_call_id: str | None = None,
    ) -> None:
        now = _utc_now().isoformat()
        async with self._transaction() as connection:
            cursor = await connection.execute(
                """
                UPDATE episode_generation_windows
                SET call_id = COALESCE(content_call_id, ?), sidecar_call_id = ?,
                    status = 'generated', updated_at = ?
                WHERE window_id = ? AND run_id = ? AND status = 'generating'
                  AND (screenplay_text IS NULL OR content_call_id IS NOT NULL)
                """,
                (call_id, sidecar_call_id, now, window_id, str(run_id)),
            )
            if cursor.rowcount != 1:
                raise DomainError(
                    "generation_window_conflict",
                    "The episode generation window is no longer active.",
                    409,
                )

    async def fail_episode_generation_window(
        self,
        run_id: UUID,
        window_id: str,
        *,
        preserve_text: bool = False,
    ) -> None:
        async with self._transaction() as connection:
            sql = """
                UPDATE episode_generation_windows
                SET status = 'failed', updated_at = ?
                WHERE window_id = ? AND run_id = ?
                  AND status IN ('generating', 'generated', 'partially_committed')
            """
            if preserve_text:
                sql += " AND screenplay_text IS NULL"
            await connection.execute(
                sql,
                (_utc_now().isoformat(), window_id, str(run_id)),
            )

    async def get_episode_generation_windows(self, run_id: UUID) -> list[dict[str, Any]]:
        async with self._connection() as connection:
            cursor = await connection.execute(
                """
                SELECT * FROM episode_generation_windows
                WHERE run_id = ? ORDER BY created_at, window_id
                """,
                (str(run_id),),
            )
            return [dict(row) for row in await cursor.fetchall()]

    async def create_script_batch(
        self,
        run_id: UUID,
        *,
        design_candidate_id: str,
        design_content_hash: str,
        design_epoch: int,
        now: datetime | None = None,
    ) -> ScriptBatchLineage:
        """Create or return the active design-bound script batch for one run.

        When the active design identity changes, the prior active batch and every
        active candidate are superseded and the active episode projection is reset,
        starting a fresh batch at episode 1 (FSW-A7).
        """
        timestamp = _timestamp(now or _utc_now())
        async with self._transaction() as connection:
            run = await self._fetchone(
                connection,
                "SELECT id, kind, creation_id FROM runs WHERE id = ?",
                (str(run_id),),
            )
            if run is None:
                raise DomainError("run_not_found", "Workflow run not found.", 404)
            prior = await self._fetch_script_batch_lineage(connection, run_id)
            if prior is not None and prior["status"] == "active":
                if (
                    prior["design_candidate_id"] == design_candidate_id
                    and prior["design_content_hash"] == design_content_hash
                    and prior["design_epoch"] == design_epoch
                ):
                    return self._script_batch_lineage_from_row(prior)
                await self._supersede_active_batch(connection, run_id, prior, timestamp)
            batch_epoch = int(prior["batch_epoch"]) + 1 if prior is not None else 1
            batch_id = new_batch_id()
            await connection.execute(
                """
                INSERT INTO script_batches(
                    batch_id, run_id, creation_id, run_kind, batch_epoch, status,
                    design_candidate_id, design_content_hash, design_epoch,
                    active_pointers_json, created_at
                ) VALUES (?, ?, ?, ?, ?, 'active', ?, ?, ?, '{}', ?)
                """,
                (
                    batch_id,
                    str(run_id),
                    str(run["creation_id"]),
                    run["kind"],
                    batch_epoch,
                    design_candidate_id,
                    design_content_hash,
                    design_epoch,
                    timestamp,
                ),
            )
            await connection.execute(
                """
                UPDATE run_progress
                SET current_episode = NULL,
                    timeout_stage = NULL,
                    timeout_count = 0,
                    recovery_reason = 'none',
                    content_repair_count = NULL,
                    pause_message = NULL,
                    updated_at = ?
                WHERE run_id = ?
                """,
                (timestamp, str(run_id)),
            )
            await connection.execute(
                "UPDATE creations SET updated_at = ? WHERE id = ?",
                (timestamp, str(run["creation_id"])),
            )
            stored = await self._fetch_script_batch_lineage(connection, run_id)
        return self._script_batch_lineage_from_row(stored)

    async def commit_episode_candidate(
        self,
        run_id: UUID,
        *,
        episode_number: int,
        content: str,
        episode_lock: EpisodeLock,
        call_id: str,
        writer_notes: str,
        generation_window_id: str | None = None,
        now: datetime | None = None,
    ) -> EpisodeCandidate:
        """Transactional/CAS commit of one immutable episode candidate.

        The candidate is deterministically validated against the retained active
        prefix and may advance the active pointer only when it binds the current
        active design batch, matches the active predecessor pointer, and carries
        the next version. A failing or late candidate never moves a pointer
        (FSW-A8/A9).
        """
        if not content.strip():
            raise DomainError(
                "invalid_episode_draft",
                "An episode draft must contain content.",
                409,
            )
        timestamp = _timestamp(now or _utc_now())
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
            batch_row = await self._fetch_script_batch_lineage(connection, run_id)
            if batch_row is None or batch_row["status"] != "active":
                raise DomainError(
                    "episode_batch_missing",
                    "A design-bound active script batch is required before committing an episode.",
                    409,
                )
            batch = self._script_batch_lineage_from_row(batch_row)
            generation_window = None
            if generation_window_id is not None:
                generation_window = await self._fetchone(
                    connection,
                    """
                    SELECT * FROM episode_generation_windows
                    WHERE window_id = ? AND run_id = ?
                    """,
                    (generation_window_id, str(run_id)),
                )
                if (
                    generation_window is None
                    or generation_window["status"] not in {"generated", "partially_committed"}
                    or generation_window["call_id"] != call_id
                    or generation_window["design_candidate_id"] != batch.design_candidate_id
                    or generation_window["design_content_hash"] != batch.design_content_hash
                    or int(generation_window["design_epoch"]) != batch.design_epoch
                    or not int(generation_window["start_episode"])
                    <= episode_number
                    <= int(generation_window["end_episode"])
                ):
                    raise DomainError(
                        "generation_window_conflict",
                        "The episode candidate does not bind the active generation window.",
                        409,
                    )
                expected_window_episode = (
                    int(generation_window["committed_through_episode"]) + 1
                    if generation_window["committed_through_episode"] is not None
                    else int(generation_window["start_episode"])
                )
                if episode_number != expected_window_episode:
                    raise DomainError(
                        "generation_window_sequence_conflict",
                        "Episode generation window commits must be contiguous.",
                        409,
                    )
            active_pointers = json.loads(batch_row["active_pointers_json"])
            committed = {int(key): value for key, value in active_pointers.items()}
            plans = await self._episode_plans(connection, run_id)
            plan_numbers = [plan.episode_number for plan in plans]
            if episode_number not in plan_numbers:
                raise DomainError(
                    "episode_not_planned",
                    "The episode is not in the approved outline.",
                    409,
                )
            first_unfinished = next(
                (number for number in plan_numbers if number not in committed),
                None,
            )
            if episode_number != first_unfinished:
                raise DomainError(
                    "episode_out_of_order",
                    "Only the first unfinished episode can be committed.",
                    409,
                )
            contract, contract_hash = await self._design_contract_for_batch(connection, batch)
            if episode_number == 1:
                prior_state = initial_series_state(contract, contract_hash)
                predecessor_id = None
                predecessor_hash = None
            else:
                predecessor_id = committed[episode_number - 1]
                predecessor_row = await self._fetch_episode_candidate(
                    connection,
                    run_id,
                    predecessor_id,
                )
                if predecessor_row is None or predecessor_row["status"] != "active":
                    raise DomainError(
                        "episode_predecessor_missing",
                        "The active predecessor candidate is missing.",
                        409,
                    )
                predecessor_hash = predecessor_row["content_sha256"]
                prior_state = SeriesState.model_validate_json(predecessor_row["series_state_json"])
            existing_versions = await self._episode_candidate_versions(
                connection,
                batch.batch_id,
                episode_number,
            )
            version = max(existing_versions, default=0) + 1
            try:
                candidate = build_episode_candidate(
                    run_id=str(run_id),
                    run_kind=batch.run_kind,
                    batch=batch,
                    contract=contract,
                    prior_state=prior_state,
                    episode_number=episode_number,
                    version=version,
                    predecessor_candidate_id=predecessor_id,
                    predecessor_sha256=predecessor_hash,
                    content=content,
                    delta=episode_lock.state_delta,
                    semantic_review=episode_lock.semantic_review,
                    repair_rounds=episode_lock.repair_rounds,
                    repair_constraints=episode_lock.series_state.repair_constraints,
                    call_id=call_id,
                    generation_window_id=generation_window_id,
                    writer_notes=writer_notes,
                    now=_datetime(timestamp),
                )
            except ContinuityViolation as exc:
                raise DomainError(
                    "episode_candidate_invalid",
                    exc.evidence,
                    409,
                ) from exc
            await self._insert_episode_candidate(connection, candidate, timestamp)
            new_pointers = {**active_pointers, str(episode_number): candidate.candidate_id}
            await connection.execute(
                """
                UPDATE script_batches
                SET active_pointers_json = ?
                WHERE batch_id = ? AND status = 'active'
                """,
                (_json(new_pointers), batch.batch_id),
            )
            await connection.execute(
                """
                UPDATE episode_candidates
                SET status = 'active', activated_at = ?
                WHERE candidate_id = ? AND status = 'unvalidated'
                """,
                (timestamp, candidate.candidate_id),
            )
            if generation_window is not None:
                window_status = (
                    "committed"
                    if episode_number == int(generation_window["end_episode"])
                    else "partially_committed"
                )
                await connection.execute(
                    """
                    UPDATE episode_generation_windows
                    SET committed_through_episode = ?, status = ?, updated_at = ?
                    WHERE window_id = ? AND run_id = ?
                    """,
                    (
                        episode_number,
                        window_status,
                        timestamp,
                        generation_window_id,
                        str(run_id),
                    ),
                )
            await self._upsert_active_draft_projection(
                connection,
                run_id,
                candidate,
            )
            next_unfinished = next(
                (number for number in plan_numbers if str(number) not in new_pointers),
                None,
            )
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
                    next_unfinished,
                    timestamp,
                    str(run_id),
                ),
            )
            await connection.execute(
                "UPDATE creations SET updated_at = ? WHERE id = ?",
                (timestamp, str(run["creation_id"])),
            )
            stored = await self._fetch_episode_candidate(connection, run_id, candidate.candidate_id)
        return self._episode_candidate_from_row(stored)

    async def record_stale_episode_candidate(
        self,
        run_id: UUID,
        *,
        episode_number: int,
        content: str,
        episode_lock: EpisodeLock,
        call_id: str,
        writer_notes: str,
        now: datetime | None = None,
    ) -> EpisodeCandidate:
        """Retain one generation that did not advance the active pointer as stale evidence.

        A late or superseded generation is stored with its usage lineage and status
        ``stale`` and can never change any active pointer (FSW-A9).
        """
        if not content.strip():
            raise DomainError(
                "invalid_episode_draft",
                "An episode draft must contain content.",
                409,
            )
        timestamp = _timestamp(now or _utc_now())
        async with self._transaction() as connection:
            batch_row = await self._fetch_script_batch_lineage(connection, run_id)
            if batch_row is None:
                raise DomainError(
                    "episode_batch_missing",
                    "A design-bound script batch is required to record stale evidence.",
                    409,
                )
            batch = self._script_batch_lineage_from_row(batch_row)
            existing_versions = await self._episode_candidate_versions(
                connection,
                batch.batch_id,
                episode_number,
            )
            version = max(existing_versions, default=0) + 1
            active_pointers = json.loads(batch_row["active_pointers_json"])
            predecessor_id = None
            predecessor_hash = None
            if episode_number > 1 and str(episode_number - 1) in active_pointers:
                predecessor_id = active_pointers[str(episode_number - 1)]
                predecessor_row = await self._fetch_episode_candidate(
                    connection,
                    run_id,
                    predecessor_id,
                )
                predecessor_hash = (
                    predecessor_row["content_sha256"] if predecessor_row is not None else None
                )
            candidate = EpisodeCandidate(
                candidate_id=new_candidate_id(),
                batch_id=batch.batch_id,
                batch_epoch=batch.batch_epoch,
                run_id=str(run_id),
                design_candidate_id=batch.design_candidate_id,
                design_content_hash=batch.design_content_hash,
                design_epoch=batch.design_epoch,
                episode_number=episode_number,
                version=version,
                content=content,
                content_sha256=_text_hash(content),
                predecessor_candidate_id=predecessor_id,
                predecessor_sha256=predecessor_hash,
                call_id=call_id,
                writer_notes=writer_notes,
                state_delta=episode_lock.state_delta,
                series_state=episode_lock.series_state,
                series_state_sha256=episode_lock.series_state_sha256,
                semantic_review=episode_lock.semantic_review,
                repair_rounds=episode_lock.repair_rounds,
                status="stale",
                created_at=_datetime(timestamp),
            )
            await self._insert_episode_candidate(connection, candidate, timestamp)
            stored = await self._fetch_episode_candidate(connection, run_id, candidate.candidate_id)
        return self._episode_candidate_from_row(stored)

    async def rewrite_episode_suffix(
        self,
        run_id: UUID,
        from_episode: int,
        *,
        now: datetime | None = None,
    ) -> Mapping[str, Any]:
        """Preserve active 1..N-1, supersede every active candidate N..end, replay state.

        The returned mapping carries the next episode to write, the folded
        SeriesState replayed strictly from the retained prefix, and the retained
        prefix candidate list. No event, knowledge, obligation, or state delta from
        the superseded suffix enters the replayed context (FSW-A5/A6).
        """
        if from_episode < 1:
            raise ValueError("Suffix rewrite requires an episode number >= 1")
        timestamp = _timestamp(now or _utc_now())
        async with self._transaction() as connection:
            run = await self._fetchone(
                connection,
                "SELECT id, creation_id FROM runs WHERE id = ?",
                (str(run_id),),
            )
            if run is None:
                raise DomainError("run_not_found", "Workflow run not found.", 404)
            batch_row = await self._fetch_script_batch_lineage(connection, run_id)
            if batch_row is None or batch_row["status"] != "active":
                raise DomainError(
                    "episode_batch_missing",
                    "An active script batch is required for suffix rewrite.",
                    409,
                )
            plans = await self._episode_plans(connection, run_id)
            plan_numbers = [plan.episode_number for plan in plans]
            if from_episode not in plan_numbers:
                raise DomainError(
                    "episode_not_planned",
                    "The rewrite target is not in the approved outline.",
                    409,
                )
            await self._start_episode_attempt_cycle(
                connection,
                run_id,
                from_episode,
                plan_numbers,
                timestamp=timestamp,
            )
            active_pointers = json.loads(batch_row["active_pointers_json"])
            for episode, candidate_id in sorted(active_pointers.items()):
                if int(episode) >= from_episode:
                    await connection.execute(
                        """
                        UPDATE episode_candidates
                        SET status = 'superseded', superseded_at = ?
                        WHERE candidate_id = ? AND status = 'active'
                        """,
                        (timestamp, candidate_id),
                    )
            new_pointers = {
                key: value for key, value in active_pointers.items() if int(key) < from_episode
            }
            await connection.execute(
                """
                UPDATE script_batches
                SET active_pointers_json = ?
                WHERE batch_id = ? AND status = 'active'
                """,
                (_json(new_pointers), batch_row["batch_id"]),
            )
            await connection.execute(
                """
                DELETE FROM episode_drafts
                WHERE run_id = ? AND episode_number >= ?
                """,
                (str(run_id), from_episode),
            )
            prefix_candidates: list[EpisodeCandidate] = []
            if from_episode == 1:
                batch = self._script_batch_lineage_from_row(batch_row)
                contract, contract_hash = await self._design_contract_for_batch(connection, batch)
                prior_state = initial_series_state(contract, contract_hash)
            else:
                for episode in range(1, from_episode):
                    candidate_id = new_pointers.get(str(episode))
                    if candidate_id is None:
                        raise DomainError(
                            "episode_predecessor_missing",
                            "The retained prefix candidate is missing.",
                            409,
                        )
                    row = await self._fetch_episode_candidate(connection, run_id, candidate_id)
                    if row is None:
                        raise DomainError(
                            "episode_predecessor_missing",
                            "The retained prefix candidate is missing.",
                            409,
                        )
                    prefix_candidates.append(self._episode_candidate_from_row(row))
                prior_state = prefix_candidates[-1].series_state
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
                    from_episode,
                    timestamp,
                    str(run_id),
                ),
            )
            await connection.execute(
                "UPDATE creations SET updated_at = ? WHERE id = ?",
                (timestamp, str(run["creation_id"])),
            )
            updated_batch = await self._fetch_script_batch_lineage(connection, run_id)
        return {
            "batch": self._script_batch_lineage_from_row(updated_batch),
            "next_episode": from_episode,
            "prior_state": prior_state,
            "prefix_candidates": prefix_candidates,
        }

    async def apply_quality_episode_patches(
        self,
        run_id: UUID,
        *,
        expected_batch_id: str,
        patched_contents: Mapping[int, tuple[str, str]],
        stage: InternalStage,
        rejection_attempt: int,
        now: datetime | None = None,
    ) -> ScriptBatchLineage:
        """Atomically activate a new batch with only bound episode content changed.

        Every episode is deterministically rebound into the new immutable batch. Non-target
        content hashes must remain identical, and every rebuilt SeriesState must equal the
        prior active state so later episodes cannot silently inherit changed continuity.
        """
        if not patched_contents:
            raise DomainError("quality_repair_invalid", "No episode patch was supplied.", 409)
        timestamp = _timestamp(now or _utc_now())
        async with self._transaction() as connection:
            run = await self._fetchone(
                connection,
                "SELECT state, creation_id, kind FROM runs WHERE id = ?",
                (str(run_id),),
            )
            if run is None:
                raise DomainError("run_not_found", "Workflow run not found.", 404)
            if run["state"] != "running":
                raise DomainError("run_not_running", "Workflow run is not running.", 409)
            batch_row = await self._fetch_script_batch_lineage(connection, run_id)
            if (
                batch_row is None
                or batch_row["status"] != "active"
                or batch_row["batch_id"] != expected_batch_id
            ):
                raise DomainError(
                    "quality_repair_stale",
                    "The active script batch changed before the quality repair committed.",
                    409,
                )
            old_batch = self._script_batch_lineage_from_row(batch_row)
            repair_row = await self._fetchone(
                connection,
                """
                SELECT plan_json, status
                FROM quality_gate_repairs
                WHERE run_id = ? AND stage = ? AND rejection_attempt = ?
                """,
                (str(run_id), stage.value, rejection_attempt),
            )
            if (
                repair_row is None
                or repair_row["status"] not in {"queued", "repairing"}
                or repair_row["plan_json"] is None
            ):
                raise DomainError(
                    "quality_repair_stale",
                    "The evidence-bound quality repair is unavailable.",
                    409,
                )
            repair_plan = QualityRepairPlan.model_validate_json(repair_row["plan_json"])
            if repair_plan.scope != "episode_content":
                raise DomainError(
                    "quality_repair_invalid",
                    "Only an episode-content plan can activate a repaired script batch.",
                    409,
                )
            planned_episodes = {issue.episode_number for issue in repair_plan.issues}
            if set(patched_contents) != planned_episodes:
                raise DomainError(
                    "quality_repair_scope_violation",
                    "The repair must cover exactly the episodes named by the saved evidence.",
                    409,
                )
            bound_excerpts = {
                episode_number: [
                    issue.exact_excerpt
                    for issue in repair_plan.issues
                    if issue.episode_number == episode_number
                ]
                for episode_number in patched_contents
            }
            if any(not excerpts for excerpts in bound_excerpts.values()):
                raise DomainError(
                    "quality_repair_scope_violation",
                    "A patched episode was not named by the saved review evidence.",
                    409,
                )
            active_pointers = json.loads(batch_row["active_pointers_json"])
            plans = await self._episode_plans(connection, run_id)
            plan_numbers = [plan.episode_number for plan in plans]
            if set(int(number) for number in active_pointers) != set(plan_numbers):
                raise DomainError(
                    "episode_sequence_incomplete",
                    "Every planned episode must be active before a quality repair.",
                    409,
                )
            if not set(patched_contents).issubset(plan_numbers):
                raise DomainError(
                    "episode_not_planned",
                    "A quality repair targeted an unplanned episode.",
                    409,
                )
            old_candidates: list[EpisodeCandidate] = []
            for episode_number in plan_numbers:
                row = await self._fetch_episode_candidate(
                    connection,
                    run_id,
                    active_pointers[str(episode_number)],
                )
                if row is None or row["status"] != "active":
                    raise DomainError(
                        "episode_predecessor_missing",
                        "The active episode candidate is missing.",
                        409,
                    )
                old_candidates.append(self._episode_candidate_from_row(row))
            new_batch = ScriptBatchLineage(
                batch_id=new_batch_id(),
                run_id=str(run_id),
                run_kind=old_batch.run_kind,
                batch_epoch=old_batch.batch_epoch + 1,
                status="active",
                design_candidate_id=old_batch.design_candidate_id,
                design_content_hash=old_batch.design_content_hash,
                design_epoch=old_batch.design_epoch,
                active_pointers={},
                created_at=_datetime(timestamp),
            )
            predecessor_id = None
            predecessor_hash = None
            rebuilt: list[EpisodeCandidate] = []
            for old in old_candidates:
                content, call_id = patched_contents.get(
                    old.episode_number,
                    (old.content, old.call_id),
                )
                if old.episode_number in patched_contents and not _is_bound_quality_patch(
                    old.content,
                    content,
                    bound_excerpts[old.episode_number],
                ):
                    raise DomainError(
                        "quality_repair_scope_violation",
                        "The repaired episode changed content outside the bound excerpts.",
                        409,
                    )
                if (
                    old.episode_number not in patched_contents
                    and _text_hash(content) != old.content_sha256
                ):
                    raise DomainError(
                        "quality_repair_scope_violation",
                        "A non-target episode changed during quality repair.",
                        409,
                    )
                candidate = old.model_copy(
                    update={
                        "candidate_id": new_candidate_id(),
                        "batch_id": new_batch.batch_id,
                        "batch_epoch": new_batch.batch_epoch,
                        "version": 1,
                        "content": content,
                        "content_sha256": _text_hash(content),
                        "predecessor_candidate_id": predecessor_id,
                        "predecessor_sha256": predecessor_hash,
                        "call_id": call_id,
                        "status": "active",
                        "created_at": _datetime(timestamp),
                        "activated_at": _datetime(timestamp),
                        "superseded_at": None,
                    }
                )
                rebuilt.append(candidate)
                predecessor_id = candidate.candidate_id
                predecessor_hash = candidate.content_sha256

            await self._supersede_active_batch(connection, run_id, batch_row, timestamp)
            await connection.execute(
                """
                INSERT INTO script_batches(
                    batch_id, run_id, creation_id, run_kind, batch_epoch, status,
                    design_candidate_id, design_content_hash, design_epoch,
                    active_pointers_json, suffix_rewrite_count, created_at
                ) VALUES (?, ?, ?, ?, ?, 'active', ?, ?, ?, ?, ?, ?)
                """,
                (
                    new_batch.batch_id,
                    str(run_id),
                    run["creation_id"],
                    run["kind"],
                    new_batch.batch_epoch,
                    new_batch.design_candidate_id,
                    new_batch.design_content_hash,
                    new_batch.design_epoch,
                    _json(
                        {
                            str(candidate.episode_number): candidate.candidate_id
                            for candidate in rebuilt
                        }
                    ),
                    int(batch_row["suffix_rewrite_count"]),
                    timestamp,
                ),
            )
            for candidate in rebuilt:
                await self._insert_episode_candidate(connection, candidate, timestamp)
                await self._upsert_active_draft_projection(connection, run_id, candidate)
            checkpoint_row = await self._fetchone(
                connection,
                """
                SELECT payload_json FROM business_checkpoints
                WHERE run_id = ? AND stage = ?
                """,
                (str(run_id), InternalStage.GENERATING_EPISODE_SCRIPTS.value),
            )
            if checkpoint_row is None:
                raise DomainError(
                    "quality_repair_stale",
                    "The approved episode checkpoint is unavailable for a bound repair.",
                    409,
                )
            previous_aggregate = json.loads(checkpoint_row["payload_json"])
            drafts = await self._episode_drafts(connection, run_id)
            try:
                aggregate = {
                    **previous_aggregate,
                    "stage": InternalStage.GENERATING_EPISODE_SCRIPTS.value,
                    "content": _aggregate_episode_scripts(plans, drafts),
                }
            except ValueError as exc:
                raise DomainError(
                    "episode_sequence_incomplete",
                    "Every planned episode must be active after a quality repair.",
                    409,
                ) from exc
            if "episode_hashes" in previous_aggregate:
                aggregate["episode_hashes"] = [
                    {
                        "episode_number": candidate.episode_number,
                        "content_sha256": candidate.content_sha256,
                        "series_state_sha256": candidate.series_state_sha256,
                    }
                    for candidate in rebuilt
                ]
                aggregate["series_state_sha256"] = rebuilt[-1].series_state_sha256
            checkpoint_cursor = await connection.execute(
                """
                UPDATE business_checkpoints
                SET payload_json = ?, payload_sha256 = ?, approved_at = ?
                WHERE run_id = ? AND stage = ?
                """,
                (
                    _json(aggregate),
                    canonical_payload_hash(aggregate),
                    timestamp,
                    str(run_id),
                    InternalStage.GENERATING_EPISODE_SCRIPTS.value,
                ),
            )
            if checkpoint_cursor.rowcount != 1:
                raise DomainError(
                    "quality_repair_stale",
                    "The approved episode checkpoint is unavailable for a bound repair.",
                    409,
                )
            await connection.execute(
                """
                DELETE FROM business_checkpoints
                WHERE run_id = ? AND stage IN ('accepting_l0', 'accepting_l4')
                """,
                (str(run_id),),
            )
            await connection.execute(
                """
                UPDATE series_reviews SET status = 'stale'
                WHERE run_id = ? AND status = 'active'
                """,
                (str(run_id),),
            )
            repair_cursor = await connection.execute(
                """
                UPDATE quality_gate_repairs
                SET status = 'applied', repaired_batch_id = ?, completed_at = ?
                WHERE run_id = ? AND stage = ? AND rejection_attempt = ?
                  AND status IN ('queued', 'repairing')
                """,
                (
                    new_batch.batch_id,
                    timestamp,
                    str(run_id),
                    stage.value,
                    rejection_attempt,
                ),
            )
            if repair_cursor.rowcount != 1:
                raise DomainError(
                    "quality_repair_stale",
                    "The queued quality repair changed before the new batch committed.",
                    409,
                )
            await connection.execute(
                """
                UPDATE run_progress
                SET current_stage = 'accepting_l0', current_episode = NULL,
                    execution_state = 'running', updated_at = ?
                WHERE run_id = ?
                """,
                (timestamp, str(run_id)),
            )
            stored = await self._fetch_script_batch_lineage(connection, run_id)
        return self._script_batch_lineage_from_row(stored)

    async def _supersede_active_batch(
        self,
        connection: aiosqlite.Connection,
        run_id: UUID,
        batch_row: aiosqlite.Row,
        timestamp: str,
    ) -> None:
        await connection.execute(
            """
            UPDATE script_batches
            SET status = 'superseded', superseded_at = ?
            WHERE batch_id = ? AND status = 'active'
            """,
            (timestamp, batch_row["batch_id"]),
        )
        await connection.execute(
            """
            UPDATE episode_candidates
            SET status = 'superseded', superseded_at = COALESCE(superseded_at, ?)
            WHERE batch_id = ? AND status = 'active'
            """,
            (timestamp, batch_row["batch_id"]),
        )
        await connection.execute(
            "DELETE FROM episode_drafts WHERE run_id = ?",
            (str(run_id),),
        )

    async def _insert_episode_candidate(
        self,
        connection: aiosqlite.Connection,
        candidate: EpisodeCandidate,
        timestamp: str,
    ) -> None:
        await connection.execute(
            """
            INSERT INTO episode_candidates(
                candidate_id, batch_id, batch_epoch, run_id,
                design_candidate_id, design_content_hash, design_epoch,
                episode_number, version, content, content_sha256,
                predecessor_candidate_id, predecessor_sha256, call_id, generation_window_id,
                writer_notes,
                state_delta_json, series_state_json, series_state_sha256,
                semantic_review_json, repair_rounds, status, created_at,
                activated_at, superseded_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                candidate.candidate_id,
                candidate.batch_id,
                candidate.batch_epoch,
                candidate.run_id,
                candidate.design_candidate_id,
                candidate.design_content_hash,
                candidate.design_epoch,
                candidate.episode_number,
                candidate.version,
                candidate.content,
                candidate.content_sha256,
                candidate.predecessor_candidate_id,
                candidate.predecessor_sha256,
                candidate.call_id,
                candidate.generation_window_id,
                candidate.writer_notes,
                _json(candidate.state_delta),
                _json(candidate.series_state),
                candidate.series_state_sha256,
                _json(candidate.semantic_review) if candidate.semantic_review is not None else None,
                candidate.repair_rounds,
                candidate.status,
                _timestamp(candidate.created_at),
                _timestamp(candidate.activated_at) if candidate.activated_at is not None else None,
                (
                    _timestamp(candidate.superseded_at)
                    if candidate.superseded_at is not None
                    else None
                ),
            ),
        )

    async def _upsert_active_draft_projection(
        self,
        connection: aiosqlite.Connection,
        run_id: UUID,
        candidate: EpisodeCandidate,
    ) -> None:
        await connection.execute(
            """
            INSERT INTO episode_drafts(
                run_id, episode_number, content, content_sha256, completed_at,
                contract_sha256, state_delta_json, series_state_json,
                series_state_sha256, semantic_review_json, repair_rounds
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(run_id, episode_number) DO UPDATE SET
                content = excluded.content,
                content_sha256 = excluded.content_sha256,
                completed_at = excluded.completed_at,
                contract_sha256 = excluded.contract_sha256,
                state_delta_json = excluded.state_delta_json,
                series_state_json = excluded.series_state_json,
                series_state_sha256 = excluded.series_state_sha256,
                semantic_review_json = excluded.semantic_review_json,
                repair_rounds = excluded.repair_rounds
            """,
            (
                str(run_id),
                candidate.episode_number,
                candidate.content,
                candidate.content_sha256,
                _timestamp(candidate.created_at),
                candidate.state_delta.contract_sha256,
                _json(candidate.state_delta),
                _json(candidate.series_state),
                candidate.series_state_sha256,
                _json(candidate.semantic_review),
                candidate.repair_rounds,
            ),
        )

    async def _design_contract_for_batch(
        self,
        connection: aiosqlite.Connection,
        batch: ScriptBatchLineage,
    ) -> tuple[StoryContract, str]:
        row = await self._fetchone(
            connection,
            "SELECT content_json FROM series_bible_candidates WHERE candidate_id = ?",
            (batch.design_candidate_id,),
        )
        if row is None:
            raise DomainError(
                "series_bible_candidate_not_found",
                "The active design candidate is missing.",
                409,
            )
        content = SeriesBibleContent.model_validate(json.loads(row["content_json"]))
        contract = content.story_contract
        return contract, story_contract_sha256(contract)

    async def _episode_candidate_versions(
        self,
        connection: aiosqlite.Connection,
        batch_id: str,
        episode_number: int,
    ) -> list[int]:
        cursor = await connection.execute(
            """
            SELECT version
            FROM episode_candidates
            WHERE batch_id = ? AND episode_number = ?
            """,
            (batch_id, episode_number),
        )
        return [int(row["version"]) for row in await cursor.fetchall()]

    @staticmethod
    async def _fetch_script_batch_lineage(
        connection: aiosqlite.Connection,
        run_id: UUID,
    ) -> aiosqlite.Row | None:
        row = await Repository._fetchone(
            connection,
            "SELECT * FROM script_batches WHERE run_id = ? AND status = 'active' LIMIT 1",
            (str(run_id),),
        )
        if row is not None:
            return row
        return await Repository._fetchone(
            connection,
            "SELECT * FROM script_batches WHERE run_id = ? ORDER BY batch_epoch DESC LIMIT 1",
            (str(run_id),),
        )

    @staticmethod
    async def _fetch_episode_candidate(
        connection: aiosqlite.Connection,
        run_id: UUID,
        candidate_id: str,
    ) -> aiosqlite.Row | None:
        return await Repository._fetchone(
            connection,
            "SELECT * FROM episode_candidates WHERE run_id = ? AND candidate_id = ?",
            (str(run_id), candidate_id),
        )

    @staticmethod
    def _script_batch_lineage_from_row(row: aiosqlite.Row) -> ScriptBatchLineage:
        return ScriptBatchLineage(
            batch_id=row["batch_id"],
            run_id=row["run_id"],
            run_kind=row["run_kind"],
            batch_epoch=int(row["batch_epoch"]),
            status=row["status"],
            design_candidate_id=row["design_candidate_id"],
            design_content_hash=row["design_content_hash"],
            design_epoch=int(row["design_epoch"]),
            active_pointers={
                int(key): value for key, value in json.loads(row["active_pointers_json"]).items()
            },
            created_at=_datetime(row["created_at"]),
            superseded_at=(_datetime(row["superseded_at"]) if row["superseded_at"] else None),
        )

    @staticmethod
    def _episode_candidate_from_row(row: aiosqlite.Row) -> EpisodeCandidate:
        return EpisodeCandidate(
            candidate_id=row["candidate_id"],
            batch_id=row["batch_id"],
            batch_epoch=int(row["batch_epoch"]),
            run_id=row["run_id"],
            design_candidate_id=row["design_candidate_id"],
            design_content_hash=row["design_content_hash"],
            design_epoch=int(row["design_epoch"]),
            episode_number=int(row["episode_number"]),
            version=int(row["version"]),
            content=row["content"],
            content_sha256=row["content_sha256"],
            predecessor_candidate_id=row["predecessor_candidate_id"],
            predecessor_sha256=row["predecessor_sha256"],
            call_id=row["call_id"],
            generation_window_id=row["generation_window_id"],
            writer_notes=row["writer_notes"] or "",
            state_delta=EpisodeStateDelta.model_validate_json(row["state_delta_json"]),
            series_state=SeriesState.model_validate_json(row["series_state_json"]),
            series_state_sha256=row["series_state_sha256"],
            semantic_review=(
                SemanticReview.model_validate_json(row["semantic_review_json"])
                if row["semantic_review_json"] is not None
                else None
            ),
            repair_rounds=(int(row["repair_rounds"]) if row["repair_rounds"] is not None else None),
            status=row["status"],
            created_at=_datetime(row["created_at"]),
            activated_at=(_datetime(row["activated_at"]) if row["activated_at"] else None),
            superseded_at=(_datetime(row["superseded_at"]) if row["superseded_at"] else None),
        )

    # ------------------------------------------------------------------
    # Bound structural reviews, shared budgets, and repair authorization
    # ------------------------------------------------------------------

    async def register_series_review(
        self,
        run_id: UUID,
        *,
        review_type: str,
        episode_number: int,
        design_candidate_id: str,
        design_content_hash: str,
        design_epoch: int,
        batch_id: str,
        batch_epoch: int,
        prefix_hash: str,
        call_id: str,
        passed: bool,
        category: str,
        evidence: str,
        earliest_affected_episode: int | None,
        now: datetime | None = None,
    ) -> BoundStructuralReview:
        """Persist one immutable bound structural review and retire superseded reviews.

        A review binds the exact design candidate, script batch/epoch, active-prefix
        hash, and model-call id it observed (RPR-A1). Any prior ``active`` review for
        a different design or batch is marked stale and can never approve, rebuild,
        rewrite, or deliver the current lineage (RPR-A11).
        """
        timestamp = _timestamp(now or _utc_now())
        async with self._transaction() as connection:
            await connection.execute(
                """
                UPDATE series_reviews
                SET status = 'stale'
                WHERE run_id = ?
                  AND status = 'active'
                  AND (
                      design_content_hash <> ?
                      OR batch_id <> ?
                      OR batch_epoch <> ?
                  )
                """,
                (
                    str(run_id),
                    design_content_hash,
                    batch_id,
                    batch_epoch,
                ),
            )
            review = BoundStructuralReview(
                review_id=new_review_id(),
                run_id=str(run_id),
                review_epoch=await self._next_review_epoch(connection, run_id),
                review_type=review_type,
                episode_number=episode_number,
                design_candidate_id=design_candidate_id,
                design_content_hash=design_content_hash,
                design_epoch=design_epoch,
                batch_id=batch_id,
                batch_epoch=batch_epoch,
                prefix_hash=prefix_hash,
                call_id=call_id,
                passed=passed,
                category=category,
                evidence=evidence,
                earliest_affected_episode=earliest_affected_episode,
                status="active",
                reviewed_at=_datetime(timestamp),
            )
            await connection.execute(
                """
                INSERT INTO series_reviews(
                    review_id, run_id, review_epoch, review_type, episode_number,
                    design_candidate_id, design_content_hash, design_epoch,
                    batch_id, batch_epoch, prefix_hash, call_id, passed, category,
                    evidence, earliest_affected_episode, status, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    review.review_id,
                    review.run_id,
                    review.review_epoch,
                    review.review_type,
                    review.episode_number,
                    review.design_candidate_id,
                    review.design_content_hash,
                    review.design_epoch,
                    review.batch_id,
                    review.batch_epoch,
                    review.prefix_hash,
                    review.call_id,
                    1 if review.passed else 0,
                    review.category,
                    review.evidence,
                    review.earliest_affected_episode,
                    review.status,
                    _timestamp(review.reviewed_at),
                ),
            )
        return review

    async def get_latest_passing_final_review(
        self,
        run_id: UUID,
        *,
        design_content_hash: str | None = None,
        batch_id: str | None = None,
        prefix_hash: str | None = None,
    ) -> BoundStructuralReview | None:
        """The latest active passing final review bound to the given lineage.

        Only a passing bound final whole-series review can freeze formal delivery
        (RPR-A13). When lineage filters are supplied, a review bound to a different
        design, batch, or prefix is not a passing gate.
        """
        async with self._connection() as connection:
            row = await self._fetchone(
                connection,
                """
                SELECT * FROM series_reviews
                WHERE run_id = ?
                  AND review_type = 'final'
                  AND status = 'active'
                  AND passed = 1
                  AND (
                      ? IS NULL OR design_content_hash = ?
                  )
                  AND (? IS NULL OR batch_id = ?)
                  AND (? IS NULL OR prefix_hash = ?)
                ORDER BY review_epoch DESC
                LIMIT 1
                """,
                (
                    str(run_id),
                    design_content_hash,
                    design_content_hash,
                    batch_id,
                    batch_id,
                    prefix_hash,
                    prefix_hash,
                ),
            )
        return self._series_review_from_row(row) if row is not None else None

    async def get_series_reviews(self, run_id: UUID) -> list[BoundStructuralReview]:
        """Every bound structural review for one run, newest first (immutable evidence)."""
        async with self._connection() as connection:
            cursor = await connection.execute(
                """
                SELECT * FROM series_reviews
                WHERE run_id = ?
                ORDER BY review_epoch DESC
                """,
                (str(run_id),),
            )
            rows = await cursor.fetchall()
        return [self._series_review_from_row(row) for row in rows]

    async def get_unresolved_script_defect_reviews(
        self,
        run_id: UUID,
    ) -> list[BoundStructuralReview]:
        """Return current-prefix script defects after its latest passing review.

        The newest review selects the exact active prefix under consideration. Reviews
        for earlier prefixes remain immutable audit evidence and may guide the rewrite
        until a replacement prefix is reviewed, but they cannot affect that replacement's
        repair range or evidence. Reviews bound to a stale design or batch are never
        eligible feedback.
        """
        async with self._connection() as connection:
            lineage = await self._fetch_series_bible_lineage(connection, run_id)
            batch = await self._fetch_script_batch_lineage(connection, run_id)
            if (
                lineage is None
                or lineage["active_candidate_id"] is None
                or lineage["active_content_hash"] is None
                or batch is None
                or batch["status"] != "active"
            ):
                return []
            candidate = await self._fetch_series_bible_candidate(
                connection,
                run_id,
                lineage["active_candidate_id"],
            )
            if candidate is None or candidate["status"] != "active":
                return []
            if (
                candidate["candidate_id"] != lineage["active_candidate_id"]
                or candidate["content_hash"] != lineage["active_content_hash"]
                or int(candidate["design_epoch"]) != int(lineage["active_design_epoch"])
                or batch["design_candidate_id"] != candidate["candidate_id"]
                or batch["design_content_hash"] != candidate["content_hash"]
                or int(batch["design_epoch"]) != int(candidate["design_epoch"])
            ):
                return []
            return await self._unresolved_script_defect_reviews_for_lineage(
                connection,
                run_id,
                design_candidate_id=candidate["candidate_id"],
                design_content_hash=candidate["content_hash"],
                design_epoch=int(candidate["design_epoch"]),
                batch_id=batch["batch_id"],
                batch_epoch=int(batch["batch_epoch"]),
            )

    async def _unresolved_script_defect_reviews_for_lineage(
        self,
        connection: aiosqlite.Connection,
        run_id: UUID,
        *,
        design_candidate_id: str,
        design_content_hash: str,
        design_epoch: int,
        batch_id: str,
        batch_epoch: int,
    ) -> list[BoundStructuralReview]:
        """Read unresolved current-prefix reviews while a transaction owns the lineage."""
        binding = (
            str(run_id),
            design_candidate_id,
            design_content_hash,
            design_epoch,
            batch_id,
            batch_epoch,
        )
        latest = await self._fetchone(
            connection,
            """
            SELECT passed, prefix_hash
            FROM series_reviews
            WHERE run_id = ?
              AND design_candidate_id = ?
              AND design_content_hash = ?
              AND design_epoch = ?
              AND batch_id = ?
              AND batch_epoch = ?
              AND status = 'active'
            ORDER BY review_epoch DESC
            LIMIT 1
            """,
            binding,
        )
        if latest is None or bool(latest["passed"]):
            return []
        latest_prefix_hash = str(latest["prefix_hash"])
        passing = await self._fetchone(
            connection,
            """
            SELECT MAX(review_epoch) AS review_epoch
            FROM series_reviews
            WHERE run_id = ?
              AND design_candidate_id = ?
              AND design_content_hash = ?
              AND design_epoch = ?
              AND batch_id = ?
              AND batch_epoch = ?
              AND prefix_hash = ?
              AND status = 'active'
              AND passed = 1
            """,
            (*binding, latest_prefix_hash),
        )
        passing_epoch = (
            int(passing["review_epoch"])
            if passing is not None and passing["review_epoch"] is not None
            else 0
        )
        cursor = await connection.execute(
            """
            SELECT *
            FROM series_reviews
            WHERE run_id = ?
              AND design_candidate_id = ?
              AND design_content_hash = ?
              AND design_epoch = ?
              AND batch_id = ?
              AND batch_epoch = ?
              AND prefix_hash = ?
              AND status = 'active'
              AND passed = 0
              AND category = 'script_defect'
              AND review_epoch > ?
            ORDER BY review_epoch ASC
            """,
            (*binding, latest_prefix_hash, passing_epoch),
        )
        rows = await cursor.fetchall()
        return [self._series_review_from_row(row) for row in rows]

    async def consume_automatic_suffix_budget(
        self,
        run_id: UUID,
        batch_id: str,
        *,
        now: datetime | None = None,
    ) -> None:
        """Consume the single automatic suffix-rewrite budget shared by all reviews.

        All milestone and final reviews share exactly one automatic suffix-rewrite
        per script batch (RPR-A6). A second consumption is rejected.
        """
        async with self._transaction() as connection:
            batch = await self._fetch_script_batch_lineage(connection, run_id)
            if batch is None or batch["batch_id"] != batch_id or batch["status"] != "active":
                raise DomainError(
                    "episode_batch_missing",
                    "The active script batch is missing.",
                    409,
                )
            cursor = await connection.execute(
                """
                UPDATE script_batches
                SET suffix_rewrite_count = 1
                WHERE batch_id = ? AND status = 'active' AND suffix_rewrite_count = 0
                """,
                (batch_id,),
            )
            if cursor.rowcount != 1:
                raise DomainError(
                    "suffix_budget_exhausted",
                    "The automatic suffix-rewrite budget for this script batch is exhausted.",
                    409,
                )

    async def has_automatic_suffix_budget(self, run_id: UUID, batch_id: str) -> bool:
        async with self._connection() as connection:
            batch = await self._fetch_script_batch_lineage(connection, run_id)
        return bool(
            batch is not None
            and batch["batch_id"] == batch_id
            and batch["status"] == "active"
            and int(batch["suffix_rewrite_count"]) == 0
        )

    async def design_rebuild_budget_available(self, run_id: UUID) -> bool:
        lineage = await self.get_series_bible_lineage(run_id)
        return lineage is not None and int(lineage["rebuild_count"]) < 1

    async def trigger_design_rebuild(
        self,
        run_id: UUID,
        *,
        evidence: str,
        now: datetime | None = None,
        authorized: bool = False,
    ) -> None:
        """Trigger the one automatic complete design regeneration for a design defect.

        The run's approved outline and every downstream stage are reset so the
        design stages regenerate; ``_sync_series_bible`` then builds and promotes a
        complete re-reviewed design (consuming the one-per-lineage rebuild budget),
        invalidates the prior script batch, and restarts writing at episode 1
        (RPR-A4). The defect evidence stays bound in ``series_reviews``.

        ``authorized`` is used by ``authorize_repair``: an explicit one-cycle
        repair authorization (RPR-A9) performs exactly one further complete
        rebuild even after the automatic budget is consumed.
        """
        timestamp = _timestamp(now or _utc_now())
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
            lineage = await self._fetch_series_bible_lineage(connection, run_id)
            if lineage is None:
                raise DomainError(
                    "series_bible_lineage_missing",
                    "The design lineage for this run is missing.",
                    409,
                )
            if int(lineage["rebuild_count"]) >= 1 and not authorized:
                raise DomainError(
                    "series_bible_rebuild_exhausted",
                    "This run lineage may rebuild the design automatically at most once.",
                    409,
                )
            batch_row = await self._fetch_script_batch_lineage(connection, run_id)
            if batch_row is not None and batch_row["status"] == "active":
                await self._supersede_active_batch(connection, run_id, batch_row, timestamp)
            await connection.execute(
                """
                DELETE FROM business_checkpoints
                WHERE run_id = ? AND stage IN (
                    'generating_episode_outline',
                    'generating_episode_scripts',
                    'accepting_l0',
                    'accepting_l4',
                    'assembling_delivery'
                )
                """,
                (str(run_id),),
            )
            await connection.execute(
                """
                DELETE FROM episode_plans WHERE run_id = ?
                """,
                (str(run_id),),
            )
            await connection.execute(
                """
                UPDATE run_progress
                SET current_stage = ?,
                    execution_state = 'queued',
                    active_started_at = NULL,
                    timeout_stage = ?,
                    timeout_count = 0,
                    recovery_reason = 'none',
                    content_repair_count = NULL,
                    pause_message = ?,
                    updated_at = ?
                WHERE run_id = ?
                """,
                (
                    InternalStage.GENERATING_EPISODE_OUTLINE.value,
                    UserStage.GENERATING_EPISODE_OUTLINE.value,
                    evidence,
                    timestamp,
                    str(run_id),
                ),
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
                (timestamp, timestamp, str(run_id)),
            )
            await connection.execute(
                "UPDATE creations SET updated_at = ? WHERE id = ?",
                (timestamp, str(run["creation_id"])),
            )

    async def pause_repair_authorization(
        self,
        run_id: UUID,
        *,
        kind: str,
        design_candidate_id: str,
        design_content_hash: str,
        design_epoch: int,
        batch_id: str,
        batch_epoch: int,
        earliest_affected_episode: int | None,
        range_episodes: int | None,
        estimated_tokens: int | None,
        evidence: str,
        review_id: str,
        now: datetime | None = None,
    ) -> None:
        """Pause the run for an exact one-cycle repair authorization.

        The authorization is bound to the active lineage and shows the evidence,
        affected range, and reference context amount at the pause (neither a lower
        bound nor a total cycle forecast) (RPR-A8). It grants at most one
        generation-plus-review cycle
        (RPR-A9).
        """
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
                SELECT COALESCE(MAX(authorization_epoch), 0)
                FROM repair_authorizations WHERE run_id = ?
                """,
                (str(run_id),),
            )
            row = await cursor.fetchone()
            authorization_epoch = int(row[0]) + 1 if row is not None else 1
            await connection.execute(
                """
                INSERT INTO repair_authorizations(
                    run_id, authorization_epoch, kind,
                    design_candidate_id, design_content_hash, design_epoch,
                    batch_id, batch_epoch, earliest_affected_episode,
                    range_episodes, estimated_tokens, evidence, review_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(run_id),
                    authorization_epoch,
                    kind,
                    design_candidate_id,
                    design_content_hash,
                    design_epoch,
                    batch_id,
                    batch_epoch,
                    earliest_affected_episode,
                    range_episodes,
                    estimated_tokens,
                    evidence,
                    review_id,
                ),
            )
            await connection.execute(
                """
                UPDATE run_progress
                SET current_stage = ?,
                    execution_state = 'paused',
                    elapsed_seconds = ?,
                    active_started_at = NULL,
                    timeout_stage = ?,
                    timeout_count = 0,
                    recovery_reason = 'repair_authorization',
                    content_repair_count = NULL,
                    pause_message = ?,
                    updated_at = ?
                WHERE run_id = ?
                """,
                (
                    InternalStage.GENERATING_EPISODE_SCRIPTS.value,
                    self._elapsed_seconds(progress, current),
                    UserStage.FINAL_REVIEW.value,
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

    async def get_repair_authorization(self, run_id: UUID) -> Mapping[str, Any] | None:
        """The latest not-yet-consumed repair authorization (granted or pending)."""
        async with self._connection() as connection:
            row = await self._fetchone(
                connection,
                """
                SELECT * FROM repair_authorizations
                WHERE run_id = ?
                ORDER BY authorization_epoch DESC LIMIT 1
                """,
                (str(run_id),),
            )
        if row is None:
            return None
        return {
            "authorization_epoch": int(row["authorization_epoch"]),
            "kind": row["kind"],
            "design_candidate_id": row["design_candidate_id"],
            "design_content_hash": row["design_content_hash"],
            "design_epoch": int(row["design_epoch"]),
            "batch_id": row["batch_id"],
            "batch_epoch": int(row["batch_epoch"]),
            "earliest_affected_episode": row["earliest_affected_episode"],
            "range_episodes": row["range_episodes"],
            "estimated_tokens": row["estimated_tokens"],
            "evidence": row["evidence"],
            "review_id": row["review_id"],
            "granted_at": row["granted_at"],
            "consumed_at": row["consumed_at"],
            "rebuild_candidate_id": row["rebuild_candidate_id"],
        }

    async def authorize_repair(
        self,
        *,
        creation_id: UUID,
        run_kind: RunKind,
        idempotency_key: str,
        now: datetime | None = None,
    ) -> RunControlAccepted:
        """Grant exactly one generation-plus-review cycle for the pending authorization.

        The authorization is bound to the active lineage; if the active design or
        batch changed since the pause, it cannot be granted (RPR-A9). Granting
        requeues the run for exactly one cycle; if a hard-constraint conflict remains,
        the next pause carries the latest prefix-bound review evidence.
        """
        timestamp = _timestamp(now or _utc_now())
        scope = f"run-control:{creation_id}:{run_kind}:authorize-repair"
        payload_hash = canonical_payload_hash({"action": "authorize-repair"})

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
            if state != "paused" or run["recovery_reason"] != "repair_authorization":
                raise DomainError(
                    "run_not_controllable",
                    "Only a repair-authorization pause can be authorized.",
                    409,
                )
            auth = await self._fetchone(
                connection,
                """
                SELECT * FROM repair_authorizations
                WHERE run_id = ? ORDER BY authorization_epoch DESC LIMIT 1
                """,
                (run["id"],),
            )
            if auth is None:
                raise DomainError(
                    "run_not_controllable",
                    "The repair authorization record is missing.",
                    409,
                )
            if auth["consumed_at"] is not None:
                raise DomainError(
                    "run_not_controllable",
                    "The repair authorization is already consumed.",
                    409,
                )
            lineage = await self._fetch_series_bible_lineage(connection, UUID(run["id"]))
            active_batch = await self._fetch_script_batch_lineage(connection, UUID(run["id"]))
            run_id = UUID(run["id"])
            active_candidate = (
                await self._fetch_series_bible_candidate(
                    connection,
                    run_id,
                    lineage["active_candidate_id"],
                )
                if lineage is not None and lineage["active_candidate_id"] is not None
                else None
            )
            if (
                lineage is None
                or lineage["active_candidate_id"] is None
                or lineage["active_content_hash"] != auth["design_content_hash"]
                or active_candidate is None
                or active_candidate["status"] != "active"
                or active_candidate["candidate_id"] != auth["design_candidate_id"]
                or active_candidate["content_hash"] != auth["design_content_hash"]
                or int(active_candidate["design_epoch"]) != int(auth["design_epoch"])
                or active_batch is None
                or active_batch["status"] != "active"
                or active_batch["batch_id"] != auth["batch_id"]
                or int(active_batch["batch_epoch"]) != int(auth["batch_epoch"])
                or active_batch["design_candidate_id"] != auth["design_candidate_id"]
                or active_batch["design_content_hash"] != auth["design_content_hash"]
                or int(active_batch["design_epoch"]) != int(auth["design_epoch"])
            ):
                raise DomainError(
                    "repair_authorization_stale",
                    "The active lineage changed; the repair authorization is stale.",
                    409,
                )
            auth_kind = auth["kind"]
            auth_episode = auth["earliest_affected_episode"]
            auth_range = auth["range_episodes"]
            auth_evidence = auth["evidence"]
            auth_review_id = auth["review_id"]
            if auth_kind == "suffix_rewrite":
                unresolved = await self._unresolved_script_defect_reviews_for_lineage(
                    connection,
                    run_id,
                    design_candidate_id=active_candidate["candidate_id"],
                    design_content_hash=active_candidate["content_hash"],
                    design_epoch=int(active_candidate["design_epoch"]),
                    batch_id=active_batch["batch_id"],
                    batch_epoch=int(active_batch["batch_epoch"]),
                )
                if unresolved:
                    auth_episode = min(
                        review.earliest_affected_episode
                        for review in unresolved
                        if review.earliest_affected_episode is not None
                    )
                    range_row = await self._fetchone(
                        connection,
                        """
                        SELECT COUNT(*) AS episode_count
                        FROM episode_plans
                        WHERE run_id = ? AND episode_number >= ?
                        """,
                        (str(run_id), auth_episode),
                    )
                    planned_range = int(range_row["episode_count"]) if range_row else 0
                    auth_range = planned_range or auth_range
                    auth_evidence = aggregate_script_defect_evidence(unresolved)
                    auth_review_id = unresolved[-1].review_id
                    await connection.execute(
                        """
                        UPDATE repair_authorizations
                        SET earliest_affected_episode = ?,
                            range_episodes = ?,
                            evidence = ?,
                            review_id = ?
                        WHERE run_id = ? AND authorization_epoch = ?
                        """,
                        (
                            auth_episode,
                            auth_range,
                            auth_evidence,
                            auth_review_id,
                            str(run_id),
                            int(auth["authorization_epoch"]),
                        ),
                    )
            await connection.execute(
                """
                UPDATE repair_authorizations
                SET granted_at = ?, consumed_at = ?
                WHERE run_id = ? AND authorization_epoch = ?
                """,
                (timestamp, timestamp, run["id"], int(auth["authorization_epoch"])),
            )
            response = RunControlAccepted(
                creation_id=creation_id,
                run_kind=run_kind,
                run_state="queued",
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
            auth_kind = auth["kind"]

        # RPR-A9: the authorization permits exactly one generation-plus-review cycle.
        # Perform the bound repair now so the run actually regenerates the affected
        # range and a fresh bound review can pass (or return to the evidence pause).
        if auth_kind == "suffix_rewrite" and auth_episode is not None:
            await self.rewrite_episode_suffix(run_id, int(auth_episode))
            await self.requeue_run_job(run_id)
        elif auth_kind == "design_rebuild":
            # RPR-A9: the authorization permits exactly one design-rebuild cycle
            # even after the automatic budget was consumed. If the rebuild trigger
            # itself fails, roll the authorization back so the same evidence pause
            # can be authorized again instead of being stranded as consumed.
            try:
                await self.trigger_design_rebuild(
                    run_id,
                    evidence=auth_evidence or "",
                    authorized=True,
                )
            except Exception:
                async with self._transaction() as connection:
                    await connection.execute(
                        """
                        UPDATE repair_authorizations
                        SET granted_at = NULL, consumed_at = NULL
                        WHERE run_id = ? AND authorization_epoch = ?
                        """,
                        (str(run_id), int(auth["authorization_epoch"])),
                    )
                raise
        return response

    async def _next_review_epoch(
        self,
        connection: aiosqlite.Connection,
        run_id: UUID,
    ) -> int:
        row = await self._fetchone(
            connection,
            "SELECT COALESCE(MAX(review_epoch), 0) AS epoch FROM series_reviews WHERE run_id = ?",
            (str(run_id),),
        )
        return int(row["epoch"]) + 1 if row is not None else 1

    @staticmethod
    def _series_review_from_row(row: aiosqlite.Row) -> BoundStructuralReview:
        return BoundStructuralReview(
            review_id=row["review_id"],
            run_id=row["run_id"],
            review_epoch=int(row["review_epoch"]),
            review_type=row["review_type"],
            episode_number=int(row["episode_number"]),
            design_candidate_id=row["design_candidate_id"],
            design_content_hash=row["design_content_hash"],
            design_epoch=int(row["design_epoch"]),
            batch_id=row["batch_id"],
            batch_epoch=int(row["batch_epoch"]),
            prefix_hash=row["prefix_hash"],
            call_id=row["call_id"],
            passed=bool(row["passed"]),
            category=row["category"],
            evidence=row["evidence"],
            earliest_affected_episode=row["earliest_affected_episode"],
            status=row["status"],
            reviewed_at=_datetime(row["created_at"]),
            consumed_at=(_datetime(row["consumed_at"]) if row["consumed_at"] else None),
        )

    async def _fetch_series_bible_candidate(
        self,
        connection: aiosqlite.Connection,
        run_id: UUID,
        candidate_id: str,
    ) -> aiosqlite.Row | None:
        return await self._fetchone(
            connection,
            """
            SELECT *
            FROM series_bible_candidates
            WHERE run_id = ? AND candidate_id = ?
            """,
            (str(run_id), candidate_id),
        )

    async def _fetch_series_bible_lineage(
        self,
        connection: aiosqlite.Connection,
        run_id: UUID,
    ) -> aiosqlite.Row | None:
        return await self._fetchone(
            connection,
            "SELECT * FROM series_bible_lineage WHERE run_id = ?",
            (str(run_id),),
        )

    @staticmethod
    def _series_bible_from_row(row: aiosqlite.Row) -> SeriesBible:
        return SeriesBible(
            candidate_id=row["candidate_id"],
            version=row["version"],
            design_epoch=row["design_epoch"],
            content_hash=row["content_hash"],
            status=row["status"],
            l0_variant=row["l0_variant"],
            genre=row["genre"],
            lineage=json.loads(row["lineage_json"]),
            content=json.loads(row["content_json"]),
            validation=json.loads(row["validation_json"]) if row["validation_json"] else None,
            global_review=(
                json.loads(row["global_review_json"]) if row["global_review_json"] else None
            ),
            created_at=_datetime(row["created_at"]),
        )

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

    async def requeue_stage_flake_retry(
        self,
        run_id: UUID,
        *,
        stage: InternalStage,
        now: datetime | None = None,
    ) -> None:
        """Requeue a run after a bounded structured-flake retry.

        The retry bound is a worker-side in-memory requeue counter (immune to
        frozen stage-attempt counters such as grouped-outline resume); this
        method only performs the immediate atomic requeue.
        """
        timestamp = _timestamp(now or _utc_now())
        async with self._transaction() as connection:
            progress = await self._fetchone(
                connection,
                """
                SELECT run_progress.*, runs.state AS run_state
                FROM run_progress JOIN runs ON runs.id = run_progress.run_id
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
                UPDATE run_progress
                SET current_stage = ?,
                    execution_state = 'queued',
                    active_started_at = NULL,
                    updated_at = ?
                WHERE run_id = ?
                """,
                (stage.value, timestamp, str(run_id)),
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
                (timestamp, timestamp, str(run_id)),
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
            grouped_outline_resume = False
            if stage is InternalStage.GENERATING_EPISODE_OUTLINE:
                season_map = await self._fetchone(
                    connection,
                    "SELECT 1 FROM outline_season_maps WHERE run_id = ?",
                    (str(run_id),),
                )
                grouped_outline_resume = season_map is not None
            next_state: RecoveryState = (
                "failed"
                if attempt_count >= MAX_STAGE_ATTEMPTS and not grouped_outline_resume
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
        repair_plan: QualityRepairPlan | None = None,
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

            if attempt_count < MAX_STAGE_ATTEMPTS:
                repair_status = (
                    "available"
                    if repair_plan is None or repair_plan.scope == "episode_content"
                    else "blocked"
                )
                await connection.execute(
                    """
                    INSERT INTO quality_gate_repairs(
                        run_id, stage, rejection_attempt, plan_json, status
                    ) VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT(run_id, stage, rejection_attempt) DO UPDATE SET
                        plan_json = COALESCE(quality_gate_repairs.plan_json, excluded.plan_json),
                        status = CASE
                            WHEN quality_gate_repairs.status IN ('queued', 'repairing', 'applied')
                                THEN quality_gate_repairs.status
                            ELSE excluded.status
                        END
                    """,
                    (
                        str(run_id),
                        stage.value,
                        attempt_count,
                        _json(repair_plan) if repair_plan is not None else None,
                        repair_status,
                    ),
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
                repair_plan=repair_plan,
                repair_state=(repair_status if attempt_count < MAX_STAGE_ATTEMPTS else None),
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
            rejection_attempt = int(attempt_row[0])
            repair = await self._fetchone(
                connection,
                """
                SELECT plan_json, status
                FROM quality_gate_repairs
                WHERE run_id = ? AND stage = ? AND rejection_attempt = ?
                """,
                (run["id"], run["current_stage"], rejection_attempt),
            )
            if repair is not None and repair["status"] == "blocked":
                raise DomainError(
                    "run_not_controllable",
                    "The quality rejection is not safely repairable as an episode patch.",
                    409,
                )
            repair_cursor = await connection.execute(
                """
                INSERT INTO quality_gate_repairs(
                    run_id, stage, rejection_attempt, plan_json, status,
                    original_batch_id, requested_at
                )
                VALUES (
                    ?, ?, ?, ?, 'queued',
                    (SELECT batch_id FROM script_batches
                     WHERE run_id = ? AND status = 'active'),
                    ?
                )
                ON CONFLICT(run_id, stage, rejection_attempt) DO UPDATE SET
                    status = 'queued',
                    original_batch_id = excluded.original_batch_id,
                    requested_at = excluded.requested_at
                """,
                (
                    run["id"],
                    run["current_stage"],
                    rejection_attempt,
                    repair["plan_json"] if repair is not None else None,
                    run["id"],
                    timestamp,
                ),
            )
            if repair_cursor.rowcount != 1:
                raise DomainError(
                    "quality_repair_stale",
                    "The active script batch is unavailable for a bound quality repair.",
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

    async def get_queued_quality_repair(self, run_id: UUID) -> Mapping[str, Any] | None:
        async with self._connection() as connection:
            row = await self._fetchone(
                connection,
                """
                SELECT repairs.*, rejections.evidence
                FROM quality_gate_repairs AS repairs
                JOIN quality_gate_rejections AS rejections
                  ON rejections.run_id = repairs.run_id
                 AND rejections.stage = repairs.stage
                 AND rejections.attempt_number = repairs.rejection_attempt
                WHERE repairs.run_id = ? AND repairs.status IN ('queued', 'repairing')
                ORDER BY repairs.rejection_attempt DESC
                LIMIT 1
                """,
                (str(run_id),),
            )
        if row is None:
            return None
        return {
            "run_id": row["run_id"],
            "stage": row["stage"],
            "rejection_attempt": int(row["rejection_attempt"]),
            "plan": (
                QualityRepairPlan.model_validate_json(row["plan_json"])
                if row["plan_json"] is not None
                else None
            ),
            "status": row["status"],
            "original_batch_id": row["original_batch_id"],
            "evidence": row["evidence"],
        }

    async def set_quality_repair_plan(
        self,
        run_id: UUID,
        *,
        stage: InternalStage,
        rejection_attempt: int,
        plan: QualityRepairPlan,
    ) -> None:
        status = "repairing" if plan.scope == "episode_content" else "blocked"
        async with self._transaction() as connection:
            cursor = await connection.execute(
                """
                UPDATE quality_gate_repairs
                SET plan_json = ?, status = ?
                WHERE run_id = ? AND stage = ? AND rejection_attempt = ?
                  AND status IN ('queued', 'repairing')
                """,
                (_json(plan), status, str(run_id), stage.value, rejection_attempt),
            )
            if cursor.rowcount != 1:
                raise DomainError(
                    "quality_repair_stale",
                    "The quality repair request is no longer active.",
                    409,
                )
            if status == "blocked":
                timestamp = _timestamp(_utc_now())
                await connection.execute(
                    """
                    UPDATE run_progress
                    SET execution_state = 'quality_rejected', active_started_at = NULL,
                        updated_at = ?
                    WHERE run_id = ?
                    """,
                    (timestamp, str(run_id)),
                )
                await connection.execute(
                    """
                    UPDATE jobs SET state = 'failed', lease_owner = NULL,
                        lease_expires_at = NULL, updated_at = ?
                    WHERE run_id = ?
                    """,
                    (timestamp, str(run_id)),
                )

    async def block_quality_repair(
        self,
        run_id: UUID,
        *,
        stage: InternalStage,
        rejection_attempt: int,
        evidence: str,
    ) -> None:
        timestamp = _timestamp(_utc_now())
        async with self._transaction() as connection:
            await connection.execute(
                """
                UPDATE quality_gate_repairs
                SET status = 'blocked', completed_at = ?, result_evidence = ?
                WHERE run_id = ? AND stage = ? AND rejection_attempt = ?
                """,
                (timestamp, evidence, str(run_id), stage.value, rejection_attempt),
            )
            await connection.execute(
                """
                UPDATE run_progress
                SET current_stage = ?, execution_state = 'quality_rejected',
                    active_started_at = NULL, updated_at = ?
                WHERE run_id = ?
                """,
                (stage.value, timestamp, str(run_id)),
            )
            await connection.execute(
                """
                UPDATE jobs SET state = 'failed', lease_owner = NULL,
                    lease_expires_at = NULL, updated_at = ?
                WHERE run_id = ?
                """,
                (timestamp, str(run_id)),
            )

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
                if run["recovery_reason"] == "repair_authorization":
                    # RPR-A10: generic Continue is for transient runtime/Relay/timeout
                    # states only and can never bypass a semantic rejection or spend a
                    # content-repair budget. A repair-authorization pause requires the
                    # exact one-cycle authorize control.
                    raise DomainError(
                        "run_not_controllable",
                        "Generic Continue cannot bypass a repair authorization.",
                        409,
                    )
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

    async def retry_run(
        self,
        *,
        creation_id: UUID,
        run_kind: RunKind,
        idempotency_key: str,
        now: datetime | None = None,
    ) -> RunControlAccepted:
        """Revive a terminally failed initial run after an operator-resolvable failure.

        The transition is the inverse of ``fail_run`` for the revivable-failure
        allowlist only: the run requeues with its original thread, approved
        business checkpoints, and attempt accounting intact, so the worker
        resumes through the same recovery path as a process restart.
        """
        timestamp = _timestamp(now or _utc_now())
        scope = f"run-control:{creation_id}:{run_kind}:retry"
        payload_hash = canonical_payload_hash({"action": "retry"})

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
            if run_kind != "initial":
                raise DomainError(
                    "run_not_controllable",
                    "A failed revision run is retried by resubmitting its frozen feedback.",
                    409,
                )
            if run["execution_state"] != "failed":
                raise DomainError(
                    "run_not_controllable",
                    "Only a failed workflow run can be retried.",
                    409,
                )
            if run["failure_code"] not in RETRYABLE_FAILURE_CODES:
                raise DomainError(
                    "run_not_controllable",
                    "The failure is not an operator-revivable failure code.",
                    409,
                )
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
                UPDATE runs
                SET state = 'queued',
                    failure_code = NULL,
                    failure_message = NULL,
                    failed_stage = NULL,
                    failure_attempt_count = NULL,
                    completed_at = NULL,
                    updated_at = ?
                WHERE id = ?
                """,
                (timestamp, run["id"]),
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
            response = RunControlAccepted(
                creation_id=creation_id,
                run_kind=run_kind,
                run_state="queued",
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
                          AND attempt_cycle = COALESCE(
                              (
                                  SELECT attempt_state.attempt_cycle
                                  FROM episode_attempt_current AS attempt_state
                                  WHERE attempt_state.run_id = episode_attempts.run_id
                                    AND attempt_state.episode_number = (
                                        episode_attempts.episode_number
                                    )
                              ),
                              0
                          )
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
        final_review_id: str | None = None,
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

            if final_review_id is not None:
                # RPR-A13: only a bound final whole-series PASS freezes formal delivery.
                review = await self._fetchone(
                    connection,
                    """
                    SELECT * FROM series_reviews
                    WHERE run_id = ? AND review_id = ?
                      AND review_type = 'final' AND status = 'active' AND passed = 1
                    """,
                    (str(run_id), final_review_id),
                )
                if review is None:
                    raise DomainError(
                        "final_review_required",
                        "Formal delivery requires a passing bound final whole-series review.",
                        409,
                    )
                active_batch = await self._fetch_script_batch_lineage(connection, run_id)
                if (
                    active_batch is None
                    or active_batch["batch_id"] != review["batch_id"]
                    or int(active_batch["batch_epoch"]) != int(review["batch_epoch"])
                ):
                    raise DomainError(
                        "final_review_required",
                        "The final review must bind the active script batch.",
                        409,
                    )

            story_checkpoint = await self._fetchone(
                connection,
                "SELECT payload_json FROM business_checkpoints WHERE run_id = ? AND stage = ?",
                (str(run_id), InternalStage.GENERATING_STORY_OUTLINE.value),
            )
            relationship_checkpoint = await self._fetchone(
                connection,
                "SELECT payload_json FROM business_checkpoints WHERE run_id = ? AND stage = ?",
                (str(run_id), InternalStage.GENERATING_CHARACTER_RELATIONSHIPS.value),
            )
            story_payload = json.loads(story_checkpoint["payload_json"]) if story_checkpoint else {}
            relationship_payload = (
                json.loads(relationship_checkpoint["payload_json"])
                if relationship_checkpoint
                else {}
            )
            presentation = compile_delivery_presentation(
                creation_id=UUID(run["creation_id"]),
                run_kind=run["kind"],
                content=delivery.content_package,
                story_hints=story_payload.get("story_navigation") or (),
                character_hints=relationship_payload.get("character_navigation") or (),
                relationship_hints=relationship_payload.get("relationship_navigation") or (),
                episode_plans=plans,
                episode_drafts=await self._episode_drafts(connection, run_id),
            )
            presentation_json = _json(presentation)
            await connection.execute(
                """
                INSERT INTO deliveries(
                    run_id, content_package_json, delivery_report_json,
                    presentation_manifest_json, presentation_manifest_sha256, created_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    str(run_id),
                    _json(delivery.content_package),
                    _json(delivery.delivery_report),
                    presentation_json,
                    _text_hash(presentation_json),
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

    async def get_delivery_presentation(
        self,
        creation_id: UUID,
        run_kind: Literal["initial", "revision"],
    ) -> DeliveryPresentation:
        async with self._connection() as connection:
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
                SELECT runs.id, runs.kind, runs.state,
                       deliveries.content_package_json,
                       deliveries.presentation_manifest_json,
                       deliveries.presentation_manifest_sha256
                FROM runs
                LEFT JOIN deliveries ON deliveries.run_id = runs.id
                WHERE runs.creation_id = ? AND runs.kind = ?
                ORDER BY runs.sequence {order}
                LIMIT 1
                """,
                (str(creation_id), run_kind),
            )
            if run is None or run["state"] != "succeeded" or run["content_package_json"] is None:
                raise DomainError(
                    "presentation_not_available",
                    "The requested run has no formal delivery to present.",
                    409,
                )
            content = ContentPackage.model_validate_json(run["content_package_json"])
            manifest_json = run["presentation_manifest_json"]
            if manifest_json is not None:
                try:
                    raw_manifest = json.loads(manifest_json)
                except (TypeError, ValueError):
                    raw_manifest = None
            else:
                raw_manifest = None
            run_id = UUID(run["id"])
            return recover_delivery_presentation(
                raw_manifest=raw_manifest,
                creation_id=creation_id,
                run_kind=run_kind,
                content=content,
                episode_plans=await self._episode_plans(connection, run_id),
                episode_drafts=await self._episode_drafts(connection, run_id),
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

    async def _reconcile_legacy_suffix_attempts(
        self,
        connection: aiosqlite.Connection,
        *,
        timestamp: str,
    ) -> list[str]:
        cursor = await connection.execute(
            """
            SELECT repair_authorizations.*,
                   runs.state AS run_state,
                   runs.failure_code,
                   runs.failed_stage,
                   run_progress.execution_state,
                   run_progress.current_episode
            FROM repair_authorizations
            JOIN runs ON runs.id = repair_authorizations.run_id
            JOIN run_progress ON run_progress.run_id = runs.id
            WHERE repair_authorizations.kind = 'suffix_rewrite'
              AND repair_authorizations.consumed_at IS NOT NULL
              AND runs.state = 'failed'
              AND runs.failure_code = 'attempts_exhausted'
              AND runs.failed_stage = ?
              AND run_progress.execution_state = 'failed'
            ORDER BY repair_authorizations.run_id,
                     repair_authorizations.authorization_epoch DESC
            """,
            (InternalStage.GENERATING_EPISODE_SCRIPTS.value,),
        )
        rows = await cursor.fetchall()
        recovered: list[str] = []
        seen_runs: set[str] = set()
        for auth in rows:
            run_id = auth["run_id"]
            if run_id in seen_runs:
                continue
            seen_runs.add(run_id)
            earliest = auth["earliest_affected_episode"]
            current_episode = auth["current_episode"]
            if earliest is None or current_episode is None:
                continue
            earliest = int(earliest)
            if int(current_episode) < earliest:
                continue

            run_uuid = UUID(run_id)
            lineage = await self._fetch_series_bible_lineage(connection, run_uuid)
            active_batch = await self._fetch_script_batch_lineage(connection, run_uuid)
            active_candidate = (
                await self._fetch_series_bible_candidate(
                    connection,
                    run_uuid,
                    lineage["active_candidate_id"],
                )
                if lineage is not None and lineage["active_candidate_id"] is not None
                else None
            )
            if (
                lineage is None
                or lineage["active_candidate_id"] is None
                or lineage["active_content_hash"] != auth["design_content_hash"]
                or active_candidate is None
                or active_candidate["status"] != "active"
                or active_candidate["candidate_id"] != auth["design_candidate_id"]
                or active_candidate["content_hash"] != auth["design_content_hash"]
                or int(active_candidate["design_epoch"]) != int(auth["design_epoch"])
                or active_batch is None
                or active_batch["status"] != "active"
                or active_batch["batch_id"] != auth["batch_id"]
                or int(active_batch["batch_epoch"]) != int(auth["batch_epoch"])
                or active_batch["design_candidate_id"] != auth["design_candidate_id"]
                or active_batch["design_content_hash"] != auth["design_content_hash"]
                or int(active_batch["design_epoch"]) != int(auth["design_epoch"])
            ):
                continue

            plan_cursor = await connection.execute(
                """
                SELECT episode_number
                FROM episode_plans
                WHERE run_id = ?
                ORDER BY episode_number
                """,
                (run_id,),
            )
            plan_numbers = [int(row["episode_number"]) for row in await plan_cursor.fetchall()]
            if not any(number >= earliest for number in plan_numbers):
                continue

            current_cycle = await self._fetchone(
                connection,
                """
                SELECT COALESCE(MAX(attempt_state.attempt_cycle), 0) AS attempt_cycle
                FROM episode_plans AS plans
                LEFT JOIN episode_attempt_current AS attempt_state
                  ON attempt_state.run_id = plans.run_id
                 AND attempt_state.episode_number = plans.episode_number
                WHERE plans.run_id = ? AND plans.episode_number >= ?
                """,
                (run_id, earliest),
            )
            if current_cycle is not None and int(current_cycle["attempt_cycle"]) != 0:
                continue

            attempt_after = await self._fetchone(
                connection,
                """
                SELECT 1
                FROM episode_attempts
                WHERE run_id = ?
                  AND episode_number >= ?
                  AND recorded_at > ?
                LIMIT 1
                """,
                (run_id, earliest, auth["consumed_at"]),
            )
            if attempt_after is not None:
                continue
            candidate_after = await self._fetchone(
                connection,
                """
                SELECT 1
                FROM episode_candidates
                WHERE run_id = ?
                  AND batch_id = ?
                  AND episode_number >= ?
                  AND status = 'active'
                  AND created_at > ?
                LIMIT 1
                """,
                (run_id, auth["batch_id"], earliest, auth["consumed_at"]),
            )
            if candidate_after is not None:
                continue

            await self._start_episode_attempt_cycle(
                connection,
                run_uuid,
                earliest,
                plan_numbers,
                timestamp=timestamp,
            )
            await connection.execute(
                """
                UPDATE runs
                SET state = 'queued',
                    failure_code = NULL,
                    failure_message = NULL,
                    failed_stage = NULL,
                    failure_attempt_count = NULL,
                    completed_at = NULL,
                    updated_at = ?
                WHERE id = ? AND state = 'failed'
                """,
                (timestamp, run_id),
            )
            await connection.execute(
                """
                UPDATE run_progress
                SET current_stage = ?,
                    execution_state = 'queued',
                    active_started_at = NULL,
                    timeout_stage = NULL,
                    timeout_count = 0,
                    recovery_reason = 'none',
                    content_repair_count = NULL,
                    pause_message = NULL,
                    updated_at = ?
                WHERE run_id = ? AND execution_state = 'failed'
                """,
                (InternalStage.GENERATING_EPISODE_SCRIPTS.value, timestamp, run_id),
            )
            await connection.execute(
                """
                UPDATE jobs
                SET state = 'queued',
                    available_at = ?,
                    lease_owner = NULL,
                    lease_expires_at = NULL,
                    updated_at = ?
                WHERE run_id = ? AND state = 'failed'
                """,
                (timestamp, timestamp, run_id),
            )
            run_creation = await self._fetchone(
                connection,
                "SELECT creation_id FROM runs WHERE id = ?",
                (run_id,),
            )
            if run_creation is not None:
                await connection.execute(
                    "UPDATE creations SET updated_at = ? WHERE id = ?",
                    (timestamp, run_creation["creation_id"]),
                )
            recovered.append(run_id)
        return recovered

    async def reconcile_startup(
        self,
        *,
        now: datetime | None = None,
    ) -> list[RunRecovery]:
        await self.requeue_expired_jobs(now=now)
        timestamp = _timestamp(now or _utc_now())
        async with self._connection() as connection:
            await connection.execute("BEGIN IMMEDIATE")
            try:
                await self._reconcile_legacy_suffix_attempts(
                    connection,
                    timestamp=timestamp,
                )
            except BaseException:
                await connection.rollback()
                raise
            else:
                await connection.commit()
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
                    authorization=await self._repair_authorization_snapshot(
                        connection,
                        UUID(run["id"]),
                        progress_row,
                    ),
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
                    authorization=await self._repair_authorization_snapshot(
                        connection,
                        UUID(run["id"]),
                        progress_row,
                    ),
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

    async def get_run_model_calls(self, run_id: UUID) -> list[ModelCallSummary]:
        async with self._connection() as connection:
            return await self._model_calls(connection, run_id)

    async def _model_calls(
        self,
        connection: aiosqlite.Connection,
        run_id: UUID,
    ) -> list[ModelCallSummary]:
        cursor = await connection.execute(
            """
            SELECT *
            FROM model_calls
            WHERE run_id = ?
            ORDER BY requested_at, call_id
            """,
            (str(run_id),),
        )
        rows = await cursor.fetchall()
        return [self._model_call_summary(row) for row in rows]

    @staticmethod
    def _model_call_summary(row: aiosqlite.Row) -> ModelCallSummary:
        response_model_ids: list[str] | None = None
        context_manifest: dict[str, Any] | None = None
        raw_response_model_ids = row["response_model_ids_json"]
        if raw_response_model_ids:
            try:
                parsed_response_model_ids = json.loads(raw_response_model_ids)
            except (TypeError, ValueError):
                parsed_response_model_ids = None
            if isinstance(parsed_response_model_ids, list):
                response_model_ids = [
                    value for value in parsed_response_model_ids if isinstance(value, str) and value
                ]
        raw_context_manifest = row["context_manifest_json"]
        if raw_context_manifest:
            try:
                parsed_context_manifest = json.loads(raw_context_manifest)
            except (TypeError, ValueError):
                parsed_context_manifest = None
            if isinstance(parsed_context_manifest, dict):
                context_manifest = parsed_context_manifest
        return ModelCallSummary(
            call_id=row["call_id"],
            operation_id=row["operation_id"],
            role=row["role"],
            adapter=row["adapter"],
            provider=row["provider"],
            model=row["model"],
            response_model_ids=response_model_ids,
            stage=row["stage"],
            episode_number=row["episode_number"],
            candidate=row["candidate"],
            batch=row["batch"],
            requested_at=_datetime(row["requested_at"]),
            finished_at=_datetime(row["finished_at"]) if row["finished_at"] else None,
            duration_seconds=row["duration_seconds"],
            estimated_input_tokens=row["estimated_input_tokens"],
            estimated_output_tokens=row["estimated_output_tokens"],
            estimated_total_tokens=row["estimated_total_tokens"],
            verified_limit_tokens=row["verified_limit_tokens"],
            preflight=row["preflight"],
            status=row["status"],
            usage=ModelCallUsage(
                input_tokens=row["actual_input_tokens"],
                output_tokens=row["actual_output_tokens"],
                cache_read_tokens=row["cache_read_tokens"],
                cache_creation_tokens=row["cache_creation_tokens"],
                status=row["usage_status"],
            ),
            finish_reason=row["finish_reason"],
            outcome=row["outcome"],
            error_code=row["error_code"],
            error_type=row["error_type"],
            safe_message=row["safe_message"],
            supersedes_call_id=row["supersedes_call_id"],
            context_bundle_sha256=row["context_bundle_sha256"],
            context_manifest=context_manifest,
        )

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
        outline_groups = (
            await self._outline_group_progress(connection, UUID(run["id"]))
            if current_internal is InternalStage.GENERATING_EPISODE_OUTLINE
            else None
        )
        model_calls = await self._model_calls(connection, UUID(run["id"]))
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
            outline_groups=outline_groups,
            model_calls=model_calls,
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
            can_retry=(
                execution_state == "failed"
                and run["kind"] == "initial"
                and run["failure_code"] in RETRYABLE_FAILURE_CODES
                and await self._has_remaining_attempts(
                    connection,
                    run_id=progress["run_id"],
                    current_stage=progress["current_stage"],
                    current_episode=progress["current_episode"],
                )
            ),
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
            design=await self._run_design_snapshot(connection, run_id),
            review_status=progress.final_review,
        )

    async def _run_design_snapshot(
        self,
        connection: aiosqlite.Connection,
        run_id: UUID,
    ) -> SeriesBibleSummary | None:
        lineage = await self._fetch_series_bible_lineage(connection, run_id)
        if lineage is None or lineage["active_candidate_id"] is None:
            return None
        candidate = await self._fetch_series_bible_candidate(
            connection,
            run_id,
            lineage["active_candidate_id"],
        )
        if candidate is None:
            return None
        return project_series_bible(self._series_bible_from_row(candidate), is_active=True)

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

    async def _outline_group_progress(
        self,
        connection: aiosqlite.Connection,
        run_id: UUID,
    ) -> OutlineGroupProgress | None:
        cursor = await connection.execute(
            """
            SELECT position, start_episode, end_episode, status
            FROM outline_generation_groups
            WHERE run_id = ?
            ORDER BY position
            """,
            (str(run_id),),
        )
        rows = await cursor.fetchall()
        if not rows:
            return None
        committed = [row for row in rows if row["status"] == "committed"]
        in_flight = next((row for row in rows if row["status"] != "committed"), None)
        return OutlineGroupProgress(
            committed_groups=len(committed),
            committed_through_episode=max(
                (int(row["end_episode"]) for row in committed), default=0
            ),
            current_group=int(in_flight["position"]) if in_flight is not None else None,
            current_start_episode=(
                int(in_flight["start_episode"]) if in_flight is not None else None
            ),
            current_end_episode=(int(in_flight["end_episode"]) if in_flight is not None else None),
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
            if stage is UserStage.GENERATING_CHARACTER_RELATIONSHIPS:
                # The character+relationships payload carries two content fields; render
                # them as one readable artifact for the draft snapshot, mirroring the
                # flattened candidate the agent reviews and repairs.
                content = _flatten_cr_draft(
                    character_biographies=payload["character_biographies"],
                    relationship_logic=payload["relationship_logic"],
                )
                return CreativeTextDraft(stage=stage.value, content=content)
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
        repair = await self._fetchone(
            connection,
            """
            SELECT plan_json, status
            FROM quality_gate_repairs
            WHERE run_id = ? AND stage = ? AND rejection_attempt = ?
            """,
            (str(run_id), stage.value, int(rejection["attempt_number"])),
        )
        can_repair = int(rejection["attempt_number"]) < MAX_STAGE_ATTEMPTS
        plan = (
            QualityRepairPlan.model_validate_json(repair["plan_json"])
            if repair is not None and repair["plan_json"] is not None
            else None
        )
        return QualityGateRejection(
            stage=rejection["stage"],
            evidence=rejection["evidence"],
            attempt_count=int(rejection["attempt_number"]),
            can_retry=can_repair,
            repair_plan=plan,
            repair_state=(repair["status"] if repair is not None else "available")
            if can_repair
            else None,
        )

    async def _repair_authorization_snapshot(
        self,
        connection: aiosqlite.Connection,
        run_id: UUID,
        progress: aiosqlite.Row,
    ) -> RepairAuthorization | None:
        if progress["recovery_reason"] != "repair_authorization":
            return None
        auth = await self._fetchone(
            connection,
            "SELECT * FROM repair_authorizations "
            "WHERE run_id = ? ORDER BY authorization_epoch DESC LIMIT 1",
            (str(run_id),),
        )
        if auth is None:
            return None
        return RepairAuthorization(
            authorization_epoch=int(auth["authorization_epoch"]),
            kind=auth["kind"],
            design_candidate_id=auth["design_candidate_id"],
            design_content_hash=auth["design_content_hash"],
            design_epoch=int(auth["design_epoch"]),
            batch_id=auth["batch_id"],
            batch_epoch=int(auth["batch_epoch"]),
            earliest_affected_episode=auth["earliest_affected_episode"],
            range_episodes=auth["range_episodes"],
            estimated_tokens=auth["estimated_tokens"],
            evidence=auth["evidence"],
            review_id=auth["review_id"],
            granted_at=(_datetime(auth["granted_at"]) if auth["granted_at"] else None),
            consumed_at=(_datetime(auth["consumed_at"]) if auth["consumed_at"] else None),
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
            "context_budget",
            "relay_identity_mismatch",
            "repair_authorization",
        }:
            raise RuntimeError("Paused workflow run is missing its recovery reason")
        if recovery_reason == "repair_authorization":
            if not progress["pause_message"]:
                raise RuntimeError("Repair-authorization pause is missing its evidence")
            return RunPause(
                message=progress["pause_message"],
                code="repair_authorization",
                stage=stage,
            )
        if recovery_reason == "context_budget":
            if not progress["pause_message"]:
                raise RuntimeError("Context-budget pause is missing safe recovery evidence")
            return RunPause(
                message=progress["pause_message"],
                code="context_budget",
                stage=stage,
                episode_number=(
                    int(progress["current_episode"])
                    if progress["current_episode"] is not None
                    else None
                ),
            )
        if recovery_reason == "relay_identity_mismatch":
            if not progress["pause_message"]:
                raise RuntimeError("Relay-identity pause is missing verification evidence")
            return RunPause(
                message=progress["pause_message"],
                code="relay_identity_mismatch",
                stage=stage,
                episode_number=(
                    int(progress["current_episode"])
                    if progress["current_episode"] is not None
                    else None
                ),
            )
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
                  AND attempt_cycle = COALESCE(
                      (
                          SELECT attempt_state.attempt_cycle
                          FROM episode_attempt_current AS attempt_state
                          WHERE attempt_state.run_id = episode_attempts.run_id
                            AND attempt_state.episode_number = episode_attempts.episode_number
                      ),
                      0
                  )
                """,
                (run_id, current_episode),
            )
            row = await cursor.fetchone()
            return int(row[0]) < MAX_EPISODE_ATTEMPTS
        if stage is InternalStage.GENERATING_EPISODE_OUTLINE:
            season_map = await Repository._fetchone(
                connection,
                "SELECT 1 FROM outline_season_maps WHERE run_id = ?",
                (run_id,),
            )
            if season_map is not None:
                return True
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
    async def _current_episode_attempt_cycle(
        connection: aiosqlite.Connection,
        run_id: UUID | str,
        episode_number: int,
    ) -> int:
        row = await Repository._fetchone(
            connection,
            """
            SELECT attempt_cycle
            FROM episode_attempt_current
            WHERE run_id = ? AND episode_number = ?
            """,
            (str(run_id), episode_number),
        )
        return int(row["attempt_cycle"]) if row is not None else 0

    @staticmethod
    async def _ensure_episode_attempt_current(
        connection: aiosqlite.Connection,
        run_id: UUID | str,
        episode_number: int,
        *,
        timestamp: str,
    ) -> int:
        current = await Repository._fetchone(
            connection,
            """
            SELECT attempt_cycle
            FROM episode_attempt_current
            WHERE run_id = ? AND episode_number = ?
            """,
            (str(run_id), episode_number),
        )
        if current is not None:
            return int(current["attempt_cycle"])

        cycle_row = await Repository._fetchone(
            connection,
            """
            SELECT MAX(attempt_cycle) AS attempt_cycle
            FROM episode_attempt_cycles
            WHERE run_id = ?
            """,
            (str(run_id),),
        )
        attempt_cycle = (
            int(cycle_row["attempt_cycle"])
            if cycle_row is not None and cycle_row["attempt_cycle"] is not None
            else 0
        )
        boundary = await Repository._fetchone(
            connection,
            """
            SELECT MIN(episode_number) AS from_episode,
                   MAX(episode_number) AS to_episode
            FROM episode_plans
            WHERE run_id = ?
            """,
            (str(run_id),),
        )
        from_episode = (
            int(boundary["from_episode"])
            if boundary is not None and boundary["from_episode"] is not None
            else episode_number
        )
        to_episode = (
            max(int(boundary["to_episode"]), episode_number)
            if boundary is not None and boundary["to_episode"] is not None
            else episode_number
        )
        await connection.execute(
            """
            INSERT OR IGNORE INTO episode_attempt_cycles(
                run_id, attempt_cycle, from_episode, to_episode, started_at
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (str(run_id), attempt_cycle, from_episode, to_episode, timestamp),
        )
        await connection.execute(
            """
            INSERT INTO episode_attempt_current(run_id, episode_number, attempt_cycle)
            VALUES (?, ?, ?)
            """,
            (str(run_id), episode_number, attempt_cycle),
        )
        return attempt_cycle

    @staticmethod
    async def _start_episode_attempt_cycle(
        connection: aiosqlite.Connection,
        run_id: UUID | str,
        from_episode: int,
        plan_numbers: list[int],
        *,
        timestamp: str,
    ) -> int:
        affected = [number for number in plan_numbers if number >= from_episode]
        if not affected:
            raise DomainError(
                "episode_not_planned",
                "The rewrite target is not in the approved outline.",
                409,
            )
        await connection.execute(
            """
            INSERT OR IGNORE INTO episode_attempt_cycles(
                run_id, attempt_cycle, from_episode, to_episode, started_at
            ) VALUES (?, 0, ?, ?, ?)
            """,
            (str(run_id), min(plan_numbers), max(plan_numbers), timestamp),
        )
        await connection.executemany(
            """
            INSERT OR IGNORE INTO episode_attempt_current(
                run_id, episode_number, attempt_cycle
            ) VALUES (?, ?, 0)
            """,
            [(str(run_id), number) for number in plan_numbers],
        )
        cycle_row = await Repository._fetchone(
            connection,
            """
            SELECT MAX(attempt_cycle) AS attempt_cycle
            FROM episode_attempt_cycles
            WHERE run_id = ?
            """,
            (str(run_id),),
        )
        latest_cycle = (
            int(cycle_row["attempt_cycle"])
            if cycle_row is not None and cycle_row["attempt_cycle"] is not None
            else 0
        )
        attempt_cycle = latest_cycle + 1
        await connection.execute(
            """
            INSERT INTO episode_attempt_cycles(
                run_id, attempt_cycle, from_episode, to_episode, started_at
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (str(run_id), attempt_cycle, from_episode, max(affected), timestamp),
        )
        await connection.executemany(
            """
            INSERT INTO episode_attempt_current(run_id, episode_number, attempt_cycle)
            VALUES (?, ?, ?)
            ON CONFLICT(run_id, episode_number) DO UPDATE SET
                attempt_cycle = excluded.attempt_cycle
            """,
            [(str(run_id), number, attempt_cycle) for number in affected],
        )
        await connection.execute(
            """
            DELETE FROM episode_timeouts
            WHERE run_id = ? AND episode_number >= ?
            """,
            (str(run_id), from_episode),
        )
        return attempt_cycle

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
                runs.failure_code,
                run_progress.current_stage,
                run_progress.current_episode,
                run_progress.execution_state,
                run_progress.recovery_reason
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
