"""Versioned episode candidates and design-bound script-batch lineage.

One design-bound script batch owns immutable episode candidate versions and one
active pointer per episode. A candidate becomes active only through a
transactional/CAS commit that validates the exact design identity, batch epoch,
predecessor pointer, and deterministic contract/state replay. Rewriting from
episode N preserves 1..N-1 and supersedes N..end with SeriesState replay; a
design hash change supersedes the whole batch and starts a fresh batch at
episode 1. A late generation from an inactive design, batch, epoch,
predecessor, or suffix is retained as stale evidence and can never move an
active pointer.

The module intentionally depends only on :mod:`pengine.continuity` so the
repository and worker can consume it without a relay cycle.
"""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from typing import Literal
from uuid import uuid4

from pydantic import Field, model_validator

from pengine.continuity import (
    ContinuityModel,
    EpisodeStateDelta,
    NonEmptyText,
    SemanticReview,
    SeriesState,
    Sha256,
    StableId,
    StoryContract,
    build_episode_lock,
    canonical_model_hash,
    story_contract_sha256,
)

ScriptBatchStatus = Literal["active", "superseded"]
EpisodeCandidateStatus = Literal["unvalidated", "validated", "active", "superseded", "stale"]
RunKind = Literal["initial", "revision"]


def _utc_now() -> datetime:
    return datetime.now(UTC)


def new_batch_id() -> StableId:
    return f"script_batch_{uuid4().hex}"


def new_candidate_id() -> StableId:
    return f"episode_candidate_{uuid4().hex}"


def _content_hash(content: str) -> str:
    return hashlib.sha256(content.encode()).hexdigest()


class ScriptBatchLineage(ContinuityModel):
    """One design-bound script batch with one active pointer per episode.

    The batch binds the exact immutable design candidate that authored it. Its
    ``active_pointers`` map every episode number to the single active candidate
    version that may feed later episodes and assembly.
    """

    batch_id: StableId
    run_id: NonEmptyText
    run_kind: RunKind
    batch_epoch: int = Field(ge=1)
    status: ScriptBatchStatus = "active"
    design_candidate_id: StableId
    design_content_hash: Sha256
    design_epoch: int = Field(ge=1)
    active_pointers: dict[int, StableId] = Field(default_factory=dict)
    created_at: datetime
    superseded_at: datetime | None = None

    def active_episodes(self) -> list[int]:
        return sorted(self.active_pointers)


class EpisodeCandidate(ContinuityModel):
    """One immutable versioned episode candidate bound to a design and batch.

    Every candidate binds the exact design candidate/hash/epoch, script batch and
    epoch, episode number, candidate version, predecessor hash, call id, writer
    notes, state delta, and folded series state (FSW-A3).
    """

    candidate_id: StableId
    batch_id: StableId
    batch_epoch: int = Field(ge=1)
    run_id: NonEmptyText
    design_candidate_id: StableId
    design_content_hash: Sha256
    design_epoch: int = Field(ge=1)
    episode_number: int = Field(ge=1)
    version: int = Field(ge=1)
    content: NonEmptyText
    content_sha256: Sha256
    predecessor_candidate_id: StableId | None = None
    predecessor_sha256: Sha256 | None = None
    call_id: NonEmptyText
    generation_window_id: StableId | None = None
    writer_notes: str = ""
    state_delta: EpisodeStateDelta
    series_state: SeriesState
    series_state_sha256: Sha256
    semantic_review: SemanticReview
    repair_rounds: int = Field(ge=0, le=2)
    status: EpisodeCandidateStatus = "unvalidated"
    created_at: datetime
    activated_at: datetime | None = None
    superseded_at: datetime | None = None

    @model_validator(mode="after")
    def validate_candidate_identity(self) -> EpisodeCandidate:
        if _content_hash(self.content) != self.content_sha256:
            raise ValueError("Episode candidate content hash does not match its content")
        if canonical_model_hash(self.series_state) != self.series_state_sha256:
            raise ValueError("Episode candidate series state hash does not match its state")
        if self.status == "stale":
            # Stale evidence retains a generation that never advanced the pointer;
            # its predecessor lineage may be intentionally partial.
            return self
        if self.episode_number == 1:
            if self.predecessor_candidate_id is not None or self.predecessor_sha256 is not None:
                raise ValueError("The first episode cannot declare a predecessor")
        elif self.predecessor_candidate_id is None or self.predecessor_sha256 is None:
            raise ValueError("A later episode requires a predecessor candidate")
        return self


def build_episode_candidate(
    *,
    run_id: str,
    run_kind: RunKind,
    batch: ScriptBatchLineage,
    contract: StoryContract,
    prior_state: SeriesState,
    episode_number: int,
    version: int,
    predecessor_candidate_id: StableId | None,
    predecessor_sha256: Sha256 | None,
    content: str,
    delta: EpisodeStateDelta,
    semantic_review: SemanticReview,
    repair_rounds: int,
    call_id: str,
    generation_window_id: StableId | None = None,
    writer_notes: str,
    now: datetime | None = None,
) -> EpisodeCandidate:
    """Build one deterministic candidate bound to an active batch.

    Deterministic contract/state validation runs through :func:`build_episode_lock`
    against the retained prefix; any failure raises :class:`ContinuityViolation`
    before the candidate can be promoted. The returned candidate is immutable and
    ``unvalidated`` until the repository commits it.
    """
    if episode_number != delta.episode_number:
        raise ValueError("Candidate episode must match its state delta episode")
    contract_hash = story_contract_sha256(contract)
    lock = build_episode_lock(
        contract=contract,
        contract_sha256=contract_hash,
        prior_state=prior_state,
        content=content,
        delta=delta,
        semantic_review=semantic_review,
        repair_rounds=repair_rounds,
    )
    return EpisodeCandidate(
        candidate_id=new_candidate_id(),
        batch_id=batch.batch_id,
        batch_epoch=batch.batch_epoch,
        run_id=run_id,
        design_candidate_id=batch.design_candidate_id,
        design_content_hash=batch.design_content_hash,
        design_epoch=batch.design_epoch,
        episode_number=episode_number,
        version=version,
        content=lock.content,
        content_sha256=lock.content_sha256,
        predecessor_candidate_id=predecessor_candidate_id,
        predecessor_sha256=predecessor_sha256,
        call_id=call_id,
        generation_window_id=generation_window_id,
        writer_notes=writer_notes,
        state_delta=lock.state_delta,
        series_state=lock.series_state,
        series_state_sha256=lock.series_state_sha256,
        semantic_review=lock.semantic_review,
        repair_rounds=lock.repair_rounds,
        status="unvalidated",
        created_at=now or _utc_now(),
    )
