"""Bound structural milestone and final reviews with repair-budget semantics.

A structural review binds one DeepSeek review to the exact design candidate,
script batch/epoch, complete active-prefix hash, and model-call id that it
observed. It carries a deterministic category (pass / design defect / script
defect) and, for a script defect, the earliest affected episode N. Protocol and
transient failures never produce a semantic rejection; they surface as
exceptions and never consume design or suffix content budgets. A late review
from an inactive lineage is retained as stale evidence and can never approve,
rebuild, rewrite, or deliver the active lineage.

The module intentionally depends only on :mod:`pengine.continuity` so the
repository and worker can consume it without a relay cycle.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from typing import Literal
from uuid import uuid4

from pydantic import Field, model_validator

from pengine.continuity import ContinuityModel, NonEmptyText, Sha256, StableId

StructuralReviewCategory = Literal[
    "pass",
    "design_defect",
    "script_defect",
    "protocol_failure",
    "transient_failure",
    "stale",
]
ReviewType = Literal["milestone", "final"]


def _utc_now() -> datetime:
    return datetime.now(UTC)


def new_review_id() -> StableId:
    return f"series_review_{uuid4().hex}"


def active_prefix_hash(episodes: Sequence[Mapping[str, str]]) -> str:
    """Deterministic SHA-256 of the complete active prefix in episode order.

    The prefix identity is the ordered sequence of active episode content hashes
    (episode number + content sha256), so a rewrite of any episode in the prefix
    changes the bound review identity (RPR-A1/A11).
    """
    encoded = json.dumps(
        [
            {
                "episode_number": int(item["episode_number"]),
                "content_sha256": item["content_sha256"],
            }
            for item in episodes
        ],
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


class StructuralReviewResult(ContinuityModel):
    """The structured DeepSeek review decision for one milestone or final prefix."""

    passed: bool = Field(
        description=(
            "False only for a direct explicit hard-Canon contradiction, an impossible required "
            "locked binding, or a proven private-runtime leak in the current reviewed prefix."
        )
    )
    category: Literal["pass", "design_defect", "script_defect"] = Field(
        description=(
            "design_defect only when the active design itself contains the blocker; "
            "script_defect only when the current script prefix contains it."
        )
    )
    evidence: NonEmptyText = Field(
        description=(
            "On pass, state the checked hard-Canon scope and that no blocker exists. On failure, "
            "list every current blocker with the conflicting screenplay or design excerpt and "
            "its explicit authoritative source; for a runtime leak, also name the matching "
            "private source."
        )
    )
    earliest_affected_episode: int | None = Field(
        default=None,
        ge=1,
        description=(
            "The earliest episode containing a current script blocker; never derive this from "
            "format, style, or a prior superseded prefix."
        ),
    )

    @model_validator(mode="after")
    def validate_decision(self) -> StructuralReviewResult:
        if self.passed:
            if self.category != "pass" or self.earliest_affected_episode is not None:
                raise ValueError(
                    "A passing structural review cannot carry a defect category or episode"
                )
            return self
        if self.category not in {"design_defect", "script_defect"}:
            raise ValueError("A failed structural review requires a defect category")
        if self.category == "design_defect":
            if self.earliest_affected_episode is not None:
                raise ValueError("A design defect cannot name an affected episode")
            return self
        if self.earliest_affected_episode is None:
            raise ValueError("A script defect requires the earliest affected episode")
        return self


class BoundStructuralReview(ContinuityModel):
    """One immutable structural review bound to the lineage it observed."""

    review_id: StableId
    run_id: NonEmptyText
    review_epoch: int = Field(ge=1)
    review_type: ReviewType
    episode_number: int = Field(ge=1)
    design_candidate_id: StableId
    design_content_hash: Sha256
    design_epoch: int = Field(ge=1)
    batch_id: StableId
    batch_epoch: int = Field(ge=1)
    prefix_hash: Sha256
    call_id: NonEmptyText
    passed: bool
    category: StructuralReviewCategory
    evidence: NonEmptyText
    earliest_affected_episode: int | None = Field(default=None, ge=1)
    status: Literal["active", "superseded", "stale"] = "active"
    reviewed_at: datetime
    consumed_at: datetime | None = None


def effective_milestones(declared: Sequence[int], episode_count: int) -> list[int]:
    """The review schedule: declared milestones plus the mandatory final episode.

    The final completion review is always a structural milestone (RPR-A1/A2).
    """
    if episode_count < 1:
        raise ValueError("A review schedule requires at least one episode")
    cleaned = [int(item) for item in declared]
    invalid = [item for item in cleaned if item < 1 or item > episode_count]
    if invalid:
        raise ValueError("Review milestones must lie within the episode range")
    return sorted(set(cleaned) | {episode_count})
