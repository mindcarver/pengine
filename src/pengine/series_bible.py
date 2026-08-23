"""Versioned hybrid SeriesBible aggregate, validation, and projections.

The SeriesBible is the atomic design package for one run lineage. It bundles the
story outline, character biographies, relationship logic, and episode outline
Markdown projections with the machine-readable :class:`StoryContract` into one
immutable candidate. A candidate is promoted (``active``) only after complete
deterministic validation and a configured global design review bound to its
candidate id and content hash. Old candidates and reviews remain immutable
evidence; a confirmed design defect triggers at most one automatic complete
rebuild per run lineage.

The module intentionally depends only on :mod:`pengine.continuity` so the
repository and worker can consume it without a relay cycle.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any, Literal
from uuid import uuid4

from pydantic import Field, ValidationError, model_validator

from pengine.continuity import (
    ContinuityModel,
    NonEmptyText,
    ReviewIssue,
    Sha256,
    StableId,
    StoryContract,
    canonical_story_contract_payload,
    character_label_base,
    render_story_contract_markdown,
    story_contract_sha256,
)

SeriesBibleGenre = Literal["mystery", "general"]
SeriesBibleStatus = Literal["unvalidated", "validated", "active", "superseded", "stale"]
MAX_SCRIPT_GENERATION_GROUP_EPISODES = 4

_MYSTERY_TERMS = (
    "悬疑",
    "推理",
    "侦探",
    "谜题",
    "案件",
    "凶案",
    "罪案",
    "凶手",
    "线索",
    "证据",
    "动机",
)

# Rules that only apply when the SeriesBible declares the mystery genre. A
# general-genre idea is never rejected for missing mystery-only clues or reveal
# mechanics (SDP-A4).
_MYSTERY_ACTIVATED_RULES = frozenset(
    {
        "mystery_reveal_required",
    }
)
_UNIVERSAL_RULES = frozenset(
    {
        "schema",
        "reference",
        "uniqueness",
        "order",
        "arithmetic",
        "projection",
    }
)

# Static rule-issue messages that are safe to surface through the API and logs.
_SAFE_VALIDATION_MESSAGES = frozenset(
    {
        "Duplicate character identifiers are not allowed",
        "Duplicate fact identifiers are not allowed",
        "Duplicate clue identifiers are not allowed",
        "Duplicate timeline event identifiers are not allowed",
        "Duplicate episode obligation identifiers are not allowed",
        "Fact, clue, and obligation IDs must be globally unique",
        "Character names must be unique",
        "Character knowledge references unknown identifiers",
        "Timeline events must be ordered and contiguous from 1",
        "Every episode requires exactly one obligation",
        "Numeric facts require an exact decimal value",
        "Numeric facts require a finite value and explicit unit",
        "Projection does not include every contract character biography",
        "A mystery candidate must declare and resolve reveal mechanics",
    }
)


def _utc_now() -> datetime:
    return datetime.now(UTC)


def detect_genre(story: str, requirements: str) -> SeriesBibleGenre:
    """Deterministically derive the declared genre from the free-form request."""
    text = f"{story}\n{requirements}"
    return "mystery" if any(term in text for term in _MYSTERY_TERMS) else "general"


def new_candidate_id() -> StableId:
    return f"candidate_{uuid4().hex}"


class DesignLineage(ContinuityModel):
    """Run-lineage identity and automatic-rebuild budget of one candidate."""

    run_id: NonEmptyText
    run_kind: Literal["initial", "revision"]
    parent_candidate_id: StableId | None = None
    rebuild_count: int = Field(default=0, ge=0, le=1)

    @model_validator(mode="after")
    def validate_rebuild_budget(self) -> DesignLineage:
        if self.rebuild_count > 0 and self.parent_candidate_id is None:
            raise ValueError("A rebuilt candidate requires a parent candidate")
        return self


class ScriptGenerationGroup(ContinuityModel):
    """One outline-authored dramatic unit generated in a single writer operation."""

    group_id: StableId
    start_episode: int = Field(ge=1)
    end_episode: int = Field(ge=1)
    dramatic_unit: NonEmptyText
    boundary_reason: NonEmptyText

    @model_validator(mode="after")
    def validate_range(self) -> ScriptGenerationGroup:
        if self.end_episode < self.start_episode:
            raise ValueError("Script generation group end must not precede its start")
        if self.end_episode - self.start_episode + 1 > MAX_SCRIPT_GENERATION_GROUP_EPISODES:
            raise ValueError(
                "Script generation groups may contain at most "
                f"{MAX_SCRIPT_GENERATION_GROUP_EPISODES} episodes"
            )
        return self


def validate_script_generation_groups(
    groups: list[ScriptGenerationGroup],
    *,
    episode_count: int,
    review_milestones: list[int],
    allow_empty: bool,
) -> None:
    """Validate complete ordered coverage and hard structural-review boundaries."""
    if not groups:
        if allow_empty:
            return
        raise ValueError("Episode outline must declare script generation groups")
    group_ids = [group.group_id for group in groups]
    if len(group_ids) != len(set(group_ids)):
        raise ValueError("Script generation group IDs must be unique")
    expected_start = 1
    milestones = set(review_milestones)
    for group in groups:
        if group.start_episode != expected_start:
            raise ValueError("Script generation groups must continuously cover episodes from 1")
        if group.end_episode > episode_count:
            raise ValueError("Script generation group exceeds the episode count")
        if any(group.start_episode <= milestone < group.end_episode for milestone in milestones):
            raise ValueError("Script generation group cannot cross a review milestone")
        expected_start = group.end_episode + 1
    if expected_start != episode_count + 1:
        raise ValueError("Script generation groups must cover every episode exactly once")


class SeriesBibleContent(ContinuityModel):
    """The complete creative content of one design candidate."""

    story_outline: NonEmptyText
    character_biographies: NonEmptyText
    relationship_logic: NonEmptyText
    episode_outline: NonEmptyText
    story_contract: StoryContract
    review_milestones: list[int] = Field(
        default_factory=list,
        description=(
            "SeriesBible-declared structural review milestone episode numbers. Empty means the "
            "only structural review is the final completion review."
        ),
    )
    script_generation_groups: list[ScriptGenerationGroup] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_milestones(self) -> SeriesBibleContent:
        milestones = [int(item) for item in self.review_milestones]
        if len(milestones) != len(set(milestones)):
            raise ValueError("Review milestones must be unique")
        for milestone in milestones:
            if milestone < 1 or milestone > self.story_contract.episode_count:
                raise ValueError("Review milestones must lie within the episode count")
        self.review_milestones = sorted(milestones)
        validate_script_generation_groups(
            self.script_generation_groups,
            episode_count=self.story_contract.episode_count,
            review_milestones=self.review_milestones,
            allow_empty=True,
        )
        return self


class ValidationIssue(ContinuityModel):
    code: StableId
    message: NonEmptyText
    refs: list[StableId] = Field(default_factory=list)


class ValidationEvidence(ContinuityModel):
    passed: bool
    validator_version: int = Field(default=1, ge=1)
    issues: list[ValidationIssue] = Field(default_factory=list)
    validated_at: datetime


class GlobalDesignReview(ContinuityModel):
    """One global design review bound to a single candidate."""

    review_call_id: NonEmptyText
    candidate_id: StableId
    candidate_hash: Sha256
    review_model_id: NonEmptyText
    passed: bool
    evidence: NonEmptyText
    issues: list[ReviewIssue] = Field(default_factory=list)
    reviewed_at: datetime

    @model_validator(mode="after")
    def validate_review_decision(self) -> GlobalDesignReview:
        if self.passed and self.issues:
            raise ValueError("A passing global design review cannot contain issues")
        if not self.passed and not self.issues:
            raise ValueError("A failed global design review requires at least one issue")
        return self


class SeriesBible(ContinuityModel):
    """One immutable versioned SeriesBible design candidate."""

    candidate_id: StableId
    version: int = Field(default=1, ge=1)
    design_epoch: int = Field(ge=1)
    content_hash: Sha256
    status: SeriesBibleStatus = "unvalidated"
    l0_variant: NonEmptyText
    genre: SeriesBibleGenre = "general"
    lineage: DesignLineage
    content: SeriesBibleContent
    validation: ValidationEvidence | None = None
    global_review: GlobalDesignReview | None = None
    created_at: datetime

    @model_validator(mode="after")
    def validate_hash_and_status(self) -> SeriesBible:
        expected_hash = canonical_series_bible_content_hash(self.content)
        if self.content_hash != expected_hash:
            raise ValueError("SeriesBible content hash does not match its content")
        if self.status == "active" and (self.validation is None or not self.validation.passed):
            raise ValueError("An active candidate requires passing deterministic validation")
        if self.status == "active" and (
            self.global_review is None or not self.global_review.passed
        ):
            raise ValueError("An active candidate requires a passing bound global design review")
        return self


class SeriesBibleProjections(ContinuityModel):
    """Readable projections exposed to the workbench from one active candidate."""

    story_outline: NonEmptyText
    character_biographies: NonEmptyText
    relationship_logic: NonEmptyText
    episode_outline: NonEmptyText
    story_contract_markdown: NonEmptyText
    review_milestones: list[int] = Field(default_factory=list)
    script_generation_groups: list[ScriptGenerationGroup] = Field(default_factory=list)


class SeriesBibleSummary(ContinuityModel):
    """Durable API/UI projection of the latest active design candidate."""

    candidate_id: StableId
    version: int = Field(ge=1)
    design_epoch: int = Field(ge=1)
    content_hash: Sha256
    status: SeriesBibleStatus
    is_active: bool
    unfinished: Literal[True] = True
    l0_variant: NonEmptyText
    genre: SeriesBibleGenre
    lineage: DesignLineage
    projections: SeriesBibleProjections
    review_milestones: list[int] = Field(default_factory=list)
    script_generation_groups: list[ScriptGenerationGroup] = Field(default_factory=list)
    validation: ValidationEvidence | None = None
    global_review: GlobalDesignReview | None = None
    created_at: datetime


def canonical_series_bible_content_hash(content: SeriesBibleContent) -> str:
    """SHA-256 of the canonical serialized design content (immutable identity)."""
    payload = content.model_dump(mode="json")
    payload["story_contract"] = canonical_story_contract_payload(content.story_contract)
    if not payload["script_generation_groups"]:
        # Historical candidates predate outline-authored generation groups. Keep
        # their immutable content hashes readable while new designs hash the field.
        payload.pop("script_generation_groups")
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def canonical_series_bible_hash(bible: SeriesBible) -> str:
    """Candidate-level SHA-256 combining content identity and lineage metadata."""
    payload = {
        "content_hash": bible.content_hash,
        "design_epoch": bible.design_epoch,
        "version": bible.version,
        "lineage": bible.lineage.model_dump(mode="json"),
    }
    encoded = json.dumps(
        payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def build_series_bible(
    *,
    run_id: str,
    run_kind: Literal["initial", "revision"],
    l0_variant: str,
    genre: SeriesBibleGenre,
    story_outline: str,
    character_biographies: str,
    relationship_logic: str,
    episode_outline: str,
    story_contract_payload: Mapping[str, Any],
    parent_candidate_id: StableId | None = None,
    rebuild_count: int = 0,
    design_epoch: int | None = None,
    candidate_id: StableId | None = None,
    review_milestones: list[int] | None = None,
    script_generation_groups: list[Mapping[str, Any]] | None = None,
    now: datetime | None = None,
) -> SeriesBible:
    """Build one immutable SeriesBible candidate from complete content.

    Constructing the candidate validates the story contract schema, stable
    identifiers, references, uniqueness, ordering, and typed arithmetic through
    the underlying :class:`StoryContract` model. Any structural violation raises
    :class:`pydantic.ValidationError` before the candidate can be reviewed or
    promoted. ``review_milestones`` declares the SeriesBible structural review
    schedule; an empty list leaves only the final completion review.
    """
    contract = StoryContract.model_validate(story_contract_payload)
    content = SeriesBibleContent(
        story_outline=story_outline,
        character_biographies=character_biographies,
        relationship_logic=relationship_logic,
        episode_outline=episode_outline,
        story_contract=contract,
        review_milestones=review_milestones or [],
        script_generation_groups=script_generation_groups or [],
    )
    return SeriesBible(
        candidate_id=candidate_id or new_candidate_id(),
        version=1,
        design_epoch=design_epoch or (2 if parent_candidate_id is not None else 1),
        content_hash=canonical_series_bible_content_hash(content),
        status="unvalidated",
        l0_variant=l0_variant,
        genre=genre,
        lineage=DesignLineage(
            run_id=run_id,
            run_kind=run_kind,
            parent_candidate_id=parent_candidate_id,
            rebuild_count=rebuild_count,
        ),
        content=content,
        created_at=now or _utc_now(),
    )


def _schema_validation_issues(error: ValidationError) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    for item in error.errors(include_url=False, include_input=False, include_context=False)[:8]:
        loc = ".".join(str(part) for part in item["loc"])
        raw_message = str(item.get("msg", "")).removeprefix("Value error, ")
        message = raw_message if raw_message in _SAFE_VALIDATION_MESSAGES else str(item.get("type"))
        issues.append(ValidationIssue(code="schema", message=f"{loc}: {message}"))
    return issues


def validate_series_bible(bible: SeriesBible) -> ValidationEvidence:
    """Deterministic universal plus genre-activated validation of one candidate.

    Universal checks (schema, references, uniqueness, ordering, explicit
    arithmetic, and projection consistency) always apply. Genre-activated rules
    declared by the SeriesBible apply only when the candidate's genre activates
    them; a general-genre idea is never rejected for missing mystery-only clues
    or reveal mechanics.
    """
    issues: list[ValidationIssue] = []
    try:
        StoryContract.model_validate(bible.content.story_contract.model_dump(mode="json"))
    except ValidationError as exc:
        issues.extend(_schema_validation_issues(exc))
    issues.extend(_projection_issues(bible))
    issues.extend(_activated_rule_issues(bible))
    return ValidationEvidence(passed=not issues, issues=issues, validated_at=_utc_now())


def _projection_issues(bible: SeriesBible) -> list[ValidationIssue]:
    """Universal projection consistency: every projection belongs to one candidate.

    A substantive biography projection (one that declares character definitions)
    must cover the contract cast. Sparse or placeholder projections that declare
    no names are treated as unspecified and are not rejected.
    """
    issues: list[ValidationIssue] = []
    contract = bible.content.story_contract
    biography = bible.content.character_biographies
    substantive = "\n" in biography or "：" in biography or ":" in biography
    if substantive:
        base_counts = Counter(
            character_label_base(character.name) for character in contract.characters
        )
        for character in contract.characters:
            base = character_label_base(character.name)
            covered = character.name in biography or (
                bool(base)
                and base != character.name
                and base_counts[base] == 1
                and base in biography
            )
            if not covered:
                issues.append(
                    ValidationIssue(
                        code="projection_missing_biography",
                        message="Projection does not include every contract character biography",
                        refs=[character.character_id],
                    )
                )
    return issues


def _activated_rule_issues(bible: SeriesBible) -> list[ValidationIssue]:
    """Clue lifecycle enforcement removed; mystery reveal cadence remains."""
    issues: list[ValidationIssue] = []
    if bible.genre == "mystery" and not bible.content.story_contract.clues:
        issues.append(
            ValidationIssue(
                code="mystery_reveal_required",
                message="A mystery candidate must declare and resolve reveal mechanics",
            )
        )
    return issues


def activated_rule_names(bible: SeriesBible) -> frozenset[str]:
    """The rule names currently activated by this candidate's declared genre."""
    return _MYSTERY_ACTIVATED_RULES if bible.genre == "mystery" else frozenset()


def universal_rule_names() -> frozenset[str]:
    return frozenset(_UNIVERSAL_RULES)


def bind_global_design_review(
    bible: SeriesBible,
    *,
    review_call_id: str,
    review_model_id: str,
    passed: bool,
    evidence: str,
    issues: list[Mapping[str, Any]] | None = None,
    now: datetime | None = None,
) -> GlobalDesignReview:
    """Bind one global design review to exactly one candidate."""
    return GlobalDesignReview(
        review_call_id=review_call_id,
        candidate_id=bible.candidate_id,
        candidate_hash=bible.content_hash,
        review_model_id=review_model_id,
        passed=passed,
        evidence=evidence,
        issues=[ReviewIssue.model_validate(item) for item in issues or []],
        reviewed_at=now or _utc_now(),
    )


def project_series_bible(bible: SeriesBible, *, is_active: bool) -> SeriesBibleSummary:
    """Project one candidate for the durable resource and workbench."""
    contract = bible.content.story_contract
    contract_hash = story_contract_sha256(contract)
    return SeriesBibleSummary(
        candidate_id=bible.candidate_id,
        version=bible.version,
        design_epoch=bible.design_epoch,
        content_hash=bible.content_hash,
        status=bible.status,
        is_active=is_active,
        l0_variant=bible.l0_variant,
        genre=bible.genre,
        lineage=bible.lineage,
        projections=SeriesBibleProjections(
            story_outline=bible.content.story_outline,
            character_biographies=bible.content.character_biographies,
            relationship_logic=bible.content.relationship_logic,
            episode_outline=bible.content.episode_outline,
            story_contract_markdown=render_story_contract_markdown(contract, contract_hash),
            review_milestones=bible.content.review_milestones,
            script_generation_groups=bible.content.script_generation_groups,
        ),
        review_milestones=bible.content.review_milestones,
        script_generation_groups=bible.content.script_generation_groups,
        validation=bible.validation,
        global_review=bible.global_review,
        created_at=bible.created_at,
    )
