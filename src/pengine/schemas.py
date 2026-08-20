import hashlib
from datetime import datetime
from enum import StrEnum
from typing import Annotated, Any, Literal
from uuid import UUID

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    model_validator,
)

from pengine.continuity import EpisodeStateDelta, SemanticReview, SeriesState
from pengine.series_bible import SeriesBibleSummary

NonEmptyText = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, pattern=r"\S"),
]
Sha256 = Annotated[str, StringConstraints(pattern=r"^[a-f0-9]{64}$")]


def _require_non_blank(value: str) -> str:
    if not value.strip():
        raise ValueError("String must contain non-whitespace text")
    return value


NonBlankPreservedText = Annotated[
    str,
    StringConstraints(min_length=1, pattern=r"\S"),
    AfterValidator(_require_non_blank),
]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class InternalStage(StrEnum):
    LOADING_PERSONA = "loading_persona"
    SELECTING_L0_VARIANT = "selecting_l0_variant"
    GENERATING_STORY_OUTLINE = "generating_story_outline"
    GENERATING_CHARACTER_RELATIONSHIPS = "generating_character_relationships"
    GENERATING_EPISODE_OUTLINE = "generating_episode_outline"
    GENERATING_EPISODE_SCRIPTS = "generating_episode_scripts"
    ACCEPTING_L0 = "accepting_l0"
    ACCEPTING_L4 = "accepting_l4"
    ASSEMBLING_DELIVERY = "assembling_delivery"


class UserStage(StrEnum):
    DETERMINING_DIRECTION = "determining_direction"
    GENERATING_STORY_OUTLINE = "generating_story_outline"
    GENERATING_CHARACTER_RELATIONSHIPS = "generating_character_relationships"
    GENERATING_EPISODE_OUTLINE = "generating_episode_outline"
    GENERATING_EPISODE_SCRIPTS = "generating_episode_scripts"
    FINAL_REVIEW = "final_review"


class CreateCreationRequest(StrictModel):
    persona_id: NonEmptyText
    story: NonEmptyText
    requirements: NonEmptyText


class RevisionRequest(StrictModel):
    feedback: NonBlankPreservedText


class PersonaSummary(StrictModel):
    persona_id: Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$")]
    display_name: NonEmptyText
    version: NonEmptyText
    snapshot_sha256: Sha256


class PersonaList(StrictModel):
    items: list[PersonaSummary]


class PersonaSnapshot(PersonaSummary):
    pass


class ContentPackage(StrictModel):
    story_outline: NonEmptyText
    character_biographies: NonEmptyText
    relationship_logic: NonEmptyText
    episode_outline: NonEmptyText
    episode_scripts: NonEmptyText


class GateResult(StrictModel):
    passed: Literal[True]
    evidence: NonEmptyText


class FeedbackHandlingItem(StrictModel):
    feedback_item: NonEmptyText
    handling: NonEmptyText
    result: NonEmptyText


class DeliveryReport(StrictModel):
    persona_id: str
    persona_version: str
    persona_snapshot_sha256: Sha256
    selected_l0_variant: NonEmptyText
    selection_rationale: NonEmptyText
    l0_gate: GateResult | None = None
    l4_gate: GateResult | None = None
    ownership_statement: NonEmptyText
    feedback_handling: list[FeedbackHandlingItem]


class Delivery(StrictModel):
    content_package: ContentPackage
    delivery_report: DeliveryReport


PresentationId = Annotated[
    str,
    StringConstraints(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$"),
]


class PresentationItem(StrictModel):
    id: PresentationId
    label: NonEmptyText
    ordinal: int = Field(ge=1)
    content: NonBlankPreservedText
    content_sha256: Sha256

    @model_validator(mode="after")
    def validate_content_hash(self) -> "PresentationItem":
        expected = hashlib.sha256(self.content.encode("utf-8")).hexdigest()
        if self.content_sha256 != expected:
            raise ValueError("Presentation item hash must match its content")
        return self


class StorySection(PresentationItem):
    level: int = Field(ge=1, le=3)


class CharacterEntry(PresentationItem):
    group: Literal["core", "supporting", "other"] = "other"


class RelationshipEntry(PresentationItem):
    group: Literal["primary", "supporting", "other"] = "other"


class EpisodeEntry(PresentationItem):
    episode_number: int = Field(ge=1)
    scenes: list[PresentationItem] = Field(default_factory=list)


class StoryOutlinePresentation(StrictModel):
    key: Literal["story_outline"] = "story_outline"
    title: Literal["故事大纲"] = "故事大纲"
    mode: Literal["structured", "source"]
    source_text: NonBlankPreservedText
    source_sha256: Sha256
    sections: list[StorySection] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_integrity(self) -> "StoryOutlinePresentation":
        _validate_presentation_artifact(
            self.source_text, self.source_sha256, self.sections, self.mode
        )
        _validate_complete_partition(self.source_text, self.sections)
        return self


class CharacterBiographiesPresentation(StrictModel):
    key: Literal["character_biographies"] = "character_biographies"
    title: Literal["人物小传"] = "人物小传"
    mode: Literal["structured", "source"]
    source_text: NonBlankPreservedText
    source_sha256: Sha256
    characters: list[CharacterEntry] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_integrity(self) -> "CharacterBiographiesPresentation":
        _validate_presentation_artifact(
            self.source_text, self.source_sha256, self.characters, self.mode
        )
        _validate_complete_partition(self.source_text, self.characters)
        return self


class RelationshipLogicPresentation(StrictModel):
    key: Literal["relationship_logic"] = "relationship_logic"
    title: Literal["人物关系"] = "人物关系"
    mode: Literal["structured", "source"]
    source_text: NonBlankPreservedText
    source_sha256: Sha256
    relationships: list[RelationshipEntry] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_integrity(self) -> "RelationshipLogicPresentation":
        _validate_presentation_artifact(
            self.source_text, self.source_sha256, self.relationships, self.mode
        )
        _validate_complete_partition(self.source_text, self.relationships)
        return self


class EpisodeOutlinePresentation(StrictModel):
    key: Literal["episode_outline"] = "episode_outline"
    title: Literal["分集大纲"] = "分集大纲"
    mode: Literal["structured", "source"]
    source_text: NonBlankPreservedText
    source_sha256: Sha256
    episodes: list[EpisodeEntry] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_integrity(self) -> "EpisodeOutlinePresentation":
        _validate_presentation_artifact(
            self.source_text,
            self.source_sha256,
            self.episodes,
            self.mode,
            require_source_membership=False,
        )
        _validate_episode_entries(self.episodes)
        return self


class EpisodeScriptsPresentation(StrictModel):
    key: Literal["episode_scripts"] = "episode_scripts"
    title: Literal["分集剧本"] = "分集剧本"
    mode: Literal["structured", "source"]
    source_text: NonBlankPreservedText
    source_sha256: Sha256
    episodes: list[EpisodeEntry] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_integrity(self) -> "EpisodeScriptsPresentation":
        _validate_presentation_artifact(
            self.source_text, self.source_sha256, self.episodes, self.mode
        )
        _validate_episode_entries(self.episodes)
        return self


def _validate_presentation_artifact(
    source_text: str,
    source_sha256: str,
    items: list[PresentationItem],
    mode: Literal["structured", "source"],
    *,
    require_source_membership: bool = True,
) -> None:
    expected_source_hash = hashlib.sha256(source_text.encode("utf-8")).hexdigest()
    if source_sha256 != expected_source_hash:
        raise ValueError("Presentation source hash must match its source text")
    if mode == "structured" and not items:
        raise ValueError("Structured presentation artifacts require items")
    if mode == "source" and items:
        raise ValueError("Source presentation artifacts cannot contain items")
    if len({item.id for item in items}) != len(items):
        raise ValueError("Presentation item IDs must be unique")
    if [item.ordinal for item in items] != list(range(1, len(items) + 1)):
        raise ValueError("Presentation item ordinals must be contiguous from 1")
    if require_source_membership:
        positions = [source_text.find(item.content) for item in items]
        if any(position < 0 for position in positions) or positions != sorted(positions):
            raise ValueError("Presentation items must occur in source order")


def _validate_complete_partition(source_text: str, items: list[PresentationItem]) -> None:
    if items and "".join(item.content for item in items) != source_text:
        raise ValueError("Structured presentation items must preserve the complete source")


def _validate_episode_entries(entries: list[EpisodeEntry]) -> None:
    if [entry.episode_number for entry in entries] != list(range(1, len(entries) + 1)):
        raise ValueError("Presentation episode numbers must be contiguous from 1")


class DeliveryPresentation(StrictModel):
    schema_version: Literal[1] = 1
    creation_id: UUID
    run_kind: Literal["initial", "revision"]
    status: Literal["complete", "partial", "source"]
    story_outline: StoryOutlinePresentation
    character_biographies: CharacterBiographiesPresentation
    relationship_logic: RelationshipLogicPresentation
    episode_outline: EpisodeOutlinePresentation
    episode_scripts: EpisodeScriptsPresentation

    @model_validator(mode="after")
    def validate_modes_and_status(self) -> "DeliveryPresentation":
        artifacts = (
            (self.story_outline, self.story_outline.sections),
            (self.character_biographies, self.character_biographies.characters),
            (self.relationship_logic, self.relationship_logic.relationships),
            (self.episode_outline, self.episode_outline.episodes),
            (self.episode_scripts, self.episode_scripts.episodes),
        )
        for artifact, items in artifacts:
            if artifact.mode == "structured" and not items:
                raise ValueError("Structured presentation artifacts require items")
            if artifact.mode == "source" and items:
                raise ValueError("Source presentation artifacts cannot contain items")
        modes = {artifact.mode for artifact, _ in artifacts}
        expected = (
            "complete"
            if modes == {"structured"}
            else "source"
            if modes == {"source"}
            else "partial"
        )
        if self.status != expected:
            raise ValueError("Presentation status must match artifact modes")
        return self


class RunFailure(StrictModel):
    code: Literal[
        "persona_package_invalid",
        "relay_unavailable",
        "relay_incompatible",
        "relay_rejected",
        "preflight_blocked",
        "structured_output_invalid",
        "stage_validation_failed",
        "content_review_rejected",
        "agent_execution_limit",
        "graph_recursion_limit",
        "quality_gate_rejected",
        "checkpoint_unavailable",
        "attempts_exhausted",
        "ended_by_user",
        "internal_error",
    ]
    message: NonEmptyText
    failed_stage: InternalStage
    attempt_count: int = Field(ge=1, le=3)


class QualityRepairIssue(StrictModel):
    issue_id: NonEmptyText
    rule_source: NonEmptyText
    episode_number: int = Field(ge=1)
    exact_excerpt: NonEmptyText
    repair_instruction: NonEmptyText


class QualityRepairPlan(StrictModel):
    scope: Literal["episode_content", "design_rebuild", "unresolved"]
    rationale: NonEmptyText
    issues: list[QualityRepairIssue] = Field(default_factory=list, max_length=6)

    @model_validator(mode="after")
    def validate_scope(self) -> "QualityRepairPlan":
        if self.scope == "episode_content" and not self.issues:
            raise ValueError("Episode-content repair requires at least one bound issue")
        if self.scope != "episode_content" and self.issues:
            raise ValueError("Only episode-content repair may name episode issues")
        return self


class QualityGateRejection(StrictModel):
    code: Literal["quality_gate_rejected"] = "quality_gate_rejected"
    stage: Literal["accepting_l0", "accepting_l4"]
    evidence: NonEmptyText | None = None
    attempt_count: int = Field(ge=1, le=3)
    can_retry: bool
    repair_plan: QualityRepairPlan | None = None
    repair_state: Literal["available", "queued", "repairing", "applied", "blocked"] | None = None

    @model_validator(mode="after")
    def validate_repair_state(self) -> "QualityGateRejection":
        if (
            self.repair_state in {"available", "queued", "repairing", "applied"}
            and self.repair_plan is not None
            and self.repair_plan.scope != "episode_content"
        ):
            raise ValueError("Only episode-content plans may enter the repair cycle")
        return self


class FinalReviewProgress(StrictModel):
    l0: Literal["pending", "running", "passed", "paused", "failed"]
    l4: Literal["pending", "running", "passed", "paused", "failed"]


class EpisodeProgress(StrictModel):
    total: int = Field(ge=1)
    completed: int = Field(ge=0)
    current: int | None = Field(default=None, ge=1)


class ModelCallUsage(StrictModel):
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    cache_read_tokens: int | None = Field(default=None, ge=0)
    cache_creation_tokens: int | None = Field(default=None, ge=0)
    status: Literal["reported", "partial", "unavailable"]


class ModelCallSummary(StrictModel):
    call_id: str
    operation_id: str | None = None
    role: Literal["generation", "review"]
    adapter: str
    provider: str
    model: str
    response_model_ids: list[str] | None = None
    stage: str | None = None
    episode_number: int | None = Field(default=None, ge=1)
    candidate: str | None = None
    batch: str | None = None
    requested_at: datetime
    finished_at: datetime | None = None
    duration_seconds: float | None = Field(default=None, ge=0)
    estimated_input_tokens: int = Field(ge=0)
    estimated_output_tokens: int = Field(ge=0)
    estimated_total_tokens: int = Field(ge=0)
    context_bundle_sha256: str | None = None
    context_manifest: dict[str, Any] | None = None
    verified_limit_tokens: int | None = Field(default=None, ge=1)
    preflight: Literal["ok", "blocked"]
    status: Literal[
        "started",
        "succeeded",
        "failed",
        "timed_out",
        "stale",
        "superseded",
        "preflight_blocked",
    ]
    usage: ModelCallUsage
    finish_reason: str | None = None
    outcome: Literal[
        "success",
        "failure",
        "timeout",
        "stale",
        "superseded",
        "blocked",
        "incomplete",
    ]
    error_code: str | None = None
    error_type: str | None = None
    safe_message: str | None = None
    supersedes_call_id: str | None = None


class RunProgress(StrictModel):
    current_stage: UserStage
    completed_stages: list[UserStage]
    elapsed_seconds: int = Field(ge=0)
    recovery_state: Literal["none", "auto_resuming", "paused"]
    recovery_reason: Literal[
        "none",
        "run_timeout",
        "relay_interruption",
        "content_rejected",
        "episode_error",
        "context_budget",
        "relay_identity_mismatch",
        "repair_authorization",
    ]
    final_review: FinalReviewProgress
    episodes: EpisodeProgress | None = None
    model_calls: list[ModelCallSummary] = Field(default_factory=list)
    can_continue: bool
    can_end: bool
    can_retry: bool = False


class CreativeDirectionDraft(StrictModel):
    stage: Literal["determining_direction"] = "determining_direction"
    selected_l0_variant: NonEmptyText
    selection_rationale: NonEmptyText


class CreativeTextDraft(StrictModel):
    stage: Literal[
        "generating_story_outline",
        "generating_character_relationships",
        "generating_episode_outline",
    ]
    content: NonEmptyText


class EpisodeDraft(StrictModel):
    episode_number: int = Field(ge=1)
    content: NonEmptyText
    content_sha256: Sha256
    completed_at: datetime
    contract_sha256: Sha256 | None = None
    state_delta: EpisodeStateDelta | None = None
    series_state: SeriesState | None = None
    series_state_sha256: Sha256 | None = None
    semantic_review: SemanticReview | None = None
    repair_rounds: int | None = Field(default=None, ge=0, le=2)


CreativeDraft = Annotated[
    CreativeDirectionDraft | CreativeTextDraft,
    Field(discriminator="stage"),
]


class RunDraftSnapshot(StrictModel):
    artifacts: list[CreativeDraft] = Field(default_factory=list)
    episodes: list[EpisodeDraft] = Field(default_factory=list)
    design: SeriesBibleSummary | None = Field(
        default=None,
        description=(
            "The latest deterministically valid unfinished SeriesBible design candidate "
            "for this run. Every projection and hash belongs to one candidate; the design "
            "package is never formal delivery."
        ),
    )
    review_status: FinalReviewProgress


class RunPause(StrictModel):
    code: Literal[
        "run_timeout",
        "relay_interruption",
        "content_rejected",
        "episode_error",
        "context_budget",
        "relay_identity_mismatch",
        "repair_authorization",
    ] = "run_timeout"
    message: NonEmptyText
    stage: UserStage
    timeout_count: int | None = Field(default=None, ge=2)
    content_repair_count: int | None = Field(default=None, ge=2, le=4)
    episode_number: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def validate_reason_counts(self) -> "RunPause":
        if self.code == "content_rejected":
            if (
                self.content_repair_count is None
                or not 2 <= self.content_repair_count <= 4
                or self.timeout_count is not None
            ):
                raise ValueError(
                    "Content rejection requires two to four content repairs and no timeout count"
                )
            if self.content_repair_count > 2 and self.stage not in {
                UserStage.GENERATING_STORY_OUTLINE,
                UserStage.GENERATING_CHARACTER_RELATIONSHIPS,
            }:
                raise ValueError("Only story creation stages allow more than two content repairs")
        elif self.code in {
            "episode_error",
            "context_budget",
            "relay_identity_mismatch",
            "repair_authorization",
        }:
            if self.timeout_count is not None or self.content_repair_count is not None:
                raise ValueError("This pause does not use timeout or repair counts")
        elif self.timeout_count is None or self.content_repair_count is not None:
            raise ValueError(
                "Timeout and relay pauses require a timeout count and no content repair count"
            )
        return self


class RepairAuthorization(StrictModel):
    """The exact one-cycle repair authorization shown at a repair-authorization pause.

    It binds the active lineage and carries the review evidence, affected range,
    and a reference token count for the active design projections and retained
    prefix at the pause. This is neither a lower bound nor a total cycle forecast;
    actual repair input and output can differ (RPR-A8/A9).
    """

    authorization_epoch: int = Field(ge=1)
    kind: Literal["design_rebuild", "suffix_rewrite"]
    design_candidate_id: str
    design_content_hash: Sha256
    design_epoch: int = Field(ge=1)
    batch_id: str
    batch_epoch: int = Field(ge=1)
    earliest_affected_episode: int | None = Field(default=None, ge=1)
    range_episodes: int | None = Field(default=None, ge=1)
    estimated_tokens: int | None = Field(
        default=None,
        ge=0,
        description=(
            "Reference token count for active design projections and the retained prefix at the "
            "pause; neither a lower bound nor a total repair-cycle forecast."
        ),
    )
    evidence: NonEmptyText
    review_id: str
    granted_at: datetime | None = None
    consumed_at: datetime | None = None


class QueuedRun(StrictModel):
    state: Literal["queued"] = "queued"
    progress: RunProgress
    drafts: RunDraftSnapshot


class RunningRun(StrictModel):
    state: Literal["running"] = "running"
    progress: RunProgress
    drafts: RunDraftSnapshot


class AutoResumingRun(StrictModel):
    state: Literal["auto_resuming"] = "auto_resuming"
    progress: RunProgress
    drafts: RunDraftSnapshot


class PausedRun(StrictModel):
    state: Literal["paused"] = "paused"
    progress: RunProgress
    drafts: RunDraftSnapshot
    pause: RunPause
    authorization: RepairAuthorization | None = None


class EndedRun(StrictModel):
    state: Literal["ended"] = "ended"
    progress: RunProgress
    drafts: RunDraftSnapshot


class SucceededRun(StrictModel):
    state: Literal["succeeded"] = "succeeded"
    progress: RunProgress
    result: Delivery


class QualityRejectedRun(StrictModel):
    state: Literal["quality_rejected"] = "quality_rejected"
    progress: RunProgress
    quality_rejection: QualityGateRejection
    drafts: RunDraftSnapshot


class FailedRun(StrictModel):
    state: Literal["failed"] = "failed"
    progress: RunProgress
    failure: RunFailure
    drafts: RunDraftSnapshot


RunStatus = Annotated[
    QueuedRun
    | RunningRun
    | AutoResumingRun
    | PausedRun
    | EndedRun
    | SucceededRun
    | QualityRejectedRun
    | FailedRun,
    Field(discriminator="state"),
]


class RevisionUnavailable(StrictModel):
    state: Literal["unavailable"] = "unavailable"
    feedback_locked: Literal[False] = False
    reason: Literal["initial_not_succeeded"] = "initial_not_succeeded"


class RevisionAvailable(StrictModel):
    state: Literal["available"] = "available"
    feedback_locked: Literal[False] = False


class RevisionQueued(StrictModel):
    state: Literal["queued"] = "queued"
    feedback_locked: Literal[True] = True
    progress: RunProgress
    drafts: RunDraftSnapshot


class RevisionRunning(StrictModel):
    state: Literal["running"] = "running"
    feedback_locked: Literal[True] = True
    progress: RunProgress
    drafts: RunDraftSnapshot


class RevisionAutoResuming(StrictModel):
    state: Literal["auto_resuming"] = "auto_resuming"
    feedback_locked: Literal[True] = True
    progress: RunProgress
    drafts: RunDraftSnapshot


class RevisionPaused(StrictModel):
    state: Literal["paused"] = "paused"
    feedback_locked: Literal[True] = True
    progress: RunProgress
    drafts: RunDraftSnapshot
    pause: RunPause
    authorization: RepairAuthorization | None = None


class RevisionEnded(StrictModel):
    state: Literal["ended"] = "ended"
    feedback_locked: Literal[True] = True
    progress: RunProgress
    drafts: RunDraftSnapshot


class RevisionFailed(StrictModel):
    state: Literal["failed"] = "failed"
    feedback_locked: Literal[True] = True
    retryable: Literal[True] = True
    progress: RunProgress
    failure: RunFailure
    drafts: RunDraftSnapshot


class RevisionSucceeded(StrictModel):
    state: Literal["succeeded"] = "succeeded"
    feedback_locked: Literal[True] = True
    progress: RunProgress
    result: Delivery


class RevisionQualityRejected(StrictModel):
    state: Literal["quality_rejected"] = "quality_rejected"
    feedback_locked: Literal[True] = True
    progress: RunProgress
    quality_rejection: QualityGateRejection
    drafts: RunDraftSnapshot


RevisionStatus = Annotated[
    RevisionUnavailable
    | RevisionAvailable
    | RevisionQueued
    | RevisionRunning
    | RevisionAutoResuming
    | RevisionPaused
    | RevisionEnded
    | RevisionQualityRejected
    | RevisionFailed
    | RevisionSucceeded,
    Field(discriminator="state"),
]


class CreationAccepted(StrictModel):
    creation_id: UUID
    initial_state: Literal["queued"] = "queued"
    resource_url: str


class RevisionAccepted(StrictModel):
    creation_id: UUID
    revision_state: Literal["queued"] = "queued"
    feedback_locked: Literal[True] = True
    resource_url: str


class RunControlAccepted(StrictModel):
    creation_id: UUID
    run_kind: Literal["initial", "revision"]
    run_state: Literal["queued", "running", "auto_resuming", "ended"]
    resource_url: str


class CreationResource(StrictModel):
    creation_id: UUID
    persona: PersonaSnapshot
    initial: RunStatus
    revision: RevisionStatus
    created_at: datetime
    updated_at: datetime


class CommandError(StrictModel):
    code: Literal[
        "invalid_request",
        "persona_not_found",
        "persona_package_unavailable",
        "creation_not_found",
        "presentation_not_available",
        "idempotency_conflict",
        "revision_not_allowed",
        "revision_feedback_locked",
        "run_not_controllable",
        "repair_authorization_stale",
        "series_bible_rebuild_exhausted",
        "service_unavailable",
    ]
    message: NonEmptyText


class L0Selection(StrictModel):
    selected_l0_variant: NonEmptyText
    selection_rationale: NonEmptyText


class TextArtifact(StrictModel):
    content: NonEmptyText


class EpisodePlan(StrictModel):
    episode_number: int = Field(ge=1)
    plan: NonEmptyText


class QualityReview(StrictModel):
    l0_gate: GateResult
    l4_gate: GateResult
    feedback_handling: list[FeedbackHandlingItem] = Field(default_factory=list)


class WorkflowResult(StrictModel):
    content_package: ContentPackage
    selected_l0_variant: NonEmptyText
    selection_rationale: NonEmptyText
    l0_gate: GateResult | None = None
    l4_gate: GateResult | None = None
    feedback_handling: list[FeedbackHandlingItem] = Field(default_factory=list)
