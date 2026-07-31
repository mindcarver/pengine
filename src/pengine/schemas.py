from datetime import datetime
from enum import StrEnum
from typing import Annotated, Literal
from uuid import UUID

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, StringConstraints

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
    GENERATING_CHARACTER_BIOGRAPHIES = "generating_character_biographies"
    GENERATING_RELATIONSHIP_LOGIC = "generating_relationship_logic"
    GENERATING_EPISODE_OUTLINE = "generating_episode_outline"
    GENERATING_EPISODE_SCRIPTS = "generating_episode_scripts"
    ACCEPTING_L0 = "accepting_l0"
    ACCEPTING_L4 = "accepting_l4"
    ASSEMBLING_DELIVERY = "assembling_delivery"


class UserStage(StrEnum):
    DETERMINING_DIRECTION = "determining_direction"
    GENERATING_STORY_OUTLINE = "generating_story_outline"
    GENERATING_CHARACTER_BIOGRAPHIES = "generating_character_biographies"
    GENERATING_RELATIONSHIPS = "generating_relationships"
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
    l0_gate: GateResult
    l4_gate: GateResult
    ownership_statement: NonEmptyText
    feedback_handling: list[FeedbackHandlingItem]


class Delivery(StrictModel):
    content_package: ContentPackage
    delivery_report: DeliveryReport


class RunFailure(StrictModel):
    code: Literal[
        "persona_package_invalid",
        "relay_unavailable",
        "relay_incompatible",
        "structured_output_invalid",
        "stage_validation_failed",
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


class QualityGateRejection(StrictModel):
    code: Literal["quality_gate_rejected"] = "quality_gate_rejected"
    stage: Literal["accepting_l0", "accepting_l4"]
    evidence: NonEmptyText | None = None
    attempt_count: int = Field(ge=1, le=3)
    can_retry: bool


class FinalReviewProgress(StrictModel):
    l0: Literal["pending", "running", "passed", "paused", "failed"]
    l4: Literal["pending", "running", "passed", "paused", "failed"]


class EpisodeProgress(StrictModel):
    total: int = Field(ge=1)
    completed: int = Field(ge=0)
    current: int | None = Field(default=None, ge=1)


class RunProgress(StrictModel):
    current_stage: UserStage
    completed_stages: list[UserStage]
    elapsed_seconds: int = Field(ge=0)
    recovery_state: Literal["none", "auto_resuming", "paused"]
    final_review: FinalReviewProgress
    episodes: EpisodeProgress | None = None
    can_continue: bool
    can_end: bool


class CreativeDirectionDraft(StrictModel):
    stage: Literal["determining_direction"] = "determining_direction"
    selected_l0_variant: NonEmptyText
    selection_rationale: NonEmptyText


class CreativeTextDraft(StrictModel):
    stage: Literal[
        "generating_story_outline",
        "generating_character_biographies",
        "generating_relationships",
        "generating_episode_outline",
    ]
    content: NonEmptyText


class EpisodeDraft(StrictModel):
    episode_number: int = Field(ge=1)
    content: NonEmptyText
    content_sha256: Sha256
    completed_at: datetime


CreativeDraft = Annotated[
    CreativeDirectionDraft | CreativeTextDraft,
    Field(discriminator="stage"),
]


class RunDraftSnapshot(StrictModel):
    artifacts: list[CreativeDraft] = Field(default_factory=list)
    episodes: list[EpisodeDraft] = Field(default_factory=list)
    review_status: FinalReviewProgress


class RunPause(StrictModel):
    code: Literal["run_timeout"] = "run_timeout"
    message: NonEmptyText
    stage: UserStage
    timeout_count: int = Field(ge=2)
    episode_number: int | None = Field(default=None, ge=1)


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
        "idempotency_conflict",
        "revision_not_allowed",
        "revision_feedback_locked",
        "run_not_controllable",
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
    l0_gate: GateResult
    l4_gate: GateResult
    feedback_handling: list[FeedbackHandlingItem] = Field(default_factory=list)
