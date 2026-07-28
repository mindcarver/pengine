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
        "checkpoint_unavailable",
        "attempts_exhausted",
        "internal_error",
    ]
    message: NonEmptyText
    failed_stage: InternalStage
    attempt_count: int = Field(ge=1, le=3)


class QueuedRun(StrictModel):
    state: Literal["queued"] = "queued"


class RunningRun(StrictModel):
    state: Literal["running"] = "running"


class SucceededRun(StrictModel):
    state: Literal["succeeded"] = "succeeded"
    result: Delivery


class FailedRun(StrictModel):
    state: Literal["failed"] = "failed"
    failure: RunFailure


RunStatus = Annotated[
    QueuedRun | RunningRun | SucceededRun | FailedRun,
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


class RevisionRunning(StrictModel):
    state: Literal["running"] = "running"
    feedback_locked: Literal[True] = True


class RevisionFailed(StrictModel):
    state: Literal["failed"] = "failed"
    feedback_locked: Literal[True] = True
    retryable: Literal[True] = True
    failure: RunFailure


class RevisionSucceeded(StrictModel):
    state: Literal["succeeded"] = "succeeded"
    feedback_locked: Literal[True] = True
    result: Delivery


RevisionStatus = Annotated[
    RevisionUnavailable
    | RevisionAvailable
    | RevisionQueued
    | RevisionRunning
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
        "service_unavailable",
    ]
    message: NonEmptyText


class L0Selection(StrictModel):
    selected_l0_variant: NonEmptyText
    selection_rationale: NonEmptyText


class TextArtifact(StrictModel):
    content: NonEmptyText


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
