import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator
from pydantic import ValidationError

from pengine.api import create_app
from pengine.config import Settings
from pengine.personas import canonical_snapshot_sha256
from pengine.schemas import RunPause, UserStage

ROOT = Path(__file__).parents[1]


def test_machine_contracts_parse_and_example_manifest_validates() -> None:
    openapi = json.loads((ROOT / "contracts/openapi.json").read_text())
    manifest_schema = json.loads((ROOT / "contracts/persona-package.schema.json").read_text())
    manifest_example = json.loads((ROOT / "contracts/examples/persona-manifest.json").read_text())

    assert openapi["openapi"] == "3.1.0"
    Draft202012Validator.check_schema(manifest_schema)
    Draft202012Validator(manifest_schema).validate(manifest_example)
    documented_snapshot = openapi["paths"]["/personas"]["get"]["responses"]["200"]["content"][
        "application/json"
    ]["examples"]["nonProduction"]["value"]["items"][0]["snapshot_sha256"]
    assert documented_snapshot == canonical_snapshot_sha256(
        manifest_example,
        manifest_example["package_sha256"],
    )


def test_openapi_exposes_creation_and_run_control_operations() -> None:
    openapi = json.loads((ROOT / "contracts/openapi.json").read_text())

    operations = {
        (method.upper(), path)
        for path, methods in openapi["paths"].items()
        for method in methods
        if method in {"get", "post", "put", "patch", "delete"}
    }

    assert operations == {
        ("POST", "/auth/register"),
        ("POST", "/auth/login"),
        ("POST", "/auth/logout"),
        ("GET", "/me"),
        ("GET", "/personas"),
        ("GET", "/creations"),
        ("POST", "/creations"),
        ("GET", "/creations/{creation_id}"),
        ("GET", "/creations/{creation_id}/runs/{run_kind}/presentation"),
        ("POST", "/creations/{creation_id}/revision"),
        ("POST", "/creations/{creation_id}/runs/{run_kind}/continue"),
        ("POST", "/creations/{creation_id}/runs/{run_kind}/retry-final-review"),
        ("POST", "/creations/{creation_id}/runs/{run_kind}/authorize-repair"),
        ("POST", "/creations/{creation_id}/runs/{run_kind}/end"),
        ("POST", "/creations/{creation_id}/runs/{run_kind}/retry"),
    }


def test_openapi_auth_errors_match_runtime_contract() -> None:
    documented = json.loads((ROOT / "contracts/openapi.json").read_text())
    generated = create_app(settings=Settings()).openapi()

    for path in ("/auth/register", "/auth/login"):
        assert documented["paths"][path]["post"] == generated["paths"][path]["post"]

    register_responses = documented["paths"]["/auth/register"]["post"]["responses"]
    login_responses = documented["paths"]["/auth/login"]["post"]["responses"]
    assert register_responses["409"]["content"]["application/json"]["schema"] == {
        "$ref": "#/components/schemas/CommandError"
    }
    assert login_responses["401"]["content"]["application/json"]["schema"] == {
        "$ref": "#/components/schemas/CommandError"
    }
    assert register_responses["422"]["content"]["application/json"]["schema"] == {
        "$ref": "#/components/schemas/CommandError"
    }
    assert login_responses["422"]["content"]["application/json"]["schema"] == {
        "$ref": "#/components/schemas/CommandError"
    }


def test_openapi_exposes_delivery_presentation_read_model() -> None:
    openapi = json.loads((ROOT / "contracts/openapi.json").read_text())
    operation = openapi["paths"]["/creations/{creation_id}/runs/{run_kind}/presentation"]["get"]
    schemas = openapi["components"]["schemas"]
    generated = create_app(settings=Settings()).openapi()

    assert operation["operationId"] == "getDeliveryPresentation"
    assert operation["responses"]["200"]["content"]["application/json"]["schema"] == {
        "$ref": "#/components/schemas/DeliveryPresentation"
    }
    assert schemas["DeliveryPresentation"]["properties"]["status"]["enum"] == [
        "complete",
        "partial",
        "source",
    ]
    assert schemas["EpisodeEntry"]["properties"]["episode_number"]["minimum"] == 1
    assert (
        operation
        == generated["paths"]["/creations/{creation_id}/runs/{run_kind}/presentation"]["get"]
    )
    for name in (
        "PresentationItem",
        "StorySection",
        "CharacterEntry",
        "RelationshipEntry",
        "EpisodeEntry",
        "StoryOutlinePresentation",
        "CharacterBiographiesPresentation",
        "RelationshipLogicPresentation",
        "EpisodeOutlinePresentation",
        "EpisodeScriptsPresentation",
        "DeliveryPresentation",
    ):
        assert schemas[name] == generated["components"]["schemas"][name]


def test_openapi_exposes_durable_episode_drafts_for_readable_runs() -> None:
    schemas = json.loads((ROOT / "contracts/openapi.json").read_text())["components"]["schemas"]

    for name in (
        "QueuedRun",
        "RunningRun",
        "AutoResumingRun",
        "PausedRun",
        "QualityRejectedRun",
        "RevisionQueued",
        "RevisionRunning",
        "RevisionAutoResuming",
        "RevisionPaused",
        "RevisionQualityRejected",
        "EndedRun",
        "FailedRun",
        "RevisionEnded",
        "RevisionFailed",
    ):
        assert schemas[name]["properties"]["drafts"] == {
            "$ref": "#/components/schemas/RunDraftSnapshot"
        }

    assert "drafts" not in schemas["SucceededRun"]["properties"]
    assert "drafts" not in schemas["RevisionSucceeded"]["properties"]
    assert schemas["RunProgress"]["properties"]["episodes"] == {
        "anyOf": [
            {"$ref": "#/components/schemas/EpisodeProgress"},
            {"type": "null"},
        ]
    }
    assert schemas["RunDraftSnapshot"]["properties"]["episodes"] == {
        "items": {"$ref": "#/components/schemas/EpisodeDraft"},
        "type": "array",
        "title": "Episodes",
    }


def test_openapi_exposes_recovery_reasons_and_pause_evidence() -> None:
    schemas = json.loads((ROOT / "contracts/openapi.json").read_text())["components"]["schemas"]

    assert schemas["RunProgress"]["properties"]["recovery_reason"]["enum"] == [
        "none",
        "run_timeout",
        "relay_interruption",
        "content_rejected",
        "episode_error",
        "context_budget",
        "relay_identity_mismatch",
        "repair_authorization",
    ]
    assert "recovery_reason" in schemas["RunProgress"]["required"]
    assert schemas["RunPause"]["properties"]["code"]["enum"] == [
        "run_timeout",
        "relay_interruption",
        "content_rejected",
        "episode_error",
        "context_budget",
        "relay_identity_mismatch",
        "repair_authorization",
    ]
    assert schemas["RunProgress"]["properties"]["model_calls"] == {
        "items": {"$ref": "#/components/schemas/ModelCallSummary"},
        "type": "array",
        "title": "Model Calls",
    }
    assert schemas["RunPause"]["properties"]["content_repair_count"]["anyOf"] == [
        {"type": "integer", "maximum": 4.0, "minimum": 2.0},
        {"type": "null"},
    ]
    assert schemas["EpisodeDraft"]["properties"]["state_delta"]["anyOf"][0] == {
        "$ref": "#/components/schemas/EpisodeStateDelta"
    }

    generated = create_app(settings=Settings()).openapi()["components"]["schemas"]
    for name in (
        "RunFailure",
        "RunPause",
        "RunProgress",
        "EpisodeDraft",
        "EpisodeStateDelta",
        "SeriesState",
        "CharacterKnowledge",
        "KnowledgeGain",
        "ScriptEvidence",
        "ReviewIssue",
        "SemanticReview",
        "RepairAuthorization",
    ):
        assert schemas[name] == generated[name]


def test_openapi_authorize_repair_description_matches_runtime() -> None:
    path = "/creations/{creation_id}/runs/{run_kind}/authorize-repair"
    documented = json.loads((ROOT / "contracts/openapi.json").read_text())["paths"][path]["post"]
    generated = create_app(settings=Settings()).openapi()["paths"][path]["post"]

    assert documented["description"] == generated["description"]
    assert "latest review evidence" in documented["description"]
    assert "same evidence" not in documented["description"]


def test_pause_counts_cannot_mix_transport_and_content_recovery() -> None:
    content_pause = RunPause(
        code="content_rejected",
        message="Continuity review still failed.",
        stage=UserStage.GENERATING_EPISODE_SCRIPTS,
        content_repair_count=2,
        episode_number=1,
    )
    assert content_pause.timeout_count is None

    max_story_content_pause = RunPause(
        code="content_rejected",
        message="Story consistency review still failed.",
        stage=UserStage.GENERATING_STORY_OUTLINE,
        content_repair_count=4,
    )
    assert max_story_content_pause.content_repair_count == 4

    with pytest.raises(ValidationError):
        RunPause(
            code="content_rejected",
            message="Story consistency review still failed.",
            stage=UserStage.GENERATING_STORY_OUTLINE,
            content_repair_count=5,
        )

    with pytest.raises(ValidationError):
        RunPause(
            code="content_rejected",
            message="Continuity review still failed.",
            stage=UserStage.GENERATING_EPISODE_SCRIPTS,
            timeout_count=2,
            content_repair_count=2,
            episode_number=1,
        )

    with pytest.raises(ValidationError):
        RunPause(
            code="relay_interruption",
            message="Relay was interrupted twice.",
            stage=UserStage.GENERATING_EPISODE_SCRIPTS,
        )


def test_openapi_exposes_quality_rejection_recovery_contract() -> None:
    openapi = json.loads((ROOT / "contracts/openapi.json").read_text())
    schemas = openapi["components"]["schemas"]
    retry = openapi["paths"]["/creations/{creation_id}/runs/{run_kind}/retry-final-review"]["post"]

    assert retry["operationId"] == "retryFinalReview"
    assert retry["responses"]["202"]["content"]["application/json"]["schema"] == {
        "$ref": "#/components/schemas/RunControlAccepted"
    }
    assert schemas["QualityGateRejection"]["properties"]["stage"] == {
        "type": "string",
        "enum": ["accepting_l0", "accepting_l4"],
        "title": "Stage",
    }
    assert schemas["QualityGateRejection"]["properties"]["can_retry"] == {
        "type": "boolean",
        "title": "Can Retry",
    }
    assert "can_retry" in schemas["QualityGateRejection"]["required"]
    assert schemas["QualityRejectedRun"]["properties"]["quality_rejection"] == {
        "$ref": "#/components/schemas/QualityGateRejection"
    }
    assert schemas["RevisionQualityRejected"]["properties"]["quality_rejection"] == {
        "$ref": "#/components/schemas/QualityGateRejection"
    }
    resource = schemas["CreationResource"]["properties"]
    assert resource["initial"]["discriminator"]["mapping"]["quality_rejected"] == (
        "#/components/schemas/QualityRejectedRun"
    )
    assert resource["revision"]["discriminator"]["mapping"]["quality_rejected"] == (
        "#/components/schemas/RevisionQualityRejected"
    )


def test_openapi_documents_outline_group_progress() -> None:
    schemas = json.loads((ROOT / "contracts/openapi.json").read_text())["components"]["schemas"]

    outline_ref = schemas["RunProgress"]["properties"]["outline_groups"]["anyOf"][0]["$ref"]
    assert outline_ref == "#/components/schemas/OutlineGroupProgress"
    assert schemas["OutlineGroupProgress"]["required"] == [
        "committed_groups",
        "committed_through_episode",
    ]
    assert set(schemas["OutlineGroupProgress"]["properties"]) == {
        "committed_groups",
        "committed_through_episode",
        "current_group",
        "current_start_episode",
        "current_end_episode",
    }
