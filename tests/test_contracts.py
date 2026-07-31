import json
from pathlib import Path

from jsonschema import Draft202012Validator

from pengine.personas import canonical_snapshot_sha256

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
        ("GET", "/personas"),
        ("POST", "/creations"),
        ("GET", "/creations/{creation_id}"),
        ("POST", "/creations/{creation_id}/revision"),
        ("POST", "/creations/{creation_id}/runs/{run_kind}/continue"),
        ("POST", "/creations/{creation_id}/runs/{run_kind}/end"),
    }


def test_openapi_exposes_durable_episode_drafts_for_readable_runs() -> None:
    schemas = json.loads((ROOT / "contracts/openapi.json").read_text())["components"]["schemas"]

    for name in (
        "QueuedRun",
        "RunningRun",
        "AutoResumingRun",
        "PausedRun",
        "RevisionQueued",
        "RevisionRunning",
        "RevisionAutoResuming",
        "RevisionPaused",
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
