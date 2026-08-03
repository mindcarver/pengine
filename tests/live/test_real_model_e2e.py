from __future__ import annotations

import copy
import json
import os
import re
import shutil
import socket
import sqlite3
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit
from uuid import uuid4

import httpx
import pytest
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langchain_core.messages import AIMessage, ToolMessage

from pengine.agents import (
    CanonReviewerResult,
    EpisodePlannerResult,
    OutlineRepairPatch,
    ScriptWriterResult,
    StoryArtifactRepairPatch,
    StructuredResultMiddleware,
    _apply_outline_repair_patch,
    _apply_story_artifact_repair_patch,
    _canon_review_with_issue_ledger,
    _outline_repair_context,
    _outline_repair_result,
    _story_repair_context,
    _story_repair_result,
    _structured_output_retry_message,
    _validate_outline_repair_patch_targets,
)
from pengine.config import Settings
from pengine.language import SIMPLIFIED_CHINESE, has_obvious_language_mismatch
from pengine.relay import build_relay_adapter
from pengine.schemas import InternalStage

_ENABLE_ENV = "PENGINE_RUN_LIVE_E2E"
_STOP_STATES = {"succeeded", "failed", "ended", "paused", "quality_rejected"}
_CONTENT_ARTIFACTS = (
    "story_outline",
    "character_biographies",
    "relationship_logic",
    "episode_outline",
    "episode_scripts",
)
_DEFAULT_STORY = (
    "海岛修表师林夏在父亲葬礼后发现一只停在九点十七分的旧表。"
    "她必须在台风封岛前查明父亲为何替失踪船员承担责任，并决定公开真相，"
    "还是保住全村赖以生存的救援站。"
)
_DEFAULT_REQUIREMENTS = (
    "创作三集现实主义悬疑短剧，每集约三分钟；人物姓名、年龄、时间线、"
    "关键证物与动机全程一致；结局不得新增关键证据；全部使用简体中文。"
)
_GENERIC_SECRET = re.compile(rb"\bsk-[A-Za-z0-9_-]{8,}\b")


def _require_dual_model_relay(
    settings: Settings,
    probe: str,
    *,
    evidence_dir: Path | None = None,
) -> None:
    evidence = f"; evidence: {evidence_dir}" if evidence_dir is not None else ""
    if not settings.relay_configured:
        pytest.fail(f"{probe} requires configured generation and review routes{evidence}")
    if settings.generation_context_limit_tokens is None:
        pytest.fail(
            f"{probe} requires PENGINE_GENERATION_CONTEXT_LIMIT_TOKENS (the verified "
            f"context window of the generation route){evidence}"
        )
    if settings.review_context_limit_tokens is None:
        pytest.fail(
            f"{probe} requires PENGINE_REVIEW_CONTEXT_LIMIT_TOKENS (the verified "
            f"context window of the review route){evidence}"
        )


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )


def _append_jsonl(path: Path, value: Any) -> None:
    with path.open("a", encoding="utf-8") as output:
        output.write(json.dumps(value, ensure_ascii=False, default=str) + "\n")


def _safe_base_url(value: str | None) -> str | None:
    if value is None:
        return None
    parsed = urlsplit(value)
    hostname = parsed.hostname or ""
    if ":" in hostname:
        hostname = f"[{hostname}]"
    netloc = hostname
    if parsed.port is not None:
        netloc = f"{netloc}:{parsed.port}"
    return urlunsplit((parsed.scheme, netloc, parsed.path, "", ""))


def _reserve_loopback_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def _prepare_live_e2e_run(
    settings: Settings,
    *,
    repository_root: Path,
    run_label: str,
) -> tuple[Path, Path, float, float]:
    """Validate all local prerequisites before creating durable evidence."""
    _require_dual_model_relay(settings, "Real-model E2E")
    if not settings.persona_root.resolve().is_dir():
        pytest.fail(f"Persona root does not exist: {settings.persona_root.resolve()}")

    timeout_seconds = float(os.getenv("PENGINE_LIVE_E2E_TIMEOUT_SECONDS", "7200"))
    poll_seconds = float(os.getenv("PENGINE_LIVE_E2E_POLL_SECONDS", "2"))
    assert timeout_seconds > 0
    assert poll_seconds > 0

    artifact_root = Path(
        os.getenv(
            "PENGINE_LIVE_E2E_ARTIFACT_ROOT",
            repository_root / ".artifacts" / "live-e2e",
        )
    )
    artifact_root.mkdir(mode=0o700, parents=True, exist_ok=True)
    evidence_dir = artifact_root.resolve() / run_label
    data_dir = evidence_dir / "data"
    evidence_dir.mkdir(mode=0o700, exist_ok=False)
    data_dir.mkdir(mode=0o700)
    return evidence_dir, data_dir, timeout_seconds, poll_seconds


def _response_body(response: httpx.Response) -> Any:
    try:
        return response.json()
    except ValueError:
        return {"text": response.text}


def _progress_summary(resource: dict[str, Any], *, observed_at: str) -> dict[str, Any]:
    initial = resource.get("initial", {})
    progress = initial.get("progress", {})
    return {
        "observed_at": observed_at,
        "updated_at": resource.get("updated_at"),
        "state": initial.get("state"),
        "current_stage": progress.get("current_stage"),
        "completed_stages": progress.get("completed_stages", []),
        "elapsed_seconds": progress.get("elapsed_seconds"),
        "recovery_state": progress.get("recovery_state"),
        "recovery_reason": progress.get("recovery_reason"),
        "final_review": progress.get("final_review"),
        "episodes": progress.get("episodes"),
        "failure": initial.get("failure"),
        "pause": initial.get("pause"),
        "quality_rejection": initial.get("quality_rejection"),
    }


def _timeline_signature(summary: dict[str, Any]) -> str:
    meaningful = {key: value for key, value in summary.items() if key != "elapsed_seconds"}
    meaningful.pop("observed_at", None)
    meaningful.pop("updated_at", None)
    return json.dumps(meaningful, ensure_ascii=False, sort_keys=True, default=str)


def _child_environment(
    settings: Settings,
    data_dir: Path,
    port: int,
    *,
    run_timeout_seconds: float,
) -> dict[str, str]:
    assert settings.relay_base_url is not None
    assert settings.relay_api_key is not None
    assert settings.generation_model_id is not None
    assert settings.review_model_id is not None

    environment = os.environ.copy()
    for obsolete_name in (
        "PENGINE_RELAY_ADAPTER",
        "PENGINE_RELAY_MODEL_ID",
        "PENGINE_RELAY_MAX_OUTPUT_TOKENS",
    ):
        environment.pop(obsolete_name, None)
    environment.update(
        {
            "PENGINE_PERSONA_ROOT": str(settings.persona_root.resolve()),
            "PENGINE_DATA_DIR": str(data_dir.resolve()),
            "PENGINE_HOST": "127.0.0.1",
            "PENGINE_PORT": str(port),
            "PENGINE_RELAY_BASE_URL": settings.relay_base_url,
            "PENGINE_RELAY_API_KEY": settings.relay_api_key.get_secret_value(),
            "PENGINE_GENERATION_MODEL_ID": settings.generation_model_id,
            "PENGINE_REVIEW_MODEL_ID": settings.review_model_id,
            "PENGINE_MODEL_TIMEOUT_SECONDS": str(settings.model_timeout_seconds),
            "PENGINE_RUN_TIMEOUT_SECONDS": str(run_timeout_seconds),
            "PENGINE_LEASE_SECONDS": str(settings.lease_seconds),
            "PENGINE_WORKER_POLL_SECONDS": "0.1",
            "PENGINE_AGENT_RECURSION_LIMIT": str(settings.agent_recursion_limit),
            "PENGINE_RETRIEVAL_LIMIT": str(settings.retrieval_limit),
        }
    )
    caps = {
        "PENGINE_GENERATION_MAX_OUTPUT_TOKENS": settings.generation_max_output_tokens,
        "PENGINE_REVIEW_MAX_OUTPUT_TOKENS": settings.review_max_output_tokens,
    }
    for name, value in caps.items():
        if value is None:
            environment.pop(name, None)
        else:
            environment[name] = str(value)
    context_limits = {
        "PENGINE_GENERATION_CONTEXT_LIMIT_TOKENS": settings.generation_context_limit_tokens,
        "PENGINE_REVIEW_CONTEXT_LIMIT_TOKENS": settings.review_context_limit_tokens,
    }
    for name, value in context_limits.items():
        if value is None:
            environment.pop(name, None)
        else:
            environment[name] = str(value)
    return environment


def _wait_until_ready(
    client: httpx.Client,
    process: subprocess.Popen[bytes],
    *,
    deadline: float,
    log_path: Path,
) -> dict[str, Any]:
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise AssertionError(
                f"Pengine exited before readiness with code {process.returncode}; "
                f"evidence: {log_path}"
            )
        try:
            response = client.get("/personas")
            if response.status_code == 200:
                body = _response_body(response)
                if isinstance(body, dict):
                    return body
                raise AssertionError("GET /personas returned a non-object JSON response")
            last_error = AssertionError(f"GET /personas returned HTTP {response.status_code}")
        except httpx.HTTPError as error:
            last_error = error
        time.sleep(0.2)
    raise AssertionError(
        f"Pengine did not become ready: {type(last_error).__name__ if last_error else 'timeout'}; "
        f"evidence: {log_path}"
    )


def _poll_initial(
    client: httpx.Client,
    *,
    resource_url: str,
    timeout_seconds: float,
    poll_seconds: float,
    polls_path: Path,
    timeline_path: Path,
    final_resource_path: Path,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_seconds
    previous_signature: str | None = None
    latest_resource: dict[str, Any] | None = None

    while time.monotonic() < deadline:
        response = client.get(resource_url)
        body = _response_body(response)
        if response.status_code != 200 or not isinstance(body, dict):
            _append_jsonl(
                polls_path,
                {"observed_at": _now(), "http_status": response.status_code, "body": body},
            )
            raise AssertionError(f"GET {resource_url} returned HTTP {response.status_code}")

        latest_resource = body
        _write_json(final_resource_path, body)
        summary = _progress_summary(body, observed_at=_now())
        _append_jsonl(polls_path, {"http_status": response.status_code, **summary})
        signature = _timeline_signature(summary)
        if signature != previous_signature:
            _append_jsonl(timeline_path, summary)
            previous_signature = signature

        state = summary["state"]
        if state in _STOP_STATES:
            return body
        time.sleep(poll_seconds)

    if latest_resource is not None:
        _write_json(polls_path.with_name("last-resource-at-timeout.json"), latest_resource)
    raise AssertionError(f"Initial run did not stop within {timeout_seconds:g} seconds")


def _assert_success(resource: dict[str, Any], *, evidence_dir: Path) -> None:
    initial = resource.get("initial", {})
    state = initial.get("state")
    if state != "succeeded":
        details = initial.get("failure") or initial.get("pause") or initial.get("quality_rejection")
        pytest.fail(
            f"Real-model E2E stopped in initial.state={state!r}, details={details!r}; "
            f"evidence: {evidence_dir}"
        )

    result = initial.get("result", {})
    content_package = result.get("content_package", {})
    missing = [
        name
        for name in _CONTENT_ARTIFACTS
        if not isinstance(content_package.get(name), str) or not content_package[name].strip()
    ]
    assert not missing, f"Blank delivery artifacts: {missing}; evidence: {evidence_dir}"
    language_mismatches = [
        name
        for name in _CONTENT_ARTIFACTS
        if has_obvious_language_mismatch(
            content_package[name],
            SIMPLIFIED_CHINESE,
        )
    ]
    assert not language_mismatches, (
        "Delivery artifacts are not Simplified Chinese: "
        f"{language_mismatches}; evidence: {evidence_dir}"
    )

    report = result.get("delivery_report", {})
    assert report.get("l0_gate", {}).get("passed") is True, (
        f"L0 gate did not pass; evidence: {evidence_dir}"
    )
    assert report.get("l4_gate", {}).get("passed") is True, (
        f"L4 gate did not pass; evidence: {evidence_dir}"
    )
    assert resource.get("revision", {}).get("state") == "available", (
        f"Revision did not become available; evidence: {evidence_dir}"
    )


def _assert_story_consistency_checkpoints(
    database_path: Path,
    *,
    creation_id: str,
    evidence_dir: Path,
) -> None:
    expected_stages = {
        "generating_story_outline",
        "generating_character_biographies",
        "generating_relationship_logic",
    }
    with sqlite3.connect(f"{database_path.resolve().as_uri()}?mode=ro", uri=True) as connection:
        rows = connection.execute(
            """
            SELECT business_checkpoints.stage,
                   business_checkpoints.payload_json,
                   business_checkpoints.payload_sha256
            FROM business_checkpoints
            JOIN runs ON runs.id = business_checkpoints.run_id
            WHERE runs.creation_id = ?
            """,
            (creation_id,),
        ).fetchall()

    checkpoints = {
        stage: (json.loads(payload_json), payload_sha256)
        for stage, payload_json, payload_sha256 in rows
        if stage in expected_stages
    }
    assert checkpoints.keys() == expected_stages, (
        f"Missing story consistency checkpoints: {sorted(expected_stages - checkpoints.keys())}; "
        f"evidence: {evidence_dir}"
    )

    summary: dict[str, Any] = {}
    for stage in sorted(expected_stages):
        payload, payload_sha256 = checkpoints[stage]
        review = payload.get("consistency_review")
        repair_rounds = payload.get("consistency_repair_rounds")
        assert isinstance(review, dict) and review.get("passed") is True, (
            f"{stage} lacks a passing independent consistency review; evidence: {evidence_dir}"
        )
        assert type(repair_rounds) is int and 0 <= repair_rounds <= 6, (
            f"{stage} has invalid consistency repair rounds; evidence: {evidence_dir}"
        )
        summary[stage] = {
            "checkpoint_sha256": payload_sha256,
            "review_passed": True,
            "issue_count": len(review.get("issues", [])),
            "repair_rounds": repair_rounds,
        }
    _write_json(evidence_dir / "story-consistency-checkpoints.json", summary)


def _redact_log(log_path: Path, secret: str | None) -> None:
    if not log_path.exists():
        return
    content = log_path.read_text(encoding="utf-8", errors="replace")
    if secret:
        content = content.replace(secret, "[REDACTED]")
    content = re.sub(
        r"(?i)(authorization\s*[:=]\s*)(?:bearer\s+)?[^\s,;}]+",
        r"\1[REDACTED]",
        content,
    )
    content = re.sub(r"(?i)(api[_-]?key\s*[:=]\s*)[^\s,;}]+", r"\1[REDACTED]", content)
    content = re.sub(r"\bsk-[A-Za-z0-9_-]{8,}\b", "sk-[REDACTED]", content)
    log_path.write_text(content, encoding="utf-8")


def _assert_model_routing_audit(
    log_path: Path,
    *,
    generation_model_id: str,
    review_model_id: str,
    evidence_dir: Path,
) -> dict[str, Any]:
    expected_models = {
        "generation": generation_model_id,
        "review": review_model_id,
    }
    calls = {role: {"started": set(), "ended": set()} for role in expected_models}
    audit_lines = [
        line
        for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines()
        if "model_call event=" in line
    ]
    assert audit_lines, f"No model-call audit records were captured; evidence: {evidence_dir}"

    for line in audit_lines:
        event_match = re.search(
            r"model_call event=(start|end|error|identity_mismatch) "
            r"role=(generation|review) requested_model_id=([^\s]+)",
            line,
        )
        assert event_match is not None, (
            f"Unparseable model-call audit record: {line!r}; evidence: {evidence_dir}"
        )
        event, role, requested_model_id = event_match.groups()
        assert requested_model_id == expected_models[role], (
            f"Unexpected {role} request model {requested_model_id!r}; evidence: {evidence_dir}"
        )
        assert event not in {"error", "identity_mismatch"}, (
            f"Model-call audit contains event={event!r} for role={role}; evidence: {evidence_dir}"
        )

        call_id_match = re.search(r"call_id=([^\s]+)", line)
        assert call_id_match is not None, (
            f"Model-call audit lacks call_id: {line!r}; evidence: {evidence_dir}"
        )
        call_id = call_id_match.group(1)
        if event == "start":
            calls[role]["started"].add(call_id)
            continue

        response_match = re.search(r"response_model_id=([^\s]+)", line)
        assert response_match is not None, (
            f"Completed model call lacks response identity: {line!r}; evidence: {evidence_dir}"
        )
        assert response_match.group(1) == expected_models[role], (
            f"Unexpected {role} response model {response_match.group(1)!r}; "
            f"evidence: {evidence_dir}"
        )
        calls[role]["ended"].add(call_id)

    summary: dict[str, Any] = {"status": "passed", "routes": {}}
    for role, model_id in expected_models.items():
        started = calls[role]["started"]
        ended = calls[role]["ended"]
        assert started, f"The {role} route was never called; evidence: {evidence_dir}"
        assert started == ended, (
            f"The {role} route has incomplete calls: "
            f"started_only={sorted(started - ended)}, ended_only={sorted(ended - started)}; "
            f"evidence: {evidence_dir}"
        )
        summary["routes"][role] = {
            "requested_model_id": model_id,
            "response_model_id": model_id,
            "completed_calls": len(ended),
        }
    return summary


def _assert_durable_usage_evidence(
    log_path: Path,
    database_path: Path,
    *,
    evidence_dir: Path,
) -> dict[str, Any]:
    """The durable model-call envelope agrees across structured logs and SQLite."""
    record_lines = [
        line
        for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines()
        if "model_call_record " in line
    ]
    assert record_lines, f"No durable model-call records were captured; evidence: {evidence_dir}"

    log_records: dict[str, dict[str, Any]] = {}
    for line in record_lines:
        fields = dict(
            token.split("=", 1)
            for token in line.split("model_call_record ", 1)[1].split()
            if "=" in token
        )
        call_id = fields.get("call_id")
        assert call_id, f"Unparseable model-call record: {line!r}; evidence: {evidence_dir}"
        log_records[call_id] = fields

    succeeded = {
        call_id: fields
        for call_id, fields in log_records.items()
        if fields.get("status") == "succeeded"
    }
    assert succeeded, f"No completed model calls with provider usage; evidence: {evidence_dir}"
    usage_mismatches = [
        call_id
        for call_id, fields in succeeded.items()
        if fields.get("usage_status") != "reported"
        or not fields.get("actual_input_tokens")
        or not fields.get("actual_output_tokens")
    ]
    assert not usage_mismatches, (
        f"Real calls are missing provider-reported usage: {sorted(usage_mismatches)}; "
        f"evidence: {evidence_dir}"
    )

    with sqlite3.connect(f"{database_path.resolve().as_uri()}?mode=ro", uri=True) as connection:
        rows = connection.execute(
            "SELECT call_id, status, usage_status, actual_input_tokens, actual_output_tokens "
            "FROM model_calls"
        ).fetchall()
    db_records = {row[0]: row for row in rows}
    assert set(log_records) <= set(db_records), (
        f"SQLite model_calls missing log call_ids: {sorted(set(log_records) - set(db_records))}; "
        f"evidence: {evidence_dir}"
    )
    for call_id, _fields in succeeded.items():
        row = db_records[call_id]
        assert row[1] == "succeeded"
        assert row[2] == "reported"
        assert row[3] is not None and row[4] is not None

    summary: dict[str, Any] = {
        "status": "passed",
        "recorded_calls": len(log_records),
        "succeeded_with_usage": len(succeeded),
        "example": {
            "call_id": next(iter(succeeded)),
            "model": succeeded[next(iter(succeeded))].get("model"),
            "input_tokens": succeeded[next(iter(succeeded))].get("actual_input_tokens"),
            "output_tokens": succeeded[next(iter(succeeded))].get("actual_output_tokens"),
            "finish_reason": succeeded[next(iter(succeeded))].get("finish_reason"),
        },
    }
    return summary


def _assert_evidence_has_no_secrets(evidence_dir: Path, secret: str | None) -> None:
    exact_secret = secret.encode() if secret else None
    violations: list[str] = []
    for path in sorted(evidence_dir.rglob("*")):
        if not path.is_file():
            continue
        content = path.read_bytes()
        if exact_secret and exact_secret in content:
            violations.append(str(path.relative_to(evidence_dir)))
            continue
        if _GENERIC_SECRET.search(content):
            violations.append(str(path.relative_to(evidence_dir)))
    if violations:
        shutil.rmtree(evidence_dir)
        raise AssertionError(
            "Credential material was written to E2E evidence files; deleted the entire "
            "run directory. Affected relative paths: " + ", ".join(violations)
        )


def _outline_probe_candidate() -> dict[str, Any]:
    return {
        "stage": "generating_episode_outline",
        "content": "第 1 集：林夏只是旁观者。",
        "episode_count": 1,
        "episodes": [{"episode_number": 1, "plan": "林夏调查旧表。"}],
        "story_contract": {
            "version": 1,
            "episode_count": 1,
            "characters": [
                {
                    "character_id": "lin_xia",
                    "name": "林夏",
                    "role": "调查者",
                    "initial_known_fact_ids": [],
                }
            ],
            "relationships": [],
            "facts": [
                {
                    "fact_id": "stopped_watch",
                    "subject": "旧表",
                    "predicate": "停在",
                    "kind": "text",
                    "value": "九点十七分",
                    "first_revealed_episode": 1,
                }
            ],
            "timeline": [
                {
                    "event_id": "watch_found",
                    "order": 1,
                    "when": "第 1 集",
                    "participant_ids": ["lin_xia"],
                    "fact_ids": ["stopped_watch"],
                }
            ],
            "knowledge_states": [
                {
                    "episode_number": 1,
                    "character_id": "lin_xia",
                    "known_fact_ids": ["stopped_watch"],
                }
            ],
            "clues": [],
            "prohibitions": [],
            "episode_obligations": [
                {
                    "obligation_id": "investigate_watch",
                    "episode_number": 1,
                    "new_information_fact_ids": ["stopped_watch"],
                    "end_hook": "林夏决定调查旧表。",
                    "required_clue_ids": [],
                }
            ],
        },
    }


def test_live_e2e_evidence_helpers_are_safe_and_validate_delivery(tmp_path: Path) -> None:
    assert (
        _safe_base_url("https://user:password@example.com/v1?api_key=secret#fragment")
        == "https://example.com/v1"
    )

    content_package = {
        name: "这是完整的简体中文成品内容，人物行动与时间线保持一致。"
        for name in _CONTENT_ARTIFACTS
    }
    resource = {
        "initial": {
            "state": "succeeded",
            "result": {
                "content_package": content_package,
                "delivery_report": {
                    "l0_gate": {"passed": True},
                    "l4_gate": {"passed": True},
                },
            },
        },
        "revision": {"state": "available"},
    }
    _assert_success(resource, evidence_dir=tmp_path)

    english_resource = copy.deepcopy(resource)
    english_resource["initial"]["result"]["content_package"]["episode_outline"] = (
        "This episode outline is entirely in English."
    )
    with pytest.raises(AssertionError, match="not Simplified Chinese.*episode_outline"):
        _assert_success(english_resource, evidence_dir=tmp_path)

    log_path = tmp_path / "server.log"
    log_path.write_text(
        "Authorization: Bearer sk-examplepartial123 api_key=visible-token\n",
        encoding="utf-8",
    )
    _redact_log(log_path, "sk-examplepartial123")
    redacted = log_path.read_text(encoding="utf-8")
    assert "examplepartial" not in redacted
    assert "visible-token" not in redacted
    _assert_evidence_has_no_secrets(tmp_path, "sk-examplepartial123")

    log_path.write_text(
        "\n".join(
            (
                "model_call event=start role=generation "
                "requested_model_id=claude-opus-5 call_id=generation-1",
                "model_call event=end role=generation requested_model_id=claude-opus-5 "
                "response_model_id=claude-opus-5 call_id=generation-1",
                "model_call event=start role=review "
                "requested_model_id=deepseek-v4-flash call_id=review-1",
                "model_call event=end role=review requested_model_id=deepseek-v4-flash "
                "response_model_id=deepseek-v4-flash call_id=review-1",
            )
        )
        + "\n",
        encoding="utf-8",
    )
    routing_audit = _assert_model_routing_audit(
        log_path,
        generation_model_id="claude-opus-5",
        review_model_id="deepseek-v4-flash",
        evidence_dir=tmp_path,
    )
    assert routing_audit["routes"]["generation"]["completed_calls"] == 1
    assert routing_audit["routes"]["review"]["completed_calls"] == 1

    log_path.write_text(
        "model_call event=identity_mismatch role=review "
        "requested_model_id=deepseek-v4-flash response_model_ids=['fallback'] "
        "call_id=review-2\n",
        encoding="utf-8",
    )
    with pytest.raises(AssertionError, match="identity_mismatch"):
        _assert_model_routing_audit(
            log_path,
            generation_model_id="claude-opus-5",
            review_model_id="deepseek-v4-flash",
            evidence_dir=tmp_path,
        )

    unsafe_dir = tmp_path / "unsafe"
    unsafe_dir.mkdir()
    (unsafe_dir / "nested.sqlite3").write_bytes(b"provider-secret-value")
    with pytest.raises(AssertionError, match="nested.sqlite3"):
        _assert_evidence_has_no_secrets(unsafe_dir, "provider-secret-value")
    assert not unsafe_dir.exists()

    unsafe_dir.mkdir()
    (unsafe_dir / "nested.sqlite3").write_bytes(b"sk-another-secret-value")
    with pytest.raises(AssertionError, match="nested.sqlite3"):
        _assert_evidence_has_no_secrets(unsafe_dir, None)
    assert not unsafe_dir.exists()


def test_live_e2e_preflight_failure_creates_no_evidence_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact_root = tmp_path / "live-e2e"
    monkeypatch.setenv("PENGINE_LIVE_E2E_ARTIFACT_ROOT", str(artifact_root))
    settings = Settings(
        _env_file=None,
        persona_root=tmp_path / "missing-personas",
    )

    with pytest.raises(
        pytest.fail.Exception,
        match="requires configured generation and review routes",
    ):
        _prepare_live_e2e_run(
            settings,
            repository_root=tmp_path,
            run_label="must-not-exist",
        )

    assert not artifact_root.exists()


@pytest.mark.live_model
@pytest.mark.asyncio
async def test_real_model_outline_patch_contract() -> None:
    if os.getenv(_ENABLE_ENV) != "1":
        pytest.skip(f"set {_ENABLE_ENV}=1 to make real, potentially billable model requests")

    settings = Settings()
    _require_dual_model_relay(settings, "Real-model outline-patch probe")
    candidate = _outline_probe_candidate()
    review = CanonReviewerResult(
        passed=False,
        evidence="可读大纲与角色合同冲突。",
        issues=[
            {
                "code": "role_mismatch",
                "message": "把可读大纲中的旁观者改为调查者。",
                "contract_refs": ["lin_xia"],
            }
        ],
    )
    adapter = build_relay_adapter(settings, role="generation")
    structured = adapter.model.with_structured_output(
        OutlineRepairPatch,
        method="function_calling",
        include_raw=True,
    )
    repair_context = _outline_repair_context(candidate, review)
    response = await structured.ainvoke(
        [
            {
                "role": "system",
                "content": (
                    "Return exactly one minimal OutlineRepairPatch tool call. Replace the exact "
                    "Chinese word 旁观者 with 调查者 in content. Do not repeat the candidate and "
                    "do not modify JSON."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    repair_context,
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
            },
        ]
    )
    patch = _outline_repair_result(response)
    _validate_outline_repair_patch_targets(patch, repair_context)

    repaired = _apply_outline_repair_patch(candidate, patch)

    assert isinstance(patch, OutlineRepairPatch)
    assert repaired.content == "第 1 集：林夏只是调查者。"


@pytest.mark.live_model
@pytest.mark.asyncio
async def test_real_model_story_artifact_repair_contract() -> None:
    if os.getenv(_ENABLE_ENV) != "1":
        pytest.skip(f"set {_ENABLE_ENV}=1 to make real, potentially billable model requests")

    settings = Settings()
    _require_dual_model_relay(settings, "Real-model story-artifact-patch probe")
    stage = InternalStage.GENERATING_RELATIONSHIP_LOGIC
    content = (
        "# 人物关系\n"
        "程远二十四岁，比程屿大六岁。\n"
        "两人的年龄关系影响程屿对兄长的依赖和调查选择。\n"
        "其余人物身份、秘密来源、时间线和证物约束均保持已批准版本。"
    )
    review = CanonReviewerResult(
        passed=False,
        evidence="关系稿年龄与已批准人物小传冲突。",
        issues=[
            {
                "code": "relative_age_conflict",
                "message": "权威人物小传为程远二十二岁、两人相差两岁。",
                "script_excerpt": "程远二十四岁，比程屿大六岁。",
            }
        ],
    )
    structured = build_relay_adapter(settings, role="generation").model.with_structured_output(
        StoryArtifactRepairPatch,
        method="function_calling",
        include_raw=True,
    )
    response = await structured.ainvoke(
        [
            {
                "role": "system",
                "content": (
                    "Return exactly one StoryArtifactRepairPatch tool call and no prose. Use the "
                    "1-based candidate_lines. Repair only line 2: replace 二十四 with 二十二 and "
                    "六 with 两. Return line 2's complete corrected text as replacement. Do not "
                    "return or change any unrelated line."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    _story_repair_context(stage=stage, content=content, review=review),
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
            },
        ]
    )
    patch = _story_repair_result(response, stage=stage)
    repaired = _apply_story_artifact_repair_patch(
        stage=stage,
        content=content,
        patch=patch,
    )

    assert isinstance(patch, StoryArtifactRepairPatch)
    assert [(item.start_line, item.end_line) for item in patch.line_replacements] == [(2, 2)]
    assert "程远二十二岁，比程屿大两岁" in repaired.content


@pytest.mark.live_model
@pytest.mark.asyncio
async def test_real_model_prior_story_issue_closure_contract() -> None:
    if os.getenv(_ENABLE_ENV) != "1":
        pytest.skip(f"set {_ENABLE_ENV}=1 to make real, potentially billable model requests")

    settings = Settings()
    _require_dual_model_relay(settings, "Real-model prior-story-issue-closure probe")
    prior = CanonReviewerResult(
        passed=False,
        evidence="人物年龄与批准小传冲突。",
        issues=[
            {
                "code": "relative_age_conflict",
                "message": "批准小传写程远二十二岁，旧候选误写二十四岁。",
                "script_excerpt": "程远二十四岁。",
            }
        ],
    )
    ledger = _canon_review_with_issue_ledger(prior)
    expected_issue_id = ledger["issue_ledger"][0]["issue_id"]
    structured = build_relay_adapter(settings, role="review").model.with_structured_output(
        CanonReviewerResult,
        method="function_calling",
    )
    review = await structured.ainvoke(
        [
            {
                "role": "system",
                "content": (
                    "独立审核当前候选，并只返回 CanonReviewerResult 结构化结果。"
                    "previous_story_review.issue_ledger 中每个 issue_id 都必须在 "
                    "prior_issue_closures 中恰好出现一次。只有当前候选和批准值证明冲突"
                    "完全消失时才标记 resolved；不要返回未知 ID。"
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "approved_biography": "程远二十二岁，比程屿大两岁。",
                        "current_candidate": "程远二十二岁，比程屿大两岁。",
                        "previous_story_review": ledger,
                    },
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
            },
        ]
    )

    assert review.passed is True
    assert review.issues == []
    assert [(item.issue_id, item.status) for item in review.prior_issue_closures] == [
        (expected_issue_id, "resolved")
    ]


@pytest.mark.live_model
@pytest.mark.asyncio
async def test_real_model_dense_story_artifact_repair_contract() -> None:
    if os.getenv(_ENABLE_ENV) != "1":
        pytest.skip(f"set {_ENABLE_ENV}=1 to make real, potentially billable model requests")

    settings = Settings()
    _require_dual_model_relay(settings, "Real-model dense story-artifact-patch probe")
    stage = InternalStage.GENERATING_STORY_OUTLINE
    target_lines = [f"- 事实{i:02d}：状态=旧。" for i in range(1, 18)]
    padding_lines = [
        f"- 背景{i:02d}：这段较长背景保持不变，用于约束修复范围并保护候选正文。"
        for i in range(1, 31)
    ]
    content = "\n".join(["# 密集修复测试", *target_lines, *padding_lines])
    review = CanonReviewerResult(
        passed=False,
        evidence="十七个独立事实位置均含同一已确认状态错误。",
        issues=[
            {
                "code": f"dense_fact_{i:02d}",
                "message": f"第 {i + 1} 行的状态必须从旧改为新。",
                "script_excerpt": target_lines[i - 1],
            }
            for i in range(1, 18)
        ],
    )
    structured = build_relay_adapter(settings, role="generation").model.with_structured_output(
        StoryArtifactRepairPatch,
        method="function_calling",
        include_raw=True,
    )
    response = await structured.ainvoke(
        [
            {
                "role": "system",
                "content": (
                    "Return exactly one StoryArtifactRepairPatch tool call and no prose. Return "
                    "exactly seventeen separate one-line replacements for candidate lines 2 "
                    "through 18, in order; do not combine ranges. In each selected line, replace "
                    "only 状态=旧 with 状态=新 and return that line's complete corrected text. "
                    "Do not include or change any other line."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    _story_repair_context(stage=stage, content=content, review=review),
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
            },
        ]
    )
    patch = _story_repair_result(response, stage=stage)
    repaired = _apply_story_artifact_repair_patch(
        stage=stage,
        content=content,
        patch=patch,
    )

    assert isinstance(patch, StoryArtifactRepairPatch)
    assert [(item.start_line, item.end_line) for item in patch.line_replacements] == [
        (line_number, line_number) for line_number in range(2, 19)
    ]
    assert "状态=旧" not in repaired.content
    assert repaired.content.count("状态=新") == 17


@pytest.mark.live_model
@pytest.mark.asyncio
async def test_real_model_corrects_structured_schema_error() -> None:
    if os.getenv(_ENABLE_ENV) != "1":
        pytest.skip(f"set {_ENABLE_ENV}=1 to make real, potentially billable model requests")

    settings = Settings()
    _require_dual_model_relay(settings, "Real-model schema-correction probe")
    invalid_args = copy.deepcopy(_outline_probe_candidate())
    invalid_args["story_contract"]["facts"][0].update(
        kind="date",
        value="第0天",
        unit="天",
    )
    invalid_message = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "EpisodePlannerResult",
                "args": invalid_args,
                "id": "invalid-outline-result",
                "type": "tool_call",
            }
        ],
    )
    generic_error = ToolMessage(
        content=(
            "Return exactly one valid structured result tool call for the requested stage. "
            "Do not return the result as prose."
        ),
        tool_call_id="invalid-outline-result",
        name="EpisodePlannerResult",
    )
    agent = create_agent(
        build_relay_adapter(settings, role="generation").model,
        tools=[],
        middleware=[StructuredResultMiddleware()],
        response_format=ToolStrategy(
            EpisodePlannerResult,
            handle_errors=_structured_output_retry_message,
        ),
    )

    result = await agent.ainvoke({"messages": [invalid_message, generic_error]})
    parsed = result["structured_response"]

    assert isinstance(parsed, EpisodePlannerResult)
    repaired = parsed.story_contract.facts[0]
    assert (repaired.kind, repaired.value, repaired.unit) != ("date", "第0天", "天")


@pytest.mark.live_model
@pytest.mark.asyncio
async def test_real_model_corrects_null_episode_state_delta() -> None:
    if os.getenv(_ENABLE_ENV) != "1":
        pytest.skip(f"set {_ENABLE_ENV}=1 to make real, potentially billable model requests")

    settings = Settings()
    _require_dual_model_relay(settings, "Real-model episode-state probe")
    invalid_message = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "ScriptWriterResult",
                "args": {
                    "stage": "generating_episode_scripts",
                    "episode_number": 1,
                    "content": "第1集：林夏发现旧怀表。",
                    "state_delta": None,
                },
                "id": "missing-episode-state",
                "type": "tool_call",
            }
        ],
    )
    generic_error = ToolMessage(
        content=(
            "Return exactly one valid structured result tool call for the requested stage. "
            "Do not return the result as prose."
        ),
        tool_call_id="missing-episode-state",
        name="ScriptWriterResult",
    )
    agent = create_agent(
        build_relay_adapter(settings, role="generation").model,
        tools=[],
        middleware=[StructuredResultMiddleware()],
        response_format=ToolStrategy(
            ScriptWriterResult,
            handle_errors=_structured_output_retry_message,
        ),
    )

    result = await agent.ainvoke({"messages": [invalid_message, generic_error]})
    parsed = result["structured_response"]

    assert isinstance(parsed, ScriptWriterResult)
    assert parsed.state_delta.episode_number == 1


@pytest.mark.live_model
def test_real_model_initial_creation_black_box() -> None:
    if os.getenv(_ENABLE_ENV) != "1":
        pytest.skip(f"set {_ENABLE_ENV}=1 to make real, potentially billable model requests")

    settings = Settings()
    repository_root = Path(__file__).resolve().parents[2]
    run_label = f"{datetime.now(UTC):%Y%m%dT%H%M%SZ}-{uuid4().hex[:8]}"
    evidence_dir, data_dir, timeout_seconds, poll_seconds = _prepare_live_e2e_run(
        settings,
        repository_root=repository_root,
        run_label=run_label,
    )
    key_present = settings.relay_api_key is not None and bool(
        settings.relay_api_key.get_secret_value()
    )
    assert settings.generation_model_id is not None
    assert settings.review_model_id is not None
    metadata: dict[str, Any] = {
        "run_label": run_label,
        "started_at": _now(),
        "status": "starting",
        "python": sys.version.split()[0],
        "relay": {
            "base_url": _safe_base_url(settings.relay_base_url),
            "key_present": key_present,
            "generation": {
                "protocol": "anthropic",
                "model_id": settings.generation_model_id,
                "max_output_tokens": settings.generation_max_output_tokens,
            },
            "review": {
                "protocol": "deepseek",
                "model_id": settings.review_model_id,
                "max_output_tokens": settings.review_max_output_tokens,
            },
        },
        "server": {"host": "127.0.0.1", "port": None},
        "data_dir": str(data_dir),
    }
    _write_json(evidence_dir / "metadata.json", metadata)

    port = _reserve_loopback_port()
    metadata["server"]["port"] = port
    metadata["timeout_seconds"] = timeout_seconds
    metadata["child_run_timeout_seconds"] = timeout_seconds
    metadata["poll_seconds"] = poll_seconds
    _write_json(evidence_dir / "metadata.json", metadata)

    log_path = evidence_dir / "server.log"
    polls_path = evidence_dir / "polls.jsonl"
    timeline_path = evidence_dir / "timeline.jsonl"
    polls_path.touch()
    timeline_path.touch()
    process: subprocess.Popen[bytes] | None = None
    log_output = log_path.open("wb")
    secret = settings.relay_api_key.get_secret_value() if settings.relay_api_key else None
    caught: BaseException | None = None
    try:
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "uvicorn",
                "pengine.main:app",
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
                "--log-level",
                "info",
            ],
            cwd=repository_root,
            env=_child_environment(
                settings,
                data_dir,
                port,
                run_timeout_seconds=timeout_seconds,
            ),
            stdout=log_output,
            stderr=subprocess.STDOUT,
        )
        base_url = f"http://127.0.0.1:{port}"
        with httpx.Client(base_url=base_url, timeout=10.0) as client:
            personas = _wait_until_ready(
                client,
                process,
                deadline=time.monotonic() + 30.0,
                log_path=log_path,
            )
            _write_json(evidence_dir / "personas.json", personas)
            items = personas.get("items", [])
            assert items, f"GET /personas returned no personas; evidence: {evidence_dir}"

            requested_persona = os.getenv("PENGINE_LIVE_E2E_PERSONA_ID")
            available = {item["persona_id"] for item in items}
            persona_id = requested_persona or sorted(available)[0]
            assert persona_id in available, (
                f"Requested persona {persona_id!r} is unavailable; choices={sorted(available)}"
            )
            metadata["persona_id"] = persona_id

            creation_response = client.post(
                "/creations",
                headers={"Idempotency-Key": f"live-e2e-{run_label}"},
                json={
                    "persona_id": persona_id,
                    "story": os.getenv("PENGINE_LIVE_E2E_STORY", _DEFAULT_STORY),
                    "requirements": os.getenv(
                        "PENGINE_LIVE_E2E_REQUIREMENTS", _DEFAULT_REQUIREMENTS
                    ),
                },
            )
            creation_body = _response_body(creation_response)
            _write_json(evidence_dir / "create-response.json", creation_body)
            assert creation_response.status_code == 202, (
                f"POST /creations returned HTTP {creation_response.status_code}; "
                f"evidence: {evidence_dir}"
            )
            assert isinstance(creation_body, dict)
            metadata["creation_id"] = creation_body.get("creation_id")
            _write_json(evidence_dir / "metadata.json", metadata)

            final_resource = _poll_initial(
                client,
                resource_url=creation_body["resource_url"],
                timeout_seconds=timeout_seconds,
                poll_seconds=poll_seconds,
                polls_path=polls_path,
                timeline_path=timeline_path,
                final_resource_path=evidence_dir / "final-resource.json",
            )
            _assert_success(final_resource, evidence_dir=evidence_dir)
            _assert_story_consistency_checkpoints(
                data_dir / "pengine.sqlite3",
                creation_id=str(creation_body["creation_id"]),
                evidence_dir=evidence_dir,
            )
            process.terminate()
            process.wait(timeout=15)
            log_output.close()
            _redact_log(log_path, secret)
            routing_audit = _assert_model_routing_audit(
                log_path,
                generation_model_id=settings.generation_model_id,
                review_model_id=settings.review_model_id,
                evidence_dir=evidence_dir,
            )
            _write_json(evidence_dir / "model-routing-audit.json", routing_audit)
            usage_audit = _assert_durable_usage_evidence(
                log_path,
                data_dir / "pengine.sqlite3",
                evidence_dir=evidence_dir,
            )
            _write_json(evidence_dir / "model-usage-audit.json", usage_audit)
            metadata["model_routing_audit"] = routing_audit
            metadata["model_usage_audit"] = usage_audit
            metadata["status"] = "succeeded"
    except BaseException as error:
        caught = error
        metadata["status"] = "failed"
        metadata["error_type"] = type(error).__name__
        message = str(error)
        if secret:
            message = message.replace(secret, "[REDACTED]")
        metadata["error"] = message
        raise
    finally:
        metadata["finished_at"] = _now()
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
        if process is not None:
            metadata["server_exit_code"] = process.returncode
        log_output.close()
        _redact_log(log_path, secret)
        if caught is None and metadata["status"] != "succeeded":
            metadata["status"] = "failed"
        _write_json(evidence_dir / "metadata.json", metadata)
        _assert_evidence_has_no_secrets(evidence_dir, secret)


@pytest.mark.live_model
@pytest.mark.asyncio
async def test_real_model_series_bible_design_review_binding() -> None:
    """Opt-in probe proving Opus generation + DeepSeek review role routing and
    candidate binding (SDP-A11).

    The Opus generation route produces an EpisodePlannerResult with a story
    contract; the DeepSeek review route produces a global design review; the
    review record binds the exact candidate id and content hash. No promotion
    can occur without a review bound to that exact candidate.
    """
    if os.getenv(_ENABLE_ENV) != "1":
        pytest.skip(f"set {_ENABLE_ENV}=1 to make real, potentially billable model requests")

    settings = Settings()
    _require_dual_model_relay(settings, "Real-model SeriesBible binding probe")

    from pengine.agents import CanonReviewerResult, EpisodePlannerResult
    from pengine.series_bible import (
        bind_global_design_review,
        build_series_bible,
        detect_genre,
        validate_series_bible,
    )

    story = "林夏回到家乡，在旧屋整理遗物时决定处理一段家族往事。"
    requirements = "创作两集现实主义短剧；人物、时间线与证据全程一致；全部使用简体中文。"
    assert detect_genre(story, requirements) == "general"

    generation = build_relay_adapter(settings, role="generation")
    structured = generation.model.with_structured_output(
        EpisodePlannerResult,
        method="function_calling",
        include_raw=True,
    )
    response = await structured.ainvoke(
        [
            {
                "role": "system",
                "content": (
                    "一次性返回 EpisodePlannerResult 结构化结果，不要输出任何正文。"
                    "自动编译最小连续性台账：人物与事实使用唯一小写 snake_case ID；"
                    "时间线从 1 开始连续编号；每集恰好一条义务；"
                    "没有证据支持的事实不要编造。"
                ),
            },
            {
                "role": "user",
                "content": f"故事：{story}\n要求：{requirements}",
            },
        ]
    )
    raw = response.get("raw", {})
    message_text = ""
    if isinstance(raw, dict) and isinstance(raw.get("raw"), list):
        blocks = [b for b in raw["raw"] if isinstance(b, dict)]
        message_text = "".join(str(b.get("text", "")) for b in blocks)
    if not message_text:
        parsed = response.get("parsed")
        if isinstance(parsed, EpisodePlannerResult):
            candidate_bible = build_series_bible(
                run_id="live-series-bible-probe",
                run_kind="initial",
                l0_variant="现实叙事",
                genre=detect_genre(story, requirements),
                story_outline="故事大纲",
                character_biographies="\n".join(
                    f"{character.name}：主要人物。"
                    for character in parsed.story_contract.characters
                ),
                relationship_logic="关系逻辑",
                episode_outline=parsed.content,
                story_contract_payload=parsed.story_contract.model_dump(mode="json"),
            )
        else:
            candidate_bible = None
    else:
        parsed = EpisodePlannerResult.model_validate_json(message_text)
        candidate_bible = build_series_bible(
            run_id="live-series-bible-probe",
            run_kind="initial",
            l0_variant="现实叙事",
            genre=detect_genre(story, requirements),
            story_outline="故事大纲",
            character_biographies="\n".join(
                f"{character.name}：主要人物。" for character in parsed.story_contract.characters
            ),
            relationship_logic="关系逻辑",
            episode_outline=parsed.content,
            story_contract_payload=parsed.story_contract.model_dump(mode="json"),
        )

    assert candidate_bible is not None, "The Opus generation route returned no EpisodePlannerResult"
    evidence = validate_series_bible(candidate_bible)
    assert evidence.passed, evidence.issues

    review_adapter = build_relay_adapter(settings, role="review")
    review_model = review_adapter.model.with_structured_output(
        CanonReviewerResult,
        method="function_calling",
    )
    review = await review_model.ainvoke(
        [
            {
                "role": "system",
                "content": (
                    "你是 DeepSeek 全局设计审核者。只返回 CanonReviewerResult 结构化结果。"
                    "审核整个设计候选：角色、关系、事实、时间线、分集义务与投影一致性。"
                    "不要要求上游留白的事实；不要修改任何候选。"
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "candidate_id": candidate_bible.candidate_id,
                        "content_hash": candidate_bible.content_hash,
                        "projections": {
                            "story_outline": candidate_bible.content.story_outline,
                            "character_biographies": candidate_bible.content.character_biographies,
                            "relationship_logic": candidate_bible.content.relationship_logic,
                            "episode_outline": candidate_bible.content.episode_outline,
                        },
                        "story_contract": candidate_bible.content.story_contract.model_dump(
                            mode="json"
                        ),
                    },
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
            },
        ]
    )

    bound = bind_global_design_review(
        candidate_bible,
        review_call_id=f"live-probe-{uuid4().hex}",
        review_model_id=settings.review_model_id,
        passed=review.passed,
        evidence=review.evidence,
        issues=[issue.model_dump(mode="json") for issue in review.issues],
    )
    assert bound.candidate_id == candidate_bible.candidate_id
    assert bound.candidate_hash == candidate_bible.content_hash
    assert bound.review_call_id.startswith("live-probe-")
    assert bound.review_model_id == settings.review_model_id
