"""Regression tests for the model-call budget + durable usage observability envelope.

Covers the fail-closed preflight path (oversized request -> zero outbound calls, run
pauses safely), estimate vs actual vs unavailable usage, call_id lineage, SQLite restart
durability, superseded/stale classification, and credential safety.
"""

from __future__ import annotations

import re
import sqlite3
from pathlib import Path
from typing import Any
from uuid import uuid4

import httpx
import openai
import pytest
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, LLMResult

from pengine.config import Settings
from pengine.model_calls import (
    ModelCallContext,
    ModelCallState,
    ModelCallStore,
    build_started_record,
    estimate_text_tokens,
    extract_provider_usage,
    usage_status_from,
)
from pengine.relay import (
    PreflightBlockedError,
    _ModelCallAuditHandler,
    build_relay_adapter,
    build_relay_routes,
)


def _settings(
    *,
    generation_context_limit: int | None = None,
    review_context_limit: int | None = None,
    relay_base_url: str = "https://relay.example/v1",
) -> Settings:
    return Settings(
        _env_file=None,
        relay_base_url=relay_base_url,
        relay_api_key="secret-value",
        generation_model_id="claude-opus-5",
        review_model_id="deepseek-v4-flash",
        generation_context_limit_tokens=generation_context_limit,
        review_context_limit_tokens=review_context_limit,
    )


def _messages(blocks: int = 1, chars_per_block: int = 20) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": "你是一个短剧创作助手。" * blocks},
        {"role": "user", "content": "请创作完整的故事。 " + "字" * chars_per_block},
    ]


def test_token_estimator_is_deterministic_and_sized() -> None:
    assert estimate_text_tokens("") == 0
    assert estimate_text_tokens("这是一段中文") == 6
    assert estimate_text_tokens("hello world") == 3
    assert estimate_text_tokens("a" * 4000) == 1000
    # 4 CJK chars + "english" (7 non-CJK -> ceil(7/4)=2) = 6
    assert estimate_text_tokens("中文english混合") == 6
    assert estimate_text_tokens("中" * 1000) == 1000


def test_preflight_blocks_oversized_request_with_zero_outbound_calls(
    tmp_path: Path,
) -> None:
    """A request that cannot fit the verified limit is never dispatched."""
    store = ModelCallStore(tmp_path / "model_calls.sqlite3")
    state = ModelCallState(store=store)
    state.context.run_id = "run-overflow"
    state.context.stage = "generating_story_outline"
    settings = _settings(
        generation_context_limit=1,
        relay_base_url="http://127.0.0.1:9/v1",  # unreachable: a sent request would error
    )
    adapter = build_relay_adapter(
        settings,
        role="generation",
        model_call_state=state,
    )

    with pytest.raises(PreflightBlockedError) as excinfo:
        adapter.model.invoke(_messages(blocks=5))

    assert excinfo.value.code == "preflight_blocked"
    assert excinfo.value.verified_limit_tokens == 1
    assert excinfo.value.required_tokens > 1
    # No HTTP request reached the (unreachable) relay; the failure is the preflight.
    assert "connection" not in type(excinfo.value).__name__.lower()

    rows = store._connection.execute(
        """
        SELECT status, preflight, stage, run_id, estimated_total_tokens, verified_limit_tokens
        FROM model_calls
        """
    ).fetchall()
    assert len(rows) == 1
    assert rows[0]["status"] == "preflight_blocked"
    assert rows[0]["preflight"] == "blocked"
    assert rows[0]["stage"] == "generating_story_outline"
    assert rows[0]["run_id"] == "run-overflow"
    assert rows[0]["estimated_total_tokens"] > 1
    assert rows[0]["verified_limit_tokens"] == 1
    store.close()


def test_preflight_fails_closed_when_no_verified_limit_exists(tmp_path: Path) -> None:
    """A route without a trustworthy verified limit blocks every outbound call."""
    store = ModelCallStore(tmp_path / "model_calls.sqlite3")
    state = ModelCallState(store=store)
    state.context.run_id = "run-unverified"
    settings = _settings(
        generation_context_limit=None,
        relay_base_url="http://127.0.0.1:9/v1",
    )
    adapter = build_relay_adapter(
        settings,
        role="generation",
        model_call_state=state,
    )

    with pytest.raises(PreflightBlockedError) as excinfo:
        adapter.model.invoke(_messages())

    assert excinfo.value.verified_limit_tokens is None
    row = store._connection.execute(
        "SELECT status FROM model_calls WHERE run_id = 'run-unverified'"
    ).fetchone()
    assert row["status"] == "preflight_blocked"
    store.close()


def test_preflight_allows_a_fitting_request_to_reach_the_relay(tmp_path: Path) -> None:
    """A fitting request is sent; the relay/transport error is recorded as failed."""
    store = ModelCallStore(tmp_path / "model_calls.sqlite3")
    state = ModelCallState(store=store)
    state.context.run_id = "run-fits"
    state.context.stage = "generating_story_outline"
    settings = _settings(
        generation_context_limit=200_000,
        relay_base_url="http://127.0.0.1:9/v1",
    )
    adapter = build_relay_adapter(
        settings,
        role="generation",
        model_call_state=state,
    )

    with pytest.raises(Exception) as excinfo:
        adapter.model.invoke([{"role": "user", "content": "你好。"}])

    assert not isinstance(excinfo.value, PreflightBlockedError)
    row = store._connection.execute(
        "SELECT status, preflight, error_type FROM model_calls WHERE run_id = 'run-fits'"
    ).fetchone()
    assert row["status"] == "failed"
    assert row["preflight"] == "ok"
    assert row["error_type"]
    store.close()


def _usage_response(*, usage_metadata: dict[str, Any] | None, llm_output: dict[str, Any] | None):
    message = AIMessage(
        content="ok",
        response_metadata={"finish_reason": "stop"},
        usage_metadata=usage_metadata,
    )
    return LLMResult(
        generations=[[ChatGeneration(message=message)]],
        llm_output=llm_output,
    )


def _audited_response(model_id: str, *, input_tokens: int, output_tokens: int) -> LLMResult:
    return LLMResult(
        generations=[
            [
                ChatGeneration(
                    message=AIMessage(
                        content="ok",
                        response_metadata={"model": model_id, "finish_reason": "stop"},
                        usage_metadata={
                            "input_tokens": input_tokens,
                            "output_tokens": output_tokens,
                            "total_tokens": input_tokens + output_tokens,
                        },
                    )
                )
            ]
        ]
    )


def test_each_provider_request_uses_callback_id_as_physical_ledger_key(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """One logical operation may span roles/stages without collapsing physical calls."""
    caplog.set_level("INFO", logger="uvicorn.error.pengine.model_calls")
    caplog.set_level("INFO", logger="uvicorn.error.pengine.model_call_records")
    store = ModelCallStore(tmp_path / "model_calls.sqlite3")
    operation_id = "episode-3-operation"
    state = ModelCallState(store=store)
    state.context.run_id = "run-physical"
    state.context.stage = "generating_episode_scripts"
    state.context.episode_number = 3
    state.context.operation_id = operation_id
    generation = _ModelCallAuditHandler(
        role="generation",
        model_id="claude-opus-5",
        adapter="anthropic",
        provider="anthropic",
        model_call_state=state,
        context_limit_tokens=200_000,
        reserved_output_tokens=100,
    )
    review = _ModelCallAuditHandler(
        role="review",
        model_id="gpt-5.5",
        adapter="openai",
        provider="openai",
        model_call_state=state,
        context_limit_tokens=200_000,
        reserved_output_tokens=100,
    )
    physical_calls = [uuid4(), uuid4(), uuid4()]

    generation.on_chat_model_start({}, [[AIMessage(content="write")]], run_id=physical_calls[0])
    generation.on_llm_end(
        _audited_response("claude-opus-5", input_tokens=101, output_tokens=11),
        run_id=physical_calls[0],
    )
    review.on_chat_model_start({}, [[AIMessage(content="review")]], run_id=physical_calls[1])
    review.on_llm_end(
        _audited_response("gpt-5.5", input_tokens=102, output_tokens=12),
        run_id=physical_calls[1],
    )
    state.context.stage = "accepting_l4"
    state.context.episode_number = None
    generation.on_chat_model_start({}, [[AIMessage(content="gate")]], run_id=physical_calls[2])
    generation.on_llm_end(
        _audited_response("claude-opus-5", input_tokens=103, output_tokens=13),
        run_id=physical_calls[2],
    )

    rows = store._connection.execute(
        "SELECT call_id, operation_id, role, stage, status, actual_input_tokens, "
        "actual_output_tokens FROM model_calls ORDER BY requested_at, call_id"
    ).fetchall()
    assert {row["call_id"] for row in rows} == {str(call_id) for call_id in physical_calls}
    assert len(rows) == len(physical_calls)
    assert {row["operation_id"] for row in rows} == {operation_id}
    assert all(row["status"] == "succeeded" for row in rows)
    assert sum(row["actual_input_tokens"] for row in rows) == 306
    assert sum(row["actual_output_tokens"] for row in rows) == 36
    assert state.latest_succeeded_call_id(
        role="review",
        run_id="run-physical",
        stage="generating_episode_scripts",
        episode_number=3,
        operation_id=operation_id,
    ) == str(physical_calls[1])
    for physical_call_id in map(str, physical_calls):
        assert "event=start role=" in caplog.text
        assert f"call_id={physical_call_id}" in caplog.text
    with pytest.raises(RuntimeError, match="Duplicate physical model call id"):
        generation.on_chat_model_start(
            {},
            [[AIMessage(content="duplicate")]],
            run_id=physical_calls[0],
        )
    store.close()


def test_provider_usage_is_extracted_exactly_when_present() -> None:
    tokens, finish_reason = extract_provider_usage(
        _usage_response(
            usage_metadata={
                "input_tokens": 123,
                "output_tokens": 45,
                "total_tokens": 168,
                "input_token_details": {"cache_read": 10},
            },
            llm_output=None,
        )
    )
    assert tokens == {
        "input_tokens": 123,
        "output_tokens": 45,
        "cache_read_tokens": 10,
        "cache_creation_tokens": None,
    }
    assert finish_reason == "stop"
    assert usage_status_from(tokens) == "reported"


def test_provider_usage_partial_is_never_backfilled_from_estimates() -> None:
    # OpenAI-compatible token_usage with only prompt tokens, no completion tokens.
    tokens, _ = extract_provider_usage(
        _usage_response(
            usage_metadata=None,
            llm_output={"token_usage": {"prompt_tokens": 77}},
        )
    )
    assert tokens["input_tokens"] == 77
    assert tokens["output_tokens"] is None
    assert usage_status_from(tokens) == "partial"


def test_provider_usage_missing_is_unavailable_never_inferred() -> None:
    tokens, _ = extract_provider_usage(_usage_response(usage_metadata=None, llm_output=None))
    assert tokens == {
        "input_tokens": None,
        "output_tokens": None,
        "cache_read_tokens": None,
        "cache_creation_tokens": None,
    }
    assert usage_status_from(tokens) == "unavailable"


def test_provider_cache_usage_maps_deepseek_and_openai_keys() -> None:
    tokens, _ = extract_provider_usage(
        _usage_response(
            usage_metadata=None,
            llm_output={
                "token_usage": {
                    "prompt_tokens": 100,
                    "completion_tokens": 20,
                    "prompt_cache_hit_tokens": 40,
                    "prompt_cache_miss_tokens": 60,
                }
            },
        )
    )
    assert tokens["cache_read_tokens"] == 40
    assert tokens["cache_creation_tokens"] == 60

    tokens, _ = extract_provider_usage(
        _usage_response(
            usage_metadata=None,
            llm_output={
                "token_usage": {
                    "prompt_tokens": 100,
                    "completion_tokens": 20,
                    "prompt_tokens_details": {"cached_tokens": 30},
                }
            },
        )
    )
    assert tokens["cache_read_tokens"] == 30


def test_current_started_call_is_never_self_superseded(tmp_path: Path) -> None:
    """A new call supersedes a prior still-started call, never itself."""
    from uuid import uuid4

    from pengine.relay import _ModelCallAuditHandler

    store = ModelCallStore(tmp_path / "model_calls.sqlite3")
    state = ModelCallState(store=store)
    state.context.run_id = "run-supersede"
    state.context.stage = "generating_story_outline"
    settings = _settings(
        generation_context_limit=200_000,
        relay_base_url="http://127.0.0.1:9/v1",
    )
    adapter = build_relay_adapter(settings, role="generation", model_call_state=state)
    handler = adapter.model.callbacks[0]
    assert isinstance(handler, _ModelCallAuditHandler)

    messages = [[{"role": "user", "content": "你好。"}]]
    handler.on_chat_model_start({}, messages, run_id=uuid4())
    handler.on_chat_model_start({}, messages, run_id=uuid4())

    rows = store._connection.execute(
        "SELECT status, outcome FROM model_calls ORDER BY requested_at"
    ).fetchall()
    assert len(rows) == 2
    # The prior still-started call is superseded; the current one stays started.
    assert rows[0]["status"] == "superseded"
    assert rows[0]["outcome"] == "superseded"
    assert rows[1]["status"] == "started"
    assert rows[1]["outcome"] == "incomplete"
    store.close()


def test_store_persists_records_across_restart_and_supersedes_overlap(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "model_calls.sqlite3"
    context = ModelCallContext(run_id="run-1", stage="generating_story_outline")

    store = ModelCallStore(db_path)
    first = build_started_record(
        role="generation",
        adapter="anthropic",
        provider="anthropic",
        model="claude-opus-5",
        context=context,
        estimated_input_tokens=10,
        estimated_output_tokens=100,
        verified_limit_tokens=200_000,
    )
    store.upsert(first)
    store.mark_superseded_pending(run_id="run-1", role="generation")
    second = build_started_record(
        role="generation",
        adapter="anthropic",
        provider="anthropic",
        model="claude-opus-5",
        context=context,
        estimated_input_tokens=10,
        estimated_output_tokens=100,
        verified_limit_tokens=200_000,
    )
    second.status = "succeeded"
    second.outcome = "success"
    second.actual_input_tokens = 11
    second.actual_output_tokens = 22
    second.usage_status = "reported"
    store.upsert(second)
    store.close()

    # A restart with a fresh store reads the same durable rows.
    restarted = ModelCallStore(db_path)
    rows = restarted._connection.execute(
        "SELECT status, outcome, actual_input_tokens, actual_output_tokens, usage_status "
        "FROM model_calls ORDER BY requested_at"
    ).fetchall()
    assert rows[0]["status"] == "superseded"
    assert rows[0]["outcome"] == "superseded"
    assert rows[0]["actual_input_tokens"] is None
    assert rows[1]["status"] == "succeeded"
    assert rows[1]["actual_input_tokens"] == 11
    assert rows[1]["actual_output_tokens"] == 22
    assert rows[1]["usage_status"] == "reported"
    restarted.close()


def test_store_finalizes_started_calls_as_timed_out(tmp_path: Path) -> None:
    db_path = tmp_path / "model_calls.sqlite3"
    store = ModelCallStore(db_path)
    context = ModelCallContext(run_id="run-timeout", stage="generating_episode_scripts")
    record = build_started_record(
        role="generation",
        adapter="anthropic",
        provider="anthropic",
        model="claude-opus-5",
        context=context,
        estimated_input_tokens=10,
        estimated_output_tokens=100,
        verified_limit_tokens=200_000,
    )
    store.upsert(record)
    store.mark_timed_out(run_id="run-timeout")
    row = store._connection.execute(
        "SELECT status, outcome, finish_reason FROM model_calls WHERE run_id = 'run-timeout'"
    ).fetchone()
    assert row["status"] == "timed_out"
    assert row["outcome"] == "timeout"
    assert row["finish_reason"] == "timeout"
    store.close()


def test_audit_records_never_contain_credential_material(tmp_path: Path) -> None:
    store = ModelCallStore(tmp_path / "model_calls.sqlite3")
    state = ModelCallState(store=store)
    state.context.run_id = "run-secret"
    settings = _settings(generation_context_limit=1)
    adapter = build_relay_adapter(settings, role="generation", model_call_state=state)

    with pytest.raises(PreflightBlockedError):
        adapter.model.invoke([{"role": "user", "content": "SECRET-PROMPT-CONTENT"}])

    with sqlite3.connect(tmp_path / "model_calls.sqlite3") as connection:
        dumped = "\n".join(
            str(row) for row in connection.execute("SELECT * FROM model_calls").fetchall()
        )
    assert "SECRET-PROMPT-CONTENT" not in dumped
    assert "secret-value" not in dumped
    assert "Authorization" not in dumped
    store.close()


def test_preflight_counts_tools_and_schema_in_the_estimate(tmp_path: Path) -> None:
    """OBS-A2: preflight covers tools/schema overhead, not only user prose."""
    from langchain_core.tools import StructuredTool

    store = ModelCallStore(tmp_path / "model_calls.sqlite3")
    state = ModelCallState(store=store)
    state.context.run_id = "run-tools"
    settings = _settings(
        generation_context_limit=None,  # fail closed so the record shows the estimate
        relay_base_url="http://127.0.0.1:9/v1",
    )
    adapter = build_relay_adapter(settings, role="generation", model_call_state=state)

    def probe(value: str) -> str:
        return value

    tool = StructuredTool.from_function(
        func=probe,
        name="probe_tool",
        description="A structured tool with a long schema description.",
    )
    bound = adapter.model.bind_tools([tool])

    with pytest.raises(PreflightBlockedError) as excinfo:
        bound.invoke([{"role": "user", "content": "你好。"}])

    # The estimate must exceed the plain message size because it includes the
    # serialized tool schema.
    assert excinfo.value.required_tokens > estimate_text_tokens("你好。")
    row = store._connection.execute(
        "SELECT estimated_input_tokens FROM model_calls WHERE run_id = 'run-tools'"
    ).fetchone()
    assert row["estimated_input_tokens"] > estimate_text_tokens("你好。")
    store.close()


def test_build_relay_routes_share_one_model_call_state() -> None:
    routes = build_relay_routes(_settings(generation_context_limit=200_000))
    assert routes.generation.model_call_state is routes.review.model_call_state
    assert routes.generation.model_call_state is routes.model_call_state


def test_model_call_id_is_unique_and_lineaged(tmp_path: Path) -> None:
    db_path = tmp_path / "model_calls.sqlite3"
    store = ModelCallStore(db_path)
    context = ModelCallContext(
        run_id="run-1",
        creation_id="creation-1",
        thread_id="thread-1",
        run_kind="initial",
        stage="generating_episode_scripts",
        episode_number=3,
        operation_id="episode-3-operation",
    )
    first = build_started_record(
        role="generation",
        adapter="anthropic",
        provider="anthropic",
        model="claude-opus-5",
        context=context,
        estimated_input_tokens=10,
        estimated_output_tokens=100,
        verified_limit_tokens=200_000,
    )
    second = build_started_record(
        role="generation",
        adapter="anthropic",
        provider="anthropic",
        model="claude-opus-5",
        context=context,
        estimated_input_tokens=10,
        estimated_output_tokens=100,
        verified_limit_tokens=200_000,
    )
    assert first.call_id != second.call_id
    assert re.fullmatch(r"[0-9a-f-]{36}", first.call_id)
    store.upsert(first)
    store.upsert(second)
    row = store._connection.execute(
        "SELECT operation_id, run_id, creation_id, thread_id, run_kind, stage, episode_number "
        "FROM model_calls WHERE call_id = ?",
        (first.call_id,),
    ).fetchone()
    assert row["operation_id"] == "episode-3-operation"
    assert row["run_id"] == "run-1"
    assert row["creation_id"] == "creation-1"
    assert row["thread_id"] == "thread-1"
    assert row["run_kind"] == "initial"
    assert row["stage"] == "generating_episode_scripts"
    assert row["episode_number"] == 3
    store.close()


_LEGACY_MODEL_CALLS_TABLE_SQL = """
CREATE TABLE model_calls (
    call_id TEXT PRIMARY KEY,
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


def test_store_migrates_legacy_table_without_provider_evidence_columns(
    tmp_path: Path,
) -> None:
    """An existing model_calls table without the provider-evidence columns must be
    upgraded in place so evidence persistence works on real databases (Issue #52)."""
    db_path = tmp_path / "legacy.sqlite3"
    with sqlite3.connect(db_path) as connection:
        connection.execute(_LEGACY_MODEL_CALLS_TABLE_SQL)
        connection.execute(
            "INSERT INTO model_calls (call_id, role, adapter, provider, model, "
            "requested_at, estimated_input_tokens, estimated_output_tokens, "
            "estimated_total_tokens, preflight, status, usage_status, outcome) "
            "VALUES ('legacy-1', 'review', 'deepseek', 'deepseek', 'deepseek-v4-flash', "
            "'2026-08-01T00:00:00+00:00', 1, 1, 2, 'ok', 'failed', 'unavailable', 'failure')"
        )

    store = ModelCallStore(db_path)

    record = build_started_record(
        role="review",
        adapter="deepseek",
        provider="deepseek",
        model="deepseek-v4-flash",
        context=ModelCallContext(run_id="run-migrated", stage="generating_story_outline"),
        estimated_input_tokens=10,
        estimated_output_tokens=100,
        verified_limit_tokens=200_000,
    )
    store.upsert(record)

    rows = store._connection.execute(
        "SELECT call_id, operation_id, http_status, provider_error_code, redacted_response "
        "FROM model_calls ORDER BY requested_at"
    ).fetchall()
    assert len(rows) == 2
    assert rows[0]["call_id"] == "legacy-1"
    assert rows[0]["operation_id"] is None
    assert rows[0]["http_status"] is None
    assert rows[0]["provider_error_code"] is None
    assert rows[0]["redacted_response"] is None
    assert rows[1]["call_id"] != "legacy-1"
    assert rows[1]["operation_id"] is None
    assert rows[1]["http_status"] is None
    assert rows[1]["provider_error_code"] is None
    assert rows[1]["redacted_response"] is None
    store.close()


def test_on_llm_error_persists_redacted_provider_400_evidence(tmp_path: Path) -> None:
    """A failed provider response (status + redacted body) must be durable and
    retrievable from the model_calls table, without leaking credentials (Issue #52)."""
    from pengine.relay import _ModelCallAuditHandler, register_redaction_secret

    register_redaction_secret("secret-value")

    store = ModelCallStore(tmp_path / "model_calls.sqlite3")
    state = ModelCallState(store=store)
    state.context.run_id = "run-provider-400"
    state.context.stage = "generating_story_outline"
    handler = _ModelCallAuditHandler(
        role="review",
        model_id="deepseek-v4-flash",
        adapter="deepseek",
        provider="deepseek",
        model_call_state=state,
        context_limit_tokens=200_000,
        reserved_output_tokens=100,
    )
    run_id = uuid4()
    handler.on_chat_model_start(
        {},
        [[{"role": "user", "content": "你好。"}]],
        run_id=run_id,
    )
    request = httpx.Request("POST", "https://relay.example/v1/chat/completions")
    body = {
        "error": {
            "message": "invalid tool input with secret-value",
            "type": "invalid_request_error",
            "code": "invalid_request_error",
        }
    }
    error = openai.BadRequestError(
        "bad",
        response=httpx.Response(400, request=request, json=body),
        body=body,
    )

    handler.on_llm_error(error, run_id=run_id)

    row = store._connection.execute(
        "SELECT status, outcome, error_type, http_status, provider_error_code, "
        "redacted_response FROM model_calls WHERE run_id = 'run-provider-400'"
    ).fetchone()
    assert row["status"] == "failed"
    assert row["outcome"] == "failure"
    assert row["http_status"] == 400
    assert row["provider_error_code"] == "invalid_request_error"
    assert row["redacted_response"] is not None
    assert "secret-value" not in row["redacted_response"]
    store.close()
