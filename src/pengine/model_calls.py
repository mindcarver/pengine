"""Shared model-call audit envelope for preflight and durable token observability.

Every real model request passes through :class:`pengine.relay._ModelCallAuditHandler`,
which builds a :class:`ModelCallRecord`, enforces a fail-closed context preflight, and
persists estimate / actual-or-unavailable usage / duration / finish reason / outcome /
lineage into the SQLite ``model_calls`` table via :class:`ModelCallStore`.

The module intentionally does not import from ``pengine.relay`` so that relay's audit
handler can import these helpers without a cycle. The fail-closed exception
(:class:`pengine.relay.PreflightBlockedError`) is raised by the relay layer.
"""

from __future__ import annotations

import json
import logging
import re
import sqlite3
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from math import ceil
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from langchain_core.outputs import LLMResult

from pengine.schemas import ModelCallUsage

logger = logging.getLogger(__name__)

# Rough deterministic token heuristic: one token per CJK character and roughly four
# non-CJK characters per token. This mirrors common "~0.75 chars/token" provider
# guidance for mixed Chinese/English prose and stays deterministic for preflight and
# audit tests. It is an *estimate*; provider-reported usage is never derived from it.
_CJK_CHARS = re.compile(r"[㐀-䶿一-鿿豈-﫿぀-ヿ가-힯]")

MODEL_CALLS_STATUS = Literal[
    "started",
    "succeeded",
    "failed",
    "timed_out",
    "stale",
    "superseded",
    "preflight_blocked",
]
USAGE_STATUS = Literal["reported", "partial", "unavailable"]
OUTCOME = Literal["success", "failure", "timeout", "stale", "superseded", "blocked", "incomplete"]

_MODEL_CALLS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS model_calls (
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

_MODEL_CALLS_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS model_calls_run_id
ON model_calls(run_id);
"""


def estimate_text_tokens(text: str) -> int:
    """Deterministic estimate of serialized text size in tokens."""
    if not text:
        return 0
    cjk_count = len(_CJK_CHARS.findall(text))
    non_cjk_count = len(_CJK_CHARS.sub("", text))
    return cjk_count + ceil(non_cjk_count / 4)


def _message_text(message: Any) -> str:
    content = getattr(message, "content", "")
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict):
                if block.get("type") in ("text", "input_text", "output_text"):
                    parts.append(str(block.get("text", "")))
                else:
                    parts.append(json.dumps(block, ensure_ascii=False, sort_keys=True, default=str))
            else:
                parts.append(str(block))
        return "\n".join(parts)
    return str(content or "")


def serialize_messages(messages: list[Any]) -> str:
    """Serialize the exact messages that will be sent to the provider."""
    lines: list[str] = []
    for message in messages:
        role = str(getattr(message, "type", None) or "unknown")
        text = _message_text(message)
        parts = [f"{role}: {text}"]
        tool_calls = getattr(message, "tool_calls", None)
        if tool_calls:
            parts.append(
                "tool_calls: "
                + json.dumps(list(tool_calls), ensure_ascii=False, sort_keys=True, default=str)
            )
        lines.append("\n".join(parts))
    return "\n".join(lines)


def estimate_messages_tokens(messages: list[Any]) -> int:
    return estimate_text_tokens(serialize_messages(messages))


def _tool_schema_text(tool: Any) -> str:
    if isinstance(tool, dict):
        return json.dumps(tool, ensure_ascii=False, sort_keys=True, default=str)
    schema = getattr(tool, "schema", None)
    if callable(schema):
        try:
            return json.dumps(schema(), ensure_ascii=False, sort_keys=True, default=str)
        except Exception:
            return repr(tool)
    return repr(tool)


def estimate_tools_tokens(tools: list[Any] | None) -> int:
    return sum(estimate_text_tokens(_tool_schema_text(tool)) for tool in tools or [])


@dataclass(slots=True)
class ModelCallContext:
    """Mutable per-run lineage context filled by the worker before each call."""

    run_id: str | None = None
    creation_id: str | None = None
    thread_id: str | None = None
    run_kind: str | None = None
    stage: str | None = None
    episode_number: int | None = None
    candidate: str | None = None
    batch: str | None = None

    def reset(self) -> None:
        self.run_id = None
        self.creation_id = None
        self.thread_id = None
        self.run_kind = None
        self.stage = None
        self.episode_number = None
        self.candidate = None
        self.batch = None


@dataclass(slots=True)
class ModelCallRecord:
    call_id: str
    role: str
    adapter: str
    provider: str
    model: str
    requested_at: str
    estimated_input_tokens: int
    estimated_output_tokens: int
    estimated_total_tokens: int
    verified_limit_tokens: int | None
    preflight: Literal["ok", "blocked"]
    status: MODEL_CALLS_STATUS
    usage_status: USAGE_STATUS = "unavailable"
    outcome: OUTCOME = "incomplete"
    run_id: str | None = None
    creation_id: str | None = None
    thread_id: str | None = None
    run_kind: str | None = None
    stage: str | None = None
    episode_number: int | None = None
    candidate: str | None = None
    batch: str | None = None
    finished_at: str | None = None
    duration_seconds: float | None = None
    actual_input_tokens: int | None = None
    actual_output_tokens: int | None = None
    cache_read_tokens: int | None = None
    cache_creation_tokens: int | None = None
    finish_reason: str | None = None
    error_code: str | None = None
    error_type: str | None = None
    safe_message: str | None = None
    supersedes_call_id: str | None = None


def new_call_id() -> str:
    return str(uuid4())


def _utc_now() -> datetime:
    return datetime.now(UTC)


def build_started_record(
    *,
    role: str,
    adapter: str,
    provider: str,
    model: str,
    context: ModelCallContext,
    estimated_input_tokens: int,
    estimated_output_tokens: int,
    verified_limit_tokens: int | None,
) -> ModelCallRecord:
    now = _utc_now()
    return ModelCallRecord(
        call_id=new_call_id(),
        role=role,
        adapter=adapter,
        provider=provider,
        model=model,
        requested_at=now.isoformat(),
        estimated_input_tokens=estimated_input_tokens,
        estimated_output_tokens=estimated_output_tokens,
        estimated_total_tokens=estimated_input_tokens + estimated_output_tokens,
        verified_limit_tokens=verified_limit_tokens,
        preflight="ok",
        status="started",
        run_id=context.run_id,
        creation_id=context.creation_id,
        thread_id=context.thread_id,
        run_kind=context.run_kind,
        stage=context.stage,
        episode_number=context.episode_number,
        candidate=context.candidate,
        batch=context.batch,
    )


def extract_provider_usage(response: LLMResult) -> tuple[dict[str, int | None], str | None]:
    """Return ``{input_tokens, output_tokens, cache_read_tokens, cache_creation_tokens}``.

    Provider-reported usage is collected from ``usage_metadata`` on generated messages
    and from ``llm_output`` (Anthropic ``usage`` / OpenAI ``token_usage``). Missing
    values stay ``None`` (displayed as ``unavailable``) and are never backfilled from
    estimates. The second return value is the finish reason when reported.
    """
    input_tokens: int | None = None
    output_tokens: int | None = None
    cache_read_tokens: int | None = None
    cache_creation_tokens: int | None = None
    finish_reason: str | None = None

    def absorb(key: str, value: Any) -> None:
        nonlocal input_tokens, output_tokens, cache_read_tokens, cache_creation_tokens
        if not isinstance(value, int):
            return
        if key in {"input_tokens", "prompt_tokens"} and value >= 0:
            input_tokens = value if input_tokens is None else input_tokens
        elif key in {"output_tokens", "completion_tokens"} and value >= 0:
            output_tokens = value if output_tokens is None else output_tokens
        elif (
            key
            in {
                "cache_read_input_tokens",
                "cache_read",
                "cached_tokens",
                "prompt_cache_hit_tokens",
            }
            and value >= 0
        ):
            cache_read_tokens = value if cache_read_tokens is None else cache_read_tokens
        elif (
            key
            in {
                "cache_creation_input_tokens",
                "cache_creation",
                "prompt_cache_miss_tokens",
            }
            and value >= 0
        ):
            cache_creation_tokens = (
                value if cache_creation_tokens is None else cache_creation_tokens
            )

    for generation_list in response.generations:
        for generation in generation_list:
            message = getattr(generation, "message", None)
            if message is None:
                continue
            metadata = getattr(message, "response_metadata", None)
            if isinstance(metadata, dict):
                for key in ("finish_reason", "stop_reason"):
                    value = metadata.get(key)
                    if isinstance(value, str) and value and finish_reason is None:
                        finish_reason = value
            usage_metadata = getattr(message, "usage_metadata", None)
            if isinstance(usage_metadata, dict):
                for key, value in usage_metadata.items():
                    if key == "input_token_details" and isinstance(value, dict):
                        for detail_key, detail_value in value.items():
                            absorb(detail_key, detail_value)
                        continue
                    absorb(key, value)

    llm_output = response.llm_output
    if isinstance(llm_output, dict):
        for key in ("finish_reason", "stop_reason"):
            value = llm_output.get(key)
            if isinstance(value, str) and value and finish_reason is None:
                finish_reason = value
        usage = llm_output.get("usage")
        if isinstance(usage, dict):
            for key, value in usage.items():
                absorb(key, value)
        token_usage = llm_output.get("token_usage")
        if isinstance(token_usage, dict):
            for key, value in token_usage.items():
                absorb(key, value)
            details = token_usage.get("prompt_tokens_details")
            if isinstance(details, dict):
                for key, value in details.items():
                    absorb(key, value)
        prompt_details = llm_output.get("prompt_tokens_details")
        if isinstance(prompt_details, dict):
            for key, value in prompt_details.items():
                absorb(key, value)

    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cache_read_tokens": cache_read_tokens,
        "cache_creation_tokens": cache_creation_tokens,
    }, finish_reason


def usage_status_from(tokens: dict[str, int | None]) -> USAGE_STATUS:
    input_tokens = tokens["input_tokens"]
    output_tokens = tokens["output_tokens"]
    if input_tokens is not None and output_tokens is not None:
        return "reported"
    if input_tokens is not None or output_tokens is not None:
        return "partial"
    return "unavailable"


def usage_from_tokens(tokens: dict[str, int | None]) -> ModelCallUsage:
    return ModelCallUsage(
        input_tokens=tokens["input_tokens"],
        output_tokens=tokens["output_tokens"],
        cache_read_tokens=tokens["cache_read_tokens"],
        cache_creation_tokens=tokens["cache_creation_tokens"],
        status=usage_status_from(tokens),
    )


def _row_values(record: ModelCallRecord) -> tuple[Any, ...]:
    payload = asdict(record)
    return tuple(payload[column] for column in _COLUMNS)


_COLUMNS = (
    "call_id",
    "run_id",
    "creation_id",
    "thread_id",
    "run_kind",
    "role",
    "adapter",
    "provider",
    "model",
    "stage",
    "episode_number",
    "candidate",
    "batch",
    "requested_at",
    "finished_at",
    "duration_seconds",
    "estimated_input_tokens",
    "estimated_output_tokens",
    "estimated_total_tokens",
    "verified_limit_tokens",
    "preflight",
    "status",
    "usage_status",
    "actual_input_tokens",
    "actual_output_tokens",
    "cache_read_tokens",
    "cache_creation_tokens",
    "finish_reason",
    "outcome",
    "error_code",
    "error_type",
    "safe_message",
    "supersedes_call_id",
)

_UPSERT_SQL = f"""
INSERT INTO model_calls ({", ".join(_COLUMNS)})
VALUES ({", ".join("?" for _ in _COLUMNS)})
ON CONFLICT(call_id) DO UPDATE SET
    {", ".join(f"{column} = excluded.{column}" for column in _COLUMNS[1:])}
"""

_SUPERSEDE_PENDING_SQL = """
UPDATE model_calls
SET status = 'superseded',
    outcome = 'superseded',
    finished_at = ?,
    duration_seconds = 0
WHERE run_id = ?
  AND role = ?
  AND status = 'started'
"""

_FINALIZE_TIMED_OUT_SQL = """
UPDATE model_calls
SET status = 'timed_out',
    outcome = 'timeout',
    finished_at = ?,
    duration_seconds = COALESCE(duration_seconds, 0),
    finish_reason = COALESCE(finish_reason, 'timeout'),
    error_type = COALESCE(error_type, 'timeout')
WHERE run_id = ?
  AND status = 'started'
"""


class ModelCallStore:
    """Synchronous, immediately durable SQLite writer for model-call audit records.

    The worker owns one store per process. Each record write is committed right away so
    a refresh or restart between call start, provider response, and run finalization
    still leaves the last durable state in the shared database.
    """

    def __init__(self, database_path: str | Path) -> None:
        self.database_path = Path(database_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(str(self.database_path), timeout=10)
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("PRAGMA busy_timeout = 5000")
        self._connection.execute(_MODEL_CALLS_TABLE_SQL)
        self._connection.execute(_MODEL_CALLS_INDEX_SQL)
        self._connection.commit()

    def upsert(self, record: ModelCallRecord) -> None:
        self._connection.execute(_UPSERT_SQL, _row_values(record))
        self._connection.commit()

    def mark_superseded_pending(self, *, run_id: str, role: str) -> None:
        """A new call for the same run/role supersedes any still-started prior call."""
        timestamp = _utc_now().isoformat()
        self._connection.execute(_SUPERSEDE_PENDING_SQL, (timestamp, run_id, role))
        self._connection.commit()

    def mark_timed_out(self, *, run_id: str) -> None:
        """Finalize every still-started call for a run as timed out."""
        timestamp = _utc_now().isoformat()
        self._connection.execute(_FINALIZE_TIMED_OUT_SQL, (timestamp, run_id))
        self._connection.commit()

    def close(self) -> None:
        self._connection.close()


@dataclass(slots=True)
class ModelCallState:
    """Shared mutable wiring between the relay audit handlers and the worker."""

    store: ModelCallStore | None = None
    context: ModelCallContext = field(default_factory=ModelCallContext)
