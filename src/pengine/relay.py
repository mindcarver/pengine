import asyncio
import concurrent.futures
import json
import logging
import re
import ssl
import threading
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from math import ceil
from typing import Any, Literal
from uuid import UUID

import httpx
from langchain_anthropic import ChatAnthropic
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.language_models import BaseChatModel
from langchain_core.outputs import LLMResult
from langchain_deepseek import ChatDeepSeek
from langchain_openai import ChatOpenAI

try:
    # Optional agent observability. The import is guarded so pengine still runs
    # when tracing is disabled or the langfuse package is absent. The handler is
    # only constructed when settings.langfuse_configured is true.
    from langfuse.langchain import CallbackHandler as _LangfuseCallbackHandler
except ImportError:  # pragma: no cover - exercised only without langfuse installed
    _LangfuseCallbackHandler = None  # type: ignore[assignment,misc]

from pengine.config import Settings
from pengine.model_calls import (
    ModelCallContext,
    ModelCallRecord,
    ModelCallState,
    StageCallBudgetExceeded,
    build_started_record,
    estimate_messages_tokens,
    estimate_tools_tokens,
    extract_provider_usage,
    usage_status_from,
)
from pengine.observability import record_model_call_event

_AUTO_TOOL_CHOICE_MODELS = frozenset({"deepseek-v4-flash", "deepseek-v4-pro"})
# Uvicorn owns the runtime log handlers. This child logger keeps the safe model-call
# audit at INFO while ensuring it reaches the same server evidence stream.
_MODEL_CALL_LOGGER = logging.getLogger("uvicorn.error.pengine.model_calls")
# Durable structured record lines that carry estimate/actual/duration/finish/outcome.
_MODEL_CALL_RECORD_LOGGER = logging.getLogger("uvicorn.error.pengine.model_call_records")
ModelRole = Literal["generation", "review"]

# The audit handler runs on the event-loop thread (the model is invoked with
# ``ainvoke``), and a synchronous SQLite write there can deadlock with the
# LangGraph AsyncSqliteSaver: the sync write blocks the loop while the saver holds
# the database write lock and needs the loop to run its queued commit. The
# persistence is therefore dispatched to this single writer thread whenever a
# running loop is present; the worker drains it before finalizing a run
# (Delivery #58 INT-A1/A2/A9). Sync tests without a running loop write inline and
# stay deterministic.
_AUDIT_EXECUTOR = concurrent.futures.ThreadPoolExecutor(
    max_workers=1,
    thread_name_prefix="pengine-audit",
)
_PENDING_AUDIT_WRITES: set[concurrent.futures.Future[Any]] = set()


def _running_loop() -> bool:
    try:
        asyncio.get_running_loop()
        return True
    except RuntimeError:
        return False


def submit_store_write(fn: Any, *args: Any, **kwargs: Any) -> None:
    """Run one synchronous SQLite audit write, off the loop when a loop is running."""
    if _running_loop():
        future = _AUDIT_EXECUTOR.submit(fn, *args, **kwargs)
        _PENDING_AUDIT_WRITES.add(future)
        future.add_done_callback(_PENDING_AUDIT_WRITES.discard)
    else:
        fn(*args, **kwargs)


def drain_audit_writes() -> None:
    """Wait for every dispatched audit write to land (used at run finalization)."""
    futures = list(_PENDING_AUDIT_WRITES)
    for future in futures:
        try:
            # Cover the store's 30s busy timeout plus scheduling margin.
            future.result(timeout=60)
        except Exception as exc:  # pragma: no cover - infrastructure failure path
            _MODEL_CALL_LOGGER.warning(
                "model_call audit write failed error_type=%s", type(exc).__name__
            )


class _SerialChatAnthropic(ChatAnthropic):
    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> Any:
        kwargs["parallel_tool_calls"] = False
        kwargs["tool_choice"] = "auto" if self.model in _AUTO_TOOL_CHOICE_MODELS else "any"
        return super().bind_tools(tools, **kwargs)


class _SerialChatDeepSeek(ChatDeepSeek):
    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> Any:
        # LangChain's ToolStrategy requests "any" for a mixed set of working and
        # result tools. DeepSeek treats that as "required", which can trap an agent
        # in repeated working-tool calls. Keep mixed sets on auto, but preserve the
        # required choice when middleware narrows the call to the one result tool.
        tool_choice = kwargs.get("tool_choice")
        if len(tools) > 1 and isinstance(tool_choice, str) and tool_choice in {"any", "required"}:
            kwargs["tool_choice"] = "auto"
        kwargs["parallel_tool_calls"] = False
        return super().bind_tools(tools, **kwargs)


class _SerialChatOpenAI(ChatOpenAI):
    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> Any:
        kwargs["parallel_tool_calls"] = False
        return super().bind_tools(tools, **kwargs)


def _build_langfuse_handler(settings: Settings) -> BaseCallbackHandler | None:
    """Build a Langfuse trace handler when tracing is configured, else None.

    The handler rides on the ChatModel instance, so a single construction per role
    covers the supervisor and every sub-agent that reuses that model — deepagents
    propagates the parent's callbacks to sub-agents automatically. Run-level
    association (session_id, run_id) is stamped from the worker via
    :func:`langfuse.propagate_attributes`, not here.
    """
    if not settings.langfuse_configured or _LangfuseCallbackHandler is None:
        return None
    # Credentials reach the SDK via LANGFUSE_* env vars set by the worker at
    # startup, so the handler needs no arguments here.
    return _LangfuseCallbackHandler()


class _ModelCallAuditHandler(BaseCallbackHandler):
    """Preflight + durable audit at the common outbound model-call boundary.

    ``on_chat_model_start`` estimates the exact serialized request plus reserved output
    against the verified route capacity and raises :class:`PreflightBlockedError` when
    the request cannot fit, so no provider request is dispatched. Every attempted call
    is recorded (estimate, actual-or-unavailable usage, duration, finish reason,
    outcome, lineage) to the shared :class:`ModelCallState` store and structured logs.
    """

    raise_error = True
    run_inline = True

    def __init__(
        self,
        *,
        role: ModelRole,
        model_id: str,
        adapter: str = "",
        provider: str = "",
        model_call_state: ModelCallState | None = None,
        context_limit_tokens: int | None = None,
        reserved_output_tokens: int = 0,
        stage_call_limit: int | None = None,
        stage_review_call_limit: int | None = None,
        script_stage_model_call_total_limit: int | None = None,
        script_stage_review_call_total_limit: int | None = None,
    ) -> None:
        self.role = role
        self.model_id = model_id
        self.adapter = adapter
        self.provider = provider
        self.state = model_call_state
        self.context_limit_tokens = context_limit_tokens
        self.reserved_output_tokens = reserved_output_tokens
        self.stage_call_limit = stage_call_limit
        self.stage_review_call_limit = stage_review_call_limit
        self.script_stage_model_call_total_limit = script_stage_model_call_total_limit
        self.script_stage_review_call_total_limit = script_stage_review_call_total_limit
        self._pending: dict[UUID, ModelCallRecord] = {}
        self._pending_lineage: dict[UUID, tuple[int | None, int | None]] = {}

    def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[Any]],
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        message_batch = messages[0] if messages else []
        context = self.state.context if self.state is not None else ModelCallContext()
        estimated_input = estimate_messages_tokens(message_batch)
        estimated_input += estimate_tools_tokens(_extract_tools(serialized, kwargs))
        estimated_total = estimated_input + self.reserved_output_tokens
        record = build_started_record(
            role=self.role,
            adapter=self.adapter,
            provider=self.provider,
            model=self.model_id,
            context=context,
            estimated_input_tokens=estimated_input,
            estimated_output_tokens=self.reserved_output_tokens,
            verified_limit_tokens=self.context_limit_tokens,
        )
        role_limit = (
            self.stage_review_call_limit if self.role == "review" else self.stage_call_limit
        )
        script_stage = context.stage == "generating_episode_scripts"
        role_total_limit = (
            (
                self.script_stage_review_call_total_limit
                if self.role == "review"
                else self.script_stage_model_call_total_limit
            )
            if script_stage
            else None
        )
        stage_sequence: int | None = None
        try:
            if self.state is not None:
                stage_sequence = self.state.reserve_stage_call(
                    role=self.role,
                    limit=role_limit,
                    total_limit=role_total_limit,
                )
        except StageCallBudgetExceeded as exc:
            record.preflight = "blocked"
            record.status = "preflight_blocked"
            record.outcome = "blocked"
            record.error_code = "agent_execution_limit"
            record.error_type = type(exc).__name__
            record.safe_message = str(exc)
            record.finished_at = _utc_now().isoformat()
            record.duration_seconds = 0.0
            self._persist(record)
            record_model_call_event(
                phase="blocked",
                role=self.role,
                stage=context.stage,
                episode_number=context.episode_number,
                sequence=exc.attempted,
                outcome="blocked",
                call_id=record.call_id,
                model=self.model_id,
                repair_round=context.repair_round,
                error_code="agent_execution_limit",
            )
            _MODEL_CALL_LOGGER.warning(
                "model_call event=error role=%s requested_model_id=%s call_id=%s "
                "error_type=%s stage=%s limit=%s attempted=%s",
                self.role,
                self.model_id,
                run_id,
                type(exc).__name__,
                exc.stage,
                exc.limit,
                exc.attempted,
            )
            raise
        blocked = self.context_limit_tokens is None or estimated_total > self.context_limit_tokens
        if blocked:
            record.preflight = "blocked"
            record.status = "preflight_blocked"
            record.outcome = "blocked"
            record.error_code = "preflight_blocked"
            record.finished_at = _utc_now().isoformat()
            record.duration_seconds = 0.0
            self._persist(record)
            record_model_call_event(
                phase="blocked",
                role=self.role,
                stage=context.stage,
                episode_number=context.episode_number,
                sequence=stage_sequence,
                outcome="blocked",
                call_id=record.call_id,
                model=self.model_id,
                repair_round=context.repair_round,
                error_code="preflight_blocked",
            )
            _MODEL_CALL_LOGGER.info(
                "model_call event=error role=%s requested_model_id=%s call_id=%s "
                "error_type=preflight_blocked http_status=none "
                "estimated_total_tokens=%s verified_limit_tokens=%s",
                self.role,
                self.model_id,
                run_id,
                estimated_total,
                self.context_limit_tokens,
            )
            raise PreflightBlockedError(
                role=self.role,
                model_id=self.model_id,
                stage=context.stage,
                episode_number=context.episode_number,
                required_tokens=estimated_total,
                verified_limit_tokens=self.context_limit_tokens,
            )
        self._pending[run_id] = record
        self._pending_lineage[run_id] = (stage_sequence, context.repair_round)
        # Supersede any prior still-started call for this run/role BEFORE persisting
        # the current record, so the current in-flight call is never self-superseded.
        if self.state is not None and self.state.store is not None and context.run_id is not None:
            submit_store_write(
                self.state.store.mark_superseded_pending,
                run_id=context.run_id,
                role=self.role,
            )
        self._persist(record)
        record_model_call_event(
            phase="start",
            role=record.role,
            stage=record.stage,
            episode_number=record.episode_number,
            sequence=stage_sequence,
            outcome="started",
            call_id=record.call_id,
            model=record.model,
            repair_round=context.repair_round,
        )
        _MODEL_CALL_LOGGER.info(
            "model_call event=start role=%s requested_model_id=%s call_id=%s "
            "stage=%s episode=%s estimated_input_tokens=%s estimated_output_tokens=%s",
            self.role,
            self.model_id,
            run_id,
            context.stage,
            context.episode_number,
            estimated_input,
            self.reserved_output_tokens,
        )

    def on_llm_end(self, response: LLMResult, *, run_id: UUID, **kwargs: Any) -> None:
        del kwargs
        record = self._pending.pop(run_id, None)
        sequence, repair_round = self._pending_lineage.pop(run_id, (None, None))
        response_model_ids = _response_model_ids(response)
        tokens, finish_reason = extract_provider_usage(response)
        if {model_id.casefold() for model_id in response_model_ids} != {self.model_id.casefold()}:
            if record is not None:
                self._finalize(
                    record,
                    status="failed",
                    outcome="failure",
                    tokens=tokens,
                    finish_reason=finish_reason,
                    error_code="relay_incompatible",
                    error_type="RelayError",
                    safe_message="The relay returned an unexpected model for the configured role.",
                    sequence=sequence,
                    repair_round=repair_round,
                )
            _MODEL_CALL_LOGGER.error(
                "model_call event=identity_mismatch role=%s requested_model_id=%s "
                "response_model_ids=%s call_id=%s",
                self.role,
                self.model_id,
                sorted(response_model_ids),
                run_id,
            )
            raise RelayError(
                code="relay_incompatible",
                safe_message="The relay returned an unexpected model for the configured role.",
            )
        if record is not None:
            self._finalize(
                record,
                status="succeeded",
                outcome="success",
                tokens=tokens,
                finish_reason=finish_reason,
                sequence=sequence,
                repair_round=repair_round,
            )
        _MODEL_CALL_LOGGER.info(
            "model_call event=end role=%s requested_model_id=%s response_model_id=%s "
            "call_id=%s usage_status=%s finish_reason=%s",
            self.role,
            self.model_id,
            self.model_id,
            run_id,
            usage_status_from(tokens) if record is not None else "unavailable",
            finish_reason,
        )

    def on_llm_error(self, error: BaseException, *, run_id: UUID, **kwargs: Any) -> None:
        del kwargs
        record = self._pending.pop(run_id, None)
        sequence, repair_round = self._pending_lineage.pop(run_id, (None, None))
        failure = _extract_provider_failure(error)
        http_status = failure.http_status if failure is not None else None
        if record is not None:
            timed_out = isinstance(error, TimeoutError) or "timeout" in type(error).__name__.lower()
            self._finalize(
                record,
                status="timed_out" if timed_out else "failed",
                outcome="timeout" if timed_out else "failure",
                tokens={
                    "input_tokens": None,
                    "output_tokens": None,
                    "cache_read_tokens": None,
                    "cache_creation_tokens": None,
                },
                finish_reason="timeout" if timed_out else None,
                error_code="relay_timeout" if timed_out else "relay_error",
                error_type=type(error).__name__,
                http_status=http_status,
                provider_error_code=(failure.provider_code if failure is not None else None),
                redacted_body=(failure.redacted_body if failure is not None else None),
                sequence=sequence,
                repair_round=repair_round,
            )
        redacted = failure.redacted_body if failure is not None else None
        _MODEL_CALL_LOGGER.warning(
            "model_call event=error role=%s requested_model_id=%s call_id=%s "
            "error_type=%s http_status=%s provider_error_code=%s redacted_response=%s",
            self.role,
            self.model_id,
            run_id,
            type(error).__name__,
            http_status if http_status is not None else "none",
            (failure.provider_code if failure is not None else "none"),
            _truncate(redacted, 300) if redacted else "none",
        )

    def _persist(self, record: ModelCallRecord) -> None:
        if self.state is not None and self.state.store is not None:
            submit_store_write(self.state.store.upsert, record)
        self._log_record(record)

    def _finalize(
        self,
        record: ModelCallRecord,
        *,
        status: str,
        outcome: str,
        tokens: dict[str, int | None],
        finish_reason: str | None,
        error_code: str | None = None,
        error_type: str | None = None,
        safe_message: str | None = None,
        http_status: int | None = None,
        provider_error_code: str | None = None,
        redacted_body: str | None = None,
        sequence: int | None = None,
        repair_round: int | None = None,
    ) -> None:
        now = _utc_now()
        record.status = status  # type: ignore[assignment]
        record.outcome = outcome  # type: ignore[assignment]
        record.finished_at = now.isoformat()
        if record.requested_at:
            try:
                started = datetime.fromisoformat(record.requested_at)
            except ValueError:
                started = now
            record.duration_seconds = max(0.0, (now - started).total_seconds())
        else:
            record.duration_seconds = 0.0
        record.actual_input_tokens = tokens["input_tokens"]
        record.actual_output_tokens = tokens["output_tokens"]
        record.cache_read_tokens = tokens["cache_read_tokens"]
        record.cache_creation_tokens = tokens["cache_creation_tokens"]
        record.usage_status = usage_status_from(tokens)
        record.finish_reason = finish_reason
        record.error_code = error_code
        record.error_type = error_type
        record.safe_message = safe_message
        record.http_status = http_status
        record.provider_error_code = provider_error_code
        record.redacted_response = redacted_body
        self._persist(record)
        record_model_call_event(
            phase="end",
            role=record.role,
            stage=record.stage,
            episode_number=record.episode_number,
            sequence=sequence,
            outcome=record.outcome,
            call_id=record.call_id,
            model=record.model,
            repair_round=repair_round,
            error_code=record.error_code,
            finish_reason=record.finish_reason,
            duration_seconds=record.duration_seconds,
        )

    def _log_record(self, record: ModelCallRecord) -> None:
        _MODEL_CALL_RECORD_LOGGER.info(
            "model_call_record call_id=%s role=%s adapter=%s provider=%s model=%s "
            "stage=%s episode=%s status=%s preflight=%s estimated_input_tokens=%s "
            "estimated_output_tokens=%s estimated_total_tokens=%s verified_limit_tokens=%s "
            "usage_status=%s actual_input_tokens=%s actual_output_tokens=%s "
            "cache_read_tokens=%s cache_creation_tokens=%s duration_ms=%s "
            "finish_reason=%s outcome=%s error_code=%s error_type=%s supersedes_call_id=%s",
            record.call_id,
            record.role,
            record.adapter,
            record.provider,
            record.model,
            record.stage,
            record.episode_number,
            record.status,
            record.preflight,
            record.estimated_input_tokens,
            record.estimated_output_tokens,
            record.estimated_total_tokens,
            record.verified_limit_tokens,
            record.usage_status,
            record.actual_input_tokens,
            record.actual_output_tokens,
            record.cache_read_tokens,
            record.cache_creation_tokens,
            None if record.duration_seconds is None else round(record.duration_seconds * 1000),
            record.finish_reason,
            record.outcome,
            record.error_code,
            record.error_type,
            record.supersedes_call_id,
        )


def _extract_tools(serialized: dict[str, Any], kwargs: dict[str, Any]) -> list[Any]:
    # The bound tool/schema definitions ride in invocation_params (and the serialized
    # model config) rather than in the message batch, so preflight pulls them from here
    # to count tool/schema overhead against the verified context limit.
    tools: list[Any] = []
    for source in (kwargs, serialized):
        if not isinstance(source, dict):
            continue
        invocation = source.get("invocation_params")
        if isinstance(invocation, dict):
            candidate = invocation.get("tools")
            if isinstance(candidate, list):
                tools.extend(candidate)
        candidate = source.get("tools")
        if isinstance(candidate, list):
            tools.extend(candidate)
    return tools


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _response_model_ids(response: LLMResult) -> set[str]:
    model_ids: set[str] = set()
    for generation_list in response.generations:
        for generation in generation_list:
            message = getattr(generation, "message", None)
            metadata = getattr(message, "response_metadata", None)
            if not isinstance(metadata, dict):
                continue
            for key in ("model", "model_name"):
                value = metadata.get(key)
                if isinstance(value, str) and value:
                    model_ids.add(value)
    if isinstance(response.llm_output, dict):
        for key in ("model", "model_name"):
            value = response.llm_output.get(key)
            if isinstance(value, str) and value:
                model_ids.add(value)
    return model_ids


@dataclass(frozen=True, slots=True)
class _ProviderFailure:
    """Precise provider failure signal extracted from an exception and its cause chain.

    ``detail`` is the provider's own error message (pre-redaction, used only for
    classification); ``raw_body`` is the full response body before redaction. Both stay
    internal: anything surfaced through :class:`RelayError` is redacted and truncated.
    """

    http_status: int | None
    provider_code: str | None
    detail: str
    raw_body: str | None

    @property
    def redacted_body(self) -> str | None:
        if self.raw_body is None:
            return None
        return redact_provider_response(self.raw_body)


_SECRET_KEY_NAME = re.compile(
    r"^(?:api[_-]?key|apikey|authorization|x-api-key|access[_-]?token|"
    r"refresh[_-]?token|bearer|secret|password|token|key)$",
    re.IGNORECASE,
)
_SK_TOKEN = re.compile(r"\bsk-[A-Za-z0-9_\-]{8,}")
_BEARER_TOKEN = re.compile(r"\bBearer [A-Za-z0-9._~+/=\-]+")
_SECRET_KEY_PAIR = re.compile(
    r"([\"']?(?:api[_-]?key|apikey|authorization|access[_-]?token|"
    r"refresh[_-]?token|secret|password)[\"']?\s*[:=]\s*)([\"'][^\"']*[\"']|\S+)",
    re.IGNORECASE,
)
# Known operator secrets (the configured relay API key) are masked exactly anywhere a
# provider response echoes them back. Registration is thread-safe; extra masks are
# harmless, so tests may register their own probe secrets.
_REDACTION_SECRETS: set[str] = set()
_REDACTION_SECRETS_LOCK = threading.Lock()


def register_redaction_secret(secret: Any) -> None:
    """Register an exact credential value that must never persist in provider evidence.

    Accepts either a plain string or a pydantic ``SecretStr`` (the configured relay key
    arrives as ``SecretStr``); only strings of at least 6 characters are registered.
    """
    if not isinstance(secret, str):
        reveal = getattr(secret, "get_secret_value", None)
        if callable(reveal):
            secret = reveal()
    if not isinstance(secret, str) or not secret or len(secret) < 6:
        return
    with _REDACTION_SECRETS_LOCK:
        _REDACTION_SECRETS.add(secret)


def _mask_known_secrets(text: str) -> str:
    with _REDACTION_SECRETS_LOCK:
        secrets = tuple(_REDACTION_SECRETS)
    for secret in secrets:
        if secret in text:
            text = text.replace(secret, "***")
    return text


def redact_provider_response(text: str | None) -> str | None:
    """Return a credential-safe copy of a provider response body.

    JSON bodies are redacted structurally (secret-named keys become ``***``, secret
    token patterns inside text are masked); non-JSON text is scrubbed with the same
    token patterns. Known operator secrets (the configured relay API key) are masked
    exactly. This keeps provider failure evidence durable and queryable while never
    persisting keys or bearer tokens (Issue #52).
    """
    if not text:
        return text
    try:
        payload = json.loads(text)
    except (ValueError, TypeError):
        return _redact_text(text)
    try:
        return json.dumps(_redact_value(payload), ensure_ascii=False, sort_keys=True)
    except (TypeError, ValueError):
        return _redact_text(text)


def _redact_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: "***" if _SECRET_KEY_NAME.fullmatch(key) else _redact_value(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_redact_value(item) for item in value]
    if isinstance(value, str):
        return _redact_text(value)
    return value


def _redact_text(text: str) -> str:
    text = _mask_known_secrets(text)
    text = _SK_TOKEN.sub("sk-***", text)
    text = _BEARER_TOKEN.sub("Bearer ***", text)
    text = _SECRET_KEY_PAIR.sub(r"\1***", text)
    return text


def _raw_body_from(candidate: BaseException) -> str | None:
    body = getattr(candidate, "body", None)
    if isinstance(body, (dict, list)):
        try:
            return json.dumps(body, ensure_ascii=False, sort_keys=True)
        except (TypeError, ValueError):
            return repr(body)
    if isinstance(body, str) and body:
        return body
    response = getattr(candidate, "response", None)
    text = getattr(response, "text", None)
    if isinstance(text, str) and text:
        return text
    return None


def _provider_error_code_from(candidate: BaseException, raw_body: str | None) -> str | None:
    if raw_body is not None:
        try:
            parsed = json.loads(raw_body)
        except (ValueError, TypeError):
            parsed = None
        if isinstance(parsed, dict):
            error = parsed.get("error")
            if isinstance(error, dict):
                for key in ("code", "type"):
                    value = error.get(key)
                    if isinstance(value, str) and value:
                        return value
    for attr in ("code", "type"):
        value = getattr(candidate, attr, None)
        if isinstance(value, str) and value:
            return value
    return None


def _detail_from(candidate: BaseException, raw_body: str | None) -> str:
    if raw_body is not None:
        try:
            parsed = json.loads(raw_body)
        except (ValueError, TypeError):
            parsed = None
        if isinstance(parsed, dict):
            error = parsed.get("error")
            if isinstance(error, dict):
                message = error.get("message")
                if isinstance(message, str) and message:
                    return message.strip()
            message = parsed.get("message")
            if isinstance(message, str) and message:
                return message.strip()
    message = getattr(candidate, "message", None)
    if isinstance(message, str) and message:
        return message.strip()
    text = str(candidate)
    return text.strip() if text and text != "()" else ""


def _extract_provider_failure(error: BaseException) -> _ProviderFailure | None:
    for candidate in (error, *_cause_chain(error)):
        status = getattr(candidate, "status_code", None)
        if not isinstance(status, int):
            response = getattr(candidate, "response", None)
            status = getattr(response, "status_code", None)
            if not isinstance(status, int):
                continue
        raw_body = _raw_body_from(candidate)
        return _ProviderFailure(
            http_status=status,
            provider_code=_provider_error_code_from(candidate, raw_body),
            detail=_detail_from(candidate, raw_body),
            raw_body=raw_body,
        )
    return None


_PROVIDER_EVIDENCE_MAX_CHARS = 400


def _provider_evidence(failure: _ProviderFailure | None) -> str:
    if failure is None:
        return "no provider detail available"
    redacted = failure.redacted_body
    if redacted:
        return _truncate(redacted, _PROVIDER_EVIDENCE_MAX_CHARS)
    detail = failure.detail
    if detail:
        return _truncate(redact_provider_response(detail) or "", _PROVIDER_EVIDENCE_MAX_CHARS)
    return "no provider detail available"


def _truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."


# Tool/function protocol rejection, not arbitrary request-content rejection. The
# pattern requires an explicit "not supported"-style signal near tool language, so
# ordinary 400s (context length, content policy, malformed input) stay classified as
# request rejection instead of a protocol mismatch (Issue #52 graph revision 4).
_TOOL_PROTOCOL_REJECTION = re.compile(
    r"\b(?:tool|tools|tool_use|tool_calls|function|functions|function_call|"
    r"structured output)\b.{0,80}\b(?:not supported|unsupported|does not support|"
    r"not allowed|not available|unavailable|disabled)\b"
    r"|\b(?:not supported|unsupported|does not support|not allowed)\b.{0,60}\b(?:tool|tools|"
    r"tool_use|tool_calls|function|functions|function_call)\b",
    re.IGNORECASE | re.DOTALL,
)


@dataclass(frozen=True, slots=True)
class RelayAdapter:
    model: BaseChatModel
    role: ModelRole
    model_id: str
    provider_profile_key: str
    model_call_state: ModelCallState | None = None


@dataclass(frozen=True, slots=True)
class RelayRoutes:
    generation: RelayAdapter
    review: RelayAdapter
    model_call_state: ModelCallState | None = None


@dataclass(slots=True)
class RelayError(Exception):
    code: Literal["relay_unavailable", "relay_incompatible", "relay_rejected", "preflight_blocked"]
    safe_message: str
    http_status: int | None = None
    provider_error_code: str | None = None
    redacted_body: str | None = None

    def __str__(self) -> str:
        return self.safe_message


class PreflightBlockedError(RelayError):
    """Fail-closed result when the serialized request cannot fit the verified route."""

    def __init__(
        self,
        *,
        role: ModelRole,
        model_id: str,
        stage: str | None,
        episode_number: int | None,
        required_tokens: int,
        verified_limit_tokens: int | None,
    ) -> None:
        self.stage = stage
        self.episode_number = episode_number
        self.required_tokens = required_tokens
        self.verified_limit_tokens = verified_limit_tokens
        self.model_id = model_id
        limit_text = (
            str(verified_limit_tokens)
            if verified_limit_tokens is not None
            else "unverified (no trustworthy verified limit)"
        )
        super().__init__(
            code="preflight_blocked",
            safe_message=(
                f"The {role} model request needs about {required_tokens} tokens "
                f"(input plus reserved output), but the verified context limit for "
                f"{model_id} is {limit_text}. No request was sent; the current run "
                "paused without changing approved work."
            ),
        )


MIN_RELAY_RETRY_DELAY_SECONDS = 10
# HTTP 408 is a provider-side request timeout (`timeout_error`); it is transient
# congestion, not a terminal relay failure, so the run recovers instead of
# failing on a single timed-out call (Issue #52 graph revision 10).
_RETRYABLE_RELAY_STATUSES = frozenset({408, 429, 502, 503, 504})


@dataclass(frozen=True, slots=True)
class RetryableRelayInterruption:
    retry_delay_seconds: int


def build_relay_routes(
    settings: Settings,
    *,
    model_call_state: ModelCallState | None = None,
) -> RelayRoutes:
    if not settings.relay_configured:
        raise RelayError(
            code="relay_unavailable",
            safe_message="Both generation and review model routes must be configured.",
        )
    if model_call_state is None:
        model_call_state = ModelCallState()
    return RelayRoutes(
        generation=build_relay_adapter(
            settings,
            role="generation",
            model_call_state=model_call_state,
        ),
        review=build_relay_adapter(
            settings,
            role="review",
            model_call_state=model_call_state,
        ),
        model_call_state=model_call_state,
    )


def build_relay_adapter(
    settings: Settings,
    *,
    role: ModelRole,
    model_call_state: ModelCallState | None = None,
) -> RelayAdapter:
    if role == "generation":
        model_id = settings.generation_model_id
        max_output_tokens = settings.generation_max_output_tokens
        context_limit_tokens = settings.generation_context_limit_tokens
        provider_profile_key = "anthropic"
    else:
        model_id = settings.review_model_id
        max_output_tokens = settings.review_max_output_tokens
        context_limit_tokens = settings.review_context_limit_tokens
        provider_profile_key = (
            "anthropic"
            if model_id == "claude-opus-5"
            else "openai"
            if model_id in {"gpt-5.5", "gpt-5.6-terra"}
            else "deepseek"
        )
    if not settings.relay_base_url or not settings.relay_api_key or not model_id:
        raise RelayError(
            code="relay_unavailable",
            safe_message=f"The {role} model route is not configured.",
        )
    if model_call_state is None:
        model_call_state = ModelCallState()
    reserved_output_tokens = max_output_tokens or 0
    # The operator's relay key is an exact credential that must never survive in
    # durable provider-error evidence, even if the relay echoes it in an error body.
    register_redaction_secret(settings.relay_api_key)

    common = {
        "model": model_id,
        "base_url": settings.relay_base_url,
        "api_key": settings.relay_api_key,
        "max_retries": 0,
        "timeout": settings.model_timeout_seconds,
        "temperature": 0,
        "callbacks": [
            _ModelCallAuditHandler(
                role=role,
                model_id=model_id,
                adapter=provider_profile_key,
                provider=provider_profile_key,
                model_call_state=model_call_state,
                context_limit_tokens=context_limit_tokens,
                reserved_output_tokens=reserved_output_tokens,
                stage_call_limit=settings.stage_model_call_limit,
                stage_review_call_limit=settings.stage_review_call_limit,
                script_stage_model_call_total_limit=settings.script_stage_model_call_total_limit,
                script_stage_review_call_total_limit=settings.script_stage_review_call_total_limit,
            ),
            *([handler] if (handler := _build_langfuse_handler(settings)) is not None else []),
        ],
    }
    if role == "review":
        if model_id in {"gpt-5.5", "gpt-5.6-terra"}:
            return RelayAdapter(
                model=_SerialChatOpenAI(
                    **common,
                    max_tokens=max_output_tokens,
                ),
                role=role,
                model_id=model_id,
                provider_profile_key=provider_profile_key,
                model_call_state=model_call_state,
            )
        if model_id == "claude-opus-5":
            return RelayAdapter(
                model=_SerialChatAnthropic(
                    **common,
                    max_tokens=max_output_tokens,
                ),
                role=role,
                model_id=model_id,
                provider_profile_key=provider_profile_key,
                model_call_state=model_call_state,
            )
        return RelayAdapter(
            model=_SerialChatDeepSeek(
                **common,
                max_tokens=max_output_tokens,
                extra_body={"thinking": {"type": "disabled"}},
            ),
            role=role,
            model_id=model_id,
            provider_profile_key=provider_profile_key,
            model_call_state=model_call_state,
        )
    return RelayAdapter(
        model=_SerialChatAnthropic(
            **common,
            max_tokens=max_output_tokens,
            # The relay 408s non-streaming requests at ~300s; long generation
            # calls (episode outlines/scripts) legitimately exceed that, so
            # stream and aggregate. model_timeout_seconds then bounds the gap
            # between chunks instead of the whole response.
            streaming=True,
        ),
        role=role,
        model_id=model_id,
        provider_profile_key=provider_profile_key,
        model_call_state=model_call_state,
    )


def build_chat_model(settings: Settings, *, role: ModelRole) -> BaseChatModel:
    return build_relay_adapter(settings, role=role).model


def is_relay_exception(exc: BaseException) -> bool:
    return any(
        base.__module__.startswith(("anthropic", "openai", "httpx", "httpcore"))
        for base in type(exc).__mro__
    )


def is_relay_connection_error(exc: BaseException) -> bool:
    if _is_retryable_transport(exc):
        return True
    return any(
        base.__module__.startswith(("anthropic", "openai"))
        and base.__name__ == "APIConnectionError"
        for base in type(exc).__mro__
    )


def classify_relay_exception(exc: Exception) -> RelayError:
    """Classify a relay failure from precise provider evidence, never type-name heuristics.

    An explicit HTTP status carries the real semantics: 400 content rejection maps to
    ``relay_rejected``, a 400 tool-protocol rejection stays ``relay_incompatible``,
    and auth/other terminal statuses map to ``relay_unavailable``. Every surfaced
    message reflects the provider's own (redacted, truncated) response so the external
    block root cause stays distinguishable (Issue #52 graph revision 4). This function
    only labels the failure truthfully; whether a status is retryable is decided
    separately by ``retryable_relay_interruption`` (e.g. HTTP 408 provider timeout is
    classified as ``relay_unavailable`` but recoverable — Issue #52 graph revision 10).
    """
    failure = _extract_provider_failure(exc)
    status = failure.http_status if failure is not None else None
    if status is not None:
        evidence = _provider_evidence(failure)
        if (
            status == 400
            and failure is not None
            and _TOOL_PROTOCOL_REJECTION.search(failure.detail)
        ):
            return RelayError(
                code="relay_incompatible",
                safe_message=(
                    "The model relay rejected the tool request (HTTP 400): the provider "
                    "does not support the required tool protocol. "
                    f"Provider response: {evidence}"
                ),
                http_status=status,
                provider_error_code=failure.provider_code,
                redacted_body=failure.redacted_body,
            )
        if status == 400:
            return RelayError(
                code="relay_rejected",
                safe_message=(
                    "The model relay rejected the request (HTTP 400): the provider "
                    "response indicates a request-content problem, not a protocol "
                    f"mismatch. Provider response: {evidence}"
                ),
                http_status=status,
                provider_error_code=failure.provider_code,
                redacted_body=failure.redacted_body,
            )
        return RelayError(
            code="relay_unavailable",
            safe_message=(
                f"The model relay request failed (HTTP {status}). Provider response: {evidence}"
            ),
            http_status=status,
            provider_error_code=failure.provider_code,
            redacted_body=failure.redacted_body,
        )
    type_name = type(exc).__name__.lower()
    if any(token in type_name for token in ("tool", "structured")):
        return RelayError(
            code="relay_incompatible",
            safe_message="The model relay does not support the required tool protocol.",
        )
    return RelayError(
        code="relay_unavailable",
        safe_message="The model relay request failed.",
    )


def retryable_relay_interruption(exc: Exception) -> RetryableRelayInterruption | None:
    if _has_tls_configuration_error(exc):
        return None
    if _is_retryable_transport(exc):
        return RetryableRelayInterruption(_retry_delay_seconds(exc))
    if is_relay_connection_error(exc) and any(
        _is_retryable_transport(candidate) for candidate in _cause_chain(exc)
    ):
        return RetryableRelayInterruption(_retry_delay_seconds(exc))
    if _is_retryable_status_error(exc):
        return RetryableRelayInterruption(_retry_delay_seconds(exc))
    return None


def _is_retryable_transport(exc: BaseException) -> bool:
    if isinstance(
        exc,
        (
            ConnectionResetError,
            httpx.ConnectError,
            httpx.ReadError,
            httpx.ConnectTimeout,
            httpx.ReadTimeout,
            httpx.RemoteProtocolError,
        ),
    ):
        return True
    return type(exc).__module__.startswith("httpcore") and type(exc).__name__ in {
        "ConnectError",
        "ReadError",
        "ConnectTimeout",
        "ReadTimeout",
        "RemoteProtocolError",
    }


def _has_tls_configuration_error(exc: BaseException) -> bool:
    return any(
        isinstance(candidate, ssl.SSLCertVerificationError)
        for candidate in (exc, *_cause_chain(exc))
    )


def _is_retryable_status_error(exc: BaseException) -> bool:
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in _RETRYABLE_RELAY_STATUSES
    return (
        any(
            base.__module__.startswith(("anthropic", "openai"))
            and base.__name__ == "APIStatusError"
            for base in type(exc).__mro__
        )
        and getattr(exc, "status_code", None) in _RETRYABLE_RELAY_STATUSES
    )


def _cause_chain(exc: BaseException):
    candidate = exc.__cause__
    seen: set[int] = set()
    while candidate is not None and id(candidate) not in seen:
        seen.add(id(candidate))
        yield candidate
        candidate = candidate.__cause__


def _retry_delay_seconds(exc: Exception) -> int:
    for candidate in (exc, *_cause_chain(exc)):
        response = getattr(candidate, "response", None)
        headers = getattr(response, "headers", None)
        retry_after = headers.get("retry-after") if headers is not None else None
        if not isinstance(retry_after, str):
            continue
        try:
            return max(MIN_RELAY_RETRY_DELAY_SECONDS, ceil(float(retry_after)))
        except ValueError:
            try:
                retry_at = parsedate_to_datetime(retry_after)
            except (TypeError, ValueError, IndexError):
                continue
            if retry_at.tzinfo is None:
                retry_at = retry_at.replace(tzinfo=UTC)
            remaining = (retry_at.astimezone(UTC) - datetime.now(UTC)).total_seconds()
            return max(MIN_RELAY_RETRY_DELAY_SECONDS, ceil(remaining))
    return MIN_RELAY_RETRY_DELAY_SECONDS
