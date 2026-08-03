import logging
import ssl
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

from pengine.config import Settings

_AUTO_TOOL_CHOICE_MODELS = frozenset({"deepseek-v4-flash", "deepseek-v4-pro"})
# Uvicorn owns the runtime log handlers. This child logger keeps the safe model-call
# audit at INFO while ensuring it reaches the same server evidence stream.
_MODEL_CALL_LOGGER = logging.getLogger("uvicorn.error.pengine.model_calls")
ModelRole = Literal["generation", "review"]


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


@dataclass(frozen=True, slots=True)
class _ModelCallAuditHandler(BaseCallbackHandler):
    raise_error = True
    run_inline = True

    role: ModelRole
    model_id: str

    def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[Any]],
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        del serialized, messages, kwargs
        _MODEL_CALL_LOGGER.info(
            "model_call event=start role=%s requested_model_id=%s call_id=%s",
            self.role,
            self.model_id,
            run_id,
        )

    def on_llm_end(self, response: LLMResult, *, run_id: UUID, **kwargs: Any) -> None:
        del kwargs
        response_model_ids = _response_model_ids(response)
        if response_model_ids != {self.model_id}:
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
        _MODEL_CALL_LOGGER.info(
            "model_call event=end role=%s requested_model_id=%s response_model_id=%s call_id=%s",
            self.role,
            self.model_id,
            self.model_id,
            run_id,
        )

    def on_llm_error(self, error: BaseException, *, run_id: UUID, **kwargs: Any) -> None:
        del kwargs
        _MODEL_CALL_LOGGER.warning(
            "model_call event=error role=%s requested_model_id=%s call_id=%s "
            "error_type=%s http_status=%s",
            self.role,
            self.model_id,
            run_id,
            type(error).__name__,
            _safe_http_status(error) or "none",
        )


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


def _safe_http_status(error: BaseException) -> int | None:
    for candidate in (error, *_cause_chain(error)):
        status = getattr(candidate, "status_code", None)
        if isinstance(status, int):
            return status
        response = getattr(candidate, "response", None)
        response_status = getattr(response, "status_code", None)
        if isinstance(response_status, int):
            return response_status
    return None


@dataclass(frozen=True, slots=True)
class RelayAdapter:
    model: BaseChatModel
    role: ModelRole
    model_id: str
    provider_profile_key: str


@dataclass(frozen=True, slots=True)
class RelayRoutes:
    generation: RelayAdapter
    review: RelayAdapter


@dataclass(slots=True)
class RelayError(Exception):
    code: Literal["relay_unavailable", "relay_incompatible"]
    safe_message: str

    def __str__(self) -> str:
        return self.safe_message


MIN_RELAY_RETRY_DELAY_SECONDS = 10
_RETRYABLE_RELAY_STATUSES = frozenset({429, 502, 503, 504})


@dataclass(frozen=True, slots=True)
class RetryableRelayInterruption:
    retry_delay_seconds: int


def build_relay_routes(settings: Settings) -> RelayRoutes:
    if not settings.relay_configured:
        raise RelayError(
            code="relay_unavailable",
            safe_message="Both generation and review model routes must be configured.",
        )
    return RelayRoutes(
        generation=build_relay_adapter(settings, role="generation"),
        review=build_relay_adapter(settings, role="review"),
    )


def build_relay_adapter(settings: Settings, *, role: ModelRole) -> RelayAdapter:
    if role == "generation":
        model_id = settings.generation_model_id
        max_output_tokens = settings.generation_max_output_tokens
    else:
        model_id = settings.review_model_id
        max_output_tokens = settings.review_max_output_tokens
    if not settings.relay_base_url or not settings.relay_api_key or not model_id:
        raise RelayError(
            code="relay_unavailable",
            safe_message=f"The {role} model route is not configured.",
        )

    common = {
        "model": model_id,
        "base_url": settings.relay_base_url,
        "api_key": settings.relay_api_key,
        "max_retries": 0,
        "timeout": settings.model_timeout_seconds,
        "temperature": 0,
        "callbacks": [_ModelCallAuditHandler(role=role, model_id=model_id)],
    }
    if role == "review":
        return RelayAdapter(
            model=_SerialChatDeepSeek(
                **common,
                max_tokens=max_output_tokens,
                extra_body={"thinking": {"type": "disabled"}},
            ),
            role=role,
            model_id=model_id,
            provider_profile_key="deepseek",
        )
    return RelayAdapter(
        model=_SerialChatAnthropic(
            **common,
            max_tokens=max_output_tokens,
        ),
        role=role,
        model_id=model_id,
        provider_profile_key="anthropic",
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
    type_name = type(exc).__name__.lower()
    if any(token in type_name for token in ("badrequest", "tool", "structured")):
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
