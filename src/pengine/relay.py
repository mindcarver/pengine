import ssl
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from math import ceil
from typing import Any, Literal

import httpx
from langchain_anthropic import ChatAnthropic

from pengine.config import Settings

_AUTO_TOOL_CHOICE_MODELS = frozenset({"deepseek-v4-flash"})


class _SerialChatAnthropic(ChatAnthropic):
    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> Any:
        kwargs["parallel_tool_calls"] = False
        kwargs["tool_choice"] = "auto" if self.model in _AUTO_TOOL_CHOICE_MODELS else "any"
        return super().bind_tools(tools, **kwargs)


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


def build_chat_model(settings: Settings) -> ChatAnthropic:
    if not settings.relay_configured:
        raise RelayError(
            code="relay_unavailable",
            safe_message="The model relay is not configured.",
        )

    return _SerialChatAnthropic(
        model=settings.relay_model_id,
        base_url=settings.relay_base_url,
        api_key=settings.relay_api_key,
        max_retries=0,
        timeout=settings.model_timeout_seconds,
        max_tokens=8192,
        temperature=0,
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
    if _is_anthropic_connection_error(exc) and any(
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
        ),
    ):
        return True
    return type(exc).__module__.startswith("httpcore") and type(exc).__name__ in {
        "ConnectError",
        "ReadError",
        "ConnectTimeout",
        "ReadTimeout",
    }


def _is_anthropic_connection_error(exc: BaseException) -> bool:
    return any(
        base.__module__.startswith("anthropic") and base.__name__ == "APIConnectionError"
        for base in type(exc).__mro__
    )


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
            base.__module__.startswith("anthropic") and base.__name__ == "APIStatusError"
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
