from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

from langchain_anthropic import ChatAnthropic

from pengine.config import Settings


class _SerialChatAnthropic(ChatAnthropic):
    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> Any:
        kwargs["parallel_tool_calls"] = False
        kwargs.setdefault("tool_choice", "any")
        return super().bind_tools(tools, **kwargs)


@dataclass(slots=True)
class RelayError(Exception):
    code: Literal["relay_unavailable", "relay_incompatible"]
    safe_message: str

    def __str__(self) -> str:
        return self.safe_message


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
