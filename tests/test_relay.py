import ssl

import anthropic
import httpx
import openai
import pytest
from langchain_anthropic import ChatAnthropic
from langchain_deepseek import ChatDeepSeek
from pydantic import BaseModel, SecretStr

from pengine.config import Settings
from pengine.relay import (
    MIN_RELAY_RETRY_DELAY_SECONDS,
    RelayError,
    build_chat_model,
    build_relay_adapter,
    classify_relay_exception,
    is_relay_connection_error,
    is_relay_exception,
    retryable_relay_interruption,
)


def test_build_chat_model_uses_only_operator_connection_fields() -> None:
    settings = Settings(
        relay_base_url="https://relay.example/anthropic",
        relay_api_key="secret-value",
        relay_model_id="model-id",
    )

    model = build_chat_model(settings)

    assert model.model == "model-id"
    assert model.anthropic_api_url == "https://relay.example/anthropic"
    assert model.max_retries == 0
    assert isinstance(model.anthropic_api_key, SecretStr)
    assert model.anthropic_api_key.get_secret_value() == "secret-value"
    assert "secret-value" not in repr(model)


def test_build_relay_adapter_preserves_anthropic_defaults() -> None:
    adapter = build_relay_adapter(
        Settings(
            relay_base_url="https://relay.example/anthropic",
            relay_api_key="secret-value",
            relay_model_id="model-id",
        )
    )

    assert isinstance(adapter.model, ChatAnthropic)
    assert adapter.provider_profile_key == "anthropic"
    assert adapter.model.max_tokens == 8192


@pytest.mark.parametrize("max_output_tokens", [None, 16384])
def test_build_relay_adapter_uses_native_deepseek_without_an_implicit_token_cap(
    max_output_tokens: int | None,
) -> None:
    adapter = build_relay_adapter(
        Settings(
            relay_adapter="deepseek",
            relay_base_url="https://relay.example/v1",
            relay_api_key="secret-value",
            relay_model_id="deepseek-v4-flash",
            relay_max_output_tokens=max_output_tokens,
        )
    )

    assert isinstance(adapter.model, ChatDeepSeek)
    assert adapter.provider_profile_key == "deepseek"
    assert adapter.model.model_name == "deepseek-v4-flash"
    assert adapter.model.openai_api_base == "https://relay.example/v1"
    assert adapter.model.max_tokens == max_output_tokens
    assert adapter.model.extra_body == {"thinking": {"type": "disabled"}}
    assert adapter.model.max_retries == 0
    assert adapter.model.openai_api_key.get_secret_value() == "secret-value"
    assert "secret-value" not in repr(adapter.model)


@pytest.mark.parametrize("tool_choice", ["any", "required"])
def test_deepseek_uses_auto_serial_tools_for_mixed_tool_strategy(tool_choice: str) -> None:
    class ProbeTool(BaseModel):
        value: str

    adapter = build_relay_adapter(
        Settings(
            relay_adapter="deepseek",
            relay_base_url="https://relay.example/v1",
            relay_api_key="secret-value",
            relay_model_id="deepseek-v4-flash",
        )
    )

    bound = adapter.model.bind_tools([ProbeTool], tool_choice=tool_choice)

    assert bound.kwargs["tool_choice"] == "auto"
    assert bound.kwargs["parallel_tool_calls"] is False


def test_deepseek_preserves_named_tool_choice() -> None:
    class ProbeTool(BaseModel):
        value: str

    adapter = build_relay_adapter(
        Settings(
            relay_adapter="deepseek",
            relay_base_url="https://relay.example/v1",
            relay_api_key="secret-value",
            relay_model_id="deepseek-v4-flash",
        )
    )
    tool_choice = {"type": "function", "function": {"name": "ProbeTool"}}

    bound = adapter.model.bind_tools([ProbeTool], tool_choice=tool_choice)

    assert bound.kwargs["tool_choice"] == tool_choice
    assert bound.kwargs["parallel_tool_calls"] is False


def test_deepseek_structured_output_forces_its_named_result_tool() -> None:
    class ProbeTool(BaseModel):
        value: str

    adapter = build_relay_adapter(
        Settings(
            relay_adapter="deepseek",
            relay_base_url="https://relay.example/v1",
            relay_api_key="secret-value",
            relay_model_id="deepseek-v4-flash",
        )
    )

    structured = adapter.model.with_structured_output(
        ProbeTool,
        method="function_calling",
    )

    assert structured.first.kwargs["tool_choice"] == {
        "type": "function",
        "function": {"name": "ProbeTool"},
    }
    assert structured.first.kwargs["parallel_tool_calls"] is False


def test_build_chat_model_requires_serial_tool_calls_and_a_tool_result() -> None:
    class ProbeTool(BaseModel):
        value: str

    settings = Settings(
        relay_base_url="https://relay.example/anthropic",
        relay_api_key="secret-value",
        relay_model_id="model-id",
    )

    bound = build_chat_model(settings).bind_tools([ProbeTool], tool_choice="auto")

    assert bound.kwargs["tool_choice"]["type"] == "any"
    assert bound.kwargs["tool_choice"]["disable_parallel_tool_use"] is True


@pytest.mark.parametrize("model_id", ["deepseek-v4-flash", "deepseek-v4-pro"])
def test_deepseek_anthropic_compat_uses_auto_tool_choice_and_serial_calls(
    model_id: str,
) -> None:
    class ProbeTool(BaseModel):
        value: str

    settings = Settings(
        relay_base_url="https://relay.example/anthropic",
        relay_api_key="secret-value",
        relay_model_id=model_id,
    )

    bound = build_chat_model(settings).bind_tools([ProbeTool], tool_choice="any")

    assert bound.kwargs["tool_choice"]["type"] == "auto"
    assert bound.kwargs["tool_choice"]["disable_parallel_tool_use"] is True


def test_build_chat_model_initializes_with_a_socks_proxy(monkeypatch) -> None:
    for variable in ("HTTP_PROXY", "HTTPS_PROXY", "NO_PROXY", "no_proxy"):
        monkeypatch.delenv(variable, raising=False)
    monkeypatch.setenv("ALL_PROXY", "socks5://127.0.0.1:1080")
    settings = Settings(
        relay_base_url="https://relay.example/anthropic",
        relay_api_key="secret-value",
        relay_model_id="model-id",
    )

    assert build_chat_model(settings)._async_client is not None


def test_missing_relay_configuration_is_safe(monkeypatch) -> None:
    monkeypatch.setenv("PENGINE_RELAY_BASE_URL", "https://ambient.example/anthropic")
    monkeypatch.setenv("PENGINE_RELAY_API_KEY", "ambient-secret")
    monkeypatch.setenv("PENGINE_RELAY_MODEL_ID", "ambient-model")
    settings = Settings(
        _env_file=None,
        relay_base_url=None,
        relay_api_key=None,
        relay_model_id=None,
    )

    try:
        build_chat_model(settings)
    except RelayError as exc:
        assert exc.code == "relay_unavailable"
        assert "key" not in exc.safe_message.lower()
    else:
        raise AssertionError("Expected missing relay configuration to fail")


def test_exception_mapping_does_not_echo_provider_body() -> None:
    class BadRequestError(Exception):
        pass

    raw = "vendor body with api-key=secret-value and full prompt"
    mapped = classify_relay_exception(BadRequestError(raw))

    assert mapped.code == "relay_incompatible"
    assert raw not in mapped.safe_message
    assert "secret-value" not in mapped.safe_message


@pytest.mark.parametrize(
    "error",
    [
        anthropic.APIConnectionError(request=httpx.Request("POST", "https://relay.example")),
        openai.APIConnectionError(request=httpx.Request("POST", "https://relay.example")),
        httpx.ConnectError("connection failed"),
    ],
)
def test_relay_exception_helpers_recognize_supported_connection_errors(error: Exception) -> None:
    assert is_relay_exception(error)
    assert is_relay_connection_error(error)


@pytest.mark.parametrize(
    "error",
    [
        httpx.ConnectError("dns unavailable"),
        httpx.ReadError("connection reset"),
        httpx.ConnectTimeout("connect timed out"),
        httpx.ReadTimeout("read timed out"),
        ConnectionResetError("connection reset"),
    ],
)
def test_retryable_relay_interruption_allows_only_approved_transport_errors(
    error: Exception,
) -> None:
    interruption = retryable_relay_interruption(error)

    assert interruption is not None
    assert interruption.retry_delay_seconds == MIN_RELAY_RETRY_DELAY_SECONDS


def test_retryable_relay_interruption_honours_retry_after() -> None:
    request = httpx.Request("POST", "https://relay.example/messages")
    response = httpx.Response(429, headers={"retry-after": "17"}, request=request)

    interruption = retryable_relay_interruption(
        httpx.HTTPStatusError("rate limited", request=request, response=response)
    )

    assert interruption is not None
    assert interruption.retry_delay_seconds == 17


def test_retryable_relay_interruption_keeps_tls_certificate_configuration_terminal() -> None:
    try:
        raise ssl.SSLCertVerificationError(1, "certificate verify failed")
    except ssl.SSLCertVerificationError as cause:
        try:
            raise httpx.ConnectError("TLS verification failed") from cause
        except httpx.ConnectError as error:
            assert retryable_relay_interruption(error) is None


@pytest.mark.parametrize("status_code", [429, 502, 503, 504])
def test_retryable_relay_interruption_accepts_allowed_statuses_and_transport_cause(
    status_code: int,
) -> None:
    request = httpx.Request("POST", "https://relay.example/messages")
    response = httpx.Response(status_code, request=request)
    status = anthropic.APIStatusError("unavailable", response=response, body={})
    assert retryable_relay_interruption(status) is not None

    try:
        raise httpx.ReadTimeout("read timed out", request=request)
    except httpx.ReadTimeout as cause:
        try:
            raise anthropic.APIConnectionError(request=request) from cause
        except anthropic.APIConnectionError as wrapped:
            assert retryable_relay_interruption(wrapped) is not None


@pytest.mark.parametrize("status_code", [429, 502, 503, 504])
def test_retryable_relay_interruption_accepts_openai_status_and_connection_errors(
    status_code: int,
) -> None:
    request = httpx.Request("POST", "https://relay.example/v1/chat/completions")
    response = httpx.Response(status_code, request=request)
    status = openai.APIStatusError("unavailable", response=response, body={})
    assert retryable_relay_interruption(status) is not None

    try:
        raise httpx.ReadTimeout("read timed out", request=request)
    except httpx.ReadTimeout as cause:
        try:
            raise openai.APIConnectionError(request=request) from cause
        except openai.APIConnectionError as wrapped:
            assert retryable_relay_interruption(wrapped) is not None


@pytest.mark.parametrize(
    "factory",
    [
        lambda request: FileNotFoundError("configured CA file is unavailable"),
        lambda request: httpx.WriteError("write failed", request=request),
        lambda request: httpx.CloseError("close failed", request=request),
        lambda request: httpx.WriteTimeout("write timed out", request=request),
        lambda request: httpx.PoolTimeout("pool timed out", request=request),
        lambda request: httpx.RemoteProtocolError("protocol failed", request=request),
        lambda request: httpx.HTTPStatusError(
            "unexpected server failure",
            request=request,
            response=httpx.Response(500, request=request),
        ),
        lambda request: anthropic.APIResponseValidationError(
            httpx.Response(502, request=request), body={}
        ),
    ],
)
def test_retryable_relay_interruption_keeps_protocol_and_unknown_failures_terminal(factory) -> None:
    request = httpx.Request("POST", "https://relay.example/messages")

    assert retryable_relay_interruption(factory(request)) is None
