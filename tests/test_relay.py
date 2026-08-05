import ssl
from uuid import uuid4

import anthropic
import httpx
import openai
import pytest
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, LLMResult
from langchain_deepseek import ChatDeepSeek
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, SecretStr

from pengine.config import Settings
from pengine.relay import (
    MIN_RELAY_RETRY_DELAY_SECONDS,
    RelayError,
    _ModelCallAuditHandler,
    build_chat_model,
    build_relay_adapter,
    build_relay_routes,
    classify_relay_exception,
    is_relay_connection_error,
    is_relay_exception,
    retryable_relay_interruption,
)


def _role_settings(
    *,
    generation_model_id: str = "claude-opus-5",
    generation_max_output_tokens: int = 128_000,
    review_model_id: str = "deepseek-v4-flash",
    review_max_output_tokens: int | None = None,
) -> Settings:
    return Settings(
        _env_file=None,
        relay_base_url="https://relay.example/v1",
        relay_api_key="secret-value",
        generation_model_id=generation_model_id,
        generation_max_output_tokens=generation_max_output_tokens,
        review_model_id=review_model_id,
        review_max_output_tokens=review_max_output_tokens,
    )


def test_build_chat_model_uses_only_operator_connection_fields() -> None:
    settings = _role_settings()

    model = build_chat_model(settings, role="generation")

    assert model.model == "claude-opus-5"
    assert model.anthropic_api_url == "https://relay.example/v1"
    assert model.max_retries == 0
    assert isinstance(model.anthropic_api_key, SecretStr)
    assert model.anthropic_api_key.get_secret_value() == "secret-value"
    assert "secret-value" not in repr(model)


def test_build_relay_adapter_preserves_anthropic_defaults() -> None:
    adapter = build_relay_adapter(
        _role_settings(),
        role="generation",
    )

    assert isinstance(adapter.model, ChatAnthropic)
    assert adapter.role == "generation"
    assert adapter.model_id == "claude-opus-5"
    assert adapter.provider_profile_key == "anthropic"
    assert adapter.model.max_tokens == 128_000


def test_build_relay_routes_keeps_generation_and_review_models_separate() -> None:
    routes = build_relay_routes(_role_settings())

    assert isinstance(routes.generation.model, ChatAnthropic)
    assert routes.generation.role == "generation"
    assert routes.generation.model_id == "claude-opus-5"
    assert isinstance(routes.review.model, ChatDeepSeek)
    assert routes.review.role == "review"
    assert routes.review.model_id == "deepseek-v4-flash"
    assert routes.generation.model is not routes.review.model


def test_model_call_audit_requires_exact_response_model_identity(caplog) -> None:
    caplog.set_level("INFO", logger="uvicorn.error.pengine.model_calls")
    handler = _ModelCallAuditHandler(role="generation", model_id="claude-opus-5")
    response = LLMResult(
        generations=[
            [
                ChatGeneration(
                    message=AIMessage(
                        content="ok",
                        response_metadata={"model": "claude-opus-5"},
                    )
                )
            ]
        ]
    )

    handler.on_llm_end(response, run_id=uuid4())

    assert "requested_model_id=claude-opus-5" in caplog.text
    assert "response_model_id=claude-opus-5" in caplog.text


@pytest.mark.parametrize("response_model", [None, "deepseek-v4-flash"])
def test_model_call_audit_rejects_missing_or_mismatched_response_identity(
    response_model: str | None,
) -> None:
    metadata = {"model_name": response_model} if response_model is not None else {}
    response = LLMResult(
        generations=[[ChatGeneration(message=AIMessage(content="bad", response_metadata=metadata))]]
    )
    handler = _ModelCallAuditHandler(role="generation", model_id="claude-opus-5")

    with pytest.raises(RelayError, match="unexpected model"):
        handler.on_llm_end(response, run_id=uuid4())


@pytest.mark.parametrize(
    "response_model",
    ["DeepSeek-V4-Flash", "deepseek-v4-flash", "DEEPSEEK-V4-FLASH"],
)
def test_model_call_audit_accepts_case_variant_model_identity(response_model: str) -> None:
    handler = _ModelCallAuditHandler(role="review", model_id="deepseek-v4-flash")
    response = LLMResult(
        generations=[
            [
                ChatGeneration(
                    message=AIMessage(
                        content="ok",
                        response_metadata={"model": response_model},
                    )
                )
            ]
        ]
    )

    handler.on_llm_end(response, run_id=uuid4())


@pytest.mark.parametrize("max_output_tokens", [None, 16384])
def test_build_relay_adapter_uses_native_deepseek_without_an_implicit_token_cap(
    max_output_tokens: int | None,
) -> None:
    adapter = build_relay_adapter(
        _role_settings(review_max_output_tokens=max_output_tokens),
        role="review",
    )

    assert isinstance(adapter.model, ChatDeepSeek)
    assert adapter.role == "review"
    assert adapter.model_id == "deepseek-v4-flash"
    assert adapter.provider_profile_key == "deepseek"
    assert adapter.model.model_name == "deepseek-v4-flash"
    assert adapter.model.openai_api_base == "https://relay.example/v1"
    assert adapter.model.max_tokens == max_output_tokens
    assert adapter.model.extra_body == {"thinking": {"type": "disabled"}}
    assert adapter.model.max_retries == 0
    assert adapter.model.openai_api_key.get_secret_value() == "secret-value"
    assert "secret-value" not in repr(adapter.model)


@pytest.mark.parametrize("max_output_tokens", [None, 16384])
def test_build_relay_adapter_uses_openai_for_gpt55_review(max_output_tokens: int | None) -> None:
    adapter = build_relay_adapter(
        _role_settings(review_model_id="gpt-5.5", review_max_output_tokens=max_output_tokens),
        role="review",
    )

    assert isinstance(adapter.model, ChatOpenAI)
    assert not isinstance(adapter.model, ChatDeepSeek)
    assert adapter.role == "review"
    assert adapter.model_id == "gpt-5.5"
    assert adapter.provider_profile_key == "openai"
    assert adapter.model.model_name == "gpt-5.5"
    assert adapter.model.openai_api_base == "https://relay.example/v1"
    assert adapter.model.max_tokens == max_output_tokens
    # GPT-5.5 must NOT receive the DeepSeek-specific "thinking" extra_body.
    assert adapter.model.extra_body is None
    assert adapter.model.max_retries == 0
    assert "secret-value" not in repr(adapter.model)


@pytest.mark.parametrize("tool_choice", ["any", "required"])
def test_deepseek_uses_auto_serial_tools_for_mixed_tool_strategy(tool_choice: str) -> None:
    class WorkTool(BaseModel):
        value: str

    class ResultTool(BaseModel):
        value: str

    adapter = build_relay_adapter(
        _role_settings(),
        role="review",
    )

    bound = adapter.model.bind_tools([WorkTool, ResultTool], tool_choice=tool_choice)

    assert bound.kwargs["tool_choice"] == "auto"
    assert bound.kwargs["parallel_tool_calls"] is False


@pytest.mark.parametrize("tool_choice", ["any", "required"])
def test_deepseek_requires_the_only_available_result_tool(tool_choice: str) -> None:
    class ResultTool(BaseModel):
        value: str

    adapter = build_relay_adapter(
        _role_settings(),
        role="review",
    )

    bound = adapter.model.bind_tools([ResultTool], tool_choice=tool_choice)

    assert bound.kwargs["tool_choice"] == "required"
    assert bound.kwargs["parallel_tool_calls"] is False


def test_deepseek_preserves_named_tool_choice() -> None:
    class ProbeTool(BaseModel):
        value: str

    adapter = build_relay_adapter(
        _role_settings(),
        role="review",
    )
    tool_choice = {"type": "function", "function": {"name": "ProbeTool"}}

    bound = adapter.model.bind_tools([ProbeTool], tool_choice=tool_choice)

    assert bound.kwargs["tool_choice"] == tool_choice
    assert bound.kwargs["parallel_tool_calls"] is False


def test_deepseek_structured_output_forces_its_named_result_tool() -> None:
    class ProbeTool(BaseModel):
        value: str

    adapter = build_relay_adapter(
        _role_settings(),
        role="review",
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

    settings = _role_settings()

    bound = build_chat_model(settings, role="generation").bind_tools(
        [ProbeTool], tool_choice="auto"
    )

    assert bound.kwargs["tool_choice"]["type"] == "any"
    assert bound.kwargs["tool_choice"]["disable_parallel_tool_use"] is True


def test_build_chat_model_initializes_with_a_socks_proxy(monkeypatch) -> None:
    for variable in ("HTTP_PROXY", "HTTPS_PROXY", "NO_PROXY", "no_proxy"):
        monkeypatch.delenv(variable, raising=False)
    monkeypatch.setenv("ALL_PROXY", "socks5://127.0.0.1:1080")
    settings = _role_settings()

    assert build_chat_model(settings, role="generation")._async_client is not None


def test_missing_relay_configuration_is_safe(monkeypatch) -> None:
    monkeypatch.setenv("PENGINE_RELAY_BASE_URL", "https://ambient.example/anthropic")
    monkeypatch.setenv("PENGINE_RELAY_API_KEY", "ambient-secret")
    monkeypatch.setenv("PENGINE_GENERATION_MODEL_ID", "ambient-generation-model")
    monkeypatch.setenv("PENGINE_REVIEW_MODEL_ID", "ambient-review-model")
    settings = Settings(
        _env_file=None,
        relay_base_url=None,
        relay_api_key=None,
        generation_model_id=None,
        review_model_id=None,
    )

    try:
        build_chat_model(settings, role="generation")
    except RelayError as exc:
        assert exc.code == "relay_unavailable"
        assert "key" not in exc.safe_message.lower()
    else:
        raise AssertionError("Expected missing relay configuration to fail")


def _openai_bad_request(
    *,
    message: str,
    code: str | None = None,
) -> openai.BadRequestError:
    request = httpx.Request("POST", "https://relay.example/v1/chat/completions")
    body: dict[str, object] = {
        "error": {
            "message": message,
            "type": "invalid_request_error",
        }
    }
    if code is not None:
        body["error"]["code"] = code  # type: ignore[index]
    response = httpx.Response(400, request=request, json=body)
    return openai.BadRequestError(message, response=response, body=body)


def test_classify_relay_exception_400_content_rejection_is_not_tool_protocol() -> None:
    """A provider 400 that rejects the *request content* must not be labeled as a
    tool-protocol incompatibility (Issue #52 graph revision 4 defect 1)."""
    error = _openai_bad_request(
        message="this model's maximum context length is 32768 tokens; you requested 52000 tokens",
        code="context_length_exceeded",
    )

    mapped = classify_relay_exception(error)

    assert mapped.http_status == 400
    assert mapped.provider_error_code == "context_length_exceeded"
    assert mapped.code == "relay_rejected"
    assert "unsupported tool protocol" not in mapped.safe_message
    assert "400" in mapped.safe_message
    assert "context length" in mapped.safe_message
    assert "secret" not in mapped.safe_message


def test_classify_relay_exception_400_tool_protocol_keeps_incompatible() -> None:
    """A provider 400 that does reject the tool protocol stays relay_incompatible,
    but now carries the precise status and truthful provider detail."""
    request = httpx.Request("POST", "https://relay.example/v1/messages")
    body = {
        "type": "error",
        "error": {"type": "invalid_request_error", "message": "tool_use blocks are not supported"},
    }
    response = httpx.Response(400, request=request, json=body)
    error = anthropic.BadRequestError("bad", response=response, body=body)

    mapped = classify_relay_exception(error)

    assert mapped.http_status == 400
    assert mapped.code == "relay_incompatible"
    assert "tool" in mapped.safe_message
    assert "400" in mapped.safe_message


def test_classify_relay_exception_401_keeps_precise_status_and_message() -> None:
    request = httpx.Request("POST", "https://relay.example/v1/messages")
    body = {
        "type": "error",
        "error": {"type": "authentication_error", "message": "invalid x-api-key"},
    }
    response = httpx.Response(401, request=request, json=body)
    error = anthropic.AuthenticationError("invalid x-api-key", response=response, body=body)

    mapped = classify_relay_exception(error)

    assert mapped.http_status == 401
    assert mapped.code == "relay_unavailable"
    assert "401" in mapped.safe_message


def test_classify_relay_exception_500_carries_precise_status_and_evidence() -> None:
    """Non-retryable 5xx responses stay relay_unavailable but still expose the exact
    status and redacted provider detail so the external block stays distinguishable."""
    request = httpx.Request("POST", "https://relay.example/v1/chat/completions")
    body = {"error": {"message": "upstream storage error", "type": "server_error"}}
    response = httpx.Response(500, request=request, json=body)
    error = openai.InternalServerError("upstream storage error", response=response, body=body)

    mapped = classify_relay_exception(error)

    assert mapped.http_status == 500
    assert mapped.code == "relay_unavailable"
    assert "500" in mapped.safe_message
    assert "storage error" in mapped.safe_message


def test_classify_relay_exception_400_redacts_provider_body() -> None:
    error = _openai_bad_request(
        message="rejected with sk-abc123456789abcdefg token material",
    )

    mapped = classify_relay_exception(error)

    assert mapped.http_status == 400
    assert mapped.redacted_body is not None
    assert "sk-abc123456789abcdefg" not in mapped.redacted_body
    assert "sk-***" in mapped.redacted_body


def test_exception_mapping_does_not_echo_provider_body() -> None:
    class BareBadRequestError(Exception):
        pass

    raw = "vendor body with api-key=secret-value and full prompt"
    mapped = classify_relay_exception(BareBadRequestError(raw))

    # A bare error without a precise HTTP status must not be heuristically
    # labeled a tool-protocol incompatibility (Issue #52 graph revision 4 defect 1).
    assert mapped.code != "relay_incompatible"
    assert mapped.http_status is None
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
        httpx.RemoteProtocolError("server disconnected"),
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


@pytest.mark.parametrize("status_code", [408, 429, 502, 503, 504])
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


@pytest.mark.parametrize("status_code", [408, 429, 502, 503, 504])
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


def test_retryable_relay_interruption_recovers_provider_408_request_timeout() -> None:
    """A provider-side HTTP 408 request timeout (`timeout_error`) is transient
    congestion, not a terminal relay failure: the run recovers instead of failing
    on a single timed-out call (Issue #52 graph revision 10; E2E
    20260804T131739Z-02ea1827 failed in `generating_character_biographies` after
    one 408 with attempt_count=1, recovery_state=none)."""
    request = httpx.Request("POST", "https://relay.example/v1/chat/completions")
    body = {
        "message": "请求超时，请稍后重试",
        "type": "timeout_error",
    }
    response = httpx.Response(408, request=request, json=body)
    error = openai.APIStatusError("request timed out", response=response, body=body)

    interruption = retryable_relay_interruption(error)

    assert interruption is not None
    assert interruption.retry_delay_seconds >= MIN_RELAY_RETRY_DELAY_SECONDS


def test_classify_relay_exception_408_keeps_unavailable_with_timeout_evidence() -> None:
    """HTTP 408 still classifies as relay_unavailable with the provider's
    timeout detail surfaced, but it is now retryable so the worker recovers
    rather than terminating the run."""
    request = httpx.Request("POST", "https://relay.example/v1/chat/completions")
    body = {
        "message": "请求超时，请稍后重试",
        "type": "timeout_error",
    }
    response = httpx.Response(408, request=request, json=body)
    error = openai.APIStatusError("request timed out", response=response, body=body)

    mapped = classify_relay_exception(error)

    assert mapped.http_status == 408
    assert mapped.code == "relay_unavailable"
    assert retryable_relay_interruption(error) is not None


@pytest.mark.parametrize(
    "factory",
    [
        lambda request: FileNotFoundError("configured CA file is unavailable"),
        lambda request: httpx.WriteError("write failed", request=request),
        lambda request: httpx.CloseError("close failed", request=request),
        lambda request: httpx.WriteTimeout("write timed out", request=request),
        lambda request: httpx.PoolTimeout("pool timed out", request=request),
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
