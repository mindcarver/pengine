import ssl
from uuid import uuid4

import anthropic
import httpx
import openai
import pytest
from anthropic.types import RawMessageDeltaEvent
from langchain_anthropic import ChatAnthropic
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, LLMResult
from langchain_deepseek import ChatDeepSeek
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, SecretStr

import pengine.relay as relay_module
from pengine.config import Settings
from pengine.model_calls import ModelCallState, StageCallBudgetExceeded
from pengine.relay import (
    MIN_RELAY_RETRY_DELAY_SECONDS,
    RelayError,
    RelayIdentityError,
    RelayStreamIncompleteError,
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


def test_build_relay_adapter_disables_anthropic_extended_thinking() -> None:
    for role in ("generation", "review"):
        adapter = build_relay_adapter(
            _role_settings(review_model_id="claude-opus-5"),
            role=role,
        )

        request_payload = adapter.model._get_request_payload([HumanMessage(content="ping")])

        assert request_payload["thinking"] == {"type": "disabled"}


def test_build_relay_adapter_uses_sonnet5_without_sampling_override() -> None:
    adapter = build_relay_adapter(
        _role_settings(generation_model_id="claude-sonnet-5"),
        role="generation",
    )

    assert isinstance(adapter.model, ChatAnthropic)
    assert adapter.model_id == "claude-sonnet-5"
    assert adapter.provider_profile_key == "anthropic"
    assert adapter.model.model == "claude-sonnet-5"
    assert adapter.model.temperature is None
    request_payload = adapter.model._get_request_payload([HumanMessage(content="ping")])
    assert "temperature" not in request_payload


def test_build_relay_adapter_configures_langfuse_without_sdk_environment(
    monkeypatch,
) -> None:
    client_calls: list[dict[str, str]] = []
    handler_calls: list[str | None] = []

    class FakeLangfuseClient:
        def __init__(self, **kwargs: str) -> None:
            client_calls.append(kwargs)

    class FakeLangfuseHandler(BaseCallbackHandler):
        def __init__(self, *, public_key: str | None = None) -> None:
            handler_calls.append(public_key)

    for name in ("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY", "LANGFUSE_BASE_URL"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(relay_module, "_LangfuseClient", FakeLangfuseClient)
    monkeypatch.setattr(relay_module, "_LangfuseCallbackHandler", FakeLangfuseHandler)
    settings = Settings(
        _env_file=None,
        relay_base_url="https://relay.example/v1",
        relay_api_key="secret-value",
        generation_model_id="claude-opus-5",
        review_model_id="deepseek-v4-flash",
        langfuse_enabled=True,
        langfuse_host="https://langfuse.example",
        langfuse_public_key="public-value",
        langfuse_secret_key="private-value",
    )

    adapter = build_relay_adapter(settings, role="review")

    assert client_calls == [
        {
            "public_key": "public-value",
            "secret_key": "private-value",
            "base_url": "https://langfuse.example",
        }
    ]
    assert handler_calls == ["public-value"]
    assert any(isinstance(callback, FakeLangfuseHandler) for callback in adapter.model.callbacks)


@pytest.mark.parametrize(
    "stage",
    ["generating_episode_outline", "generating_episode_scripts"],
)
def test_generation_uses_call_specific_output_budget_for_preflight_and_provider(
    stage: str,
) -> None:
    state = ModelCallState()
    state.context.stage = stage
    state.context.requested_output_tokens = 20_480
    adapter = build_relay_adapter(
        _role_settings(),
        role="generation",
        model_call_state=state,
    )
    handler = next(
        callback
        for callback in adapter.model.callbacks
        if isinstance(callback, _ModelCallAuditHandler)
    )
    handler.context_limit_tokens = 200_000
    run_id = uuid4()

    handler.on_chat_model_start(
        {},
        [[HumanMessage(content="当前剧情组")]],
        run_id=run_id,
    )

    assert handler._pending[run_id].estimated_output_tokens == 20_480
    assert adapter.model._with_call_output_budget({})["max_tokens"] == 20_480


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


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("response_models", "accepted"),
    [
        (["claude-opus-5", "claude-opus-5"], True),
        (["claude-opus-5", "relay-fallback"], False),
    ],
)
async def test_anthropic_stream_deduplicates_only_identical_model_identity_chunks(
    monkeypatch, response_models: list[str], accepted: bool
) -> None:
    async def fake_astream(*args, **kwargs):
        del args, kwargs
        for model_id in response_models:
            yield ChatGenerationChunk(
                message=AIMessageChunk(
                    content="",
                    response_metadata={"model_name": model_id},
                )
            )
        yield ChatGenerationChunk(
            message=AIMessageChunk(
                content="完成",
                response_metadata={"model_name": "claude-opus-5", "stop_reason": "end_turn"},
            )
        )

    monkeypatch.setattr(ChatAnthropic, "_astream", fake_astream)
    model = build_chat_model(_role_settings(), role="generation")
    chunks = [chunk async for chunk in model._astream([])]
    combined = chunks[0]
    for chunk in chunks[1:]:
        combined += chunk
    response = LLMResult(generations=[[combined]])
    handler = _ModelCallAuditHandler(role="generation", model_id="claude-opus-5")

    if accepted:
        handler.on_llm_end(response, run_id=uuid4())
        assert combined.message.response_metadata["model_name"] == "claude-opus-5"
    else:
        with pytest.raises(RelayIdentityError, match="identity did not match"):
            handler.on_llm_end(response, run_id=uuid4())


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("stop_reason", "content"),
    [
        ("end_turn", "完整文本"),
        ("end_turn", ""),
        ("tool_use", ""),
        ("max_tokens", "截断文本"),
    ],
)
async def test_anthropic_completed_streams_pass_the_completion_gate(
    monkeypatch, stop_reason: str, content: str
) -> None:
    async def fake_astream(*args, **kwargs):
        del args, kwargs
        yield ChatGenerationChunk(
            message=AIMessageChunk(
                content=content,
                response_metadata={"model_name": "claude-opus-5", "stop_reason": stop_reason},
            )
        )

    monkeypatch.setattr(ChatAnthropic, "_astream", fake_astream)
    model = build_chat_model(_role_settings(), role="generation")
    chunks = [chunk async for chunk in model._astream([])]
    assert chunks


@pytest.mark.asyncio
async def test_anthropic_stream_with_usage_but_no_stop_reason_passes(monkeypatch) -> None:
    async def fake_astream(*args, **kwargs):
        del args, kwargs
        yield ChatGenerationChunk(
            message=AIMessageChunk(
                content="",
                response_metadata={"model_name": "claude-opus-5"},
                usage_metadata={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
            )
        )

    monkeypatch.setattr(ChatAnthropic, "_astream", fake_astream)
    model = build_chat_model(_role_settings(), role="generation")
    chunks = [chunk async for chunk in model._astream([])]
    assert chunks


@pytest.mark.asyncio
async def test_anthropic_empty_stream_fails_as_relay_unavailable(monkeypatch) -> None:
    """Identity without content or completion evidence is a transport failure."""

    async def fake_astream(*args, **kwargs):
        del args, kwargs
        yield ChatGenerationChunk(
            message=AIMessageChunk(
                content="",
                response_metadata={"model_name": "claude-opus-5"},
            )
        )

    monkeypatch.setattr(ChatAnthropic, "_astream", fake_astream)
    model = build_chat_model(_role_settings(), role="generation")
    with pytest.raises(RelayError) as error:
        async for _ in model._astream([]):
            pass
    assert error.value.code == "relay_unavailable"
    assert "without a finish reason" in error.value.safe_message
    assert retryable_relay_interruption(error.value) is not None


@pytest.mark.asyncio
async def test_anthropic_partially_streamed_text_without_completion_fails_as_relay(
    monkeypatch,
) -> None:
    async def fake_astream(*args, **kwargs):
        del args, kwargs
        yield ChatGenerationChunk(message=AIMessageChunk(content="写到一半的"))
        yield ChatGenerationChunk(message=AIMessageChunk(content="正文"))

    monkeypatch.setattr(ChatAnthropic, "_astream", fake_astream)
    model = build_chat_model(_role_settings(), role="generation")
    with pytest.raises(RelayError) as error:
        async for _ in model._astream([]):
            pass
    assert error.value.code == "relay_unavailable"


def test_anthropic_incomplete_stream_is_a_bounded_resume_interruption() -> None:
    """The stream-completion guard is transport evidence: it auto-recovers once.

    A stream dropped without the terminal message_delta is the same provider-side
    transport failure as upstream_stream_error, so it joins the bounded recovery
    path instead of failing the run outright (Issue #264).
    """
    interruption = retryable_relay_interruption(RelayStreamIncompleteError())
    assert interruption is not None
    assert interruption.retry_delay_seconds == MIN_RELAY_RETRY_DELAY_SECONDS


def test_plain_relay_unavailable_without_the_stream_guard_stays_terminal() -> None:
    """A generic relay_unavailable (quota, unconfigured route) never auto-resumes."""
    error = RelayError(
        code="relay_unavailable",
        safe_message="The model relay closed the generation stream without a finish reason.",
    )
    assert retryable_relay_interruption(error) is None


def test_anthropic_sync_stream_deduplicates_identical_model_identity_chunks(
    monkeypatch,
) -> None:
    def fake_stream(*args, **kwargs):
        del args, kwargs
        for _ in range(2):
            yield ChatGenerationChunk(
                message=AIMessageChunk(
                    content="",
                    response_metadata={"model_name": "claude-opus-5"},
                )
            )
        yield ChatGenerationChunk(
            message=AIMessageChunk(
                content="完成",
                response_metadata={"model_name": "claude-opus-5", "stop_reason": "end_turn"},
            )
        )

    monkeypatch.setattr(ChatAnthropic, "_stream", fake_stream)
    model = build_chat_model(_role_settings(), role="generation")
    chunks = list(model._stream([]))
    combined = chunks[0]
    for chunk in chunks[1:]:
        combined += chunk

    assert combined.message.response_metadata["model_name"] == "claude-opus-5"


def test_anthropic_message_delta_normalizes_mapping_context_management() -> None:
    """Exercise the SDK event -> LangChain chunk boundary used by real streams."""
    model = build_chat_model(_role_settings(), role="generation")
    event = RawMessageDeltaEvent.model_validate(
        {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn", "stop_sequence": None},
            "usage": {"output_tokens": 17},
            "context_management": {
                "applied_edits": [{"type": "clear_tool_uses_20250919", "cleared_tool_uses": 3}]
            },
        }
    )

    chunk, block_start = model._make_message_chunk_from_anthropic_event(
        event,
        stream_usage=True,
        coerce_content_to_string=True,
    )

    assert block_start is None
    assert chunk is not None
    assert chunk.response_metadata["context_management"] == {
        "applied_edits": [{"type": "clear_tool_uses_20250919", "cleared_tool_uses": 3}]
    }
    assert chunk.usage_metadata is not None
    assert chunk.usage_metadata["output_tokens"] == 17


def test_anthropic_message_delta_rejects_invalid_context_management_shape() -> None:
    model = build_chat_model(_role_settings(), role="generation")
    event = RawMessageDeltaEvent.model_validate(
        {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn", "stop_sequence": None},
            "usage": {"output_tokens": 17},
            "context_management": ["unexpected"],
        }
    )

    with pytest.raises(RelayError) as caught:
        model._make_message_chunk_from_anthropic_event(
            event,
            stream_usage=True,
            coerce_content_to_string=True,
        )

    assert caught.value.code == "relay_incompatible"
    assert "context_management" in caught.value.safe_message
    assert "model_dump" not in caught.value.safe_message


def test_anthropic_message_delta_preserves_model_context_management() -> None:
    class ContextManagementMetadata(BaseModel):
        applied_edits: list[str]

    model = build_chat_model(_role_settings(), role="generation")
    event = RawMessageDeltaEvent.model_validate(
        {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn", "stop_sequence": None},
            "usage": {"output_tokens": 17},
        }
    ).model_copy(
        update={"context_management": ContextManagementMetadata(applied_edits=["clear_tool_uses"])}
    )

    chunk, _ = model._make_message_chunk_from_anthropic_event(
        event,
        stream_usage=True,
        coerce_content_to_string=True,
    )

    assert chunk is not None
    assert chunk.response_metadata["context_management"] == {"applied_edits": ["clear_tool_uses"]}


def test_model_call_audit_accepts_official_deepseek_flash_snapshot_identity(caplog) -> None:
    """The relay reports the dated snapshot for the canonical deepseek route."""
    caplog.set_level("INFO", logger="uvicorn.error.pengine.model_calls")
    handler = _ModelCallAuditHandler(role="review", model_id="deepseek-v4-flash")
    response = LLMResult(
        generations=[
            [
                ChatGeneration(
                    message=AIMessage(
                        content="",
                        response_metadata={"model_name": "deepseek-v4-flash-0731"},
                        tool_calls=[
                            {
                                "name": "SemanticReview",
                                "args": {"passed": True, "evidence": "通过", "issues": []},
                                "id": "review-result",
                                "type": "tool_call",
                            }
                        ],
                    )
                )
            ]
        ]
    )

    handler.on_llm_end(response, run_id=uuid4())

    assert "requested_model_id=deepseek-v4-flash" in caplog.text
    assert "response_model_id=deepseek-v4-flash-0731" in caplog.text
    assert "identity_match=explicit_equivalent" in caplog.text


@pytest.mark.parametrize(
    "response_models",
    [
        [],
        ["deepseek-v4-flash-2099"],
        ["deepseek-v4-flash-pro"],
        ["deepseek-v4-pro"],
        ["deepseek-v4-flash", "deepseek-v4-flash-0731"],
        ["gpt-5.5-2026-04-23", "deepseek-v4-flash"],
    ],
)
def test_model_call_audit_rejects_unapproved_or_ambiguous_deepseek_identity(
    response_models: list[str],
) -> None:
    response = LLMResult(
        generations=[
            [
                ChatGeneration(
                    message=AIMessage(
                        content="untrusted",
                        response_metadata={"model_name": model_id} if model_id else {},
                    )
                )
                for model_id in (response_models or [""])
            ]
        ]
    )
    handler = _ModelCallAuditHandler(role="review", model_id="deepseek-v4-flash")

    with pytest.raises(RelayIdentityError, match="identity did not match"):
        handler.on_llm_end(response, run_id=uuid4())


def test_model_call_audit_accepts_official_gpt55_snapshot_identity(caplog) -> None:
    caplog.set_level("INFO", logger="uvicorn.error.pengine.model_calls")
    handler = _ModelCallAuditHandler(role="review", model_id="gpt-5.5")
    response = LLMResult(
        generations=[
            [
                ChatGeneration(
                    message=AIMessage(
                        content="",
                        response_metadata={"model_name": "gpt-5.5-2026-04-23"},
                        tool_calls=[
                            {
                                "name": "CanonReviewerResult",
                                "args": {"passed": True, "evidence": "通过", "issues": []},
                                "id": "review-result",
                                "type": "tool_call",
                            }
                        ],
                    )
                )
            ]
        ]
    )

    handler.on_llm_end(response, run_id=uuid4())

    assert "requested_model_id=gpt-5.5" in caplog.text
    assert "response_model_id=gpt-5.5-2026-04-23" in caplog.text
    assert "identity_match=explicit_equivalent" in caplog.text


@pytest.mark.parametrize(
    "response_models",
    [
        [],
        ["gpt-5.5-2099-01-01"],
        ["gpt-5.5-pro-2026-04-23"],
        ["gpt-5.6-terra"],
        ["gpt-5.5", "gpt-5.5-2026-04-23"],
        ["gpt-5.5-2026-04-23", "deepseek-v4-flash"],
    ],
)
def test_model_call_audit_rejects_unapproved_or_ambiguous_gpt55_identity(
    response_models: list[str],
) -> None:
    response = LLMResult(
        generations=[
            [
                ChatGeneration(
                    message=AIMessage(
                        content="untrusted",
                        response_metadata={"model_name": model_id} if model_id else {},
                    )
                )
                for model_id in (response_models or [""])
            ]
        ]
    )
    handler = _ModelCallAuditHandler(role="review", model_id="gpt-5.5")

    with pytest.raises(RelayIdentityError, match="identity did not match"):
        handler.on_llm_end(response, run_id=uuid4())


def test_model_call_audit_enforces_the_review_budget_before_dispatch() -> None:
    state = ModelCallState()
    state.context.stage = "generating_character_relationships"
    handler = _ModelCallAuditHandler(
        role="review",
        model_id="gpt-5.5",
        model_call_state=state,
        context_limit_tokens=1_000,
        stage_review_call_limit=2,
    )
    response = LLMResult(
        generations=[
            [
                ChatGeneration(
                    message=AIMessage(
                        content="ok",
                        response_metadata={"model": "gpt-5.5"},
                    )
                )
            ]
        ]
    )

    for _ in range(2):
        call_id = uuid4()
        handler.on_chat_model_start(
            {},
            [[HumanMessage(content="review this candidate")]],
            run_id=call_id,
        )
        handler.on_llm_end(response, run_id=call_id)

    with pytest.raises(StageCallBudgetExceeded, match="model-call budget"):
        handler.on_chat_model_start(
            {},
            [[HumanMessage(content="review this candidate again")]],
            run_id=uuid4(),
        )


def test_model_call_audit_scopes_script_budget_per_episode_and_stage(monkeypatch) -> None:
    state = ModelCallState()
    state.context.stage = "generating_episode_scripts"
    state.context.episode_number = 1
    events: list[dict[str, object]] = []
    monkeypatch.setattr(
        "pengine.relay.record_model_call_event",
        lambda **kwargs: events.append(kwargs),
    )
    handler = _ModelCallAuditHandler(
        role="generation",
        model_id="claude-opus-5",
        model_call_state=state,
        context_limit_tokens=1_000,
        stage_call_limit=2,
        script_stage_model_call_total_limit=5,
    )
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

    def call() -> None:
        call_id = uuid4()
        handler.on_chat_model_start(
            {},
            [[HumanMessage(content="write the episode")]],
            run_id=call_id,
        )
        handler.on_llm_end(response, run_id=call_id)

    call()
    call()
    state.context.episode_number = 2
    call()
    call()
    state.context.episode_number = 1
    with pytest.raises(StageCallBudgetExceeded, match="episode 1"):
        call()

    state.context.episode_number = 3
    call()
    with pytest.raises(StageCallBudgetExceeded, match="stage reached"):
        call()

    assert [event["phase"] for event in events[:2]] == ["start", "end"]
    assert events[0]["stage"] == "generating_episode_scripts"
    assert events[0]["episode_number"] == 1
    assert events[0]["sequence"] == 1
    assert events[1]["outcome"] == "success"
    blocked = [event for event in events if event["phase"] == "blocked"]
    assert blocked[0]["episode_number"] == 1
    assert blocked[0]["sequence"] == 3
    assert blocked[0]["error_code"] == "agent_execution_limit"


@pytest.mark.parametrize(
    "response_model",
    [None, "deepseek-v4-flash", "claude-opus-5claude-opus-5"],
)
def test_model_call_audit_rejects_missing_or_mismatched_response_identity(
    response_model: str | None,
) -> None:
    metadata = {"model_name": response_model} if response_model is not None else {}
    response = LLMResult(
        generations=[[ChatGeneration(message=AIMessage(content="bad", response_metadata=metadata))]]
    )
    handler = _ModelCallAuditHandler(role="generation", model_id="claude-opus-5")

    with pytest.raises(RelayIdentityError, match="identity did not match") as excinfo:
        handler.on_llm_end(response, run_id=uuid4())
    assert excinfo.value.requested_model_id == "claude-opus-5"
    assert list(excinfo.value.response_model_ids) == (
        [] if response_model is None else [response_model]
    )


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


@pytest.mark.parametrize("model_id", ["claude-opus-5", "claude-sonnet-5"])
@pytest.mark.parametrize("max_output_tokens", [None, 16384])
def test_build_relay_adapter_uses_anthropic_for_supported_review_models(
    model_id: str,
    max_output_tokens: int | None,
) -> None:
    adapter = build_relay_adapter(
        _role_settings(review_model_id=model_id, review_max_output_tokens=max_output_tokens),
        role="review",
    )

    assert isinstance(adapter.model, ChatAnthropic)
    assert adapter.role == "review"
    assert adapter.model_id == model_id
    assert adapter.provider_profile_key == "anthropic"
    assert adapter.model.model == model_id
    assert adapter.model.temperature == (None if model_id == "claude-sonnet-5" else 0)


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


def test_generic_http_400_upstream_failure_is_retryable_unavailable() -> None:
    request = httpx.Request("POST", "https://relay.example/v1/chat/completions")
    body = {
        "code": None,
        "message": "Upstream request failed",
        "param": "",
        "type": "upstream_error",
    }
    error = openai.BadRequestError(
        "Upstream request failed",
        response=httpx.Response(400, request=request, json=body),
        body=body,
    )

    mapped = classify_relay_exception(error)
    interruption = retryable_relay_interruption(error)

    assert mapped.code == "relay_unavailable"
    assert mapped.http_status == 400
    assert mapped.provider_error_code == "upstream_error"
    assert "temporary upstream failure" in mapped.safe_message
    assert interruption is not None
    assert interruption.retry_delay_seconds == MIN_RELAY_RETRY_DELAY_SECONDS


@pytest.mark.parametrize(
    ("message", "code"),
    [
        ("insufficient tool messages following tool_calls message", "upstream_error"),
        ("maximum context length exceeded", "context_length_exceeded"),
        ("Upstream request failed", "invalid_request_error"),
    ],
)
def test_specific_http_400_request_failures_remain_terminal(
    message: str,
    code: str,
) -> None:
    error = _openai_bad_request(message=message, code=code)

    mapped = classify_relay_exception(error)

    assert mapped.code == "relay_rejected"
    assert retryable_relay_interruption(error) is None


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


@pytest.mark.parametrize("provider_error", [anthropic.APIStatusError, openai.APIStatusError])
def test_http_200_upstream_stream_error_is_a_relay_interruption(provider_error) -> None:
    request = httpx.Request("POST", "https://relay.example/v1/messages")
    body = {
        "error": {
            "message": "error decoding response body",
            "type": "upstream_stream_error",
        },
        "type": "error",
    }
    response = httpx.Response(200, request=request, json=body)
    error = provider_error("error decoding response body", response=response, body=body)

    assert is_relay_exception(error)
    assert is_relay_connection_error(error)
    interruption = retryable_relay_interruption(error)
    assert interruption is not None
    assert interruption.retry_delay_seconds == MIN_RELAY_RETRY_DELAY_SECONDS

    mapped = classify_relay_exception(error)
    assert mapped.code == "relay_unavailable"
    assert mapped.http_status == 200
    assert mapped.provider_error_code == "upstream_stream_error"
    assert mapped.redacted_body is not None
    assert "upstream_stream_error" in mapped.redacted_body
    assert "decoding response body" in mapped.safe_message


@pytest.mark.parametrize("provider_error", [anthropic.APIStatusError, openai.APIStatusError])
def test_http_200_exact_upstream_timeout_is_a_relay_interruption(provider_error) -> None:
    request = httpx.Request("POST", "https://relay.example/v1/messages")
    body = {"error": "upstream_timeout"}
    response = httpx.Response(200, request=request, json=body)
    error = provider_error("upstream timeout", response=response, body=body)

    assert is_relay_connection_error(error)
    assert retryable_relay_interruption(error) is not None


@pytest.mark.parametrize(
    "body",
    [
        {"error": "timeout"},
        {"error": "upstream_timeout", "extra": True},
        {"error": {"type": "upstream_timeout"}},
    ],
)
def test_http_200_similar_upstream_timeout_shapes_are_not_retryable(
    body: dict[str, object],
) -> None:
    request = httpx.Request("POST", "https://relay.example/v1/messages")
    response = httpx.Response(200, request=request, json=body)
    error = anthropic.APIStatusError("upstream timeout", response=response, body=body)

    assert retryable_relay_interruption(error) is None
    assert not is_relay_connection_error(error)


@pytest.mark.parametrize(
    ("status_code", "body"),
    [
        (
            200,
            {"error": {"message": "error decoding response body", "type": "server_error"}},
        ),
        (
            200,
            {
                "error": {
                    "message": "error decoding response body",
                    "type": "authentication_error",
                }
            },
        ),
        (
            400,
            {
                "error": {
                    "message": "error decoding response body",
                    "type": "upstream_stream_error",
                }
            },
        ),
    ],
)
def test_http_200_stream_like_messages_and_http_400_are_not_retryable(
    status_code: int,
    body: dict[str, object],
) -> None:
    request = httpx.Request("POST", "https://relay.example/v1/messages")
    response = httpx.Response(status_code, request=request, json=body)
    error = anthropic.APIStatusError("error decoding response body", response=response, body=body)

    assert retryable_relay_interruption(error) is None
    assert not is_relay_connection_error(error)


def test_structured_output_validation_error_is_not_a_stream_interruption() -> None:
    request = httpx.Request("POST", "https://relay.example/v1/messages")
    body = {
        "error": {
            "message": "error decoding response body",
            "type": "upstream_stream_error",
        }
    }
    error = anthropic.APIResponseValidationError(
        httpx.Response(200, request=request, json=body),
        body=body,
    )

    assert retryable_relay_interruption(error) is None
    assert not is_relay_connection_error(error)


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
    20260804T131739Z-02ea1827 failed in `generating_story_outline` after
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
