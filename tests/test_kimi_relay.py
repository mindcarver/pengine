from typing import Any
from uuid import uuid4

import pytest
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, LLMResult
from openai.types.chat import ChatCompletion

from pengine.config import Settings
from pengine.model_calls import ModelCallState, estimate_messages_tokens
from pengine.relay import (
    RelayIdentityError,
    _ModelCallAuditHandler,
    _stream_chunk_text_chars,
    build_relay_adapter,
    build_relay_routes,
)

KIMI = "moonshotai/kimi-k3"
DEEPSEEK = "deepseek/deepseek-v4-flash"
REASONING = {
    "reasoning_content": "先验证工具结果。",
    "reasoning": "先验证工具结果。",
    "reasoning_details": [
        {"type": "reasoning.text", "text": "先验证工具结果。", "index": 0, "format": "unknown"}
    ],
}


def _settings(**overrides: Any) -> Settings:
    return Settings(
        _env_file=None,
        relay_base_url="https://relay.example/v1",
        relay_api_key="test-key",
        generation_model_id=overrides.pop("generation_model_id", KIMI),
        review_model_id=overrides.pop("review_model_id", KIMI),
        generation_context_limit_tokens=1_048_576,
        review_context_limit_tokens=1_048_576,
        **overrides,
    )


@pytest.mark.parametrize("generation,review", [(KIMI, KIMI), (KIMI, DEEPSEEK), (DEEPSEEK, KIMI)])
def test_kimi_and_deepseek_routes_coexist(generation: str, review: str) -> None:
    routes = build_relay_routes(
        _settings(generation_model_id=generation, review_model_id=review, outline_model_id=KIMI)
    )
    assert routes.generation.model_id == generation
    assert routes.review.model_id == review
    assert routes.outline.model_id == KIMI
    assert routes.generation.model is not routes.review.model
    for adapter in (routes.generation, routes.review, routes.outline):
        assert adapter.provider_profile_key == "openrouter"
        payload = adapter.model._get_request_payload([HumanMessage(content="ping")])
        if adapter.model_id == KIMI:
            assert "temperature" not in payload
            assert payload["extra_body"]["reasoning"] == {"effort": "low"}
            assert adapter.model._pengine_prompt_cache_warmup is False
            assert adapter.model._pengine_stream_continuation is False
        else:
            assert payload["temperature"] == 0
            assert payload["extra_body"]["reasoning"] == {"enabled": False}


@pytest.mark.parametrize("role", ["generation", "review"])
@pytest.mark.parametrize("typed", [False, True])
def test_kimi_receives_and_replays_complete_tool_reasoning(role: str, typed: bool) -> None:
    model = build_relay_adapter(_settings(), role=role).model
    assistant = {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {"id": "call_1", "type": "function", "function": {"name": "probe", "arguments": "{}"}}
        ],
        **REASONING,
    }
    response = {
        "id": "completion_1",
        "object": "chat.completion",
        "created": 1,
        "model": KIMI,
        "choices": [{"index": 0, "finish_reason": "tool_calls", "message": assistant}],
    }
    result = model._create_chat_result(ChatCompletion(**response) if typed else response)
    message = result.generations[0].message
    for key, value in REASONING.items():
        assert message.additional_kwargs[key] == value
    payload = model._get_request_payload(
        [HumanMessage(content="probe"), message, ToolMessage(content="ok", tool_call_id="call_1")]
    )
    replay = payload["messages"][1]
    for key, value in REASONING.items():
        assert replay[key] == value
    assert replay["tool_calls"] == assistant["tool_calls"]
    assert payload["messages"][2]["tool_call_id"] == "call_1"


def test_kimi_stream_merges_reasoning_and_tool_arguments_for_replay() -> None:
    model = build_relay_adapter(_settings(), role="generation").model
    chunks = []
    for index, text in enumerate(("先验证", "工具结果。")):
        delta = {
            "role": "assistant",
            "content": "",
            "reasoning": text,
            "reasoning_details": [
                {"type": "reasoning.text", "text": text, "index": 0, "format": "unknown"}
            ],
            "tool_calls": [
                {
                    "index": 0,
                    "id": "call_1" if index == 0 else None,
                    "type": "function",
                    "function": {
                        "name": "probe" if index == 0 else None,
                        "arguments": "{" if index == 0 else "}",
                    },
                }
            ],
        }
        chunk = model._convert_chunk_to_generation_chunk(
            {"model": KIMI, "choices": [{"delta": delta, "finish_reason": None}]},
            AIMessageChunk,
            None,
        )
        assert _stream_chunk_text_chars(chunk) > 0
        chunks.append(chunk)
    combined = chunks[0] + chunks[1]
    payload = model._get_request_payload([combined.message])
    assert payload["messages"][0]["reasoning"] == "先验证工具结果。"
    assert payload["messages"][0]["reasoning_details"][0]["text"] == "先验证工具结果。"
    assert payload["messages"][0]["tool_calls"][0]["function"]["arguments"] == "{}"


def test_kimi_preserves_serial_tools_and_per_call_output_budget() -> None:
    state = ModelCallState()
    state.context.requested_output_tokens = 4096
    model = build_relay_adapter(_settings(), role="generation", model_call_state=state).model
    bound = model.bind_tools(
        [
            {
                "type": "function",
                "function": {
                    "name": "probe",
                    "description": "Probe",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ],
        tool_choice="required",
    )
    assert bound.kwargs["parallel_tool_calls"] is False
    assert bound.kwargs["tool_choice"] == "required"
    assert model._with_call_output_budget({})["max_tokens"] == 4096


def test_kimi_reasoning_history_counts_against_preflight() -> None:
    plain = AIMessage(content="ok")
    reasoning = AIMessage(content="ok", additional_kwargs=REASONING)
    assert estimate_messages_tokens([reasoning]) > estimate_messages_tokens([plain])


@pytest.mark.parametrize("response_model", [KIMI, "kimi-k3", DEEPSEEK])
def test_kimi_identity_requires_exact_configured_slug(response_model: str) -> None:
    handler = _ModelCallAuditHandler(role="review", model_id=KIMI)
    response = LLMResult(
        generations=[
            [
                ChatGeneration(
                    message=AIMessage(content="ok", response_metadata={"model": response_model})
                )
            ]
        ]
    )
    if response_model == KIMI:
        handler.on_llm_end(response, run_id=uuid4())
    else:
        with pytest.raises(RelayIdentityError):
            handler.on_llm_end(response, run_id=uuid4())


@pytest.mark.asyncio
async def test_kimi_does_not_replay_incomplete_reasoning_after_stream_interruption(
    monkeypatch,
) -> None:
    from langchain_openai import ChatOpenAI

    from pengine.relay import RelayStreamStalledError

    calls = 0

    async def broken_stream(*args, **kwargs):
        nonlocal calls
        calls += 1
        yield model._convert_chunk_to_generation_chunk(
            {
                "choices": [
                    {
                        "delta": {"role": "assistant", "reasoning": "unfinished"},
                        "finish_reason": None,
                    }
                ]
            },
            AIMessageChunk,
            None,
        )
        raise RelayStreamStalledError(reason="stall", detail="probe interruption")

    monkeypatch.setattr(ChatOpenAI, "_astream", broken_stream)
    model = build_relay_adapter(_settings(stream_max_retries=2), role="generation").model
    with pytest.raises(RelayStreamStalledError):
        async for _ in model._astream([HumanMessage(content="probe")]):
            pass
    assert calls == 1
