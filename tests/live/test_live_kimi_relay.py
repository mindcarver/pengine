"""Opt-in K3 protocol probe using synthetic inputs and no application database."""

import json
import os
from pathlib import Path

import pytest
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langchain_core.messages import HumanMessage, ToolMessage
from pydantic import BaseModel

from pengine.config import KIMI_MODEL_ID, Settings
from pengine.relay import build_relay_routes


@pytest.mark.live_model
@pytest.mark.asyncio
@pytest.mark.skipif(
    os.getenv("PENGINE_RUN_KIMI_PROBE") != "1",
    reason="Opt-in probe makes billable Kimi K3 requests",
)
async def test_real_kimi_stream_tool_replay_and_review(tmp_path: Path) -> None:
    settings = Settings(
        _env_file=os.getenv("PENGINE_KIMI_ENV_FILE", ".env"),
        generation_model_id=KIMI_MODEL_ID,
        review_model_id=KIMI_MODEL_ID,
        outline_model_id=KIMI_MODEL_ID,
        generation_max_output_tokens=4096,
        review_max_output_tokens=4096,
        generation_context_limit_tokens=1_048_576,
        review_context_limit_tokens=1_048_576,
        openrouter_provider="",
        langfuse_enabled=False,
    )
    routes = build_relay_routes(settings)
    tool = {
        "type": "function",
        "function": {
            "name": "add",
            "description": "Add two integers.",
            "parameters": {
                "type": "object",
                "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
                "required": ["a", "b"],
                "additionalProperties": False,
            },
        },
    }
    prompt = HumanMessage(content="Use add to calculate 17 + 25. After the tool result, answer 42.")
    first = await routes.generation.model.bind_tools([tool], tool_choice="required").ainvoke(
        [prompt]
    )
    assert len(first.tool_calls) == 1
    call = first.tool_calls[0]
    assert call["name"] == "add"
    assert call["args"] == {"a": 17, "b": 25}
    reasoning = {
        key: first.additional_kwargs[key]
        for key in ("reasoning_content", "reasoning", "reasoning_details")
        if first.additional_kwargs.get(key)
    }
    assert reasoning, "K3 tool response must preserve reasoning for replay"
    final = await routes.generation.model.bind_tools([tool], tool_choice="none").ainvoke(
        [prompt, first, ToolMessage(content="42", tool_call_id=call["id"])]
    )
    assert "42" in str(final.content)
    assert not final.tool_calls

    class ReviewResult(BaseModel):
        correct: bool
        result: int

    review = await routes.review.model.bind_tools([ReviewResult], tool_choice="required").ainvoke(
        [HumanMessage(content="Check the calculation 17 + 25 = 42. Return ReviewResult.")]
    )
    assert len(review.tool_calls) == 1
    assert review.tool_calls[0]["name"] == "ReviewResult"
    assert review.tool_calls[0]["args"] == {"correct": True, "result": 42}

    def add(a: int, b: int) -> int:
        """Add two integers."""
        return a + b

    agent = create_agent(
        model=routes.generation.model,
        tools=[add],
        response_format=ToolStrategy(ReviewResult),
    )
    agent_result = await agent.ainvoke(
        {"messages": [HumanMessage(content="Call add for 17 + 25, then return ReviewResult.")]},
        {"recursion_limit": 8},
    )
    assert agent_result["structured_response"] == ReviewResult(correct=True, result=42)
    assert any(
        isinstance(message, ToolMessage) and message.name == "add"
        for message in agent_result["messages"]
    )
    evidence = {
        "model": KIMI_MODEL_ID,
        "streaming_generation": routes.generation.model.streaming,
        "nonstreaming_review": not routes.review.model.streaming,
        "tool_arguments": call["args"],
        "reasoning_fields": sorted(reasoning),
        "tool_replay_result": "42",
        "structured_review": review.tool_calls[0]["args"],
        "agent_tool_strategy": agent_result["structured_response"].model_dump(),
        "response_models": [
            message.response_metadata.get("model_name") for message in (first, final, review)
        ],
        "usage": [message.usage_metadata for message in (first, final, review)],
    }
    assert evidence["response_models"] == [KIMI_MODEL_ID] * 3
    destination = Path(os.getenv("PENGINE_KIMI_EVIDENCE", str(tmp_path / "kimi-probe.json")))
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(evidence, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(evidence, ensure_ascii=False))
