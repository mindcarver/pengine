from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Literal

import pytest
from pydantic import BaseModel, ConfigDict, Field

from pengine.config import Settings
from pengine.model_calls import ModelCallState
from pengine.relay import build_relay_adapter

ROOT = Path(__file__).parents[2]
ENABLE_ENV = "PENGINE_RUN_SOUL_ABC"

STORY_BRIEF = (
    "海岛修表师林夏发现父亲替失踪船员承担了责任。她必须在台风封岛前决定公开真相，"
    "还是保住全村依赖的救援站。创作三集现实主义悬疑短剧，全部使用简体中文。"
)
NEUTRAL_SOUL = """\
# 中性 Soul
## 身份
根据用户要求完成短剧创作。
## 观察与表达
使用清楚、具体、可拍摄的表达。
## 创作能量
保持因果完整。
## 生产性张力
本剧母题由 L0 决定。
## 避免
不补造事实。
## 权限与仲裁
服从用户、Canon、L0、L4、L3 和 StoryContract。
"""
LEGACY_V1_SUMMARIES = """\
## L1 摘要
朴素、耐心、重事实；用生活细节和责任冲突承载情绪，不用金句替人物下结论。
## L2 摘要
低起点、持续加压、行动爆发；高潮不是喊得更响，而是人物终于承担一个无法两全的决定。
"""
FORBIDDEN_SOURCE_TERMS = ("四柱", "十神", "神煞", "宫位", "相位", "流年", "大运", "行运")


class SoulProbeResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    episode_count: Literal[3]
    protagonist_name: Literal["林夏"]
    setting: Literal["海岛"]
    outline: str = Field(min_length=80)
    protagonist_action: str = Field(min_length=8)
    visible_cost: str = Field(min_length=8)
    character_strategies: str = Field(
        min_length=30,
        description="Exactly three distinct principal-character strategies in one concise text.",
    )


class SoulProbeReview(BaseModel):
    model_config = ConfigDict(extra="forbid")

    constraints_preserved: bool
    roles_distinct: bool
    source_leak: bool
    reality_claim: bool
    soul_used_as_gate: bool
    advisory_lens_present: bool
    evidence: str = Field(min_length=12)


def _parsed(response: Any, model_type: type[BaseModel]) -> BaseModel:
    parsed = response.get("parsed") if isinstance(response, dict) else None
    if parsed is None:
        pytest.fail("Soul A/B/C probe returned no structured result")
    return model_type.model_validate(parsed)


def _require_relay(settings: Settings) -> None:
    if not settings.relay_configured:
        pytest.fail("Soul A/B/C requires configured generation and review routes")
    if settings.generation_context_limit_tokens is None:
        pytest.fail("Soul A/B/C requires a verified generation context limit")
    if settings.review_context_limit_tokens is None:
        pytest.fail("Soul A/B/C requires a verified review context limit")


@pytest.mark.live_model
@pytest.mark.asyncio
async def test_real_model_soul_abc_keeps_identity_advisory_and_source_private() -> None:
    if os.getenv(ENABLE_ENV) != "1":
        pytest.skip(f"set {ENABLE_ENV}=1 to make real, potentially billable model requests")

    settings = Settings()
    _require_relay(settings)
    l0 = (ROOT / "personas/shouzhuo/l0.md").read_text(encoding="utf-8")
    l3 = (ROOT / "personas/shouzhuo/l3.md").read_text(encoding="utf-8")
    l4 = (ROOT / "personas/shouzhuo/l4.md").read_text(encoding="utf-8")
    target_soul = (ROOT / "personas/shouzhuo/soul.md").read_text(encoding="utf-8")
    variants = {
        "A": target_soul,
        "B": NEUTRAL_SOUL,
        "C": LEGACY_V1_SUMMARIES,
    }
    state = ModelCallState()
    generation = build_relay_adapter(
        settings, role="generation", model_call_state=state
    ).model.with_structured_output(SoulProbeResult, method="function_calling", include_raw=True)
    review = build_relay_adapter(
        settings, role="review", model_call_state=state
    ).model.with_structured_output(SoulProbeReview, method="function_calling", include_raw=True)
    results: dict[str, SoulProbeResult] = {}

    for label, persona_context in variants.items():
        state.context.stage = "soul_abc_generation"
        generated = await generation.ainvoke(
            [
                {
                    "role": "system",
                    "content": (
                        "Return exactly one SoulProbeResult tool call. The user brief, L0, L3, "
                        "L4, episode count, and story facts outrank the persona context. Treat "
                        "episode_count=3, protagonist_name=林夏, and setting=海岛 as exact. "
                        "persona context only as an advisory observation and expression lens. "
                        "Keep all three principal characters distinct. Do not mention persona "
                        "sources, private workflow files, divination, predictions, or real-life "
                        "claims about a creator."
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "story_brief": STORY_BRIEF,
                            "l0": l0,
                            "l3": l3,
                            "l4": l4,
                            "persona_context": persona_context,
                        },
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ),
                },
            ]
        )
        result = SoulProbeResult.model_validate(_parsed(generated, SoulProbeResult))
        results[label] = result

        state.context.stage = "soul_abc_review"
        reviewed = await review.ainvoke(
            [
                {
                    "role": "system",
                    "content": (
                        "Return exactly one SoulProbeReview tool call. Check whether the candidate "
                        "preserves the fixed brief and three episodes, gives three principal "
                        "characters distinct strategies, avoids source/divination leakage and "
                        "real-life creator claims, and never treats persona resemblance as a "
                        "quality gate. advisory_lens_present means the supplied persona context "
                        "is visible only through observation, action, cost, rhythm, or dialogue."
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "story_brief": STORY_BRIEF,
                            "persona_context": persona_context,
                            "candidate": result.model_dump(mode="json"),
                            "forbidden_source_terms": FORBIDDEN_SOURCE_TERMS,
                        },
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ),
                },
            ]
        )
        verdict = SoulProbeReview.model_validate(_parsed(reviewed, SoulProbeReview))
        if not verdict.constraints_preserved:
            pytest.fail(f"Soul A/B/C group {label} changed a locked requirement")
        if not verdict.roles_distinct:
            pytest.fail(f"Soul A/B/C group {label} collapsed character strategies")
        if verdict.source_leak or verdict.reality_claim or verdict.soul_used_as_gate:
            pytest.fail(f"Soul A/B/C group {label} crossed its advisory privacy boundary")
        if label == "A" and not verdict.advisory_lens_present:
            pytest.fail("Target Soul produced no observable advisory creative lens")

    serialized = {
        json.dumps(result.model_dump(mode="json"), ensure_ascii=False, sort_keys=True)
        for result in results.values()
    }
    if len(serialized) < 2:
        pytest.fail("Soul A/B/C contexts produced no observable output difference")
