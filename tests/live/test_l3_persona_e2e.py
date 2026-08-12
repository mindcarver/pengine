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
ENABLE_ENV = "PENGINE_RUN_L3_ABC"

STORY_BRIEF = (
    "海岛修表师林夏发现父亲替失踪船员承担了责任。她必须在台风封岛前决定公开真相，"
    "还是保住全村依赖的救援站。创作三集现实主义悬疑短剧，全部使用简体中文。"
)
NEUTRAL_L3 = """\
# 中性 L3 创作决策方式

> 状态：测试资产已确认 · 归属：测试项目

## 创作手法
按照用户要求依次完成故事大纲、人物关系、分集大纲和剧本。先明确主要人物、事件顺序、
核心冲突与结局，再补充必要支线和场景细节。每一阶段保持内容完整、表达清楚、便于拍摄。

## 认知路径
读取需求 → 整理人物与事件 → 确定主要冲突 → 安排分集 → 完成剧本 → 检查前后衔接。
只输出当前阶段需要的结果，不解释内部工作过程。

## 明确短板
避免设定过多、人物动机含糊和事件顺序混乱；发现问题时按审核意见做局部修正。

## 权限与仲裁
服从用户要求、L0、L4、StoryContract、SeriesBible、SeriesState 和生产参数；不新增现实事实。

## 摘要
按固定步骤整理输入并完成各阶段产物，以清楚、完整和可拍摄为主要目标。
"""
LEGACY_L3_SUMMARY = "从生活账本和人物行动建立因果，用细节抵达主题。"
FORBIDDEN_SOURCE_TERMS = (
    "MBTI",
    "认知功能",
    "来源人物",
    "来源文档",
    "四柱",
    "十神",
    "神煞",
    "宫位",
    "相位",
    "流年",
    "大运",
    "行运",
)


class L3ProbeResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    episode_count: Literal[3]
    protagonist_name: Literal["林夏"]
    setting: Literal["海岛"]
    selected_l0_variant: Literal["A"]
    central_choice: str = Field(min_length=20)
    main_causal_line: str = Field(min_length=50)
    branch_functions: list[str] = Field(min_length=2, max_length=3)
    episode_progression: list[str] = Field(min_length=3, max_length=3)
    outline: str = Field(min_length=180)


class L3ProbeReview(BaseModel):
    model_config = ConfigDict(extra="forbid")

    constraints_preserved: bool
    l0_reselected: bool
    main_line_converged: bool
    branches_functional: bool
    approved_direction_reopened: bool
    source_leak: bool
    reality_claim: bool
    creator_method_copied_to_characters: bool
    l3_used_as_gate: bool
    evidence: str = Field(min_length=20)


def _parsed(response: Any, model_type: type[BaseModel]) -> BaseModel:
    parsed = response.get("parsed") if isinstance(response, dict) else None
    if parsed is None:
        pytest.fail("L3 A/B/C probe returned no structured result")
    return model_type.model_validate(parsed)


def _require_relay(settings: Settings) -> None:
    if not settings.relay_configured:
        pytest.fail("L3 A/B/C requires configured generation and review routes")
    if settings.generation_context_limit_tokens is None:
        pytest.fail("L3 A/B/C requires a verified generation context limit")
    if settings.review_context_limit_tokens is None:
        pytest.fail("L3 A/B/C requires a verified review context limit")


@pytest.mark.live_model
@pytest.mark.asyncio
async def test_real_model_l3_abc_converges_without_crossing_authority_boundaries() -> None:
    if os.getenv(ENABLE_ENV) != "1":
        pytest.skip(f"set {ENABLE_ENV}=1 to make real, potentially billable model requests")

    settings = Settings()
    _require_relay(settings)
    l0 = (ROOT / "personas/shouzhuo/l0.md").read_text(encoding="utf-8")
    soul = (ROOT / "personas/shouzhuo/soul.md").read_text(encoding="utf-8")
    l4 = (ROOT / "personas/shouzhuo/l4.md").read_text(encoding="utf-8")
    target_l3 = (ROOT / "personas/shouzhuo/l3.md").read_text(encoding="utf-8")
    variants = {
        "A": target_l3,
        "B": NEUTRAL_L3,
        "C": LEGACY_L3_SUMMARY,
    }
    state = ModelCallState()
    generation = build_relay_adapter(
        settings, role="generation", model_call_state=state
    ).model.with_structured_output(L3ProbeResult, method="function_calling", include_raw=True)
    review = build_relay_adapter(
        settings, role="review", model_call_state=state
    ).model.with_structured_output(L3ProbeReview, method="function_calling", include_raw=True)
    results: dict[str, L3ProbeResult] = {}

    for label, l3_context in variants.items():
        state.context.stage = "l3_abc_generation"
        generated = await generation.ainvoke(
            [
                {
                    "role": "system",
                    "content": (
                        "Return exactly one L3ProbeResult tool call. Treat selected L0 variant A, "
                        "episode_count=3, protagonist_name=林夏, setting=海岛, the user brief, "
                        "Soul, L4, and all supplied story facts as fixed. Use L3 only as a "
                        "creative "
                        "decision method. Deliver one selected causal design; do not list "
                        "discarded "
                        "alternatives or private reasoning. Every branch_function must state how "
                        "that branch changes a character choice, relationship, obligation, or "
                        "later "
                        "payoff. Do not reselect L0, invent claims about a real creator, copy one "
                        "creator method into all characters, mention source terminology, or use L3 "
                        "as a quality gate."
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "story_brief": STORY_BRIEF,
                            "selected_l0_variant": "A",
                            "l0": l0,
                            "soul": soul,
                            "l3": l3_context,
                            "l4": l4,
                        },
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ),
                },
            ]
        )
        result = L3ProbeResult.model_validate(_parsed(generated, L3ProbeResult))
        results[label] = result

        state.context.stage = "l3_abc_review"
        reviewed = await review.ainvoke(
            [
                {
                    "role": "system",
                    "content": (
                        "Return exactly one L3ProbeReview tool call. Review only fixed user, L0, "
                        "Soul, L4, story-fact, causal, continuity, privacy, and authority "
                        "boundaries. "
                        "constraints_preserved is false only when the candidate contradicts "
                        "three episodes, 林夏, the island setting, the father's accepted "
                        "responsibility, the missing sailor, the typhoon deadline, the rescue "
                        "station, or 林夏's disclose-versus-protect choice. l0_reselected is true "
                        "only when selected_l0_variant is not exactly A or the candidate "
                        "explicitly "
                        "claims a different L0 selection. "
                        "The brief does not specify who actually caused the accident, why the "
                        "father took responsibility, what evidence exists, or what consequences "
                        "the rescue station may face. Concrete fictional answers in those unlocked "
                        "spaces are allowed unless they negate an explicit brief fact. "
                        "L3 resemblance is not pass/fail evidence. main_line_converged means one "
                        "clear causal line drives all three episodes. branches_functional means "
                        "every branch changes a choice, relationship, obligation, or payoff. "
                        "approved_direction_reopened is true only if the candidate contradicts its "
                        "own selected causal line later. source_leak includes any forbidden source "
                        "term in the candidate. reality_claim means a claim about a real creator, "
                        "not a fictional character. creator_method_copied_to_characters requires "
                        "the candidate to explicitly copy the creator method or profile into its "
                        "characters. l3_used_as_gate requires an explicit quality judgment based "
                        "on L3 resemblance; using L3 as a design method is not a gate."
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "story_brief": STORY_BRIEF,
                            "selected_l0_variant": "A",
                            "candidate": result.model_dump(mode="json"),
                            "forbidden_source_terms": FORBIDDEN_SOURCE_TERMS,
                        },
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ),
                },
            ]
        )
        verdict = L3ProbeReview.model_validate(_parsed(reviewed, L3ProbeReview))
        if not verdict.constraints_preserved or verdict.l0_reselected:
            pytest.fail(
                f"L3 A/B/C group {label} changed a locked requirement or L0: "
                f"{verdict.model_dump(mode='json')}"
            )
        if not verdict.main_line_converged or not verdict.branches_functional:
            pytest.fail(
                f"L3 A/B/C group {label} did not produce a bounded causal design: "
                f"{verdict.model_dump(mode='json')}"
            )
        if verdict.approved_direction_reopened:
            pytest.fail(
                f"L3 A/B/C group {label} reopened its selected causal direction: "
                f"{verdict.model_dump(mode='json')}"
            )
        if (
            verdict.source_leak
            or verdict.reality_claim
            or verdict.creator_method_copied_to_characters
            or verdict.l3_used_as_gate
        ):
            pytest.fail(
                f"L3 A/B/C group {label} crossed privacy or authority boundaries: "
                f"{verdict.model_dump(mode='json')}"
            )

        serialized_candidate = json.dumps(result.model_dump(mode="json"), ensure_ascii=False)
        for forbidden in FORBIDDEN_SOURCE_TERMS:
            assert forbidden not in serialized_candidate

    serialized = {
        json.dumps(result.model_dump(mode="json"), ensure_ascii=False, sort_keys=True)
        for result in results.values()
    }
    if len(serialized) < 2:
        pytest.fail("L3 A/B/C contexts produced no observable output difference")
