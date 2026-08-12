from __future__ import annotations

import asyncio
import json
import os
import shutil
from pathlib import Path
from typing import Any, Literal

import pytest
from pydantic import BaseModel, ConfigDict, Field

from pengine.config import Settings
from pengine.model_calls import ModelCallState, ModelCallStore
from pengine.relay import build_relay_adapter, drain_audit_writes

ROOT = Path(__file__).parents[2]
ENABLE_ENV = "PENGINE_RUN_L4_ABC"


class L4CaseVerdict(BaseModel):
    model_config = ConfigDict(extra="forbid")

    case: Literal["A_hard_conflict", "B_advice_deviation", "C_user_parameter_override"]
    passed: bool
    blocking_rule_owner: Literal["守拙", "none"]
    parameter_owner: Literal["Pengine", "not_applicable"]
    user_override_respected: bool
    product_parameter_attributed_to_creator: bool
    evidence: str = Field(min_length=20)


class L4ProbeReview(BaseModel):
    model_config = ConfigDict(extra="forbid")

    verdicts: list[L4CaseVerdict] = Field(min_length=3, max_length=3)


def _parsed(response: Any) -> L4ProbeReview:
    parsed = response.get("parsed") if isinstance(response, dict) else None
    if parsed is None:
        pytest.fail("L4 A/B/C probe returned no structured result")
    return L4ProbeReview.model_validate(parsed)


def _require_relay(settings: Settings) -> None:
    if not settings.relay_configured:
        pytest.fail("L4 A/B/C requires configured generation and review routes")
    if settings.review_context_limit_tokens is None:
        pytest.fail("L4 A/B/C requires a verified review context limit")


@pytest.mark.live_model
@pytest.mark.asyncio
async def test_real_model_l4_abc_separates_hard_advice_and_product_authority(
    tmp_path: Path,
) -> None:
    if os.getenv(ENABLE_ENV) != "1":
        pytest.skip(f"set {ENABLE_ENV}=1 to make a real, potentially billable model request")

    settings = Settings()
    _require_relay(settings)

    isolated_persona = tmp_path / "personas" / "shouzhuo"
    shutil.copytree(ROOT / "personas" / "shouzhuo", isolated_persona)
    l4 = (isolated_persona / "l4.md").read_text(encoding="utf-8")
    store = ModelCallStore(tmp_path / "l4-abc.sqlite3")
    state = ModelCallState(store=store)
    state.context.run_id = "l4-abc-run"
    state.context.creation_id = "l4-abc-creation"
    state.context.thread_id = "l4-abc-thread"
    state.context.run_kind = "live_probe"
    state.context.stage = "l4_abc_review"
    state.context.persona_schema_version = "3.0.0"
    state.context.persona_id = "shouzhuo"
    state.context.persona_version = "0.5.0-l4"

    review_model = build_relay_adapter(
        settings,
        role="review",
        model_call_state=state,
    ).model.with_structured_output(
        L4ProbeReview,
        method="function_calling",
        include_raw=True,
    )
    try:
        response = await review_model.ainvoke(
            [
                {
                    "role": "system",
                    "content": (
                        "Return exactly one L4ProbeReview tool call with exactly one verdict for "
                        "each named case. Review only the supplied compiled L4. Only clauses under "
                        "硬规则 may block. Clauses under 已确认创作建议 are advisory and cannot "
                        "block. Episode count, duration, and scene-count defaults are owned by "
                        "Pengine, never by 守拙; an explicit user production parameter overrides "
                        "the Pengine default. Case A must fail only if it directly contradicts a "
                        "守拙 hard rule. Case B must pass if it differs only from advice. Case C "
                        "must pass when the candidate follows the explicit three-episode user "
                        "requirement instead of the six-episode Pengine default. Set "
                        "product_parameter_attributed_to_creator=true only if the candidate or "
                        "your own judgment wrongly attributes that product default to 守拙."
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "compiled_l4": l4,
                            "cases": [
                                {
                                    "case": "A_hard_conflict",
                                    "user_requirement": "创作三集现实主义短剧。",
                                    "candidate": (
                                        "人物昨天仍是仇敌，今天没有发生任何事件或行动，"
                                        "只用一句旁白宣布两人已经成为生死盟友。"
                                    ),
                                },
                                {
                                    "case": "B_advice_deviation",
                                    "user_requirement": "创作三集现实主义短剧。",
                                    "candidate": (
                                        "人物变化均由可见事件和行动推动，三集因果完整；"
                                        "表达直接，没有安排华彩或浪漫时刻。"
                                    ),
                                },
                                {
                                    "case": "C_user_parameter_override",
                                    "user_requirement": (
                                        "明确创作三集，每集约三分钟、四场；该生产参数已锁定。"
                                    ),
                                    "candidate": "完整交付三集，每集约三分钟、四场。",
                                },
                            ],
                        },
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ),
                },
            ]
        )
        verdicts = {verdict.case: verdict for verdict in _parsed(response).verdicts}

        hard = verdicts["A_hard_conflict"]
        assert hard.passed is False
        assert hard.blocking_rule_owner == "守拙"

        advice = verdicts["B_advice_deviation"]
        assert advice.passed is True
        assert advice.blocking_rule_owner == "none"

        parameter = verdicts["C_user_parameter_override"]
        assert parameter.passed is True
        assert parameter.parameter_owner == "Pengine"
        assert parameter.user_override_respected is True
        assert parameter.product_parameter_attributed_to_creator is False

        await asyncio.to_thread(drain_audit_writes)
        rows = store._connection.execute(
            "SELECT role, stage, status, model FROM model_calls ORDER BY requested_at"
        ).fetchall()
        assert len(rows) == 1
        assert dict(rows[0]) == {
            "role": "review",
            "stage": "l4_abc_review",
            "status": "succeeded",
            "model": settings.review_model_id,
        }
    finally:
        store.close()
