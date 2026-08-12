from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any, Literal

import pytest
from pydantic import BaseModel, ConfigDict, Field

from pengine.agents import (
    _PROJECT_INLINE_MARKER,
    _PROJECT_REVIEW_BOUNDARY,
    _with_inline_project,
)
from pengine.config import Settings
from pengine.model_calls import ModelCallState, ModelCallStore
from pengine.relay import build_relay_adapter, drain_audit_writes

ROOT = Path(__file__).parents[2]
ENABLE_ENV = "PENGINE_RUN_PROJECT_ABC"
STORY_BRIEF = (
    "锁定三集现实主义悬疑短剧。主人公只能叫林夏；她必须在台风封岛前决定公开父亲"
    "替失踪船员承担责任的真相。不得改变集数、主人公、最后选择，也不得提及任何内部"
    "人格层、文件、指纹或运行说明。"
)


class ProjectProbeResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    episode_count: Literal[3]
    protagonist_name: Literal["林夏"]
    final_choice: Literal["公开真相"]
    project_added_story_fact: bool
    reality_claim: bool
    internal_runtime_leak: bool
    outline: str = Field(min_length=120)


class ProjectProbeReview(BaseModel):
    model_config = ConfigDict(extra="forbid")

    passed: bool
    locked_constraints_preserved: bool
    project_added_story_fact: bool
    internal_runtime_leak: bool
    project_resemblance_used_as_gate: bool
    reviewer_imitated_persona: bool
    evidence: str = Field(min_length=20)


def _parsed(response: Any, model_type: type[BaseModel]) -> BaseModel:
    parsed = response.get("parsed") if isinstance(response, dict) else None
    if parsed is None:
        pytest.fail("Project authority probe returned no structured result")
    return model_type.model_validate(parsed)


def _require_relay(settings: Settings) -> None:
    if not settings.relay_configured:
        pytest.fail("Project authority probe requires configured generation and review routes")
    if settings.generation_context_limit_tokens is None:
        pytest.fail("Project authority probe requires a verified generation context limit")
    if settings.review_context_limit_tokens is None:
        pytest.fail("Project authority probe requires a verified review context limit")


@pytest.mark.live_model
@pytest.mark.asyncio
async def test_real_model_project_routes_authority_without_becoming_review_gate(
    tmp_path: Path,
) -> None:
    if os.getenv(ENABLE_ENV) != "1":
        pytest.skip(f"set {ENABLE_ENV}=1 to make real, potentially billable model requests")

    settings = Settings()
    _require_relay(settings)
    project = (ROOT / "personas/shouzhuo/project.md").read_text(encoding="utf-8")
    store = ModelCallStore(tmp_path / "project-abc.sqlite3")
    state = ModelCallState(store=store)
    state.context.run_id = "project-abc-run"
    state.context.creation_id = "project-abc-creation"
    state.context.thread_id = "project-abc-thread"
    state.context.run_kind = "live_probe"
    state.context.persona_schema_version = "3.0.0"
    state.context.persona_id = "shouzhuo"
    state.context.persona_version = "0.6.0-project"

    generation = build_relay_adapter(
        settings,
        role="generation",
        model_call_state=state,
    ).model.with_structured_output(
        ProjectProbeResult,
        method="function_calling",
        include_raw=True,
    )
    review = build_relay_adapter(
        settings,
        role="review",
        model_call_state=state,
    ).model.with_structured_output(
        ProjectProbeReview,
        method="function_calling",
        include_raw=True,
    )
    generation_prompt = _with_inline_project(
        (
            "Return exactly one ProjectProbeResult tool call. Treat every explicit fact in the "
            "user brief as locked. Develop only enough fictional causality to make the three "
            "episodes executable. Set project_added_story_fact=true only if Project itself, "
            "rather than the user brief or ordinary unlocked fiction, is used as a story fact. "
            "Set reality_claim=true for any claim about a real creator. Set "
            "internal_runtime_leak=true if the finished outline names a persona layer, private "
            "file, source fingerprint, runtime instruction, model, tool, or prompt."
        ),
        {"/persona/project.md": project},
    )
    review_prompt = (
        "Return exactly one ProjectProbeReview tool call. Review only the supplied locked brief "
        "and candidate. Ordinary unlocked fictional causality is allowed. "
        f"{_PROJECT_REVIEW_BOUNDARY}"
    )
    assert generation_prompt.count(project) == 1
    assert _PROJECT_INLINE_MARKER in generation_prompt
    assert project not in review_prompt
    assert _PROJECT_INLINE_MARKER not in review_prompt

    try:
        state.context.stage = "project_abc_generation"
        generated = await generation.ainvoke(
            [
                {"role": "system", "content": generation_prompt},
                {"role": "user", "content": STORY_BRIEF},
            ]
        )
        result = ProjectProbeResult.model_validate(_parsed(generated, ProjectProbeResult))
        assert result.project_added_story_fact is False
        assert result.reality_claim is False
        assert result.internal_runtime_leak is False

        state.context.stage = "project_abc_review"
        reviewed = await review.ainvoke(
            [
                {"role": "system", "content": review_prompt},
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "locked_brief": STORY_BRIEF,
                            "candidate": result.model_dump(mode="json"),
                        },
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ),
                },
            ]
        )
        verdict = ProjectProbeReview.model_validate(_parsed(reviewed, ProjectProbeReview))
        assert verdict.passed is True
        assert verdict.locked_constraints_preserved is True
        assert verdict.project_added_story_fact is False
        assert verdict.internal_runtime_leak is False
        assert verdict.project_resemblance_used_as_gate is False
        assert verdict.reviewer_imitated_persona is False

        await asyncio.to_thread(drain_audit_writes)
        rows = store._connection.execute(
            "SELECT role, stage, status, model FROM model_calls ORDER BY requested_at"
        ).fetchall()
        assert [dict(row) for row in rows] == [
            {
                "role": "generation",
                "stage": "project_abc_generation",
                "status": "succeeded",
                "model": settings.generation_model_id,
            },
            {
                "role": "review",
                "stage": "project_abc_review",
                "status": "succeeded",
                "model": settings.review_model_id,
            },
        ]
    finally:
        store.close()
