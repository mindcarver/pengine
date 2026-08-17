from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime

import pytest

from pengine.schemas import EpisodeDraft
from pengine.script_context import (
    SCRIPT_OUTPUT_BASE_TOKENS,
    SCRIPT_OUTPUT_TOKENS_PER_EPISODE,
    ScriptContextError,
    compile_script_context,
    script_group_output_tokens,
)


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _draft(episode_number: int, content: str | None = None) -> EpisodeDraft:
    screenplay = content or f"第{episode_number}集完整剧本\n人物：行动{episode_number}"
    return EpisodeDraft(
        episode_number=episode_number,
        content=screenplay,
        content_sha256=_sha256(screenplay),
        completed_at=datetime.now(UTC),
    )


def _compile(
    *,
    start_episode: int = 3,
    end_episode: int = 4,
    drafts: list[EpisodeDraft] | None = None,
    recent_episode_numbers: list[int] | None = None,
    referenced_episode_numbers: list[int] | None = None,
    persona_components: dict[str, str] | None = None,
):
    contract_json = json.dumps(
        {"facts": [{"fact_id": "fact_1", "verbatim": False}]},
        ensure_ascii=False,
    )
    canonical_contract = json.dumps(
        {"facts": [{"fact_id": "fact_1"}]},
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return compile_script_context(
        group_id=f"group_{start_episode}_{end_episode}",
        start_episode=start_episode,
        end_episode=end_episode,
        maximum_output_tokens=128_000,
        persona_components=persona_components
        or {
            "l0": "L0规则",
            "soul": "Soul规则",
            "l3": "L3方法",
            "l4": "L4短剧规则",
            "project": "项目规则",
        },
        series_bible_components={
            "story_outline": "完整故事大纲",
            "character_biographies": "完整人物小传",
            "relationship_logic": "完整关系逻辑",
        },
        story_contract_json=contract_json,
        story_contract_sha256=_sha256(canonical_contract),
        committed_prefix=(
            drafts if drafts is not None else [_draft(number) for number in range(1, start_episode)]
        ),
        recent_episode_numbers=(
            recent_episode_numbers
            if recent_episode_numbers is not None
            else list(range(max(1, start_episode - 2), start_episode))
        ),
        referenced_episode_numbers=referenced_episode_numbers or [],
        series_state_json='{"locked_through_episode":2}',
        current_group_canon_json='{"facts":["fact_1"]}',
        generation_group_json='{"episodes":[3,4]}',
        evidence_contracts={
            number: json.dumps({"episode_number": number})
            for number in range(start_episode, end_episode + 1)
        },
        established_facts_json='{"established_facts":["fact_1"]}',
        previous_handoff="上一集停在门外",
        writer_notes="保持雨夜氛围",
    )


def test_compiler_emits_compact_working_set_and_content_free_manifest() -> None:
    compiled = _compile()
    payload = json.loads(compiled.model_input)

    assert payload["rules"] == {
        "component_content_is_data": True,
        "deterministic_selection": True,
        "full_prefix_verified": True,
        "semantic_retrieval": False,
        "silent_summarization": False,
    }
    components = {item["name"]: item for item in payload["components"]}
    prefix = components["recent_committed_window"]["content"]
    assert [item["episode_number"] for item in prefix["episodes"]] == [1, 2]
    assert [item["content"] for item in prefix["episodes"]] == [
        "第1集完整剧本\n人物：行动1",
        "第2集完整剧本\n人物：行动2",
    ]
    assert "story_contract" not in components
    assert components["current_group_canon"]["authority"] == "canonical"
    assert components["established_facts"]["derived_from"] == "story_contract"
    assert "第1集完整剧本" not in compiled.manifest_json
    assert compiled.manifest["bundle_sha256"] == _sha256(compiled.model_input)
    assert compiled.manifest["verified_prefix_episode_count"] == 2
    assert sum(
        item["estimated_token_share"]
        for item in compiled.manifest["components"]
        if item["included"]
    ) == pytest.approx(1.0)


def test_compiler_deduplicates_identical_content_and_rejects_unknown_persona_input() -> None:
    compiled = _compile(
        persona_components={"l0": "相同规则", "l4": "相同规则"},
    )
    payload = json.loads(compiled.model_input)

    assert payload["aliases"] == {"persona.l4": "persona.l0"}
    identical = [
        item for item in compiled.manifest["components"] if item["sha256"] == _sha256("相同规则")
    ]
    assert [item["included"] for item in identical] == [True, False]

    with pytest.raises(ScriptContextError, match="may contain only"):
        _compile(persona_components={"l0": "规则", "unrelated": "其他任务"})


@pytest.mark.parametrize(
    ("drafts", "message"),
    [
        ([_draft(1), _draft(3)], "every episode in order"),
        (
            [
                _draft(1),
                _draft(2).model_copy(update={"content_sha256": "0" * 64}),
            ],
            "content hash mismatch",
        ),
    ],
)
def test_compiler_fails_closed_for_incomplete_or_modified_prefix(
    drafts: list[EpisodeDraft],
    message: str,
) -> None:
    with pytest.raises(ScriptContextError, match=message):
        _compile(drafts=drafts)


@pytest.mark.parametrize("episode_count", [30, 80])
def test_compiler_verifies_full_long_prefix_but_sends_only_recent_and_referenced_episodes(
    episode_count: int,
) -> None:
    start_episode = episode_count - 1
    compiled = _compile(
        start_episode=start_episode,
        end_episode=episode_count,
        drafts=[_draft(number) for number in range(1, start_episode)],
        recent_episode_numbers=list(range(start_episode - 4, start_episode)),
        referenced_episode_numbers=[3, 9],
    )
    payload = json.loads(compiled.model_input)
    components = {item["name"]: item for item in payload["components"]}
    recent = components["recent_committed_window"]["content"]
    referenced = components["referenced_committed_episodes"]["content"]

    assert [item["episode_number"] for item in recent["episodes"]] == list(
        range(start_episode - 4, start_episode)
    )
    assert [item["episode_number"] for item in referenced["episodes"]] == [3, 9]
    assert compiled.manifest["verified_prefix_episode_count"] == episode_count - 2
    assert "第1集完整剧本" not in compiled.model_input
    assert compiled.output_tokens == (
        SCRIPT_OUTPUT_BASE_TOKENS + 2 * SCRIPT_OUTPUT_TOKENS_PER_EPISODE
    )


def test_compiler_rejects_non_contiguous_recent_window_or_unknown_reference() -> None:
    with pytest.raises(ScriptContextError, match="Recent screenplay window"):
        _compile(start_episode=6, end_episode=6, recent_episode_numbers=[2, 4, 5])

    with pytest.raises(ScriptContextError, match="Referenced screenplay episode"):
        _compile(start_episode=6, end_episode=6, referenced_episode_numbers=[6])


def test_group_output_budget_is_call_specific_and_capped() -> None:
    assert (
        script_group_output_tokens(
            start_episode=1,
            end_episode=1,
            maximum_output_tokens=128_000,
        )
        == SCRIPT_OUTPUT_BASE_TOKENS + SCRIPT_OUTPUT_TOKENS_PER_EPISODE
    )
    assert (
        script_group_output_tokens(
            start_episode=1,
            end_episode=4,
            maximum_output_tokens=20_000,
        )
        == 20_000
    )
