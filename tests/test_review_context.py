from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime

import pytest

from pengine.review_context import ReviewContextError, compile_review_context
from pengine.schemas import EpisodeDraft
from pengine.series_review import BoundStructuralReview, active_prefix_hash


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _draft(episode_number: int, content: str | None = None) -> EpisodeDraft:
    screenplay = content or f"第{episode_number}集完整剧本\n人物完成行动{episode_number}"
    return EpisodeDraft(
        episode_number=episode_number,
        content=screenplay,
        content_sha256=_sha256(screenplay),
        completed_at=datetime.now(UTC),
    )


def _receipt(
    drafts: list[EpisodeDraft],
    *,
    status: str = "active",
    passed: bool = True,
    prefix_hash: str | None = None,
) -> BoundStructuralReview:
    episode_number = len(drafts)
    return BoundStructuralReview(
        review_id="series_review_boundary",
        run_id="run-1",
        review_epoch=1,
        review_type="milestone",
        episode_number=episode_number,
        design_candidate_id="candidate_1",
        design_content_hash="a" * 64,
        design_epoch=1,
        batch_id="batch_1",
        batch_epoch=1,
        prefix_hash=prefix_hash
        or active_prefix_hash(
            [
                {
                    "episode_number": draft.episode_number,
                    "content_sha256": draft.content_sha256,
                }
                for draft in drafts
            ]
        ),
        call_id="review-call-1",
        passed=passed,
        category="pass" if passed else "script_defect",
        evidence="L4硬规则：已核对。\nL4价值观：已核对。\nL4创作建议：已核对。",
        earliest_affected_episode=None if passed else episode_number,
        status=status,
        reviewed_at=datetime.now(UTC),
    )


def _compile(
    *,
    drafts: list[EpisodeDraft] | None = None,
    context_limit_tokens: int = 100_000,
    receipt: BoundStructuralReview | None = None,
    series_bible_components: dict[str, str] | None = None,
):
    committed = drafts or [_draft(number) for number in range(1, 5)]
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
    contract_hash = _sha256(canonical_contract)
    return compile_review_context(
        review_type="milestone",
        episode_number=4,
        context_limit_tokens=context_limit_tokens,
        maximum_output_tokens=64_000,
        series_bible_components=series_bible_components
        or {
            "story_outline": "完整故事大纲",
            "character_biographies": "完整人物小传",
            "relationship_logic": "完整关系逻辑",
        },
        design_content_hash="a" * 64,
        design_epoch=1,
        story_contract_json=contract_json,
        story_contract_sha256=contract_hash,
        committed_prefix=committed,
        series_state_json=json.dumps(
            {
                "contract_sha256": contract_hash,
                "locked_through_episode": 4,
                "established_fact_ids": ["fact_1"],
                "repair_constraints": [],
            }
        ),
        current_episode_plan="第4集计划",
        previous_receipt=receipt,
    )


def test_review_context_prefers_complete_prefix_and_manifest_excludes_content() -> None:
    compiled = _compile()

    assert compiled.mode == "full_prefix"
    assert 'name="active_screenplay_prefix"' in compiled.model_input
    assert all(f"第{number}集完整剧本" in compiled.model_input for number in range(1, 5))
    assert "第1集完整剧本" not in compiled.manifest_json
    assert compiled.manifest["mode"] == "full_prefix"
    assert compiled.manifest["episode_number"] == 4
    assert compiled.bundle_sha256 == _sha256(compiled.model_input)


def test_review_context_uses_exact_passing_receipt_when_complete_prefix_is_too_large() -> None:
    drafts = [
        _draft(1, "甲" * 10_000),
        _draft(2, "乙" * 10_000),
        _draft(3),
        _draft(4),
    ]

    compiled = _compile(
        drafts=drafts,
        context_limit_tokens=12_000,
        receipt=_receipt(drafts[:2]),
    )

    assert compiled.mode == "milestone_receipt"
    assert 'name="prior_milestone_receipt"' in compiled.model_input
    assert 'name="current_review_window"' in compiled.model_input
    assert "甲" * 100 not in compiled.model_input
    assert "第3集完整剧本" in compiled.model_input
    assert "第4集完整剧本" in compiled.model_input
    components = {item["name"]: item for item in compiled.manifest["components"]}
    assert components["prior_milestone_receipt"]["episode_end"] == 2
    assert components["current_review_window"]["episode_start"] == 3


@pytest.mark.parametrize(
    ("receipt_factory", "message"),
    [
        (lambda drafts: None, "No valid prior milestone receipt"),
        (lambda drafts: _receipt(drafts[:2], status="stale"), "not an active passing"),
        (
            lambda drafts: _receipt(drafts[:2], prefix_hash="f" * 64),
            "prefix hash mismatch",
        ),
        (
            lambda drafts: _receipt(drafts[:2]).model_copy(
                update={"design_content_hash": "c" * 64}
            ),
            "not an active passing",
        ),
    ],
)
def test_review_context_fails_closed_for_missing_or_invalid_receipt(
    receipt_factory,
    message: str,
) -> None:
    drafts = [
        _draft(1, "甲" * 10_000),
        _draft(2, "乙" * 10_000),
        _draft(3),
        _draft(4),
    ]

    with pytest.raises(ReviewContextError, match=message):
        _compile(
            drafts=drafts,
            context_limit_tokens=12_000,
            receipt=receipt_factory(drafts),
        )


def test_review_context_rejects_unknown_series_bible_component() -> None:
    with pytest.raises(ReviewContextError, match="unknown projections"):
        _compile(
            series_bible_components={
                "story_outline": "故事",
                "character_biographies": "人物",
                "relationship_logic": "关系",
                "supervisor_history": "与剧本审查无关的内部消息",
            }
        )


def test_review_context_rejects_tampered_active_screenplay_before_dispatch() -> None:
    drafts = [_draft(number) for number in range(1, 5)]
    drafts[2] = drafts[2].model_copy(update={"content": "被篡改"})

    with pytest.raises(ReviewContextError, match="content hash mismatch"):
        _compile(drafts=drafts)
