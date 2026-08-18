"""Deterministic context assembly for structural milestone reviews.

The compiler prefers the complete active screenplay prefix. When that cannot fit the
verified review route, it may replace only an already-approved historical prefix with
one immutable bound review receipt while retaining every screenplay after that boundary.
It never accepts an arbitrary workspace tree or a model-written summary.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from pengine.model_calls import estimate_text_tokens
from pengine.schemas import EpisodeDraft
from pengine.series_review import BoundStructuralReview, active_prefix_hash

REVIEW_CONTEXT_SCHEMA_VERSION = 1
REVIEW_OUTPUT_TOKENS = 4_096
REVIEW_CONTEXT_FIXED_RESERVE_TOKENS = 2_048

ReviewContextAuthority = Literal["canonical", "committed", "derived"]
ReviewContextMode = Literal["full_prefix", "milestone_receipt"]


class ReviewContextError(ValueError):
    """Raised before provider dispatch when review context cannot be proven."""


@dataclass(frozen=True, slots=True)
class ReviewContextComponentInput:
    name: str
    source: str
    authority: ReviewContextAuthority
    content: str
    reason: str
    episode_start: int | None = None
    episode_end: int | None = None


@dataclass(frozen=True, slots=True)
class CompiledReviewContext:
    model_input: str
    bundle_sha256: str
    manifest: dict[str, Any]
    manifest_json: str
    output_tokens: int
    mode: ReviewContextMode


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _component(
    *,
    name: str,
    source: str,
    authority: ReviewContextAuthority,
    content: str,
    reason: str,
    episode_start: int | None = None,
    episode_end: int | None = None,
) -> ReviewContextComponentInput:
    if not content:
        raise ReviewContextError(f"Required review context component is empty: {name}")
    return ReviewContextComponentInput(
        name=name,
        source=source,
        authority=authority,
        content=content,
        reason=reason,
        episode_start=episode_start,
        episode_end=episode_end,
    )


def _verified_drafts(
    drafts: Sequence[EpisodeDraft],
    *,
    through_episode: int,
) -> list[EpisodeDraft]:
    ordered = sorted(drafts, key=lambda draft: draft.episode_number)
    actual = [draft.episode_number for draft in ordered]
    expected = list(range(1, through_episode + 1))
    if actual != expected:
        raise ReviewContextError(
            "Structural review requires a complete active screenplay prefix "
            f"from 1 through {through_episode}; received {actual}."
        )
    for draft in ordered:
        if _sha256(draft.content) != draft.content_sha256:
            raise ReviewContextError(
                f"Committed screenplay episode {draft.episode_number} content hash mismatch."
            )
    return ordered


def _screenplays_json(drafts: Sequence[EpisodeDraft]) -> str:
    return json.dumps(
        {
            "start_episode": drafts[0].episode_number if drafts else None,
            "end_episode": drafts[-1].episode_number if drafts else None,
            "episodes": [
                {
                    "episode_number": draft.episode_number,
                    "content_sha256": draft.content_sha256,
                    "content": draft.content,
                }
                for draft in drafts
            ],
        },
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _receipt_json(receipt: BoundStructuralReview) -> str:
    return json.dumps(
        {
            "review_id": receipt.review_id,
            "review_type": receipt.review_type,
            "episode_number": receipt.episode_number,
            "design_content_hash": receipt.design_content_hash,
            "design_epoch": receipt.design_epoch,
            "batch_id": receipt.batch_id,
            "batch_epoch": receipt.batch_epoch,
            "prefix_hash": receipt.prefix_hash,
            "call_id": receipt.call_id,
            "passed": receipt.passed,
            "category": receipt.category,
            "evidence": receipt.evidence,
            "status": receipt.status,
            "reviewed_at": receipt.reviewed_at.isoformat(),
        },
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _render(
    components: Sequence[ReviewContextComponentInput],
    *,
    mode: ReviewContextMode,
    review_type: str,
    episode_number: int,
    context_limit_tokens: int,
    requested_output_tokens: int,
) -> CompiledReviewContext:
    rendered = "\n\n".join(
        (
            f'<review-context-component name="{item.name}" authority="{item.authority}" '
            f'source="{item.source}">\n{item.content}\n</review-context-component>'
        )
        for item in components
    )
    manifest_components = [
        {
            "name": item.name,
            "source": item.source,
            "authority": item.authority,
            "reason": item.reason,
            "episode_start": item.episode_start,
            "episode_end": item.episode_end,
            "included": True,
            "characters": len(item.content),
            "estimated_tokens": estimate_text_tokens(item.content),
            "sha256": _sha256(item.content),
        }
        for item in components
    ]
    input_tokens = estimate_text_tokens(rendered)
    required_tokens = input_tokens + requested_output_tokens + REVIEW_CONTEXT_FIXED_RESERVE_TOKENS
    if required_tokens > context_limit_tokens:
        raise ReviewContextError(
            "Review context exceeds the verified route after deterministic compilation: "
            f"required={required_tokens}, limit={context_limit_tokens}, mode={mode}."
        )
    manifest = {
        "schema_version": REVIEW_CONTEXT_SCHEMA_VERSION,
        "mode": mode,
        "review_type": review_type,
        "episode_number": episode_number,
        "components": manifest_components,
        "bundle_characters": len(rendered),
        "bundle_estimated_tokens": input_tokens,
        "requested_output_tokens": requested_output_tokens,
        "fixed_reserve_tokens": REVIEW_CONTEXT_FIXED_RESERVE_TOKENS,
        "required_tokens": required_tokens,
        "verified_context_limit_tokens": context_limit_tokens,
        "bundle_sha256": _sha256(rendered),
    }
    manifest_json = json.dumps(
        manifest,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return CompiledReviewContext(
        model_input=rendered,
        bundle_sha256=_sha256(rendered),
        manifest=manifest,
        manifest_json=manifest_json,
        output_tokens=requested_output_tokens,
        mode=mode,
    )


def compile_review_context(
    *,
    review_type: Literal["milestone", "final"],
    episode_number: int,
    context_limit_tokens: int | None,
    maximum_output_tokens: int | None,
    series_bible_components: Mapping[str, str],
    design_content_hash: str,
    design_epoch: int,
    story_contract_json: str,
    story_contract_sha256: str,
    committed_prefix: Sequence[EpisodeDraft],
    series_state_json: str,
    current_episode_plan: str,
    previous_receipt: BoundStructuralReview | None,
) -> CompiledReviewContext:
    """Compile a complete-prefix or receipt-bounded structural review packet."""
    if context_limit_tokens is None:
        raise ReviewContextError("A verified review context limit is required.")
    if episode_number < 1:
        raise ReviewContextError("Review episode number must be positive.")
    if set(series_bible_components) != {
        "story_outline",
        "character_biographies",
        "relationship_logic",
    }:
        raise ReviewContextError(
            "Review SeriesBible context is incomplete or contains unknown projections."
        )
    if len(design_content_hash) != 64 or design_epoch < 1:
        raise ReviewContextError("Active SeriesBible review identity is invalid.")
    try:
        contract = json.loads(story_contract_json)
        state = json.loads(series_state_json)
    except json.JSONDecodeError as exc:
        raise ReviewContextError("Review canonical JSON component is invalid.") from exc
    canonical_payload = dict(contract)
    facts = canonical_payload.get("facts")
    if isinstance(facts, list):
        canonical_payload["facts"] = [
            {key: value for key, value in fact.items() if key != "verbatim" or value is not False}
            if isinstance(fact, dict)
            else fact
            for fact in facts
        ]
    canonical_contract = json.dumps(
        canonical_payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    if _sha256(canonical_contract) != story_contract_sha256:
        raise ReviewContextError("StoryContract review context hash mismatch.")
    if state.get("contract_sha256") != story_contract_sha256:
        raise ReviewContextError("SeriesState is not bound to the reviewed StoryContract.")
    if state.get("locked_through_episode") != episode_number:
        raise ReviewContextError("SeriesState does not cover the reviewed milestone.")

    drafts = _verified_drafts(committed_prefix, through_episode=episode_number)
    output_tokens = min(maximum_output_tokens or REVIEW_OUTPUT_TOKENS, REVIEW_OUTPUT_TOKENS)
    common = [
        *(
            _component(
                name=f"series_bible.{name}",
                source=f"active_series_bible:{name}",
                authority="canonical",
                content=content,
                reason="当前活动 SeriesBible 的完整审查投影",
            )
            for name, content in sorted(series_bible_components.items())
        ),
        _component(
            name="story_contract",
            source="approved_episode_outline:story_contract",
            authority="canonical",
            content=canonical_contract,
            reason="锁定的全季机器权威合同",
        ),
        _component(
            name="series_state",
            source="folded_series_state",
            authority="committed",
            content=series_state_json,
            reason="截至当前里程碑的完整连续性状态与活动修复约束",
            episode_start=1,
            episode_end=episode_number,
        ),
        _component(
            name="current_episode_plan",
            source=f"approved_episode_outline:episode_{episode_number}",
            authority="canonical",
            content=current_episode_plan,
            reason="当前结构里程碑对应的分集计划",
            episode_start=episode_number,
            episode_end=episode_number,
        ),
    ]
    full_components = [
        *common,
        _component(
            name="active_screenplay_prefix",
            source="active_episode_candidates:complete_prefix",
            authority="committed",
            content=_screenplays_json(drafts),
            reason="截至当前里程碑的全部活动剧本原文",
            episode_start=1,
            episode_end=episode_number,
        ),
    ]
    try:
        return _render(
            full_components,
            mode="full_prefix",
            review_type=review_type,
            episode_number=episode_number,
            context_limit_tokens=context_limit_tokens,
            requested_output_tokens=output_tokens,
        )
    except ReviewContextError as full_error:
        if previous_receipt is None:
            raise ReviewContextError(
                f"{full_error} No valid prior milestone receipt is available."
            ) from full_error

    receipt = previous_receipt
    if (
        not receipt.passed
        or receipt.category != "pass"
        or receipt.status != "active"
        or receipt.episode_number >= episode_number
        or receipt.design_content_hash != design_content_hash
        or receipt.design_epoch != design_epoch
    ):
        raise ReviewContextError("Prior milestone receipt is not an active passing boundary.")
    receipt_prefix = drafts[: receipt.episode_number]
    expected_prefix_hash = active_prefix_hash(
        [
            {
                "episode_number": draft.episode_number,
                "content_sha256": draft.content_sha256,
            }
            for draft in receipt_prefix
        ]
    )
    if receipt.prefix_hash != expected_prefix_hash:
        raise ReviewContextError("Prior milestone receipt prefix hash mismatch.")
    window = drafts[receipt.episode_number :]
    layered_components = [
        *common,
        _component(
            name="prior_milestone_receipt",
            source=f"series_reviews:{receipt.review_id}",
            authority="committed",
            content=_receipt_json(receipt),
            reason="精确绑定已通过历史前缀的不可变结构审查收据",
            episode_start=1,
            episode_end=receipt.episode_number,
        ),
        _component(
            name="current_review_window",
            source="active_episode_candidates:after_receipt_boundary",
            authority="committed",
            content=_screenplays_json(window),
            reason="上次通过里程碑之后的全部活动剧本原文",
            episode_start=window[0].episode_number,
            episode_end=window[-1].episode_number,
        ),
    ]
    return _render(
        layered_components,
        mode="milestone_receipt",
        review_type=review_type,
        episode_number=episode_number,
        context_limit_tokens=context_limit_tokens,
        requested_output_tokens=output_tokens,
    )
