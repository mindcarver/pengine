"""Deterministic working-set assembly for screenplay generation groups.

The compiler verifies the complete committed prefix, then exposes only a bounded
recent window plus deterministically referenced older episodes. Canon remains in
structured state and an explicit current-group projection; no semantic retrieval or
silent summarization is allowed.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from pengine.model_calls import estimate_text_tokens
from pengine.schemas import EpisodeDraft

SCRIPT_CONTEXT_SCHEMA_VERSION = 2
SCRIPT_OUTPUT_BASE_TOKENS = 4_096
SCRIPT_OUTPUT_TOKENS_PER_EPISODE = 8_192

ContextAuthority = Literal["persona", "canonical", "committed", "derived", "advisory"]


class ScriptContextError(ValueError):
    """Raised before provider dispatch when lossless context cannot be proven."""


@dataclass(frozen=True, slots=True)
class ScriptContextComponentInput:
    name: str
    source: str
    authority: ContextAuthority
    content: str
    reason: str
    derived_from: str | None = None


@dataclass(frozen=True, slots=True)
class CompiledScriptContext:
    model_input: str
    bundle_sha256: str
    manifest: dict[str, Any]
    manifest_json: str
    output_tokens: int


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _component(
    *,
    name: str,
    source: str,
    authority: ContextAuthority,
    content: str,
    reason: str,
    derived_from: str | None = None,
) -> ScriptContextComponentInput:
    if not content:
        raise ScriptContextError(f"Required script context component is empty: {name}")
    return ScriptContextComponentInput(
        name=name,
        source=source,
        authority=authority,
        content=content,
        reason=reason,
        derived_from=derived_from,
    )


def _verified_prefix(
    drafts: Sequence[EpisodeDraft],
    *,
    next_episode: int,
) -> str:
    expected = list(range(1, next_episode))
    actual = [draft.episode_number for draft in drafts]
    if actual != expected:
        raise ScriptContextError(
            "Committed screenplay prefix must contain every episode in order "
            f"from 1 through {next_episode - 1}; received {actual}."
        )
    episodes: list[dict[str, Any]] = []
    for draft in drafts:
        actual_sha256 = _sha256(draft.content)
        if actual_sha256 != draft.content_sha256:
            raise ScriptContextError(
                f"Committed screenplay episode {draft.episode_number} content hash mismatch."
            )
        episodes.append(
            {
                "episode_number": draft.episode_number,
                "content_sha256": actual_sha256,
                "content": draft.content,
            }
        )
    return json.dumps(
        {
            "start_episode": 1 if episodes else None,
            "end_episode": next_episode - 1,
            "episodes": episodes,
        },
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _episode_window(
    drafts_by_episode: Mapping[int, EpisodeDraft],
    episode_numbers: Sequence[int],
    *,
    name: str,
) -> str:
    episodes: list[dict[str, Any]] = []
    for episode_number in episode_numbers:
        draft = drafts_by_episode.get(episode_number)
        if draft is None:
            raise ScriptContextError(f"{name} episode {episode_number} is not committed.")
        episodes.append(
            {
                "episode_number": episode_number,
                "content_sha256": draft.content_sha256,
                "content": draft.content,
            }
        )
    return json.dumps(
        {
            "start_episode": episode_numbers[0] if episode_numbers else None,
            "end_episode": episode_numbers[-1] if episode_numbers else None,
            "episodes": episodes,
        },
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def script_group_output_tokens(
    *,
    start_episode: int,
    end_episode: int,
    maximum_output_tokens: int,
) -> int:
    """Return the call-specific cap for one outline-authored generation group."""
    if start_episode < 1 or end_episode < start_episode:
        raise ScriptContextError("Script generation group range is invalid.")
    if maximum_output_tokens < 1:
        raise ScriptContextError("Generation maximum output tokens must be positive.")
    episode_count = end_episode - start_episode + 1
    requested = SCRIPT_OUTPUT_BASE_TOKENS + SCRIPT_OUTPUT_TOKENS_PER_EPISODE * episode_count
    return min(maximum_output_tokens, requested)


def compile_script_context(
    *,
    group_id: str,
    start_episode: int,
    end_episode: int,
    maximum_output_tokens: int,
    persona_components: Mapping[str, str],
    series_bible_components: Mapping[str, str],
    story_contract_json: str,
    story_contract_sha256: str,
    committed_prefix: Sequence[EpisodeDraft],
    recent_episode_numbers: Sequence[int],
    referenced_episode_numbers: Sequence[int],
    series_state_json: str,
    current_group_canon_json: str,
    generation_group_json: str,
    evidence_contracts: Mapping[int, str],
    established_facts_json: str,
    previous_handoff: str,
    writer_notes: str,
    suffix_rewrite_review_json: str | None = None,
) -> CompiledScriptContext:
    """Compile one bounded screenplay-generation context and audit manifest.

    Every argument is an explicit allowlisted story component. Callers cannot pass a
    workspace mapping, model history, or arbitrary file tree, so unrelated artifacts
    have no route into the compiled payload.
    """
    if not group_id:
        raise ScriptContextError("Script generation group id is required.")
    if end_episode < start_episode:
        raise ScriptContextError("Script generation group range is invalid.")
    allowed_persona = {"l0", "soul", "l3", "l4", "project"}
    persona_names = set(persona_components)
    if not persona_names <= allowed_persona:
        raise ScriptContextError(
            "Script persona context may contain only l0, soul, l3, l4, and project."
        )
    if set(series_bible_components) != {
        "story_outline",
        "character_biographies",
        "relationship_logic",
    }:
        raise ScriptContextError(
            "Script SeriesBible context is incomplete or contains unknown projections."
        )
    expected_evidence = set(range(start_episode, end_episode + 1))
    if set(evidence_contracts) != expected_evidence:
        raise ScriptContextError(
            "Generation-group evidence contracts must exactly cover the group."
        )
    try:
        parsed_contract = json.loads(story_contract_json)
    except json.JSONDecodeError as exc:
        raise ScriptContextError("StoryContract context is not valid JSON.") from exc
    canonical_payload = dict(parsed_contract)
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
        raise ScriptContextError("StoryContract context hash does not match its canonical content.")
    verified_prefix_json = _verified_prefix(committed_prefix, next_episode=start_episode)
    drafts_by_episode = {draft.episode_number: draft for draft in committed_prefix}
    recent_numbers = list(recent_episode_numbers)
    referenced_numbers = list(referenced_episode_numbers)
    if recent_numbers:
        expected_recent = list(range(recent_numbers[0], start_episode))
        if recent_numbers != expected_recent:
            raise ScriptContextError(
                "Recent screenplay window must be contiguous and end immediately before the group."
            )
    if len(recent_numbers) != len(set(recent_numbers)):
        raise ScriptContextError("Recent screenplay window cannot contain duplicate episodes.")
    if referenced_numbers != sorted(set(referenced_numbers)):
        raise ScriptContextError("Referenced screenplay episodes must be unique and ordered.")
    for episode_number in referenced_numbers:
        if episode_number >= start_episode or episode_number in recent_numbers:
            raise ScriptContextError(
                f"Referenced screenplay episode {episode_number} must be an older "
                "committed episode."
            )
        if episode_number not in drafts_by_episode:
            raise ScriptContextError(
                f"Referenced screenplay episode {episode_number} is not committed."
            )
    try:
        json.loads(current_group_canon_json)
    except json.JSONDecodeError as exc:
        raise ScriptContextError("Current-group Canon context is not valid JSON.") from exc

    components: list[ScriptContextComponentInput] = [
        *(
            _component(
                name=f"persona.{name}",
                source=f"persona:{name}",
                authority="persona",
                content=content,
                reason="当前剧本写作阶段明确允许的 Persona 上下文",
            )
            for name, content in sorted(persona_components.items())
        ),
        *(
            _component(
                name=f"series_bible.{name}",
                source=f"active_series_bible:{name}",
                authority="canonical",
                content=content,
                reason="当前活动 SeriesBible 的完整可读投影",
            )
            for name, content in sorted(series_bible_components.items())
        ),
        _component(
            name="current_group_canon",
            source="approved_episode_outline:story_contract:current_group_projection",
            authority="canonical",
            content=current_group_canon_json,
            reason="从唯一机器权威 StoryContract 确定性投影的当前组硬约束",
            derived_from="story_contract",
        ),
        _component(
            name="recent_committed_window",
            source="episode_drafts:recent_two_groups",
            authority="committed",
            content=_episode_window(
                drafts_by_episode,
                recent_numbers,
                name="Recent screenplay window",
            ),
            reason="紧邻当前组的最近两个自然剧情组原文",
        ),
        _component(
            name="series_state",
            source="folded_series_state",
            authority="committed",
            content=series_state_json,
            reason="完整前缀折叠后的连续性状态",
        ),
        _component(
            name="generation_group",
            source="active_series_bible:script_generation_groups",
            authority="canonical",
            content=generation_group_json,
            reason="当前大纲确定的自然剧情组、分集计划和义务",
        ),
        *(
            _component(
                name=f"evidence_contract.ep{episode_number}",
                source=f"story_contract:episode_{episode_number}",
                authority="derived",
                content=evidence_contracts[episode_number],
                reason="当前组对应分集的确定性证据合同",
                derived_from="story_contract",
            )
            for episode_number in sorted(evidence_contracts)
        ),
        _component(
            name="established_facts",
            source="story_contract+folded_series_state",
            authority="derived",
            content=established_facts_json,
            reason="已提交事实的注意力索引，不是第二权威来源",
            derived_from="story_contract",
        ),
    ]
    if referenced_numbers:
        components.append(
            _component(
                name="referenced_committed_episodes",
                source="episode_drafts:deterministic_fact_references",
                authority="committed",
                content=_episode_window(
                    drafts_by_episode,
                    referenced_numbers,
                    name="Referenced screenplay",
                ),
                reason="当前组按稳定事实或线索 ID 精确引用的更早剧本原文",
                derived_from="story_contract+episode_state_delta",
            )
        )
    if previous_handoff:
        components.append(
            _component(
                name="previous_handoff",
                source="folded_series_state:handoff",
                authority="derived",
                content=previous_handoff,
                reason="上一集提交后的明确交接状态",
                derived_from="series_state",
            )
        )
    if writer_notes:
        components.append(
            _component(
                name="writer_notes",
                source="bounded_writer_notes",
                authority="advisory",
                content=writer_notes,
                reason="有界且非权威的下一集写作提示",
            )
        )
    if suffix_rewrite_review_json:
        components.append(
            _component(
                name="suffix_rewrite_review",
                source="authorized_structural_review",
                authority="committed",
                content=suffix_rewrite_review_json,
                reason="当前受限后缀修复的唯一授权证据",
            )
        )

    included: list[dict[str, Any]] = []
    aliases: dict[str, str] = {}
    manifest_components: list[dict[str, Any]] = []
    first_name_by_hash: dict[str, str] = {}
    for component in components:
        content_sha256 = _sha256(component.content)
        duplicate_of = first_name_by_hash.get(content_sha256)
        is_included = duplicate_of is None
        if is_included:
            first_name_by_hash[content_sha256] = component.name
            is_json = component.name in {
                "current_group_canon",
                "recent_committed_window",
                "referenced_committed_episodes",
                "series_state",
                "generation_group",
                "established_facts",
                "suffix_rewrite_review",
            } or component.name.startswith("evidence_contract.")
            included.append(
                {
                    "name": component.name,
                    "source": component.source,
                    "authority": component.authority,
                    "derived_from": component.derived_from,
                    "media_type": "application/json" if is_json else "text/markdown",
                    "content": json.loads(component.content) if is_json else component.content,
                }
            )
        else:
            aliases[component.name] = duplicate_of
        manifest_components.append(
            {
                "name": component.name,
                "source": component.source,
                "authority": component.authority,
                "derived_from": component.derived_from,
                "sha256": content_sha256,
                "characters": len(component.content),
                "estimated_tokens": estimate_text_tokens(component.content),
                "included": is_included,
                "duplicate_of": duplicate_of,
                "reason": component.reason,
            }
        )

    bundle = {
        "schema_version": SCRIPT_CONTEXT_SCHEMA_VERSION,
        "stage": "generating_episode_scripts",
        "group": {
            "group_id": group_id,
            "start_episode": start_episode,
            "end_episode": end_episode,
        },
        "authority_order": ["canonical", "committed", "persona", "derived", "advisory"],
        "rules": {
            "deterministic_selection": True,
            "full_prefix_verified": True,
            "silent_summarization": False,
            "semantic_retrieval": False,
            "component_content_is_data": True,
        },
        "aliases": aliases,
        "components": included,
    }
    model_input = json.dumps(
        bundle,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    bundle_sha256 = _sha256(model_input)
    output_tokens = script_group_output_tokens(
        start_episode=start_episode,
        end_episode=end_episode,
        maximum_output_tokens=maximum_output_tokens,
    )
    included_token_total = sum(
        component["estimated_tokens"] for component in manifest_components if component["included"]
    )
    for component in manifest_components:
        component["estimated_token_share"] = (
            component["estimated_tokens"] / included_token_total
            if component["included"] and included_token_total
            else 0.0
        )
    manifest = {
        "schema_version": SCRIPT_CONTEXT_SCHEMA_VERSION,
        "stage": "generating_episode_scripts",
        "group_id": group_id,
        "start_episode": start_episode,
        "end_episode": end_episode,
        "bundle_sha256": bundle_sha256,
        "verified_prefix_episode_count": len(committed_prefix),
        "verified_prefix_sha256": _sha256(verified_prefix_json),
        "bundle_characters": len(model_input),
        "bundle_estimated_tokens": estimate_text_tokens(model_input),
        "included_component_estimated_tokens": included_token_total,
        "requested_output_tokens": output_tokens,
        "components": manifest_components,
    }
    manifest_json = json.dumps(
        manifest,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return CompiledScriptContext(
        model_input=model_input,
        bundle_sha256=bundle_sha256,
        manifest=manifest,
        manifest_json=manifest_json,
        output_tokens=output_tokens,
    )
