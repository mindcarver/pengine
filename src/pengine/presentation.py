from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from typing import Any, Literal
from uuid import UUID

from pengine.schemas import (
    CharacterBiographiesPresentation,
    CharacterEntry,
    ContentPackage,
    DeliveryPresentation,
    EpisodeDraft,
    EpisodeEntry,
    EpisodeOutlinePresentation,
    EpisodePlan,
    EpisodeScriptsPresentation,
    RelationshipEntry,
    RelationshipLogicPresentation,
    StoryOutlinePresentation,
    StorySection,
)

_ARTIFACT_MODELS = {
    "story_outline": StoryOutlinePresentation,
    "character_biographies": CharacterBiographiesPresentation,
    "relationship_logic": RelationshipLogicPresentation,
    "episode_outline": EpisodeOutlinePresentation,
    "episode_scripts": EpisodeScriptsPresentation,
}


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _source_story(source: str) -> StoryOutlinePresentation:
    return StoryOutlinePresentation(
        mode="source", source_text=source, source_sha256=_sha256(source)
    )


def _source_characters(source: str) -> CharacterBiographiesPresentation:
    return CharacterBiographiesPresentation(
        mode="source", source_text=source, source_sha256=_sha256(source)
    )


def _source_relationships(source: str) -> RelationshipLogicPresentation:
    return RelationshipLogicPresentation(
        mode="source", source_text=source, source_sha256=_sha256(source)
    )


def _source_episode_outline(source: str) -> EpisodeOutlinePresentation:
    return EpisodeOutlinePresentation(
        mode="source", source_text=source, source_sha256=_sha256(source)
    )


def _source_episode_scripts(source: str) -> EpisodeScriptsPresentation:
    return EpisodeScriptsPresentation(
        mode="source", source_text=source, source_sha256=_sha256(source)
    )


def _split_by_anchors(
    source: str,
    hints: Sequence[Mapping[str, Any]],
) -> list[tuple[Mapping[str, Any], str]]:
    if not hints:
        return []
    positions: list[int] = []
    for hint in hints:
        anchor = hint.get("anchor")
        label = hint.get("label")
        if not isinstance(anchor, str) or not anchor.strip():
            return []
        if not isinstance(label, str) or not label.strip():
            return []
        position = source.find(anchor)
        if position < 0 or source.find(anchor, position + 1) >= 0:
            return []
        positions.append(position)
    if positions != sorted(positions) or len(set(positions)) != len(positions):
        return []
    return [
        (
            hint,
            source[
                0 if index == 0 else position : positions[index + 1]
                if index + 1 < len(positions)
                else None
            ],
        )
        for index, (hint, position) in enumerate(zip(hints, positions, strict=True))
    ]


def _story(source: str, hints: Sequence[Mapping[str, Any]]) -> StoryOutlinePresentation:
    chunks = _split_by_anchors(source, hints)
    if not chunks:
        return _source_story(source)
    sections = [
        StorySection(
            id=f"story-{index}",
            label=str(hint["label"]),
            ordinal=index,
            content=content,
            content_sha256=_sha256(content),
            level=int(hint.get("level", 1)),
        )
        for index, (hint, content) in enumerate(chunks, start=1)
    ]
    return StoryOutlinePresentation(
        mode="structured",
        source_text=source,
        source_sha256=_sha256(source),
        sections=sections,
    )


def _characters(
    source: str,
    hints: Sequence[Mapping[str, Any]],
) -> CharacterBiographiesPresentation:
    chunks = _split_by_anchors(source, hints)
    if not chunks:
        return _source_characters(source)
    entries = [
        CharacterEntry(
            id=f"character-{index}",
            label=str(hint["label"]),
            ordinal=index,
            content=content,
            content_sha256=_sha256(content),
            group=hint.get("group", "other"),
        )
        for index, (hint, content) in enumerate(chunks, start=1)
    ]
    return CharacterBiographiesPresentation(
        mode="structured",
        source_text=source,
        source_sha256=_sha256(source),
        characters=entries,
    )


def _relationships(
    source: str,
    hints: Sequence[Mapping[str, Any]],
) -> RelationshipLogicPresentation:
    chunks = _split_by_anchors(source, hints)
    if not chunks:
        return _source_relationships(source)
    entries = [
        RelationshipEntry(
            id=f"relationship-{index}",
            label=str(hint["label"]),
            ordinal=index,
            content=content,
            content_sha256=_sha256(content),
            group=hint.get("group", "other"),
        )
        for index, (hint, content) in enumerate(chunks, start=1)
    ]
    return RelationshipLogicPresentation(
        mode="structured",
        source_text=source,
        source_sha256=_sha256(source),
        relationships=entries,
    )


def _episodes(
    source: str,
    values: Sequence[tuple[int, str]],
    *,
    require_source_membership: bool = True,
) -> list[EpisodeEntry]:
    expected = list(range(1, len(values) + 1))
    if [number for number, _ in values] != expected:
        return []
    if require_source_membership:
        positions: list[int] = []
        for _, content in values:
            position = source.find(content)
            if position < 0 or source.find(content, position + 1) >= 0:
                return []
            positions.append(position)
        if positions != sorted(positions) or len(set(positions)) != len(positions):
            return []
    return [
        EpisodeEntry(
            id=f"episode-{number}",
            label=f"第 {number} 集",
            ordinal=number,
            content=content,
            content_sha256=_sha256(content),
            episode_number=number,
        )
        for number, content in values
    ]


def compile_delivery_presentation(
    *,
    creation_id: UUID,
    run_kind: Literal["initial", "revision"],
    content: ContentPackage,
    story_hints: Sequence[Mapping[str, Any]] = (),
    character_hints: Sequence[Mapping[str, Any]] = (),
    relationship_hints: Sequence[Mapping[str, Any]] = (),
    episode_plans: Sequence[EpisodePlan] = (),
    episode_drafts: Sequence[EpisodeDraft] = (),
) -> DeliveryPresentation:
    story = _story(content.story_outline, story_hints)
    characters = _characters(content.character_biographies, character_hints)
    relationships = _relationships(content.relationship_logic, relationship_hints)

    outline_entries = _episodes(
        content.episode_outline,
        [(plan.episode_number, plan.plan) for plan in episode_plans],
        require_source_membership=False,
    )
    episode_outline = (
        EpisodeOutlinePresentation(
            mode="structured",
            source_text=content.episode_outline,
            source_sha256=_sha256(content.episode_outline),
            episodes=outline_entries,
        )
        if outline_entries
        else _source_episode_outline(content.episode_outline)
    )

    script_entries = _episodes(
        content.episode_scripts,
        [(draft.episode_number, draft.content) for draft in episode_drafts],
    )
    episode_scripts = (
        EpisodeScriptsPresentation(
            mode="structured",
            source_text=content.episode_scripts,
            source_sha256=_sha256(content.episode_scripts),
            episodes=script_entries,
        )
        if script_entries
        else _source_episode_scripts(content.episode_scripts)
    )

    modes = {
        story.mode,
        characters.mode,
        relationships.mode,
        episode_outline.mode,
        episode_scripts.mode,
    }
    status: Literal["complete", "partial", "source"]
    if modes == {"structured"}:
        status = "complete"
    elif modes == {"source"}:
        status = "source"
    else:
        status = "partial"
    return DeliveryPresentation(
        creation_id=creation_id,
        run_kind=run_kind,
        status=status,
        story_outline=story,
        character_biographies=characters,
        relationship_logic=relationships,
        episode_outline=episode_outline,
        episode_scripts=episode_scripts,
    )


def recover_delivery_presentation(
    *,
    raw_manifest: Any,
    creation_id: UUID,
    run_kind: Literal["initial", "revision"],
    content: ContentPackage,
    episode_plans: Sequence[EpisodePlan] = (),
    episode_drafts: Sequence[EpisodeDraft] = (),
) -> DeliveryPresentation:
    """Keep valid stored artifacts and downgrade only invalid ones to source mode."""
    fallback = compile_delivery_presentation(
        creation_id=creation_id,
        run_kind=run_kind,
        content=content,
        episode_plans=episode_plans,
        episode_drafts=episode_drafts,
    )
    if not isinstance(raw_manifest, Mapping):
        return fallback
    if raw_manifest.get("schema_version") != 1:
        return fallback
    if str(raw_manifest.get("creation_id")) != str(creation_id):
        return fallback
    if raw_manifest.get("run_kind") != run_kind:
        return fallback

    source_by_key = {
        "story_outline": content.story_outline,
        "character_biographies": content.character_biographies,
        "relationship_logic": content.relationship_logic,
        "episode_outline": content.episode_outline,
        "episode_scripts": content.episode_scripts,
    }
    recovered: dict[str, Any] = {}
    for key, model in _ARTIFACT_MODELS.items():
        try:
            artifact = model.model_validate(raw_manifest.get(key))
        except ValueError:
            artifact = getattr(fallback, key)
        if artifact.source_text != source_by_key[key]:
            artifact = getattr(fallback, key)
        recovered[key] = artifact

    modes = {artifact.mode for artifact in recovered.values()}
    status = (
        "complete" if modes == {"structured"} else "source" if modes == {"source"} else "partial"
    )
    return DeliveryPresentation(
        creation_id=creation_id,
        run_kind=run_kind,
        status=status,
        **recovered,
    )
