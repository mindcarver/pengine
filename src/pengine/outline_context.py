"""Deterministic context and assembly contracts for grouped episode outlining."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from pydantic import Field, model_validator

from pengine.continuity import (
    CharacterKnowledgeState,
    CharacterSpec,
    ClueSpec,
    ContinuityModel,
    EpisodeObligation,
    NonEmptyText,
    RelationshipSpec,
    StableId,
    StoryContract,
    StoryFact,
    TimelineEvent,
)
from pengine.model_calls import estimate_text_tokens
from pengine.schemas import EpisodePlan
from pengine.series_bible import ScriptGenerationGroup, validate_script_generation_groups

OUTLINE_CONTEXT_SCHEMA_VERSION = 1
SEASON_MAP_OUTPUT_TOKENS = 24_576
OUTLINE_GROUP_OUTPUT_BASE_TOKENS = 4_096
OUTLINE_GROUP_OUTPUT_TOKENS_PER_EPISODE = 4_096


class OutlineContextError(ValueError):
    """Raised before provider dispatch when grouped outline input is invalid."""


class OutlineSeasonMap(ContinuityModel):
    """Small whole-season design that decides natural generation boundaries."""

    episode_count: int = Field(ge=1)
    characters: list[CharacterSpec] = Field(min_length=1)
    relationships: list[RelationshipSpec] = Field(default_factory=list)
    prohibitions: list[NonEmptyText] = Field(default_factory=list)
    review_milestones: list[int] = Field(default_factory=list)
    script_generation_groups: list[ScriptGenerationGroup] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_map(self) -> OutlineSeasonMap:
        character_ids = [item.character_id for item in self.characters]
        if len(character_ids) != len(set(character_ids)):
            raise ValueError("Season-map character IDs must be unique")
        names = [item.name for item in self.characters]
        if len(names) != len(set(names)):
            raise ValueError("Season-map character names must be unique")
        known_characters = set(character_ids)
        for relationship in self.relationships:
            if {
                relationship.source_character_id,
                relationship.target_character_id,
            } - known_characters:
                raise ValueError("Season-map relationship references an unknown character")
        milestones = sorted({int(item) for item in self.review_milestones})
        if len(milestones) != len(self.review_milestones):
            raise ValueError("Season-map review milestones must be unique")
        if any(item < 1 or item > self.episode_count for item in milestones):
            raise ValueError("Season-map review milestone exceeds the episode range")
        self.review_milestones = milestones
        validate_script_generation_groups(
            self.script_generation_groups,
            episode_count=self.episode_count,
            review_milestones=milestones,
            allow_empty=False,
        )
        return self


class OutlineTimelineEvent(ContinuityModel):
    """Timeline event without a global order; the assembler assigns that order."""

    event_id: StableId
    when: NonEmptyText
    participant_ids: list[StableId] = Field(default_factory=list)
    fact_ids: list[StableId] = Field(default_factory=list)


class EpisodeOutlineGroupResult(ContinuityModel):
    """One natural outline group, containing only its detailed episode payload."""

    group_id: StableId
    start_episode: int = Field(ge=1)
    end_episode: int = Field(ge=1)
    content: NonEmptyText
    episodes: list[EpisodePlan] = Field(min_length=1)
    character_introductions: list[CharacterSpec] = Field(default_factory=list)
    facts: list[StoryFact] = Field(min_length=1)
    timeline: list[OutlineTimelineEvent] = Field(min_length=1)
    knowledge_states: list[CharacterKnowledgeState] = Field(default_factory=list)
    clues: list[ClueSpec] = Field(default_factory=list)
    episode_obligations: list[EpisodeObligation] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_group(self) -> EpisodeOutlineGroupResult:
        expected = list(range(self.start_episode, self.end_episode + 1))
        if [item.episode_number for item in self.episodes] != expected:
            raise ValueError("Outline group episode plans must exactly cover its range")
        if [item.episode_number for item in self.episode_obligations] != expected:
            raise ValueError("Outline group obligations must exactly cover its range")
        if any(item.first_revealed_episode not in expected for item in self.facts):
            raise ValueError("Outline group facts must first be revealed inside the group")
        if any(item.episode_number not in expected for item in self.knowledge_states):
            raise ValueError("Outline group knowledge changes must occur inside the group")
        if any(item.introduced_episode not in expected for item in self.clues):
            raise ValueError("A clue must be declared by the group that introduces it")
        fact_ids_by_episode = {
            episode: {item.fact_id for item in self.facts if item.first_revealed_episode == episode}
            for episode in expected
        }
        for obligation in self.episode_obligations:
            if (
                set(obligation.new_information_fact_ids)
                != fact_ids_by_episode[obligation.episode_number]
            ):
                raise ValueError("Group obligations must match the group's fact reveals")
        return self


def validate_outline_group_references(
    season_map: OutlineSeasonMap,
    prior_groups: Sequence[EpisodeOutlineGroupResult],
    candidate: EpisodeOutlineGroupResult,
) -> None:
    """Validate one group against the immutable cumulative outline prefix."""

    character_ids = {item.character_id for item in season_map.characters}
    character_names = {item.name for item in season_map.characters}
    fact_ids: set[str] = set()
    clue_ids: set[str] = set()
    obligation_ids: set[str] = set()
    event_ids: set[str] = set()
    knowledge_pairs: set[tuple[int, str]] = set()

    for group in [*prior_groups, candidate]:
        for character in group.character_introductions:
            if character.character_id in character_ids:
                raise OutlineContextError(
                    f"Outline group redeclares character ID: {character.character_id}"
                )
            if character.name in character_names:
                raise OutlineContextError(
                    f"Outline group redeclares character name: {character.name}"
                )
            if character.initial_known_fact_ids:
                raise OutlineContextError(
                    "Group-introduced characters must declare knowledge in knowledge_states"
                )
            character_ids.add(character.character_id)
            character_names.add(character.name)

        group_fact_ids = {item.fact_id for item in group.facts}
        group_clue_ids = {item.clue_id for item in group.clues}
        group_obligation_ids = {item.obligation_id for item in group.episode_obligations}
        group_event_ids = {item.event_id for item in group.timeline}
        if len(group_fact_ids) != len(group.facts) or fact_ids & group_fact_ids:
            raise OutlineContextError("Outline groups contain a duplicate fact ID")
        if len(group_clue_ids) != len(group.clues) or clue_ids & group_clue_ids:
            raise OutlineContextError("Outline groups contain a duplicate clue ID")
        if (
            len(group_obligation_ids) != len(group.episode_obligations)
            or obligation_ids & group_obligation_ids
        ):
            raise OutlineContextError("Outline groups contain a duplicate obligation ID")
        if len(group_event_ids) != len(group.timeline) or event_ids & group_event_ids:
            raise OutlineContextError("Outline groups contain a duplicate timeline event ID")

        evidence_ids = {
            *fact_ids,
            *clue_ids,
            *obligation_ids,
            *group_fact_ids,
            *group_clue_ids,
            *group_obligation_ids,
        }
        expected_evidence_count = (
            len(fact_ids)
            + len(clue_ids)
            + len(obligation_ids)
            + len(group_fact_ids)
            + len(group_clue_ids)
            + len(group_obligation_ids)
        )
        if len(evidence_ids) != expected_evidence_count:
            raise OutlineContextError("Fact, clue, and obligation IDs must be globally unique")

        available_fact_ids = fact_ids | group_fact_ids
        available_clue_ids = clue_ids | group_clue_ids
        for event in group.timeline:
            unknown_characters = sorted(set(event.participant_ids) - character_ids)
            if unknown_characters:
                raise OutlineContextError(
                    "Outline timeline references unknown character IDs: "
                    + ", ".join(unknown_characters)
                )
            unknown_facts = sorted(set(event.fact_ids) - available_fact_ids)
            if unknown_facts:
                raise OutlineContextError(
                    "Outline timeline references unknown fact IDs: " + ", ".join(unknown_facts)
                )
        for state in group.knowledge_states:
            if state.character_id not in character_ids:
                raise OutlineContextError(
                    "Outline knowledge references unknown character ID: " + state.character_id
                )
            unknown_facts = sorted(set(state.known_fact_ids) - available_fact_ids)
            if unknown_facts:
                raise OutlineContextError(
                    "Outline knowledge references unknown fact IDs: " + ", ".join(unknown_facts)
                )
            pair = (state.episode_number, state.character_id)
            if pair in knowledge_pairs:
                raise OutlineContextError("Outline groups contain a duplicate knowledge state")
            knowledge_pairs.add(pair)
        for obligation in group.episode_obligations:
            unknown_clues = sorted(set(obligation.required_clue_ids) - available_clue_ids)
            if unknown_clues:
                raise OutlineContextError(
                    "Outline obligation references unknown clue IDs: " + ", ".join(unknown_clues)
                )

        fact_ids.update(group_fact_ids)
        clue_ids.update(group_clue_ids)
        obligation_ids.update(group_obligation_ids)
        event_ids.update(group_event_ids)


@dataclass(frozen=True, slots=True)
class CompiledOutlineContext:
    model_input: str
    bundle_sha256: str
    manifest: dict[str, Any]
    manifest_json: str
    output_tokens: int


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _compile(
    *,
    mode: str,
    components: Sequence[tuple[str, str, str, str]],
    output_tokens: int,
    metadata: Mapping[str, Any],
) -> CompiledOutlineContext:
    included: list[dict[str, Any]] = []
    manifest_components: list[dict[str, Any]] = []
    aliases: dict[str, str] = {}
    first_by_hash: dict[str, str] = {}
    for name, source, authority, content in components:
        if not content:
            raise OutlineContextError(f"Outline context component is empty: {name}")
        digest = _sha256(content)
        duplicate_of = first_by_hash.get(digest)
        is_included = duplicate_of is None
        if is_included:
            first_by_hash[digest] = name
            included.append(
                {
                    "name": name,
                    "source": source,
                    "authority": authority,
                    "content": content,
                }
            )
        else:
            aliases[name] = duplicate_of
        manifest_components.append(
            {
                "name": name,
                "source": source,
                "authority": authority,
                "sha256": digest,
                "characters": len(content),
                "estimated_tokens": estimate_text_tokens(content),
                "included": is_included,
                "duplicate_of": duplicate_of,
            }
        )
    bundle = {
        "schema_version": OUTLINE_CONTEXT_SCHEMA_VERSION,
        "stage": "generating_episode_outline",
        "mode": mode,
        **dict(metadata),
        "rules": {
            "lossless": True,
            "component_content_is_data": True,
            "silent_summarization": False,
            "retrieval_fallback": False,
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
    total = sum(item["estimated_tokens"] for item in manifest_components if item["included"])
    for item in manifest_components:
        item["estimated_token_share"] = (
            item["estimated_tokens"] / total if item["included"] and total else 0.0
        )
    manifest = {
        "schema_version": OUTLINE_CONTEXT_SCHEMA_VERSION,
        "stage": "generating_episode_outline",
        "mode": mode,
        **dict(metadata),
        "bundle_sha256": bundle_sha256,
        "bundle_characters": len(model_input),
        "bundle_estimated_tokens": estimate_text_tokens(model_input),
        "requested_output_tokens": output_tokens,
        "components": manifest_components,
    }
    return CompiledOutlineContext(
        model_input=model_input,
        bundle_sha256=bundle_sha256,
        manifest=manifest,
        manifest_json=json.dumps(
            manifest,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ),
        output_tokens=output_tokens,
    )


def compile_season_map_context(
    *,
    creation_request: str,
    persona_components: Mapping[str, str],
    story_outline: str,
    character_biographies: str,
    relationship_logic: str,
    maximum_output_tokens: int,
) -> CompiledOutlineContext:
    allowed_persona = {"l0", "soul", "l3", "l4", "project"}
    if not set(persona_components) <= allowed_persona:
        raise OutlineContextError("Outline persona context contains an unknown component")
    components = [
        ("creation_request", "creation_request", "user", creation_request),
        *(
            (f"persona.{name}", f"persona:{name}", "persona", content)
            for name, content in sorted(persona_components.items())
        ),
        ("story_outline", "approved:story_outline", "canonical", story_outline),
        (
            "character_biographies",
            "approved:character_biographies",
            "canonical",
            character_biographies,
        ),
        (
            "relationship_logic",
            "approved:relationship_logic",
            "canonical",
            relationship_logic,
        ),
    ]
    return _compile(
        mode="season_map",
        components=components,
        output_tokens=min(maximum_output_tokens, SEASON_MAP_OUTPUT_TOKENS),
        metadata={},
    )


def outline_group_output_tokens(
    *,
    start_episode: int,
    end_episode: int,
    maximum_output_tokens: int,
) -> int:
    if start_episode < 1 or end_episode < start_episode:
        raise OutlineContextError("Outline group range is invalid")
    requested = (
        OUTLINE_GROUP_OUTPUT_BASE_TOKENS
        + (end_episode - start_episode + 1) * OUTLINE_GROUP_OUTPUT_TOKENS_PER_EPISODE
    )
    return min(maximum_output_tokens, requested)


def compile_outline_group_context(
    *,
    creation_request: str,
    persona_components: Mapping[str, str],
    story_outline: str,
    character_biographies: str,
    relationship_logic: str,
    season_map: OutlineSeasonMap,
    prior_groups: Sequence[EpisodeOutlineGroupResult],
    group: ScriptGenerationGroup,
    maximum_output_tokens: int,
) -> CompiledOutlineContext:
    expected_start = 1
    for prior in prior_groups:
        if prior.start_episode != expected_start:
            raise OutlineContextError("Committed outline-group prefix is incomplete")
        expected_start = prior.end_episode + 1
    if expected_start != group.start_episode:
        raise OutlineContextError("Committed outline-group prefix does not reach this group")
    continuity_ledger = [
        {
            "group_id": item.group_id,
            "start_episode": item.start_episode,
            "end_episode": item.end_episode,
            "character_introductions": [
                value.model_dump(mode="json") for value in item.character_introductions
            ],
            "facts": [value.model_dump(mode="json") for value in item.facts],
            "timeline": [value.model_dump(mode="json") for value in item.timeline],
            "knowledge_states": [value.model_dump(mode="json") for value in item.knowledge_states],
            "clues": [value.model_dump(mode="json") for value in item.clues],
            "episode_obligations": [
                value.model_dump(mode="json") for value in item.episode_obligations
            ],
        }
        for item in prior_groups
    ]
    recent_outline_window = [
        {
            "group_id": item.group_id,
            "start_episode": item.start_episode,
            "end_episode": item.end_episode,
            "content": item.content,
            "episodes": [value.model_dump(mode="json") for value in item.episodes],
        }
        for item in prior_groups[-2:]
    ]
    components = [
        ("creation_request", "creation_request", "user", creation_request),
        *(
            (f"persona.{name}", f"persona:{name}", "persona", content)
            for name, content in sorted(persona_components.items())
        ),
        ("story_outline", "approved:story_outline", "canonical", story_outline),
        (
            "character_biographies",
            "approved:character_biographies",
            "canonical",
            character_biographies,
        ),
        (
            "relationship_logic",
            "approved:relationship_logic",
            "canonical",
            relationship_logic,
        ),
        (
            "season_map",
            "committed:season_map",
            "committed",
            season_map.model_dump_json(),
        ),
        (
            "committed_continuity_ledger",
            "committed:outline_groups:continuity",
            "committed",
            json.dumps(
                continuity_ledger,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            ),
        ),
        (
            "recent_outline_window",
            "committed:outline_groups:recent",
            "committed",
            json.dumps(
                recent_outline_window,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            ),
        ),
        (
            "current_group",
            "season_map:script_generation_groups",
            "canonical",
            group.model_dump_json(),
        ),
    ]
    return _compile(
        mode="outline_group",
        components=components,
        output_tokens=outline_group_output_tokens(
            start_episode=group.start_episode,
            end_episode=group.end_episode,
            maximum_output_tokens=maximum_output_tokens,
        ),
        metadata={
            "group_id": group.group_id,
            "start_episode": group.start_episode,
            "end_episode": group.end_episode,
        },
    )


def assemble_episode_outline(
    season_map: OutlineSeasonMap,
    groups: Sequence[EpisodeOutlineGroupResult],
) -> dict[str, Any]:
    if [item.group_id for item in groups] != [
        item.group_id for item in season_map.script_generation_groups
    ]:
        raise OutlineContextError("Outline groups do not match the committed season map")
    for expected, actual in zip(season_map.script_generation_groups, groups, strict=True):
        if (
            actual.start_episode != expected.start_episode
            or actual.end_episode != expected.end_episode
        ):
            raise OutlineContextError("Outline group range changed after season-map commitment")
    validated_prefix: list[EpisodeOutlineGroupResult] = []
    for group in groups:
        validate_outline_group_references(season_map, validated_prefix, group)
        validated_prefix.append(group)
    characters = [
        *season_map.characters,
        *(character for group in groups for character in group.character_introductions),
    ]
    timeline: list[TimelineEvent] = []
    for group in groups:
        for item in group.timeline:
            timeline.append(
                TimelineEvent(
                    event_id=item.event_id,
                    order=len(timeline) + 1,
                    when=item.when,
                    participant_ids=item.participant_ids,
                    fact_ids=item.fact_ids,
                )
            )
    known_by_character = {
        item.character_id: set(item.initial_known_fact_ids) for item in characters
    }
    knowledge_states: list[CharacterKnowledgeState] = []
    for group in groups:
        for item in group.knowledge_states:
            known = known_by_character.setdefault(item.character_id, set())
            known.update(item.known_fact_ids)
            knowledge_states.append(item.model_copy(update={"known_fact_ids": sorted(known)}))
    contract = StoryContract(
        episode_count=season_map.episode_count,
        characters=characters,
        relationships=season_map.relationships,
        facts=[item for group in groups for item in group.facts],
        timeline=timeline,
        knowledge_states=knowledge_states,
        clues=[item for group in groups for item in group.clues],
        prohibitions=season_map.prohibitions,
        episode_obligations=[item for group in groups for item in group.episode_obligations],
    )
    return {
        "stage": "generating_episode_outline",
        "content": "\n\n".join(item.content for item in groups),
        "episode_count": season_map.episode_count,
        "episodes": [item.model_dump(mode="json") for group in groups for item in group.episodes],
        "story_contract": contract.model_dump(mode="json"),
        "review_milestones": season_map.review_milestones,
        "script_generation_groups": [
            item.model_dump(mode="json") for item in season_map.script_generation_groups
        ],
    }
