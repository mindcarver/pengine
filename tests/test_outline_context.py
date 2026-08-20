import json

import pytest

from pengine.outline_context import (
    EpisodeOutlineGroupResult,
    EpisodeOutlineGroupSidecar,
    OutlineContextError,
    OutlineSeasonMap,
    assemble_episode_outline,
    assemble_outline_group_result,
    compile_outline_group_context,
    compile_season_map_context,
    outline_group_output_tokens,
    parse_outline_group_markdown,
    validate_outline_group_references,
)
from pengine.series_bible import ScriptGenerationGroup


def make_season_map(episode_count: int = 30) -> OutlineSeasonMap:
    sizes = [2, 3, 4, 1]
    groups = []
    start = 1
    index = 0
    while start <= episode_count:
        end = min(episode_count, start + sizes[index % len(sizes)] - 1)
        groups.append(
            {
                "group_id": f"group_{index + 1}",
                "start_episode": start,
                "end_episode": end,
                "dramatic_unit": f"第{start}至{end}集完成一次行动转折",
                "boundary_reason": "行动结果改变下一阶段目标",
            }
        )
        start = end + 1
        index += 1
    return OutlineSeasonMap.model_validate(
        {
            "episode_count": episode_count,
            "characters": [
                {
                    "character_id": "lin_lan",
                    "name": "林岚",
                    "role": "返乡调查者",
                    "initial_known_fact_ids": [],
                }
            ],
            "relationships": [],
            "prohibitions": ["不得凭空增加幕后主使"],
            "review_milestones": [],
            "script_generation_groups": groups,
        }
    )


def make_group(group: ScriptGenerationGroup) -> EpisodeOutlineGroupResult:
    episodes = range(group.start_episode, group.end_episode + 1)
    return EpisodeOutlineGroupResult.model_validate(
        {
            "group_id": group.group_id,
            "start_episode": group.start_episode,
            "end_episode": group.end_episode,
            "content": f"{group.group_id} 的分集大纲正文",
            "episodes": [
                {"episode_number": episode, "plan": f"林岚推进线索 {episode}"}
                for episode in episodes
            ],
            "facts": [
                {
                    "fact_id": f"fact_{episode}",
                    "subject": "林岚",
                    "predicate": "发现线索",
                    "kind": "text",
                    "value": f"线索{episode}",
                    "first_revealed_episode": episode,
                }
                for episode in episodes
            ],
            "timeline": [
                {
                    "event_id": f"event_{episode}",
                    "when": f"第{episode}天",
                    "participant_ids": ["lin_lan"],
                    "fact_ids": [f"fact_{episode}"],
                }
                for episode in episodes
            ],
            "knowledge_states": [],
            "clues": [],
            "episode_obligations": [
                {
                    "obligation_id": f"obligation_{episode}",
                    "episode_number": episode,
                    "new_information_fact_ids": [f"fact_{episode}"],
                    "end_hook": f"第{episode}集行动钩子",
                    "required_clue_ids": [],
                }
                for episode in episodes
            ],
        }
    )


def test_season_map_accepts_natural_variable_groups_for_thirty_episodes() -> None:
    season_map = make_season_map()

    assert season_map.episode_count == 30
    assert {
        group.end_episode - group.start_episode + 1 for group in season_map.script_generation_groups
    } > {2}
    assert season_map.script_generation_groups[0].start_episode == 1
    assert season_map.script_generation_groups[-1].end_episode == 30


@pytest.mark.parametrize(
    ("genre", "start_episode", "end_episode"),
    [("现实", 1, 1), ("悬疑", 7, 9), ("古装", 17, 20)],
)
def test_outline_markdown_accepts_varied_content_and_natural_group_sizes(
    genre: str,
    start_episode: int,
    end_episode: int,
) -> None:
    raw_text = "\n\n".join(
        f"## 第{episode_number}集\n\n{genre}行动 {episode_number}\n\n### 集尾钩子\n选择产生代价"
        for episode_number in range(start_episode, end_episode + 1)
    )

    parsed = parse_outline_group_markdown(
        raw_text,
        group_id="natural_turn",
        start_episode=start_episode,
        end_episode=end_episode,
    )

    assert [item.episode_number for item in parsed.episodes] == list(
        range(start_episode, end_episode + 1)
    )
    assert all(genre in item.content for item in parsed.episodes)
    assert parsed.raw_text == raw_text


@pytest.mark.parametrize(
    "raw_text",
    [
        "## 第1集\n正文",
        "## 第2集\n正文\n\n## 第2集\n重复",
        "## 第2集\n正文\n\n## 第4集\n越界",
        "## 第2集\n\n## 第3集\n正文",
        "额外前缀\n\n## 第2集\n正文\n\n## 第3集\n正文",
    ],
)
def test_outline_markdown_rejects_missing_duplicate_out_of_range_or_empty_sections(
    raw_text: str,
) -> None:
    with pytest.raises(OutlineContextError):
        parse_outline_group_markdown(
            raw_text,
            group_id="current_group",
            start_episode=2,
            end_episode=3,
        )


def test_outline_markdown_ignores_episode_like_text_and_fenced_headings() -> None:
    parsed = parse_outline_group_markdown(
        "## 第1集\n\n人物说第2集才会离开。\n\n```markdown\n## 第2集\n```\n\n本集仍是第一集。",
        group_id="single_episode",
        start_episode=1,
        end_episode=1,
    )

    assert len(parsed.episodes) == 1
    assert "## 第2集" in parsed.episodes[0].content


def test_outline_sidecar_excludes_engine_coordinates_and_markdown() -> None:
    assert set(EpisodeOutlineGroupSidecar.model_json_schema()["properties"]) == {
        "character_introductions",
        "facts",
        "timeline",
        "knowledge_states",
        "clues",
        "episode_obligations",
    }


def test_outline_group_assembly_binds_markdown_and_preserves_character_introductions() -> None:
    group = make_season_map(2).script_generation_groups[0]
    markdown = parse_outline_group_markdown(
        "## 第1集\n\n村医帮林岚核对伤情。\n\n## 第2集\n\n林岚确认了新线索。",
        group_id=group.group_id,
        start_episode=group.start_episode,
        end_episode=group.end_episode,
    )
    payload = make_group(group).model_dump(mode="json")
    payload["character_introductions"] = [
        {
            "character_id": "village_doctor",
            "name": "村医",
            "role": "提供伤情判断",
            "initial_known_fact_ids": [],
        }
    ]
    sidecar = EpisodeOutlineGroupSidecar.model_validate(
        {field: payload[field] for field in EpisodeOutlineGroupSidecar.model_fields}
    )

    result = assemble_outline_group_result(group, markdown, sidecar)

    assert result.group_id == group.group_id
    assert result.content == markdown.raw_text
    assert [item.plan for item in result.episodes] == [
        "村医帮林岚核对伤情。",
        "林岚确认了新线索。",
    ]
    assert result.character_introductions[0].character_id == "village_doctor"


def test_outline_compilers_include_only_named_components_and_deduplicate() -> None:
    compiled = compile_season_map_context(
        creation_request="相同故事资料",
        persona_components={"l0": "L0", "l4": "L4"},
        story_outline="相同故事资料",
        character_biographies="人物小传",
        relationship_logic="人物关系",
        maximum_output_tokens=128_000,
    )
    bundle = json.loads(compiled.model_input)

    assert "approved-checkpoints" not in compiled.model_input
    assert {item["name"] for item in bundle["components"]} == {
        "creation_request",
        "persona.l0",
        "persona.l4",
        "character_biographies",
        "relationship_logic",
    }
    assert bundle["aliases"] == {"story_outline": "creation_request"}
    assert compiled.manifest["bundle_sha256"] == compiled.bundle_sha256


def test_group_context_keeps_full_ledger_but_only_two_recent_outline_groups() -> None:
    season_map = make_season_map()
    prior = [make_group(group) for group in season_map.script_generation_groups[:3]]
    current = season_map.script_generation_groups[3]

    compiled = compile_outline_group_context(
        creation_request="故事需求",
        persona_components={"l0": "L0", "l4": "L4"},
        story_outline="故事梗概",
        character_biographies="人物小传",
        relationship_logic="人物关系",
        season_map=season_map,
        prior_groups=prior,
        group=current,
        maximum_output_tokens=128_000,
    )
    components = {
        item["name"]: item["content"] for item in json.loads(compiled.model_input)["components"]
    }
    ledger = json.loads(components["committed_continuity_ledger"])
    recent = json.loads(components["recent_outline_window"])

    assert [item["group_id"] for item in ledger] == ["group_1", "group_2", "group_3"]
    assert [item["group_id"] for item in recent] == ["group_2", "group_3"]
    assert "content" not in ledger[0]
    assert recent[0]["content"] == "group_2 的分集大纲正文"


def test_group_context_carries_prior_character_introductions_in_the_ledger() -> None:
    season_map = make_season_map(5)
    first = make_group(season_map.script_generation_groups[0])
    payload = first.model_dump(mode="json")
    payload["character_introductions"] = [
        {
            "character_id": "village_doctor",
            "name": "村医",
            "role": "提供伤情判断",
            "initial_known_fact_ids": [],
        }
    ]
    first = EpisodeOutlineGroupResult.model_validate(payload)

    compiled = compile_outline_group_context(
        creation_request="乡村悬疑故事",
        persona_components={},
        story_outline="调查旧案",
        character_biographies="调查者返乡",
        relationship_logic="村民互相隐瞒",
        season_map=season_map,
        prior_groups=[first],
        group=season_map.script_generation_groups[1],
        maximum_output_tokens=128_000,
    )
    components = {
        item["name"]: item["content"] for item in json.loads(compiled.model_input)["components"]
    }
    ledger = json.loads(components["committed_continuity_ledger"])

    assert ledger[0]["character_introductions"][0]["character_id"] == "village_doctor"


def test_group_reference_validation_rejects_an_undeclared_character_before_commit() -> None:
    season_map = make_season_map(3)
    group = make_group(season_map.script_generation_groups[0])
    payload = group.model_dump(mode="json")
    payload["timeline"][0]["participant_ids"].append("new_witness")
    candidate = EpisodeOutlineGroupResult.model_validate(payload)

    with pytest.raises(OutlineContextError, match="unknown character IDs.*new_witness"):
        validate_outline_group_references(season_map, [], candidate)


def test_group_can_declare_a_character_for_same_and_later_group_references() -> None:
    season_map = make_season_map(5)
    groups = [make_group(group) for group in season_map.script_generation_groups]
    first_payload = groups[0].model_dump(mode="json")
    first_payload["character_introductions"] = [
        {
            "character_id": "new_witness",
            "name": "新证人",
            "role": "掌握旧案线索的证人",
            "initial_known_fact_ids": [],
        }
    ]
    first_payload["timeline"][0]["participant_ids"].append("new_witness")
    groups[0] = EpisodeOutlineGroupResult.model_validate(first_payload)
    validate_outline_group_references(season_map, [], groups[0])

    second_payload = groups[1].model_dump(mode="json")
    second_payload["timeline"][0]["participant_ids"].append("new_witness")
    groups[1] = EpisodeOutlineGroupResult.model_validate(second_payload)
    validate_outline_group_references(season_map, groups[:1], groups[1])

    payload = assemble_episode_outline(season_map, groups)

    assert [item["character_id"] for item in payload["story_contract"]["characters"]].count(
        "new_witness"
    ) == 1


@pytest.mark.parametrize(
    ("introduction", "message"),
    [
        (
            {
                "character_id": "lin_lan",
                "name": "另一个林岚",
                "role": "冲突人物",
                "initial_known_fact_ids": [],
            },
            "character ID",
        ),
        (
            {
                "character_id": "other_lin_lan",
                "name": "林岚",
                "role": "重名人物",
                "initial_known_fact_ids": [],
            },
            "character name",
        ),
    ],
)
def test_group_reference_validation_rejects_character_redeclarations(
    introduction: dict[str, object],
    message: str,
) -> None:
    season_map = make_season_map(3)
    payload = make_group(season_map.script_generation_groups[0]).model_dump(mode="json")
    payload["character_introductions"] = [introduction]
    candidate = EpisodeOutlineGroupResult.model_validate(payload)

    with pytest.raises(OutlineContextError, match=message):
        validate_outline_group_references(season_map, [], candidate)


def test_group_reference_validation_rejects_unknown_fact_and_clue_references() -> None:
    season_map = make_season_map(3)
    base = make_group(season_map.script_generation_groups[0]).model_dump(mode="json")
    base["timeline"][0]["fact_ids"] = ["unknown_fact"]
    with pytest.raises(OutlineContextError, match="unknown fact IDs.*unknown_fact"):
        validate_outline_group_references(
            season_map,
            [],
            EpisodeOutlineGroupResult.model_validate(base),
        )

    base = make_group(season_map.script_generation_groups[0]).model_dump(mode="json")
    base["episode_obligations"][0]["required_clue_ids"] = ["unknown_clue"]
    with pytest.raises(OutlineContextError, match="unknown clue IDs.*unknown_clue"):
        validate_outline_group_references(
            season_map,
            [],
            EpisodeOutlineGroupResult.model_validate(base),
        )


def test_group_reference_validation_rejects_cross_type_evidence_id_collisions() -> None:
    season_map = make_season_map(3)
    payload = make_group(season_map.script_generation_groups[0]).model_dump(mode="json")
    payload["clues"] = [
        {
            "clue_id": payload["facts"][0]["fact_id"],
            "description": "与事实错误共用 ID 的线索",
            "introduced_episode": 1,
            "explained_episode": 2,
            "callback_episode": 3,
            "introduction_is_visible_or_audible": True,
        }
    ]
    candidate = EpisodeOutlineGroupResult.model_validate(payload)

    with pytest.raises(OutlineContextError, match="globally unique"):
        validate_outline_group_references(season_map, [], candidate)


def test_group_context_rejects_a_non_contiguous_committed_prefix() -> None:
    season_map = make_season_map()
    current = season_map.script_generation_groups[1]

    with pytest.raises(OutlineContextError, match="prefix"):
        compile_outline_group_context(
            creation_request="故事需求",
            persona_components={},
            story_outline="故事梗概",
            character_biographies="人物小传",
            relationship_logic="人物关系",
            season_map=season_map,
            prior_groups=[],
            group=current,
            maximum_output_tokens=128_000,
        )


@pytest.mark.parametrize("episode_count", [30, 80])
def test_assembler_builds_one_complete_long_form_contract(episode_count: int) -> None:
    season_map = make_season_map(episode_count)
    groups = [make_group(group) for group in season_map.script_generation_groups]

    payload = assemble_episode_outline(season_map, groups)

    assert payload["episode_count"] == episode_count
    assert [item["episode_number"] for item in payload["episodes"]] == list(
        range(1, episode_count + 1)
    )
    assert len(payload["story_contract"]["episode_obligations"]) == episode_count
    assert payload["story_contract"]["timeline"][-1]["order"] == episode_count


def test_assembler_expands_group_local_knowledge_deltas_cumulatively() -> None:
    season_map = make_season_map(5)
    groups = [make_group(group) for group in season_map.script_generation_groups]
    first_payload = groups[0].model_dump(mode="json")
    first_payload["knowledge_states"] = [
        {
            "episode_number": 1,
            "character_id": "lin_lan",
            "known_fact_ids": ["fact_1"],
        }
    ]
    groups[0] = EpisodeOutlineGroupResult.model_validate(first_payload)
    second_payload = groups[1].model_dump(mode="json")
    second_payload["knowledge_states"] = [
        {
            "episode_number": 3,
            "character_id": "lin_lan",
            "known_fact_ids": ["fact_3"],
        }
    ]
    groups[1] = EpisodeOutlineGroupResult.model_validate(second_payload)

    payload = assemble_episode_outline(season_map, groups)
    states = {
        (item["episode_number"], item["character_id"]): item["known_fact_ids"]
        for item in payload["story_contract"]["knowledge_states"]
    }

    assert states[(1, "lin_lan")] == ["fact_1"]
    assert states[(2, "lin_lan")] == ["fact_1"]
    assert states[(3, "lin_lan")] == ["fact_1", "fact_3"]


def test_group_output_budget_scales_with_the_natural_group_size() -> None:
    assert (
        outline_group_output_tokens(
            start_episode=1,
            end_episode=1,
            maximum_output_tokens=128_000,
        )
        == 8_192
    )
    assert (
        outline_group_output_tokens(
            start_episode=1,
            end_episode=4,
            maximum_output_tokens=12_000,
        )
        == 12_000
    )
