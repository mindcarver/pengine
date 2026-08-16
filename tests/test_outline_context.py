import json

import pytest

from pengine.outline_context import (
    EpisodeOutlineGroupResult,
    OutlineContextError,
    OutlineSeasonMap,
    assemble_episode_outline,
    compile_outline_group_context,
    compile_season_map_context,
    outline_group_output_tokens,
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
