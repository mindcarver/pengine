from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from pengine.continuity import (
    ContinuityViolation,
    EpisodeStateDelta,
    SemanticReview,
    SeriesState,
    StoryContract,
    bind_episode_delta_to_contract,
    build_episode_lock,
    initial_series_state,
    render_story_contract_markdown,
    story_contract_sha256,
    validate_episode_candidate,
)


def make_contract(*, numeric_facts: list[dict] | None = None) -> StoryContract:
    facts = numeric_facts or [
        {
            "fact_id": "arrival_date",
            "subject": "林岚",
            "predicate": "回乡日期",
            "kind": "date",
            "value": "2015-08-12",
            "first_revealed_episode": 1,
        },
        {
            "fact_id": "meeting_time",
            "subject": "林岚",
            "predicate": "抵达旧屋时间",
            "kind": "time",
            "value": "21:40",
            "first_revealed_episode": 1,
        },
        {
            "fact_id": "locked_duration",
            "subject": "监控记录",
            "predicate": "持续时间",
            "kind": "duration",
            "value": "80",
            "unit": "分钟",
            "first_revealed_episode": 1,
        },
    ]
    fact_ids = [fact["fact_id"] for fact in facts]
    return StoryContract.model_validate(
        {
            "version": 1,
            "episode_count": 1,
            "characters": [
                {
                    "character_id": "lin_lan",
                    "name": "林岚",
                    "role": "调查者",
                    "initial_known_fact_ids": [],
                }
            ],
            "relationships": [],
            "facts": facts,
            "timeline": [
                {
                    "event_id": "return_home",
                    "order": 1,
                    "when": "2015-08-12",
                    "participant_ids": ["lin_lan"],
                    "fact_ids": fact_ids,
                }
            ],
            "knowledge_states": [
                {
                    "episode_number": 1,
                    "character_id": "lin_lan",
                    "known_fact_ids": fact_ids,
                }
            ],
            "clues": [],
            "prohibitions": ["不得增加角色"],
            "episode_obligations": [
                {
                    "obligation_id": "episode_one_obligation",
                    "episode_number": 1,
                    "new_information_fact_ids": fact_ids,
                    "end_hook": "门后传来第二次敲击",
                    "required_clue_ids": [],
                }
            ],
        }
    )


def make_sparse_knowledge_contract(knowledge_states: list[dict]) -> StoryContract:
    facts = [
        {
            "fact_id": f"fact_{number}",
            "subject": "旧案",
            "predicate": f"事实{number}",
            "kind": "text",
            "value": f"事实{number}",
            "first_revealed_episode": episode,
        }
        for number, episode in (("one", 1), ("two", 2), ("three", 3))
    ]
    fact_ids = [fact["fact_id"] for fact in facts]
    return StoryContract.model_validate(
        {
            "version": 1,
            "episode_count": 3,
            "characters": [
                {
                    "character_id": "alice",
                    "name": "阿丽",
                    "role": "调查者",
                    "initial_known_fact_ids": ["fact_one"],
                },
                {
                    "character_id": "bob",
                    "name": "阿博",
                    "role": "证人",
                    "initial_known_fact_ids": [],
                },
            ],
            "relationships": [],
            "facts": facts,
            "timeline": [
                {
                    "event_id": "event_one",
                    "order": 1,
                    "when": "故事开始",
                    "participant_ids": ["alice", "bob"],
                    "fact_ids": fact_ids,
                }
            ],
            "knowledge_states": knowledge_states,
            "clues": [],
            "prohibitions": [],
            "episode_obligations": [
                {
                    "obligation_id": f"obligation_{episode}",
                    "episode_number": episode,
                    "new_information_fact_ids": [f"fact_{number}"],
                    "end_hook": f"第{episode}集钩子",
                    "required_clue_ids": [],
                }
                for number, episode in (("one", 1), ("two", 2), ("three", 3))
            ],
        }
    )


def make_delta(contract: StoryContract) -> EpisodeStateDelta:
    fact_ids = [fact.fact_id for fact in contract.facts]
    return EpisodeStateDelta(
        episode_number=1,
        contract_sha256=story_contract_sha256(contract),
        established_fact_ids=fact_ids,
        knowledge_gains=[{"character_id": "lin_lan", "fact_ids": fact_ids}],
        introduced_clue_ids=[],
        resolved_clue_ids=[],
        satisfied_obligation_ids=["episode_one_obligation"],
        evidence=[
            {"target_id": fact_id, "excerpt": _excerpt_for_fact(fact_id)} for fact_id in fact_ids
        ]
        + [
            {
                "target_id": "episode_one_obligation",
                "excerpt": "门后传来第二次敲击",
            }
        ],
        handoff="林岚停在门前。",
    )


def test_episode_delta_contract_binding_replaces_cumulative_model_metadata() -> None:
    contract = make_sparse_knowledge_contract(
        [
            {
                "episode_number": 2,
                "character_id": "alice",
                "known_fact_ids": ["fact_one", "fact_two"],
            },
            {
                "episode_number": 2,
                "character_id": "bob",
                "known_fact_ids": ["fact_two"],
            },
        ]
    )
    contract_hash = story_contract_sha256(contract)
    prior = SeriesState(
        contract_sha256=contract_hash,
        locked_through_episode=1,
        established_fact_ids=["fact_one"],
        character_knowledge=[
            {"character_id": "alice", "known_fact_ids": ["fact_one"]},
            {"character_id": "bob", "known_fact_ids": []},
        ],
    )
    model_delta = EpisodeStateDelta(
        episode_number=2,
        contract_sha256=contract_hash,
        established_fact_ids=["fact_one", "fact_two", "fact_three"],
        knowledge_gains=[{"character_id": "alice", "fact_ids": ["fact_one", "fact_two"]}],
        introduced_clue_ids=["invented_clue"],
        resolved_clue_ids=["invented_clue"],
        satisfied_obligation_ids=["obligation_1", "obligation_2"],
        evidence=[{"target_id": "fact_two", "excerpt": "事实证据"}],
        handoff="第二集交接",
    )

    bound = bind_episode_delta_to_contract(
        contract=contract,
        prior_state=prior,
        delta=model_delta,
    )

    assert bound.established_fact_ids == ["fact_two"]
    assert [item.model_dump(mode="json") for item in bound.knowledge_gains] == [
        {"character_id": "alice", "fact_ids": ["fact_two"]},
        {"character_id": "bob", "fact_ids": ["fact_two"]},
    ]
    assert bound.introduced_clue_ids == []
    assert bound.resolved_clue_ids == []
    assert bound.satisfied_obligation_ids == ["obligation_2"]
    assert bound.evidence == model_delta.evidence
    assert bound.handoff == model_delta.handoff


def _excerpt_for_fact(fact_id: str) -> str:
    return {
        "arrival_date": "2015-08-12",
        "meeting_time": "21:40",
        "locked_duration": "80分钟",
    }.get(fact_id, "事实证据")


def test_story_contract_allows_same_number_with_distinct_kinds() -> None:
    same_value = [
        {
            "fact_id": "current_age",
            "subject": "林汐",
            "predicate": "当前年龄",
            "kind": "count",
            "value": "26",
            "unit": "years",
            "first_revealed_episode": 1,
        },
        {
            "fact_id": "elapsed_years",
            "subject": "旧案",
            "predicate": "距今年数",
            "kind": "duration",
            "value": "26",
            "unit": "years",
            "first_revealed_episode": 1,
        },
    ]

    contract = make_contract(numeric_facts=same_value)

    assert [(fact.value, fact.unit, fact.kind) for fact in contract.facts] == [
        ("26", "years", "count"),
        ("26", "years", "duration"),
    ]


def test_story_contract_completes_fully_sparse_knowledge_matrix() -> None:
    contract = make_sparse_knowledge_contract([])

    assert [
        (state.episode_number, state.character_id, state.known_fact_ids)
        for state in contract.knowledge_states
    ] == [
        (1, "alice", ["fact_one"]),
        (1, "bob", []),
        (2, "alice", ["fact_one"]),
        (2, "bob", []),
        (3, "alice", ["fact_one"]),
        (3, "bob", []),
    ]
    dumped = contract.model_dump(mode="json")
    assert StoryContract.model_validate(dumped).model_dump(mode="json") == dumped


def test_story_contract_completes_partially_sparse_knowledge_matrix() -> None:
    contract = make_sparse_knowledge_contract(
        [
            {
                "episode_number": 2,
                "character_id": "alice",
                "known_fact_ids": ["fact_one", "fact_two"],
            },
            {
                "episode_number": 3,
                "character_id": "bob",
                "known_fact_ids": ["fact_three"],
            },
        ]
    )

    assert [
        (state.episode_number, state.character_id, state.known_fact_ids)
        for state in contract.knowledge_states
    ] == [
        (1, "alice", ["fact_one"]),
        (1, "bob", []),
        (2, "alice", ["fact_one", "fact_two"]),
        (2, "bob", []),
        (3, "alice", ["fact_one", "fact_two"]),
        (3, "bob", ["fact_three"]),
    ]


def test_story_contract_canonicalizes_initial_knowledge_fact_order() -> None:
    payload = make_sparse_knowledge_contract([]).model_dump(mode="json")
    payload["characters"][0]["initial_known_fact_ids"] = ["fact_two", "fact_one"]
    payload["knowledge_states"] = []
    reversed_payload = json.loads(json.dumps(payload))
    reversed_payload["characters"][0]["initial_known_fact_ids"] = ["fact_one", "fact_two"]

    contract = StoryContract.model_validate(payload)
    reversed_contract = StoryContract.model_validate(reversed_payload)

    assert contract.characters[0].initial_known_fact_ids == ["fact_one", "fact_two"]
    assert story_contract_sha256(contract) == story_contract_sha256(reversed_contract)


def test_story_contract_canonicalizes_shuffled_dense_knowledge() -> None:
    shuffled_dense = [
        {
            "episode_number": 3,
            "character_id": "bob",
            "known_fact_ids": ["fact_three"],
        },
        {
            "episode_number": 1,
            "character_id": "alice",
            "known_fact_ids": ["fact_one"],
        },
        {
            "episode_number": 2,
            "character_id": "bob",
            "known_fact_ids": [],
        },
        {
            "episode_number": 3,
            "character_id": "alice",
            "known_fact_ids": ["fact_two", "fact_one"],
        },
        {
            "episode_number": 1,
            "character_id": "bob",
            "known_fact_ids": [],
        },
        {
            "episode_number": 2,
            "character_id": "alice",
            "known_fact_ids": ["fact_one", "fact_two"],
        },
    ]

    contract = make_sparse_knowledge_contract(shuffled_dense)
    dumped = contract.model_dump(mode="json")
    canonical_knowledge = [
        {
            "episode_number": 1,
            "character_id": "alice",
            "known_fact_ids": ["fact_one"],
        },
        {"episode_number": 1, "character_id": "bob", "known_fact_ids": []},
        {
            "episode_number": 2,
            "character_id": "alice",
            "known_fact_ids": ["fact_one", "fact_two"],
        },
        {"episode_number": 2, "character_id": "bob", "known_fact_ids": []},
        {
            "episode_number": 3,
            "character_id": "alice",
            "known_fact_ids": ["fact_one", "fact_two"],
        },
        {
            "episode_number": 3,
            "character_id": "bob",
            "known_fact_ids": ["fact_three"],
        },
    ]
    round_tripped = StoryContract.model_validate(dumped)

    assert dumped["knowledge_states"] == canonical_knowledge
    assert round_tripped.model_dump(mode="json") == dumped
    assert story_contract_sha256(round_tripped) == story_contract_sha256(contract)


def test_story_contract_rejects_duplicate_sparse_knowledge_pair() -> None:
    duplicate = {
        "episode_number": 2,
        "character_id": "alice",
        "known_fact_ids": ["fact_one", "fact_two"],
    }

    with pytest.raises(ValidationError, match="Duplicate knowledge state"):
        make_sparse_knowledge_contract([duplicate, duplicate])


def test_story_contract_rejects_evidence_target_id_collision() -> None:
    payload = make_contract().model_dump(mode="json")
    payload["clues"].append(
        {
            "clue_id": "arrival_date",
            "description": "与事实 ID 冲突的线索",
            "introduced_episode": 1,
            "explained_episode": 1,
            "callback_episode": None,
            "introduction_is_visible_or_audible": True,
        }
    )

    with pytest.raises(ValidationError, match="must be globally unique"):
        StoryContract.model_validate(payload)


def test_episode_state_delta_rejects_duplicate_evidence_targets() -> None:
    payload = make_delta(make_contract()).model_dump(mode="json")
    payload["evidence"].append(dict(payload["evidence"][0]))

    with pytest.raises(ValidationError, match="evidence target IDs must be unique"):
        EpisodeStateDelta.model_validate(payload)


def test_story_contract_rejects_sparse_knowledge_regression() -> None:
    with pytest.raises(ValidationError, match="cannot silently disappear"):
        make_sparse_knowledge_contract(
            [
                {
                    "episode_number": 1,
                    "character_id": "alice",
                    "known_fact_ids": ["fact_one", "fact_two"],
                },
                {
                    "episode_number": 3,
                    "character_id": "alice",
                    "known_fact_ids": ["fact_one"],
                },
            ]
        )


@pytest.mark.parametrize(
    ("knowledge_state", "message"),
    [
        (
            {
                "episode_number": 1,
                "character_id": "outsider",
                "known_fact_ids": [],
            },
            "references unknown identifiers",
        ),
        (
            {
                "episode_number": 1,
                "character_id": "alice",
                "known_fact_ids": ["unknown_fact"],
            },
            "references unknown identifiers",
        ),
        (
            {
                "episode_number": 4,
                "character_id": "alice",
                "known_fact_ids": ["fact_one"],
            },
            "episode exceeds the contract episode count",
        ),
    ],
)
def test_story_contract_rejects_invalid_sparse_knowledge_entries(
    knowledge_state: dict,
    message: str,
) -> None:
    with pytest.raises(ValidationError, match=message):
        make_sparse_knowledge_contract([knowledge_state])


def test_story_contract_hash_and_markdown_projection_are_deterministic() -> None:
    contract = make_contract()
    contract_hash = story_contract_sha256(contract)

    assert story_contract_sha256(StoryContract.model_validate_json(contract.model_dump_json())) == (
        contract_hash
    )
    markdown = render_story_contract_markdown(contract, contract_hash)
    assert f"SHA-256: `{contract_hash}`" in markdown
    assert "80 分钟 (duration)" in markdown


def test_episode_validation_rejects_uncontracted_time_unknown_speaker_and_missing_evidence() -> (
    None
):
    contract = make_contract()
    contract_hash = story_contract_sha256(contract)
    prior = initial_series_state(contract, contract_hash)
    delta = make_delta(contract).model_copy(
        update={
            "evidence": [
                item for item in make_delta(contract).evidence if item.target_id != "meeting_time"
            ]
        }
    )
    content = (
        "林岚：我在2015-08-12的21:40到这里，监控持续80分钟。\n"
        "陌生人：其实是22:12。\n"
        "门后传来第二次敲击"
    )

    issues = validate_episode_candidate(
        contract=contract,
        contract_sha256=contract_hash,
        prior_state=prior,
        content=content,
        delta=delta,
    )

    assert {issue.code for issue in issues} >= {
        "evidence_coverage_mismatch",
        "uncontracted_time",
        "unknown_speaker",
    }


@pytest.mark.parametrize(
    "heading",
    [
        "场景一：旧屋",
        "第2场：雨夜",
        "第一集：归来",
        "第12集：终局",
        "EP01：归来",
    ],
)
def test_episode_validation_does_not_treat_scene_heading_as_speaker(heading: str) -> None:
    contract = make_contract()
    contract_hash = story_contract_sha256(contract)
    issues = validate_episode_candidate(
        contract=contract,
        contract_sha256=contract_hash,
        prior_state=initial_series_state(contract, contract_hash),
        content=(
            f"{heading}\n林岚：我在2015-08-12的21:40到这里，监控持续80分钟。\n门后传来第二次敲击"
        ),
        delta=make_delta(contract),
    )

    assert "unknown_speaker" not in {issue.code for issue in issues}


def test_episode_validation_accepts_exact_annotated_character_speaker_label() -> None:
    payload = make_contract().model_dump(mode="json")
    payload["characters"][0]["name"] = "林岚（主角）"
    contract = StoryContract.model_validate(payload)
    contract_hash = story_contract_sha256(contract)
    issues = validate_episode_candidate(
        contract=contract,
        contract_sha256=contract_hash,
        prior_state=initial_series_state(contract, contract_hash),
        content=(
            "林岚（主角）：我在2015-08-12的21:40到这里，监控持续80分钟。\n"
            "林岚（主角）（低声）：门后又响了一次。\n"
            "门后传来第二次敲击"
        ),
        delta=make_delta(contract),
    )

    assert "unknown_speaker" not in {issue.code for issue in issues}


@pytest.mark.parametrize(
    "format_line",
    [
        "特写：物证袋上的编号被胶带遮住。",
        "人物：林岚站在长条桌前。",
        "内景：审讯室。",
        "外景：旧屋门前。",
        "近景：林岚的手指。",
        "远景：码头渐渐隐去。",
        "全景：物证室一片安静。",
        "中景：林岚合上笔记本。",
        "画外音：门后传来一声轻响。",
        "独白：我不能再等了。",
        "闪回：雨夜的旧屋。",
        "转场：翌日清晨。",
        "镜头贴着装订线看进去：纸页边缘有撕裂痕迹。",
        "长条桌上物证袋排成一列：林岚没有伸手。",
        "所以结论我写下来：她把笔记本合上。",
        "我一直照着一个假设查：林岚望向窗外。",
    ],
)
def test_episode_validation_does_not_treat_format_or_narration_as_speaker(
    format_line: str,
) -> None:
    contract = make_contract()
    contract_hash = story_contract_sha256(contract)
    issues = validate_episode_candidate(
        contract=contract,
        contract_sha256=contract_hash,
        prior_state=initial_series_state(contract, contract_hash),
        content=(
            f"{format_line}\n"
            "林岚：我在2015-08-12的21:40到这里，监控持续80分钟。\n"
            "门后传来第二次敲击"
        ),
        delta=make_delta(contract),
    )

    assert "unknown_speaker" not in {issue.code for issue in issues}


@pytest.mark.parametrize(
    "invalid_line",
    [
        (
            "陌生人（低声）：发生日期是二〇一四年十月三日，官方时间是二十二点整，"
            "距今时长十一年，当前年龄十三岁。"
        ),
        (
            "陌生人（低声）：发生日期是2014年10月3日，官方时间是22点整，"
            "距今时长十一年，当前年龄十三岁。"
        ),
    ],
)
def test_episode_validation_normalizes_chinese_continuity_values_and_parenthetical_speakers(
    invalid_line: str,
) -> None:
    contract = make_contract(
        numeric_facts=[
            {
                "fact_id": "event_date",
                "subject": "旧案",
                "predicate": "发生日期",
                "kind": "date",
                "value": "2015-08-12",
                "first_revealed_episode": 1,
            },
            {
                "fact_id": "official_time",
                "subject": "旧案",
                "predicate": "官方时间",
                "kind": "time",
                "value": "22:50",
                "first_revealed_episode": 1,
            },
            {
                "fact_id": "elapsed_years",
                "subject": "旧案",
                "predicate": "距今时长",
                "kind": "duration",
                "value": "10",
                "unit": "年",
                "first_revealed_episode": 1,
            },
            {
                "fact_id": "current_age",
                "subject": "林岚",
                "predicate": "当前年龄",
                "kind": "count",
                "value": "26",
                "unit": "岁",
                "first_revealed_episode": 1,
            },
        ]
    )
    contract_hash = story_contract_sha256(contract)
    prior = initial_series_state(contract, contract_hash)
    delta = EpisodeStateDelta(
        episode_number=1,
        contract_sha256=contract_hash,
        established_fact_ids=[fact.fact_id for fact in contract.facts],
        knowledge_gains=[
            {
                "character_id": "lin_lan",
                "fact_ids": [fact.fact_id for fact in contract.facts],
            }
        ],
        satisfied_obligation_ids=["episode_one_obligation"],
        evidence=[
            {"target_id": "event_date", "excerpt": "2015-08-12"},
            {"target_id": "official_time", "excerpt": "22:50"},
            {"target_id": "elapsed_years", "excerpt": "10年"},
            {"target_id": "current_age", "excerpt": "26岁"},
            {"target_id": "episode_one_obligation", "excerpt": "门后传来第二次敲击"},
        ],
        handoff="林岚停在门前。",
    )
    content = f"林岚：锁定值是2015-08-12、22:50、10年、26岁。\n{invalid_line}\n门后传来第二次敲击"

    issues = validate_episode_candidate(
        contract=contract,
        contract_sha256=contract_hash,
        prior_state=prior,
        content=content,
        delta=delta,
    )

    codes = [issue.code for issue in issues]
    assert "unknown_speaker" in codes
    assert codes.count("uncontracted_time") == 2
    assert codes.count("uncontracted_number") == 2


def test_episode_validation_accepts_chinese_equivalents_of_locked_values() -> None:
    contract = make_contract(
        numeric_facts=[
            {
                "fact_id": "event_date",
                "subject": "旧案",
                "predicate": "发生日期",
                "kind": "date",
                "value": "2015-08-12",
                "first_revealed_episode": 1,
            },
            {
                "fact_id": "official_time",
                "subject": "旧案",
                "predicate": "官方时间",
                "kind": "time",
                "value": "22:50",
                "first_revealed_episode": 1,
            },
            {
                "fact_id": "elapsed_years",
                "subject": "旧案",
                "predicate": "距今时长",
                "kind": "duration",
                "value": "10",
                "unit": "年",
                "first_revealed_episode": 1,
            },
            {
                "fact_id": "current_age",
                "subject": "林岚",
                "predicate": "当前年龄",
                "kind": "count",
                "value": "26",
                "unit": "岁",
                "first_revealed_episode": 1,
            },
        ]
    )
    contract_hash = story_contract_sha256(contract)
    prior = initial_series_state(contract, contract_hash)
    delta = EpisodeStateDelta(
        episode_number=1,
        contract_sha256=contract_hash,
        established_fact_ids=[fact.fact_id for fact in contract.facts],
        knowledge_gains=[
            {
                "character_id": "lin_lan",
                "fact_ids": [fact.fact_id for fact in contract.facts],
            }
        ],
        satisfied_obligation_ids=["episode_one_obligation"],
        evidence=[
            {"target_id": "event_date", "excerpt": "二〇一五年八月十二日"},
            {"target_id": "official_time", "excerpt": "二十二点五十分"},
            {"target_id": "elapsed_years", "excerpt": "十年"},
            {"target_id": "current_age", "excerpt": "二十六岁"},
            {"target_id": "episode_one_obligation", "excerpt": "门后传来第二次敲击"},
        ],
        handoff="林岚停在门前。",
    )
    content = (
        "林岚（低声）：旧案发生在二〇一五年八月十二日二十二点五十分。\n"
        "林岚：那是十年前，我现在二十六岁。\n"
        "门后传来第二次敲击"
    )

    issues = validate_episode_candidate(
        contract=contract,
        contract_sha256=contract_hash,
        prior_state=prior,
        content=content,
        delta=delta,
    )

    assert not {
        "uncontracted_time",
        "uncontracted_number",
        "unknown_speaker",
    } & {issue.code for issue in issues}


@pytest.mark.parametrize(
    "natural_expression",
    [
        "林岚：我想问你一件事。",
        "两人并肩站在雨里。",
        "林岚：咖啡只要三分甜。",
        "林岚：我一天都没安心过。",
        "林岚：哪一次不是我们先到？",
    ],
)
def test_episode_validation_leaves_unlocked_units_to_semantic_review(
    natural_expression: str,
) -> None:
    contract = make_contract()
    contract_hash = story_contract_sha256(contract)
    prior = initial_series_state(contract, contract_hash)
    delta = make_delta(contract)
    content = (
        "林岚：我在2015-08-12的21:40到这里，监控持续80分钟。\n"
        f"{natural_expression}\n"
        "门后传来第二次敲击"
    )

    issues = validate_episode_candidate(
        contract=contract,
        contract_sha256=contract_hash,
        prior_state=prior,
        content=content,
        delta=delta,
    )

    assert "uncontracted_number" not in {issue.code for issue in issues}


def test_episode_validation_rejects_wrong_value_for_locked_numeric_unit() -> None:
    contract = make_contract()
    contract_hash = story_contract_sha256(contract)
    prior = initial_series_state(contract, contract_hash)
    delta = make_delta(contract)
    content = (
        "林岚：我在2015-08-12的21:40到这里，监控持续80分钟。\n"
        "林岚：另一份记录却写着监控记录的持续时间是81分钟。\n"
        "门后传来第二次敲击"
    )

    issues = validate_episode_candidate(
        contract=contract,
        contract_sha256=contract_hash,
        prior_state=prior,
        content=content,
        delta=delta,
    )

    assert "uncontracted_number" in {issue.code for issue in issues}


def test_episode_validation_rejects_wrong_cross_unit_duration() -> None:
    contract = make_contract()
    contract_hash = story_contract_sha256(contract)
    prior = initial_series_state(contract, contract_hash)
    delta = make_delta(contract)
    content = (
        "林岚：我在2015-08-12的21:40到这里，监控持续80分钟。\n"
        "林岚：另一份记录却说监控记录的持续时间是2小时。\n"
        "门后传来第二次敲击"
    )

    issues = validate_episode_candidate(
        contract=contract,
        contract_sha256=contract_hash,
        prior_state=prior,
        content=content,
        delta=delta,
    )

    assert "uncontracted_number" in {issue.code for issue in issues}


def test_episode_validation_accepts_equivalent_cross_unit_duration() -> None:
    contract = make_contract(
        numeric_facts=[
            {
                "fact_id": "locked_duration",
                "subject": "监控记录",
                "predicate": "持续时间",
                "kind": "duration",
                "value": "120",
                "unit": "分钟",
                "first_revealed_episode": 1,
            }
        ]
    )
    contract_hash = story_contract_sha256(contract)
    prior = initial_series_state(contract, contract_hash)
    delta = make_delta(contract)
    delta.evidence[0].excerpt = "持续时间是2小时"
    content = "林岚：监控记录的持续时间是2小时。\n门后传来第二次敲击"

    issues = validate_episode_candidate(
        contract=contract,
        contract_sha256=contract_hash,
        prior_state=prior,
        content=content,
        delta=delta,
    )

    assert "uncontracted_number" not in {issue.code for issue in issues}


def test_episode_validation_rejects_wrong_standalone_calendar_year() -> None:
    contract = make_contract()
    contract_hash = story_contract_sha256(contract)
    prior = initial_series_state(contract, contract_hash)
    delta = make_delta(contract)
    content = (
        "林岚：我在2015-08-12的21:40到这里，监控持续80分钟。\n"
        "林岚：另一份记录却说我的回乡日期是在2016年。\n"
        "门后传来第二次敲击"
    )

    issues = validate_episode_candidate(
        contract=contract,
        contract_sha256=contract_hash,
        prior_state=prior,
        content=content,
        delta=delta,
    )

    assert "uncontracted_number" in {issue.code for issue in issues}


@pytest.mark.parametrize(
    "unrelated_number",
    [
        "林岚：等你解释这件事，我只在门口等了2分钟。",
        "林岚：看完监控后，我又在码头等了2小时。",
        "林岚：石碑记着这座岛已有2000年历史。",
    ],
)
def test_episode_validation_leaves_unrelated_cross_unit_numbers_to_semantic_review(
    unrelated_number: str,
) -> None:
    contract = make_contract()
    contract_hash = story_contract_sha256(contract)
    prior = initial_series_state(contract, contract_hash)
    delta = make_delta(contract)
    content = (
        "林岚：我在2015-08-12的21:40到这里，监控持续80分钟。\n"
        f"{unrelated_number}\n"
        "门后传来第二次敲击"
    )

    issues = validate_episode_candidate(
        contract=contract,
        contract_sha256=contract_hash,
        prior_state=prior,
        content=content,
        delta=delta,
    )

    assert "uncontracted_number" not in {issue.code for issue in issues}


def test_episode_validation_scopes_numeric_predicate_to_the_same_clause() -> None:
    contract = make_contract(
        numeric_facts=[
            {
                "fact_id": "current_age",
                "subject": "林岚",
                "predicate": "年龄",
                "kind": "count",
                "value": "28",
                "unit": "岁",
                "first_revealed_episode": 1,
            },
            {
                "fact_id": "father_age",
                "subject": "林岚父亲",
                "predicate": "享年",
                "kind": "count",
                "value": "58",
                "unit": "岁",
                "first_revealed_episode": 1,
            },
        ]
    )
    contract_hash = story_contract_sha256(contract)
    prior = initial_series_state(contract, contract_hash)
    delta = make_delta(contract)
    content = "林岚：事实证据。她二十八岁，眉眼清冷；父亲享年五十八岁。\n门后传来第二次敲击"

    issues = validate_episode_candidate(
        contract=contract,
        contract_sha256=contract_hash,
        prior_state=prior,
        content=content,
        delta=delta,
    )

    assert "uncontracted_number" not in {issue.code for issue in issues}


def test_episode_validation_allows_chinese_time_locked_as_text_fact() -> None:
    contract = make_contract(
        numeric_facts=[
            {
                "fact_id": "watch_time",
                "subject": "旧怀表",
                "predicate": "表针停在九点十七分",
                "kind": "text",
                "value": "九点十七分",
                "first_revealed_episode": 1,
            }
        ]
    )
    contract_hash = story_contract_sha256(contract)
    prior = initial_series_state(contract, contract_hash)
    delta = make_delta(contract)
    content = "林岚：事实证据，表针停在九点十七分。\n门后传来第二次敲击"

    issues = validate_episode_candidate(
        contract=contract,
        contract_sha256=contract_hash,
        prior_state=prior,
        content=content,
        delta=delta,
    )

    assert "uncontracted_time" not in {issue.code for issue in issues}


def test_episode_validation_accepts_unqualified_twelve_hour_alias_for_locked_pm_time() -> None:
    contract = make_contract()
    contract_hash = story_contract_sha256(contract)
    prior = initial_series_state(contract, contract_hash)
    delta = make_delta(contract)
    delta.evidence[1].excerpt = "九点四十分"
    content = "林岚：我在2015-08-12九点四十分到这里，监控持续80分钟。\n门后传来第二次敲击"

    issues = validate_episode_candidate(
        contract=contract,
        contract_sha256=contract_hash,
        prior_state=prior,
        content=content,
        delta=delta,
    )

    assert "uncontracted_time" not in {issue.code for issue in issues}


def test_episode_validation_keeps_explicit_morning_distinct_from_locked_pm_time() -> None:
    contract = make_contract()
    contract_hash = story_contract_sha256(contract)
    prior = initial_series_state(contract, contract_hash)
    delta = make_delta(contract)
    delta.evidence[1].excerpt = "上午九点四十分"
    content = "林岚：我在2015-08-12上午九点四十分到这里，监控持续80分钟。\n门后传来第二次敲击"

    issues = validate_episode_candidate(
        contract=contract,
        contract_sha256=contract_hash,
        prior_state=prior,
        content=content,
        delta=delta,
    )

    assert "uncontracted_time" in {issue.code for issue in issues}


def test_episode_validation_keeps_numeric_am_time_distinct_from_locked_pm_time() -> None:
    contract = make_contract()
    contract_hash = story_contract_sha256(contract)
    prior = initial_series_state(contract, contract_hash)
    delta = make_delta(contract)
    delta.evidence[1].excerpt = "09:40"
    content = "林岚：我在2015-08-12的09:40到这里，监控持续80分钟。\n门后传来第二次敲击"

    issues = validate_episode_candidate(
        contract=contract,
        contract_sha256=contract_hash,
        prior_state=prior,
        content=content,
        delta=delta,
    )

    assert "uncontracted_time" in {issue.code for issue in issues}


def test_episode_validation_rejects_knowledge_gain_outside_locked_cast() -> None:
    contract = make_contract()
    contract_hash = story_contract_sha256(contract)
    prior = initial_series_state(contract, contract_hash)
    content = "林岚：日期是2015-08-12，时间是21:40，记录持续80分钟。\n门后传来第二次敲击"
    delta_payload = make_delta(contract).model_dump(mode="json")
    delta_payload["knowledge_gains"].append(
        {"character_id": "outsider", "fact_ids": ["arrival_date"]}
    )
    delta = EpisodeStateDelta.model_validate(delta_payload)

    issues = validate_episode_candidate(
        contract=contract,
        contract_sha256=contract_hash,
        prior_state=prior,
        content=content,
        delta=delta,
    )

    assert "unknown_knowledge_character" in {issue.code for issue in issues}
    with pytest.raises(ContinuityViolation, match="continuity validation failed"):
        build_episode_lock(
            contract=contract,
            contract_sha256=contract_hash,
            prior_state=prior,
            content=content,
            delta=delta,
            semantic_review=SemanticReview(passed=True, evidence="独立审查通过", issues=[]),
            repair_rounds=0,
        )


def test_valid_episode_lock_folds_exact_knowledge_and_hashes() -> None:
    contract = make_contract()
    contract_hash = story_contract_sha256(contract)
    prior = initial_series_state(contract, contract_hash)
    content = "林岚：日期是2015-08-12，时间是21:40，记录持续80分钟。\n门后传来第二次敲击"
    delta = make_delta(contract)
    review = SemanticReview(passed=True, evidence="独立审查通过", issues=[])

    episode_lock = build_episode_lock(
        contract=contract,
        contract_sha256=contract_hash,
        prior_state=prior,
        content=content,
        delta=delta,
        semantic_review=review,
        repair_rounds=1,
    )

    assert episode_lock.contract_sha256 == contract_hash
    assert episode_lock.series_state.locked_through_episode == 1
    assert episode_lock.series_state.character_knowledge[0].known_fact_ids == [
        "arrival_date",
        "locked_duration",
        "meeting_time",
    ]
    assert episode_lock.repair_rounds == 1
