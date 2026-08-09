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


def make_required_callback_case(
    *, callback_episode: int | None = 3, obligation_requires_clue: bool = True
) -> tuple[StoryContract, str, SeriesState, EpisodeStateDelta, str]:
    payload = make_sparse_knowledge_contract([]).model_dump(mode="json")
    payload["clues"] = [
        {
            "clue_id": "callback_tape",
            "description": "第一集出现、第二集解释、第三集回扣的旧磁带",
            "introduced_episode": 1,
            "explained_episode": 2,
            "callback_episode": callback_episode,
            "introduction_is_visible_or_audible": True,
        }
    ]
    if obligation_requires_clue:
        payload["episode_obligations"][2]["required_clue_ids"] = ["callback_tape"]
    contract = StoryContract.model_validate(payload)
    contract_hash = story_contract_sha256(contract)
    prior = SeriesState(
        contract_sha256=contract_hash,
        locked_through_episode=2,
        established_fact_ids=["fact_one", "fact_two"],
        character_knowledge=[
            {"character_id": "alice", "known_fact_ids": ["fact_one"]},
            {"character_id": "bob", "known_fact_ids": []},
        ],
        introduced_clue_ids=["callback_tape"],
        resolved_clue_ids=["callback_tape"],
        handoff="第二集结束。",
    )
    delta = EpisodeStateDelta(
        episode_number=3,
        contract_sha256=contract_hash,
        established_fact_ids=["fact_three"],
        satisfied_obligation_ids=["obligation_3"],
        evidence=[
            {"target_id": "fact_three", "excerpt": "事实three"},
            {"target_id": "callback_tape", "excerpt": "旧磁带再次出现"},
            {"target_id": "obligation_3", "excerpt": "第3集钩子"},
        ],
        handoff="故事完成。",
    )
    content = "阿博：事实three。\n旧磁带再次出现。\n第3集钩子"
    return contract, contract_hash, prior, delta, content


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


def test_episode_validation_does_not_infer_semantics_from_unbound_content() -> None:
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

    codes = {issue.code for issue in issues}
    assert "missing_evidence_targets" in codes
    assert "locked_temporal_evidence_mismatch" not in codes
    assert "unknown_speaker" not in codes


@pytest.mark.parametrize(
    ("callback_episode", "obligation_requires_clue"),
    [(3, False), (2, True)],
)
def test_episode_validation_accepts_required_clue_evidence(
    callback_episode: int, obligation_requires_clue: bool
) -> None:
    contract, contract_hash, prior, delta, content = make_required_callback_case(
        callback_episode=callback_episode,
        obligation_requires_clue=obligation_requires_clue,
    )

    issues = validate_episode_candidate(
        contract=contract,
        contract_sha256=contract_hash,
        prior_state=prior,
        content=content,
        delta=delta,
    )

    assert issues == []


def test_episode_validation_reports_only_missing_evidence_targets() -> None:
    contract, contract_hash, prior, delta, content = make_required_callback_case()
    delta = delta.model_copy(
        update={"evidence": [item for item in delta.evidence if item.target_id != "callback_tape"]}
    )

    issues = validate_episode_candidate(
        contract=contract,
        contract_sha256=contract_hash,
        prior_state=prior,
        content=content,
        delta=delta,
    )

    missing = next(issue for issue in issues if issue.code == "missing_evidence_targets")
    assert missing.contract_refs == ["callback_tape"]
    assert "unexpected_evidence_targets" not in {issue.code for issue in issues}


def test_episode_validation_reports_only_unexpected_evidence_targets() -> None:
    contract, contract_hash, prior, delta, content = make_required_callback_case()
    payload = delta.model_dump(mode="json")
    payload["evidence"].append({"target_id": "stale_target", "excerpt": "旧版本残留说明"})
    delta = EpisodeStateDelta.model_validate(payload)

    issues = validate_episode_candidate(
        contract=contract,
        contract_sha256=contract_hash,
        prior_state=prior,
        content=f"{content}\n旧版本残留说明",
        delta=delta,
    )

    unexpected = next(issue for issue in issues if issue.code == "unexpected_evidence_targets")
    assert unexpected.contract_refs == ["stale_target"]
    assert "missing_evidence_targets" not in {issue.code for issue in issues}


@pytest.mark.parametrize(
    "colon_form_line",
    [
        "场景一：旧屋",
        "第一幕：归来",
        "注：这里保留停顿。",
        "陌生人：其实是22:12。",
        "阿卜杜勒·卡迪尔：门外有人。",
        "ALEXANDER：Cut the feed.",
    ],
)
def test_episode_validation_does_not_parse_colon_forms_as_speakers(
    colon_form_line: str,
) -> None:
    contract = make_contract()
    contract_hash = story_contract_sha256(contract)
    issues = validate_episode_candidate(
        contract=contract,
        contract_sha256=contract_hash,
        prior_state=initial_series_state(contract, contract_hash),
        content=(
            f"{colon_form_line}\n"
            "林岚：我在2015-08-12的21:40到这里，监控持续80分钟。\n"
            "门后传来第二次敲击"
        ),
        delta=make_delta(contract),
    )

    assert "unknown_speaker" not in {issue.code for issue in issues}


def test_episode_validation_allows_screenplay_markers_reasoning_and_code_subject_matter() -> None:
    contract = make_contract()
    contract_hash = story_contract_sha256(contract)
    content = (
        "第一集《纸上的B型》\n"
        '剪辑师：节目字幕里写着 JSON {"episode": 1}。\n'
        "她把时间差默算了两遍：结果是一年零四个月。\n"
        "林岚：我在2015-08-12的21:40到这里，监控持续80分钟。\n"
        "门后传来第二次敲击\n（本集终）"
    )

    issues = validate_episode_candidate(
        contract=contract,
        contract_sha256=contract_hash,
        prior_state=initial_series_state(contract, contract_hash),
        content=content,
        delta=make_delta(contract),
    )

    assert issues == []


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
def test_episode_validation_leaves_disputed_claims_and_speakers_to_semantic_review(
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
    assert "unknown_speaker" not in codes
    assert "locked_temporal_evidence_mismatch" not in codes
    assert "locked_numeric_evidence_mismatch" not in codes


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
        "locked_temporal_evidence_mismatch",
        "locked_numeric_evidence_mismatch",
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

    assert "locked_numeric_evidence_mismatch" not in {issue.code for issue in issues}


def test_episode_validation_rejects_wrong_value_in_locked_numeric_evidence() -> None:
    contract = make_contract()
    contract_hash = story_contract_sha256(contract)
    prior = initial_series_state(contract, contract_hash)
    delta = make_delta(contract)
    next(item for item in delta.evidence if item.target_id == "locked_duration").excerpt = "81分钟"
    content = "林岚：我在2015-08-12的21:40到这里，监控持续81分钟。\n门后传来第二次敲击"

    issues = validate_episode_candidate(
        contract=contract,
        contract_sha256=contract_hash,
        prior_state=prior,
        content=content,
        delta=delta,
    )

    assert "locked_numeric_evidence_mismatch" in {issue.code for issue in issues}


def test_episode_validation_rejects_wrong_cross_unit_locked_numeric_evidence() -> None:
    contract = make_contract()
    contract_hash = story_contract_sha256(contract)
    prior = initial_series_state(contract, contract_hash)
    delta = make_delta(contract)
    next(item for item in delta.evidence if item.target_id == "locked_duration").excerpt = "2小时"
    content = "林岚：我在2015-08-12的21:40到这里，监控持续2小时。\n门后传来第二次敲击"

    issues = validate_episode_candidate(
        contract=contract,
        contract_sha256=contract_hash,
        prior_state=prior,
        content=content,
        delta=delta,
    )

    assert "locked_numeric_evidence_mismatch" in {issue.code for issue in issues}


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

    assert "locked_numeric_evidence_mismatch" not in {issue.code for issue in issues}


@pytest.mark.parametrize(
    "disputed_claim",
    [
        "林岚：另一份记录却写着监控记录的持续时间是81分钟。",
        "林岚：另一份记录却说监控记录的持续时间是2小时。",
        "林岚：另一份记录却说我的回乡日期是在2016年。",
    ],
)
def test_episode_validation_does_not_treat_disputed_values_as_bound_evidence(
    disputed_claim: str,
) -> None:
    contract = make_contract()
    contract_hash = story_contract_sha256(contract)
    prior = initial_series_state(contract, contract_hash)
    delta = make_delta(contract)
    content = (
        f"林岚：我在2015-08-12的21:40到这里，监控持续80分钟。\n{disputed_claim}\n门后传来第二次敲击"
    )

    issues = validate_episode_candidate(
        contract=contract,
        contract_sha256=contract_hash,
        prior_state=prior,
        content=content,
        delta=delta,
    )

    assert not {
        "locked_temporal_evidence_mismatch",
        "locked_numeric_evidence_mismatch",
    } & {issue.code for issue in issues}


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

    assert "locked_numeric_evidence_mismatch" not in {issue.code for issue in issues}


def test_episode_validation_accepts_distinct_numeric_fact_evidence() -> None:
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

    assert "locked_numeric_evidence_mismatch" not in {issue.code for issue in issues}


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

    assert "locked_temporal_evidence_mismatch" not in {issue.code for issue in issues}


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

    assert "locked_temporal_evidence_mismatch" not in {issue.code for issue in issues}


def test_episode_validation_rejects_wrong_explicit_period_in_locked_time_evidence() -> None:
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

    assert "locked_temporal_evidence_mismatch" in {issue.code for issue in issues}


def test_episode_validation_rejects_wrong_numeric_locked_time_evidence() -> None:
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

    assert "locked_temporal_evidence_mismatch" in {issue.code for issue in issues}


@pytest.mark.parametrize(
    ("excerpt", "rejected"),
    [
        ("2015-08-12 21:40", False),
        ("二〇一五年八月十二日晚上九点四十分", False),
        ("2015-08-12 08:00", True),
        ("2015-08-12", True),
        ("就在那一刻", False),
    ],
)
def test_episode_validation_requires_all_parseable_locked_datetime_components(
    excerpt: str,
    rejected: bool,
) -> None:
    contract = make_contract(
        numeric_facts=[
            {
                "fact_id": "arrival_moment",
                "subject": "林岚",
                "predicate": "抵达旧屋时刻",
                "kind": "datetime",
                "value": "2015-08-12T21:40",
                "first_revealed_episode": 1,
            }
        ]
    )
    contract_hash = story_contract_sha256(contract)
    prior = initial_series_state(contract, contract_hash)
    delta = make_delta(contract)
    delta.evidence[0].excerpt = excerpt
    content = f"林岚：抵达旧屋时刻是{excerpt}。\n门后传来第二次敲击"

    issues = validate_episode_candidate(
        contract=contract,
        contract_sha256=contract_hash,
        prior_state=prior,
        content=content,
        delta=delta,
    )

    codes = {issue.code for issue in issues}
    assert ("locked_temporal_evidence_mismatch" in codes) is rejected


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
