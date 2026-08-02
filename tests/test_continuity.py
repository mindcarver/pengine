from __future__ import annotations

import pytest
from pydantic import ValidationError

from pengine.continuity import (
    ContinuityViolation,
    EpisodeStateDelta,
    SemanticReview,
    StoryContract,
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


def _excerpt_for_fact(fact_id: str) -> str:
    return {
        "arrival_date": "2015-08-12",
        "meeting_time": "21:40",
        "locked_duration": "80分钟",
    }.get(fact_id, "事实证据")


def test_story_contract_rejects_one_number_with_two_semantic_meanings() -> None:
    duplicate_value = [
        {
            "fact_id": "tide_reading",
            "subject": "潮位表",
            "predicate": "读数",
            "kind": "measurement",
            "value": "1378",
            "unit": "毫米",
            "first_revealed_episode": 1,
        },
        {
            "fact_id": "minute_count",
            "subject": "时间换算",
            "predicate": "午夜后分钟数",
            "kind": "duration",
            "value": "1378",
            "unit": "分钟",
            "first_revealed_episode": 1,
        },
    ]

    with pytest.raises(ValidationError, match="different kinds or units"):
        make_contract(numeric_facts=duplicate_value)


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
    "invalid_line",
    [
        "陌生人（低声）：其实是二〇一四年十月三日、二十二点整、十一年、十三岁。",
        "陌生人（低声）：其实是2014年10月3日、22点整、十一年、十三岁。",
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
