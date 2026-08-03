from pengine.language import (
    SIMPLIFIED_CHINESE,
    has_obvious_language_mismatch,
    infer_output_language,
    language_instruction,
)


def test_infer_output_language_locks_chinese_from_story_or_requirements() -> None:
    assert infer_output_language("一个海边故事", "10 episodes") == SIMPLIFIED_CHINESE
    assert infer_output_language("A seaside mystery", "当代背景，10 集") == SIMPLIFIED_CHINESE


def test_infer_output_language_leaves_english_only_request_unlocked() -> None:
    assert infer_output_language("A seaside mystery", "Ten episodes") is None


def test_language_instruction_covers_user_text_and_delegated_tasks() -> None:
    instruction = language_instruction(SIMPLIFIED_CHINESE)

    assert "简体中文" in instruction
    assert "标题" in instruction
    assert "审核证据" in instruction
    assert "稳定 ID" in instruction
    assert "每个委派任务" in instruction
    assert language_instruction(None) == ""


def test_chinese_lock_rejects_output_without_han_characters() -> None:
    assert has_obvious_language_mismatch(
        "Relationship logic for the daughter and her adoptive father.",
        SIMPLIFIED_CHINESE,
    )


def test_chinese_lock_rejects_long_output_dominated_by_english() -> None:
    english_body = "The relationship remains hidden beneath the surface. " * 12

    assert has_obvious_language_mismatch(
        f"人物关系：{english_body}",
        SIMPLIFIED_CHINESE,
    )


def test_chinese_lock_rejects_moderate_english_body_after_chinese_title() -> None:
    english_body = "The relationship stays hidden beneath the surface."

    assert has_obvious_language_mismatch(
        f"人物关系：{english_body}",
        SIMPLIFIED_CHINESE,
    )


def test_chinese_lock_rejects_short_english_sentence_after_chinese_label() -> None:
    assert has_obvious_language_mismatch("审核：The script passes.", SIMPLIFIED_CHINESE)
    assert has_obvious_language_mismatch("审核：Not passed.", SIMPLIFIED_CHINESE)
    assert has_obvious_language_mismatch("结论：Failed.", SIMPLIFIED_CHINESE)
    assert has_obvious_language_mismatch("审核：Fail", SIMPLIFIED_CHINESE)
    assert has_obvious_language_mismatch("结论：Pass", SIMPLIFIED_CHINESE)
    assert has_obvious_language_mismatch("审核：No", SIMPLIFIED_CHINESE)
    assert has_obvious_language_mismatch("not-passed", SIMPLIFIED_CHINESE)
    assert has_obvious_language_mismatch("review-failed", SIMPLIFIED_CHINESE)


def test_chinese_lock_rejects_obvious_traditional_chinese_output() -> None:
    assert has_obvious_language_mismatch("審核證據：劇本未通過", SIMPLIFIED_CHINESE)


def test_chinese_lock_allows_one_verbatim_traditional_name_character() -> None:
    assert not has_obvious_language_mismatch(
        "角色名中的龍字按用户原文保留，其余内容使用简体中文。",
        SIMPLIFIED_CHINESE,
    )


def test_chinese_lock_allows_chinese_with_english_ids_and_proper_names() -> None:
    content = (
        "人物关系以林汐和养父之间的秘密为核心。"
        "角色 ID lin_xi 在 E1 发现照片，lin_jianguo 在 E3 承认说谎。"
        "Beneath the Tide 只是作品的英文副标题，正文仍然使用简体中文。"
    )

    assert not has_obvious_language_mismatch(content, SIMPLIFIED_CHINESE)


def test_chinese_lock_excludes_many_stable_ids_from_language_ratio() -> None:
    content = (
        "契约 knowledge_states 中 lin_mu（ep6）known_fact_ids 含 fact_keep_secret，"
        "candidate_state_delta 亦列为林母 knowledge_gain；但林母仅在场景六出现，"
        "台词仅体现 fact_stay_island，未体现获知不公开真相（fact_keep_secret）。"
    )

    assert not has_obvious_language_mismatch(content, SIMPLIFIED_CHINESE)


def test_chinese_lock_allows_pure_machine_identifiers() -> None:
    assert not has_obvious_language_mismatch("L0-B", SIMPLIFIED_CHINESE)
    assert not has_obvious_language_mismatch("ep_01", SIMPLIFIED_CHINESE)
    assert not has_obvious_language_mismatch("E10", SIMPLIFIED_CHINESE)


def test_chinese_lock_allows_language_neutral_time_and_date_values() -> None:
    assert not has_obvious_language_mismatch("21:40", SIMPLIFIED_CHINESE)
    assert not has_obvious_language_mismatch("2014-07-06", SIMPLIFIED_CHINESE)
    assert not has_obvious_language_mismatch("2024-05-01T10:30:00Z", SIMPLIFIED_CHINESE)
    assert not has_obvious_language_mismatch(
        "2015-08-12T21:40:00+08:00",
        SIMPLIFIED_CHINESE,
    )


def test_chinese_lock_ignores_empty_optional_fields() -> None:
    assert not has_obvious_language_mismatch("   ", SIMPLIFIED_CHINESE)


def test_unlocked_language_does_not_reject_english_output() -> None:
    assert not has_obvious_language_mismatch("English output", None)
