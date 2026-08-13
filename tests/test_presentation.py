from datetime import UTC, datetime
from uuid import uuid4

from pengine.presentation import compile_delivery_presentation, recover_delivery_presentation
from pengine.schemas import ContentPackage, EpisodeDraft, EpisodePlan


def test_compiler_builds_structured_navigation_without_changing_sources() -> None:
    content = ContentPackage(
        story_outline="故事核\n归乡。\n冲突\n旧照被撕毁。",
        character_biographies="林岚\n返乡者。\n周伯\n守屋人。",
        relationship_logic="林岚与周伯\n互相隐瞒。\n林岚与母亲\n多年隔阂。",
        episode_outline="第一集计划\n第二集计划",
        episode_scripts="第一集剧本\n第二集剧本",
    )
    presentation = compile_delivery_presentation(
        creation_id=uuid4(),
        run_kind="initial",
        content=content,
        story_hints=[
            {"label": "故事核", "anchor": "故事核", "level": 1},
            {"label": "冲突", "anchor": "冲突", "level": 1},
        ],
        character_hints=[
            {"label": "林岚", "anchor": "林岚", "group": "core"},
            {"label": "周伯", "anchor": "周伯", "group": "supporting"},
        ],
        relationship_hints=[
            {"label": "林岚与周伯", "anchor": "林岚与周伯", "group": "primary"},
            {"label": "林岚与母亲", "anchor": "林岚与母亲", "group": "supporting"},
        ],
        episode_plans=[
            EpisodePlan(episode_number=1, plan="第一集计划"),
            EpisodePlan(episode_number=2, plan="第二集计划"),
        ],
        episode_drafts=[
            EpisodeDraft(
                episode_number=1,
                content="第一集剧本",
                content_sha256="a" * 64,
                completed_at=datetime.now(UTC),
            ),
            EpisodeDraft(
                episode_number=2,
                content="第二集剧本",
                content_sha256="b" * 64,
                completed_at=datetime.now(UTC),
            ),
        ],
    )

    assert presentation.status == "complete"
    assert [item.label for item in presentation.story_outline.sections] == ["故事核", "冲突"]
    assert [item.label for item in presentation.character_biographies.characters] == [
        "林岚",
        "周伯",
    ]
    assert [item.episode_number for item in presentation.episode_scripts.episodes] == [1, 2]
    assert presentation.episode_scripts.source_text == content.episode_scripts


def test_first_structured_item_preserves_text_before_its_anchor() -> None:
    content = ContentPackage(
        story_outline="前言\n故事核\n归乡。",
        character_biographies="人物",
        relationship_logic="关系",
        episode_outline="分集",
        episode_scripts="剧本",
    )

    presentation = compile_delivery_presentation(
        creation_id=uuid4(),
        run_kind="initial",
        content=content,
        story_hints=[{"label": "故事核", "anchor": "故事核", "level": 1}],
    )

    assert presentation.story_outline.mode == "structured"
    assert presentation.story_outline.sections[0].content == content.story_outline


def test_invalid_anchor_degrades_only_affected_artifact() -> None:
    content = ContentPackage(
        story_outline="完整故事",
        character_biographies="林岚\n返乡者。",
        relationship_logic="人物关系",
        episode_outline="分集大纲",
        episode_scripts="分集剧本",
    )
    presentation = compile_delivery_presentation(
        creation_id=uuid4(),
        run_kind="revision",
        content=content,
        story_hints=[{"label": "不存在", "anchor": "不存在", "level": 1}],
        character_hints=[{"label": "林岚", "anchor": "林岚", "group": "core"}],
    )

    assert presentation.status == "partial"
    assert presentation.story_outline.mode == "source"
    assert presentation.story_outline.sections == []
    assert presentation.story_outline.source_text == "完整故事"
    assert presentation.character_biographies.mode == "structured"


def test_missing_manifest_inputs_return_complete_source_fallback() -> None:
    content = ContentPackage(
        story_outline="故事",
        character_biographies="人物",
        relationship_logic="关系",
        episode_outline="分集",
        episode_scripts="剧本",
    )
    presentation = compile_delivery_presentation(
        creation_id=uuid4(), run_kind="initial", content=content
    )

    assert presentation.status == "source"
    assert presentation.episode_scripts.episodes == []
    assert presentation.relationship_logic.source_text == "关系"


def test_recovery_downgrades_only_artifact_with_corrupted_hash() -> None:
    creation_id = uuid4()
    content = ContentPackage(
        story_outline="故事核\n归乡。",
        character_biographies="林岚\n返乡者。",
        relationship_logic="林岚与周伯\n互相隐瞒。",
        episode_outline="第一集计划\n第二集计划",
        episode_scripts="第一集剧本\n第二集剧本",
    )
    plans = [
        EpisodePlan(episode_number=1, plan="第一集计划"),
        EpisodePlan(episode_number=2, plan="第二集计划"),
    ]
    drafts = [
        EpisodeDraft(
            episode_number=number,
            content=f"第{'一' if number == 1 else '二'}集剧本",
            content_sha256="a" * 64,
            completed_at=datetime.now(UTC),
        )
        for number in (1, 2)
    ]
    stored = compile_delivery_presentation(
        creation_id=creation_id,
        run_kind="initial",
        content=content,
        story_hints=[{"label": "故事核", "anchor": "故事核", "level": 1}],
        character_hints=[{"label": "林岚", "anchor": "林岚", "group": "core"}],
        relationship_hints=[{"label": "林岚与周伯", "anchor": "林岚与周伯", "group": "primary"}],
        episode_plans=plans,
        episode_drafts=drafts,
    ).model_dump(mode="json")
    stored["character_biographies"]["characters"][0]["content_sha256"] = "0" * 64

    recovered = recover_delivery_presentation(
        raw_manifest=stored,
        creation_id=creation_id,
        run_kind="initial",
        content=content,
    )

    assert recovered.status == "partial"
    assert recovered.character_biographies.mode == "source"
    assert recovered.story_outline.mode == "structured"
    assert recovered.relationship_logic.mode == "structured"
    assert recovered.episode_outline.mode == "structured"
    assert recovered.episode_scripts.mode == "structured"


def test_recovery_downgrades_artifact_with_inconsistent_mode() -> None:
    creation_id = uuid4()
    content = ContentPackage(
        story_outline="故事核\n归乡。",
        character_biographies="林岚\n返乡者。",
        relationship_logic="林岚与周伯\n互相隐瞒。",
        episode_outline="分集",
        episode_scripts="剧本",
    )
    stored = compile_delivery_presentation(
        creation_id=creation_id,
        run_kind="initial",
        content=content,
        story_hints=[{"label": "故事核", "anchor": "故事核", "level": 1}],
        character_hints=[{"label": "林岚", "anchor": "林岚", "group": "core"}],
        relationship_hints=[{"label": "林岚与周伯", "anchor": "林岚与周伯", "group": "primary"}],
    ).model_dump(mode="json")
    stored["character_biographies"]["mode"] = "source"

    recovered = recover_delivery_presentation(
        raw_manifest=stored,
        creation_id=creation_id,
        run_kind="initial",
        content=content,
    )

    assert recovered.status == "partial"
    assert recovered.story_outline.mode == "structured"
    assert recovered.character_biographies.mode == "source"
    assert recovered.relationship_logic.mode == "structured"


def test_missing_or_unknown_manifest_returns_all_source_artifacts() -> None:
    creation_id = uuid4()
    content = ContentPackage(
        story_outline="故事",
        character_biographies="人物",
        relationship_logic="关系",
        episode_outline="第一集计划",
        episode_scripts="第一集剧本",
    )

    for raw_manifest in (None, {"schema_version": 2}):
        recovered = recover_delivery_presentation(
            raw_manifest=raw_manifest,
            creation_id=creation_id,
            run_kind="initial",
            content=content,
        )

        assert recovered.status == "source"
        assert recovered.episode_outline.mode == "source"
        assert recovered.episode_scripts.mode == "source"
