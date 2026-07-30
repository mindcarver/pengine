from pathlib import Path

from pengine.personas import LOGICAL_FILES, PersonaCatalog, validate_persona_package

ROOT = Path(__file__).parents[1]
EXPECTED_PERSONAS = {
    "shouzhuo": "守拙 · 正剧",
    "wuzhen": "雾枕 · 悬疑",
    "sanfentian": "三分甜 · 言情",
    "xinggui": "星轨 · 科幻",
}


def test_bundled_prototype_personas_are_valid_and_traceable(tmp_path: Path) -> None:
    catalog = PersonaCatalog(ROOT / "personas", tmp_path / "snapshots")

    summaries = {summary.persona_id: summary for summary in catalog.discover()}

    assert {
        persona_id: summary.display_name for persona_id, summary in summaries.items()
    } == EXPECTED_PERSONAS
    for persona_id in EXPECTED_PERSONAS:
        package = validate_persona_package(ROOT / "personas" / persona_id)
        content = "\n".join(package.text(logical_name) for logical_name, _ in LOGICAL_FILES)
        assert "仅作本地原型临时启用" in content
        assert "非创作者人格定稿" in content
        assert "归属：原型项目方" in content
        assert "原型基线：6 集" in package.text("l4")
        assert "24 集" not in package.text("l4")
