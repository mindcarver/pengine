import re
from pathlib import Path

from pengine.personas import (
    LOGICAL_FILES,
    PersonaCatalog,
    extract_l0_variant_ids,
    validate_persona_package,
)

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
        assert package.manifest["schema_version"] == "2.0.0"
        assert set(package.manifest["files"]) == {
            "paradigm",
            "project",
            "l0",
            "soul",
            "l3",
            "l4",
            "l5",
            "l6",
        }
        content = "\n".join(package.text(logical_name) for logical_name, _ in LOGICAL_FILES)
        assert "仅作本地原型临时启用" in content
        assert "非创作者人格定稿" in content
        assert "归属：原型项目方" in content
        assert "原型基线：6 集" in package.text("l4")
        assert "24 集" not in package.text("l4")
        assert "## Soul 状态" in package.text("project")
        assert "## 观察与表达" not in package.text("project")
        assert "## 创作能量" not in package.text("project")
        assert len(package.text("soul")) <= 8_000


def test_shouzhuo_alone_carries_the_confirmed_l0_source() -> None:
    packages = {
        persona_id: validate_persona_package(ROOT / "personas" / persona_id)
        for persona_id in EXPECTED_PERSONAS
    }

    shouzhuo_l0 = packages["shouzhuo"].text("l0")
    shouzhuo_soul = packages["shouzhuo"].text("soul")
    assert extract_l0_variant_ids(shouzhuo_l0) == ("A", "B", "C", "D")
    assert "初心 vs 现实的落差" in shouzhuo_l0
    assert "每集至多一次情绪峰值" in shouzhuo_l0
    assert "3fc9ebd9c0ca293e8979d644ceb72993b216661a354f660f659f4502e799e5bf" in (shouzhuo_soul)
    assert "8ebaca0bf45a6c6e99d52baecd4cbdbccf14d384acfa3bbcdd07b77a94586e07" in (shouzhuo_soul)
    assert re.search(r"\b(?:19|20)\d{2}\b", shouzhuo_soul) is None
    for private_source_fragment in ("出生：", "四柱", "十神", "行运"):
        assert private_source_fragment not in shouzhuo_soul

    for persona_id in EXPECTED_PERSONAS.keys() - {"shouzhuo"}:
        other_l0 = packages[persona_id].text("l0")
        assert extract_l0_variant_ids(other_l0) == ()
        assert "初心 vs 现实的落差" not in other_l0
