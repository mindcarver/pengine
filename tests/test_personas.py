from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path

import pytest
from persona_factory import (
    LOGICAL_FILES,
    NON_PRODUCTION_CONTENT,
    create_persona_package,
    package_bytes,
)

import pengine.personas as persona_module
from pengine.personas import (
    DEFAULT_SCHEMA_PATH,
    PersonaCatalog,
    PersonaPackageError,
    canonical_package_sha256,
    extract_l0_variant_ids,
    validate_persona_package,
)
from pengine.schemas import InternalStage


def _catalog(tmp_path: Path) -> tuple[PersonaCatalog, Path]:
    source_root = tmp_path / "personas"
    package_dir = create_persona_package(source_root / "active")
    catalog = PersonaCatalog(
        source_root,
        tmp_path / "data" / "snapshots",
        schema_path=DEFAULT_SCHEMA_PATH,
    )
    return catalog, package_dir


def test_discover_returns_only_complete_valid_package(tmp_path: Path) -> None:
    catalog, _ = _catalog(tmp_path)
    create_persona_package(tmp_path / "personas" / "missing", persona_id="missing")
    (tmp_path / "personas" / "missing" / "l6.md").unlink()
    create_persona_package(tmp_path / "personas" / "extra", persona_id="extra")
    (tmp_path / "personas" / "extra" / "notes.txt").write_text("unexpected", encoding="utf-8")

    personas = catalog.discover()

    assert len(personas) == 1
    assert personas[0].persona_id == "test-persona"
    assert personas[0].display_name == "非生产测试人格"
    assert personas[0].version == "2.0.0-test"
    assert len(personas[0].snapshot_sha256) == 64


@pytest.mark.parametrize(
    ("mutate", "expected_code"),
    [
        (
            lambda package: (package / "soul.md").write_bytes(b"\xff"),
            "markdown_not_utf8",
        ),
        (
            lambda package: (package / "soul.md").write_text(
                "# Soul\n\n## 身份\n已改变。\n", encoding="utf-8"
            ),
            "file_hash_mismatch",
        ),
    ],
)
def test_invalid_encoding_or_hash_is_rejected(
    tmp_path: Path,
    mutate: object,
    expected_code: str,
) -> None:
    package = create_persona_package(tmp_path / "persona")
    mutate(package)  # type: ignore[operator]

    with pytest.raises(PersonaPackageError) as exc_info:
        validate_persona_package(package)

    assert exc_info.value.code == expected_code


def test_missing_required_heading_is_rejected_even_with_matching_hashes(
    tmp_path: Path,
) -> None:
    package = create_persona_package(
        tmp_path / "persona",
        content_overrides={
            "l3": """\
# 非生产测试 L3
## 创作手法
测试手法。
## 认知路径
测试路径。
## 摘要
测试摘要。
"""
        },
    )

    with pytest.raises(PersonaPackageError) as exc_info:
        validate_persona_package(package)

    assert exc_info.value.code == "required_heading_missing"


def test_l0_items_require_explicit_status_and_ownership(tmp_path: Path) -> None:
    invalid_l0 = NON_PRODUCTION_CONTENT["l0"].replace(
        "[真人已定][归属:创作者] 在困境中主动选择。",
        "在困境中主动选择。",
    )
    package = create_persona_package(tmp_path / "persona", content_overrides={"l0": invalid_l0})

    with pytest.raises(PersonaPackageError) as exc_info:
        validate_persona_package(package)

    assert exc_info.value.code == "l0_marker_invalid"


def test_l0_explicit_variant_ids_are_extracted_from_confirmed_items() -> None:
    l0 = NON_PRODUCTION_CONTENT["l0"].replace(
        "- [真人已定][归属:创作者] 在困境中主动选择。",
        "- [ID:A][真人已定][归属:创作者] 在困境中主动选择。\n"
        "- [ID:B][真人已定][归属:创作者] 在规则中守住承诺。",
    )

    assert extract_l0_variant_ids(l0) == ("A", "B")


@pytest.mark.parametrize(
    ("replacement", "expected_code"),
    [
        (
            "- [ID:A][真人已定][归属:创作者] 在困境中主动选择。\n"
            "- [ID:A][真人已定][归属:创作者] 在规则中守住承诺。",
            "l0_variant_id_duplicate",
        ),
        (
            "- [ID: A][真人已定][归属:创作者] 在困境中主动选择。",
            "l0_variant_id_invalid",
        ),
        (
            "- [ID:A][真人已定][归属:创作者] 在困境中主动选择。\n"
            "- [真人已定][归属:创作者] 在规则中守住承诺。",
            "l0_variant_id_missing",
        ),
    ],
)
def test_l0_explicit_variant_ids_reject_ambiguous_contracts(
    tmp_path: Path,
    replacement: str,
    expected_code: str,
) -> None:
    l0 = NON_PRODUCTION_CONTENT["l0"].replace(
        "- [真人已定][归属:创作者] 在困境中主动选择。",
        replacement,
    )
    package = create_persona_package(tmp_path / "persona", content_overrides={"l0": l0})

    with pytest.raises(PersonaPackageError) as exc_info:
        validate_persona_package(package)

    assert exc_info.value.code == expected_code


def test_manifest_path_escape_and_symlink_are_rejected(tmp_path: Path) -> None:
    escaped = create_persona_package(
        tmp_path / "escaped",
        manifest_mutator=lambda manifest: manifest["files"]["l0"].update({"path": "../outside.md"}),
    )
    with pytest.raises(PersonaPackageError) as escaped_error:
        validate_persona_package(escaped)
    assert escaped_error.value.code == "manifest_schema_invalid"

    linked = create_persona_package(tmp_path / "linked")
    original_l6 = (linked / "l6.md").read_bytes()
    external = tmp_path / "external.md"
    external.write_bytes(original_l6)
    (linked / "l6.md").unlink()
    (linked / "l6.md").symlink_to(external)
    with pytest.raises(PersonaPackageError) as linked_error:
        validate_persona_package(linked)
    assert linked_error.value.code == "unsafe_package_entry"


def test_duplicate_persona_ids_are_not_selectable(tmp_path: Path) -> None:
    source_root = tmp_path / "personas"
    create_persona_package(source_root / "one", persona_id="duplicate", version="1")
    create_persona_package(source_root / "two", persona_id="duplicate", version="2")

    catalog = PersonaCatalog(source_root, tmp_path / "snapshots")

    assert catalog.discover() == []
    assert catalog.get("duplicate") is None


def test_discovery_logs_only_safe_rejection_metadata(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    source_root = tmp_path / "personas"
    invalid = create_persona_package(source_root / "invalid-secret")
    (invalid / "l6.md").unlink()
    create_persona_package(source_root / "duplicate-one", persona_id="duplicate", version="1")
    create_persona_package(source_root / "duplicate-two", persona_id="duplicate", version="2")
    catalog = PersonaCatalog(source_root, tmp_path / "snapshots")
    caplog.set_level(logging.WARNING, logger="pengine.personas")

    assert catalog.discover() == []

    assert "path=invalid-secret code=invalid_package_entries" in caplog.text
    assert "path=duplicate-one code=duplicate_persona_id" in caplog.text
    assert "path=duplicate-two code=duplicate_persona_id" in caplog.text
    assert "NON-PRODUCTION TEST FIXTURE" not in caplog.text


def test_snapshot_is_idempotent_and_source_is_never_mutated(tmp_path: Path) -> None:
    catalog, package_dir = _catalog(tmp_path)
    before = package_bytes(package_dir)

    first = catalog.create_snapshot("test-persona")
    second = catalog.create_snapshot("test-persona")

    assert package_bytes(package_dir) == before
    assert first.path == second.path
    assert first.summary == second.summary
    assert first.path.name == first.summary.snapshot_sha256
    assert package_bytes(first.path) == before


def test_identical_content_keeps_distinct_persona_snapshot_identities(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "personas"
    first_source = create_persona_package(
        source_root / "first",
        persona_id="writer-one",
        display_name="编剧一",
    )
    second_source = create_persona_package(
        source_root / "second",
        persona_id="writer-two",
        display_name="编剧二",
    )
    catalog = PersonaCatalog(source_root, tmp_path / "snapshots")

    first = catalog.create_snapshot("writer-one")
    second = catalog.create_snapshot("writer-two")

    first_manifest = json.loads((first_source / "manifest.json").read_text(encoding="utf-8"))
    second_manifest = json.loads((second_source / "manifest.json").read_text(encoding="utf-8"))
    assert first_manifest["package_sha256"] == second_manifest["package_sha256"]
    assert first.summary.snapshot_sha256 != second.summary.snapshot_sha256
    assert first.path != second.path
    assert first.summary.persona_id == "writer-one"
    assert second.summary.persona_id == "writer-two"
    assert catalog.resolve_snapshot(first.summary.snapshot_sha256).manifest["persona_id"] == (
        "writer-one"
    )
    assert catalog.resolve_snapshot(second.summary.snapshot_sha256).manifest["persona_id"] == (
        "writer-two"
    )


def test_version_only_change_creates_new_identity_preserving_snapshot(
    tmp_path: Path,
) -> None:
    catalog, package_dir = _catalog(tmp_path)
    first = catalog.create_snapshot("test-persona")
    first_package_hash = first.manifest["package_sha256"]

    create_persona_package(package_dir, version="2.0.1-test")
    restarted_catalog = PersonaCatalog(catalog.source_root, catalog.snapshot_root)
    second = restarted_catalog.create_snapshot("test-persona")

    assert second.manifest["package_sha256"] == first_package_hash
    assert second.summary.snapshot_sha256 != first.summary.snapshot_sha256
    assert restarted_catalog.resolve_snapshot(first.summary.snapshot_sha256).summary.version == (
        "2.0.0-test"
    )
    assert restarted_catalog.resolve_snapshot(second.summary.snapshot_sha256).summary.version == (
        "2.0.1-test"
    )


def test_later_source_version_does_not_change_existing_snapshot(tmp_path: Path) -> None:
    catalog, package_dir = _catalog(tmp_path)
    first = catalog.create_snapshot("test-persona")
    first_bytes = package_bytes(first.path)

    create_persona_package(
        package_dir,
        version="2.0.0-test",
        content_overrides={"l5": NON_PRODUCTION_CONTENT["l5"].replace("《霜桥》", "《新霜桥》")},
    )
    restarted_catalog = PersonaCatalog(catalog.source_root, catalog.snapshot_root)
    second = restarted_catalog.create_snapshot("test-persona")
    resolved_first = restarted_catalog.resolve_snapshot(first.summary.snapshot_sha256)

    assert second.summary.version == "2.0.0-test"
    assert second.summary.snapshot_sha256 != first.summary.snapshot_sha256
    assert package_bytes(resolved_first.path) == first_bytes
    assert resolved_first.text("l5") == first.text("l5")


def test_restart_scoped_snapshot_and_projection_caches_avoid_revalidation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    catalog, _ = _catalog(tmp_path)
    snapshot = catalog.create_snapshot("test-persona")
    restarted_catalog = PersonaCatalog(catalog.source_root, catalog.snapshot_root)

    validation_calls = 0
    chunk_calls = 0
    validate = persona_module.validate_persona_package
    markdown_chunks = persona_module._markdown_chunks

    def count_validation(*args: object, **kwargs: object) -> object:
        nonlocal validation_calls
        validation_calls += 1
        return validate(*args, **kwargs)

    def count_chunks(text: str) -> tuple[tuple[str, str], ...]:
        nonlocal chunk_calls
        chunk_calls += 1
        return markdown_chunks(text)

    monkeypatch.setattr(persona_module, "validate_persona_package", count_validation)
    monkeypatch.setattr(persona_module, "_markdown_chunks", count_chunks)

    story_context = restarted_catalog.load_stage_context(
        snapshot.summary.snapshot_sha256,
        InternalStage.GENERATING_STORY_OUTLINE,
    )
    same_story_context = restarted_catalog.load_stage_context(
        snapshot.summary.snapshot_sha256,
        InternalStage.GENERATING_STORY_OUTLINE,
    )
    script_context = restarted_catalog.load_stage_context(
        snapshot.summary.snapshot_sha256,
        InternalStage.GENERATING_EPISODE_SCRIPTS,
    )
    restarted_catalog.retrieve_references(snapshot.summary.snapshot_sha256, "霜桥反转")
    restarted_catalog.retrieve_references(snapshot.summary.snapshot_sha256, "场景压缩")

    assert validation_calls == 1
    assert chunk_calls == 2
    assert same_story_context is story_context
    assert "/persona/l5.md" not in story_context.files
    assert "/persona/l6.md" not in story_context.files
    assert "必须呈现主角的主动选择" in story_context.files["/persona/l4.md"]
    assert "场景必须承担叙事功能" not in story_context.files["/persona/l4.md"]
    assert "场景必须承担叙事功能" in script_context.files["/persona/l4.md"]
    assert "必须呈现主角的主动选择" not in script_context.files["/persona/l4.md"]


def test_stage_context_is_bounded_read_only_projection(tmp_path: Path) -> None:
    catalog, _ = _catalog(tmp_path)
    snapshot = catalog.create_snapshot("test-persona")

    context = catalog.load_stage_context(
        snapshot.summary.snapshot_sha256,
        InternalStage.GENERATING_STORY_OUTLINE,
    )

    assert context.total_chars == sum(len(value) for value in context.files.values())
    assert "/persona/project.md" in context.files
    assert "/persona/l0.md" in context.files
    assert "/persona/l5.md" not in context.files
    assert "/persona/l6.md" not in context.files
    assert "待定变体不得作为确认规则" not in context.files["/persona/l0.md"]
    assert "必须呈现主角的主动选择" in context.files["/persona/l4.md"]
    assert "场景必须承担叙事功能" not in context.files["/persona/l4.md"]
    with pytest.raises(TypeError):
        context.files["/persona/other.md"] = "blocked"  # type: ignore[index]
    with pytest.raises(PersonaPackageError) as exc_info:
        catalog.load_stage_context(
            snapshot.summary.snapshot_sha256,
            InternalStage.GENERATING_STORY_OUTLINE,
            max_chars=10,
        )
    assert exc_info.value.code == "stage_context_too_large"


@pytest.mark.parametrize(
    "stage",
    [
        InternalStage.SELECTING_L0_VARIANT,
        InternalStage.GENERATING_STORY_OUTLINE,
        InternalStage.GENERATING_CHARACTER_RELATIONSHIPS,
        InternalStage.GENERATING_EPISODE_OUTLINE,
        InternalStage.GENERATING_EPISODE_SCRIPTS,
        InternalStage.ACCEPTING_L0,
        InternalStage.ACCEPTING_L4,
    ],
)
def test_complete_l0_is_available_to_every_specialist_stage(
    tmp_path: Path,
    stage: InternalStage,
) -> None:
    catalog, _ = _catalog(tmp_path)
    snapshot = catalog.create_snapshot("test-persona")

    context = catalog.load_stage_context(snapshot.summary.snapshot_sha256, stage)

    assert context.files["/persona/l0.md"] == (
        NON_PRODUCTION_CONTENT["l0"]
        .replace(
            "- [AI草稿待真人确认][归属:创作者] 待定变体不得作为确认规则。\n",
            "",
        )
        .strip()
    )


@pytest.mark.parametrize(
    "stage",
    [
        InternalStage.SELECTING_L0_VARIANT,
        InternalStage.GENERATING_STORY_OUTLINE,
        InternalStage.GENERATING_CHARACTER_RELATIONSHIPS,
        InternalStage.GENERATING_EPISODE_OUTLINE,
        InternalStage.GENERATING_EPISODE_SCRIPTS,
        InternalStage.ACCEPTING_L0,
        InternalStage.ACCEPTING_L4,
    ],
)
def test_complete_soul_is_available_to_every_specialist_stage(
    tmp_path: Path,
    stage: InternalStage,
) -> None:
    catalog, _ = _catalog(tmp_path)
    snapshot = catalog.create_snapshot("test-persona")

    context = catalog.load_stage_context(snapshot.summary.snapshot_sha256, stage)

    assert context.files["/persona/soul.md"] == NON_PRODUCTION_CONTENT["soul"]
    assert "/persona/l1-summary.md" not in context.files
    assert "/persona/l2-summary.md" not in context.files


def test_l5_l6_retrieval_is_query_and_output_bounded(tmp_path: Path) -> None:
    catalog, _ = _catalog(tmp_path)
    snapshot = catalog.create_snapshot("test-persona")

    hits = catalog.retrieve_references(
        snapshot.summary.snapshot_sha256,
        "霜桥反转",
        limit=1,
        max_chars=40,
        per_result_chars=40,
    )

    assert len(hits) == 1
    assert hits[0].source == "l5"
    assert "霜桥反转" in hits[0].excerpt
    assert sum(len(hit.excerpt) for hit in hits) <= 40
    with pytest.raises(PersonaPackageError):
        catalog.retrieve_references(snapshot.summary.snapshot_sha256, "", limit=1)
    with pytest.raises(PersonaPackageError):
        catalog.retrieve_references(
            snapshot.summary.snapshot_sha256,
            "测试",
            sources=("l4",),
        )


@pytest.mark.parametrize(
    ("mutate", "expected_code"),
    [
        (
            lambda package: (package / "l1.md").write_text("legacy", encoding="utf-8"),
            "mixed_persona_schema",
        ),
        (lambda package: (package / "soul.md").unlink(), "invalid_package_entries"),
    ],
)
def test_v2_rejects_mixed_or_missing_soul_package(
    tmp_path: Path,
    mutate: object,
    expected_code: str,
) -> None:
    package = create_persona_package(tmp_path / "persona")
    mutate(package)  # type: ignore[operator]

    with pytest.raises(PersonaPackageError) as exc_info:
        validate_persona_package(package)

    assert exc_info.value.code == expected_code


def test_v2_rejects_manifest_that_declares_soul_and_legacy_layers(tmp_path: Path) -> None:
    package = create_persona_package(
        tmp_path / "persona",
        manifest_mutator=lambda manifest: manifest["files"].update(
            {"l1": dict(manifest["files"]["soul"])}
        ),
    )

    with pytest.raises(PersonaPackageError) as exc_info:
        validate_persona_package(package)

    assert exc_info.value.code == "mixed_persona_schema"


@pytest.mark.parametrize(
    ("soul", "expected_code"),
    [
        (
            NON_PRODUCTION_CONTENT["soul"].replace("状态：创作者已确认", "状态：AI草稿待真人确认"),
            "soul_pending",
        ),
        (NON_PRODUCTION_CONTENT["soul"] + ("扩" * 8_000), "soul_too_large"),
    ],
)
def test_v2_soul_status_and_size_fail_closed(
    tmp_path: Path,
    soul: str,
    expected_code: str,
) -> None:
    package = create_persona_package(tmp_path / "persona", content_overrides={"soul": soul})

    with pytest.raises(PersonaPackageError) as exc_info:
        validate_persona_package(package)

    assert exc_info.value.code == expected_code


def test_v1_source_is_not_selectable_but_snapshot_keeps_legacy_projection(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "personas"
    source = create_persona_package(
        source_root / "legacy",
        schema_version="1.0.0",
        version="1.0.0-test",
    )
    package = validate_persona_package(source)
    snapshot_root = tmp_path / "snapshots"
    snapshot_root.mkdir()
    shutil.copytree(source, snapshot_root / package.summary.snapshot_sha256)
    catalog = PersonaCatalog(source_root, snapshot_root)

    assert catalog.discover() == []
    assert catalog.get("test-persona") is None

    snapshot = catalog.resolve_snapshot(package.summary.snapshot_sha256)
    context = catalog.load_stage_context(
        snapshot.summary.snapshot_sha256,
        InternalStage.GENERATING_STORY_OUTLINE,
    )

    assert "/persona/soul.md" not in context.files
    assert "表达具有向前推动的能量" in context.files["/persona/l1-summary.md"]
    assert "冲突表现克制" in context.files["/persona/l2-summary.md"]


def test_canonical_hash_is_order_sensitive_and_manifest_is_excluded(tmp_path: Path) -> None:
    package = create_persona_package(tmp_path / "persona")
    manifest = json.loads((package / "manifest.json").read_text(encoding="utf-8"))
    ordered_hashes = [manifest["files"][name]["sha256"] for name, _ in LOGICAL_FILES]

    assert canonical_package_sha256(ordered_hashes) == manifest["package_sha256"]
    assert canonical_package_sha256(reversed(ordered_hashes)) != manifest["package_sha256"]
