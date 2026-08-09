"""Tests for the atomic SeriesBible design-package aggregate (Issue #55).

Covers the deterministic universal + genre-activated validation, one candidate
identity across every projection, DeepSeek review binding, CAS promotion,
one-automatic-rebuild budget, stale retention, restart retention, the design
epoch-change signal, and the durable unfinished design projection.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError
from test_continuity import make_contract
from test_repository import create_and_lease_initial, locked_outline_payload

from pengine.config import Settings
from pengine.errors import DomainError
from pengine.personas import PersonaCatalog
from pengine.repository import LeasedJob, Repository
from pengine.schemas import CreateCreationRequest, InternalStage, PersonaSnapshot
from pengine.series_bible import (
    activated_rule_names,
    bind_global_design_review,
    build_series_bible,
    canonical_series_bible_content_hash,
    detect_genre,
    project_series_bible,
    validate_series_bible,
)
from pengine.worker import Worker

NOW = datetime(2026, 7, 28, 12, 0, tzinfo=UTC)
SNAPSHOT_HASH = "a" * 64


@pytest.fixture
async def repository(tmp_path):
    value = Repository(tmp_path / "pengine.sqlite3")
    await value.initialize()
    return value


def make_candidate(
    *,
    run_id: str = "run-1",
    run_kind: str = "initial",
    genre: str = "general",
    l0_variant: str = "归返",
    story_outline: str = "离乡者回到旧屋处理旧事。",
    biographies: str = "林岚：回乡调查旧案的主角。",
    relationships: str = "林岚与旧屋守护者存在监护关系。",
    episode_outline: str = "第 1 集：林岚回到旧屋。",
    story_contract_payload: dict | None = None,
    **kwargs,
):
    contract = make_contract() if story_contract_payload is None else story_contract_payload
    return build_series_bible(
        run_id=run_id,
        run_kind=run_kind,
        l0_variant=l0_variant,
        genre=genre,
        story_outline=story_outline,
        character_biographies=biographies,
        relationship_logic=relationships,
        episode_outline=episode_outline,
        story_contract_payload=contract,
        now=NOW,
        **kwargs,
    )


def make_contract_payload(*, clue: bool = False) -> dict:
    contract = make_contract()
    payload = contract.model_dump(mode="json")
    if clue:
        payload["clues"] = [
            {
                "clue_id": "pocket_watch_clue",
                "description": "旧表停在九点十七分。",
                "introduced_episode": 1,
                "explained_episode": 1,
                "callback_episode": None,
                "introduction_is_visible_or_audible": True,
            }
        ]
        payload["episode_obligations"][0]["required_clue_ids"] = ["pocket_watch_clue"]
    return payload


def passing_review(
    candidate, *, call_id: str = "review-call-1", model_id: str = "deepseek-v4-flash"
):
    return bind_global_design_review(
        candidate,
        review_call_id=call_id,
        review_model_id=model_id,
        passed=True,
        evidence="独立设计审查通过。",
    )


def broken_reference_payload() -> dict:
    payload = make_contract_payload()
    payload["relationships"] = [
        {
            "source_character_id": "lin_lan",
            "target_character_id": "unknown_person",
            "relation": "相邻",
        }
    ]
    return payload


def duplicate_fact_payload() -> dict:
    payload = make_contract_payload()
    payload["facts"].append(dict(payload["facts"][0]))
    return payload


def broken_arithmetic_payload() -> dict:
    payload = make_contract_payload()
    payload["facts"].append(
        {
            "fact_id": "bad_amount",
            "subject": "旧屋",
            "predicate": "估值",
            "kind": "amount",
            "value": "not-a-number",
            "unit": "元",
            "first_revealed_episode": 1,
        }
    )
    return payload


def broken_timeline_order_payload() -> dict:
    payload = make_contract_payload()
    payload["timeline"][0]["order"] = 2
    return payload


async def create_and_lease(repository: Repository):
    return await create_and_lease_initial(
        repository,
        PersonaSnapshot(
            persona_id="fixture-writer",
            display_name="非生产测试人格",
            version="fixture-1",
            snapshot_sha256=SNAPSHOT_HASH,
        ),
        CreateCreationRequest(
            persona_id="fixture-writer",
            story="一个离乡的人回家处理旧屋。",
            requirements="创作一部当代短剧。",
        ),
    )


async def register_validated(repository: Repository, run_id, candidate):
    evidence = validate_series_bible(candidate)
    stored = await repository.register_series_bible_candidate(
        str(run_id), run_id, candidate, evidence
    )
    return stored, evidence


# ---------------------------------------------------------------------------
# Module-level behavior
# ---------------------------------------------------------------------------


def test_detect_genre_is_deterministic_and_mystery_aware() -> None:
    assert detect_genre("一个人回家。", "创作三集短剧。") == "general"
    assert detect_genre("海岛悬疑旧案。", "查明凶手动机。") == "mystery"


def test_sparse_story_produces_complete_series_bible_candidate() -> None:
    candidate = make_candidate()
    assert candidate.candidate_id.startswith("candidate_")
    assert candidate.version == 1
    assert candidate.design_epoch == 1
    assert len(candidate.content_hash) == 64
    assert candidate.content.story_outline
    assert candidate.content.character_biographies
    assert candidate.content.relationship_logic
    assert candidate.content.episode_outline
    assert candidate.content.story_contract.episode_count >= 1
    assert canonical_series_bible_content_hash(candidate.content) == candidate.content_hash


def test_all_projections_share_one_candidate_identity() -> None:
    candidate = make_candidate()
    summary = project_series_bible(candidate, is_active=True)
    assert summary.candidate_id == candidate.candidate_id
    assert summary.version == candidate.version
    assert summary.content_hash == candidate.content_hash
    assert summary.design_epoch == candidate.design_epoch
    assert summary.projections.story_outline == candidate.content.story_outline
    assert summary.projections.character_biographies == candidate.content.character_biographies
    assert summary.projections.relationship_logic == candidate.content.relationship_logic
    assert summary.projections.episode_outline == candidate.content.episode_outline
    assert summary.projections.story_contract_markdown
    assert summary.unfinished is True


def test_universal_validation_rejects_broken_references() -> None:
    with pytest.raises(ValidationError):
        make_candidate(story_contract_payload=broken_reference_payload())


def test_universal_validation_rejects_duplicate_ids() -> None:
    with pytest.raises(ValidationError):
        make_candidate(story_contract_payload=duplicate_fact_payload())


def test_universal_validation_rejects_broken_arithmetic() -> None:
    with pytest.raises(ValidationError):
        make_candidate(story_contract_payload=broken_arithmetic_payload())


def test_universal_validation_rejects_broken_ordering() -> None:
    with pytest.raises(ValidationError):
        make_candidate(story_contract_payload=broken_timeline_order_payload())


def test_universal_validation_rejects_projection_mismatch() -> None:
    candidate = build_series_bible(
        run_id="run-1",
        run_kind="initial",
        l0_variant="归返",
        genre="general",
        story_outline="大纲",
        character_biographies="李雷：一位住在旧屋对面的邻居。",
        relationship_logic="关系逻辑",
        episode_outline="分集大纲",
        story_contract_payload=make_contract_payload(),
        now=NOW,
    )
    evidence = validate_series_bible(candidate)
    assert not evidence.passed
    codes = {issue.code for issue in evidence.issues}
    assert "projection_missing_biography" in codes


def test_projection_validation_matches_unique_annotated_character_label() -> None:
    payload = make_contract_payload()
    payload["characters"][0]["name"] = "林岚（主角）"
    candidate = make_candidate(
        biographies="林岚（现用名／主角）：回乡调查旧案的主角。",
        story_contract_payload=payload,
    )

    assert validate_series_bible(candidate).passed


def test_projection_validation_requires_exact_label_when_bases_collide() -> None:
    payload = make_contract_payload()
    payload["characters"][0]["name"] = "林岚（青年）"
    payload["characters"].append(
        {
            "character_id": "lin_lan_elder",
            "name": "林岚（老年）",
            "role": "多年后的调查者",
            "initial_known_fact_ids": [],
        }
    )
    payload["timeline"][0]["participant_ids"].append("lin_lan_elder")
    payload["knowledge_states"].append(
        {
            "episode_number": 1,
            "character_id": "lin_lan_elder",
            "known_fact_ids": [fact["fact_id"] for fact in payload["facts"]],
        }
    )
    candidate = make_candidate(
        biographies="林岚（青年）：回乡调查旧案。",
        story_contract_payload=payload,
    )

    evidence = validate_series_bible(candidate)

    assert not evidence.passed
    missing_refs = {
        reference
        for issue in evidence.issues
        if issue.code == "projection_missing_biography"
        for reference in issue.refs
    }
    assert missing_refs == {"lin_lan_elder"}


def test_genre_activation_never_rejects_general_idea_for_missing_clues() -> None:
    general = make_candidate(genre="general")
    assert activated_rule_names(general) == frozenset()
    assert validate_series_bible(general).passed

    mystery = make_candidate(genre="mystery")
    assert "mystery_reveal_required" in activated_rule_names(mystery)
    evidence = validate_series_bible(mystery)
    assert not evidence.passed
    assert "mystery_reveal_required" in {issue.code for issue in evidence.issues}


def test_mystery_with_declared_clues_passes_genre_activation() -> None:
    mystery = make_candidate(
        genre="mystery", story_contract_payload=make_contract_payload(clue=True)
    )
    assert validate_series_bible(mystery).passed


def test_candidate_hash_changes_when_any_projection_changes() -> None:
    first = make_candidate(story_outline="离乡者回到旧屋处理旧事。")
    second = make_candidate(story_outline="离乡者回到旧屋重建生活。")
    assert first.content_hash != second.content_hash


# ---------------------------------------------------------------------------
# Repository-level promotion / rebuild / restart / stale / CAS behavior
# ---------------------------------------------------------------------------


async def test_promotion_requires_validation_and_passing_bound_review(
    repository: Repository,
) -> None:
    _accepted, lease = await create_and_lease(repository)
    candidate = make_candidate(run_id=str(lease.run_id))
    stored, evidence = await register_validated(repository, lease.run_id, candidate)
    assert evidence.passed
    with pytest.raises(DomainError) as error:
        await repository.promote_series_bible(lease.run_id, stored.candidate_id)
    assert error.value.code == "series_bible_review_required"


async def test_review_must_bind_the_candidate_it_records(repository: Repository) -> None:
    _accepted, lease = await create_and_lease(repository)
    candidate = make_candidate(run_id=str(lease.run_id))
    stored, _evidence = await register_validated(repository, lease.run_id, candidate)
    other = make_candidate(run_id=str(lease.run_id), story_outline="不同的设计内容。")
    foreign_review = passing_review(other, call_id="other-candidate-review")
    with pytest.raises(DomainError) as error:
        await repository.record_series_bible_review(
            lease.run_id, stored.candidate_id, foreign_review
        )
    assert error.value.code == "series_bible_review_mismatch"


async def test_promotion_supersedes_prior_active_and_marks_stale(repository: Repository) -> None:
    _accepted, lease = await create_and_lease(repository)
    first = make_candidate(run_id=str(lease.run_id))
    stored, _evidence = await register_validated(repository, lease.run_id, first)
    await repository.record_series_bible_review(
        lease.run_id, stored.candidate_id, passing_review(stored)
    )
    active = await repository.promote_series_bible(lease.run_id, stored.candidate_id)
    assert active.status == "active"

    rebuilt = make_candidate(
        run_id=str(lease.run_id),
        story_outline="设计师确认首个候选存在结构性缺陷，整体重建设计。",
        parent_candidate_id=stored.candidate_id,
        rebuild_count=1,
        design_epoch=2,
    )
    rebuilt_evidence = validate_series_bible(rebuilt)
    stored_rebuilt = await repository.rebuild_series_bible(
        str(lease.run_id), lease.run_id, rebuilt, rebuilt_evidence
    )
    await repository.record_series_bible_review(
        lease.run_id, stored_rebuilt.candidate_id, passing_review(stored_rebuilt)
    )
    promoted = await repository.promote_series_bible(lease.run_id, stored_rebuilt.candidate_id)
    assert promoted.design_epoch == 2
    assert promoted.status == "active"

    candidates = await repository.get_run_series_bible_candidates(lease.run_id)
    by_id = {item.candidate_id: item for item in candidates}
    assert by_id[stored.candidate_id].status == "superseded"
    assert by_id[promoted.candidate_id].status == "active"

    lineage = await repository.get_series_bible_lineage(lease.run_id)
    assert lineage is not None and lineage["rebuild_count"] == 1


async def test_same_run_lineage_cannot_rebuild_a_second_time(repository: Repository) -> None:
    _accepted, lease = await create_and_lease(repository)
    first = make_candidate(run_id=str(lease.run_id))
    stored, _evidence = await register_validated(repository, lease.run_id, first)
    await repository.record_series_bible_review(
        lease.run_id, stored.candidate_id, passing_review(stored)
    )
    await repository.promote_series_bible(lease.run_id, stored.candidate_id)

    rebuilt = make_candidate(
        run_id=str(lease.run_id),
        story_outline="设计师确认首个候选存在结构性缺陷，整体重建设计。",
        parent_candidate_id=stored.candidate_id,
        rebuild_count=1,
        design_epoch=2,
    )
    rebuilt_evidence = validate_series_bible(rebuilt)
    stored_rebuilt = await repository.rebuild_series_bible(
        str(lease.run_id), lease.run_id, rebuilt, rebuilt_evidence
    )
    await repository.record_series_bible_review(
        lease.run_id, stored_rebuilt.candidate_id, passing_review(stored_rebuilt)
    )
    await repository.promote_series_bible(lease.run_id, stored_rebuilt.candidate_id)

    second_defect = make_candidate(
        run_id=str(lease.run_id),
        story_outline="再次整体重建将需要显式授权。",
        parent_candidate_id=stored_rebuilt.candidate_id,
        rebuild_count=1,
        design_epoch=3,
    )
    with pytest.raises(DomainError) as error:
        await repository.rebuild_series_bible(
            str(lease.run_id), lease.run_id, second_defect, validate_series_bible(second_defect)
        )
    assert error.value.code == "series_bible_rebuild_exhausted"


async def test_late_candidate_cannot_promote_and_is_retained_stale(repository: Repository) -> None:
    _accepted, lease = await create_and_lease(repository)
    first = make_candidate(run_id=str(lease.run_id))
    stored, _evidence = await register_validated(repository, lease.run_id, first)
    await repository.record_series_bible_review(
        lease.run_id, stored.candidate_id, passing_review(stored)
    )
    await repository.promote_series_bible(lease.run_id, stored.candidate_id)

    late = make_candidate(
        run_id=str(lease.run_id), story_outline="迟到的旧生成响应。", design_epoch=1
    )
    stored_late, _evidence = await register_validated(repository, lease.run_id, late)
    await repository.record_series_bible_review(
        lease.run_id, stored_late.candidate_id, passing_review(stored_late)
    )
    with pytest.raises(DomainError) as error:
        await repository.promote_series_bible(lease.run_id, stored_late.candidate_id)
    assert error.value.code == "series_bible_stale_promotion"

    await repository.mark_series_bible_stale(lease.run_id, active_candidate_id=stored.candidate_id)
    candidates = await repository.get_run_series_bible_candidates(lease.run_id)
    statuses = {item.candidate_id: item.status for item in candidates}
    assert statuses[stored_late.candidate_id] == "stale"


async def test_restart_retains_active_candidate_evidence_and_rebuild_budget(tmp_path: Path) -> None:
    repository = Repository(tmp_path / "pengine.sqlite3")
    await repository.initialize()
    _accepted, lease = await create_and_lease(repository)
    first = make_candidate(run_id=str(lease.run_id))
    stored, _evidence = await register_validated(repository, lease.run_id, first)
    await repository.record_series_bible_review(
        lease.run_id, stored.candidate_id, passing_review(stored)
    )
    active = await repository.promote_series_bible(lease.run_id, stored.candidate_id)

    reopened = Repository(tmp_path / "pengine.sqlite3")
    await reopened.initialize()
    restored = await reopened.get_run_series_bible(lease.run_id)
    assert restored is not None
    assert restored.candidate_id == active.candidate_id
    assert restored.content_hash == active.content_hash
    assert restored.design_epoch == active.design_epoch
    assert restored.validation is not None and restored.validation.passed is True
    assert restored.global_review is not None and restored.global_review.passed is True
    lineage = await reopened.get_series_bible_lineage(lease.run_id)
    assert lineage is not None and lineage["rebuild_count"] == 0


async def test_design_epoch_change_signals_prior_script_batch_ineligibility(
    repository: Repository,
) -> None:
    _accepted, lease = await create_and_lease(repository)
    first = make_candidate(run_id=str(lease.run_id))
    stored, _evidence = await register_validated(repository, lease.run_id, first)
    await repository.record_series_bible_review(
        lease.run_id, stored.candidate_id, passing_review(stored)
    )
    active = await repository.promote_series_bible(lease.run_id, stored.candidate_id)
    assert await repository.assert_episode_batch_current(lease.run_id) == active.content_hash

    rebuilt = make_candidate(
        run_id=str(lease.run_id),
        story_outline="设计变更后旧脚本批次失效。",
        parent_candidate_id=stored.candidate_id,
        rebuild_count=1,
        design_epoch=2,
    )
    rebuilt_evidence = validate_series_bible(rebuilt)
    stored_rebuilt = await repository.rebuild_series_bible(
        str(lease.run_id), lease.run_id, rebuilt, rebuilt_evidence
    )
    await repository.record_series_bible_review(
        lease.run_id, stored_rebuilt.candidate_id, passing_review(stored_rebuilt)
    )
    await repository.promote_series_bible(lease.run_id, stored_rebuilt.candidate_id)

    current = await repository.assert_episode_batch_current(lease.run_id)
    assert current == rebuilt.content_hash
    assert current != active.content_hash


# ---------------------------------------------------------------------------
# Worker integration: one atomic candidate promoted from approved checkpoints
# ---------------------------------------------------------------------------


def _make_worker(tmp_path: Path) -> tuple[Settings, Repository, Worker]:
    settings = Settings(
        _env_file=None,
        data_dir=tmp_path,
        persona_root=tmp_path / "personas",
        generation_model_id="claude-opus-5",
        review_model_id="deepseek-v4-flash",
    )
    catalog = PersonaCatalog(settings.persona_root, settings.snapshot_root)
    repository = Repository(settings.database_path)
    return (
        settings,
        repository,
        Worker(
            settings=settings,
            repository=repository,
            catalog=catalog,
        ),
    )


async def _approve_design_stages(repository: Repository, run_id) -> dict:
    _contract, outline_payload = locked_outline_payload()
    design_payloads = {
        InternalStage.SELECTING_L0_VARIANT: {
            "stage": "selecting_l0_variant",
            "selected_l0_variant": "主动选择",
            "selection_rationale": "符合测试故事",
        },
        InternalStage.GENERATING_STORY_OUTLINE: {
            "stage": "generating_story_outline",
            "content": "故事大纲",
        },
        InternalStage.GENERATING_CHARACTER_RELATIONSHIPS: {
            "stage": "generating_character_relationships",
            "character_biographies": "林岚：回乡调查旧案的主角。",
            "relationship_logic": "关系逻辑",
        },
        InternalStage.GENERATING_EPISODE_OUTLINE: outline_payload,
    }
    for stage, payload in design_payloads.items():
        await repository.approve_business_checkpoint(run_id, stage, payload, now=NOW)
    return {stage: dict(payload) for stage, payload in design_payloads.items()}


async def test_worker_sync_promotes_one_active_series_bible(tmp_path: Path) -> None:
    settings, repository, worker = _make_worker(tmp_path)
    await repository.initialize()
    accepted, lease = await create_and_lease(repository)
    await repository.mark_run_running(lease.run_id, now=NOW)
    approved = await _approve_design_stages(repository, lease.run_id)
    work = await repository.get_run_work_item(lease.run_id)

    await worker._sync_series_bible(work, approved)

    resource = await repository.get_creation(accepted.creation_id, now=NOW)
    assert resource is not None
    assert resource.initial.state == "running"
    design = resource.initial.drafts.design
    assert design is not None
    assert design.status == "active"
    assert design.is_active is True
    assert design.unfinished is True
    assert design.projections.story_outline == "故事大纲"
    assert design.projections.episode_outline
    assert design.global_review is not None
    assert design.global_review.candidate_id == design.candidate_id
    assert design.global_review.candidate_hash == design.content_hash

    lineage = await repository.get_series_bible_lineage(lease.run_id)
    assert lineage is not None and lineage["active_design_epoch"] == 1


async def test_worker_sync_is_idempotent_across_restart(tmp_path: Path) -> None:
    settings, repository, worker = _make_worker(tmp_path)
    await repository.initialize()
    accepted, lease = await create_and_lease(repository)
    await repository.mark_run_running(lease.run_id, now=NOW)
    approved = await _approve_design_stages(repository, lease.run_id)
    work = await repository.get_run_work_item(lease.run_id)

    await worker._sync_series_bible(work, approved)
    await worker._sync_series_bible(work, approved)

    reopened = Repository(settings.database_path)
    await reopened.initialize()
    restored = await reopened.get_creation(accepted.creation_id, now=NOW)
    assert restored is not None
    design = restored.initial.drafts.design
    assert design is not None and design.status == "active"

    candidates = await reopened.get_run_series_bible_candidates(lease.run_id)
    assert len(candidates) == 1


async def test_worker_never_syncs_a_legacy_outline_without_contract(tmp_path: Path) -> None:
    settings, repository, worker = _make_worker(tmp_path)
    await repository.initialize()
    _accepted, lease = await create_and_lease(repository)
    await repository.mark_run_running(lease.run_id, now=NOW)
    approved = {
        InternalStage.SELECTING_L0_VARIANT: {
            "stage": "selecting_l0_variant",
            "selected_l0_variant": "主动选择",
            "selection_rationale": "符合测试故事",
        },
        InternalStage.GENERATING_STORY_OUTLINE: {
            "stage": "generating_story_outline",
            "content": "大纲",
        },
        InternalStage.GENERATING_CHARACTER_RELATIONSHIPS: {
            "stage": "generating_character_relationships",
            "character_biographies": "小传",
            "relationship_logic": "关系",
        },
        InternalStage.GENERATING_EPISODE_OUTLINE: {
            "stage": "generating_episode_outline",
            "content": "分集大纲",
            "episode_count": 1,
            "episodes": [{"episode_number": 1, "plan": "林岚回到旧屋。"}],
        },
    }
    work = await repository.get_run_work_item(lease.run_id)
    await worker._sync_series_bible(work, approved)
    assert await repository.get_run_series_bible(lease.run_id) is None


async def test_restart_before_promotion_recovers_active_design(tmp_path: Path) -> None:
    """A crash after the outline commit but before design promotion must recover.

    The approve hook that syncs the SeriesBible only fires during live
    delegation. On restart the outline is already approved and never
    re-delegated, so _process_job must re-run the idempotent sync itself or the
    run would proceed with no active design (SDP-A8 restart-before-promotion).
    """
    from persona_factory import create_persona_package

    persona_root = tmp_path / "personas"
    create_persona_package(persona_root / "active")
    settings = Settings(persona_root=persona_root, data_dir=tmp_path / "data")
    catalog = PersonaCatalog(persona_root, settings.snapshot_root)
    repository = Repository(settings.database_path)
    await repository.initialize()
    snapshot = catalog.create_snapshot("test-persona")
    accepted, lease = await create_and_lease_initial(
        repository,
        snapshot.summary,
        CreateCreationRequest(
            persona_id="test-persona",
            story="一个离乡的人回家处理旧屋。",
            requirements="创作一部当代短剧。",
        ),
    )
    del accepted
    await repository.mark_run_running(lease.run_id, now=NOW)
    await _approve_design_stages(repository, lease.run_id)
    # Simulate the crash window: the outline checkpoint is durable but the
    # design candidate was never registered, reviewed, or promoted.
    assert await repository.get_run_series_bible(lease.run_id) is None

    reopened = Repository(settings.database_path)
    await reopened.initialize()
    worker2 = Worker(
        settings=settings,
        repository=reopened,
        catalog=catalog,
        workflow=None,
        worker_id="restart-before-promotion-worker",
    )
    job = LeasedJob(
        job_id=lease.job_id,
        run_id=lease.run_id,
        creation_id=lease.creation_id,
        run_kind="initial",
        run_sequence=lease.run_sequence,
        thread_id=lease.thread_id,
        lease_owner="restart-before-promotion-worker",
        lease_expires_at=NOW,
    )
    # workflow=None fails the run with relay_unavailable at the next unapproved
    # stage (episode scripts) instead of raising to the caller; the important
    # assertion is that the approved-outline recovery sync ran first and
    # promoted the design candidate before that failure.
    await worker2._process_job(job)

    restored = await reopened.get_run_series_bible(lease.run_id)
    assert restored is not None
    assert restored.status == "active"
    assert restored.is_active is True
    assert restored.global_review is not None and restored.global_review.passed is True
