from pathlib import Path

import pytest
from test_script_batch import (
    build_episode_lock_for,
    create_leased_run,
    initial_series_state,
    seed_active_design_and_batch,
    seed_batch_with_episodes,
    story_contract_sha256,
    three_episode_contract,
)
from test_series_bible import make_candidate

from pengine.config import Settings
from pengine.errors import DomainError
from pengine.personas import PersonaCatalog
from pengine.repository import Repository
from pengine.schemas import InternalStage
from pengine.series_bible import (
    bind_global_design_review,
    build_series_bible,
    validate_series_bible,
)
from pengine.series_review import active_prefix_hash, effective_milestones
from pengine.worker import AgentProtocolError, StageValidationError, Worker, _classify_failure


@pytest.fixture
async def repository(tmp_path):
    value = Repository(tmp_path / "pengine.sqlite3")
    await value.initialize()
    return value


def prefix_hash_of(candidates) -> str:
    return active_prefix_hash(
        [
            {"episode_number": candidate.episode_number, "content_sha256": candidate.content_sha256}
            for candidate in candidates
        ]
    )


# ---------------------------------------------------------------------------
# RPR-A1 / RPR-A11: bound review identity + stale lineage
# ---------------------------------------------------------------------------


async def test_review_binds_design_batch_prefix_and_call_id(repository: Repository) -> None:
    accepted, lease = await create_leased_run(repository)
    contract, active = await seed_active_design_and_batch(repository, lease.run_id)
    prior_state = initial_series_state(contract, story_contract_sha256(contract))
    lock = build_episode_lock_for(contract, 1, prior_state)
    candidate = await repository.commit_episode_candidate(
        lease.run_id,
        episode_number=1,
        content=lock.content,
        episode_lock=lock,
        call_id="episode-1-call",
        writer_notes="",
    )
    batch = await repository.get_script_batch_lineage(lease.run_id)
    prefix_hash = prefix_hash_of([candidate])

    bound = await repository.register_series_review(
        lease.run_id,
        review_type="milestone",
        episode_number=1,
        design_candidate_id=active.candidate_id,
        design_content_hash=active.content_hash,
        design_epoch=active.design_epoch,
        batch_id=batch.batch_id,
        batch_epoch=batch.batch_epoch,
        prefix_hash=prefix_hash,
        call_id="review-call-1",
        passed=True,
        category="pass",
        evidence="里程碑审查通过",
        earliest_affected_episode=None,
    )
    assert bound.review_type == "milestone"
    assert bound.design_content_hash == active.content_hash
    assert bound.batch_id == batch.batch_id
    assert bound.prefix_hash == prefix_hash
    assert bound.call_id == "review-call-1"
    assert bound.status == "active"


async def test_new_lineage_review_marks_prior_active_reviews_stale(
    repository: Repository,
) -> None:
    accepted, lease = await create_leased_run(repository)
    contract, active = await seed_active_design_and_batch(repository, lease.run_id)
    batch = await repository.get_script_batch_lineage(lease.run_id)
    await repository.register_series_review(
        lease.run_id,
        review_type="milestone",
        episode_number=1,
        design_candidate_id=active.candidate_id,
        design_content_hash=active.content_hash,
        design_epoch=active.design_epoch,
        batch_id=batch.batch_id,
        batch_epoch=batch.batch_epoch,
        prefix_hash="a" * 64,
        call_id="review-1",
        passed=True,
        category="pass",
        evidence="第一份审查",
        earliest_affected_episode=None,
    )
    # A review bound to a DIFFERENT design hash retires the prior active review.
    await repository.register_series_review(
        lease.run_id,
        review_type="final",
        episode_number=1,
        design_candidate_id=active.candidate_id,
        design_content_hash="f" * 64,
        design_epoch=active.design_epoch,
        batch_id=batch.batch_id,
        batch_epoch=batch.batch_epoch,
        prefix_hash="b" * 64,
        call_id="review-2",
        passed=True,
        category="pass",
        evidence="新设计审查",
        earliest_affected_episode=None,
    )
    reviews = await repository.get_series_reviews(lease.run_id)
    statuses = {review.call_id: review.status for review in reviews}
    assert statuses["review-1"] == "stale"
    assert statuses["review-2"] == "active"


# ---------------------------------------------------------------------------
# RPR-A6: one shared automatic suffix-rewrite budget per script batch
# ---------------------------------------------------------------------------


async def test_suffix_budget_is_shared_and_consumed_once(repository: Repository) -> None:
    accepted, lease = await create_leased_run(repository)
    contract, active = await seed_active_design_and_batch(repository, lease.run_id)
    batch = await repository.get_script_batch_lineage(lease.run_id)
    assert await repository.has_automatic_suffix_budget(lease.run_id, batch.batch_id)

    await repository.consume_automatic_suffix_budget(lease.run_id, batch.batch_id)
    assert not await repository.has_automatic_suffix_budget(lease.run_id, batch.batch_id)
    with pytest.raises(DomainError) as exhausted:
        await repository.consume_automatic_suffix_budget(lease.run_id, batch.batch_id)
    assert exhausted.value.code == "suffix_budget_exhausted"


async def test_new_batch_gets_a_fresh_suffix_budget(repository: Repository) -> None:
    accepted, lease = await create_leased_run(repository)
    contract, active = await seed_active_design_and_batch(repository, lease.run_id)
    batch = await repository.get_script_batch_lineage(lease.run_id)
    await repository.consume_automatic_suffix_budget(lease.run_id, batch.batch_id)

    new_batch = await repository.create_script_batch(
        lease.run_id,
        design_candidate_id=active.candidate_id,
        design_content_hash=active.content_hash,
        design_epoch=active.design_epoch,
    )
    assert new_batch.batch_epoch == batch.batch_epoch  # same design -> idempotent
    assert not await repository.has_automatic_suffix_budget(lease.run_id, new_batch.batch_id)


# ---------------------------------------------------------------------------
# RPR-A4: automatic design rebuild
# ---------------------------------------------------------------------------


async def test_design_rebuild_budget_and_trigger(repository: Repository) -> None:
    accepted, lease = await create_leased_run(repository)
    contract, active, committed = await seed_batch_with_episodes(repository, lease.run_id, up_to=1)
    assert await repository.design_rebuild_budget_available(lease.run_id)

    await repository.trigger_design_rebuild(lease.run_id, evidence="设计缺陷证据")
    # The script batch is superseded, active projection cleared, and the outline
    # checkpoint + episode plans reset so the design regenerates from episode 1.
    assert await repository.get_episode_drafts(lease.run_id) == []
    checkpoints = await repository.get_business_checkpoints(lease.run_id)
    assert InternalStage.GENERATING_EPISODE_OUTLINE not in checkpoints


# ---------------------------------------------------------------------------
# RPR-A8 / RPR-A9: one-cycle repair authorization
# ---------------------------------------------------------------------------


async def test_repair_authorization_pauses_and_grants_one_cycle(
    repository: Repository,
) -> None:
    accepted, lease = await create_leased_run(repository)
    contract, active, committed = await seed_batch_with_episodes(repository, lease.run_id, up_to=3)
    first, second, third = committed
    batch = await repository.get_script_batch_lineage(lease.run_id)

    await repository.pause_repair_authorization(
        lease.run_id,
        kind="suffix_rewrite",
        design_candidate_id=active.candidate_id,
        design_content_hash=active.content_hash,
        design_epoch=active.design_epoch,
        batch_id=batch.batch_id,
        batch_epoch=batch.batch_epoch,
        earliest_affected_episode=2,
        range_episodes=2,
        estimated_tokens=12_000,
        evidence="脚本缺陷证据",
        review_id="review-x",
    )
    resource = await repository.get_creation(accepted.creation_id)
    assert resource is not None
    assert resource.initial.state == "paused"
    assert resource.initial.pause.code == "repair_authorization"
    auth = resource.initial.authorization
    assert auth is not None
    assert auth.kind == "suffix_rewrite"
    assert auth.earliest_affected_episode == 2
    assert auth.estimated_tokens == 12_000
    assert auth.evidence == "脚本缺陷证据"

    # Generic Continue cannot bypass the authorization (RPR-A10).
    with pytest.raises(DomainError) as cannot_continue:
        await repository.continue_run(
            creation_id=accepted.creation_id,
            run_kind="initial",
            idempotency_key="continue-bypass",
        )
    assert cannot_continue.value.code == "run_not_controllable"

    # Authorize exactly one cycle: the bound suffix rewrite is actually performed
    # (preserve 1..N-1, supersede N..end) and the run requeues (RPR-A9).
    accepted_control = await repository.authorize_repair(
        creation_id=accepted.creation_id,
        run_kind="initial",
        idempotency_key="authorize-1",
    )
    assert accepted_control.run_state == "queued"
    active_batch = await repository.get_script_batch_lineage(lease.run_id)
    assert active_batch is not None
    assert active_batch.active_pointers == {1: first.candidate_id}
    drafts = await repository.get_episode_drafts(lease.run_id)
    assert [draft.episode_number for draft in drafts] == [1]
    candidates = await repository.get_episode_candidates(lease.run_id)
    by_id = {candidate.candidate_id: candidate for candidate in candidates}
    assert by_id[first.candidate_id].status == "active"
    assert by_id[second.candidate_id].status == "superseded"
    assert by_id[third.candidate_id].status == "superseded"
    assert await repository.first_unfinished_episode(lease.run_id) == 2

    # A second grant for the same authorization epoch is refused.
    with pytest.raises(DomainError) as consumed:
        await repository.authorize_repair(
            creation_id=accepted.creation_id,
            run_kind="initial",
            idempotency_key="authorize-2",
        )
    assert consumed.value.code in {"run_not_controllable", "repair_authorization_stale"}


async def test_authorization_is_stale_when_lineage_changes(
    repository: Repository,
) -> None:
    accepted, lease = await create_leased_run(repository)
    contract, active = await seed_active_design_and_batch(repository, lease.run_id)
    batch = await repository.get_script_batch_lineage(lease.run_id)
    await repository.pause_repair_authorization(
        lease.run_id,
        kind="design_rebuild",
        design_candidate_id=active.candidate_id,
        design_content_hash=active.content_hash,
        design_epoch=active.design_epoch,
        batch_id=batch.batch_id,
        batch_epoch=batch.batch_epoch,
        earliest_affected_episode=None,
        range_episodes=None,
        estimated_tokens=None,
        evidence="设计缺陷证据",
        review_id="review-y",
    )
    # Promote a NEW design so the paused authorization's lineage is stale.
    new_design = build_series_bible(
        run_id=str(lease.run_id),
        run_kind="initial",
        l0_variant="归返",
        genre="general",
        story_outline="新的完整故事梗概。",
        character_biographies="阿丽：回乡调查旧案的主角。\n阿博：见证旧事的证人。",
        relationship_logic="阿丽与阿博为搭档。",
        episode_outline="三集连续写作。",
        story_contract_payload=three_episode_contract().model_dump(mode="json"),
        parent_candidate_id=active.candidate_id,
        rebuild_count=1,
    )
    evidence = validate_series_bible(new_design)
    stored = await repository.register_series_bible_candidate(
        str(lease.run_id), lease.run_id, new_design, evidence
    )
    review = bind_global_design_review(
        stored,
        review_call_id="design-review-2",
        review_model_id="deepseek-v4-flash",
        passed=True,
        evidence="新设计审查通过",
    )
    await repository.record_series_bible_review(lease.run_id, stored.candidate_id, review)
    await repository.promote_series_bible(lease.run_id, stored.candidate_id)

    with pytest.raises(DomainError) as stale:
        await repository.authorize_repair(
            creation_id=accepted.creation_id,
            run_kind="initial",
            idempotency_key="authorize-stale",
        )
    assert stale.value.code == "repair_authorization_stale"


async def test_authorized_design_rebuild_allows_one_cycle_after_automatic_budget(
    repository: Repository,
) -> None:
    # Delivery #57 INT-A8: after the one automatic design rebuild is consumed, an
    # explicit repair authorization must still grant exactly one design-rebuild
    # cycle (RPR-A9) instead of being refused as "series_bible_rebuild_exhausted".
    accepted, lease = await create_leased_run(repository)
    contract, active = await seed_active_design_and_batch(repository, lease.run_id)
    batch = await repository.get_script_batch_lineage(lease.run_id)
    # Simulate the consumed automatic design-rebuild budget (rebuild_count = 1).
    async with repository._transaction() as connection:
        await connection.execute(
            "UPDATE series_bible_lineage SET rebuild_count = 1 WHERE run_id = ?",
            (str(lease.run_id),),
        )
    assert not await repository.design_rebuild_budget_available(lease.run_id)

    await repository.pause_repair_authorization(
        lease.run_id,
        kind="design_rebuild",
        design_candidate_id=active.candidate_id,
        design_content_hash=active.content_hash,
        design_epoch=active.design_epoch,
        batch_id=batch.batch_id,
        batch_epoch=batch.batch_epoch,
        earliest_affected_episode=None,
        range_episodes=None,
        estimated_tokens=9_000,
        evidence="设计缺陷证据",
        review_id="review-auth",
    )

    # Authorize exactly one cycle: the design stage resets and the run requeues,
    # while the automatic budget stays consumed (no implicit second automatic
    # rebuild and no third version).
    accepted_control = await repository.authorize_repair(
        creation_id=accepted.creation_id,
        run_kind="initial",
        idempotency_key="authorize-design-1",
    )
    assert accepted_control.run_state == "queued"
    checkpoints = await repository.get_business_checkpoints(lease.run_id)
    assert InternalStage.GENERATING_EPISODE_OUTLINE not in checkpoints
    lineage = await repository.get_series_bible_lineage(lease.run_id)
    assert lineage["rebuild_count"] == 1
    assert not await repository.design_rebuild_budget_available(lease.run_id)
    drafts = await repository.get_episode_drafts(lease.run_id)
    assert [draft.episode_number for draft in drafts] == []


async def test_worker_authorized_design_rebuild_promotes_a_fresh_design(
    repository: Repository,
    tmp_path: Path,
) -> None:
    # The authorized design-rebuild cycle must actually regenerate and promote a
    # complete fresh design (RPR-A9) when the worker syncs the design after the
    # run is requeued, invalidate the prior script batch (episodes restart at 1),
    # and must not restore the automatic budget.
    accepted, lease = await create_leased_run(repository)
    contract, active, committed = await seed_batch_with_episodes(
        repository,
        lease.run_id,
        up_to=2,
    )
    batch = await repository.get_script_batch_lineage(lease.run_id)
    assert len(await repository.get_active_episode_candidates(lease.run_id)) == 2
    async with repository._transaction() as connection:
        await connection.execute(
            "UPDATE series_bible_lineage SET rebuild_count = 1 WHERE run_id = ?",
            (str(lease.run_id),),
        )
    await repository.pause_repair_authorization(
        lease.run_id,
        kind="design_rebuild",
        design_candidate_id=active.candidate_id,
        design_content_hash=active.content_hash,
        design_epoch=active.design_epoch,
        batch_id=batch.batch_id,
        batch_epoch=batch.batch_epoch,
        earliest_affected_episode=None,
        range_episodes=None,
        estimated_tokens=9_000,
        evidence="设计缺陷证据",
        review_id="review-worker",
    )
    await repository.authorize_repair(
        creation_id=accepted.creation_id,
        run_kind="initial",
        idempotency_key="authorize-worker-1",
    )

    settings = Settings(
        persona_root=tmp_path / "personas",
        data_dir=tmp_path / "data",
    )
    worker = Worker(
        settings=settings,
        repository=repository,
        catalog=PersonaCatalog(
            settings.persona_root,
            settings.snapshot_root,
        ),
    )
    work = await repository.get_run_work_item(lease.run_id)
    rebuilt_contract = three_episode_contract().model_copy(deep=True)
    for fact in rebuilt_contract.facts:
        fact.value = f"{fact.value}（重建版）"
    approved = {
        InternalStage.GENERATING_STORY_OUTLINE: {
            "content": "新的完整故事梗概。\n其余人物设定与已批准大纲保持一致。",
        },
        InternalStage.GENERATING_CHARACTER_RELATIONSHIPS: {
            "character_biographies": "阿丽：回乡调查旧案的主角。\n阿博：见证旧事的证人。",
            "relationship_logic": "阿丽与阿博为搭档。",
        },
        InternalStage.SELECTING_L0_VARIANT: {"selected_l0_variant": "归返"},
        InternalStage.GENERATING_EPISODE_OUTLINE: {
            "content": "三集连续写作。",
            "story_contract": rebuilt_contract.model_dump(mode="json"),
            "contract_review": {
                "passed": True,
                "evidence": "独立设计审查通过。",
                "issues": [],
            },
        },
    }
    await worker._sync_series_bible(work, approved)

    fresh = await repository.get_run_series_bible(lease.run_id)
    assert fresh is not None
    assert fresh.content_hash != active.content_hash
    assert fresh.design_epoch == active.design_epoch + 1
    lineage = await repository.get_series_bible_lineage(lease.run_id)
    assert int(lineage["rebuild_count"]) == 1
    assert not await repository.design_rebuild_budget_available(lease.run_id)
    # The prior script batch is invalidated and writing restarts at episode 1.
    assert await repository.get_active_episode_candidates(lease.run_id) == []


async def test_worker_refuses_unauthorized_rebuild_after_automatic_budget(
    repository: Repository,
    tmp_path: Path,
) -> None:
    # With the automatic budget consumed and no granted design_rebuild
    # authorization bound to the active design, the worker must refuse the
    # rebuild instead of silently performing a second automatic one.
    accepted, lease = await create_leased_run(repository)
    contract, active = await seed_active_design_and_batch(repository, lease.run_id)
    async with repository._transaction() as connection:
        await connection.execute(
            "UPDATE series_bible_lineage SET rebuild_count = 1 WHERE run_id = ?",
            (str(lease.run_id),),
        )

    settings = Settings(
        persona_root=tmp_path / "personas",
        data_dir=tmp_path / "data",
    )
    worker = Worker(
        settings=settings,
        repository=repository,
        catalog=PersonaCatalog(
            settings.persona_root,
            settings.snapshot_root,
        ),
    )
    work = await repository.get_run_work_item(lease.run_id)
    approved = {
        InternalStage.GENERATING_STORY_OUTLINE: {
            "content": "新的完整故事梗概。\n其余人物设定与已批准大纲保持一致。",
        },
        InternalStage.GENERATING_CHARACTER_RELATIONSHIPS: {
            "character_biographies": "阿丽：回乡调查旧案的主角。\n阿博：见证旧事的证人。",
            "relationship_logic": "阿丽与阿博为搭档。",
        },
        InternalStage.SELECTING_L0_VARIANT: {"selected_l0_variant": "归返"},
        InternalStage.GENERATING_EPISODE_OUTLINE: {
            "content": "三集连续写作。",
            "story_contract": contract.model_dump(mode="json"),
            "contract_review": {
                "passed": True,
                "evidence": "独立设计审查通过。",
                "issues": [],
            },
        },
    }
    with pytest.raises(
        AgentProtocolError, match="may rebuild the design automatically at most once"
    ):
        await worker._sync_series_bible(work, approved)
    # Nothing changed: no second candidate, budget still consumed.
    lineage = await repository.get_series_bible_lineage(lease.run_id)
    assert int(lineage["rebuild_count"]) == 1
    active_now = await repository.get_run_series_bible(lease.run_id)
    assert active_now is not None
    assert active_now.content_hash == active.content_hash


async def test_worker_stops_invalid_series_bible_at_design_boundary(
    repository: Repository,
    tmp_path: Path,
) -> None:
    _, lease = await create_leased_run(repository)
    settings = Settings(
        persona_root=tmp_path / "personas",
        data_dir=tmp_path / "data",
    )
    worker = Worker(
        settings=settings,
        repository=repository,
        catalog=PersonaCatalog(
            settings.persona_root,
            settings.snapshot_root,
        ),
    )
    work = await repository.get_run_work_item(lease.run_id)
    approved = {
        InternalStage.GENERATING_STORY_OUTLINE: {"content": "完整故事梗概。"},
        InternalStage.GENERATING_CHARACTER_RELATIONSHIPS: {
            "character_biographies": "陌生人：合同角色都未出现在这份小传中。",
            "relationship_logic": "阿丽与阿博为搭档。",
        },
        InternalStage.SELECTING_L0_VARIANT: {"selected_l0_variant": "归返"},
        InternalStage.GENERATING_EPISODE_OUTLINE: {
            "content": "三集连续写作。",
            "story_contract": three_episode_contract().model_dump(mode="json"),
            "contract_review": {
                "passed": True,
                "evidence": "独立设计审查通过。",
                "issues": [],
            },
        },
    }

    with pytest.raises(StageValidationError) as error:
        await worker._sync_series_bible(work, approved)

    assert error.value.stage is InternalStage.GENERATING_EPISODE_OUTLINE
    assert _classify_failure(error.value) == (
        "stage_validation_failed",
        "系列设计未通过确定性校验，未进入分集剧本生成。",
    )
    assert await repository.get_run_series_bible(lease.run_id) is None
    candidates = await repository.get_run_series_bible_candidates(lease.run_id)
    assert len(candidates) == 1
    assert candidates[0].status == "unvalidated"


async def test_authorized_rebuild_is_idempotent_after_crash_before_promotion(
    repository: Repository,
) -> None:
    # Re-running the same authorized one-cycle rebuild (a crash between INSERT and
    # promotion) must resume the same candidate instead of failing on a duplicate
    # key or creating an orphan rebuild candidate.
    accepted, lease = await create_leased_run(repository)
    contract, active = await seed_active_design_and_batch(repository, lease.run_id)
    batch = await repository.get_script_batch_lineage(lease.run_id)
    async with repository._transaction() as connection:
        await connection.execute(
            "UPDATE series_bible_lineage SET rebuild_count = 1 WHERE run_id = ?",
            (str(lease.run_id),),
        )
    await repository.pause_repair_authorization(
        lease.run_id,
        kind="design_rebuild",
        design_candidate_id=active.candidate_id,
        design_content_hash=active.content_hash,
        design_epoch=active.design_epoch,
        batch_id=batch.batch_id,
        batch_epoch=batch.batch_epoch,
        earliest_affected_episode=None,
        range_episodes=None,
        estimated_tokens=9_000,
        evidence="设计缺陷证据",
        review_id="review-idem",
    )
    rebuilt = make_candidate(
        run_id=str(lease.run_id),
        story_outline="设计师确认首个候选存在结构性缺陷，整体重建设计。",
        parent_candidate_id=active.candidate_id,
        rebuild_count=1,
        design_epoch=active.design_epoch + 1,
    )
    evidence = validate_series_bible(rebuilt)
    first = await repository.rebuild_series_bible(
        str(accepted.creation_id),
        lease.run_id,
        rebuilt,
        evidence,
        authorized=True,
    )
    # The granted authorization now points at the rebuild candidate it produced.
    auth = await repository.get_repair_authorization(lease.run_id)
    assert auth is not None
    assert auth["rebuild_candidate_id"] == first.candidate_id

    # Simulate a crash between INSERT and promotion: the worker re-runs and
    # builds a fresh candidate object but reuses the persisted candidate id
    # bound to the authorization, so the same cycle resumes without an orphan.
    retried_build = make_candidate(
        run_id=str(lease.run_id),
        story_outline="设计师确认首个候选存在结构性缺陷，整体重建设计。",
        parent_candidate_id=active.candidate_id,
        rebuild_count=1,
        design_epoch=active.design_epoch + 1,
    )
    assert retried_build.candidate_id != rebuilt.candidate_id
    retried_build = retried_build.model_copy(update={"candidate_id": auth["rebuild_candidate_id"]})
    resumed = await repository.rebuild_series_bible(
        str(accepted.creation_id),
        lease.run_id,
        retried_build,
        evidence,
        authorized=True,
    )
    assert resumed.candidate_id == first.candidate_id
    candidates = await repository.get_run_series_bible_candidates(lease.run_id)
    assert len(candidates) == 2  # the active design plus exactly one rebuild
    lineage = await repository.get_series_bible_lineage(lease.run_id)
    assert int(lineage["rebuild_count"]) == 1
    assert not await repository.design_rebuild_budget_available(lease.run_id)


# ---------------------------------------------------------------------------
# RPR-A13: only a bound final whole-series PASS freezes formal delivery
# ---------------------------------------------------------------------------


async def test_succeed_run_requires_a_passing_bound_final_review(
    repository: Repository,
    tmp_path,
) -> None:
    from pengine.schemas import ContentPackage, Delivery, DeliveryReport, GateResult

    repository = Repository(tmp_path / "pengine.sqlite3")
    await repository.initialize()
    accepted, lease = await create_leased_run(repository)
    contract, active, committed = await seed_batch_with_episodes(repository, lease.run_id, up_to=3)
    batch = await repository.get_script_batch_lineage(lease.run_id)
    prefix_hash = prefix_hash_of(committed)
    # Approve the assembled-scripts checkpoint so the freeze validation reaches the
    # final-review gate.
    aggregate = dict(await repository.episode_aggregate_checkpoint_payload(lease.run_id))
    await repository.approve_business_checkpoint(
        lease.run_id,
        InternalStage.GENERATING_EPISODE_SCRIPTS,
        {"stage": InternalStage.GENERATING_EPISODE_SCRIPTS.value, **aggregate},
    )
    delivery = Delivery(
        content_package=ContentPackage(
            story_outline="大纲",
            character_biographies="小传",
            relationship_logic="关系",
            episode_outline="分集",
            episode_scripts=aggregate["content"],
        ),
        delivery_report=DeliveryReport(
            persona_id="fixture-writer",
            persona_version="fixture-1",
            persona_snapshot_sha256="a" * 64,
            selected_l0_variant="归返",
            selection_rationale="符合输入",
            l0_gate=GateResult(passed=True, evidence="L0"),
            l4_gate=GateResult(passed=True, evidence="L4"),
            ownership_statement="本次创作归当前任务所有。",
            feedback_handling=[],
        ),
    )
    # No passing bound final review -> freeze is refused.
    with pytest.raises(DomainError) as no_review:
        await repository.succeed_run(lease.run_id, delivery, final_review_id="missing")
    assert no_review.value.code == "final_review_required"

    review = await repository.register_series_review(
        lease.run_id,
        review_type="final",
        episode_number=3,
        design_candidate_id=active.candidate_id,
        design_content_hash=active.content_hash,
        design_epoch=active.design_epoch,
        batch_id=batch.batch_id,
        batch_epoch=batch.batch_epoch,
        prefix_hash=prefix_hash,
        call_id="final-review-call",
        passed=True,
        category="pass",
        evidence="全系列审查通过",
        earliest_affected_episode=None,
    )
    await repository.succeed_run(lease.run_id, delivery, final_review_id=review.review_id)
    # A later review bound to a different prefix hash is retained as evidence but
    # cannot change the frozen delivery.
    review_wrong = await repository.register_series_review(
        lease.run_id,
        review_type="final",
        episode_number=3,
        design_candidate_id=active.candidate_id,
        design_content_hash=active.content_hash,
        design_epoch=active.design_epoch,
        batch_id=batch.batch_id,
        batch_epoch=batch.batch_epoch,
        prefix_hash="9" * 64,
        call_id="final-review-wrong-prefix",
        passed=True,
        category="pass",
        evidence="不同前缀的审查",
        earliest_affected_episode=None,
    )
    # Both bound reviews are retained as immutable evidence (newest first).
    reviews = await repository.get_series_reviews(lease.run_id)
    assert {item.review_id for item in reviews} == {review.review_id, review_wrong.review_id}


# ---------------------------------------------------------------------------
# Schedule
# ---------------------------------------------------------------------------


def test_effective_milestones_always_include_the_final_episode() -> None:
    assert effective_milestones([], 1) == [1]
    assert effective_milestones([2, 1], 3) == [1, 2, 3]
    assert effective_milestones([1, 3], 3) == [1, 3]
    import pytest as _pytest

    with _pytest.raises(ValueError):
        effective_milestones([0], 3)
    with _pytest.raises(ValueError):
        effective_milestones([4], 3)
