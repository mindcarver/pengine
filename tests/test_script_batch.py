from datetime import UTC, datetime

import pytest
from test_continuity import make_sparse_knowledge_contract
from test_repository import create_and_lease_initial, persist_succeeded_outline_review

from pengine.continuity import (
    EpisodeStateDelta,
    ScriptEvidence,
    SemanticReview,
    StoryContract,
    bind_episode_delta_to_contract,
    build_episode_lock,
    initial_series_state,
    render_story_contract_markdown,
    story_contract_sha256,
)
from pengine.errors import DomainError
from pengine.repository import SCHEMA_VERSION, Repository
from pengine.schemas import (
    CreateCreationRequest,
    InternalStage,
    PersonaSnapshot,
)
from pengine.series_bible import (
    SeriesBible,
    bind_global_design_review,
    build_series_bible,
    validate_series_bible,
)

NOW = datetime(2026, 8, 3, 12, 0, tzinfo=UTC)
SNAPSHOT_HASH = "a" * 64


@pytest.fixture
async def repository(tmp_path):
    value = Repository(tmp_path / "pengine.sqlite3")
    await value.initialize()
    return value


def make_persona() -> PersonaSnapshot:
    return PersonaSnapshot(
        persona_id="fixture-writer",
        display_name="非生产测试人格",
        version="fixture-1",
        snapshot_sha256=SNAPSHOT_HASH,
    )


def make_request() -> CreateCreationRequest:
    return CreateCreationRequest(
        persona_id="fixture-writer",
        story="一个离乡的人回家处理旧屋。",
        requirements="创作一部三集短剧。",
    )


async def create_leased_run(repository: Repository):
    accepted, lease = await create_and_lease_initial(
        repository,
        make_persona(),
        make_request(),
    )
    return accepted, lease


def make_design(contract: StoryContract, *, run_id: str, **kwargs) -> SeriesBible:
    base = dict(
        run_id=run_id,
        run_kind="initial",
        l0_variant="归返",
        genre="general",
        story_outline="离乡者回到旧屋处理旧事。",
        character_biographies="阿丽：回乡调查旧案的主角。\n阿博：见证旧事的证人。",
        relationship_logic="阿丽与阿博为搭档。",
        episode_outline="三集连续写作。",
        story_contract_payload=contract.model_dump(mode="json"),
        now=NOW,
    )
    base.update(kwargs)
    return build_series_bible(**base)


async def promote_active_design(repository: Repository, run_id, contract: StoryContract):
    design = make_design(contract, run_id=str(run_id))
    evidence = validate_series_bible(design)
    stored = await repository.register_series_bible_candidate(
        str(run_id), run_id, design, evidence, now=NOW
    )
    review = bind_global_design_review(
        stored,
        review_call_id="design-review-call-1",
        review_model_id="deepseek-v4-flash",
        passed=True,
        evidence="独立设计审查通过。",
        now=NOW,
    )
    await repository.record_series_bible_review(run_id, stored.candidate_id, review, now=NOW)
    active = await repository.promote_series_bible(run_id, stored.candidate_id, now=NOW)
    return active


def obligation_hook(contract: StoryContract, episode: int) -> str:
    return next(
        item.end_hook for item in contract.episode_obligations if item.episode_number == episode
    )


def fact_value(contract: StoryContract, fact_id: str) -> str:
    return next(fact.value for fact in contract.facts if fact.fact_id == fact_id)


def build_episode_delta(
    contract: StoryContract,
    episode: int,
    prior_state,
) -> EpisodeStateDelta:
    obligation = next(
        item for item in contract.episode_obligations if item.episode_number == episode
    )
    delta = EpisodeStateDelta(
        episode_number=episode,
        contract_sha256=story_contract_sha256(contract),
        established_fact_ids=[],
        knowledge_gains=[],
        introduced_clue_ids=[],
        resolved_clue_ids=[],
        satisfied_obligation_ids=[],
        evidence=[],
        handoff=f"第{episode}集结束。",
    )
    delta = bind_episode_delta_to_contract(
        contract=contract,
        prior_state=prior_state,
        delta=delta,
    )
    revealed_facts = [
        fact.fact_id for fact in contract.facts if fact.first_revealed_episode == episode
    ]
    excerpts = {fact_id: fact_value(contract, fact_id) for fact_id in revealed_facts}
    excerpts[obligation.obligation_id] = obligation.end_hook
    return delta.model_copy(
        update={
            "evidence": [
                ScriptEvidence(target_id=target_id, excerpt=excerpt)
                for target_id, excerpt in excerpts.items()
            ]
        }
    )


def episode_content(delta: EpisodeStateDelta) -> str:
    return "\n".join(item.excerpt for item in delta.evidence)


def build_episode_lock_for(
    contract: StoryContract,
    episode: int,
    prior_state,
    *,
    repair_rounds: int = 0,
):
    delta = build_episode_delta(contract, episode, prior_state)
    content = episode_content(delta)
    return build_episode_lock(
        contract=contract,
        contract_sha256=story_contract_sha256(contract),
        prior_state=prior_state,
        content=content,
        delta=delta,
        semantic_review=SemanticReview(
            passed=True,
            evidence="独立分集审查通过",
            issues=[],
        ),
        repair_rounds=repair_rounds,
    )


def three_episode_contract() -> StoryContract:
    return make_sparse_knowledge_contract(
        [
            {"episode_number": 1, "character_id": "alice", "known_fact_ids": ["fact_one"]},
            {"episode_number": 1, "character_id": "bob", "known_fact_ids": ["fact_one"]},
            {
                "episode_number": 2,
                "character_id": "alice",
                "known_fact_ids": ["fact_one", "fact_two"],
            },
            {
                "episode_number": 2,
                "character_id": "bob",
                "known_fact_ids": ["fact_one", "fact_two"],
            },
            {
                "episode_number": 3,
                "character_id": "alice",
                "known_fact_ids": ["fact_one", "fact_two", "fact_three"],
            },
            {
                "episode_number": 3,
                "character_id": "bob",
                "known_fact_ids": ["fact_one", "fact_two", "fact_three"],
            },
        ]
    )


async def _approve_episode_outline(repository: Repository, run_id, contract: StoryContract):
    contract_hash = story_contract_sha256(contract)
    payload = {
        "stage": InternalStage.GENERATING_EPISODE_OUTLINE.value,
        "content": "三集分集大纲",
        "episode_count": contract.episode_count,
        "episodes": [
            {"episode_number": number, "plan": f"第{number}集计划。"}
            for number in range(1, contract.episode_count + 1)
        ],
        "story_contract": contract.model_dump(mode="json"),
        "story_contract_sha256": contract_hash,
        "story_contract_markdown": render_story_contract_markdown(contract, contract_hash),
        "contract_review": {"passed": True, "evidence": "独立合同审查通过", "issues": []},
        "contract_repair_rounds": 0,
    }
    await repository.record_stage_attempt(
        run_id,
        InternalStage.GENERATING_EPISODE_OUTLINE,
        now=NOW,
    )
    review_call_id = persist_succeeded_outline_review(repository, run_id)
    await repository.approve_business_checkpoint(
        run_id,
        InternalStage.GENERATING_EPISODE_OUTLINE,
        payload,
        review_call_id=review_call_id,
        now=NOW,
    )


async def seed_active_design_and_batch(repository: Repository, run_id):
    """Approve the outline, promote an active design, and create the script batch."""
    contract = three_episode_contract()
    await _approve_episode_outline(repository, run_id, contract)
    active = await promote_active_design(repository, run_id, contract)
    await repository.create_script_batch(
        run_id,
        design_candidate_id=active.candidate_id,
        design_content_hash=active.content_hash,
        design_epoch=active.design_epoch,
        now=NOW,
    )
    return contract, active


async def seed_batch_with_episodes(
    repository: Repository,
    run_id,
    *,
    up_to: int = 0,
):
    """Seed a design-bound batch and commit episodes 1..up_to."""
    contract, active = await seed_active_design_and_batch(repository, run_id)
    prior_state = initial_series_state(contract, story_contract_sha256(contract))
    committed = []
    for episode in range(1, up_to + 1):
        episode_lock = build_episode_lock_for(contract, episode, prior_state)
        candidate = await repository.commit_episode_candidate(
            run_id,
            episode_number=episode,
            content=episode_lock.content,
            episode_lock=episode_lock,
            call_id=f"episode-{episode}-call",
            writer_notes=f"第{episode}集写作备注",
            now=NOW,
        )
        committed.append(candidate)
        prior_state = candidate.series_state
    return contract, active, committed


# ---------------------------------------------------------------------------
# FSW-A3 / FSW-A8: candidate identity, deterministic commit, active pointer
# ---------------------------------------------------------------------------


async def test_commit_binds_design_batch_epoch_predecessor_and_call_id(
    repository: Repository,
) -> None:
    accepted, lease = await create_leased_run(repository)
    contract, active, committed = await seed_batch_with_episodes(repository, lease.run_id, up_to=2)
    first, second = committed

    assert first.episode_number == 1
    assert first.version == 1
    assert first.predecessor_candidate_id is None
    assert first.predecessor_sha256 is None
    assert first.design_candidate_id == active.candidate_id
    assert first.design_content_hash == active.content_hash
    assert first.design_epoch == active.design_epoch
    assert first.call_id == "episode-1-call"
    assert first.status == "active"

    assert second.episode_number == 2
    assert second.version == 1
    assert second.predecessor_candidate_id == first.candidate_id
    assert second.predecessor_sha256 == first.content_sha256
    assert second.call_id == "episode-2-call"

    lineage = await repository.get_script_batch_lineage(lease.run_id)
    assert lineage is not None
    assert lineage.batch_epoch == 1
    assert lineage.status == "active"
    assert lineage.active_pointers == {1: first.candidate_id, 2: second.candidate_id}

    candidates = await repository.get_episode_candidates(lease.run_id)
    assert {candidate.candidate_id for candidate in candidates} == {
        first.candidate_id,
        second.candidate_id,
    }


async def test_commit_rejects_a_candidate_that_fails_deterministic_validation(
    repository: Repository,
) -> None:
    accepted, lease = await create_leased_run(repository)
    contract, active = await seed_active_design_and_batch(repository, lease.run_id)
    prior_state = initial_series_state(contract, story_contract_sha256(contract))
    first_lock = build_episode_lock_for(contract, 1, prior_state)
    first = await repository.commit_episode_candidate(
        lease.run_id,
        episode_number=1,
        content=first_lock.content,
        episode_lock=first_lock,
        call_id="episode-1-call",
        writer_notes="",
        now=NOW,
    )

    # A candidate whose content omits the locked evidence fails deterministic
    # contract/state validation and must never advance the active pointer (A8).
    second_lock = build_episode_lock_for(contract, 2, first.series_state)
    tampered_lock = second_lock.model_copy(update={"content": "缺少逐字证据的剧本。"})
    with pytest.raises(DomainError) as invalid:
        await repository.commit_episode_candidate(
            lease.run_id,
            episode_number=2,
            content=tampered_lock.content,
            episode_lock=tampered_lock,
            call_id="episode-2-call",
            writer_notes="",
            now=NOW,
        )
    assert invalid.value.code == "episode_candidate_invalid"

    # The pointer never advanced; episode 2 is still unfinished.
    assert await repository.first_unfinished_episode(lease.run_id) == 2
    assert [
        draft.episode_number for draft in await repository.get_episode_drafts(lease.run_id)
    ] == [1]


async def test_commit_requires_first_unfinished_and_running_run(
    repository: Repository,
) -> None:
    accepted, lease = await create_leased_run(repository)
    contract, active, committed = await seed_batch_with_episodes(repository, lease.run_id, up_to=1)
    with pytest.raises(DomainError) as out_of_order:
        await repository.commit_episode_candidate(
            lease.run_id,
            episode_number=3,
            content="跳过第二集",
            episode_lock=committed[0],
            call_id="episode-3-call",
            writer_notes="",
            now=NOW,
        )
    assert out_of_order.value.code == "episode_out_of_order"

    async with repository._connection() as connection:
        await connection.execute(
            "UPDATE runs SET state = 'succeeded' WHERE id = ?",
            (str(lease.run_id),),
        )
        await connection.commit()
    with pytest.raises(DomainError) as not_running:
        await repository.commit_episode_candidate(
            lease.run_id,
            episode_number=2,
            content="结束后的提交",
            episode_lock=committed[0],
            call_id="episode-2-call",
            writer_notes="",
            now=NOW,
        )
    assert not_running.value.code == "run_not_running"


# ---------------------------------------------------------------------------
# FSW-A4: restart resumes the same batch at the next episode
# ---------------------------------------------------------------------------


async def test_restart_resumes_the_same_batch_without_rewriting_prefix(
    tmp_path,
) -> None:
    repository = Repository(tmp_path / "pengine.sqlite3")
    await repository.initialize()
    accepted, lease = await create_leased_run(repository)
    contract, active, committed = await seed_batch_with_episodes(repository, lease.run_id, up_to=3)
    first, second, third = committed

    reopened = Repository(tmp_path / "pengine.sqlite3")
    await reopened.initialize()

    lineage = await reopened.get_script_batch_lineage(lease.run_id)
    assert lineage is not None
    assert lineage.batch_epoch == 1
    assert lineage.active_pointers == {
        1: first.candidate_id,
        2: second.candidate_id,
        3: third.candidate_id,
    }
    assert await reopened.first_unfinished_episode(lease.run_id) is None

    active_drafts = await reopened.get_episode_drafts(lease.run_id)
    assert [draft.episode_number for draft in active_drafts] == [1, 2, 3]
    assert [draft.content for draft in active_drafts] == [
        first.content,
        second.content,
        third.content,
    ]


# ---------------------------------------------------------------------------
# FSW-A5 / FSW-A6: suffix rewrite preserves the prefix and replays state
# ---------------------------------------------------------------------------


async def test_suffix_rewrite_preserves_prefix_and_supersedes_suffix(
    repository: Repository,
) -> None:
    accepted, lease = await create_leased_run(repository)
    contract, active, committed = await seed_batch_with_episodes(repository, lease.run_id, up_to=3)
    first, second, third = committed

    result = await repository.rewrite_episode_suffix(lease.run_id, 2, now=NOW)
    assert result["next_episode"] == 2
    assert result["prior_state"] == first.series_state
    assert [candidate.episode_number for candidate in result["prefix_candidates"]] == [1]

    # 1..N-1 active hashes are preserved.
    lineage = await repository.get_script_batch_lineage(lease.run_id)
    assert lineage is not None
    assert lineage.active_pointers == {1: first.candidate_id}

    # Active projection is reset to the prefix.
    drafts = await repository.get_episode_drafts(lease.run_id)
    assert [draft.episode_number for draft in drafts] == [1]

    # Every active suffix candidate is superseded but retained as evidence.
    candidates = await repository.get_episode_candidates(lease.run_id)
    by_id = {candidate.candidate_id: candidate for candidate in candidates}
    assert by_id[second.candidate_id].status == "superseded"
    assert by_id[third.candidate_id].status == "superseded"
    assert by_id[first.candidate_id].status == "active"

    # A rewritten episode 2 starts a new ordered suffix at version 2.
    prior_state = first.series_state
    rewritten_lock = build_episode_lock_for(contract, 2, prior_state)
    rewritten = await repository.commit_episode_candidate(
        lease.run_id,
        episode_number=2,
        content=rewritten_lock.content,
        episode_lock=rewritten_lock,
        call_id="rewrite-episode-2-call",
        writer_notes="重写第2集",
        now=NOW,
    )
    assert rewritten.version == 2
    assert rewritten.predecessor_candidate_id == first.candidate_id
    assert rewritten.predecessor_sha256 == first.content_sha256
    assert rewritten.series_state.locked_through_episode == 2


async def test_suffix_rewrite_starts_new_episode_attempt_cycle_without_losing_history(
    repository: Repository,
) -> None:
    _, lease = await create_leased_run(repository)
    _, _, committed = await seed_batch_with_episodes(repository, lease.run_id, up_to=3)
    first, _, _ = committed

    async with repository._transaction() as connection:
        await connection.executemany(
            """
            INSERT INTO episode_attempts(
                run_id, episode_number, attempt_cycle, attempt_number, recorded_at
            ) VALUES (?, ?, 0, ?, ?)
            """,
            [
                (str(lease.run_id), episode, attempt, NOW.isoformat())
                for episode in (2, 3)
                for attempt in range(1, 4)
            ],
        )

    await repository.rewrite_episode_suffix(lease.run_id, 2, now=NOW)

    assert await repository.get_episode_attempt_cycles(lease.run_id) == {1: 0, 2: 1, 3: 1}
    assert await repository.get_episode_attempt_counts(lease.run_id) == {}
    async with repository._connection() as connection:
        rows = await (
            await connection.execute(
                """
                SELECT episode_number, attempt_cycle, COUNT(*) AS count
                FROM episode_attempts
                WHERE run_id = ?
                GROUP BY episode_number, attempt_cycle
                ORDER BY episode_number, attempt_cycle
                """,
                (str(lease.run_id),),
            )
        ).fetchall()
    assert [(row["episode_number"], row["attempt_cycle"], row["count"]) for row in rows] == [
        (2, 0, 3),
        (3, 0, 3),
    ]

    assert await repository.record_episode_attempt(lease.run_id, 2, now=NOW) == 1
    assert await repository.record_episode_attempt(lease.run_id, 2, now=NOW) == 2
    assert await repository.record_episode_attempt(lease.run_id, 2, now=NOW) == 3
    with pytest.raises(DomainError) as exhausted:
        await repository.record_episode_attempt(lease.run_id, 2, now=NOW)
    assert exhausted.value.code == "attempts_exhausted"
    assert await repository.get_episode_attempt_counts(lease.run_id) == {2: 3}

    lineage = await repository.get_script_batch_lineage(lease.run_id)
    assert lineage is not None
    assert lineage.active_pointers == {1: first.candidate_id}


async def test_suffix_rewrite_state_never_borrows_superseded_suffix_deltas(
    repository: Repository,
) -> None:
    accepted, lease = await create_leased_run(repository)
    contract, active, committed = await seed_batch_with_episodes(repository, lease.run_id, up_to=3)
    first, second, third = committed
    third_state = third.series_state

    result = await repository.rewrite_episode_suffix(lease.run_id, 2, now=NOW)
    replayed = result["prior_state"]
    # The replayed state is exactly the retained prefix; no fact, knowledge,
    # clue, or obligation from the superseded suffix leaks into it.
    assert replayed == first.series_state
    assert replayed.locked_through_episode == 1
    assert set(replayed.established_fact_ids) == {"fact_one"}
    assert third_state != replayed
    assert "fact_three" not in replayed.established_fact_ids


async def test_suffix_rewrite_from_first_episode_replays_initial_state(
    repository: Repository,
) -> None:
    accepted, lease = await create_leased_run(repository)
    contract, active, committed = await seed_batch_with_episodes(repository, lease.run_id, up_to=2)
    contract_hash = story_contract_sha256(contract)
    result = await repository.rewrite_episode_suffix(lease.run_id, 1, now=NOW)
    assert result["prior_state"] == initial_series_state(contract, contract_hash)
    assert result["prefix_candidates"] == []
    assert await repository.first_unfinished_episode(lease.run_id) == 1
    assert await repository.get_episode_drafts(lease.run_id) == []


# ---------------------------------------------------------------------------
# FSW-A7: a design change supersedes the whole batch and restarts at episode 1
# ---------------------------------------------------------------------------


async def test_design_change_supersedes_the_whole_batch_and_restarts_at_one(
    repository: Repository,
) -> None:
    accepted, lease = await create_leased_run(repository)
    contract, active, committed = await seed_batch_with_episodes(repository, lease.run_id, up_to=2)
    first, second = committed

    # A new design (different content hash / epoch) becomes active.
    rebuilt = make_design(
        contract,
        run_id=str(lease.run_id),
        parent_candidate_id=active.candidate_id,
        rebuild_count=1,
        story_outline="重构后的完整故事梗概。",
    )
    evidence = validate_series_bible(rebuilt)
    stored_rebuilt = await repository.register_series_bible_candidate(
        str(lease.run_id), lease.run_id, rebuilt, evidence, now=NOW
    )
    review = bind_global_design_review(
        stored_rebuilt,
        review_call_id="design-review-call-2",
        review_model_id="deepseek-v4-flash",
        passed=True,
        evidence="独立设计审查通过。",
        now=NOW,
    )
    await repository.record_series_bible_review(
        lease.run_id, stored_rebuilt.candidate_id, review, now=NOW
    )
    active_rebuilt = await repository.promote_series_bible(
        lease.run_id, stored_rebuilt.candidate_id, now=NOW
    )
    assert active_rebuilt.design_epoch == active.design_epoch + 1

    # The new batch supersedes the prior batch and starts fresh at episode 1.
    new_batch = await repository.create_script_batch(
        lease.run_id,
        design_candidate_id=active_rebuilt.candidate_id,
        design_content_hash=active_rebuilt.content_hash,
        design_epoch=active_rebuilt.design_epoch,
        now=NOW,
    )
    assert new_batch.batch_epoch == 2
    assert new_batch.active_pointers == {}
    assert await repository.first_unfinished_episode(lease.run_id) == 1
    assert await repository.get_episode_drafts(lease.run_id) == []

    old_lineage = await repository.get_script_batch_lineage(lease.run_id)
    assert old_lineage is not None and old_lineage.batch_id == new_batch.batch_id

    candidates = await repository.get_episode_candidates(lease.run_id)
    by_id = {candidate.candidate_id: candidate for candidate in candidates}
    assert by_id[first.candidate_id].status == "superseded"
    assert by_id[second.candidate_id].status == "superseded"


# ---------------------------------------------------------------------------
# FSW-A9: a late generation is retained as stale and never advances a pointer
# ---------------------------------------------------------------------------


async def test_late_generation_is_stored_stale_and_cannot_advance(
    repository: Repository,
) -> None:
    accepted, lease = await create_leased_run(repository)
    contract, active, committed = await seed_batch_with_episodes(repository, lease.run_id, up_to=1)
    first = committed[0]

    # A stale generation of episode 2 (from a superseded context) is recorded.
    stale_lock = build_episode_lock_for(contract, 2, first.series_state, repair_rounds=1)
    stale = await repository.record_stale_episode_candidate(
        lease.run_id,
        episode_number=2,
        content=stale_lock.content,
        episode_lock=stale_lock,
        call_id="late-episode-2-call",
        writer_notes="迟到生成",
        now=NOW,
    )
    assert stale.status == "stale"
    assert stale.call_id == "late-episode-2-call"

    # The stale record cannot move the active pointer.
    lineage = await repository.get_script_batch_lineage(lease.run_id)
    assert lineage is not None
    assert lineage.active_pointers == {1: first.candidate_id}
    assert await repository.first_unfinished_episode(lease.run_id) == 2
    assert [
        draft.episode_number for draft in await repository.get_episode_drafts(lease.run_id)
    ] == [1]

    candidates = await repository.get_episode_candidates(lease.run_id)
    by_id = {candidate.candidate_id: candidate for candidate in candidates}
    assert by_id[stale.candidate_id].status == "stale"
    assert by_id[first.candidate_id].status == "active"


# ---------------------------------------------------------------------------
# FSW-A11: assembly requires the complete active batch, never incomplete drafts
# ---------------------------------------------------------------------------


async def test_incomplete_active_batch_cannot_assemble_formal_delivery(
    repository: Repository,
) -> None:
    accepted, lease = await create_leased_run(repository)
    contract, active, committed = await seed_batch_with_episodes(repository, lease.run_id, up_to=2)
    with pytest.raises(DomainError) as incomplete:
        await repository.assemble_episode_scripts(lease.run_id)
    assert incomplete.value.code == "episode_sequence_incomplete"


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


async def test_v13_migration_preserves_existing_database(repository) -> None:
    assert SCHEMA_VERSION == 19
    accepted, lease = await create_leased_run(repository)
    contract, active, committed = await seed_batch_with_episodes(repository, lease.run_id, up_to=1)
    async with repository._connection() as connection:
        row = await (await connection.execute("SELECT MAX(version) FROM pengine_schema")).fetchone()
        assert row[0] == SCHEMA_VERSION


# ---------------------------------------------------------------------------
# FSW-A12: a context-overflow pause preserves every committed candidate
# ---------------------------------------------------------------------------


async def test_context_budget_pause_preserves_committed_candidates(
    repository: Repository,
) -> None:
    accepted, lease = await create_leased_run(repository)
    contract, active, committed = await seed_batch_with_episodes(repository, lease.run_id, up_to=2)
    first, second = committed

    await repository.pause_context_budget(
        lease.run_id,
        stage=InternalStage.GENERATING_EPISODE_SCRIPTS,
        safe_message="the verified context limit would be exceeded",
        episode_number=3,
        now=NOW,
    )

    # Zero outbound calls and no candidate is lost or altered.
    lineage = await repository.get_script_batch_lineage(lease.run_id)
    assert lineage is not None
    assert lineage.active_pointers == {1: first.candidate_id, 2: second.candidate_id}
    drafts = await repository.get_episode_drafts(lease.run_id)
    assert [draft.episode_number for draft in drafts] == [1, 2]
    assert drafts[0].content == first.content
    assert drafts[1].content == second.content
    assert await repository.first_unfinished_episode(lease.run_id) == 3
