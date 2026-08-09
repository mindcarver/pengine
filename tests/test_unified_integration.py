"""Integration acceptance for the unified full-series workflow (Delivery #58).

This module assembles the cross-delivery behavior of the atomic SeriesBible design
package, the design-bound versioned script batch, and the bounded structural
review/repair runtime and proves it through the real Worker + Repository + SQLite
+ HTTP path with deterministic fault/race fixtures.

Covered acceptance items:

- INT-A4: a controllable stop after episode 3 resumes the same run at episode 4
  without rewriting episodes 1..3.
- INT-A5: a verified script defect at N preserves 1..N-1, replaces N..end, and
  replays SeriesState; a verified design defect produces one complete new design
  and leaves no active old scripts.
- INT-A6: stale late generations and stale late reviews (released after a new
  epoch) cannot change active content or delivery, while their evidence and token
  usage remain queryable.
- INT-A7: an over-context request produces zero outbound calls, a browser-visible
  required-versus-limit pause, and unchanged readable candidates.
- INT-A8: automatic budget exhaustion and explicit one-cycle authorization each
  complete an HTTP/worker path without an implicit additional content attempt.
- INT-A10: the current-main schema migrates forward in an isolated fixture;
  older records are not falsely represented as verified new-lineage candidates.
- INT-A12: integration evidence stays under an isolated directory with no
  credential-bearing material.

The real configured Opus/DeepSeek routes and the browser acceptance are covered by
``tests/live`` and the acceptance evidence produced against a running server.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
from pathlib import Path
from typing import Any
from uuid import UUID

import pytest
from httpx import ASGITransport, AsyncClient
from persona_factory import create_persona_package

from pengine.agents import EpisodeTimeoutError, MilestoneRejectedError
from pengine.api import create_app
from pengine.config import Settings
from pengine.continuity import (
    EpisodeStateDelta,
    ScriptEvidence,
    SemanticReview,
    StoryContract,
    bind_episode_delta_to_contract,
    build_episode_lock,
    canonical_model_hash,
    initial_series_state,
    render_story_contract_markdown,
    story_contract_sha256,
)
from pengine.personas import PersonaCatalog
from pengine.relay import PreflightBlockedError
from pengine.repository import SCHEMA_VERSION, Repository
from pengine.schemas import (
    ContentPackage,
    CreateCreationRequest,
    EpisodePlan,
    GateResult,
    InternalStage,
    PersonaSnapshot,
    WorkflowResult,
)
from pengine.series_review import active_prefix_hash, effective_milestones
from pengine.worker import Worker

SNAPSHOT_HASH = "a" * 64
_PASS_DECISION: dict[str, Any] = {"category": "pass"}


def _sparse_contract(episode_count: int) -> StoryContract:
    """A general sparse-knowledge contract for ``episode_count`` episodes."""
    fact_names = ("one", "two", "three", "four", "five")
    facts = [
        {
            "fact_id": f"fact_{fact_names[episode - 1]}",
            "subject": "旧案",
            "predicate": f"事实{episode}",
            "kind": "text",
            "value": f"事实{fact_names[episode - 1]}",
            "first_revealed_episode": episode,
        }
        for episode in range(1, episode_count + 1)
    ]
    fact_ids = [fact["fact_id"] for fact in facts]
    return StoryContract.model_validate(
        {
            "version": 1,
            "episode_count": episode_count,
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
            "knowledge_states": [
                {
                    "episode_number": episode,
                    "character_id": character_id,
                    "known_fact_ids": fact_ids[:episode],
                }
                for episode in range(1, episode_count + 1)
                for character_id in ("alice", "bob")
            ],
            "clues": [],
            "prohibitions": [],
            "episode_obligations": [
                {
                    "obligation_id": f"obligation_{episode}",
                    "episode_number": episode,
                    "new_information_fact_ids": [f"fact_{fact_names[episode - 1]}"],
                    "end_hook": f"第{episode}集钩子",
                    "required_clue_ids": [],
                }
                for episode in range(1, episode_count + 1)
            ],
        }
    )


def _three_episode_contract() -> StoryContract:
    return _sparse_contract(3)


def _contract_b_rebuild() -> StoryContract:
    """A complete replacement design contract used by the automatic design rebuild."""
    contract = _three_episode_contract().model_copy(deep=True)
    for fact in contract.facts:
        fact.value = f"{fact.value}（重建版）"
    return contract


def _episode_content(contract: StoryContract, delta: EpisodeStateDelta) -> str:
    """Deterministic Chinese script content containing every evidence excerpt verbatim."""
    lines = [f"第 {delta.episode_number} 集"]
    lines.extend(item.excerpt for item in delta.evidence)
    return "\n".join(lines)


def _episode_lock(
    contract: StoryContract,
    episode: int,
    prior_state,
    *,
    repair_rounds: int = 0,
):
    contract_hash = story_contract_sha256(contract)
    delta = EpisodeStateDelta(
        episode_number=episode,
        contract_sha256=contract_hash,
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
    revealed = [fact.fact_id for fact in contract.facts if fact.first_revealed_episode == episode]
    excerpts = {
        fact_id: next(f.value for f in contract.facts if f.fact_id == fact_id)
        for fact_id in revealed
    }
    obligation = next(
        item for item in contract.episode_obligations if item.episode_number == episode
    )
    excerpts[obligation.obligation_id] = obligation.end_hook
    delta = delta.model_copy(
        update={
            "evidence": [
                ScriptEvidence(target_id=target_id, excerpt=excerpt)
                for target_id, excerpt in excerpts.items()
            ]
        }
    )
    content = _episode_content(contract, delta)
    return build_episode_lock(
        contract=contract,
        contract_sha256=contract_hash,
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


class UnifiedWorkflow:
    """Deterministic workflow driving the full unified SeriesBible path.

    It approves the design stages (including a locked story contract), commits
    versioned episode candidates through the Worker hooks, and registers bound
    structural milestone/final reviews. Behavior is programmable per integration
    acceptance branch:

    - ``decisions`` maps a milestone episode to a review decision. A defect
      raises :class:`MilestoneRejectedError` so the Worker classifies it.
    - ``stop_after_episode`` raises :class:`EpisodeTimeoutError` after that
      episode is committed, exercising restart recovery at the next episode.
    - ``rebuild_contract`` supplies the complete replacement design used on the
      second pass after an automatic design rebuild.
    - ``preflight_block`` raises :class:`PreflightBlockedError` before any
      generation, proving zero outbound content attempts.
    """

    def __init__(
        self,
        *,
        episode_count: int = 3,
        review_milestones: tuple[int, ...] = (),
        decisions: dict[int, dict[str, Any]] | None = None,
        stop_after_episode: int | None = None,
        rebuild_contract: StoryContract | None = None,
        preflight_block: bool = False,
        persistent_defects: bool = False,
    ) -> None:
        self.episode_count = episode_count
        self.review_milestones = list(review_milestones)
        self.decisions = decisions or {}
        self.stop_after_episode = stop_after_episode
        self.rebuild_contract = rebuild_contract
        self.preflight_block = preflight_block
        self.persistent_defects = persistent_defects
        self.pass_count = 0
        self.execution_count = 0
        self.committed: list[int] = []
        self.registered_reviews: list[dict[str, Any]] = []
        self.outbound_content_calls = 0
        self.consumed_defects: set[int] = set()
        self.contract_a = _sparse_contract(episode_count)

    def _pick_contract(self, approved: dict[InternalStage, Any]) -> StoryContract:
        outline = approved.get(InternalStage.GENERATING_EPISODE_OUTLINE)
        if outline is not None:
            return StoryContract.model_validate(outline["story_contract"])
        self.pass_count += 1
        if self.pass_count >= 2 and self.rebuild_contract is not None:
            return self.rebuild_contract
        return self.contract_a

    async def _approve_design(
        self,
        before_stage,
        approve_stage,
        approved: dict[InternalStage, Any],
        contract: StoryContract,
    ) -> None:
        contract_hash = story_contract_sha256(contract)
        payloads: dict[InternalStage, dict[str, Any]] = {
            InternalStage.SELECTING_L0_VARIANT: {
                "stage": "selecting_l0_variant",
                "selected_l0_variant": "主动选择",
                "selection_rationale": "符合测试故事。",
            },
            InternalStage.GENERATING_STORY_OUTLINE: {
                "stage": "generating_story_outline",
                "content": "离乡者回到旧屋处理旧事。",
            },
            InternalStage.GENERATING_CHARACTER_RELATIONSHIPS: {
                "stage": "generating_character_relationships",
                "character_biographies": "\n".join(
                    f"{character.name}：主要人物。" for character in contract.characters
                ),
                "relationship_logic": "主要人物围绕旧案形成合作关系。",
            },
            InternalStage.GENERATING_EPISODE_OUTLINE: {
                "stage": "generating_episode_outline",
                "content": f"{self.episode_count} 集分集大纲",
                "episode_count": self.episode_count,
                "episodes": [
                    {"episode_number": number, "plan": f"第{number}集计划。"}
                    for number in range(1, self.episode_count + 1)
                ],
                "story_contract": contract.model_dump(mode="json"),
                "story_contract_sha256": contract_hash,
                "story_contract_markdown": render_story_contract_markdown(contract, contract_hash),
                "contract_review": {
                    "passed": True,
                    "evidence": "独立合同审查通过。",
                    "issues": [],
                },
                "contract_repair_rounds": 0,
                "review_milestones": self.review_milestones,
            },
        }
        for stage in (
            InternalStage.SELECTING_L0_VARIANT,
            InternalStage.GENERATING_STORY_OUTLINE,
            InternalStage.GENERATING_CHARACTER_RELATIONSHIPS,
            InternalStage.GENERATING_EPISODE_OUTLINE,
        ):
            if stage not in approved:
                await before_stage(stage)
                await approve_stage(stage, payloads[stage])
                approved[stage] = payloads[stage]

    async def execute(self, **kwargs: Any) -> WorkflowResult:
        before_stage = kwargs["before_stage"]
        approve_stage = kwargs["approve_stage"]
        approved = dict(kwargs.get("approved_checkpoints") or {})
        episode_drafts = list(kwargs.get("episode_drafts") or [])
        before_episode = kwargs.get("before_episode")
        commit_episode = kwargs.get("commit_episode")
        assemble_episode_scripts = kwargs.get("assemble_episode_scripts")
        register_series_review = kwargs.get("register_series_review")
        get_series_bible = kwargs.get("get_series_bible")
        reset_episode_deadline = kwargs.get("reset_episode_deadline")

        self.execution_count += 1

        if self.preflight_block:
            await before_stage(InternalStage.GENERATING_STORY_OUTLINE)
            raise PreflightBlockedError(
                role="generation",
                model_id="claude-opus-5",
                stage="generating_story_outline",
                episode_number=None,
                required_tokens=2_000_000,
                verified_limit_tokens=200_000,
            )

        contract = self._pick_contract(approved)
        await self._approve_design(before_stage, approve_stage, approved, contract)
        contract_hash = story_contract_sha256(contract)

        committed = {draft.episode_number: draft for draft in episode_drafts}
        prior_state = initial_series_state(contract, contract_hash)
        for number in range(1, self.episode_count + 1):
            if number in committed:
                prior_state = committed[number].series_state

        milestones: set[int] = set()
        if get_series_bible is not None:
            active = await get_series_bible()
            if active is not None:
                milestones = set(
                    effective_milestones(active.review_milestones, contract.episode_count)
                )

        for episode_number in range(1, self.episode_count + 1):
            if episode_number in committed:
                continue
            if reset_episode_deadline is not None:
                await reset_episode_deadline()
            if before_episode is not None:
                await before_episode(EpisodePlan(episode_number=episode_number, plan="测试计划"))
            lock = _episode_lock(contract, episode_number, prior_state)
            self.outbound_content_calls += 1
            draft = await commit_episode(
                episode_number,
                lock.content,
                lock,
                call_id=f"generation-episode-{episode_number}-pass{self.pass_count}",
                writer_notes=f"第{episode_number}集写作备注",
            )
            self.committed.append(episode_number)
            committed[episode_number] = draft
            prior_state = draft.series_state

            if self.stop_after_episode is not None and episode_number == self.stop_after_episode:
                raise EpisodeTimeoutError(episode_number + 1)

            if episode_number in milestones:
                decision = self.decisions.get(episode_number, _PASS_DECISION)
                if (
                    decision["category"] != "pass"
                    and not self.persistent_defects
                    and episode_number in self.consumed_defects
                ):
                    decision = _PASS_DECISION
                review_id = None
                if register_series_review is not None:
                    review_id = await register_series_review(
                        review_type=(
                            "final" if episode_number == contract.episode_count else "milestone"
                        ),
                        episode_number=episode_number,
                        passed=decision["category"] == "pass",
                        category=decision["category"],
                        evidence=decision.get("evidence", "全系列结构一致。"),
                        earliest_affected_episode=decision.get("earliest_affected_episode"),
                    )
                    self.registered_reviews.append(
                        {
                            "episode_number": episode_number,
                            "decision": decision,
                            "review_id": review_id,
                        }
                    )
                if decision["category"] != "pass":
                    self.consumed_defects.add(episode_number)
                    raise MilestoneRejectedError(
                        category=decision["category"],
                        evidence=decision.get("evidence", "结构审查未通过。"),
                        earliest_affected_episode=decision.get("earliest_affected_episode"),
                        review_id=review_id,
                    )

        aggregate = await assemble_episode_scripts()
        payload: dict[str, Any] = {
            "stage": InternalStage.GENERATING_EPISODE_SCRIPTS.value,
            "content": aggregate,
            "contract_sha256": contract_hash,
            "episode_hashes": [
                {
                    "episode_number": number,
                    "content_sha256": draft.content_sha256,
                    "series_state_sha256": draft.series_state_sha256,
                }
                for number, draft in sorted(committed.items())
            ],
            "series_state_sha256": canonical_model_hash(prior_state),
        }
        if InternalStage.GENERATING_EPISODE_SCRIPTS in approved:
            if approved[InternalStage.GENERATING_EPISODE_SCRIPTS] != payload:
                raise AssertionError(
                    "Episode scripts checkpoint conflicts with committed candidates"
                )
        else:
            # The episode-scripts stage is guarded per episode; it has no stage attempt.
            await approve_stage(InternalStage.GENERATING_EPISODE_SCRIPTS, payload)
            approved[InternalStage.GENERATING_EPISODE_SCRIPTS] = payload

        l0_payload = {"stage": "accepting_l0", "passed": True, "evidence": "L0 证据"}
        l4_payload = {
            "stage": "accepting_l4",
            "passed": True,
            "evidence": "L4 证据",
            "feedback_handling": [],
        }
        for stage, stage_payload in (
            (InternalStage.ACCEPTING_L0, l0_payload),
            (InternalStage.ACCEPTING_L4, l4_payload),
        ):
            if stage not in approved:
                await before_stage(stage)
                await approve_stage(stage, stage_payload)
                approved[stage] = stage_payload

        return WorkflowResult(
            content_package=ContentPackage(
                story_outline=approved[InternalStage.GENERATING_STORY_OUTLINE]["content"],
                character_biographies=approved[InternalStage.GENERATING_CHARACTER_RELATIONSHIPS][
                    "character_biographies"
                ],
                relationship_logic=approved[InternalStage.GENERATING_CHARACTER_RELATIONSHIPS][
                    "relationship_logic"
                ],
                episode_outline=approved[InternalStage.GENERATING_EPISODE_OUTLINE]["content"],
                episode_scripts=aggregate,
            ),
            selected_l0_variant=approved[InternalStage.SELECTING_L0_VARIANT]["selected_l0_variant"],
            selection_rationale=approved[InternalStage.SELECTING_L0_VARIANT]["selection_rationale"],
            l0_gate=GateResult(passed=True, evidence="L0 证据"),
            l4_gate=GateResult(passed=True, evidence="L4 证据"),
            feedback_handling=[],
        )


# ---------------------------------------------------------------------------
# Test services
# ---------------------------------------------------------------------------


def _persona() -> PersonaSnapshot:
    return PersonaSnapshot(
        persona_id="test-persona",
        display_name="非生产测试人格",
        version="fixture-1",
        snapshot_sha256=SNAPSHOT_HASH,
    )


def _request() -> CreateCreationRequest:
    return CreateCreationRequest(
        persona_id="test-persona",
        story="海岛修表师回乡调查旧案。",
        requirements="创作一部完整短剧；全部使用简体中文。",
    )


async def _services(tmp_path: Path):
    persona_root = tmp_path / "personas"
    create_persona_package(persona_root / "active")
    settings = Settings(persona_root=persona_root, data_dir=tmp_path / "data")
    catalog = PersonaCatalog(persona_root, settings.snapshot_root)
    repository = Repository(settings.database_path)
    await repository.initialize()
    return settings, catalog, repository


async def _create_creation(repository: Repository, catalog: PersonaCatalog, key: str):
    snapshot = catalog.create_snapshot("test-persona")
    return await repository.create_creation(
        key,
        _request(),
        snapshot.summary,
    )


def _sqlite_rows(database_path: Path, sql: str, params: tuple = ()) -> list[sqlite3.Row]:
    with sqlite3.connect(f"{database_path.resolve().as_uri()}?mode=ro", uri=True) as connection:
        connection.row_factory = sqlite3.Row
        return connection.execute(sql, params).fetchall()


def _sqlite_schema_version(database_path: Path) -> int:
    rows = _sqlite_rows(database_path, "SELECT MAX(version) AS v FROM pengine_schema")
    return int(rows[0]["v"])


# ---------------------------------------------------------------------------
# INT-A4: resume after episode 3 without rewriting 1..3
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_int_a4_resume_after_episode_three_does_not_rewrite_prefix(
    tmp_path: Path,
) -> None:
    """A controllable stop after episode 3 resumes the same run at episode 4."""
    settings, catalog, repository = await _services(tmp_path)
    workflow = UnifiedWorkflow(episode_count=4, stop_after_episode=3)
    worker = Worker(
        settings=settings,
        repository=repository,
        catalog=catalog,
        workflow=workflow,
        worker_id="a4-worker",
    )

    accepted = await _create_creation(repository, catalog, "a4-create")

    # First pass: the worker commits episodes 1..3 then stops with EpisodeTimeoutError.
    assert await worker.run_once() is True
    timeout_resource = await repository.get_creation(accepted.creation_id)
    assert timeout_resource is not None
    assert timeout_resource.initial.state == "auto_resuming"

    before = {
        row["episode_number"]: (row["candidate_id"], row["version"], row["status"])
        for row in _sqlite_rows(
            settings.database_path,
            "SELECT episode_number, candidate_id, version, status FROM episode_candidates",
        )
    }
    assert sorted(before) == [1, 2, 3]

    # Resume: the same run/batch continues at episode 4.
    await worker.run_once()
    final = await repository.get_creation(accepted.creation_id)
    assert final is not None
    assert final.initial.state == "succeeded", final.initial.failure

    after = {
        row["episode_number"]: (row["candidate_id"], row["version"], row["status"])
        for row in _sqlite_rows(
            settings.database_path,
            "SELECT episode_number, candidate_id, version, status FROM episode_candidates",
        )
    }
    # Episodes 1..3 keep the exact same immutable candidates (no rewrite).
    for episode in (1, 2, 3):
        assert after[episode] == before[episode], f"episode {episode} was rewritten"
    assert 4 in after
    assert workflow.committed == [1, 2, 3, 4]


# ---------------------------------------------------------------------------
# INT-A5: script-defect suffix rewrite and design-defect rebuild
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_int_a5_script_defect_preserves_prefix_and_replays_state(
    tmp_path: Path,
) -> None:
    """A script defect at episode 2 preserves 1, supersedes 2..3, and replays state."""
    settings, catalog, repository = await _services(tmp_path)
    workflow = UnifiedWorkflow(
        episode_count=3,
        decisions={
            3: {
                "category": "script_defect",
                "evidence": "第2集违背合同连续性与义务闭环。",
                "earliest_affected_episode": 2,
            }
        },
    )
    worker = Worker(
        settings=settings,
        repository=repository,
        catalog=catalog,
        workflow=workflow,
        worker_id="a5-script-worker",
    )
    accepted = await _create_creation(repository, catalog, "a5-script")

    # Pass 1: writes 1..3, the milestone at 3 raises a script defect at episode 2.
    await worker.run_once()
    mid = await repository.get_creation(accepted.creation_id)
    assert mid is not None
    assert mid.initial.state in {"queued", "auto_resuming", "running"}

    before_prefix = {
        row["episode_number"]: (row["candidate_id"], row["version"], row["status"])
        for row in _sqlite_rows(
            settings.database_path,
            "SELECT episode_number, candidate_id, version, status FROM episode_candidates",
        )
    }

    # Pass 2: automatic suffix rewrite preserves 1, replaces 2..3.
    await worker.run_once()
    final = await repository.get_creation(accepted.creation_id)
    assert final is not None
    assert final.initial.state == "succeeded", final.initial.failure

    after = {
        row["episode_number"]: (row["candidate_id"], row["version"], row["status"])
        for row in _sqlite_rows(
            settings.database_path,
            "SELECT episode_number, candidate_id, version, status FROM episode_candidates",
        )
    }
    assert after[1] == before_prefix[1], "episode 1 was rewritten"
    assert after[2][1] == before_prefix[2][1] + 1, after[2]
    assert after[3][1] == before_prefix[3][1] + 1, after[3]
    superseded = {
        row["episode_number"]
        for row in _sqlite_rows(
            settings.database_path,
            "SELECT episode_number FROM episode_candidates WHERE status='superseded'",
        )
    }
    assert superseded == {2, 3}

    # SeriesState was replayed strictly from the retained prefix: the rewritten
    # episode 2 binds the retained episode 1 candidate as its predecessor.
    ep2_row = _sqlite_rows(
        settings.database_path,
        "SELECT predecessor_candidate_id, series_state_json FROM episode_candidates "
        "WHERE status='active' AND episode_number=2",
    )[0]
    assert ep2_row["predecessor_candidate_id"] == after[1][0]
    assert json.loads(ep2_row["series_state_json"])["locked_through_episode"] == 2


@pytest.mark.asyncio
async def test_int_a5_design_defect_builds_complete_new_design_and_supersedes_old_scripts(
    tmp_path: Path,
) -> None:
    """A verified design defect produces one complete new design and no active old scripts."""
    settings, catalog, repository = await _services(tmp_path)
    workflow = UnifiedWorkflow(
        episode_count=3,
        review_milestones=(2,),
        decisions={
            2: {
                "category": "design_defect",
                "evidence": "设计审查发现核心事实链不可用。",
            }
        },
        rebuild_contract=_contract_b_rebuild(),
    )
    worker = Worker(
        settings=settings,
        repository=repository,
        catalog=catalog,
        workflow=workflow,
        worker_id="a5-design-worker",
    )
    accepted = await _create_creation(repository, catalog, "a5-design")

    await worker.run_once()  # pass 1 writes up to milestone 2 then rejects the design
    await worker.run_once()  # pass 2 rebuilds the design and writes the series

    final = await repository.get_creation(accepted.creation_id)
    assert final is not None
    assert final.initial.state == "succeeded", final.initial.failure

    designs = _sqlite_rows(
        settings.database_path,
        "SELECT candidate_id, content_hash, status, design_epoch "
        "FROM series_bible_candidates ORDER BY created_at",
    )
    assert len(designs) == 2
    assert designs[0]["status"] == "superseded"
    assert designs[1]["status"] == "active"
    assert designs[1]["design_epoch"] == designs[0]["design_epoch"] + 1
    assert designs[0]["content_hash"] != designs[1]["content_hash"]

    batches = _sqlite_rows(
        settings.database_path,
        "SELECT batch_id, status FROM script_batches ORDER BY created_at",
    )
    assert len(batches) == 2
    assert batches[0]["status"] == "superseded"
    assert batches[1]["status"] == "active"

    active = _sqlite_rows(
        settings.database_path,
        "SELECT episode_number, design_candidate_id FROM episode_candidates WHERE status='active'",
    )
    assert len(active) == 3
    assert all(row["design_candidate_id"] == designs[1]["candidate_id"] for row in active)

    lineage = _sqlite_rows(
        settings.database_path,
        "SELECT rebuild_count FROM series_bible_lineage",
    )
    assert lineage and int(lineage[0]["rebuild_count"]) == 1


# ---------------------------------------------------------------------------
# INT-A6: stale late responses cannot change active content or delivery
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_int_a6_stale_late_generation_and_review_remain_queryable(
    tmp_path: Path,
) -> None:
    """Old generation/review released after a new epoch cannot change active content."""
    from test_repository import create_and_lease_initial
    from test_script_batch import seed_batch_with_episodes

    from pengine.series_bible import (
        bind_global_design_review,
        build_series_bible,
        validate_series_bible,
    )

    settings, catalog, repository = await _services(tmp_path)
    accepted, lease = await create_and_lease_initial(repository, _persona(), _request())
    run_id = lease.run_id
    contract, active, committed = await seed_batch_with_episodes(repository, run_id, up_to=3)

    batch = await repository.get_script_batch_lineage(run_id)
    assert batch is not None
    prefix_hash = active_prefix_hash(
        [
            {"episode_number": c.episode_number, "content_sha256": c.content_sha256}
            for c in committed
        ]
    )
    # The real passing final review that authorizes delivery on the active lineage.
    real_review = await repository.register_series_review(
        run_id,
        review_type="final",
        episode_number=3,
        design_candidate_id=active.candidate_id,
        design_content_hash=active.content_hash,
        design_epoch=active.design_epoch,
        batch_id=batch.batch_id,
        batch_epoch=batch.batch_epoch,
        prefix_hash=prefix_hash,
        call_id="real-final-review",
        passed=True,
        category="pass",
        evidence="完整系列通过。",
        earliest_affected_episode=None,
    )

    # ---- Epoch change: a rebuilt design supersedes the old design and batch ----
    rebuilt = build_series_bible(
        run_id=str(run_id),
        run_kind="initial",
        l0_variant="主动选择",
        genre="general",
        story_outline="离乡者回到旧屋处理旧事。",
        character_biographies="\n".join(f"{c.name}：主要人物。" for c in contract.characters),
        relationship_logic="主要人物围绕旧案形成合作关系。",
        episode_outline="三集分集大纲",
        story_contract_payload=_contract_b_rebuild().model_dump(mode="json"),
        parent_candidate_id=active.candidate_id,
        rebuild_count=1,
        design_epoch=active.design_epoch + 1,
    )
    evidence = validate_series_bible(rebuilt)
    assert evidence.passed
    await repository.rebuild_series_bible(str(accepted.creation_id), run_id, rebuilt, evidence)
    design_b_review = bind_global_design_review(
        rebuilt,
        review_call_id="design-b-review",
        review_model_id="deepseek-v4-flash",
        passed=True,
        evidence="重建设计通过。",
    )
    await repository.record_series_bible_review(run_id, rebuilt.candidate_id, design_b_review)
    await repository.promote_series_bible(run_id, rebuilt.candidate_id)
    await repository.mark_series_bible_stale(run_id, active_candidate_id=rebuilt.candidate_id)

    # Supersede the old script batch and open a fresh batch for the new design.
    async with repository._connection() as connection:
        await repository._supersede_active_batch(
            connection,
            run_id,
            await repository._fetch_script_batch_lineage(connection, run_id),
            "2026-08-04T00:00:00+00:00",
        )
    new_batch = await repository.create_script_batch(
        run_id,
        design_candidate_id=rebuilt.candidate_id,
        design_content_hash=rebuilt.content_hash,
        design_epoch=rebuilt.design_epoch,
    )

    # ---- A late generation bound to the old epoch is retained stale ----
    contract_hash = story_contract_sha256(contract)
    prior = initial_series_state(contract, contract_hash)
    lock_ep1 = _episode_lock(contract, 1, prior)
    late_lock = _episode_lock(contract, 2, lock_ep1.series_state)
    stale_candidate = await repository.record_stale_episode_candidate(
        run_id,
        episode_number=2,
        content="迟到的旧时代生成内容。",
        episode_lock=late_lock,
        call_id="late-generation-call",
        writer_notes="旧时代迟到结果",
    )
    assert stale_candidate.status == "stale"
    assert stale_candidate.batch_id == new_batch.batch_id  # bound to the current batch
    # The active pointer is unchanged (the new design's batch has no episode 2 yet).
    stale_active_ep2 = _sqlite_rows(
        settings.database_path,
        "SELECT COUNT(*) AS n FROM episode_candidates WHERE status='active' AND episode_number=2",
    )[0]["n"]
    assert stale_active_ep2 == 0

    # ---- A late review bound to the old design cannot approve the new lineage ----
    await repository.register_series_review(
        run_id,
        review_type="final",
        episode_number=3,
        design_candidate_id=active.candidate_id,
        design_content_hash=active.content_hash,
        design_epoch=active.design_epoch,
        batch_id=batch.batch_id,
        batch_epoch=batch.batch_epoch,
        prefix_hash=prefix_hash,
        call_id="late-review-call",
        passed=True,
        category="pass",
        evidence="旧时代的迟到通过。",
        earliest_affected_episode=None,
    )
    # A new-lineage final review retires the late old-lineage review.
    new_prefix = active_prefix_hash([])
    new_review = await repository.register_series_review(
        run_id,
        review_type="final",
        episode_number=3,
        design_candidate_id=rebuilt.candidate_id,
        design_content_hash=rebuilt.content_hash,
        design_epoch=rebuilt.design_epoch,
        batch_id=new_batch.batch_id,
        batch_epoch=new_batch.batch_epoch,
        prefix_hash=new_prefix,
        call_id="new-final-review",
        passed=True,
        category="pass",
        evidence="新系列通过。",
        earliest_affected_episode=None,
    )
    statuses = {
        row["call_id"]: row["status"]
        for row in _sqlite_rows(
            settings.database_path,
            "SELECT call_id, status FROM series_reviews",
        )
    }
    assert statuses["late-review-call"] == "stale"
    assert statuses["new-final-review"] == "active"

    # The late old-lineage review can never be the delivery gate for the new lineage.
    gate = await repository.get_latest_passing_final_review(
        run_id,
        design_content_hash=rebuilt.content_hash,
        batch_id=new_batch.batch_id,
        prefix_hash=new_prefix,
    )
    assert gate is not None and gate.review_id == new_review.review_id

    # ---- Stale evidence and token usage remain queryable ----
    assert _sqlite_rows(
        settings.database_path,
        "SELECT call_id FROM episode_candidates WHERE call_id='late-generation-call'",
    )
    assert _sqlite_rows(
        settings.database_path,
        "SELECT call_id FROM series_reviews WHERE call_id='late-review-call'",
    )
    assert _sqlite_rows(
        settings.database_path,
        "SELECT call_id FROM series_reviews WHERE call_id='real-final-review'",
    )
    assert real_review.review_id is not None


# ---------------------------------------------------------------------------
# INT-A7: over-context zero-outbound pause
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_int_a7_context_overflow_pauses_with_zero_outbound_calls(
    tmp_path: Path,
) -> None:
    """An over-context request produces zero outbound calls and unchanged candidates."""
    settings, catalog, repository = await _services(tmp_path)
    workflow = UnifiedWorkflow(episode_count=3, preflight_block=True)
    worker = Worker(
        settings=settings,
        repository=repository,
        catalog=catalog,
        workflow=workflow,
        worker_id="a7-worker",
    )
    accepted = await _create_creation(repository, catalog, "a7-context")

    await worker.run_once()
    resource = await repository.get_creation(accepted.creation_id)
    assert resource is not None
    assert resource.initial.state == "paused"
    pause = resource.initial.pause
    assert pause is not None
    assert pause.code == "context_budget"
    assert "2,000,000" in pause.message or "2000000" in pause.message

    # Zero outbound content attempts and zero committed candidates.
    assert workflow.outbound_content_calls == 0
    assert (
        _sqlite_rows(
            settings.database_path,
            "SELECT COUNT(*) AS n FROM episode_candidates",
        )[0]["n"]
        == 0
    )
    assert (
        _sqlite_rows(
            settings.database_path,
            "SELECT COUNT(*) AS n FROM series_bible_candidates",
        )[0]["n"]
        == 0
    )
    assert resource.initial.progress.recovery_reason == "context_budget"


# ---------------------------------------------------------------------------
# INT-A8: budget exhaustion and exact one-cycle authorization
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_int_a8_suffix_budget_exhaustion_and_one_cycle_authorization(
    tmp_path: Path,
) -> None:
    """Automatic budget exhaustion pauses; one authorize cycle permits one attempt."""
    settings, catalog, repository = await _services(tmp_path)
    workflow = UnifiedWorkflow(
        episode_count=3,
        decisions={
            3: {
                "category": "script_defect",
                "evidence": "终局未满足义务闭环。",
                "earliest_affected_episode": 3,
            }
        },
        persistent_defects=True,
    )
    worker = Worker(
        settings=settings,
        repository=repository,
        catalog=catalog,
        workflow=workflow,
        worker_id="a8-suffix-worker",
    )
    app = create_app(settings=settings, repository=repository, catalog=catalog)
    async with (
        app.router.lifespan_context(app),
        AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client,
    ):
        accepted = await client.post(
            "/creations",
            headers={"Idempotency-Key": "a8-create"},
            json=_request().model_dump(mode="json"),
        )
        assert accepted.status_code == 202
        creation_id = UUID(accepted.json()["creation_id"])

        # Pass 1: automatic suffix rewrite consumes the shared budget.
        await worker.run_once()
        mid = await repository.get_creation(creation_id)
        assert mid is not None
        assert mid.initial.state in {"queued", "auto_resuming", "running"}

        # Pass 2: the automatic budget is exhausted -> repair-authorization pause.
        await worker.run_once()
        paused = await repository.get_creation(creation_id)
        assert paused is not None
        assert paused.initial.state == "paused"
        assert paused.initial.pause.code == "repair_authorization"
        authorization = paused.initial.authorization
        assert authorization is not None
        assert authorization.kind == "suffix_rewrite"
        assert authorization.earliest_affected_episode == 3
        assert authorization.consumed_at is None
        cycles_before = workflow.execution_count

        # HTTP: authorize exactly one generation-plus-review cycle.
        authorized = await client.post(
            f"/creations/{creation_id}/runs/initial/authorize-repair",
            headers={"Idempotency-Key": "a8-authorize"},
        )
        assert authorized.status_code == 202
        assert authorized.json()["run_state"] == "queued"
        await worker.run_once()

        # Exactly one cycle ran, then the fresh review produced a new authorization pause.
        assert workflow.execution_count == cycles_before + 1
        paused_again = await repository.get_creation(creation_id)
        assert paused_again is not None
        assert paused_again.initial.state == "paused"
        assert paused_again.initial.pause.code == "repair_authorization"
        consumed = _sqlite_rows(
            settings.database_path,
            "SELECT consumed_at FROM repair_authorizations "
            "WHERE kind='suffix_rewrite' ORDER BY authorization_epoch",
        )
        assert consumed and consumed[0]["consumed_at"] is not None


@pytest.mark.asyncio
async def test_int_a8_design_budget_exhaustion_and_authorization_cycle(
    tmp_path: Path,
) -> None:
    """A second design defect after the automatic rebuild pauses for authorization."""
    settings, catalog, repository = await _services(tmp_path)
    workflow = UnifiedWorkflow(
        episode_count=3,
        review_milestones=(2,),
        decisions={
            2: {
                "category": "design_defect",
                "evidence": "设计仍存在结构性缺陷。",
            }
        },
        rebuild_contract=_contract_b_rebuild(),
        persistent_defects=True,
    )
    worker = Worker(
        settings=settings,
        repository=repository,
        catalog=catalog,
        workflow=workflow,
        worker_id="a8-design-worker",
    )
    accepted = await _create_creation(repository, catalog, "a8-design")

    # Pass 1 triggers the automatic rebuild; pass 2 sees the same defect -> pause.
    await worker.run_once()
    await worker.run_once()
    paused = await repository.get_creation(accepted.creation_id)
    assert paused is not None
    assert paused.initial.state == "paused"
    assert paused.initial.pause.code == "repair_authorization"
    assert paused.initial.authorization.kind == "design_rebuild"

    # #57 repair: an explicit one-cycle authorization (RPR-A9) permits exactly one
    # further complete design rebuild even after the automatic budget is consumed.
    # The HTTP surface accepts the authorization (202, run requeued) and the next
    # worker cycle completes that one rebuild before pausing again at the same
    # repair-authorization evidence pause. The automatic budget stays consumed
    # (rebuild_count remains 1).
    cycles_before = workflow.execution_count
    app = create_app(settings=settings, repository=repository, catalog=catalog)
    async with (
        app.router.lifespan_context(app),
        AsyncClient(
            transport=ASGITransport(app=app, raise_app_exceptions=False),
            base_url="http://test",
        ) as client,
    ):
        authorized = await client.post(
            f"/creations/{accepted.creation_id}/runs/initial/authorize-repair",
            headers={"Idempotency-Key": "a8-design-authorize"},
        )
        assert authorized.status_code == 202
        assert authorized.json()["run_state"] == "queued"

        # A real provider would produce fresh design content for the granted
        # rebuild cycle. The deterministic workflow keeps one fixed rebuild
        # contract, so a rebuild would regenerate byte-identical content and be
        # skipped as a no-op (same content hash). Swap in a distinct second-round
        # contract to prove the granted cycle actually rebuilds the design into a
        # new active epoch instead of short-circuiting on the hash match.
        second_round = _contract_b_rebuild()
        for fact in second_round.facts:
            fact.value = f"{fact.value}（授权重建版）"
        workflow.rebuild_contract = second_round

        await worker.run_once()

        # Exactly one authorized design-rebuild cycle ran, then the fresh review
        # produced a new repair-authorization pause.
        assert workflow.execution_count == cycles_before + 1
    still_paused = await repository.get_creation(accepted.creation_id)
    assert still_paused is not None
    assert still_paused.initial.state == "paused"
    assert still_paused.initial.pause.code == "repair_authorization"

    # The authorized one-cycle rebuild completed: the consumed authorization is
    # bound to the rebuilt candidate, that candidate is the single active design
    # one epoch above the granted design, the automatic budget stays consumed,
    # and the new evidence pause raised a fresh unconsumed authorization.
    consumed = _sqlite_rows(
        settings.database_path,
        "SELECT consumed_at, rebuild_candidate_id, design_epoch "
        "FROM repair_authorizations "
        "WHERE kind='design_rebuild' AND consumed_at IS NOT NULL "
        "ORDER BY authorization_epoch DESC LIMIT 1",
    )
    assert consumed and consumed[0]["consumed_at"] is not None
    assert consumed[0]["rebuild_candidate_id"]
    active = _sqlite_rows(
        settings.database_path,
        "SELECT candidate_id, design_epoch FROM series_bible_candidates WHERE status='active'",
    )
    assert len(active) == 1
    assert active[0]["candidate_id"] == consumed[0]["rebuild_candidate_id"]
    assert int(active[0]["design_epoch"]) == int(consumed[0]["design_epoch"]) + 1
    lineage = _sqlite_rows(
        settings.database_path,
        "SELECT rebuild_count FROM series_bible_lineage",
    )
    assert lineage and int(lineage[0]["rebuild_count"]) == 1
    unconsumed = _sqlite_rows(
        settings.database_path,
        "SELECT COUNT(*) AS n FROM repair_authorizations "
        "WHERE kind='design_rebuild' AND consumed_at IS NULL",
    )
    assert int(unconsumed[0]["n"]) == 1


# ---------------------------------------------------------------------------
# INT-A10: current-main schema migration without false new-lineage claims
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_int_a10_migration_preserves_legacy_records_without_false_lineage(
    tmp_path: Path,
) -> None:
    """A current-main database migrates forward; legacy records stay legacy."""
    from test_worker import DeterministicWorkflow

    settings, catalog, repository = await _services(tmp_path)

    # Create a legacy (pre-unified) run and delivery through the legacy path.
    accepted = await _create_creation(repository, catalog, "migration-1")
    legacy_worker = Worker(
        settings=settings,
        repository=repository,
        catalog=catalog,
        workflow=DeterministicWorkflow(episode_count=2),
        worker_id="legacy-worker",
    )
    assert await legacy_worker.run_once() is True
    legacy = await repository.get_creation(accepted.creation_id)
    assert legacy is not None and legacy.initial.state == "succeeded"

    # Simulate the pre-#54 "current-main" schema: drop every new-lineage table and
    # roll the schema version back to 10 so re-initialization is a real migration.
    async with repository._connection() as connection:
        for table in (
            "model_calls",
            "series_bible_candidates",
            "series_bible_lineage",
            "script_batches",
            "episode_candidates",
            "series_reviews",
            "repair_authorizations",
        ):
            await connection.execute(f"DROP TABLE IF EXISTS {table}")
        for version in range(11, SCHEMA_VERSION + 1):
            await connection.execute("DELETE FROM pengine_schema WHERE version = ?", (version,))
        await connection.commit()
    assert _sqlite_schema_version(settings.database_path) == 10

    migrated = Repository(settings.database_path)
    await migrated.initialize()
    assert _sqlite_schema_version(settings.database_path) == SCHEMA_VERSION

    # The legacy delivery and run remain readable.
    migrated_resource = await migrated.get_creation(accepted.creation_id)
    assert migrated_resource is not None
    assert migrated_resource.initial.state == "succeeded"
    assert migrated_resource.initial.result.delivery_report.persona_id == "test-persona"

    # Older records are not falsely represented as verified new-lineage candidates.
    async with migrated._connection() as connection:
        for table in (
            "series_bible_candidates",
            "script_batches",
            "episode_candidates",
            "series_reviews",
            "repair_authorizations",
        ):
            cursor = await connection.execute(f"SELECT COUNT(*) FROM {table}")
            (count,) = await cursor.fetchone()
            assert count == 0, f"{table} gained {count} false new-lineage rows"

    # No legacy generation path is maintained: a fresh unified run produces only
    # new-lineage records through the new worker path.
    new_workflow = UnifiedWorkflow(episode_count=3)
    new_worker = Worker(
        settings=settings,
        repository=migrated,
        catalog=catalog,
        workflow=new_workflow,
        worker_id="new-worker",
    )
    fresh = await _create_creation(migrated, catalog, "migration-2")
    assert await new_worker.run_once() is True
    fresh_resource = await migrated.get_creation(fresh.creation_id)
    assert fresh_resource is not None and fresh_resource.initial.state == "succeeded"
    async with migrated._connection() as connection:
        cursor = await connection.execute(
            "SELECT COUNT(*) FROM series_bible_candidates WHERE status='active'"
        )
        (active_designs,) = await cursor.fetchone()
        cursor = await connection.execute(
            "SELECT COUNT(*) FROM series_reviews WHERE review_type='final' AND status='active'"
        )
        (active_final_reviews,) = await cursor.fetchone()
    assert active_designs == 1
    assert active_final_reviews == 1


# ---------------------------------------------------------------------------
# Live-harness helper validation (used by INT-A2 / INT-A9 real-model runs)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_unified_delivery_facts_helper_validates_seeded_run(
    tmp_path: Path,
) -> None:
    """The real-model unified-delivery evidence helper accepts a genuine unified run."""
    import sys

    live_dir = Path(__file__).resolve().parent / "live"
    if str(live_dir) not in sys.path:
        sys.path.insert(0, str(live_dir))
    from test_real_model_e2e import _assert_unified_delivery_facts

    settings, catalog, repository = await _services(tmp_path)
    workflow = UnifiedWorkflow(episode_count=3)
    worker = Worker(
        settings=settings,
        repository=repository,
        catalog=catalog,
        workflow=workflow,
        worker_id="unified-helper-worker",
    )
    accepted = await _create_creation(repository, catalog, "unified-helper")
    await worker.run_once()
    resource = await repository.get_creation(accepted.creation_id)
    assert resource is not None and resource.initial.state == "succeeded"

    # Seed provider-reported usage rows for both roles on this run (as the relay
    # audit would have recorded for real calls).
    review_model = settings.review_model_id or "gpt-5.5"
    with sqlite3.connect(settings.database_path) as connection:
        for role, model in (("generation", "claude-opus-5"), ("review", review_model)):
            connection.execute(
                """
                INSERT INTO model_calls(
                    call_id, run_id, creation_id, role, adapter, provider, model,
                    stage, requested_at, estimated_input_tokens, estimated_output_tokens,
                    estimated_total_tokens, verified_limit_tokens, preflight, status,
                    usage_status, actual_input_tokens, actual_output_tokens,
                    finish_reason, outcome, finished_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 100, 50, 150, 200000, 'ok',
                          'succeeded', 'reported', 120, 60, 'stop', 'success', ?)
                """,
                (
                    f"{role}-usage-call",
                    None,
                    str(accepted.creation_id),
                    role,
                    "anthropic"
                    if role == "generation"
                    else ("openai" if review_model == "gpt-5.5" else "deepseek"),
                    model.split("-")[0],
                    model,
                    "generating_episode_scripts",
                    "2026-08-04T00:00:00+00:00",
                    "2026-08-04T00:00:01+00:00",
                ),
            )
        connection.commit()

    # Fill run_id after obtaining it from the runs table.
    with sqlite3.connect(settings.database_path) as connection:
        run_id = connection.execute(
            "SELECT id FROM runs WHERE creation_id = ? AND kind='initial'",
            (str(accepted.creation_id),),
        ).fetchone()[0]
        connection.execute(
            "UPDATE model_calls SET run_id = ? WHERE call_id LIKE '%-usage-call'", (run_id,)
        )
        connection.commit()

    (tmp_path / "evidence").mkdir(exist_ok=True)
    summary = _assert_unified_delivery_facts(
        settings.database_path,
        creation_id=str(accepted.creation_id),
        generation_model_id="claude-opus-5",
        review_model_id=review_model,
        evidence_dir=tmp_path / "evidence",
    )
    assert summary["status"] == "passed"
    assert summary["episode_candidates"] == 3
    assert summary["active_final_review_id"]
    assert summary["provider_reported_usage"]["generation"]["calls"] >= 1
    assert summary["provider_reported_usage"]["review"]["calls"] >= 1
    assert (tmp_path / "evidence" / "unified-delivery-facts.json").exists()


# ---------------------------------------------------------------------------
# INT-A12: evidence isolation and credential safety
# ---------------------------------------------------------------------------


def test_int_a12_evidence_isolation_and_credential_safety(tmp_path: Path) -> None:
    """Integration evidence stays under an isolated directory and never leaks secrets."""
    import sys

    live_dir = Path(__file__).resolve().parent / "live"
    sys.path.insert(0, str(live_dir))
    from test_real_model_e2e import _assert_evidence_has_no_secrets, _redact_log

    evidence_dir = tmp_path / "isolated-evidence"
    evidence_dir.mkdir()
    (evidence_dir / "metadata.json").write_text(
        json.dumps({"relay": {"base_url": "https://example.invalid/v1", "key_present": True}}),
        encoding="utf-8",
    )
    log_path = evidence_dir / "server.log"
    log_path.write_text(
        "Authorization: Bearer sk-example-not-a-real-secret\n",
        encoding="utf-8",
    )
    (evidence_dir / "resource.json").write_text("{}", encoding="utf-8")

    secret = "sk-example-not-a-real-secret"
    _redact_log(log_path, secret)
    assert secret not in log_path.read_text(encoding="utf-8")
    assert "Bearer" not in log_path.read_text(encoding="utf-8")
    _assert_evidence_has_no_secrets(evidence_dir, secret)

    # Evidence lives under the isolated run directory, never the repo data dir.
    relative = evidence_dir.relative_to(tmp_path)
    assert "data" not in relative.parts


# ---------------------------------------------------------------------------
# INT-A1/A9 enabler: the audit store never deadlocks with the LangGraph saver
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_audit_store_write_does_not_deadlock_with_checkpointer(
    tmp_path: Path,
) -> None:
    """The model-call audit persistence is dispatched off the loop thread.

    A synchronous SQLite write on the loop thread deadlocks against the LangGraph
    AsyncSqliteSaver: the sync write blocks the loop while the saver holds the
    database write lock and needs the loop to run its queued commit. The audit
    handler must therefore persist through a writer thread and the run must drain
    it before finalization (Delivery #58 INT-A1/A2/A9).
    """
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

    from pengine.model_calls import (
        ModelCallContext,
        ModelCallStore,
        build_started_record,
        new_call_id,
    )
    from pengine.relay import drain_audit_writes, submit_store_write
    from pengine.repository import Repository

    database = tmp_path / "pengine.sqlite3"
    repository = Repository(database)
    await repository.initialize()
    store = ModelCallStore(database)
    context = ModelCallContext(run_id="run-1", stage="generating_story_outline")

    async with AsyncSqliteSaver.from_conn_string(str(database)) as saver:
        await saver.setup()

        async def saver_write(index: int) -> None:
            async with saver.lock:
                await saver.conn.execute(
                    "INSERT OR REPLACE INTO checkpoints "
                    "(thread_id, checkpoint_ns, checkpoint_id, parent_checkpoint_id, "
                    "type, checkpoint, metadata) VALUES (?, ?, ?, ?, ?, ?, ?)",
                    ("t1", "", f"checkpoint-{index}", None, "json", b"{}", b"{}"),
                )
                await asyncio.sleep(0.005)
                await saver.conn.commit()

        for index in range(20):
            task = asyncio.create_task(saver_write(index))
            record = build_started_record(
                role="generation",
                adapter="anthropic",
                provider="anthropic",
                model="claude-opus-5",
                context=context,
                estimated_input_tokens=10,
                estimated_output_tokens=100,
                verified_limit_tokens=200_000,
            )
            record.call_id = new_call_id()
            submit_store_write(store.upsert, record)
            await task
        await asyncio.to_thread(drain_audit_writes)

    rows = _sqlite_rows(
        database,
        "SELECT COUNT(*) AS n FROM model_calls WHERE run_id = 'run-1'",
    )
    assert rows[0]["n"] == 20
    store.close()
