import hashlib
import sqlite3
from datetime import UTC, datetime, timedelta

import aiosqlite
import pytest
from test_continuity import make_contract, make_delta

from pengine.continuity import (
    SemanticReview,
    build_episode_lock,
    initial_series_state,
    render_story_contract_markdown,
    story_contract_sha256,
)
from pengine.errors import DomainError
from pengine.model_calls import (
    ModelCallContext,
    ModelCallRecord,
    ModelCallStore,
    build_started_record,
    new_call_id,
)
from pengine.repository import MAX_STAGE_ATTEMPTS, SCHEMA_VERSION, Repository
from pengine.schemas import (
    ContentPackage,
    CreateCreationRequest,
    Delivery,
    DeliveryReport,
    FeedbackHandlingItem,
    GateResult,
    InternalStage,
    PersonaSnapshot,
    RevisionRequest,
    RunFailure,
)

NOW = datetime(2026, 7, 28, 12, 0, tzinfo=UTC)
SNAPSHOT_HASH = "a" * 64


@pytest.fixture
async def repository(tmp_path):
    value = Repository(tmp_path / "pengine.sqlite3")
    await value.initialize()
    return value


@pytest.fixture
def persona() -> PersonaSnapshot:
    return PersonaSnapshot(
        persona_id="fixture-writer",
        display_name="非生产测试人格",
        version="fixture-1",
        snapshot_sha256=SNAPSHOT_HASH,
    )


@pytest.fixture
def creation_request() -> CreateCreationRequest:
    return CreateCreationRequest(
        persona_id="fixture-writer",
        story="一个离乡的人回家处理旧屋。",
        requirements="创作一部当代短剧。",
    )


def make_delivery(*, revised: bool = False) -> Delivery:
    handling = (
        [
            FeedbackHandlingItem(
                feedback_item="加强结尾行动。",
                handling="重写终场。",
                result="主角以行动完成选择。",
            )
        ]
        if revised
        else []
    )
    return Delivery(
        content_package=ContentPackage(
            story_outline="故事梗概",
            character_biographies="人物小传",
            relationship_logic="人物关系逻辑",
            episode_outline="分集大纲",
            episode_scripts="分集剧本",
        ),
        delivery_report=DeliveryReport(
            persona_id="fixture-writer",
            persona_version="fixture-1",
            persona_snapshot_sha256=SNAPSHOT_HASH,
            selected_l0_variant="归返",
            selection_rationale="与故事输入一致。",
            l0_gate=GateResult(passed=True, evidence="L0 证据"),
            l4_gate=GateResult(passed=True, evidence="L4 证据"),
            ownership_statement="本次创作归当前任务所有。",
            feedback_handling=handling,
        ),
    )


async def register_test_user(
    repository: Repository,
    username: str,
    *,
    now: datetime = NOW,
):
    return await repository.register_user(
        username=username,
        password_hash=f"hash-{username}",
        session_token_sha256=f"token-{username}",
        session_expires_at=now + timedelta(days=1),
        now=now,
    )


@pytest.mark.asyncio
async def test_account_fair_leasing_and_queue_positions(
    repository,
    persona,
    creation_request,
) -> None:
    alice = await register_test_user(repository, "alice")
    bob = await register_test_user(repository, "bob")
    alice_first = await repository.create_creation(
        "alice-first",
        creation_request,
        persona,
        owner_id=alice.user_id,
        now=NOW,
    )
    alice_second = await repository.create_creation(
        "alice-second",
        creation_request,
        persona,
        owner_id=alice.user_id,
        now=NOW + timedelta(seconds=1),
    )
    bob_first = await repository.create_creation(
        "bob-first",
        creation_request,
        persona,
        owner_id=bob.user_id,
        now=NOW + timedelta(seconds=2),
    )

    alice_resource = await repository.get_creation(alice_first.creation_id)
    alice_library = await repository.list_creations(alice.user_id)
    bob_resource = await repository.get_creation(bob_first.creation_id)
    assert alice_resource is not None
    assert bob_resource is not None
    assert alice_resource.initial.state == "queued"
    assert alice_resource.initial.queue_position == 1
    assert alice_library.items[0].creation_id == alice_second.creation_id
    assert alice_library.items[0].queue_position == 2
    assert bob_resource.initial.queue_position == 3

    first_lease = await repository.lease_next_job("slot-1", 30, now=NOW)
    second_lease = await repository.lease_next_job(
        "slot-2",
        30,
        now=NOW + timedelta(seconds=2),
    )

    assert first_lease is not None
    assert second_lease is not None
    assert first_lease.creation_id == alice_first.creation_id
    assert second_lease.creation_id == bob_first.creation_id
    assert (
        await repository.lease_next_job(
            "slot-3",
            30,
            now=NOW + timedelta(seconds=2),
        )
        is None
    )

    queued_alice = await repository.get_creation(alice_second.creation_id)
    assert queued_alice is not None
    assert queued_alice.initial.state == "queued"
    assert queued_alice.initial.queue_position == 1


async def test_record_outline_markdown_failure_persists_immutable_evidence(
    repository,
    persona,
    creation_request,
) -> None:
    accepted, lease = await create_and_lease_initial(repository, persona, creation_request)
    await repository.record_outline_markdown_failure(
        lease.run_id,
        group_id="grp_broken",
        operation_id="op-1",
        attempt_index=3,
        raw_text="开场导语。\n\n## 第1集：标题\n\n正文",
        normalized_text="## 第1集\n\n正文",
        parse_error="Outline Markdown headings must exactly cover the current group in order",
        now=NOW,
    )
    async with repository._connection() as connection:
        rows = await (
            await connection.execute(
                """
                SELECT group_id, attempt_index, raw_text, normalized_text, parse_error,
                       raw_text_sha256, normalized_text_sha256
                FROM outline_markdown_failures WHERE run_id = ?
                """,
                (str(lease.run_id),),
            )
        ).fetchall()
    assert len(rows) == 1
    row = rows[0]
    assert row["group_id"] == "grp_broken"
    assert row["attempt_index"] == 3
    assert "## 第1集：标题" in row["raw_text"]
    assert row["normalized_text"].startswith("## 第1集\n")
    assert "开场导语" not in row["normalized_text"]
    assert len(row["raw_text_sha256"]) == 64
    assert len(row["normalized_text_sha256"]) == 64


async def test_outline_group_rejection_roundtrip_and_scoping(
    repository,
    persona,
    creation_request,
) -> None:
    accepted, lease = await create_and_lease_initial(repository, persona, creation_request)
    await repository.pause_content_rejection(
        lease.run_id,
        stage=InternalStage.GENERATING_EPISODE_OUTLINE,
        evidence="committed 第 3 集已锁定原片在静秋口袋里，第 4 集不得写从饼干盒首次取出。",
        repair_rounds=2,
        outline_group_id="g_third_person",
        now=NOW,
    )

    row = await repository.get_outline_group_rejection(lease.run_id, group_id="g_third_person")
    assert row is not None
    assert row["repair_rounds"] == 2
    assert "原片在静秋口袋里" in row["evidence"]
    # Rejections scope to one group: other groups see nothing.
    assert await repository.get_outline_group_rejection(lease.run_id, group_id="other") is None

    # A later rejection for the same group supersedes the earlier one on read.
    await repository.continue_run(
        creation_id=accepted.creation_id,
        run_kind="initial",
        idempotency_key="continue-rejection",
        now=NOW + timedelta(seconds=10),
    )
    resumed = await repository.lease_next_job(
        "worker-rejection", 30, now=NOW + timedelta(seconds=11)
    )
    assert resumed is not None
    await repository.pause_content_rejection(
        resumed.run_id,
        stage=InternalStage.GENERATING_EPISODE_OUTLINE,
        evidence="第二轮拒绝：仍写成了首次取出。",
        repair_rounds=2,
        outline_group_id="g_third_person",
        now=NOW + timedelta(seconds=20),
    )
    latest = await repository.get_outline_group_rejection(resumed.run_id, group_id="g_third_person")
    assert latest is not None
    assert "第二轮拒绝" in latest["evidence"]

    # A rejection without an outline group id (story/episode stages) never
    # writes the group rejection table.
    await repository.continue_run(
        creation_id=accepted.creation_id,
        run_kind="initial",
        idempotency_key="continue-rejection-2",
        now=NOW + timedelta(seconds=30),
    )
    resumed_again = await repository.lease_next_job(
        "worker-rejection-2", 30, now=NOW + timedelta(seconds=31)
    )
    assert resumed_again is not None
    await repository.pause_content_rejection(
        resumed_again.run_id,
        stage=InternalStage.GENERATING_STORY_OUTLINE,
        evidence="故事大纲连续性未收敛。",
        repair_rounds=4,
        now=NOW + timedelta(seconds=40),
    )
    async with repository._connection() as connection:
        count = await (
            await connection.execute(
                "SELECT COUNT(*) FROM outline_group_rejections WHERE run_id = ?",
                (str(resumed_again.run_id),),
            )
        ).fetchone()
    assert count[0] == 2  # only the two group-scoped rejections above


def make_failure() -> RunFailure:
    return RunFailure(
        code="attempts_exhausted",
        message="The stage attempt limit was exhausted.",
        failed_stage=InternalStage.GENERATING_STORY_OUTLINE,
        attempt_count=3,
    )


async def create_and_lease_initial(
    repository: Repository,
    persona: PersonaSnapshot,
    request: CreateCreationRequest,
):
    accepted = await repository.create_creation(
        idempotency_key="create-1",
        request=request,
        persona_snapshot=persona,
        now=NOW,
    )
    lease = await repository.lease_next_job(
        worker_id="worker-1",
        lease_seconds=30,
        now=NOW,
    )
    assert lease is not None
    return accepted, lease


def locked_outline_payload():
    contract = make_contract()
    contract_hash = story_contract_sha256(contract)
    return contract, {
        "stage": "generating_episode_outline",
        "content": "单集锁定分集大纲",
        "episode_count": 1,
        "episodes": [{"episode_number": 1, "plan": "林岚回到旧屋。"}],
        "story_contract": contract.model_dump(mode="json"),
        "story_contract_sha256": contract_hash,
        "story_contract_markdown": render_story_contract_markdown(
            contract,
            contract_hash,
        ),
        "contract_review": {
            "passed": True,
            "evidence": "独立合同审查通过",
            "issues": [],
        },
        "contract_repair_rounds": 0,
    }


def persist_succeeded_model_call(
    repository: Repository,
    *,
    context: ModelCallContext,
    role: str,
    call_id: str | None = None,
) -> ModelCallRecord:
    store = ModelCallStore(repository.database_path)
    record = build_started_record(
        call_id=call_id or new_call_id(),
        role=role,
        adapter="fake",
        provider="fake",
        model="gpt-5.5" if role == "review" else "claude-opus-5",
        context=context,
        estimated_input_tokens=10,
        estimated_output_tokens=20,
        verified_limit_tokens=200_000,
    )
    record.status = "succeeded"
    record.outcome = "success"
    store.upsert(record)
    store.close()
    return record


def persist_succeeded_outline_review(
    repository: Repository,
    run_id,
    *,
    call_id: str | None = None,
) -> str:
    return persist_succeeded_model_call(
        repository,
        call_id=call_id,
        role="review",
        context=ModelCallContext(
            run_id=str(run_id),
            stage=InternalStage.GENERATING_EPISODE_OUTLINE.value,
            operation_id="outline-operation",
        ),
    ).call_id


async def test_initialize_enables_wal_foreign_keys_and_domain_tables(repository) -> None:
    assert SCHEMA_VERSION == 33
    async with repository._connection() as connection:
        failure_columns = await (
            await connection.execute("PRAGMA table_info(outline_markdown_failures)")
        ).fetchall()
        assert {"raw_text", "normalized_text", "parse_error"} <= {
            row["name"] for row in failure_columns
        }
    async with repository._connection() as connection:
        journal = await (await connection.execute("PRAGMA journal_mode")).fetchone()
        foreign_keys = await (await connection.execute("PRAGMA foreign_keys")).fetchone()
        rows = await (
            await connection.execute(
                """
                SELECT name FROM sqlite_master
                WHERE type = 'table' AND name NOT LIKE 'sqlite_%'
                """
            )
        ).fetchall()

    assert journal[0] == "wal"
    assert foreign_keys[0] == 1
    timestamp = NOW.isoformat()
    with pytest.raises(sqlite3.IntegrityError):
        async with repository._connection() as connection:
            await connection.execute(
                """
                INSERT INTO jobs(id, run_id, state, available_at, created_at, updated_at)
                VALUES ('dangling-job', 'missing-run', 'queued', ?, ?, ?)
                """,
                (timestamp, timestamp, timestamp),
            )
    assert {
        "creations",
        "runs",
        "jobs",
        "stage_attempts",
        "business_checkpoints",
        "deliveries",
        "frozen_revisions",
        "idempotency_records",
        "run_progress",
        "episode_plans",
        "episode_drafts",
        "episode_attempts",
        "episode_attempt_cycles",
        "episode_attempt_current",
        "episode_timeouts",
        "quality_gate_rejections",
        "quality_gate_repairs",
        "series_bible_projection_repairs",
        "episode_generation_windows",
        "outline_season_maps",
        "outline_generation_groups",
        "model_calls",
    } <= {row[0] for row in rows}


async def test_schema_v24_to_v25_adds_episode_generation_windows(repository) -> None:
    async with repository._connection() as connection:
        await connection.execute("DROP TABLE episode_generation_windows")
        await connection.execute("ALTER TABLE episode_candidates DROP COLUMN generation_window_id")
        await connection.execute("DELETE FROM pengine_schema WHERE version = 25")
        await connection.execute("DELETE FROM pengine_schema WHERE version = 26")
        await connection.execute("DELETE FROM pengine_schema WHERE version = 27")
        await connection.execute("DELETE FROM pengine_schema WHERE version = 28")
        await connection.execute("DELETE FROM pengine_schema WHERE version = 29")
        await connection.execute("DELETE FROM pengine_schema WHERE version = 30")
        await connection.execute("DELETE FROM pengine_schema WHERE version = 31")
        await connection.execute("DELETE FROM pengine_schema WHERE version = 32")
        await connection.execute("DELETE FROM pengine_schema WHERE version = 33")
        await connection.commit()

    restarted = Repository(repository.database_path)
    await restarted.initialize()
    async with restarted._connection() as connection:
        tables = await (
            await connection.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
        ).fetchall()
        candidate_columns = await (
            await connection.execute("PRAGMA table_info(episode_candidates)")
        ).fetchall()
        version = await (
            await connection.execute("SELECT MAX(version) FROM pengine_schema")
        ).fetchone()

    assert "episode_generation_windows" in {row[0] for row in tables}
    assert "generation_window_id" in {column[1] for column in candidate_columns}
    assert version[0] == SCHEMA_VERSION


async def test_schema_v32_to_v33_clears_history_and_adds_account_ownership(
    repository,
    persona,
    creation_request,
) -> None:
    accepted = await repository.create_creation(
        idempotency_key="legacy-create",
        request=creation_request,
        persona_snapshot=persona,
        now=NOW,
    )
    async with repository._connection() as connection:
        run = await (await connection.execute("SELECT id FROM runs LIMIT 1")).fetchone()
        await connection.execute(
            """
            INSERT INTO model_calls(
                call_id, run_id, creation_id, role, adapter, provider, model,
                requested_at, estimated_input_tokens, estimated_output_tokens,
                estimated_total_tokens, preflight, status, usage_status, outcome
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0, 0, 0, ?, ?, ?, ?)
            """,
            (
                "legacy-call",
                run["id"],
                str(accepted.creation_id),
                "generation",
                "test",
                "test",
                "test",
                NOW.isoformat(),
                "ok",
                "succeeded",
                "unavailable",
                "succeeded",
            ),
        )
        await connection.execute(
            """
            INSERT INTO series_bible_projection_repairs(
                run_id, original_candidate_id, status, target_character_ids_json,
                validation_json, created_at, attempt_count
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (run["id"], "legacy-candidate", "failed", "[]", "{}", NOW.isoformat(), 1),
        )
        await connection.execute("DROP INDEX creations_owner_updated")
        await connection.execute("ALTER TABLE creations DROP COLUMN owner_id")
        await connection.execute("DROP TABLE sessions")
        await connection.execute("DROP TABLE users")
        await connection.execute("DELETE FROM pengine_schema WHERE version = 33")
        await connection.commit()

    restarted = Repository(repository.database_path)
    await restarted.initialize()
    async with restarted._connection() as connection:
        counts = {
            table: (await (await connection.execute(f"SELECT COUNT(*) FROM {table}")).fetchone())[0]
            for table in (
                "creations",
                "runs",
                "jobs",
                "idempotency_records",
                "model_calls",
                "series_bible_projection_repairs",
            )
        }
        tables = {
            row[0]
            for row in await (
                await connection.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
            ).fetchall()
        }
        creation_columns = {
            row[1]
            for row in await (await connection.execute("PRAGMA table_info(creations)")).fetchall()
        }
        version = (
            await (await connection.execute("SELECT MAX(version) FROM pengine_schema")).fetchone()
        )[0]

    assert counts == {
        "creations": 0,
        "runs": 0,
        "jobs": 0,
        "idempotency_records": 0,
        "model_calls": 0,
        "series_bible_projection_repairs": 0,
    }
    assert {"users", "sessions"} <= tables
    assert "owner_id" in creation_columns
    assert version == 33


async def test_schema_v32_to_v33_rolls_back_when_destructive_cutover_fails(
    repository,
    persona,
    creation_request,
) -> None:
    accepted = await repository.create_creation(
        idempotency_key="legacy-create-rollback",
        request=creation_request,
        persona_snapshot=persona,
        now=NOW,
    )
    async with repository._connection() as connection:
        await connection.execute("DROP INDEX creations_owner_updated")
        await connection.execute("ALTER TABLE creations DROP COLUMN owner_id")
        await connection.execute("DROP TABLE sessions")
        await connection.execute("DROP TABLE users")
        await connection.execute("DELETE FROM pengine_schema WHERE version = 33")
        await connection.execute(
            """
            CREATE TRIGGER reject_v33_cutover
            BEFORE DELETE ON creations
            BEGIN
                SELECT RAISE(ABORT, 'simulated cutover failure');
            END
            """
        )
        await connection.commit()

    restarted = Repository(repository.database_path)
    with pytest.raises(aiosqlite.IntegrityError, match="simulated cutover failure"):
        await restarted.initialize()

    async with repository._connection() as connection:
        creation = await (
            await connection.execute(
                "SELECT id FROM creations WHERE id = ?", (str(accepted.creation_id),)
            )
        ).fetchone()
        account_tables = await (
            await connection.execute(
                "SELECT name FROM sqlite_master "
                "WHERE type = 'table' AND name IN ('users', 'sessions')"
            )
        ).fetchall()
        version = await (
            await connection.execute("SELECT 1 FROM pengine_schema WHERE version = 33")
        ).fetchone()

    assert creation is not None
    assert account_tables == []
    assert version is None


async def test_schema_v25_to_v26_adds_script_context_audit_columns(repository) -> None:
    async with repository._connection() as connection:
        await connection.execute("ALTER TABLE model_calls DROP COLUMN context_bundle_sha256")
        await connection.execute("ALTER TABLE model_calls DROP COLUMN context_manifest_json")
        await connection.execute("DELETE FROM pengine_schema WHERE version = 26")
        await connection.execute("DELETE FROM pengine_schema WHERE version = 27")
        await connection.execute("DELETE FROM pengine_schema WHERE version = 28")
        await connection.execute("DELETE FROM pengine_schema WHERE version = 29")
        await connection.execute("DELETE FROM pengine_schema WHERE version = 30")
        await connection.execute("DELETE FROM pengine_schema WHERE version = 31")
        await connection.execute("DELETE FROM pengine_schema WHERE version = 32")
        await connection.execute("DELETE FROM pengine_schema WHERE version = 33")
        await connection.commit()

    restarted = Repository(repository.database_path)
    await restarted.initialize()
    async with restarted._connection() as connection:
        columns = await (await connection.execute("PRAGMA table_info(model_calls)")).fetchall()
        version = await (
            await connection.execute("SELECT MAX(version) FROM pengine_schema")
        ).fetchone()

    column_names = {row[1] for row in columns}
    assert {"context_bundle_sha256", "context_manifest_json"} <= column_names
    assert version[0] == SCHEMA_VERSION


async def test_schema_v26_to_v27_adds_grouped_outline_tables(repository) -> None:
    async with repository._connection() as connection:
        await connection.execute("DROP TABLE outline_generation_groups")
        await connection.execute("DROP TABLE outline_season_maps")
        await connection.execute("DELETE FROM pengine_schema WHERE version = 27")
        await connection.execute("DELETE FROM pengine_schema WHERE version = 28")
        await connection.execute("DELETE FROM pengine_schema WHERE version = 29")
        await connection.execute("DELETE FROM pengine_schema WHERE version = 30")
        await connection.execute("DELETE FROM pengine_schema WHERE version = 31")
        await connection.execute("DELETE FROM pengine_schema WHERE version = 32")
        await connection.execute("DELETE FROM pengine_schema WHERE version = 33")
        await connection.commit()

    restarted = Repository(repository.database_path)
    await restarted.initialize()
    async with restarted._connection() as connection:
        tables = await (
            await connection.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
        ).fetchall()
        version = await (
            await connection.execute("SELECT MAX(version) FROM pengine_schema")
        ).fetchone()

    assert {"outline_season_maps", "outline_generation_groups"} <= {row[0] for row in tables}
    assert version[0] == SCHEMA_VERSION


async def test_schema_v27_to_v28_adds_plaintext_sidecar_columns(repository) -> None:
    columns_to_remove = (
        "screenplay_text",
        "screenplay_nonce",
        "screenplay_manifest_json",
        "content_call_id",
        "sidecar_call_id",
        "context_bundle_sha256",
    )
    async with repository._connection() as connection:
        for column in columns_to_remove:
            await connection.execute(f"ALTER TABLE episode_generation_windows DROP COLUMN {column}")
        await connection.execute("DELETE FROM pengine_schema WHERE version = 28")
        await connection.execute("DELETE FROM pengine_schema WHERE version = 29")
        await connection.execute("DELETE FROM pengine_schema WHERE version = 30")
        await connection.execute("DELETE FROM pengine_schema WHERE version = 31")
        await connection.execute("DELETE FROM pengine_schema WHERE version = 32")
        await connection.execute("DELETE FROM pengine_schema WHERE version = 33")
        await connection.commit()

    restarted = Repository(repository.database_path)
    await restarted.initialize()
    async with restarted._connection() as connection:
        columns = await (
            await connection.execute("PRAGMA table_info(episode_generation_windows)")
        ).fetchall()
        version = await (
            await connection.execute("SELECT MAX(version) FROM pengine_schema")
        ).fetchone()

    assert set(columns_to_remove) <= {row[1] for row in columns}
    assert version[0] == SCHEMA_VERSION


async def test_schema_v28_to_v29_preserves_legacy_committed_outline_groups(
    repository,
    persona,
    creation_request,
) -> None:
    _, lease = await create_and_lease_initial(repository, persona, creation_request)
    content_json = '{"content":"legacy","group_id":"opening"}'
    content_sha256 = hashlib.sha256(content_json.encode()).hexdigest()
    async with repository._connection() as connection:
        await connection.execute("DROP TABLE outline_generation_groups")
        await connection.executescript(
            """
            CREATE TABLE outline_generation_groups (
                run_id TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
                group_id TEXT NOT NULL,
                position INTEGER NOT NULL,
                start_episode INTEGER NOT NULL,
                end_episode INTEGER NOT NULL,
                operation_id TEXT NOT NULL,
                call_id TEXT,
                status TEXT NOT NULL CHECK (status IN ('generating', 'committed', 'failed')),
                attempt_count INTEGER NOT NULL,
                content_json TEXT,
                content_sha256 TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY (run_id, group_id),
                UNIQUE (run_id, position)
            );
            CREATE INDEX outline_generation_groups_run
            ON outline_generation_groups(run_id, position);
            """
        )
        await connection.execute(
            """
            INSERT INTO outline_generation_groups(
                run_id, group_id, position, start_episode, end_episode, operation_id,
                call_id, status, attempt_count, content_json, content_sha256,
                created_at, updated_at
            ) VALUES (?, 'opening', 1, 1, 1, 'legacy-op', 'legacy-call',
                      'committed', 1, ?, ?, ?, ?)
            """,
            (str(lease.run_id), content_json, content_sha256, NOW.isoformat(), NOW.isoformat()),
        )
        await connection.execute("DELETE FROM pengine_schema WHERE version = 29")
        await connection.execute("DELETE FROM pengine_schema WHERE version = 30")
        await connection.execute("DELETE FROM pengine_schema WHERE version = 31")
        await connection.execute("DELETE FROM pengine_schema WHERE version = 32")
        await connection.execute("DELETE FROM pengine_schema WHERE version = 33")
        await connection.commit()

    restarted = Repository(repository.database_path)
    await restarted.initialize()
    groups = await restarted.get_committed_outline_groups(lease.run_id)
    async with restarted._connection() as connection:
        columns = await (
            await connection.execute("PRAGMA table_info(outline_generation_groups)")
        ).fetchall()

    assert {
        "outline_markdown",
        "outline_markdown_sha256",
        "body_call_id",
        "sidecar_call_id",
    } <= {row[1] for row in columns}
    assert groups[0]["content"] == {"content": "legacy", "group_id": "opening"}
    assert groups[0]["outline_markdown"] is None
    assert groups[0]["body_call_id"] is None


async def test_plaintext_generation_window_survives_sidecar_failure_and_reuses_attempt(
    repository,
    persona,
    creation_request,
) -> None:
    _, lease = await create_and_lease_initial(repository, persona, creation_request)
    manifest = [{"episode_number": 1, "screenplay_sha256": "a" * 64}]
    window_id = await repository.begin_episode_generation_window(
        lease.run_id,
        design_candidate_id="series_bible_candidate_test",
        design_content_hash="b" * 64,
        design_epoch=1,
        group_id="opening_unit",
        start_episode=1,
        end_episode=1,
        operation_id="content-operation",
    )
    await repository.save_episode_generation_text(
        lease.run_id,
        window_id,
        screenplay_text="<<<START>>>\n剧本正文\n<<<END>>>",
        nonce="nonce",
        manifest=manifest,
        content_call_id="content-call",
        context_bundle_sha256="c" * 64,
    )
    await repository.fail_episode_generation_window(
        lease.run_id,
        window_id,
        preserve_text=True,
    )

    resumed_window_id = await repository.begin_episode_generation_window(
        lease.run_id,
        design_candidate_id="series_bible_candidate_test",
        design_content_hash="b" * 64,
        design_epoch=1,
        group_id="opening_unit",
        start_episode=1,
        end_episode=1,
        operation_id="new-operation-must-not-replace-content",
    )
    stored = await repository.get_episode_generation_text(lease.run_id, resumed_window_id)

    assert resumed_window_id == window_id
    assert stored == {
        "raw_text": "<<<START>>>\n剧本正文\n<<<END>>>",
        "nonce": "nonce",
        "manifest": manifest,
        "content_call_id": "content-call",
        "context_bundle_sha256": "c" * 64,
        "operation_id": "content-operation",
    }

    await repository.bind_episode_generation_window_call(
        lease.run_id,
        window_id,
        call_id="content-call",
        sidecar_call_id="sidecar-call",
    )
    window = (await repository.get_episode_generation_windows(lease.run_id))[0]
    assert window["status"] == "generated"
    assert window["call_id"] == "content-call"
    assert window["content_call_id"] == "content-call"
    assert window["sidecar_call_id"] == "sidecar-call"


async def test_grouped_outline_checkpoints_are_immutable_and_resume_failed_group(
    repository,
    persona,
    creation_request,
) -> None:
    _, lease = await create_and_lease_initial(repository, persona, creation_request)
    season_map = {"episode_count": 3, "script_generation_groups": ["opening", "pursuit"]}
    await repository.commit_outline_season_map(
        lease.run_id,
        season_map,
        call_id="season-call",
    )
    await repository.commit_outline_season_map(
        lease.run_id,
        season_map,
        call_id="season-call",
    )

    assert (await repository.get_outline_season_map(lease.run_id))["content"] == season_map
    with pytest.raises(DomainError, match="cannot be replaced"):
        await repository.commit_outline_season_map(
            lease.run_id,
            {"episode_count": 4},
            call_id="replacement-call",
        )

    assert (
        await repository.begin_outline_group(
            lease.run_id,
            group_id="opening",
            position=1,
            start_episode=1,
            end_episode=1,
            operation_id="opening-op",
        )
        == 1
    )
    opening_markdown = "## 第1集\n\n第一组"
    await repository.save_outline_group_body(
        lease.run_id,
        group_id="opening",
        operation_id="opening-op",
        outline_markdown=opening_markdown,
        outline_markdown_sha256=hashlib.sha256(opening_markdown.encode()).hexdigest(),
        body_call_id="opening-body-call",
    )
    await repository.complete_outline_group(
        lease.run_id,
        group_id="opening",
        operation_id="opening-op",
        payload={"group_id": "opening", "content": "第一组"},
        sidecar_call_id="opening-sidecar-call",
    )
    assert (
        await repository.begin_outline_group(
            lease.run_id,
            group_id="pursuit",
            position=2,
            start_episode=2,
            end_episode=3,
            operation_id="pursuit-op-1",
        )
        == 1
    )
    await repository.fail_outline_group(
        lease.run_id,
        group_id="pursuit",
        operation_id="pursuit-op-1",
    )
    assert (
        await repository.begin_outline_group(
            lease.run_id,
            group_id="pursuit",
            position=2,
            start_episode=2,
            end_episode=3,
            operation_id="pursuit-op-2",
        )
        == 2
    )
    pursuit_markdown = "## 第2集\n\n追踪\n\n## 第3集\n\n结果"
    await repository.save_outline_group_body(
        lease.run_id,
        group_id="pursuit",
        operation_id="pursuit-op-2",
        outline_markdown=pursuit_markdown,
        outline_markdown_sha256=hashlib.sha256(pursuit_markdown.encode()).hexdigest(),
        body_call_id="pursuit-body-call",
    )
    pursuit_hash = hashlib.sha256(pursuit_markdown.encode()).hexdigest()
    repaired_markdown = "## 第2集\n\n追踪修正\n\n## 第3集\n\n结果"
    repaired_hash = hashlib.sha256(repaired_markdown.encode()).hexdigest()
    with pytest.raises(DomainError, match="no longer matches"):
        await repository.replace_outline_group_body(
            lease.run_id,
            group_id="pursuit",
            operation_id="pursuit-op-2",
            expected_outline_markdown_sha256="0" * 64,
            outline_markdown=repaired_markdown,
            outline_markdown_sha256=repaired_hash,
            body_call_id="pursuit-repair-body-call",
        )
    await repository.replace_outline_group_body(
        lease.run_id,
        group_id="pursuit",
        operation_id="pursuit-op-2",
        expected_outline_markdown_sha256=pursuit_hash,
        outline_markdown=repaired_markdown,
        outline_markdown_sha256=repaired_hash,
        body_call_id="pursuit-repair-body-call",
    )
    replaced_body = await repository.get_outline_group_body(
        lease.run_id,
        group_id="pursuit",
    )
    assert replaced_body is not None
    assert replaced_body["outline_markdown"] == repaired_markdown
    pursuit_markdown = repaired_markdown
    await repository.fail_outline_group(
        lease.run_id,
        group_id="pursuit",
        operation_id="pursuit-op-2",
    )
    assert (
        await repository.begin_outline_group(
            lease.run_id,
            group_id="pursuit",
            position=2,
            start_episode=2,
            end_episode=3,
            operation_id="pursuit-op-3",
        )
        == 2
    )
    stored_body = await repository.get_outline_group_body(
        lease.run_id,
        group_id="pursuit",
    )
    assert stored_body is not None
    assert stored_body["outline_markdown"] == pursuit_markdown
    await repository.complete_outline_group(
        lease.run_id,
        group_id="pursuit",
        operation_id="pursuit-op-3",
        payload={"group_id": "pursuit", "content": "第二组"},
        sidecar_call_id="pursuit-sidecar-call",
    )

    groups = await repository.get_committed_outline_groups(lease.run_id)
    assert [item["group_id"] for item in groups] == ["opening", "pursuit"]
    assert [item["attempt_count"] for item in groups] == [1, 2]
    assert [item["call_id"] for item in groups] == [
        "opening-body-call",
        "pursuit-repair-body-call",
    ]
    assert [item["sidecar_call_id"] for item in groups] == [
        "opening-sidecar-call",
        "pursuit-sidecar-call",
    ]
    with pytest.raises(DomainError, match="already committed"):
        await repository.begin_outline_group(
            lease.run_id,
            group_id="opening",
            position=1,
            start_episode=1,
            end_episode=1,
            operation_id="opening-op-2",
        )


async def test_grouped_outline_recovery_does_not_exhaust_whole_stage_attempts(
    repository,
    persona,
    creation_request,
) -> None:
    accepted, lease = await create_and_lease_initial(repository, persona, creation_request)
    stage = InternalStage.GENERATING_EPISODE_OUTLINE
    assert await repository.record_stage_attempt(lease.run_id, stage, now=NOW) == 1
    assert (
        await repository.record_stage_attempt(
            lease.run_id,
            stage,
            now=NOW + timedelta(seconds=1),
        )
        == 2
    )
    assert (
        await repository.record_stage_attempt(
            lease.run_id,
            stage,
            now=NOW + timedelta(seconds=2),
        )
        == 3
    )
    await repository.commit_outline_season_map(
        lease.run_id,
        {"episode_count": 30, "script_generation_groups": ["opening"]},
        call_id="season-call",
    )

    assert (
        await repository.record_stage_attempt(
            lease.run_id,
            stage,
            now=NOW + timedelta(seconds=3),
        )
        == 3
    )
    assert await repository.get_stage_attempt_counts(lease.run_id) == {stage: 3}
    assert (
        await repository.handle_run_timeout(
            lease.run_id,
            stage,
            now=NOW + timedelta(seconds=4),
        )
        == "auto_resuming"
    )

    resumed = await repository.lease_next_job(
        "worker-after-outline-timeout",
        30,
        now=NOW + timedelta(seconds=5),
    )
    assert resumed is not None
    assert resumed.run_id == lease.run_id
    assert (
        await repository.record_stage_attempt(
            lease.run_id,
            stage,
            now=NOW + timedelta(seconds=6),
        )
        == 3
    )
    assert (
        await repository.handle_run_timeout(
            lease.run_id,
            stage,
            now=NOW + timedelta(seconds=7),
        )
        == "paused"
    )

    continued = await repository.continue_run(
        creation_id=accepted.creation_id,
        run_kind="initial",
        idempotency_key="continue-grouped-outline",
        now=NOW + timedelta(seconds=8),
    )
    assert continued.creation_id == accepted.creation_id
    assert continued.run_state == "queued"


async def test_schema_v18_migrates_episode_attempts_to_cycle_zero(
    repository,
    persona,
    creation_request,
) -> None:
    _, lease = await create_and_lease_initial(repository, persona, creation_request)
    _, outline = locked_outline_payload()
    review_call_id = persist_succeeded_outline_review(repository, lease.run_id)
    await repository.approve_business_checkpoint(
        lease.run_id,
        InternalStage.GENERATING_EPISODE_OUTLINE,
        outline,
        review_call_id=review_call_id,
        now=NOW,
    )
    await repository.record_episode_attempt(lease.run_id, 1, now=NOW)

    async with repository._transaction() as connection:
        await connection.execute(
            """
            CREATE TABLE episode_attempts_legacy (
                run_id TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
                episode_number INTEGER NOT NULL CHECK (episode_number >= 1),
                attempt_number INTEGER NOT NULL CHECK (
                    attempt_number >= 1 AND attempt_number <= 3
                ),
                recorded_at TEXT NOT NULL,
                PRIMARY KEY (run_id, episode_number, attempt_number)
            )
            """
        )
        await connection.execute(
            """
            INSERT INTO episode_attempts_legacy(
                run_id, episode_number, attempt_number, recorded_at
            )
            SELECT run_id, episode_number, attempt_number, recorded_at
            FROM episode_attempts
            """
        )
        await connection.execute("DROP TABLE episode_attempts")
        await connection.execute("ALTER TABLE episode_attempts_legacy RENAME TO episode_attempts")
        await connection.execute("DROP TABLE episode_attempt_current")
        await connection.execute("DROP TABLE episode_attempt_cycles")
        await connection.execute("DELETE FROM pengine_schema WHERE version = 18")
        await connection.execute("DELETE FROM pengine_schema WHERE version = 19")
        await connection.execute("DELETE FROM pengine_schema WHERE version = 20")
        await connection.execute("DELETE FROM pengine_schema WHERE version = 21")
        await connection.execute("DELETE FROM pengine_schema WHERE version = 22")
        await connection.execute(
            "DELETE FROM pengine_schema WHERE version IN "
            "(23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33)"
        )

    await repository.initialize()
    assert await repository.get_episode_attempt_counts(lease.run_id) == {1: 1}
    assert await repository.get_episode_attempt_cycles(lease.run_id) == {1: 0}
    async with repository._connection() as connection:
        columns = await (await connection.execute("PRAGMA table_info(episode_attempts)")).fetchall()
    assert "attempt_cycle" in {row["name"] for row in columns}


async def test_schema_v19_to_v20_adds_identity_evidence_and_pause_reason(
    repository,
    persona,
    creation_request,
) -> None:
    accepted, lease = await create_and_lease_initial(repository, persona, creation_request)
    async with repository._transaction() as connection:
        await connection.execute(
            """
            CREATE TABLE run_progress_v19 (
                run_id TEXT PRIMARY KEY REFERENCES runs(id) ON DELETE CASCADE,
                current_stage TEXT NOT NULL,
                execution_state TEXT NOT NULL,
                elapsed_seconds REAL NOT NULL DEFAULT 0,
                active_started_at TEXT,
                timeout_stage TEXT,
                timeout_count INTEGER NOT NULL DEFAULT 0,
                updated_at TEXT NOT NULL,
                current_episode INTEGER,
                recovery_reason TEXT NOT NULL DEFAULT 'none' CHECK (
                    recovery_reason IN (
                        'none', 'run_timeout', 'relay_interruption', 'content_rejected',
                        'episode_error', 'context_budget', 'repair_authorization'
                    )
                ),
                content_repair_count INTEGER,
                pause_message TEXT
            )
            """
        )
        await connection.execute(
            """
            INSERT INTO run_progress_v19 SELECT * FROM run_progress
            """
        )
        await connection.execute("DROP TABLE run_progress")
        await connection.execute("ALTER TABLE run_progress_v19 RENAME TO run_progress")
        await connection.execute("ALTER TABLE model_calls DROP COLUMN response_model_ids_json")
        await connection.execute("DELETE FROM pengine_schema WHERE version = 20")
        await connection.execute("DELETE FROM pengine_schema WHERE version = 21")
        await connection.execute("DELETE FROM pengine_schema WHERE version = 22")
        await connection.execute(
            "DELETE FROM pengine_schema WHERE version IN "
            "(23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33)"
        )

    await repository.initialize()
    async with repository._connection() as connection:
        version = await (
            await connection.execute("SELECT MAX(version) FROM pengine_schema")
        ).fetchone()
        columns = await (await connection.execute("PRAGMA table_info(model_calls)")).fetchall()
    assert version[0] == SCHEMA_VERSION
    assert "response_model_ids_json" in {column["name"] for column in columns}

    await repository.mark_run_running(lease.run_id, now=NOW)
    await repository.pause_relay_identity_mismatch(
        lease.run_id,
        stage=InternalStage.GENERATING_STORY_OUTLINE,
        safe_message="Identity mismatch; response discarded.",
        now=NOW + timedelta(seconds=1),
    )
    resource = await repository.get_creation(
        accepted.creation_id,
        now=NOW + timedelta(seconds=2),
    )
    assert resource is not None
    assert resource.initial.state == "paused"
    assert resource.initial.pause.code == "relay_identity_mismatch"


async def test_creation_is_idempotent_and_payload_conflicts(
    repository,
    persona,
    creation_request,
) -> None:
    first = await repository.create_creation(
        idempotency_key="same-key",
        request=creation_request,
        persona_snapshot=persona,
        payload_hash="payload-a",
        now=NOW,
    )
    replay = await repository.create_creation(
        idempotency_key="same-key",
        request=creation_request,
        persona_snapshot=persona,
        payload_hash="payload-a",
        now=NOW + timedelta(seconds=1),
    )

    assert replay == first

    with pytest.raises(DomainError, match="Idempotency") as error:
        await repository.create_creation(
            idempotency_key="same-key",
            request=creation_request,
            persona_snapshot=persona,
            payload_hash="payload-b",
            now=NOW + timedelta(seconds=2),
        )
    assert error.value.code == "idempotency_conflict"

    async with aiosqlite.connect(repository.database_path) as connection:
        count = await (await connection.execute("SELECT COUNT(*) FROM creations")).fetchone()
    assert count == (1,)


async def test_new_creation_freezes_chinese_output_language_for_revisions(
    repository,
    persona,
    creation_request,
) -> None:
    accepted, initial_lease = await create_and_lease_initial(
        repository,
        persona,
        creation_request,
    )

    initial_work = await repository.get_run_work_item(initial_lease.run_id)
    assert initial_work.output_language == "zh-CN"

    await repository.succeed_run(initial_lease.run_id, make_delivery(), now=NOW)
    await repository.create_or_retry_revision(
        creation_id=accepted.creation_id,
        idempotency_key="language-revision",
        request=RevisionRequest(feedback="保持简体中文。"),
        now=NOW + timedelta(seconds=1),
    )
    revision_lease = await repository.lease_next_job(
        worker_id="language-revision-worker",
        lease_seconds=30,
        now=NOW + timedelta(seconds=2),
    )
    assert revision_lease is not None

    revision_work = await repository.get_run_work_item(revision_lease.run_id)
    assert revision_work.output_language == "zh-CN"


async def test_new_english_creation_keeps_output_language_unset(repository, persona) -> None:
    request = CreateCreationRequest(
        persona_id=persona.persona_id,
        story="A daughter returns to her seaside hometown.",
        requirements="Write a contemporary ten-episode short drama.",
    )
    _, lease = await create_and_lease_initial(repository, persona, request)

    work = await repository.get_run_work_item(lease.run_id)

    assert work.output_language is None


async def test_attempt_is_recorded_before_invocation_and_fourth_is_rejected(
    repository,
    persona,
    creation_request,
) -> None:
    _, lease = await create_and_lease_initial(repository, persona, creation_request)
    stage = InternalStage.GENERATING_STORY_OUTLINE

    assert await repository.record_stage_attempt(lease.run_id, stage) == 1
    assert await repository.record_stage_attempt(lease.run_id, stage) == 2
    assert await repository.record_stage_attempt(lease.run_id, stage) == 3

    with pytest.raises(DomainError) as error:
        await repository.record_stage_attempt(lease.run_id, stage)
    assert error.value.code == "attempts_exhausted"
    assert await repository.get_stage_attempt_counts(lease.run_id) == {stage: 3}


async def test_approved_checkpoint_is_immutable_and_blocks_regeneration(
    repository,
    persona,
    creation_request,
) -> None:
    _, lease = await create_and_lease_initial(repository, persona, creation_request)
    stage = InternalStage.SELECTING_L0_VARIANT
    await repository.record_stage_attempt(lease.run_id, stage)

    payload = {"selected_l0_variant": "归返", "selection_rationale": "匹配母题"}
    assert await repository.approve_business_checkpoint(lease.run_id, stage, payload) == payload
    assert await repository.approve_business_checkpoint(lease.run_id, stage, payload) == payload

    with pytest.raises(DomainError) as conflict:
        await repository.approve_business_checkpoint(
            lease.run_id,
            stage,
            {"selected_l0_variant": "异化"},
        )
    assert conflict.value.code == "checkpoint_conflict"

    with pytest.raises(DomainError) as approved:
        await repository.record_stage_attempt(lease.run_id, stage)
    assert approved.value.code == "stage_already_approved"


async def test_checkpoint_model_call_provenance_is_hidden_and_immutable(
    repository,
    persona,
    creation_request,
) -> None:
    _, lease = await create_and_lease_initial(repository, persona, creation_request)
    stage = InternalStage.GENERATING_EPISODE_OUTLINE
    _, payload = locked_outline_payload()

    with pytest.raises(DomainError, match="physical review provenance"):
        await repository.approve_business_checkpoint(lease.run_id, stage, payload)
    with pytest.raises(DomainError, match="not a successful physical review call"):
        await repository.approve_business_checkpoint(
            lease.run_id,
            stage,
            payload,
            review_call_id="forged-review",
        )

    review_call_id = persist_succeeded_outline_review(
        repository,
        lease.run_id,
        call_id="physical-review-1",
    )
    await repository.approve_business_checkpoint(
        lease.run_id,
        stage,
        payload,
        review_call_id=review_call_id,
    )

    assert (await repository.get_business_checkpoints(lease.run_id))[stage] == payload
    assert await repository.get_checkpoint_review_call_id(lease.run_id, stage) == (
        "physical-review-1"
    )
    with pytest.raises(DomainError, match="model-call provenance"):
        await repository.approve_business_checkpoint(
            lease.run_id,
            stage,
            payload,
            review_call_id="physical-review-2",
        )


async def test_story_checkpoint_persists_sixth_semantic_repair_round(
    repository,
    persona,
    creation_request,
) -> None:
    _, lease = await create_and_lease_initial(repository, persona, creation_request)
    outline_stage = InternalStage.GENERATING_STORY_OUTLINE
    character_relationships_stage = InternalStage.GENERATING_CHARACTER_RELATIONSHIPS
    await repository.record_stage_attempt(lease.run_id, outline_stage)
    await repository.record_stage_attempt(lease.run_id, character_relationships_stage)
    outline_payload = {
        "stage": outline_stage.value,
        "content": "经四轮语义修订后通过的故事大纲。",
        "consistency_review": {
            "passed": True,
            "evidence": "第四轮修订后大纲与时间线一致。",
            "issues": [],
        },
        "consistency_repair_rounds": 4,
    }
    character_relationships_payload = {
        "stage": character_relationships_stage.value,
        "character_biographies": "人物小传",
        "relationship_logic": "关系逻辑",
        "consistency_review": {
            "passed": True,
            "evidence": "第四轮修订后角色与时间线一致。",
            "issues": [],
        },
        "consistency_repair_rounds": 4,
    }

    await repository.approve_business_checkpoint(lease.run_id, outline_stage, outline_payload)
    await repository.approve_business_checkpoint(
        lease.run_id,
        character_relationships_stage,
        character_relationships_payload,
    )

    checkpoints = await repository.get_business_checkpoints(lease.run_id)
    assert checkpoints[outline_stage] == outline_payload
    assert checkpoints[character_relationships_stage] == character_relationships_payload


async def test_failed_revision_retry_uses_frozen_feedback_and_new_thread(
    repository,
    persona,
    creation_request,
) -> None:
    accepted, initial_lease = await create_and_lease_initial(
        repository,
        persona,
        creation_request,
    )
    await repository.succeed_run(initial_lease.run_id, make_delivery(), now=NOW)

    revision_request = RevisionRequest(feedback="  加强结尾行动。\n")
    assert revision_request.feedback == "  加强结尾行动。\n"
    await repository.create_or_retry_revision(
        creation_id=accepted.creation_id,
        idempotency_key="revision-1",
        request=revision_request,
        now=NOW + timedelta(minutes=1),
    )
    first_revision = await repository.lease_next_job(
        worker_id="worker-1",
        lease_seconds=30,
        now=NOW + timedelta(minutes=1),
    )
    assert first_revision is not None
    assert first_revision.thread_id != initial_lease.thread_id
    await repository.fail_run(first_revision.run_id, make_failure())

    with pytest.raises(DomainError) as locked:
        await repository.create_or_retry_revision(
            creation_id=accepted.creation_id,
            idempotency_key="revision-changed",
            request=RevisionRequest(feedback="加强结尾行动。"),
        )
    assert locked.value.code == "revision_feedback_locked"

    await repository.create_or_retry_revision(
        creation_id=accepted.creation_id,
        idempotency_key="revision-retry",
        request=revision_request,
    )
    retry = await repository.lease_next_job(
        worker_id="worker-1",
        lease_seconds=30,
    )
    assert retry is not None
    assert retry.run_sequence == 2
    assert retry.thread_id not in {initial_lease.thread_id, first_revision.thread_id}


async def test_revision_rules_and_initial_delivery_remains_visible(
    repository,
    persona,
    creation_request,
) -> None:
    accepted, initial_lease = await create_and_lease_initial(
        repository,
        persona,
        creation_request,
    )
    before_success = await repository.get_creation(accepted.creation_id)
    assert before_success.initial.state == "running"
    assert before_success.revision.state == "unavailable"

    initial_delivery = make_delivery()
    await repository.succeed_run(initial_lease.run_id, initial_delivery)
    after_success = await repository.get_creation(accepted.creation_id)
    assert after_success.initial.result == initial_delivery
    assert after_success.revision.state == "available"

    request = RevisionRequest(feedback="加强结尾行动。")
    await repository.create_or_retry_revision(
        creation_id=accepted.creation_id,
        idempotency_key="revision-1",
        request=request,
    )
    queued = await repository.get_creation(accepted.creation_id)
    assert queued.initial.result == initial_delivery
    assert queued.revision.state == "queued"

    with pytest.raises(DomainError) as duplicate:
        await repository.create_or_retry_revision(
            creation_id=accepted.creation_id,
            idempotency_key="revision-duplicate",
            request=request,
        )
    assert duplicate.value.code == "revision_not_allowed"

    revision_lease = await repository.lease_next_job(
        worker_id="worker-1",
        lease_seconds=30,
    )
    assert revision_lease is not None
    await repository.fail_run(revision_lease.run_id, make_failure())
    failed = await repository.get_creation(accepted.creation_id)
    assert failed.initial.result == initial_delivery
    assert failed.revision.state == "failed"


async def test_revision_becomes_available_from_authoritative_initial_status(
    repository,
    persona,
    creation_request,
    monkeypatch,
) -> None:
    accepted, initial_lease = await create_and_lease_initial(
        repository,
        persona,
        creation_request,
    )
    async with repository._connection() as connection:
        stale_initial = await repository._fetch_initial_run(
            connection,
            accepted.creation_id,
        )
    assert stale_initial is not None
    assert stale_initial["state"] == "running"

    await repository.succeed_run(initial_lease.run_id, make_delivery())

    async def fetch_stale_initial(connection, creation_id):
        return stale_initial

    monkeypatch.setattr(repository, "_fetch_initial_run", fetch_stale_initial)
    aggregate = await repository.get_creation(accepted.creation_id)

    assert aggregate.initial.state == "succeeded"
    assert aggregate.revision.state == "available"


async def test_successful_revision_consumes_entitlement(
    repository,
    persona,
    creation_request,
) -> None:
    accepted, initial_lease = await create_and_lease_initial(
        repository,
        persona,
        creation_request,
    )
    await repository.succeed_run(initial_lease.run_id, make_delivery())
    request = RevisionRequest(feedback="加强结尾行动。")
    await repository.queue_revision(
        accepted.creation_id,
        "revision-success",
        request,
    )
    revision_lease = await repository.lease_next_job("worker-1", 30)
    assert revision_lease is not None
    revised_delivery = make_delivery(revised=True)
    await repository.succeed_run(revision_lease.run_id, revised_delivery)

    aggregate = await repository.get_creation(accepted.creation_id)
    assert aggregate is not None
    assert aggregate.initial.result == make_delivery()
    assert aggregate.revision.state == "succeeded"
    assert aggregate.revision.result == revised_delivery

    with pytest.raises(DomainError) as consumed:
        await repository.queue_revision(
            accepted.creation_id,
            "revision-after-success",
            request,
        )
    assert consumed.value.code == "revision_not_allowed"


async def test_expired_lease_requeues_same_thread_with_durable_business_state(
    repository,
    persona,
    creation_request,
) -> None:
    _, lease = await create_and_lease_initial(repository, persona, creation_request)
    outline_stage = InternalStage.GENERATING_STORY_OUTLINE
    character_relationships_stage = InternalStage.GENERATING_CHARACTER_RELATIONSHIPS
    outline_payload = {"content": "已批准的故事梗概"}
    character_relationships_payload = {
        "character_biographies": "已批准的人物小传",
        "relationship_logic": "已批准的人物关系",
    }
    await repository.record_stage_attempt(lease.run_id, outline_stage, now=NOW)
    await repository.approve_business_checkpoint(
        lease.run_id,
        outline_stage,
        outline_payload,
        now=NOW,
    )
    await repository.record_stage_attempt(
        lease.run_id,
        character_relationships_stage,
        now=NOW,
    )
    await repository.approve_business_checkpoint(
        lease.run_id,
        character_relationships_stage,
        character_relationships_payload,
        now=NOW,
    )

    recoverable = await repository.reconcile_startup(
        now=NOW + timedelta(seconds=31),
    )
    assert len(recoverable) == 1
    assert recoverable[0].thread_id == lease.thread_id
    assert recoverable[0].stage_attempts == {
        outline_stage: 1,
        character_relationships_stage: 1,
    }
    assert recoverable[0].business_checkpoints == {
        outline_stage: outline_payload,
        character_relationships_stage: character_relationships_payload,
    }

    resumed = await repository.lease_next_job(
        worker_id="worker-after-restart",
        lease_seconds=30,
        now=NOW + timedelta(seconds=31),
    )
    assert resumed is not None
    assert resumed.run_id == lease.run_id
    assert resumed.thread_id == lease.thread_id


async def test_progress_timeout_pause_continue_and_end_are_durable_and_idempotent(
    repository,
    persona,
    creation_request,
) -> None:
    accepted, lease = await create_and_lease_initial(
        repository,
        persona,
        creation_request,
    )
    await repository.record_stage_attempt(
        lease.run_id,
        InternalStage.SELECTING_L0_VARIANT,
        now=NOW + timedelta(seconds=1),
    )
    await repository.approve_business_checkpoint(
        lease.run_id,
        InternalStage.SELECTING_L0_VARIANT,
        {"selected_l0_variant": "归返", "selection_rationale": "匹配母题"},
        now=NOW + timedelta(seconds=2),
    )
    await repository.record_stage_attempt(
        lease.run_id,
        InternalStage.GENERATING_STORY_OUTLINE,
        now=NOW + timedelta(seconds=3),
    )

    running = await repository.get_creation(
        accepted.creation_id,
        now=NOW + timedelta(seconds=20),
    )
    assert running is not None
    assert running.initial.state == "running"
    assert running.initial.progress.current_stage == "generating_story_outline"
    assert running.initial.progress.completed_stages == ["determining_direction"]
    assert running.initial.progress.elapsed_seconds == 20
    assert not hasattr(running.initial, "result")
    assert running.initial.drafts.model_dump() == {
        "artifacts": [
            {
                "stage": "determining_direction",
                "selected_l0_variant": "归返",
                "selection_rationale": "匹配母题",
            }
        ],
        "episodes": [],
        "design": None,
        "review_status": {"l0": "pending", "l4": "pending"},
    }

    first_timeout = await repository.handle_run_timeout(
        lease.run_id,
        InternalStage.GENERATING_STORY_OUTLINE,
        now=NOW + timedelta(seconds=30),
    )
    assert first_timeout == "auto_resuming"
    recovering = await repository.get_creation(
        accepted.creation_id,
        now=NOW + timedelta(seconds=40),
    )
    assert recovering is not None
    assert recovering.initial.state == "auto_resuming"
    assert recovering.initial.progress.elapsed_seconds == 30
    assert recovering.initial.progress.recovery_state == "auto_resuming"
    assert recovering.initial.drafts == running.initial.drafts

    resumed = await repository.lease_next_job(
        "worker-2",
        30,
        now=NOW + timedelta(seconds=40),
    )
    assert resumed is not None
    await repository.record_stage_attempt(
        resumed.run_id,
        InternalStage.GENERATING_STORY_OUTLINE,
        now=NOW + timedelta(seconds=41),
    )
    second_timeout = await repository.handle_run_timeout(
        resumed.run_id,
        InternalStage.GENERATING_STORY_OUTLINE,
        now=NOW + timedelta(seconds=55),
    )
    assert second_timeout == "paused"

    paused = await repository.get_creation(accepted.creation_id)
    assert paused is not None
    assert paused.initial.state == "paused"
    assert paused.initial.pause.code == "run_timeout"
    assert paused.initial.pause.timeout_count == 2
    assert paused.initial.progress.can_continue is True
    assert paused.initial.progress.can_end is True
    assert paused.initial.drafts == running.initial.drafts

    first_continue = await repository.continue_run(
        creation_id=accepted.creation_id,
        run_kind="initial",
        idempotency_key="continue-paused",
        now=NOW + timedelta(seconds=60),
    )
    replay_continue = await repository.continue_run(
        creation_id=accepted.creation_id,
        run_kind="initial",
        idempotency_key="continue-paused",
        now=NOW + timedelta(seconds=61),
    )
    duplicate_continue = await repository.continue_run(
        creation_id=accepted.creation_id,
        run_kind="initial",
        idempotency_key="continue-paused-again",
        now=NOW + timedelta(seconds=61),
    )
    assert replay_continue == first_continue == duplicate_continue
    assert first_continue.run_state == "queued"
    continued = await repository.get_creation(accepted.creation_id)
    assert continued is not None
    assert continued.initial.state == "queued"
    assert continued.initial.drafts == running.initial.drafts

    resumed_again = await repository.lease_next_job(
        "worker-3",
        30,
        now=NOW + timedelta(seconds=62),
    )
    assert resumed_again is not None
    await repository.record_stage_attempt(
        resumed_again.run_id,
        InternalStage.GENERATING_STORY_OUTLINE,
        now=NOW + timedelta(seconds=63),
    )
    assert (
        await repository.handle_run_timeout(
            resumed_again.run_id,
            InternalStage.GENERATING_STORY_OUTLINE,
            now=NOW + timedelta(seconds=70),
        )
        == "failed"
    )
    exhausted = await repository.get_creation(accepted.creation_id)
    assert exhausted is not None
    assert exhausted.initial.state == "failed"
    assert exhausted.initial.failure.code == "attempts_exhausted"
    assert exhausted.initial.progress.can_continue is False
    assert exhausted.initial.progress.can_end is False
    assert exhausted.initial.drafts == running.initial.drafts
    assert await repository.reconcile_startup(now=NOW + timedelta(minutes=5)) == []
    with pytest.raises(DomainError) as cannot_continue:
        await repository.continue_run(
            creation_id=accepted.creation_id,
            run_kind="initial",
            idempotency_key="continue-ended",
        )
    assert cannot_continue.value.code == "run_not_controllable"


async def test_retry_run_revives_external_relay_failure_and_requeues_the_same_thread(
    repository,
    persona,
    creation_request,
) -> None:
    accepted, lease = await create_and_lease_initial(repository, persona, creation_request)
    await repository.approve_business_checkpoint(
        lease.run_id,
        InternalStage.SELECTING_L0_VARIANT,
        {
            "stage": "selecting_l0_variant",
            "selected_l0_variant": "主动选择",
            "selection_rationale": "符合测试故事。",
        },
        now=NOW,
    )
    await repository.record_stage_attempt(
        lease.run_id, InternalStage.GENERATING_STORY_OUTLINE, now=NOW
    )
    await repository.fail_run(
        lease.run_id,
        RunFailure(
            code="relay_unavailable",
            message="The model relay request failed (HTTP 402).",
            failed_stage=InternalStage.GENERATING_STORY_OUTLINE,
            attempt_count=1,
        ),
        now=NOW,
    )
    failed = await repository.get_creation(accepted.creation_id)
    assert failed is not None
    assert failed.initial.state == "failed"
    assert failed.initial.progress.can_retry is True

    first = await repository.retry_run(
        creation_id=accepted.creation_id,
        run_kind="initial",
        idempotency_key="retry-relay",
        now=NOW + timedelta(seconds=10),
    )
    replay = await repository.retry_run(
        creation_id=accepted.creation_id,
        run_kind="initial",
        idempotency_key="retry-relay",
        now=NOW + timedelta(seconds=11),
    )
    assert replay == first
    assert first.run_state == "queued"

    revived = await repository.get_creation(accepted.creation_id)
    assert revived is not None
    assert revived.initial.state == "queued"
    checkpoints = await repository.get_business_checkpoints(lease.run_id)
    assert checkpoints[InternalStage.SELECTING_L0_VARIANT]["selected_l0_variant"] == "主动选择"

    resumed = await repository.lease_next_job(
        "worker-2",
        30,
        now=NOW + timedelta(seconds=12),
    )
    assert resumed is not None
    assert resumed.run_id == lease.run_id
    assert resumed.thread_id == lease.thread_id


async def test_retry_run_revives_stage_validation_failure_with_durable_outline(
    repository,
    persona,
    creation_request,
) -> None:
    """A stage_validation_failed run keeps its approved checkpoints and can requeue.

    The outline checkpoint is durable and the worker re-enters the sync through
    the restart-recovery path, so an operator may revive the run after fixing the
    validation cause instead of regenerating the whole outline (issue #222).
    """
    accepted, lease = await create_and_lease_initial(repository, persona, creation_request)
    await repository.approve_business_checkpoint(
        lease.run_id,
        InternalStage.SELECTING_L0_VARIANT,
        {
            "stage": "selecting_l0_variant",
            "selected_l0_variant": "主动选择",
            "selection_rationale": "符合测试故事。",
        },
        now=NOW,
    )
    await repository.record_stage_attempt(
        lease.run_id, InternalStage.GENERATING_STORY_OUTLINE, now=NOW
    )
    await repository.fail_run(
        lease.run_id,
        RunFailure(
            code="stage_validation_failed",
            message="SeriesBible biography projection repair did not pass.",
            failed_stage=InternalStage.GENERATING_EPISODE_OUTLINE,
            attempt_count=1,
        ),
        now=NOW,
    )
    failed = await repository.get_creation(accepted.creation_id)
    assert failed is not None
    assert failed.initial.state == "failed"
    assert failed.initial.progress.can_retry is True

    revived = await repository.retry_run(
        creation_id=accepted.creation_id,
        run_kind="initial",
        idempotency_key="retry-stage-validation",
        now=NOW + timedelta(seconds=10),
    )
    assert revived.run_state == "queued"
    checkpoints = await repository.get_business_checkpoints(lease.run_id)
    assert checkpoints[InternalStage.SELECTING_L0_VARIANT]["selected_l0_variant"] == "主动选择"


async def test_retry_run_rejects_running_and_non_external_failures(
    repository,
    persona,
    creation_request,
) -> None:
    accepted, lease = await create_and_lease_initial(repository, persona, creation_request)

    with pytest.raises(DomainError) as running_reject:
        await repository.retry_run(
            creation_id=accepted.creation_id,
            run_kind="initial",
            idempotency_key="retry-running",
        )
    assert running_reject.value.code == "run_not_controllable"

    for index, code in enumerate(
        ("ended_by_user", "attempts_exhausted", "content_review_rejected")
    ):
        accepted = await repository.create_creation(
            idempotency_key=f"create-retry-code-{index}",
            request=creation_request,
            persona_snapshot=persona,
            now=NOW + timedelta(minutes=index),
        )
        lease = await repository.lease_next_job(
            worker_id=f"worker-code-{index}",
            lease_seconds=30,
            now=NOW + timedelta(minutes=index),
        )
        assert lease is not None
        await repository.fail_run(
            lease.run_id,
            RunFailure(
                code=code,
                message="Terminal failure under test.",
                failed_stage=InternalStage.GENERATING_STORY_OUTLINE,
                attempt_count=1,
            ),
            now=NOW + timedelta(minutes=index),
        )
        failed = await repository.get_creation(accepted.creation_id)
        assert failed is not None
        assert failed.initial.state == "failed"
        assert failed.initial.progress.can_retry is False
        with pytest.raises(DomainError) as code_reject:
            await repository.retry_run(
                creation_id=accepted.creation_id,
                run_kind="initial",
                idempotency_key=f"retry-{code}",
            )
        assert code_reject.value.code == "run_not_controllable"


async def test_retry_run_rejects_revision_kind_and_exhausted_attempts(
    repository,
    persona,
    creation_request,
) -> None:
    accepted, initial_lease = await create_and_lease_initial(
        repository,
        persona,
        creation_request,
    )
    await repository.succeed_run(initial_lease.run_id, make_delivery(), now=NOW)
    await repository.create_or_retry_revision(
        creation_id=accepted.creation_id,
        idempotency_key="revision-1",
        request=RevisionRequest(feedback="加强结尾行动。"),
        now=NOW + timedelta(minutes=1),
    )
    revision_lease = await repository.lease_next_job(
        worker_id="worker-1",
        lease_seconds=30,
        now=NOW + timedelta(minutes=1),
    )
    assert revision_lease is not None
    await repository.fail_run(
        revision_lease.run_id,
        RunFailure(
            code="relay_unavailable",
            message="The model relay request failed (HTTP 402).",
            failed_stage=InternalStage.GENERATING_STORY_OUTLINE,
            attempt_count=1,
        ),
    )
    with pytest.raises(DomainError) as revision_reject:
        await repository.retry_run(
            creation_id=accepted.creation_id,
            run_kind="revision",
            idempotency_key="retry-revision",
        )
    assert revision_reject.value.code == "run_not_controllable"
    assert "frozen feedback" in revision_reject.value.message

    accepted = await repository.create_creation(
        idempotency_key="create-retry-exhausted",
        request=creation_request,
        persona_snapshot=persona,
        now=NOW + timedelta(minutes=2),
    )
    lease = await repository.lease_next_job(
        worker_id="worker-retry-exhausted",
        lease_seconds=30,
        now=NOW + timedelta(minutes=2),
    )
    assert lease is not None
    for index in range(MAX_STAGE_ATTEMPTS):
        await repository.record_stage_attempt(
            lease.run_id,
            InternalStage.GENERATING_STORY_OUTLINE,
            now=NOW + timedelta(seconds=index),
        )
    await repository.fail_run(
        lease.run_id,
        RunFailure(
            code="relay_unavailable",
            message="The model relay request failed (HTTP 402).",
            failed_stage=InternalStage.GENERATING_STORY_OUTLINE,
            attempt_count=MAX_STAGE_ATTEMPTS,
        ),
        now=NOW + timedelta(seconds=30),
    )
    failed = await repository.get_creation(accepted.creation_id)
    assert failed is not None
    assert failed.initial.progress.can_retry is False
    with pytest.raises(DomainError) as exhausted_reject:
        await repository.retry_run(
            creation_id=accepted.creation_id,
            run_kind="initial",
            idempotency_key="retry-exhausted",
        )
    assert exhausted_reject.value.code == "run_not_controllable"
    assert "attempt limit" in exhausted_reject.value.message


async def test_relay_interruption_is_delayed_and_shares_the_stage_recovery_budget(
    repository,
    persona,
    creation_request,
) -> None:
    accepted, lease = await create_and_lease_initial(repository, persona, creation_request)
    await repository.record_stage_attempt(
        lease.run_id,
        InternalStage.SELECTING_L0_VARIANT,
        now=NOW,
    )
    await repository.approve_business_checkpoint(
        lease.run_id,
        InternalStage.SELECTING_L0_VARIANT,
        {"selected_l0_variant": "归返", "selection_rationale": "匹配母题"},
        now=NOW + timedelta(seconds=1),
    )
    await repository.record_stage_attempt(
        lease.run_id,
        InternalStage.GENERATING_STORY_OUTLINE,
        now=NOW + timedelta(seconds=2),
    )

    assert (
        await repository.handle_run_relay_interruption(
            lease.run_id,
            InternalStage.GENERATING_STORY_OUTLINE,
            retry_delay_seconds=17,
            now=NOW + timedelta(seconds=3),
        )
        == "auto_resuming"
    )
    recovering = await repository.get_creation(accepted.creation_id, now=NOW + timedelta(seconds=4))
    assert recovering is not None
    assert recovering.initial.state == "auto_resuming"
    assert recovering.initial.progress.recovery_reason == "relay_interruption"
    assert recovering.initial.drafts.artifacts[0].selected_l0_variant == "归返"

    restarted = Repository(repository.database_path)
    await restarted.initialize()
    after_restart = await restarted.get_creation(
        accepted.creation_id,
        now=NOW + timedelta(seconds=5),
    )
    assert after_restart == recovering
    assert await restarted.lease_next_job("too-early", 30, now=NOW + timedelta(seconds=19)) is None
    resumed = await restarted.lease_next_job("after-delay", 30, now=NOW + timedelta(seconds=20))
    assert resumed is not None
    assert resumed.run_id == lease.run_id
    assert resumed.thread_id == lease.thread_id

    await restarted.record_stage_attempt(
        resumed.run_id,
        InternalStage.GENERATING_STORY_OUTLINE,
        now=NOW + timedelta(seconds=21),
    )
    assert (
        await restarted.handle_run_timeout(
            resumed.run_id,
            InternalStage.GENERATING_STORY_OUTLINE,
            now=NOW + timedelta(seconds=22),
        )
        == "paused"
    )
    paused = await restarted.get_creation(accepted.creation_id, now=NOW + timedelta(seconds=23))
    assert paused is not None
    assert paused.initial.pause.code == "run_timeout"
    assert paused.initial.pause.timeout_count == 2
    assert paused.initial.progress.recovery_reason == "run_timeout"
    assert paused.initial.progress.can_continue is True
    assert paused.initial.progress.can_end is True


@pytest.mark.asyncio
async def test_quality_rejection_is_durable_and_retries_the_same_run(
    repository,
    persona,
    creation_request,
) -> None:
    accepted, lease = await create_and_lease_initial(repository, persona, creation_request)
    checkpoints = [
        (
            InternalStage.SELECTING_L0_VARIANT,
            {"selected_l0_variant": "归返", "selection_rationale": "匹配母题"},
        ),
        (
            InternalStage.GENERATING_STORY_OUTLINE,
            {"content": "故事梗概"},
        ),
        (
            InternalStage.GENERATING_CHARACTER_RELATIONSHIPS,
            {
                "character_biographies": "人物小传",
                "relationship_logic": "人物关系",
            },
        ),
        (
            InternalStage.GENERATING_EPISODE_OUTLINE,
            {
                "content": "单集大纲",
                "episode_count": 1,
                "episodes": [{"episode_number": 1, "plan": "第一集计划"}],
            },
        ),
    ]
    for stage, payload in checkpoints:
        await repository.record_stage_attempt(lease.run_id, stage, now=NOW)
        await repository.approve_business_checkpoint(lease.run_id, stage, payload, now=NOW)
    await repository.record_episode_attempt(lease.run_id, 1, now=NOW)
    draft = await repository.commit_episode_draft(lease.run_id, 1, "第一集剧本", now=NOW)
    await repository.approve_business_checkpoint(
        lease.run_id,
        InternalStage.GENERATING_EPISODE_SCRIPTS,
        {"content": await repository.assemble_episode_scripts(lease.run_id)},
        now=NOW,
    )
    await repository.record_stage_attempt(lease.run_id, InternalStage.ACCEPTING_L0, now=NOW)
    approved_before = await repository.get_business_checkpoints(lease.run_id)

    rejection = await repository.reject_quality_gate(
        lease.run_id,
        stage=InternalStage.ACCEPTING_L0,
        evidence="L0 创作内核与人物选择不一致。",
        now=NOW + timedelta(seconds=1),
    )
    assert rejection.model_dump() == {
        "code": "quality_gate_rejected",
        "stage": "accepting_l0",
        "evidence": "L0 创作内核与人物选择不一致。",
        "attempt_count": 1,
        "can_retry": True,
        "repair_plan": None,
        "repair_state": "available",
    }
    rejected = await repository.get_creation(accepted.creation_id, now=NOW + timedelta(seconds=2))
    assert rejected is not None
    assert rejected.initial.state == "quality_rejected"
    assert rejected.initial.quality_rejection == rejection
    assert rejected.initial.progress.final_review.model_dump() == {"l0": "failed", "l4": "pending"}
    assert rejected.initial.drafts.episodes == [draft]
    assert await repository.get_business_checkpoints(lease.run_id) == approved_before

    first_retry = await repository.retry_final_review(
        creation_id=accepted.creation_id,
        run_kind="initial",
        idempotency_key="retry-final-review",
        now=NOW + timedelta(seconds=3),
    )
    replay_retry = await repository.retry_final_review(
        creation_id=accepted.creation_id,
        run_kind="initial",
        idempotency_key="retry-final-review",
        now=NOW + timedelta(seconds=4),
    )
    with pytest.raises(DomainError) as duplicate_retry:
        await repository.retry_final_review(
            creation_id=accepted.creation_id,
            run_kind="initial",
            idempotency_key="retry-final-review-again",
            now=NOW + timedelta(seconds=4),
        )
    assert first_retry == replay_retry
    assert first_retry.run_state == "queued"
    assert duplicate_retry.value.code == "run_not_controllable"
    resumed = await repository.lease_next_job(
        "quality-review-worker", 30, now=NOW + timedelta(seconds=5)
    )
    assert resumed is not None
    assert resumed.run_id == lease.run_id
    assert resumed.thread_id == lease.thread_id
    assert await repository.get_business_checkpoints(lease.run_id) == approved_before


@pytest.mark.asyncio
async def test_quality_rejection_does_not_queue_a_fourth_final_review(
    repository,
    persona,
    creation_request,
) -> None:
    accepted, lease = await create_and_lease_initial(repository, persona, creation_request)
    for attempt in range(1, 4):
        if attempt > 1:
            retry = await repository.retry_final_review(
                creation_id=accepted.creation_id,
                run_kind="initial",
                idempotency_key=f"quality-retry-{attempt}",
                now=NOW + timedelta(seconds=attempt),
            )
            assert retry.run_state == "queued"
            next_lease = await repository.lease_next_job(
                f"quality-retry-worker-{attempt}",
                30,
                now=NOW + timedelta(seconds=attempt + 1),
            )
            assert next_lease is not None
            assert next_lease.run_id == lease.run_id
            await repository.mark_run_running(
                next_lease.run_id, now=NOW + timedelta(seconds=attempt + 1)
            )

        await repository.record_stage_attempt(
            lease.run_id,
            InternalStage.ACCEPTING_L0,
            now=NOW + timedelta(seconds=attempt),
        )
        await repository.reject_quality_gate(
            lease.run_id,
            stage=InternalStage.ACCEPTING_L0,
            evidence=f"L0 第 {attempt} 次未通过。",
            now=NOW + timedelta(seconds=attempt),
        )

    with pytest.raises(DomainError) as exhausted:
        await repository.retry_final_review(
            creation_id=accepted.creation_id,
            run_kind="initial",
            idempotency_key="quality-retry-fourth",
            now=NOW + timedelta(seconds=5),
        )

    assert exhausted.value.code == "run_not_controllable"
    rejected = await repository.get_creation(accepted.creation_id, now=NOW + timedelta(seconds=5))
    assert rejected is not None
    assert rejected.initial.state == "quality_rejected"
    assert rejected.initial.quality_rejection.attempt_count == 3
    ended = await repository.end_run(
        creation_id=accepted.creation_id,
        run_kind="initial",
        idempotency_key="quality-retry-end",
        now=NOW + timedelta(seconds=6),
    )
    assert ended.run_state == "ended"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("legacy_attempt_count", "expected_attempt_count", "can_retry"),
    [(1, 1, True), (4, 3, False)],
)
async def test_schema_v3_migrates_legacy_quality_rejection_without_changing_drafts(
    repository,
    persona,
    creation_request,
    legacy_attempt_count,
    expected_attempt_count,
    can_retry,
) -> None:
    accepted, lease = await create_and_lease_initial(repository, persona, creation_request)
    await repository.record_stage_attempt(
        lease.run_id,
        InternalStage.SELECTING_L0_VARIANT,
        now=NOW,
    )
    await repository.approve_business_checkpoint(
        lease.run_id,
        InternalStage.SELECTING_L0_VARIANT,
        {"selected_l0_variant": "归返", "selection_rationale": "匹配母题"},
        now=NOW,
    )
    await repository.record_stage_attempt(lease.run_id, InternalStage.ACCEPTING_L0, now=NOW)
    original_checkpoints = await repository.get_business_checkpoints(lease.run_id)
    await repository.reject_quality_gate(
        lease.run_id,
        stage=InternalStage.ACCEPTING_L0,
        evidence="新版本会保存这条意见。",
        now=NOW + timedelta(seconds=1),
    )

    async with repository._connection() as connection:
        await connection.executescript(
            """
            CREATE TABLE run_progress_v3 (
                run_id TEXT PRIMARY KEY REFERENCES runs(id) ON DELETE CASCADE,
                current_stage TEXT NOT NULL,
                execution_state TEXT NOT NULL CHECK (
                    execution_state IN (
                        'queued', 'running', 'auto_resuming', 'paused',
                        'ended', 'succeeded', 'failed'
                    )
                ),
                elapsed_seconds REAL NOT NULL DEFAULT 0 CHECK (elapsed_seconds >= 0),
                active_started_at TEXT,
                timeout_stage TEXT,
                timeout_count INTEGER NOT NULL DEFAULT 0 CHECK (timeout_count >= 0),
                updated_at TEXT NOT NULL,
                current_episode INTEGER CHECK (current_episode IS NULL OR current_episode >= 1)
            );
            INSERT INTO run_progress_v3(
                run_id, current_stage, execution_state, elapsed_seconds,
                active_started_at, timeout_stage, timeout_count, updated_at, current_episode
            )
            SELECT
                run_id, current_stage, 'failed', elapsed_seconds,
                NULL, timeout_stage, timeout_count, updated_at, current_episode
            FROM run_progress;
            DROP TABLE run_progress;
            ALTER TABLE run_progress_v3 RENAME TO run_progress;
            DROP TABLE quality_gate_rejections;
            DROP TABLE content_rejections;
            ALTER TABLE episode_drafts DROP COLUMN contract_sha256;
            ALTER TABLE episode_drafts DROP COLUMN state_delta_json;
            ALTER TABLE episode_drafts DROP COLUMN series_state_json;
            ALTER TABLE episode_drafts DROP COLUMN series_state_sha256;
            ALTER TABLE episode_drafts DROP COLUMN semantic_review_json;
            ALTER TABLE episode_drafts DROP COLUMN repair_rounds;
            DELETE FROM pengine_schema WHERE version = 4;
            DELETE FROM pengine_schema WHERE version = 5;
            DELETE FROM pengine_schema WHERE version = 6;
            DELETE FROM pengine_schema WHERE version = 7;
            DELETE FROM pengine_schema WHERE version = 8;
            DELETE FROM pengine_schema WHERE version = 9;
            DELETE FROM pengine_schema WHERE version = 10;
            DELETE FROM pengine_schema WHERE version = 11;
            DELETE FROM pengine_schema WHERE version = 12;
            DELETE FROM pengine_schema WHERE version = 13;
            DELETE FROM pengine_schema WHERE version = 14;
            DELETE FROM pengine_schema WHERE version = 15;
            DELETE FROM pengine_schema WHERE version = 16;
            DELETE FROM pengine_schema WHERE version = 17;
            DELETE FROM pengine_schema WHERE version = 18;
            DELETE FROM pengine_schema WHERE version = 19;
            DELETE FROM pengine_schema WHERE version = 20;
            DELETE FROM pengine_schema WHERE version = 21;
            DELETE FROM pengine_schema WHERE version = 22;
                DELETE FROM pengine_schema
                WHERE version IN (23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33);
            ALTER TABLE creations DROP COLUMN output_language;
            """
        )
        await connection.execute(
            """
            UPDATE runs
            SET state = 'failed',
                failure_code = 'quality_gate_rejected',
                failure_message = 'The final quality gate rejected the generated work.',
                failed_stage = 'accepting_l0',
                failure_attempt_count = ?,
                completed_at = ?
            WHERE id = ?
            """,
            (
                legacy_attempt_count,
                (NOW + timedelta(seconds=1)).isoformat(),
                str(lease.run_id),
            ),
        )
        await connection.commit()

    await repository.initialize()
    await repository.initialize()

    migrated = await repository.get_creation(accepted.creation_id, now=NOW + timedelta(seconds=2))
    assert migrated is not None
    assert migrated.initial.state == "quality_rejected"
    assert migrated.initial.quality_rejection.model_dump() == {
        "code": "quality_gate_rejected",
        "stage": "accepting_l0",
        "evidence": None,
        "attempt_count": expected_attempt_count,
        "can_retry": can_retry,
        "repair_plan": None,
        "repair_state": "available" if can_retry else None,
    }
    assert await repository.get_business_checkpoints(lease.run_id) == original_checkpoints
    if can_retry:
        retry = await repository.retry_final_review(
            creation_id=accepted.creation_id,
            run_kind="initial",
            idempotency_key="legacy-quality-retry",
            now=NOW + timedelta(seconds=3),
        )
        assert retry.run_state == "queued"
        resumed = await repository.lease_next_job(
            "legacy-quality-review-worker",
            30,
            now=NOW + timedelta(seconds=4),
        )
        assert resumed is not None
        assert resumed.run_id == lease.run_id
        assert resumed.thread_id == lease.thread_id
    else:
        with pytest.raises(DomainError) as exhausted:
            await repository.retry_final_review(
                creation_id=accepted.creation_id,
                run_kind="initial",
                idempotency_key="legacy-quality-retry",
                now=NOW + timedelta(seconds=3),
            )
        assert exhausted.value.code == "run_not_controllable"
        ended = await repository.end_run(
            creation_id=accepted.creation_id,
            run_kind="initial",
            idempotency_key="legacy-quality-end",
            now=NOW + timedelta(seconds=4),
        )
        assert ended.run_state == "ended"
    assert await repository.get_business_checkpoints(lease.run_id) == original_checkpoints


async def test_episode_drafts_are_ordered_and_isolate_a_revision(
    repository,
    persona,
    creation_request,
) -> None:
    accepted, initial_lease = await create_and_lease_initial(
        repository,
        persona,
        creation_request,
    )
    await repository.succeed_run(initial_lease.run_id, make_delivery())
    await repository.create_or_retry_revision(
        creation_id=accepted.creation_id,
        idempotency_key="revision-drafts",
        request=RevisionRequest(feedback="加强人物选择。"),
    )
    revision_lease = await repository.lease_next_job("worker-2", 30)
    assert revision_lease is not None
    checkpoints = [
        (
            InternalStage.SELECTING_L0_VARIANT,
            {"selected_l0_variant": "新方向", "selection_rationale": "响应反馈"},
        ),
        (
            InternalStage.GENERATING_STORY_OUTLINE,
            {"content": "修订故事大纲"},
        ),
        (
            InternalStage.GENERATING_CHARACTER_RELATIONSHIPS,
            {
                "character_biographies": "修订人物小传",
                "relationship_logic": "修订人物关系",
            },
        ),
        (
            InternalStage.GENERATING_EPISODE_OUTLINE,
            {
                "content": "修订分集大纲",
                "episode_count": 2,
                "episodes": [
                    {"episode_number": 1, "plan": "修订第一集"},
                    {"episode_number": 2, "plan": "修订第二集"},
                ],
            },
        ),
    ]
    for stage, payload in checkpoints:
        await repository.approve_business_checkpoint(revision_lease.run_id, stage, payload)
    for episode_number in (1, 2):
        await repository.record_episode_attempt(revision_lease.run_id, episode_number)
        await repository.commit_episode_draft(
            revision_lease.run_id,
            episode_number,
            f"修订第 {episode_number} 集剧本",
        )
    await repository.approve_business_checkpoint(
        revision_lease.run_id,
        InternalStage.GENERATING_EPISODE_SCRIPTS,
        {
            "content": await repository.assemble_episode_scripts(revision_lease.run_id),
        },
    )

    resource = await repository.get_creation(accepted.creation_id)
    assert resource is not None
    assert resource.initial.state == "succeeded"
    assert resource.initial.result == make_delivery()
    assert resource.revision.state == "running"
    assert [artifact.model_dump() for artifact in resource.revision.drafts.artifacts] == [
        {
            "stage": "determining_direction",
            "selected_l0_variant": "新方向",
            "selection_rationale": "响应反馈",
        },
        {"stage": "generating_story_outline", "content": "修订故事大纲"},
        {
            "stage": "generating_character_relationships",
            "content": (
                "# 人物小传 / Character Biographies\n修订人物小传\n\n"
                "# 人物关系 / Relationship Logic\n修订人物关系"
            ),
        },
        {"stage": "generating_episode_outline", "content": "修订分集大纲"},
    ]
    revision_episodes = [
        (draft.episode_number, draft.content) for draft in resource.revision.drafts.episodes
    ]
    assert revision_episodes == [
        (1, "修订第 1 集剧本"),
        (2, "修订第 2 集剧本"),
    ]
    assert all(len(draft.content_sha256) == 64 for draft in resource.revision.drafts.episodes)
    assert resource.revision.drafts.review_status.model_dump() == {
        "l0": "running",
        "l4": "pending",
    }


async def test_episode_drafts_require_ordered_attempts_and_complete_aggregate(
    repository,
    persona,
    creation_request,
) -> None:
    accepted, lease = await create_and_lease_initial(repository, persona, creation_request)
    outline = {
        "content": "两集分集大纲",
        "episode_count": 2,
        "episodes": [
            {"episode_number": 1, "plan": "第一集计划"},
            {"episode_number": 2, "plan": "第二集计划"},
        ],
    }
    await repository.approve_business_checkpoint(
        lease.run_id,
        InternalStage.GENERATING_EPISODE_OUTLINE,
        outline,
        now=NOW,
    )

    assert [plan.model_dump() for plan in await repository.get_episode_plans(lease.run_id)] == [
        {"episode_number": 1, "plan": "第一集计划"},
        {"episode_number": 2, "plan": "第二集计划"},
    ]
    planned = await repository.get_creation(accepted.creation_id, now=NOW)
    assert planned is not None
    assert planned.initial.progress.episodes.model_dump() == {
        "total": 2,
        "completed": 0,
        "current": None,
    }

    with pytest.raises(DomainError) as without_attempt:
        await repository.commit_episode_draft(lease.run_id, 1, "第一集剧本", now=NOW)
    assert without_attempt.value.code == "episode_attempt_required"
    with pytest.raises(DomainError) as out_of_order:
        await repository.record_episode_attempt(lease.run_id, 2, now=NOW)
    assert out_of_order.value.code == "episode_out_of_order"

    assert await repository.record_episode_attempt(lease.run_id, 1, now=NOW) == 1
    first = await repository.commit_episode_draft(
        lease.run_id,
        1,
        "第一集剧本",
        now=NOW + timedelta(seconds=1),
    )
    assert (
        await repository.commit_episode_draft(
            lease.run_id,
            1,
            "第一集剧本",
            now=NOW + timedelta(seconds=2),
        )
        == first
    )
    with pytest.raises(DomainError) as immutable:
        await repository.commit_episode_draft(lease.run_id, 1, "替换的第一集剧本")
    assert immutable.value.code == "episode_conflict"
    with pytest.raises(DomainError) as incomplete:
        await repository.assemble_episode_scripts(lease.run_id)
    assert incomplete.value.code == "episode_sequence_incomplete"
    with pytest.raises(DomainError) as premature_gate:
        await repository.approve_business_checkpoint(
            lease.run_id,
            InternalStage.ACCEPTING_L0,
            {"passed": True, "evidence": "尚未完成的检查"},
        )
    assert premature_gate.value.code == "episode_sequence_incomplete"

    assert await repository.record_episode_attempt(lease.run_id, 2, now=NOW) == 1
    assert await repository.record_episode_attempt(lease.run_id, 2, now=NOW) == 2
    assert await repository.record_episode_attempt(lease.run_id, 2, now=NOW) == 3
    with pytest.raises(DomainError) as exhausted:
        await repository.record_episode_attempt(lease.run_id, 2, now=NOW)
    assert exhausted.value.code == "attempts_exhausted"
    second = await repository.commit_episode_draft(
        lease.run_id,
        2,
        "第二集剧本",
        now=NOW + timedelta(seconds=3),
    )
    assert await repository.get_episode_attempt_counts(lease.run_id) == {1: 1, 2: 3}
    assert await repository.get_episode_attempt_cycles(lease.run_id) == {1: 0, 2: 0}

    aggregate = "第 1 集\n第一集剧本\n\n---\n\n第 2 集\n第二集剧本"
    assert await repository.assemble_episode_scripts(lease.run_id) == aggregate
    await repository.approve_business_checkpoint(
        lease.run_id,
        InternalStage.GENERATING_EPISODE_SCRIPTS,
        {"content": aggregate},
        now=NOW + timedelta(seconds=4),
    )

    active = await repository.get_creation(accepted.creation_id, now=NOW + timedelta(seconds=4))
    assert active is not None
    assert active.initial.state == "running"
    assert active.initial.progress.episodes.model_dump() == {
        "total": 2,
        "completed": 2,
        "current": None,
    }
    assert active.initial.drafts.episodes == [first, second]
    with pytest.raises(DomainError) as delivery_without_aggregate:
        await repository.succeed_run(lease.run_id, make_delivery())
    assert delivery_without_aggregate.value.code == "episode_aggregate_conflict"


async def test_episode_timeout_recovers_first_unfinished_and_ended_run_keeps_drafts(
    repository,
    persona,
    creation_request,
) -> None:
    accepted, lease = await create_and_lease_initial(repository, persona, creation_request)
    await repository.approve_business_checkpoint(
        lease.run_id,
        InternalStage.GENERATING_EPISODE_OUTLINE,
        {
            "content": "两集分集大纲",
            "episode_count": 2,
            "episodes": [
                {"episode_number": 1, "plan": "第一集计划"},
                {"episode_number": 2, "plan": "第二集计划"},
            ],
        },
        now=NOW,
    )
    await repository.record_episode_attempt(lease.run_id, 1, now=NOW)
    first = await repository.commit_episode_draft(
        lease.run_id,
        1,
        "第一集剧本",
        now=NOW + timedelta(seconds=1),
    )
    await repository.record_episode_attempt(lease.run_id, 2, now=NOW + timedelta(seconds=2))

    assert (
        await repository.handle_episode_relay_interruption(
            lease.run_id,
            2,
            retry_delay_seconds=10,
            now=NOW + timedelta(seconds=3),
        )
        == "auto_resuming"
    )
    recovering = await repository.get_creation(accepted.creation_id, now=NOW + timedelta(seconds=3))
    assert recovering is not None
    assert recovering.initial.state == "auto_resuming"
    assert recovering.initial.progress.recovery_reason == "relay_interruption"
    assert recovering.initial.progress.episodes.model_dump() == {
        "total": 2,
        "completed": 1,
        "current": 2,
    }
    assert recovering.initial.drafts.episodes == [first]
    recovery = (await repository.reconcile_startup(now=NOW + timedelta(seconds=13)))[0]
    assert recovery.run_id == lease.run_id
    assert recovery.episode_drafts == [first]

    resumed = await repository.lease_next_job("worker-2", 30, now=NOW + timedelta(seconds=13))
    assert resumed is not None
    assert resumed.run_id == lease.run_id
    assert (
        await repository.record_episode_attempt(
            resumed.run_id,
            2,
            now=NOW + timedelta(seconds=14),
        )
        == 2
    )
    assert (
        await repository.handle_episode_timeout(
            resumed.run_id,
            2,
            now=NOW + timedelta(seconds=15),
        )
        == "paused"
    )

    paused = await repository.get_creation(accepted.creation_id, now=NOW + timedelta(seconds=15))
    assert paused is not None
    assert paused.initial.state == "paused"
    assert paused.initial.pause.episode_number == 2
    assert paused.initial.pause.timeout_count == 2
    assert paused.initial.pause.code == "run_timeout"
    assert paused.initial.drafts.episodes == [first]
    await repository.end_run(
        creation_id=accepted.creation_id,
        run_kind="initial",
        idempotency_key="end-episode-timeout",
        now=NOW + timedelta(seconds=16),
    )
    ended = await repository.get_creation(accepted.creation_id, now=NOW + timedelta(seconds=16))
    assert ended is not None
    assert ended.initial.state == "ended"
    assert ended.initial.drafts.episodes == [first]


async def test_contract_bound_episode_lock_persists_and_controls_aggregate(
    repository,
    persona,
    creation_request,
) -> None:
    accepted, lease = await create_and_lease_initial(repository, persona, creation_request)
    contract, outline = locked_outline_payload()
    await repository.record_stage_attempt(
        lease.run_id,
        InternalStage.GENERATING_EPISODE_OUTLINE,
        now=NOW,
    )
    review_call_id = persist_succeeded_outline_review(repository, lease.run_id)
    await repository.approve_business_checkpoint(
        lease.run_id,
        InternalStage.GENERATING_EPISODE_OUTLINE,
        outline,
        review_call_id=review_call_id,
        now=NOW,
    )
    await repository.record_episode_attempt(lease.run_id, 1, now=NOW)
    content = "林岚：日期是2015-08-12，时间是21:40，记录持续80分钟。\n门后传来第二次敲击"
    contract_hash = story_contract_sha256(contract)
    episode_lock = build_episode_lock(
        contract=contract,
        contract_sha256=contract_hash,
        prior_state=initial_series_state(contract, contract_hash),
        content=content,
        delta=make_delta(contract),
        semantic_review=SemanticReview(
            passed=True,
            evidence="独立分集审查通过",
            issues=[],
        ),
        repair_rounds=1,
    )

    with pytest.raises(DomainError) as missing_lock:
        await repository.commit_episode_draft(lease.run_id, 1, content, now=NOW)
    assert missing_lock.value.code == "episode_lock_required"

    draft = await repository.commit_episode_draft(
        lease.run_id,
        1,
        content,
        episode_lock=episode_lock,
        now=NOW,
    )
    assert draft.contract_sha256 == contract_hash
    assert draft.series_state == episode_lock.series_state
    assert draft.semantic_review is not None and draft.semantic_review.passed

    conflicting_lock = build_episode_lock(
        contract=contract,
        contract_sha256=contract_hash,
        prior_state=initial_series_state(contract, contract_hash),
        content=content,
        delta=make_delta(contract),
        semantic_review=SemanticReview(
            passed=True,
            evidence="另一份不可替换的审查证据",
            issues=[],
        ),
        repair_rounds=1,
    )
    with pytest.raises(DomainError) as lock_conflict:
        await repository.commit_episode_draft(
            lease.run_id,
            1,
            content,
            episode_lock=conflicting_lock,
            now=NOW,
        )
    assert lock_conflict.value.code == "episode_conflict"

    aggregate = dict(await repository.episode_aggregate_checkpoint_payload(lease.run_id))
    assert aggregate["contract_sha256"] == contract_hash
    assert aggregate["episode_hashes"] == [
        {
            "episode_number": 1,
            "content_sha256": draft.content_sha256,
            "series_state_sha256": draft.series_state_sha256,
        }
    ]
    with pytest.raises(DomainError) as mismatched_hash:
        await repository.approve_business_checkpoint(
            lease.run_id,
            InternalStage.GENERATING_EPISODE_SCRIPTS,
            {
                "stage": "generating_episode_scripts",
                **aggregate,
                "contract_sha256": "f" * 64,
            },
        )
    assert mismatched_hash.value.code == "episode_aggregate_conflict"

    checkpoint = {
        "stage": "generating_episode_scripts",
        **aggregate,
    }
    await repository.approve_business_checkpoint(
        lease.run_id,
        InternalStage.GENERATING_EPISODE_SCRIPTS,
        checkpoint,
    )
    reopened = Repository(repository.database_path)
    await reopened.initialize()
    assert await reopened.get_episode_drafts(lease.run_id) == [draft]
    assert (await reopened.get_business_checkpoints(lease.run_id))[
        InternalStage.GENERATING_EPISODE_SCRIPTS
    ] == checkpoint
    assert (await reopened.get_creation(accepted.creation_id)) is not None


async def test_content_rejection_pauses_with_evidence_without_consuming_writer_attempts(
    repository,
    persona,
    creation_request,
) -> None:
    accepted, lease = await create_and_lease_initial(repository, persona, creation_request)
    _, outline = locked_outline_payload()
    await repository.record_stage_attempt(
        lease.run_id,
        InternalStage.GENERATING_EPISODE_OUTLINE,
        now=NOW,
    )
    review_call_id = persist_succeeded_outline_review(repository, lease.run_id)
    await repository.approve_business_checkpoint(
        lease.run_id,
        InternalStage.GENERATING_EPISODE_OUTLINE,
        outline,
        review_call_id=review_call_id,
        now=NOW,
    )
    await repository.record_episode_attempt(lease.run_id, 1, now=NOW)

    for repair_rounds in (3, 6):
        with pytest.raises(ValueError, match="Episode content rejection requires two"):
            await repository.pause_content_rejection(
                lease.run_id,
                stage=InternalStage.GENERATING_EPISODE_SCRIPTS,
                evidence="分集语义审查仍失败。",
                repair_rounds=repair_rounds,
                episode_number=1,
                now=NOW + timedelta(milliseconds=500),
            )

    await repository.pause_content_rejection(
        lease.run_id,
        stage=InternalStage.GENERATING_EPISODE_SCRIPTS,
        evidence="人物知识状态仍与合同冲突。",
        repair_rounds=2,
        episode_number=1,
        now=NOW + timedelta(seconds=1),
    )

    paused = await repository.get_creation(
        accepted.creation_id,
        now=NOW + timedelta(seconds=1),
    )
    assert paused is not None
    assert paused.initial.state == "paused"
    assert paused.initial.pause.code == "content_rejected"
    assert paused.initial.pause.message == "人物知识状态仍与合同冲突。"
    assert paused.initial.pause.content_repair_count == 2
    assert paused.initial.pause.timeout_count is None
    assert paused.initial.progress.can_continue is True
    assert await repository.get_episode_attempt_counts(lease.run_id) == {1: 1}

    continued = await repository.continue_run(
        creation_id=accepted.creation_id,
        run_kind="initial",
        idempotency_key="continue-after-content-review",
        now=NOW + timedelta(seconds=2),
    )
    assert continued.run_state == "queued"
    resumed = await repository.get_creation(
        accepted.creation_id,
        now=NOW + timedelta(seconds=2),
    )
    assert resumed is not None
    assert resumed.initial.state == "queued"
    assert resumed.initial.progress.recovery_reason == "none"
    assert await repository.get_episode_attempt_counts(lease.run_id) == {1: 1}


@pytest.mark.parametrize("repair_rounds", [3, 6])
async def test_episode_outline_content_rejection_keeps_two_round_limit(
    repository,
    persona,
    creation_request,
    repair_rounds,
) -> None:
    _, lease = await create_and_lease_initial(repository, persona, creation_request)
    stage = InternalStage.GENERATING_EPISODE_OUTLINE
    await repository.record_stage_attempt(lease.run_id, stage, now=NOW)

    with pytest.raises(ValueError, match="Episode content rejection requires two"):
        await repository.pause_content_rejection(
            lease.run_id,
            stage=stage,
            evidence="分集大纲语义审查仍失败。",
            repair_rounds=repair_rounds,
            now=NOW + timedelta(seconds=1),
        )


@pytest.mark.parametrize(
    ("stage", "user_stage"),
    [
        (InternalStage.GENERATING_STORY_OUTLINE, "generating_story_outline"),
        (
            InternalStage.GENERATING_CHARACTER_RELATIONSHIPS,
            "generating_character_relationships",
        ),
    ],
)
async def test_story_text_content_rejection_pauses_and_continues_without_episode(
    repository,
    persona,
    creation_request,
    stage,
    user_stage,
) -> None:
    accepted, lease = await create_and_lease_initial(repository, persona, creation_request)
    await repository.record_stage_attempt(lease.run_id, stage, now=NOW)

    with pytest.raises(ValueError, match="required only for episode script"):
        await repository.pause_content_rejection(
            lease.run_id,
            stage=stage,
            evidence="故事文本仍有内容冲突。",
            repair_rounds=2,
            episode_number=1,
            now=NOW + timedelta(seconds=1),
        )

    with pytest.raises(ValueError, match="Story content rejection requires two to four"):
        await repository.pause_content_rejection(
            lease.run_id,
            stage=stage,
            evidence="故事文本仍有内容冲突。",
            repair_rounds=5,
            now=NOW + timedelta(seconds=1),
        )

    await repository.pause_content_rejection(
        lease.run_id,
        stage=stage,
        evidence="故事文本仍有内容冲突。",
        repair_rounds=4,
        now=NOW + timedelta(seconds=1),
    )

    paused = await repository.get_creation(
        accepted.creation_id,
        now=NOW + timedelta(seconds=1),
    )
    assert paused is not None
    assert paused.initial.state == "paused"
    assert paused.initial.pause.code == "content_rejected"
    assert paused.initial.pause.stage == user_stage
    assert paused.initial.pause.episode_number is None
    assert paused.initial.progress.can_continue is True
    assert await repository.get_stage_attempt_counts(lease.run_id) == {stage: 1}
    async with repository._connection() as connection:
        rejection = await (
            await connection.execute(
                """
                SELECT stage, episode_number, repair_rounds, evidence
                FROM content_rejections
                WHERE run_id = ?
                """,
                (str(lease.run_id),),
            )
        ).fetchone()
    assert paused.initial.pause.content_repair_count == 4
    assert tuple(rejection) == (stage.value, None, 4, "故事文本仍有内容冲突。")

    continued = await repository.continue_run(
        creation_id=accepted.creation_id,
        run_kind="initial",
        idempotency_key=f"continue-after-{stage.value}-review",
        now=NOW + timedelta(seconds=2),
    )
    assert continued.run_state == "queued"
    resumed = await repository.get_creation(
        accepted.creation_id,
        now=NOW + timedelta(seconds=2),
    )
    assert resumed is not None
    assert resumed.initial.state == "queued"
    assert resumed.initial.progress.recovery_reason == "none"
    assert await repository.get_stage_attempt_counts(lease.run_id) == {stage: 1}


async def test_failed_run_keeps_committed_episode_drafts(
    repository,
    persona,
    creation_request,
) -> None:
    accepted, lease = await create_and_lease_initial(repository, persona, creation_request)
    await repository.approve_business_checkpoint(
        lease.run_id,
        InternalStage.GENERATING_EPISODE_OUTLINE,
        {
            "content": "单集分集大纲",
            "episode_count": 1,
            "episodes": [{"episode_number": 1, "plan": "第一集计划"}],
        },
    )
    await repository.record_episode_attempt(lease.run_id, 1)
    draft = await repository.commit_episode_draft(lease.run_id, 1, "第一集剧本")
    await repository.fail_run(
        lease.run_id,
        RunFailure(
            code="internal_error",
            message="The workflow failed safely.",
            failed_stage=InternalStage.GENERATING_EPISODE_SCRIPTS,
            attempt_count=1,
        ),
    )

    failed = await repository.get_creation(accepted.creation_id)
    assert failed is not None
    assert failed.initial.state == "failed"
    assert failed.initial.drafts.episodes == [draft]


async def test_episode_execution_error_pauses_and_continues_only_unfinished_episode(
    repository,
    persona,
    creation_request,
) -> None:
    accepted, lease = await create_and_lease_initial(repository, persona, creation_request)
    await repository.approve_business_checkpoint(
        lease.run_id,
        InternalStage.GENERATING_EPISODE_OUTLINE,
        {
            "stage": "generating_episode_outline",
            "content": "两集分集大纲",
            "episode_count": 2,
            "episodes": [
                {"episode_number": 1, "plan": "第一集计划"},
                {"episode_number": 2, "plan": "第二集计划"},
            ],
        },
    )
    await repository.record_episode_attempt(lease.run_id, 1, now=NOW)
    first = await repository.commit_episode_draft(
        lease.run_id,
        1,
        "第一集不可替换剧本",
        now=NOW + timedelta(seconds=1),
    )
    await repository.record_episode_attempt(
        lease.run_id,
        2,
        now=NOW + timedelta(seconds=2),
    )

    await repository.pause_episode_error(
        lease.run_id,
        episode_number=2,
        safe_message=(
            "算术工具收到非十进制参数。需要先把时刻换算为当日经过分钟数后再计算；"
            "已完成分集不受影响。"
        ),
        now=NOW + timedelta(seconds=3),
    )

    paused = await repository.get_creation(
        accepted.creation_id,
        now=NOW + timedelta(seconds=3),
    )
    assert paused is not None
    assert paused.initial.state == "paused"
    assert paused.initial.pause.code == "episode_error"
    assert paused.initial.pause.episode_number == 2
    assert "非十进制参数" in paused.initial.pause.message
    assert paused.initial.progress.episodes.model_dump() == {
        "total": 2,
        "completed": 1,
        "current": 2,
    }
    assert paused.initial.progress.can_continue is True
    assert paused.initial.drafts.episodes == [first]

    continued = await repository.continue_run(
        creation_id=accepted.creation_id,
        run_kind="initial",
        idempotency_key="continue-episode-error",
        now=NOW + timedelta(seconds=4),
    )
    assert continued.run_state == "queued"
    resumed = await repository.lease_next_job(
        "worker-2",
        30,
        now=NOW + timedelta(seconds=4),
    )
    assert resumed is not None and resumed.run_id == lease.run_id
    assert (
        await repository.record_episode_attempt(
            resumed.run_id,
            2,
            now=NOW + timedelta(seconds=5),
        )
        == 2
    )
    assert await repository.get_episode_drafts(resumed.run_id) == [first]


async def test_episode_error_pause_requires_a_recorded_writer_attempt(
    repository,
    persona,
    creation_request,
) -> None:
    _, lease = await create_and_lease_initial(repository, persona, creation_request)
    await repository.approve_business_checkpoint(
        lease.run_id,
        InternalStage.GENERATING_EPISODE_OUTLINE,
        {
            "stage": "generating_episode_outline",
            "content": "单集分集大纲",
            "episode_count": 1,
            "episodes": [{"episode_number": 1, "plan": "第一集计划"}],
        },
    )

    with pytest.raises(DomainError) as missing_attempt:
        await repository.pause_episode_error(
            lease.run_id,
            episode_number=1,
            safe_message="当前集遇到可恢复错误。",
        )

    assert missing_attempt.value.code == "episode_attempt_required"


async def test_schema_v6_recovers_legacy_failed_episode_without_replacing_drafts(
    repository,
    persona,
    creation_request,
) -> None:
    accepted, lease = await create_and_lease_initial(repository, persona, creation_request)
    await repository.approve_business_checkpoint(
        lease.run_id,
        InternalStage.GENERATING_EPISODE_OUTLINE,
        {
            "stage": "generating_episode_outline",
            "content": "两集旧版分集大纲",
            "episode_count": 2,
            "episodes": [
                {"episode_number": 1, "plan": "第一集计划"},
                {"episode_number": 2, "plan": "第二集计划"},
            ],
        },
    )
    await repository.record_episode_attempt(lease.run_id, 1, now=NOW)
    first = await repository.commit_episode_draft(
        lease.run_id,
        1,
        "旧版第一集不可替换剧本",
        now=NOW + timedelta(seconds=1),
    )
    await repository.record_episode_attempt(
        lease.run_id,
        2,
        now=NOW + timedelta(seconds=2),
    )
    await repository.fail_run(
        lease.run_id,
        RunFailure(
            code="internal_error",
            message="The workflow failed safely.",
            failed_stage=InternalStage.GENERATING_EPISODE_SCRIPTS,
            attempt_count=1,
        ),
        now=NOW + timedelta(seconds=3),
    )
    async with repository._connection() as connection:
        await connection.execute("DELETE FROM pengine_schema WHERE version = 7")
        await connection.execute("DELETE FROM pengine_schema WHERE version = 8")
        await connection.execute("DELETE FROM pengine_schema WHERE version = 9")
        await connection.execute("DELETE FROM pengine_schema WHERE version = 10")
        await connection.execute("DELETE FROM pengine_schema WHERE version = 11")
        await connection.execute("DELETE FROM pengine_schema WHERE version = 12")
        await connection.execute("DELETE FROM pengine_schema WHERE version = 13")
        await connection.execute("DELETE FROM pengine_schema WHERE version = 14")
        await connection.execute("DELETE FROM pengine_schema WHERE version = 15")
        await connection.execute("DELETE FROM pengine_schema WHERE version = 16")
        await connection.execute("DELETE FROM pengine_schema WHERE version = 17")
        await connection.execute("DELETE FROM pengine_schema WHERE version = 18")
        await connection.execute("DELETE FROM pengine_schema WHERE version = 19")
        await connection.execute("DELETE FROM pengine_schema WHERE version = 20")
        await connection.execute("DELETE FROM pengine_schema WHERE version = 21")
        await connection.execute("DELETE FROM pengine_schema WHERE version = 22")
        await connection.execute(
            "DELETE FROM pengine_schema WHERE version IN "
            "(23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33)"
        )
        await connection.execute("ALTER TABLE creations DROP COLUMN output_language")
        await connection.commit()

    restarted = Repository(repository.database_path)
    await restarted.initialize()
    recovered = await restarted.get_creation(
        accepted.creation_id,
        now=NOW + timedelta(seconds=4),
    )

    assert recovered is not None
    assert recovered.initial.state == "paused"
    assert recovered.initial.pause.code == "episode_error"
    assert recovered.initial.pause.episode_number == 2
    assert "旧版本" in recovered.initial.pause.message
    assert recovered.initial.progress.can_continue is True
    assert recovered.initial.drafts.episodes == [first]
    assert await restarted.get_episode_attempt_counts(lease.run_id) == {1: 1, 2: 1}

    continued = await restarted.continue_run(
        creation_id=accepted.creation_id,
        run_kind="initial",
        idempotency_key="continue-migrated-episode",
        now=NOW + timedelta(seconds=5),
    )
    assert continued.run_state == "queued"
    assert await restarted.get_episode_drafts(lease.run_id) == [first]


async def test_schema_v7_creation_is_backfilled_to_chinese_output_language(
    repository,
    persona,
    creation_request,
) -> None:
    _, lease = await create_and_lease_initial(repository, persona, creation_request)
    async with repository._connection() as connection:
        await connection.execute("DELETE FROM pengine_schema WHERE version = 8")
        await connection.execute("DELETE FROM pengine_schema WHERE version = 9")
        await connection.execute("DELETE FROM pengine_schema WHERE version = 10")
        await connection.execute("DELETE FROM pengine_schema WHERE version = 11")
        await connection.execute("DELETE FROM pengine_schema WHERE version = 12")
        await connection.execute("DELETE FROM pengine_schema WHERE version = 13")
        await connection.execute("DELETE FROM pengine_schema WHERE version = 14")
        await connection.execute("DELETE FROM pengine_schema WHERE version = 15")
        await connection.execute("DELETE FROM pengine_schema WHERE version = 16")
        await connection.execute("DELETE FROM pengine_schema WHERE version = 17")
        await connection.execute("DELETE FROM pengine_schema WHERE version = 18")
        await connection.execute("DELETE FROM pengine_schema WHERE version = 19")
        await connection.execute("DELETE FROM pengine_schema WHERE version = 20")
        await connection.execute("DELETE FROM pengine_schema WHERE version = 21")
        await connection.execute("DELETE FROM pengine_schema WHERE version = 22")
        await connection.execute(
            "DELETE FROM pengine_schema WHERE version IN "
            "(23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33)"
        )
        await connection.execute("ALTER TABLE creations DROP COLUMN output_language")
        await connection.commit()

    restarted = Repository(repository.database_path)
    await restarted.initialize()
    work = await restarted.get_run_work_item(lease.run_id)

    assert work.output_language == "zh-CN"
    async with restarted._connection() as connection:
        version = await (
            await connection.execute("SELECT MAX(version) FROM pengine_schema")
        ).fetchone()
        row = await (
            await connection.execute(
                "SELECT output_language FROM creations WHERE id = ?",
                (str(lease.creation_id),),
            )
        ).fetchone()
        with pytest.raises(sqlite3.IntegrityError):
            await connection.execute(
                "UPDATE creations SET output_language = 'en-US' WHERE id = ?",
                (str(lease.creation_id),),
            )

    assert version[0] == SCHEMA_VERSION
    assert row[0] == "zh-CN"


async def test_schema_v8_migration_resumes_when_column_already_exists(
    repository,
    persona,
    creation_request,
) -> None:
    _, lease = await create_and_lease_initial(repository, persona, creation_request)
    async with repository._connection() as connection:
        await connection.execute("DELETE FROM pengine_schema WHERE version = 8")
        await connection.execute("DELETE FROM pengine_schema WHERE version = 9")
        await connection.execute("DELETE FROM pengine_schema WHERE version = 10")
        await connection.execute("DELETE FROM pengine_schema WHERE version = 11")
        await connection.execute("DELETE FROM pengine_schema WHERE version = 12")
        await connection.execute("DELETE FROM pengine_schema WHERE version = 13")
        await connection.execute("DELETE FROM pengine_schema WHERE version = 14")
        await connection.execute("DELETE FROM pengine_schema WHERE version = 15")
        await connection.execute("DELETE FROM pengine_schema WHERE version = 16")
        await connection.execute("DELETE FROM pengine_schema WHERE version = 17")
        await connection.execute("DELETE FROM pengine_schema WHERE version = 18")
        await connection.execute("DELETE FROM pengine_schema WHERE version = 19")
        await connection.execute("DELETE FROM pengine_schema WHERE version = 20")
        await connection.execute("DELETE FROM pengine_schema WHERE version = 21")
        await connection.execute("DELETE FROM pengine_schema WHERE version = 22")
        await connection.execute(
            "DELETE FROM pengine_schema WHERE version IN "
            "(23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33)"
        )
        await connection.execute(
            "UPDATE creations SET output_language = NULL WHERE id = ?",
            (str(lease.creation_id),),
        )
        await connection.commit()

    restarted = Repository(repository.database_path)
    await restarted.initialize()
    work = await restarted.get_run_work_item(lease.run_id)

    assert work.output_language == "zh-CN"


async def test_schema_v8_content_rejections_migrate_without_losing_rows(
    repository,
    persona,
    creation_request,
) -> None:
    accepted, lease = await create_and_lease_initial(repository, persona, creation_request)
    await repository.record_stage_attempt(
        lease.run_id,
        InternalStage.GENERATING_EPISODE_OUTLINE,
        now=NOW,
    )
    await repository.pause_content_rejection(
        lease.run_id,
        stage=InternalStage.GENERATING_EPISODE_OUTLINE,
        evidence="旧版合同审查证据必须保留。",
        repair_rounds=2,
        now=NOW + timedelta(seconds=1),
    )

    async with repository._connection() as connection:
        before = await (
            await connection.execute(
                """
                SELECT run_id, stage, episode_number, repair_rounds, evidence, rejected_at
                FROM content_rejections
                WHERE run_id = ?
                """,
                (str(lease.run_id),),
            )
        ).fetchone()
        await connection.executescript(
            """
            CREATE TABLE content_rejections_v8 (
                run_id TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
                stage TEXT NOT NULL CHECK (
                    stage IN ('generating_episode_outline', 'generating_episode_scripts')
                ),
                episode_number INTEGER CHECK (
                    episode_number IS NULL OR episode_number >= 1
                ),
                repair_rounds INTEGER NOT NULL CHECK (repair_rounds = 2),
                evidence TEXT NOT NULL,
                rejected_at TEXT NOT NULL,
                PRIMARY KEY (run_id, stage, episode_number, rejected_at)
            );
            INSERT INTO content_rejections_v8(
                run_id, stage, episode_number, repair_rounds, evidence, rejected_at
            )
            SELECT
                run_id, stage, episode_number, repair_rounds, evidence, rejected_at
            FROM content_rejections;
            DROP TABLE content_rejections;
            ALTER TABLE content_rejections_v8 RENAME TO content_rejections;
            DELETE FROM pengine_schema WHERE version = 9;
            DELETE FROM pengine_schema WHERE version = 10;
            DELETE FROM pengine_schema WHERE version = 11;
            DELETE FROM pengine_schema WHERE version = 12;
            DELETE FROM pengine_schema WHERE version = 13;
            DELETE FROM pengine_schema WHERE version = 14;
            DELETE FROM pengine_schema WHERE version = 15;
            DELETE FROM pengine_schema WHERE version = 16;
            DELETE FROM pengine_schema WHERE version = 17;
            DELETE FROM pengine_schema WHERE version = 18;
            DELETE FROM pengine_schema WHERE version = 19;
            DELETE FROM pengine_schema WHERE version = 20;
            DELETE FROM pengine_schema WHERE version = 21;
            DELETE FROM pengine_schema WHERE version = 22;
                DELETE FROM pengine_schema
                WHERE version IN (23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33);
            """
        )
        await connection.commit()

    restarted = Repository(repository.database_path)
    await restarted.initialize()

    async with restarted._connection() as connection:
        after = await (
            await connection.execute(
                """
                SELECT run_id, stage, episode_number, repair_rounds, evidence, rejected_at
                FROM content_rejections
                WHERE run_id = ?
                """,
                (str(lease.run_id),),
            )
        ).fetchone()
        version = await (
            await connection.execute("SELECT MAX(version) FROM pengine_schema")
        ).fetchone()
        with pytest.raises(sqlite3.IntegrityError):
            await connection.execute(
                """
                INSERT INTO content_rejections(
                    run_id, stage, episode_number, repair_rounds, evidence, rejected_at
                ) VALUES (?, 'generating_story_outline', 1, 2, '无效', ?)
                """,
                (str(lease.run_id), (NOW + timedelta(seconds=2)).isoformat()),
            )
        with pytest.raises(sqlite3.IntegrityError):
            await connection.execute(
                """
                INSERT INTO content_rejections(
                    run_id, stage, episode_number, repair_rounds, evidence, rejected_at
                ) VALUES (?, 'generating_episode_scripts', NULL, 2, '无效', ?)
                """,
                (str(lease.run_id), (NOW + timedelta(seconds=3)).isoformat()),
            )

    assert tuple(after) == tuple(before)
    assert version[0] == SCHEMA_VERSION
    continued = await restarted.continue_run(
        creation_id=accepted.creation_id,
        run_kind="initial",
        idempotency_key="continue-after-v9-migration",
        now=NOW + timedelta(seconds=4),
    )
    assert continued.run_state == "queued"


async def test_schema_v9_repair_limits_migrate_without_losing_pause(
    repository,
    persona,
    creation_request,
) -> None:
    accepted, lease = await create_and_lease_initial(repository, persona, creation_request)
    stage = InternalStage.GENERATING_STORY_OUTLINE
    await repository.record_stage_attempt(lease.run_id, stage, now=NOW)
    await repository.pause_content_rejection(
        lease.run_id,
        stage=stage,
        evidence="v9 故事审查证据必须保留。",
        repair_rounds=2,
        now=NOW + timedelta(seconds=1),
    )

    async with repository._connection() as connection:
        before_rejection = await (
            await connection.execute(
                """
                SELECT run_id, stage, episode_number, repair_rounds, evidence, rejected_at
                FROM content_rejections
                WHERE run_id = ?
                """,
                (str(lease.run_id),),
            )
        ).fetchone()
        await connection.executescript(
            """
            CREATE TABLE run_progress_v9_legacy (
                run_id TEXT PRIMARY KEY REFERENCES runs(id) ON DELETE CASCADE,
                current_stage TEXT NOT NULL,
                execution_state TEXT NOT NULL CHECK (
                    execution_state IN (
                        'queued', 'running', 'auto_resuming', 'paused',
                        'quality_rejected', 'ended', 'succeeded', 'failed'
                    )
                ),
                elapsed_seconds REAL NOT NULL DEFAULT 0 CHECK (elapsed_seconds >= 0),
                active_started_at TEXT,
                timeout_stage TEXT,
                timeout_count INTEGER NOT NULL DEFAULT 0 CHECK (timeout_count >= 0),
                updated_at TEXT NOT NULL,
                current_episode INTEGER CHECK (
                    current_episode IS NULL OR current_episode >= 1
                ),
                recovery_reason TEXT NOT NULL DEFAULT 'none' CHECK (
                    recovery_reason IN (
                        'none', 'run_timeout', 'relay_interruption', 'content_rejected',
                        'episode_error'
                    )
                ),
                content_repair_count INTEGER CHECK (
                    content_repair_count IS NULL OR content_repair_count = 2
                ),
                pause_message TEXT
            );
            INSERT INTO run_progress_v9_legacy(
                run_id, current_stage, execution_state, elapsed_seconds,
                active_started_at, timeout_stage, timeout_count, updated_at,
                current_episode, recovery_reason, content_repair_count, pause_message
            )
            SELECT
                run_id, current_stage, execution_state, elapsed_seconds,
                active_started_at, timeout_stage, timeout_count, updated_at,
                current_episode, recovery_reason, content_repair_count, pause_message
            FROM run_progress;
            DROP TABLE run_progress;
            ALTER TABLE run_progress_v9_legacy RENAME TO run_progress;

            CREATE TABLE content_rejections_v9_legacy (
                run_id TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
                stage TEXT NOT NULL CHECK (
                    stage IN (
                        'generating_story_outline',
                        'generating_character_biographies',
                        'generating_relationship_logic',
                        'generating_story_design',
                        'generating_episode_outline',
                        'generating_episode_scripts'
                    )
                ),
                episode_number INTEGER,
                repair_rounds INTEGER NOT NULL CHECK (repair_rounds = 2),
                evidence TEXT NOT NULL,
                rejected_at TEXT NOT NULL,
                PRIMARY KEY (run_id, stage, episode_number, rejected_at),
                CHECK (
                    (
                        stage = 'generating_episode_scripts'
                        AND episode_number IS NOT NULL
                        AND episode_number >= 1
                    )
                    OR (
                        stage != 'generating_episode_scripts'
                        AND episode_number IS NULL
                    )
                )
            );
            INSERT INTO content_rejections_v9_legacy(
                run_id, stage, episode_number, repair_rounds, evidence, rejected_at
            )
            SELECT
                run_id, stage, episode_number, repair_rounds, evidence, rejected_at
            FROM content_rejections;
            DROP TABLE content_rejections;
            ALTER TABLE content_rejections_v9_legacy RENAME TO content_rejections;
            DELETE FROM pengine_schema WHERE version = 10;
            DELETE FROM pengine_schema WHERE version = 11;
            DELETE FROM pengine_schema WHERE version = 12;
            DELETE FROM pengine_schema WHERE version = 13;
            DELETE FROM pengine_schema WHERE version = 14;
            DELETE FROM pengine_schema WHERE version = 15;
            DELETE FROM pengine_schema WHERE version = 16;
            DELETE FROM pengine_schema WHERE version = 17;
            DELETE FROM pengine_schema WHERE version = 18;
            DELETE FROM pengine_schema WHERE version = 19;
            DELETE FROM pengine_schema WHERE version = 20;
            DELETE FROM pengine_schema WHERE version = 21;
            DELETE FROM pengine_schema WHERE version = 22;
                DELETE FROM pengine_schema
                WHERE version IN (23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33);
            """
        )
        await connection.commit()

    restarted = Repository(repository.database_path)
    await restarted.initialize()

    migrated = await restarted.get_creation(
        accepted.creation_id,
        now=NOW + timedelta(seconds=2),
    )
    assert migrated is not None
    assert migrated.initial.state == "paused"
    assert migrated.initial.pause.content_repair_count == 2
    async with restarted._connection() as connection:
        after_rejection = await (
            await connection.execute(
                """
                SELECT run_id, stage, episode_number, repair_rounds, evidence, rejected_at
                FROM content_rejections
                WHERE run_id = ?
                """,
                (str(lease.run_id),),
            )
        ).fetchone()
        version = await (
            await connection.execute("SELECT MAX(version) FROM pengine_schema")
        ).fetchone()
    assert tuple(after_rejection) == tuple(before_rejection)
    assert version[0] == SCHEMA_VERSION

    await restarted.continue_run(
        creation_id=accepted.creation_id,
        run_kind="initial",
        idempotency_key="continue-after-v10-migration",
        now=NOW + timedelta(seconds=3),
    )
    resumed_lease = await restarted.lease_next_job(
        worker_id="worker-after-v10",
        lease_seconds=30,
        now=NOW + timedelta(seconds=3),
    )
    assert resumed_lease is not None and resumed_lease.run_id == lease.run_id
    await restarted.pause_content_rejection(
        lease.run_id,
        stage=stage,
        evidence="第四轮故事语义修订后仍有冲突。",
        repair_rounds=4,
        now=NOW + timedelta(seconds=4),
    )
    paused = await restarted.get_creation(
        accepted.creation_id,
        now=NOW + timedelta(seconds=4),
    )
    assert paused is not None
    assert paused.initial.pause.content_repair_count == 4

    async with restarted._connection() as connection:
        for repair_rounds in (3, 6):
            with pytest.raises(sqlite3.IntegrityError):
                await connection.execute(
                    """
                    INSERT INTO content_rejections(
                        run_id, stage, episode_number, repair_rounds, evidence, rejected_at
                    ) VALUES (?, 'generating_episode_outline', NULL, ?, '不允许', ?)
                    """,
                    (
                        str(lease.run_id),
                        repair_rounds,
                        (NOW + timedelta(seconds=repair_rounds + 2)).isoformat(),
                    ),
                )
        with pytest.raises(sqlite3.IntegrityError):
            await connection.execute(
                """
                INSERT INTO content_rejections(
                    run_id, stage, episode_number, repair_rounds, evidence, rejected_at
                ) VALUES (?, 'generating_story_outline', NULL, 7, '超过故事上限', ?)
                """,
                (str(lease.run_id), (NOW + timedelta(seconds=10)).isoformat()),
            )


async def test_schema_v1_database_is_backfilled_without_losing_creation(
    repository,
    persona,
    creation_request,
) -> None:
    accepted = await repository.create_creation(
        idempotency_key="migration-create",
        request=creation_request,
        persona_snapshot=persona,
        now=NOW,
    )
    async with repository._connection() as connection:
        await connection.execute("DROP TABLE run_progress")
        await connection.execute("DELETE FROM pengine_schema WHERE version = 2")
        await connection.execute("DELETE FROM pengine_schema WHERE version = 3")
        await connection.execute("DELETE FROM pengine_schema WHERE version = 4")
        await connection.execute("DELETE FROM pengine_schema WHERE version = 5")
        await connection.execute("DELETE FROM pengine_schema WHERE version = 6")
        await connection.execute("DELETE FROM pengine_schema WHERE version = 7")
        await connection.execute("DELETE FROM pengine_schema WHERE version = 8")
        await connection.execute("DELETE FROM pengine_schema WHERE version = 9")
        await connection.execute("DELETE FROM pengine_schema WHERE version = 10")
        await connection.execute("DELETE FROM pengine_schema WHERE version = 11")
        await connection.execute("DELETE FROM pengine_schema WHERE version = 12")
        await connection.execute("DELETE FROM pengine_schema WHERE version = 13")
        await connection.execute("DELETE FROM pengine_schema WHERE version = 14")
        await connection.execute("DELETE FROM pengine_schema WHERE version = 15")
        await connection.execute("DELETE FROM pengine_schema WHERE version = 16")
        await connection.execute("DELETE FROM pengine_schema WHERE version = 17")
        await connection.execute("DELETE FROM pengine_schema WHERE version = 18")
        await connection.execute("DELETE FROM pengine_schema WHERE version = 19")
        await connection.execute("DELETE FROM pengine_schema WHERE version = 20")
        await connection.execute("DELETE FROM pengine_schema WHERE version = 21")
        await connection.execute("DELETE FROM pengine_schema WHERE version = 22")
        await connection.execute(
            "DELETE FROM pengine_schema WHERE version IN "
            "(23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33)"
        )
        await connection.execute("ALTER TABLE creations DROP COLUMN output_language")
        await connection.execute("DROP TABLE quality_gate_rejections")
        await connection.execute("DROP TABLE episode_timeouts")
        await connection.execute("DROP TABLE episode_attempts")
        await connection.execute("DROP TABLE episode_drafts")
        await connection.execute("DROP TABLE episode_plans")
        await connection.commit()

    await repository.initialize()

    migrated = await repository.get_creation(accepted.creation_id, now=NOW)
    assert migrated is not None
    assert migrated.initial.state == "queued"
    assert migrated.initial.progress.current_stage == "determining_direction"
    async with repository._connection() as connection:
        version = await (
            await connection.execute("SELECT MAX(version) FROM pengine_schema")
        ).fetchone()
    assert version[0] == SCHEMA_VERSION


async def test_schema_v2_database_migrates_to_current_schema_idempotently(
    repository,
    persona,
    creation_request,
) -> None:
    accepted = await repository.create_creation(
        idempotency_key="migration-v2-create",
        request=creation_request,
        persona_snapshot=persona,
        now=NOW,
    )
    async with repository._connection() as connection:
        await connection.execute("DROP TABLE episode_timeouts")
        await connection.execute("DROP TABLE episode_attempts")
        await connection.execute("DROP TABLE episode_drafts")
        await connection.execute("DROP TABLE episode_plans")
        await connection.execute("ALTER TABLE run_progress DROP COLUMN current_episode")
        await connection.execute("DELETE FROM pengine_schema WHERE version = 3")
        await connection.execute("DELETE FROM pengine_schema WHERE version = 4")
        await connection.execute("DELETE FROM pengine_schema WHERE version = 5")
        await connection.execute("DELETE FROM pengine_schema WHERE version = 6")
        await connection.execute("DELETE FROM pengine_schema WHERE version = 7")
        await connection.execute("DELETE FROM pengine_schema WHERE version = 8")
        await connection.execute("DELETE FROM pengine_schema WHERE version = 9")
        await connection.execute("DELETE FROM pengine_schema WHERE version = 10")
        await connection.execute("DELETE FROM pengine_schema WHERE version = 11")
        await connection.execute("DELETE FROM pengine_schema WHERE version = 12")
        await connection.execute("DELETE FROM pengine_schema WHERE version = 13")
        await connection.execute("DELETE FROM pengine_schema WHERE version = 14")
        await connection.execute("DELETE FROM pengine_schema WHERE version = 15")
        await connection.execute("DELETE FROM pengine_schema WHERE version = 16")
        await connection.execute("DELETE FROM pengine_schema WHERE version = 17")
        await connection.execute("DELETE FROM pengine_schema WHERE version = 18")
        await connection.execute("DELETE FROM pengine_schema WHERE version = 19")
        await connection.execute("DELETE FROM pengine_schema WHERE version = 20")
        await connection.execute("DELETE FROM pengine_schema WHERE version = 21")
        await connection.execute("DELETE FROM pengine_schema WHERE version = 22")
        await connection.execute(
            "DELETE FROM pengine_schema WHERE version IN "
            "(23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33)"
        )
        await connection.execute("ALTER TABLE creations DROP COLUMN output_language")
        await connection.execute("DROP TABLE quality_gate_rejections")
        await connection.commit()

    await repository.initialize()
    await repository.initialize()

    migrated = await repository.get_creation(accepted.creation_id, now=NOW)
    assert migrated is not None
    assert migrated.initial.progress.episodes is None
    async with repository._connection() as connection:
        version = await (
            await connection.execute("SELECT MAX(version) FROM pengine_schema")
        ).fetchone()
        current_episode = await (
            await connection.execute("PRAGMA table_info(run_progress)")
        ).fetchall()
    assert version[0] == SCHEMA_VERSION
    assert "current_episode" in {row["name"] for row in current_episode}


async def test_schema_v4_recovery_rows_gain_the_timeout_reason_without_losing_drafts(
    repository,
    persona,
    creation_request,
) -> None:
    accepted, lease = await create_and_lease_initial(repository, persona, creation_request)
    await repository.record_stage_attempt(
        lease.run_id,
        InternalStage.SELECTING_L0_VARIANT,
        now=NOW,
    )
    await repository.approve_business_checkpoint(
        lease.run_id,
        InternalStage.SELECTING_L0_VARIANT,
        {"selected_l0_variant": "归返", "selection_rationale": "匹配母题"},
        now=NOW + timedelta(seconds=1),
    )
    await repository.record_stage_attempt(
        lease.run_id,
        InternalStage.GENERATING_STORY_OUTLINE,
        now=NOW + timedelta(seconds=2),
    )
    assert (
        await repository.handle_run_timeout(
            lease.run_id,
            InternalStage.GENERATING_STORY_OUTLINE,
            now=NOW + timedelta(seconds=3),
        )
        == "auto_resuming"
    )
    before_migration = await repository.get_creation(
        accepted.creation_id,
        now=NOW + timedelta(seconds=4),
    )
    assert before_migration is not None

    async with repository._connection() as connection:
        await connection.execute("ALTER TABLE run_progress DROP COLUMN recovery_reason")
        await connection.execute("ALTER TABLE run_progress DROP COLUMN content_repair_count")
        await connection.execute("ALTER TABLE run_progress DROP COLUMN pause_message")
        await connection.execute("ALTER TABLE episode_drafts DROP COLUMN contract_sha256")
        await connection.execute("ALTER TABLE episode_drafts DROP COLUMN state_delta_json")
        await connection.execute("ALTER TABLE episode_drafts DROP COLUMN series_state_json")
        await connection.execute("ALTER TABLE episode_drafts DROP COLUMN series_state_sha256")
        await connection.execute("ALTER TABLE episode_drafts DROP COLUMN semantic_review_json")
        await connection.execute("ALTER TABLE episode_drafts DROP COLUMN repair_rounds")
        await connection.execute("DROP TABLE content_rejections")
        await connection.execute("DELETE FROM pengine_schema WHERE version = 5")
        await connection.execute("DELETE FROM pengine_schema WHERE version = 6")
        await connection.execute("DELETE FROM pengine_schema WHERE version = 7")
        await connection.execute("DELETE FROM pengine_schema WHERE version = 8")
        await connection.execute("DELETE FROM pengine_schema WHERE version = 9")
        await connection.execute("DELETE FROM pengine_schema WHERE version = 10")
        await connection.execute("DELETE FROM pengine_schema WHERE version = 11")
        await connection.execute("DELETE FROM pengine_schema WHERE version = 12")
        await connection.execute("DELETE FROM pengine_schema WHERE version = 13")
        await connection.execute("DELETE FROM pengine_schema WHERE version = 14")
        await connection.execute("DELETE FROM pengine_schema WHERE version = 15")
        await connection.execute("DELETE FROM pengine_schema WHERE version = 16")
        await connection.execute("DELETE FROM pengine_schema WHERE version = 17")
        await connection.execute("DELETE FROM pengine_schema WHERE version = 18")
        await connection.execute("DELETE FROM pengine_schema WHERE version = 19")
        await connection.execute("DELETE FROM pengine_schema WHERE version = 20")
        await connection.execute("DELETE FROM pengine_schema WHERE version = 21")
        await connection.execute("DELETE FROM pengine_schema WHERE version = 22")
        await connection.execute(
            "DELETE FROM pengine_schema WHERE version IN "
            "(23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33)"
        )
        await connection.execute("ALTER TABLE creations DROP COLUMN output_language")
        await connection.commit()

    await repository.initialize()
    await repository.initialize()
    migrated = await repository.get_creation(accepted.creation_id, now=NOW + timedelta(seconds=5))
    assert migrated is not None
    assert migrated.initial.state == "auto_resuming"
    assert migrated.initial.progress.recovery_reason == "run_timeout"
    assert migrated.initial.drafts == before_migration.initial.drafts


async def test_schema_v10_to_v11_preserves_run_progress_and_model_calls(
    repository,
    persona,
    creation_request,
) -> None:
    from pengine.model_calls import (
        ModelCallContext,
        ModelCallStore,
        build_started_record,
    )

    accepted, lease = await create_and_lease_initial(repository, persona, creation_request)
    await repository.mark_run_running(lease.run_id)
    store = ModelCallStore(repository.database_path)
    record = build_started_record(
        role="generation",
        adapter="anthropic",
        provider="anthropic",
        model="claude-opus-5",
        context=ModelCallContext(
            run_id=str(lease.run_id),
            creation_id=str(accepted.creation_id),
            thread_id=lease.thread_id,
            run_kind="initial",
            stage="selecting_l0_variant",
        ),
        estimated_input_tokens=5,
        estimated_output_tokens=100,
        verified_limit_tokens=200_000,
    )
    record.status = "succeeded"
    record.outcome = "success"
    store.upsert(record)
    store.close()

    # Simulate a v10 database by removing the v11 schema marker only.
    async with repository._connection() as connection:
        await connection.execute("DELETE FROM pengine_schema WHERE version = 11")
        await connection.execute("DELETE FROM pengine_schema WHERE version = 12")
        await connection.execute("DELETE FROM pengine_schema WHERE version = 13")
        await connection.execute("DELETE FROM pengine_schema WHERE version = 14")
        await connection.execute("DELETE FROM pengine_schema WHERE version = 15")
        await connection.execute("DELETE FROM pengine_schema WHERE version = 16")
        await connection.execute("DELETE FROM pengine_schema WHERE version = 17")
        await connection.execute("DELETE FROM pengine_schema WHERE version = 18")
        await connection.execute("DELETE FROM pengine_schema WHERE version = 19")
        await connection.execute("DELETE FROM pengine_schema WHERE version = 20")
        await connection.execute("DELETE FROM pengine_schema WHERE version = 21")
        await connection.execute("DELETE FROM pengine_schema WHERE version = 22")
        await connection.execute(
            "DELETE FROM pengine_schema WHERE version IN "
            "(23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33)"
        )
        await connection.commit()

    restarted = Repository(repository.database_path)
    await restarted.initialize()
    calls = await restarted.get_run_model_calls(lease.run_id)
    assert len(calls) == 1
    assert calls[0].status == "succeeded"
    assert calls[0].usage.status == "unavailable"
    resource = await restarted.get_creation(accepted.creation_id, now=NOW)
    assert resource is not None
    assert resource.initial.state == "running"
    assert len(resource.initial.progress.model_calls) == 1
    assert resource.initial.progress.model_calls[0].call_id == calls[0].call_id


async def test_schema_v20_to_v21_adds_l3_audit_columns_without_changing_old_rows(
    repository,
) -> None:
    store = ModelCallStore(repository.database_path)
    record = build_started_record(
        call_id="legacy-v20-call",
        role="generation",
        adapter="fake",
        provider="fake",
        model="claude-opus-5",
        context=ModelCallContext(run_id="legacy-run", stage="generating_story_outline"),
        estimated_input_tokens=11,
        estimated_output_tokens=22,
        verified_limit_tokens=200_000,
    )
    record.status = "succeeded"
    record.outcome = "success"
    store.upsert(record)
    store.close()

    async with repository._connection() as connection:
        before = await (
            await connection.execute(
                "SELECT call_id, role, model, status, estimated_input_tokens, "
                "estimated_output_tokens FROM model_calls ORDER BY call_id"
            )
        ).fetchall()
        await connection.execute("ALTER TABLE model_calls DROP COLUMN l3_sha256")
        await connection.execute("ALTER TABLE model_calls DROP COLUMN l3_char_count")
        await connection.execute("ALTER TABLE model_calls DROP COLUMN l3_mount_path")
        await connection.execute("ALTER TABLE model_calls DROP COLUMN l3_full_text_mounted")
        await connection.execute("DELETE FROM pengine_schema WHERE version = 21")
        await connection.execute("DELETE FROM pengine_schema WHERE version = 22")
        await connection.execute(
            "DELETE FROM pengine_schema WHERE version IN "
            "(23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33)"
        )
        await connection.commit()

    restarted = Repository(repository.database_path)
    await restarted.initialize()
    await restarted.initialize()
    async with restarted._connection() as connection:
        after = await (
            await connection.execute(
                "SELECT call_id, role, model, status, estimated_input_tokens, "
                "estimated_output_tokens FROM model_calls ORDER BY call_id"
            )
        ).fetchall()
        columns = await (await connection.execute("PRAGMA table_info(model_calls)")).fetchall()
        version = await (
            await connection.execute("SELECT MAX(version) FROM pengine_schema")
        ).fetchone()
        integrity = await (await connection.execute("PRAGMA integrity_check")).fetchone()
        foreign_key_rows = await (await connection.execute("PRAGMA foreign_key_check")).fetchall()
        l3_values = await (
            await connection.execute(
                "SELECT l3_sha256, l3_char_count, l3_mount_path, l3_full_text_mounted "
                "FROM model_calls WHERE call_id = 'legacy-v20-call'"
            )
        ).fetchone()

    assert [tuple(row) for row in after] == [tuple(row) for row in before]
    assert {
        "l3_sha256",
        "l3_char_count",
        "l3_mount_path",
        "l3_full_text_mounted",
    }.issubset({column[1] for column in columns})
    assert tuple(l3_values) == (None, None, None, 0)
    assert version[0] == SCHEMA_VERSION
    assert integrity[0] == "ok"
    assert foreign_key_rows == []


async def test_schema_v21_to_v22_adds_nullable_delivery_presentation(
    repository,
    persona,
    creation_request,
) -> None:
    _, lease = await create_and_lease_initial(repository, persona, creation_request)
    await repository.succeed_run(lease.run_id, make_delivery(), now=NOW)
    async with repository._connection() as connection:
        before = await (
            await connection.execute(
                "SELECT content_package_json, delivery_report_json FROM deliveries "
                "WHERE run_id = ?",
                (str(lease.run_id),),
            )
        ).fetchone()
        await connection.execute("ALTER TABLE deliveries DROP COLUMN presentation_manifest_json")
        await connection.execute("ALTER TABLE deliveries DROP COLUMN presentation_manifest_sha256")
        await connection.execute("DELETE FROM pengine_schema WHERE version = 22")
        await connection.execute(
            "DELETE FROM pengine_schema WHERE version IN "
            "(23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33)"
        )
        await connection.commit()

    restarted = Repository(repository.database_path)
    await restarted.initialize()
    await restarted.initialize()
    async with restarted._connection() as connection:
        columns = await (await connection.execute("PRAGMA table_info(deliveries)")).fetchall()
        after = await (
            await connection.execute(
                "SELECT content_package_json, delivery_report_json, "
                "presentation_manifest_json, presentation_manifest_sha256 "
                "FROM deliveries WHERE run_id = ?",
                (str(lease.run_id),),
            )
        ).fetchone()
        version = await (
            await connection.execute("SELECT MAX(version) FROM pengine_schema")
        ).fetchone()

    assert {"presentation_manifest_json", "presentation_manifest_sha256"}.issubset(
        {column[1] for column in columns}
    )
    assert tuple(after[:2]) == tuple(before)
    assert tuple(after[2:]) == (None, None)
    assert version[0] == SCHEMA_VERSION


async def test_schema_v17_to_v18_adds_hidden_model_call_provenance(repository) -> None:
    async with repository._connection() as connection:
        await connection.execute("DROP INDEX model_calls_operation_id")
        await connection.execute("ALTER TABLE model_calls DROP COLUMN operation_id")
        await connection.execute("ALTER TABLE business_checkpoints DROP COLUMN review_call_id")
        await connection.execute("DELETE FROM pengine_schema WHERE version = 18")
        await connection.execute("DELETE FROM pengine_schema WHERE version = 19")
        await connection.execute("DELETE FROM pengine_schema WHERE version = 20")
        await connection.execute("DELETE FROM pengine_schema WHERE version = 21")
        await connection.execute("DELETE FROM pengine_schema WHERE version = 22")
        await connection.execute(
            "DELETE FROM pengine_schema WHERE version IN "
            "(23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33)"
        )
        await connection.commit()

    restarted = Repository(repository.database_path)
    await restarted.initialize()
    async with restarted._connection() as connection:
        columns = await (await connection.execute("PRAGMA table_info(model_calls)")).fetchall()
        checkpoint_columns = await (
            await connection.execute("PRAGMA table_info(business_checkpoints)")
        ).fetchall()
        indexes = await (await connection.execute("PRAGMA index_list(model_calls)")).fetchall()
        version = await (
            await connection.execute("SELECT MAX(version) FROM pengine_schema")
        ).fetchone()

    assert "operation_id" in {column[1] for column in columns}
    assert "review_call_id" in {column[1] for column in checkpoint_columns}
    assert "model_calls_operation_id" in {index[1] for index in indexes}
    assert version[0] == SCHEMA_VERSION


async def test_schema_v18_collision_repairs_missing_model_call_provenance(repository) -> None:
    async with repository._connection() as connection:
        await connection.execute("DROP INDEX model_calls_operation_id")
        await connection.execute("ALTER TABLE model_calls DROP COLUMN operation_id")
        await connection.execute("ALTER TABLE business_checkpoints DROP COLUMN review_call_id")
        await connection.execute("DELETE FROM pengine_schema WHERE version = 19")
        await connection.execute("DELETE FROM pengine_schema WHERE version = 20")
        await connection.execute("DELETE FROM pengine_schema WHERE version = 21")
        await connection.execute("DELETE FROM pengine_schema WHERE version = 22")
        await connection.execute(
            "DELETE FROM pengine_schema WHERE version IN "
            "(23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33)"
        )
        await connection.commit()

    restarted = Repository(repository.database_path)
    await restarted.initialize()
    async with restarted._connection() as connection:
        columns = await (await connection.execute("PRAGMA table_info(model_calls)")).fetchall()
        checkpoint_columns = await (
            await connection.execute("PRAGMA table_info(business_checkpoints)")
        ).fetchall()
        indexes = await (await connection.execute("PRAGMA index_list(model_calls)")).fetchall()
        version = await (
            await connection.execute("SELECT MAX(version) FROM pengine_schema")
        ).fetchone()

    assert "operation_id" in {column[1] for column in columns}
    assert "review_call_id" in {column[1] for column in checkpoint_columns}
    assert "model_calls_operation_id" in {index[1] for index in indexes}
    assert version[0] == SCHEMA_VERSION


async def test_schema_v18_collision_repairs_missing_episode_attempt_cycles(
    repository,
    persona,
    creation_request,
) -> None:
    _, lease = await create_and_lease_initial(repository, persona, creation_request)
    async with repository._connection() as connection:
        await connection.execute("DROP TABLE episode_attempt_current")
        await connection.execute("DROP TABLE episode_attempt_cycles")
        await connection.execute(
            """
            CREATE TABLE episode_attempts_v17 (
                run_id TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
                episode_number INTEGER NOT NULL CHECK (episode_number >= 1),
                attempt_number INTEGER NOT NULL CHECK (
                    attempt_number >= 1 AND attempt_number <= 3
                ),
                recorded_at TEXT NOT NULL,
                PRIMARY KEY (run_id, episode_number, attempt_number)
            )
            """
        )
        await connection.execute(
            """
            INSERT INTO episode_attempts_v17(
                run_id, episode_number, attempt_number, recorded_at
            ) VALUES (?, 1, 1, ?)
            """,
            (str(lease.run_id), NOW.isoformat()),
        )
        await connection.execute("DROP TABLE episode_attempts")
        await connection.execute("ALTER TABLE episode_attempts_v17 RENAME TO episode_attempts")
        await connection.execute("DELETE FROM pengine_schema WHERE version = 19")
        await connection.execute("DELETE FROM pengine_schema WHERE version = 20")
        await connection.execute("DELETE FROM pengine_schema WHERE version = 21")
        await connection.execute("DELETE FROM pengine_schema WHERE version = 22")
        await connection.execute(
            "DELETE FROM pengine_schema WHERE version IN "
            "(23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33)"
        )
        await connection.commit()

    restarted = Repository(repository.database_path)
    await restarted.initialize()
    async with restarted._connection() as connection:
        attempt = await (
            await connection.execute(
                """
                SELECT attempt_cycle, attempt_number
                FROM episode_attempts
                WHERE run_id = ? AND episode_number = 1
                """,
                (str(lease.run_id),),
            )
        ).fetchone()
        cycle = await (
            await connection.execute(
                """
                SELECT attempt_cycle, from_episode, to_episode
                FROM episode_attempt_cycles
                WHERE run_id = ?
                """,
                (str(lease.run_id),),
            )
        ).fetchone()
        current = await (
            await connection.execute(
                """
                SELECT attempt_cycle
                FROM episode_attempt_current
                WHERE run_id = ? AND episode_number = 1
                """,
                (str(lease.run_id),),
            )
        ).fetchone()

    assert tuple(attempt) == (0, 1)
    assert tuple(cycle) == (0, 1, 1)
    assert tuple(current) == (0,)


async def test_run_progress_reports_outline_group_coverage(
    repository,
    persona,
    creation_request,
) -> None:
    accepted, lease = await create_and_lease_initial(
        repository,
        persona,
        creation_request,
    )
    await repository.record_stage_attempt(
        lease.run_id,
        InternalStage.GENERATING_EPISODE_OUTLINE,
        now=NOW + timedelta(seconds=1),
    )
    await repository.begin_outline_group(
        lease.run_id,
        group_id="g01",
        position=1,
        start_episode=1,
        end_episode=3,
        operation_id="operation-1",
    )
    await repository.begin_outline_group(
        lease.run_id,
        group_id="g02",
        position=2,
        start_episode=4,
        end_episode=5,
        operation_id="operation-2",
    )
    markdown = "## 第1集\n第一组大纲正文。"
    await repository.save_outline_group_body(
        lease.run_id,
        group_id="g01",
        operation_id="operation-1",
        outline_markdown=markdown,
        outline_markdown_sha256=hashlib.sha256(markdown.encode()).hexdigest(),
        body_call_id="call-body-1",
    )
    await repository.complete_outline_group(
        lease.run_id,
        group_id="g01",
        operation_id="operation-1",
        payload={"episodes": []},
        sidecar_call_id="call-sidecar-1",
    )

    creation = await repository.get_creation(
        accepted.creation_id,
        now=NOW + timedelta(seconds=30),
    )
    assert creation is not None
    progress = creation.initial.progress
    assert progress.current_stage == "generating_episode_outline"
    assert progress.outline_groups is not None
    assert progress.outline_groups.model_dump() == {
        "committed_groups": 1,
        "committed_through_episode": 3,
        "current_group": 2,
        "current_start_episode": 4,
        "current_end_episode": 5,
    }
