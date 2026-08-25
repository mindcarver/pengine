import hashlib
import shutil
import threading
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import pytest
from httpx import ASGITransport, AsyncClient
from persona_factory import create_persona_package
from test_script_batch import create_leased_run, seed_active_design_and_batch

from pengine.api import _error_response, create_app
from pengine.config import Settings
from pengine.errors import DomainError
from pengine.repository import Repository
from pengine.schemas import (
    ContentPackage,
    Delivery,
    DeliveryReport,
    GateResult,
    InternalStage,
    RunFailure,
)


def _app(tmp_path: Path):
    persona_root = tmp_path / "personas"
    create_persona_package(persona_root / "active")
    settings = Settings(
        persona_root=persona_root,
        data_dir=tmp_path / "data",
    )
    return create_app(settings=settings)


@pytest.mark.asyncio
async def test_frontend_and_assets_are_served_with_run_control_openapi(
    tmp_path: Path,
) -> None:
    app = _app(tmp_path)
    async with (
        app.router.lifespan_context(app),
        AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
        ) as client,
    ):
        page = await client.get("/")
        stylesheet = await client.get("/static/styles.css")
        script = await client.get("/static/app.js")

    assert page.status_code == 200
    assert page.headers["content-type"].startswith("text/html")
    assert page.headers["cache-control"] == "no-store"
    assert "<!doctype html>" in page.text.lower()
    assert stylesheet.status_code == 200
    assert stylesheet.headers["content-type"].startswith("text/css")
    assert stylesheet.headers["cache-control"] == "no-cache"
    assert script.status_code == 200
    assert "javascript" in script.headers["content-type"]
    assert script.headers["cache-control"] == "no-cache"
    assert "重试本次修订" not in script.text
    assert 'revisionState === "available" && !state.pendingFeedback' in script.text

    operations = {
        (method.upper(), path)
        for path, methods in app.openapi()["paths"].items()
        for method in methods
        if method in {"get", "post", "put", "patch", "delete"}
    }
    assert operations == {
        ("GET", "/personas"),
        ("POST", "/creations"),
        ("GET", "/creations/{creation_id}"),
        ("GET", "/creations/{creation_id}/runs/{run_kind}/presentation"),
        ("POST", "/creations/{creation_id}/revision"),
        ("POST", "/creations/{creation_id}/runs/{run_kind}/continue"),
        ("POST", "/creations/{creation_id}/runs/{run_kind}/retry-final-review"),
        ("POST", "/creations/{creation_id}/runs/{run_kind}/authorize-repair"),
        ("POST", "/creations/{creation_id}/runs/{run_kind}/end"),
        ("POST", "/creations/{creation_id}/runs/{run_kind}/retry"),
    }

    retry = app.openapi()["paths"]["/creations/{creation_id}/runs/{run_kind}/retry-final-review"][
        "post"
    ]
    assert retry["summary"] == "Repair the rejected evidence scope and re-run final review"
    assert "never re-reviews unchanged drafts" in retry["description"]
    assert retry["responses"]["202"]["description"] == (
        "Bounded repair queued or an identical accepted command replayed"
    )
    assert retry["responses"]["409"]["content"]["application/json"]["schema"] == {
        "$ref": "#/components/schemas/CommandError"
    }


@pytest.mark.asyncio
async def test_snapshot_creation_runs_outside_event_loop_thread(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = _app(tmp_path)
    event_loop_thread = threading.get_ident()
    snapshot_threads: list[int] = []
    create_snapshot = app.state.catalog.create_snapshot

    def record_snapshot_thread(persona_id: str):
        snapshot_threads.append(threading.get_ident())
        return create_snapshot(persona_id)

    monkeypatch.setattr(app.state.catalog, "create_snapshot", record_snapshot_thread)
    async with (
        app.router.lifespan_context(app),
        AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
        ) as client,
    ):
        accepted = await client.post(
            "/creations",
            headers={"Idempotency-Key": "threaded-snapshot"},
            json={
                "persona_id": "test-persona",
                "story": "一个人回乡。",
                "requirements": "生成完整短剧。",
            },
        )

    assert accepted.status_code == 202
    assert snapshot_threads
    assert snapshot_threads[0] != event_loop_thread


@pytest.mark.asyncio
async def test_delivery_presentation_is_read_only_and_historical_safe(tmp_path: Path) -> None:
    app = _app(tmp_path)
    async with (
        app.router.lifespan_context(app),
        AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client,
    ):
        missing = await client.get(f"/creations/{uuid4()}/runs/initial/presentation")
        assert missing.status_code == 404
        assert missing.json()["code"] == "creation_not_found"

        accepted = await client.post(
            "/creations",
            headers={"Idempotency-Key": "presentation-create"},
            json={
                "persona_id": "test-persona",
                "story": "一个人回乡。",
                "requirements": "生成完整短剧。",
            },
        )
        creation_id = accepted.json()["creation_id"]
        unavailable = await client.get(f"/creations/{creation_id}/runs/initial/presentation")
        assert unavailable.status_code == 409
        assert unavailable.json()["code"] == "presentation_not_available"

        lease = await app.state.repository.lease_next_job("presentation-worker", 30)
        assert lease is not None
        creation = (await client.get(f"/creations/{creation_id}")).json()
        persona = creation["persona"]
        delivery = Delivery(
            content_package=ContentPackage(
                story_outline="故事大纲",
                character_biographies="人物小传",
                relationship_logic="人物关系",
                episode_outline="第一集详细大纲\n第二集详细大纲",
                episode_scripts="第 1 集剧本\n第 2 集剧本",
            ),
            delivery_report=DeliveryReport(
                persona_id=persona["persona_id"],
                persona_version=persona["version"],
                persona_snapshot_sha256=persona["snapshot_sha256"],
                selected_l0_variant="归返",
                selection_rationale="符合故事。",
                l0_gate=GateResult(passed=True, evidence="L0 通过"),
                l4_gate=GateResult(passed=True, evidence="L4 通过"),
                ownership_statement="由操作人员保留最终判断。",
                feedback_handling=[],
            ),
        )
        await app.state.repository.succeed_run(lease.run_id, delivery)
        async with app.state.repository._transaction() as connection:
            now = datetime.now(UTC).isoformat()
            for episode_number in (1, 2):
                plan = f"第 {episode_number} 集计划"
                script = f"第 {episode_number} 集剧本"
                await connection.execute(
                    "INSERT INTO episode_plans(run_id, episode_number, plan, plan_sha256) "
                    "VALUES (?, ?, ?, ?)",
                    (
                        str(lease.run_id),
                        episode_number,
                        plan,
                        hashlib.sha256(plan.encode()).hexdigest(),
                    ),
                )
                await connection.execute(
                    "INSERT INTO episode_drafts(run_id, episode_number, content, "
                    "content_sha256, completed_at) VALUES (?, ?, ?, ?, ?)",
                    (
                        str(lease.run_id),
                        episode_number,
                        script,
                        hashlib.sha256(script.encode()).hexdigest(),
                        now,
                    ),
                )
            await connection.execute(
                "UPDATE deliveries SET presentation_manifest_json = NULL, "
                "presentation_manifest_sha256 = NULL WHERE run_id = ?",
                (str(lease.run_id),),
            )

        async with app.state.repository._connection() as observer:
            before = (await (await observer.execute("PRAGMA data_version")).fetchone())[0]
            presented = await client.get(f"/creations/{creation_id}/runs/initial/presentation")
            after = (await (await observer.execute("PRAGMA data_version")).fetchone())[0]

        assert presented.status_code == 200
        assert presented.json()["status"] == "partial"
        assert presented.json()["story_outline"]["source_text"] == "故事大纲"
        assert len(presented.json()["episode_outline"]["episodes"]) == 2
        assert len(presented.json()["episode_scripts"]["episodes"]) == 2
        assert before == after
        assert await app.state.repository.get_run_model_calls(lease.run_id) == []


@pytest.mark.asyncio
async def test_persona_creation_and_query_contract(tmp_path: Path) -> None:
    app = _app(tmp_path)
    async with (
        app.router.lifespan_context(app),
        AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
        ) as client,
    ):
        personas = await client.get("/personas")
        assert personas.status_code == 200
        assert personas.json()["items"][0]["persona_id"] == "test-persona"

        request = {
            "persona_id": "test-persona",
            "story": "一个人回乡面对旧事。",
            "requirements": "生成完整短剧。",
        }
        accepted = await client.post(
            "/creations",
            headers={"Idempotency-Key": "create-1"},
            json=request,
        )
        assert accepted.status_code == 202
        creation_id = accepted.json()["creation_id"]

        replay = await client.post(
            "/creations",
            headers={"Idempotency-Key": "create-1"},
            json=request,
        )
        assert replay.status_code == 202
        assert replay.json()["creation_id"] == creation_id

        resource = await client.get(f"/creations/{creation_id}")
        assert resource.status_code == 200
        body = resource.json()
        assert body["initial"] == {
            "state": "queued",
            "progress": {
                "current_stage": "determining_direction",
                "completed_stages": [],
                "elapsed_seconds": 0,
                "recovery_state": "none",
                "recovery_reason": "none",
                "final_review": {"l0": "pending", "l4": "pending"},
                "episodes": None,
                "outline_groups": None,
                "model_calls": [],
                "can_continue": False,
                "can_end": False,
                "can_retry": False,
            },
            "drafts": {
                "artifacts": [],
                "episodes": [],
                "design": None,
                "review_status": {"l0": "pending", "l4": "pending"},
            },
        }
        assert body["revision"] == {
            "state": "unavailable",
            "feedback_locked": False,
            "reason": "initial_not_succeeded",
        }
        assert "result" not in body["initial"]


@pytest.mark.asyncio
async def test_paused_run_continue_and_end_commands_are_idempotent(tmp_path: Path) -> None:
    app = _app(tmp_path)
    async with (
        app.router.lifespan_context(app),
        AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
        ) as client,
    ):
        accepted = await client.post(
            "/creations",
            headers={"Idempotency-Key": "control-create"},
            json={
                "persona_id": "test-persona",
                "story": "一个人回乡面对旧事。",
                "requirements": "生成完整短剧。",
            },
        )
        creation_id = accepted.json()["creation_id"]
        repository = app.state.repository
        lease = await repository.lease_next_job("control-worker-1", 30)
        assert lease is not None
        await repository.approve_business_checkpoint(
            lease.run_id,
            InternalStage.SELECTING_L0_VARIANT,
            {
                "selected_l0_variant": "归返",
                "selection_rationale": "匹配故事母题。",
            },
        )
        await repository.record_stage_attempt(
            lease.run_id,
            InternalStage.GENERATING_STORY_OUTLINE,
        )
        assert (
            await repository.handle_run_timeout(
                lease.run_id,
                InternalStage.GENERATING_STORY_OUTLINE,
            )
            == "auto_resuming"
        )
        resumed = await repository.lease_next_job("control-worker-2", 30)
        assert resumed is not None
        await repository.record_stage_attempt(
            resumed.run_id,
            InternalStage.GENERATING_STORY_OUTLINE,
        )
        assert (
            await repository.handle_run_timeout(
                resumed.run_id,
                InternalStage.GENERATING_STORY_OUTLINE,
            )
            == "paused"
        )

        paused = await client.get(f"/creations/{creation_id}")
        assert paused.json()["initial"]["state"] == "paused"
        assert paused.json()["initial"]["progress"]["can_continue"] is True
        assert "result" not in paused.json()["initial"]
        assert paused.json()["initial"]["drafts"] == {
            "artifacts": [
                {
                    "stage": "determining_direction",
                    "selected_l0_variant": "归返",
                    "selection_rationale": "匹配故事母题。",
                }
            ],
            "episodes": [],
            "design": None,
            "review_status": {"l0": "pending", "l4": "pending"},
        }

        first_continue = await client.post(
            f"/creations/{creation_id}/runs/initial/continue",
            headers={"Idempotency-Key": "control-continue"},
        )
        replay_continue = await client.post(
            f"/creations/{creation_id}/runs/initial/continue",
            headers={"Idempotency-Key": "control-continue"},
        )
        assert first_continue.status_code == 202
        assert replay_continue.json() == first_continue.json()
        assert first_continue.json()["run_state"] == "queued"
        continued = await client.get(f"/creations/{creation_id}")
        assert continued.json()["initial"]["state"] == "queued"
        assert continued.json()["initial"]["drafts"] == paused.json()["initial"]["drafts"]

        resumed_again = await repository.lease_next_job("control-worker-3", 30)
        assert resumed_again is not None
        await repository.record_stage_attempt(
            resumed_again.run_id,
            InternalStage.GENERATING_STORY_OUTLINE,
        )
        assert (
            await repository.handle_run_timeout(
                resumed_again.run_id,
                InternalStage.GENERATING_STORY_OUTLINE,
            )
            == "failed"
        )
        exhausted = await client.get(f"/creations/{creation_id}")
        assert exhausted.json()["initial"]["state"] == "failed"
        assert exhausted.json()["initial"]["failure"]["code"] == "attempts_exhausted"
        assert exhausted.json()["initial"]["progress"]["can_continue"] is False
        assert exhausted.json()["initial"]["progress"]["can_end"] is False
        cannot_continue = await client.post(
            f"/creations/{creation_id}/runs/initial/continue",
            headers={"Idempotency-Key": "control-continue-ended"},
        )
        assert cannot_continue.status_code == 409
        assert cannot_continue.json()["code"] == "run_not_controllable"


@pytest.mark.asyncio
async def test_failed_relay_run_retry_command_is_idempotent(tmp_path: Path) -> None:
    app = _app(tmp_path)
    async with (
        app.router.lifespan_context(app),
        AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
        ) as client,
    ):
        accepted = await client.post(
            "/creations",
            headers={"Idempotency-Key": "retry-create"},
            json={
                "persona_id": "test-persona",
                "story": "一个人回乡面对旧事。",
                "requirements": "生成完整短剧。",
            },
        )
        creation_id = accepted.json()["creation_id"]
        repository = app.state.repository
        lease = await repository.lease_next_job("retry-worker-1", 30)
        assert lease is not None
        await repository.record_stage_attempt(
            lease.run_id,
            InternalStage.GENERATING_STORY_OUTLINE,
        )
        await repository.fail_run(
            lease.run_id,
            RunFailure(
                code="relay_unavailable",
                message="The model relay request failed (HTTP 402).",
                failed_stage=InternalStage.GENERATING_STORY_OUTLINE,
                attempt_count=1,
            ),
        )

        failed = await client.get(f"/creations/{creation_id}")
        assert failed.json()["initial"]["state"] == "failed"
        assert failed.json()["initial"]["progress"]["can_retry"] is True

        first = await client.post(
            f"/creations/{creation_id}/runs/initial/retry",
            headers={"Idempotency-Key": "retry-relay"},
        )
        replay = await client.post(
            f"/creations/{creation_id}/runs/initial/retry",
            headers={"Idempotency-Key": "retry-relay"},
        )
        assert first.status_code == 202
        assert replay.json() == first.json()
        assert first.json()["run_state"] == "queued"

        revived = await client.get(f"/creations/{creation_id}")
        assert revived.json()["initial"]["state"] == "queued"
        assert revived.json()["initial"]["progress"]["can_retry"] is False

        not_failed = await client.post(
            f"/creations/{creation_id}/runs/initial/retry",
            headers={"Idempotency-Key": "retry-relay-queued"},
        )
        assert not_failed.status_code == 409
        assert not_failed.json()["code"] == "run_not_controllable"


@pytest.mark.asyncio
async def test_paused_run_end_command_is_idempotent(tmp_path: Path) -> None:
    app = _app(tmp_path)
    async with (
        app.router.lifespan_context(app),
        AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
        ) as client,
    ):
        accepted = await client.post(
            "/creations",
            headers={"Idempotency-Key": "end-control-create"},
            json={
                "persona_id": "test-persona",
                "story": "一个人回乡面对旧事。",
                "requirements": "生成完整短剧。",
            },
        )
        creation_id = accepted.json()["creation_id"]
        repository = app.state.repository
        lease = await repository.lease_next_job("end-worker-1", 30)
        assert lease is not None
        await repository.record_stage_attempt(lease.run_id, InternalStage.GENERATING_STORY_OUTLINE)
        assert (
            await repository.handle_run_timeout(
                lease.run_id,
                InternalStage.GENERATING_STORY_OUTLINE,
            )
            == "auto_resuming"
        )
        resumed = await repository.lease_next_job("end-worker-2", 30)
        assert resumed is not None
        await repository.record_stage_attempt(
            resumed.run_id,
            InternalStage.GENERATING_STORY_OUTLINE,
        )
        assert (
            await repository.handle_run_timeout(
                resumed.run_id,
                InternalStage.GENERATING_STORY_OUTLINE,
            )
            == "paused"
        )

        first_end = await client.post(
            f"/creations/{creation_id}/runs/initial/end",
            headers={"Idempotency-Key": "control-end"},
        )
        replay_end = await client.post(
            f"/creations/{creation_id}/runs/initial/end",
            headers={"Idempotency-Key": "control-end"},
        )
        assert first_end.status_code == 202
        assert replay_end.json() == first_end.json()
        assert first_end.json()["run_state"] == "ended"
        ended = await client.get(f"/creations/{creation_id}")
        assert ended.json()["initial"]["state"] == "ended"


@pytest.mark.asyncio
async def test_quality_rejected_final_review_can_retry_the_same_run(tmp_path: Path) -> None:
    app = _app(tmp_path)
    async with (
        app.router.lifespan_context(app),
        AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
        ) as client,
    ):
        accepted = await client.post(
            "/creations",
            headers={"Idempotency-Key": "quality-review-create"},
            json={
                "persona_id": "test-persona",
                "story": "一个人回乡面对旧事。",
                "requirements": "生成完整短剧。",
            },
        )
        creation_id = accepted.json()["creation_id"]
        repository = app.state.repository
        lease = await repository.lease_next_job("quality-review-worker", 30)
        assert lease is not None

        while_running = await client.post(
            f"/creations/{creation_id}/runs/initial/retry-final-review",
            headers={"Idempotency-Key": "quality-review-too-early"},
        )
        assert while_running.status_code == 409
        assert while_running.json()["code"] == "run_not_controllable"

        await repository.record_stage_attempt(lease.run_id, InternalStage.ACCEPTING_L0)
        await repository.reject_quality_gate(
            lease.run_id,
            stage=InternalStage.ACCEPTING_L0,
            evidence="L0 创作内核与人物选择不一致。",
        )
        rejected = await client.get(f"/creations/{creation_id}")
        assert rejected.json()["initial"]["state"] == "quality_rejected"
        assert rejected.json()["initial"]["quality_rejection"] == {
            "code": "quality_gate_rejected",
            "stage": "accepting_l0",
            "evidence": "L0 创作内核与人物选择不一致。",
            "attempt_count": 1,
            "can_retry": True,
            "repair_plan": None,
            "repair_state": "available",
        }

        first_retry = await client.post(
            f"/creations/{creation_id}/runs/initial/retry-final-review",
            headers={"Idempotency-Key": "quality-review-retry"},
        )
        replay_retry = await client.post(
            f"/creations/{creation_id}/runs/initial/retry-final-review",
            headers={"Idempotency-Key": "quality-review-retry"},
        )
        assert first_retry.status_code == 202
        assert replay_retry.json() == first_retry.json()
        assert first_retry.json()["run_state"] == "queued"
        queued = await client.get(f"/creations/{creation_id}")
        assert queued.json()["initial"]["state"] == "queued"
        resumed = await repository.lease_next_job("quality-review-retry-worker", 30)
        assert resumed is not None
        assert resumed.run_id == lease.run_id


@pytest.mark.asyncio
async def test_episode_progress_and_committed_drafts_remain_readable_after_end(
    tmp_path: Path,
) -> None:
    app = _app(tmp_path)
    async with (
        app.router.lifespan_context(app),
        AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
        ) as client,
    ):
        accepted = await client.post(
            "/creations",
            headers={"Idempotency-Key": "episode-progress-create"},
            json={
                "persona_id": "test-persona",
                "story": "一个人回乡面对旧事。",
                "requirements": "生成完整短剧。",
            },
        )
        creation_id = accepted.json()["creation_id"]
        repository = app.state.repository
        lease = await repository.lease_next_job("episode-progress-worker-1", 30)
        assert lease is not None
        await repository.approve_business_checkpoint(
            lease.run_id,
            InternalStage.GENERATING_EPISODE_OUTLINE,
            {
                "content": "两集大纲",
                "episode_count": 2,
                "episodes": [
                    {"episode_number": 1, "plan": "第一集计划"},
                    {"episode_number": 2, "plan": "第二集计划"},
                ],
            },
        )
        await repository.record_episode_attempt(lease.run_id, 1)
        await repository.commit_episode_draft(lease.run_id, 1, "第一集剧本")
        await repository.record_episode_attempt(lease.run_id, 2)
        assert await repository.handle_episode_timeout(lease.run_id, 2) == "auto_resuming"

        recovering = await client.get(f"/creations/{creation_id}")
        initial = recovering.json()["initial"]
        assert initial["state"] == "auto_resuming"
        assert initial["progress"]["episodes"] == {
            "total": 2,
            "completed": 1,
            "current": 2,
        }
        assert initial["drafts"]["episodes"][0]["episode_number"] == 1
        assert initial["drafts"]["episodes"][0]["content"] == "第一集剧本"
        assert len(initial["drafts"]["episodes"][0]["content_sha256"]) == 64

        resumed = await repository.lease_next_job("episode-progress-worker-2", 30)
        assert resumed is not None
        await repository.record_episode_attempt(resumed.run_id, 2)
        assert await repository.handle_episode_timeout(resumed.run_id, 2) == "paused"
        paused = await client.get(f"/creations/{creation_id}")
        assert paused.json()["initial"]["pause"]["episode_number"] == 2
        assert paused.json()["initial"]["drafts"] == initial["drafts"]

        ended = await client.post(
            f"/creations/{creation_id}/runs/initial/end",
            headers={"Idempotency-Key": "end-episode-progress"},
        )
        assert ended.status_code == 202
        terminal = await client.get(f"/creations/{creation_id}")
        assert terminal.json()["initial"]["state"] == "ended"
        assert terminal.json()["initial"]["drafts"] == initial["drafts"]


@pytest.mark.asyncio
async def test_creation_replay_survives_restart_without_persona_source(
    tmp_path: Path,
) -> None:
    first_app = _app(tmp_path)
    request = {
        "persona_id": "test-persona",
        "story": "一个人回乡面对旧事。",
        "requirements": "生成完整短剧。",
    }
    async with (
        first_app.router.lifespan_context(first_app),
        AsyncClient(
            transport=ASGITransport(app=first_app),
            base_url="http://test",
        ) as client,
    ):
        accepted = await client.post(
            "/creations",
            headers={"Idempotency-Key": "restart-replay"},
            json=request,
        )
        assert accepted.status_code == 202

    shutil.rmtree(tmp_path / "personas" / "active")
    restarted_app = create_app(
        settings=Settings(
            persona_root=tmp_path / "personas",
            data_dir=tmp_path / "data",
        )
    )
    async with (
        restarted_app.router.lifespan_context(restarted_app),
        AsyncClient(
            transport=ASGITransport(app=restarted_app),
            base_url="http://test",
        ) as client,
    ):
        replay = await client.post(
            "/creations",
            headers={"Idempotency-Key": "restart-replay"},
            json=request,
        )
        assert replay.status_code == 202
        assert replay.json() == accepted.json()

        conflict = await client.post(
            "/creations",
            headers={"Idempotency-Key": "restart-replay"},
            json={**request, "story": "不同故事"},
        )
        assert conflict.status_code == 409
        assert conflict.json()["code"] == "idempotency_conflict"


@pytest.mark.asyncio
async def test_api_returns_stable_command_errors(tmp_path: Path) -> None:
    app = _app(tmp_path)
    async with (
        app.router.lifespan_context(app),
        AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
        ) as client,
    ):
        missing_header = await client.post(
            "/creations",
            json={
                "persona_id": "test-persona",
                "story": "故事",
                "requirements": "要求",
            },
        )
        assert missing_header.status_code == 422
        assert missing_header.json() == {
            "code": "invalid_request",
            "message": "One or more request fields are invalid.",
        }

        unknown = await client.get(f"/creations/{uuid4()}")
        assert unknown.status_code == 404
        assert unknown.json()["code"] == "creation_not_found"

        first = await client.post(
            "/creations",
            headers={"Idempotency-Key": "conflict-key"},
            json={
                "persona_id": "test-persona",
                "story": "原故事",
                "requirements": "要求",
            },
        )
        assert first.status_code == 202

        conflict = await client.post(
            "/creations",
            headers={"Idempotency-Key": "conflict-key"},
            json={
                "persona_id": "test-persona",
                "story": "不同故事",
                "requirements": "要求",
            },
        )
        assert conflict.status_code == 409
        assert conflict.json()["code"] == "idempotency_conflict"

        revision = await client.post(
            f"/creations/{first.json()['creation_id']}/revision",
            headers={"Idempotency-Key": "revision-too-early"},
            json={"feedback": "加强代价。"},
        )
        assert revision.status_code == 409
        assert revision.json()["code"] == "revision_not_allowed"

        blank_feedback = await client.post(
            f"/creations/{first.json()['creation_id']}/revision",
            headers={"Idempotency-Key": "revision-blank"},
            json={"feedback": " \t\n"},
        )
        assert blank_feedback.status_code == 422
        assert blank_feedback.json()["code"] == "invalid_request"


@pytest.mark.asyncio
async def test_authorize_repair_design_rebuild_returns_202_never_500(
    tmp_path: Path,
) -> None:
    # Delivery #57 INT-A8: authorizing a design-rebuild cycle after the one
    # automatic rebuild is consumed must succeed with a bounded 202 response, not
    # an HTTP 500 from a series_bible_rebuild_exhausted DomainError escaping the
    # CommandError contract. The paused resource still carries the evidence,
    # affected range, and token estimate.
    persona_root = tmp_path / "personas"
    create_persona_package(persona_root / "active")
    settings = Settings(persona_root=persona_root, data_dir=tmp_path / "data")
    repository = Repository(settings.database_path)
    await repository.initialize()

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
        review_id="review-a8",
    )
    paused_resource = await repository.get_creation(accepted.creation_id)
    assert paused_resource is not None
    assert paused_resource.initial.state == "paused"
    assert paused_resource.initial.pause.code == "repair_authorization"
    assert paused_resource.initial.authorization is not None
    assert paused_resource.initial.authorization.kind == "design_rebuild"
    assert paused_resource.initial.authorization.estimated_tokens == 9_000
    assert paused_resource.initial.authorization.evidence == "设计缺陷证据"

    app = create_app(settings=settings, repository=repository)
    async with (
        app.router.lifespan_context(app),
        AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
        ) as client,
    ):
        authorized = await client.post(
            f"/creations/{accepted.creation_id}/runs/initial/authorize-repair",
            headers={"Idempotency-Key": "a8-authorize"},
        )

    assert authorized.status_code == 202
    body = authorized.json()
    assert body["run_state"] == "queued"
    assert body["resource_url"] == f"/creations/{accepted.creation_id}"

    resource = await repository.get_creation(accepted.creation_id)
    assert resource is not None
    assert resource.initial.state == "queued"


def test_error_response_maps_unknown_domain_error_code_without_500() -> None:
    # Delivery #57 INT-A8: a DomainError that is not part of the CommandError
    # contract must still surface as a bounded displayable error, never a 500.
    response = _error_response(DomainError("series_bible_rebuild_exhausted", "消息", 409))
    assert response.status_code == 409
    assert response.body.decode() == '{"code":"series_bible_rebuild_exhausted","message":"消息"}'

    response = _error_response(DomainError("some_future_code", "消息", 409))
    assert response.status_code == 409
    assert response.body.decode() == '{"code":"service_unavailable","message":"消息"}'
