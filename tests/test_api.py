import shutil
import threading
from pathlib import Path
from uuid import uuid4

import pytest
from httpx import ASGITransport, AsyncClient
from persona_factory import create_persona_package

from pengine.api import create_app
from pengine.config import Settings


def _app(tmp_path: Path):
    persona_root = tmp_path / "personas"
    create_persona_package(persona_root / "active")
    settings = Settings(
        persona_root=persona_root,
        data_dir=tmp_path / "data",
    )
    return create_app(settings=settings)


@pytest.mark.asyncio
async def test_frontend_and_assets_are_served_without_expanding_business_openapi(
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
    assert "<!doctype html>" in page.text.lower()
    assert stylesheet.status_code == 200
    assert stylesheet.headers["content-type"].startswith("text/css")
    assert script.status_code == 200
    assert "javascript" in script.headers["content-type"]

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
        ("POST", "/creations/{creation_id}/revision"),
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
        assert body["initial"] == {"state": "queued"}
        assert body["revision"] == {
            "state": "unavailable",
            "feedback_locked": False,
            "reason": "initial_not_succeeded",
        }
        assert "result" not in body["initial"]


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
