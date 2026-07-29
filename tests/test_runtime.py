import asyncio
from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient
from persona_factory import create_persona_package
from test_worker import DeterministicWorkflow

from pengine.api import create_app
from pengine.config import Settings
from pengine.personas import PersonaCatalog
from pengine.repository import Repository
from pengine.worker import Worker


async def _wait_for_state(
    client: AsyncClient,
    creation_id: str,
    field: str,
    expected: str,
) -> dict:
    deadline = asyncio.get_running_loop().time() + 10.0
    while asyncio.get_running_loop().time() < deadline:
        response = await client.get(f"/creations/{creation_id}")
        assert response.status_code == 200
        body = response.json()
        if body[field]["state"] == expected:
            return body
        await asyncio.sleep(0.01)
    raise AssertionError(f"{field} did not reach {expected}")


@pytest.mark.asyncio
async def test_three_interaction_runtime_flow(tmp_path: Path) -> None:
    persona_root = tmp_path / "personas"
    create_persona_package(persona_root / "active")
    settings = Settings(
        persona_root=persona_root,
        data_dir=tmp_path / "data",
        worker_poll_seconds=0.01,
    )
    repository = Repository(settings.database_path)
    catalog = PersonaCatalog(persona_root, settings.snapshot_root)
    worker = Worker(
        settings=settings,
        repository=repository,
        catalog=catalog,
        workflow=DeterministicWorkflow(),
        worker_id="runtime-test-worker",
    )
    app = create_app(
        settings=settings,
        repository=repository,
        catalog=catalog,
        worker=worker,
    )

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

        accepted = await client.post(
            "/creations",
            headers={"Idempotency-Key": "runtime-create"},
            json={
                "persona_id": "test-persona",
                "story": "一个人回乡面对旧事。",
                "requirements": "生成完整短剧。",
            },
        )
        assert accepted.status_code == 202
        creation_id = accepted.json()["creation_id"]
        initial = await _wait_for_state(
            client,
            creation_id,
            "initial",
            "succeeded",
        )
        assert initial["initial"]["result"]["content_package"]["episode_scripts"]
        assert initial["revision"]["state"] == "available"

        revision = await client.post(
            f"/creations/{creation_id}/revision",
            headers={"Idempotency-Key": "runtime-revision"},
            json={"feedback": "让主角付出更明确的代价。"},
        )
        assert revision.status_code == 202
        final = await _wait_for_state(
            client,
            creation_id,
            "revision",
            "succeeded",
        )
        assert final["initial"] == initial["initial"]
        assert final["revision"]["result"]["delivery_report"]["feedback_handling"]
