import asyncio
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from persona_factory import create_persona_package
from test_worker import DeterministicWorkflow

from pengine.config import Settings
from pengine.model_calls import ModelCallState
from pengine.personas import PersonaCatalog
from pengine.relay import RelayError
from pengine.repository import Repository
from pengine.schemas import CreateCreationRequest
from pengine.worker import Worker, WorkerPool


class GatedWorkflow(DeterministicWorkflow):
    def __init__(self, gate: asyncio.Event) -> None:
        super().__init__(episode_count=1)
        self.gate = gate
        self.entered = asyncio.Event()

    async def execute(self, **kwargs: Any):
        self.entered.set()
        await self.gate.wait()
        return await super().execute(**kwargs)


class NotifyingWorkflow(DeterministicWorkflow):
    def __init__(self, entered: asyncio.Event) -> None:
        super().__init__(episode_count=1)
        self.entered = entered

    async def execute(self, **kwargs: Any):
        self.entered.set()
        return await super().execute(**kwargs)


class CoordinatedRelayFailureWorkflow:
    def __init__(self, healthy_entered: asyncio.Event) -> None:
        self.healthy_entered = healthy_entered

    async def execute(self, **kwargs: Any):
        del kwargs
        await self.healthy_entered.wait()
        raise RelayError(
            code="relay_incompatible",
            safe_message="isolated relay failure",
        )


async def wait_for_job_counts(
    repository: Repository,
    *,
    leased: int,
    queued: int,
    timeout: float = 15.0,
) -> None:
    deadline = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < deadline:
        async with repository._connection() as connection:
            rows = await (
                await connection.execute("SELECT state, COUNT(*) AS count FROM jobs GROUP BY state")
            ).fetchall()
        counts = {row["state"]: int(row["count"]) for row in rows}
        if counts.get("leased", 0) == leased and counts.get("queued", 0) == queued:
            return
        await asyncio.sleep(0.02)
    raise AssertionError(f"job counts did not become leased={leased}, queued={queued}")


async def wait_for_terminal_runs(
    repository: Repository,
    expected: int,
    timeout: float = 20.0,
) -> list[str]:
    deadline = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < deadline:
        async with repository._connection() as connection:
            rows = await (
                await connection.execute(
                    "SELECT state FROM runs WHERE state IN ('succeeded', 'failed')"
                )
            ).fetchall()
        if len(rows) == expected:
            return [str(row["state"]) for row in rows]
        await asyncio.sleep(0.02)
    raise AssertionError(f"{expected} runs did not reach a terminal state")


async def wait_for_workflows_to_enter(
    workflows: list[GatedWorkflow],
    timeout: float = 10.0,
) -> None:
    await asyncio.wait_for(
        asyncio.gather(*(workflow.entered.wait() for workflow in workflows)),
        timeout=timeout,
    )


async def create_owned_job(
    repository: Repository,
    snapshot,
    *,
    username: str,
    index: int,
    owner_id=None,
) -> tuple[object, object]:
    now = datetime.now(UTC) - timedelta(minutes=1) + timedelta(milliseconds=index)
    if owner_id is None:
        user = await repository.register_user(
            username=username,
            password_hash=f"hash-{username}",
            session_token_sha256=f"token-{username}",
            session_expires_at=now + timedelta(days=1),
            now=now,
        )
        owner_id = user.user_id
    accepted = await repository.create_creation(
        f"create-{username}-{index}",
        CreateCreationRequest(
            persona_id="test-persona",
            story=f"用户 {username} 的故事。",
            requirements="生成一集短剧。",
        ),
        snapshot,
        owner_id=owner_id,
        now=now,
    )
    return owner_id, accepted


async def services(tmp_path: Path):
    persona_root = tmp_path / "personas"
    create_persona_package(persona_root / "active")
    settings = Settings(
        persona_root=persona_root,
        data_dir=tmp_path / "data",
        worker_poll_seconds=0.01,
    )
    catalog = PersonaCatalog(persona_root, settings.snapshot_root)
    repository = Repository(settings.database_path)
    await repository.initialize()
    snapshot = catalog.create_snapshot("test-persona")
    return settings, catalog, repository, snapshot


@pytest.mark.asyncio
async def test_five_isolated_workers_run_five_accounts_and_leave_sixth_queued(
    tmp_path: Path,
) -> None:
    settings, catalog, repository, snapshot = await services(tmp_path)
    for index in range(6):
        await create_owned_job(
            repository,
            snapshot.summary,
            username=f"user-{index}",
            index=index,
        )
    gate = asyncio.Event()
    workflows = [GatedWorkflow(gate) for _ in range(5)]
    workers = [
        Worker(
            settings=settings,
            repository=repository,
            catalog=catalog,
            workflow=workflow,
            worker_id=f"slot-{index + 1}",
        )
        for index, workflow in enumerate(workflows)
    ]
    for worker in workers:
        worker._model_call_state = ModelCallState()
    pool = WorkerPool(workers)

    await pool.start()
    try:
        await wait_for_job_counts(repository, leased=5, queued=1)
        await wait_for_workflows_to_enter(workflows)
        assert len({id(worker._model_call_state) for worker in workers}) == 5
        async with repository._connection() as connection:
            rows = await (
                await connection.execute("SELECT lease_owner FROM jobs WHERE state = 'leased'")
            ).fetchall()
        assert {row["lease_owner"] for row in rows} == {f"slot-{index}" for index in range(1, 6)}

        gate.set()
        assert await wait_for_terminal_runs(repository, 6) == ["succeeded"] * 6
    finally:
        gate.set()
        await pool.stop()


@pytest.mark.asyncio
async def test_pool_builds_a_distinct_checkpoint_saver_for_each_slot(tmp_path: Path) -> None:
    settings, catalog, repository, _ = await services(tmp_path)
    workers = [
        Worker(
            settings=settings,
            repository=repository,
            catalog=catalog,
            worker_id=f"saver-slot-{index}",
        )
        for index in range(5)
    ]
    pool = WorkerPool(workers)

    await pool.start()
    try:
        assert all(worker._saver is not None for worker in workers)
        assert len({id(worker._saver) for worker in workers}) == 5
    finally:
        await pool.stop()


@pytest.mark.asyncio
async def test_pool_runs_only_one_job_per_account_while_other_account_progresses(
    tmp_path: Path,
) -> None:
    settings, catalog, repository, snapshot = await services(tmp_path)
    owner_id, _ = await create_owned_job(
        repository,
        snapshot.summary,
        username="shared",
        index=0,
    )
    await create_owned_job(
        repository,
        snapshot.summary,
        username="shared",
        index=1,
        owner_id=owner_id,
    )
    await create_owned_job(
        repository,
        snapshot.summary,
        username="other",
        index=2,
    )
    gate = asyncio.Event()
    pool = WorkerPool(
        [
            Worker(
                settings=settings,
                repository=repository,
                catalog=catalog,
                workflow=GatedWorkflow(gate),
                worker_id=f"fair-slot-{index}",
            )
            for index in range(2)
        ]
    )

    await pool.start()
    try:
        await wait_for_job_counts(repository, leased=2, queued=1)
        async with repository._connection() as connection:
            rows = await (
                await connection.execute(
                    """
                    SELECT creations.owner_id
                    FROM jobs
                    JOIN runs ON runs.id = jobs.run_id
                    JOIN creations ON creations.id = runs.creation_id
                    WHERE jobs.state = 'leased'
                    """
                )
            ).fetchall()
        assert len({row["owner_id"] for row in rows}) == 2
        gate.set()
        await wait_for_terminal_runs(repository, 3)
    finally:
        gate.set()
        await pool.stop()


@pytest.mark.asyncio
async def test_one_worker_failure_does_not_cancel_another_account(tmp_path: Path) -> None:
    settings, catalog, repository, snapshot = await services(tmp_path)
    for index in range(2):
        await create_owned_job(
            repository,
            snapshot.summary,
            username=f"failure-user-{index}",
            index=index,
        )
    healthy_entered = asyncio.Event()
    pool = WorkerPool(
        [
            Worker(
                settings=settings,
                repository=repository,
                catalog=catalog,
                workflow=CoordinatedRelayFailureWorkflow(healthy_entered),
                worker_id="failing-slot",
            ),
            Worker(
                settings=settings,
                repository=repository,
                catalog=catalog,
                workflow=NotifyingWorkflow(healthy_entered),
                worker_id="healthy-slot",
            ),
        ]
    )

    await pool.start()
    try:
        assert sorted(await wait_for_terminal_runs(repository, 2)) == [
            "failed",
            "succeeded",
        ]
    finally:
        await pool.stop()
