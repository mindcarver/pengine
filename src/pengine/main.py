import uvicorn
from fastapi import FastAPI

from pengine.api import create_app
from pengine.config import Settings, get_settings
from pengine.personas import PersonaCatalog
from pengine.repository import Repository
from pengine.worker import Worker, WorkerPool


def build_app(settings: Settings | None = None) -> FastAPI:
    resolved = settings or get_settings()
    repository = Repository(resolved.database_path)
    catalog = PersonaCatalog(resolved.persona_root, resolved.snapshot_root)
    worker = WorkerPool(
        [
            Worker(
                settings=resolved,
                repository=repository,
                catalog=catalog,
                worker_id=f"pengine-slot-{slot + 1}",
            )
            for slot in range(resolved.worker_concurrency)
        ]
    )
    return create_app(
        settings=resolved,
        repository=repository,
        catalog=catalog,
        worker=worker,
    )


app = build_app()


def run() -> None:
    settings = get_settings()
    uvicorn.run(
        app,
        host=settings.host,
        port=settings.port,
        log_level="info",
    )
