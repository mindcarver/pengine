import copy

import uvicorn
from fastapi import FastAPI
from uvicorn.config import LOGGING_CONFIG

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


def build_log_config(level: str) -> dict:
    """Uvicorn's default config only routes its own loggers; ``pengine.*`` INFO lines
    (run started/failed, relay recovery, repair loops) were silently dropped."""
    config = copy.deepcopy(LOGGING_CONFIG)
    config["formatters"]["default"]["fmt"] = "%(asctime)s %(levelprefix)s %(message)s"
    config["formatters"]["access"]["fmt"] = (
        '%(asctime)s %(levelprefix)s %(client_addr)s - "%(request_line)s" %(status_code)s'
    )
    config["loggers"]["pengine"] = {"handlers": ["default"], "level": level, "propagate": False}
    return config


def run() -> None:
    settings = get_settings()
    uvicorn.run(
        app,
        host=settings.host,
        port=settings.port,
        log_level="info",
        log_config=build_log_config(settings.log_level),
    )
