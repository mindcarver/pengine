import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Protocol
from uuid import UUID

from fastapi import FastAPI, Header
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from pengine.config import Settings, get_settings
from pengine.errors import DomainError
from pengine.personas import PersonaCatalog, PersonaPackageError
from pengine.repository import Repository
from pengine.schemas import (
    CommandError,
    CreateCreationRequest,
    CreationAccepted,
    CreationResource,
    PersonaList,
    RevisionAccepted,
    RevisionRequest,
)

IdempotencyKey = Annotated[
    str,
    Header(alias="Idempotency-Key", min_length=1, max_length=128),
]
_WEB_ROOT = Path(__file__).with_name("web")


class WorkerControl(Protocol):
    async def start(self) -> None: ...

    async def stop(self) -> None: ...


def _error_response(error: DomainError) -> JSONResponse:
    body = CommandError(code=error.code, message=error.message)
    return JSONResponse(status_code=error.status_code, content=body.model_dump(mode="json"))


def create_app(
    *,
    settings: Settings | None = None,
    repository: Repository | None = None,
    catalog: PersonaCatalog | None = None,
    worker: WorkerControl | None = None,
) -> FastAPI:
    resolved_settings = settings or get_settings()
    resolved_repository = repository or Repository(resolved_settings.database_path)
    resolved_catalog = catalog or PersonaCatalog(
        resolved_settings.persona_root,
        resolved_settings.snapshot_root,
    )

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        await resolved_repository.initialize()
        resolved_catalog.discover()
        if worker is not None:
            await worker.start()
        try:
            yield
        finally:
            if worker is not None:
                await worker.stop()

    app = FastAPI(
        title="Pengine Local API",
        version="1.0.0",
        lifespan=lifespan,
    )
    app.state.settings = resolved_settings
    app.state.repository = resolved_repository
    app.state.catalog = resolved_catalog
    app.state.worker = worker
    app.mount("/static", StaticFiles(directory=_WEB_ROOT), name="static")

    @app.get("/", include_in_schema=False)
    async def frontend() -> FileResponse:
        return FileResponse(_WEB_ROOT / "index.html", media_type="text/html")

    @app.exception_handler(DomainError)
    async def handle_domain_error(_, exc: DomainError) -> JSONResponse:
        return _error_response(exc)

    @app.exception_handler(PersonaPackageError)
    async def handle_persona_error(_, exc: PersonaPackageError) -> JSONResponse:
        if exc.code == "persona_not_found":
            return _error_response(DomainError("persona_not_found", exc.message, 404))
        return _error_response(
            DomainError(
                "persona_package_unavailable",
                "The selected persona package is unavailable.",
                503,
            )
        )

    @app.exception_handler(RequestValidationError)
    async def handle_validation_error(_, __: RequestValidationError) -> JSONResponse:
        return _error_response(
            DomainError(
                "invalid_request",
                "One or more request fields are invalid.",
                422,
            )
        )

    @app.get("/personas", operation_id="listPersonas", response_model=PersonaList)
    async def list_personas() -> PersonaList:
        return PersonaList(items=resolved_catalog.discover())

    @app.post(
        "/creations",
        operation_id="createCreation",
        status_code=202,
        response_model=CreationAccepted,
    )
    async def create_creation(
        request: CreateCreationRequest,
        idempotency_key: IdempotencyKey,
    ) -> CreationAccepted:
        replay = await resolved_repository.replay_create_creation(
            idempotency_key,
            request,
        )
        if replay is not None:
            return replay
        snapshot = await asyncio.to_thread(
            resolved_catalog.create_snapshot,
            request.persona_id,
        )
        return await resolved_repository.create_creation(
            idempotency_key=idempotency_key,
            request=request,
            persona_snapshot=snapshot.summary,
        )

    @app.get(
        "/creations/{creation_id}",
        operation_id="getCreation",
        response_model=CreationResource,
    )
    async def get_creation(creation_id: UUID) -> CreationResource:
        resource = await resolved_repository.get_creation(creation_id)
        if resource is None:
            raise DomainError("creation_not_found", "Creation not found.", 404)
        return resource

    @app.post(
        "/creations/{creation_id}/revision",
        operation_id="createOrRetryRevision",
        status_code=202,
        response_model=RevisionAccepted,
    )
    async def create_or_retry_revision(
        creation_id: UUID,
        request: RevisionRequest,
        idempotency_key: IdempotencyKey,
    ) -> RevisionAccepted:
        return await resolved_repository.create_or_retry_revision(
            creation_id=creation_id,
            idempotency_key=idempotency_key,
            request=request,
        )

    return app
