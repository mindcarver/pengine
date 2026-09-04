import asyncio
import logging
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Literal, Protocol, get_args
from uuid import UUID

from fastapi import Cookie, Depends, FastAPI, Header, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from pengine.auth import (
    SESSION_COOKIE,
    SESSION_TTL,
    hash_password,
    new_session,
    session_token_sha256,
    verify_password,
)
from pengine.config import Settings, get_settings
from pengine.errors import DomainError
from pengine.personas import PersonaCatalog, PersonaPackageError
from pengine.repository import Repository
from pengine.schemas import (
    CommandError,
    CreateCreationRequest,
    CreationAccepted,
    CreationList,
    CreationResource,
    CurrentUser,
    DeliveryPresentation,
    LoginRequest,
    PersonaList,
    RegisterRequest,
    RevisionAccepted,
    RevisionRequest,
    RunControlAccepted,
)

IdempotencyKey = Annotated[
    str,
    Header(alias="Idempotency-Key", min_length=1, max_length=128),
]
_WEB_ROOT = Path(__file__).with_name("web")
logger = logging.getLogger(__name__)

# The bounded, browser-displayable CommandError codes that the API may return.
# Any DomainError raised by a route must map onto this contract so a defect can
# never surface as an HTTP 500 (INT-A8).
_COMMAND_ERROR_CODES: frozenset[str] = frozenset(
    get_args(CommandError.model_fields["code"].annotation)
)


class WorkerControl(Protocol):
    async def start(self) -> None: ...

    async def stop(self) -> None: ...


def _error_response(error: DomainError) -> JSONResponse:
    code = error.code if error.code in _COMMAND_ERROR_CODES else "service_unavailable"
    body = CommandError(code=code, message=error.message)
    return JSONResponse(status_code=error.status_code, content=body.model_dump(mode="json"))


def create_app(
    *,
    settings: Settings | None = None,
    repository: Repository | None = None,
    catalog: PersonaCatalog | None = None,
    worker: WorkerControl | None = None,
    authentication_enabled: bool = True,
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

    @app.middleware("http")
    async def require_web_asset_revalidation(request: Request, call_next) -> Response:
        response = await call_next(request)
        if request.url.path == "/":
            response.headers["Cache-Control"] = "no-store"
        elif request.url.path.startswith("/static/"):
            response.headers["Cache-Control"] = "no-cache"
        return response

    @app.middleware("http")
    async def log_commands(request: Request, call_next) -> Response:
        # Uvicorn's access log lacks timing and is drowned out by GET polling;
        # only the state-changing commands (create/continue/end/retry) matter for
        # reconstructing what an operator did to a run.
        if request.method not in {"POST", "PUT", "PATCH", "DELETE"}:
            return await call_next(request)
        started = time.monotonic()
        response = await call_next(request)
        logger.info(
            "command %s %s status=%s duration=%.2fs",
            request.method,
            request.url.path,
            response.status_code,
            time.monotonic() - started,
        )
        return response

    @app.get("/", include_in_schema=False)
    async def frontend() -> FileResponse:
        return FileResponse(
            _WEB_ROOT / "index.html",
            media_type="text/html",
            headers={"Cache-Control": "no-store"},
        )

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

    async def resolve_current_user(
        session_token: Annotated[str | None, Cookie(alias=SESSION_COOKIE)] = None,
    ) -> CurrentUser:
        if not authentication_enabled:
            return CurrentUser(
                user_id=UUID("00000000-0000-0000-0000-000000000000"),
                username="test-user",
            )
        if not session_token:
            raise DomainError("authentication_required", "请先登录。", 401)
        user = await resolved_repository.resolve_session(session_token_sha256(session_token))
        if user is None:
            raise DomainError("authentication_required", "登录已失效，请重新登录。", 401)
        return user

    AuthenticatedUser = Annotated[CurrentUser, Depends(resolve_current_user)]

    def scoped_key(user: CurrentUser, idempotency_key: str) -> str:
        if not authentication_enabled:
            return idempotency_key
        return f"{user.user_id}:{idempotency_key}"

    async def require_owner(creation_id: UUID, user: CurrentUser) -> None:
        if authentication_enabled:
            await resolved_repository.require_creation_owner(creation_id, user.user_id)

    def set_session_cookie(response: Response, token: str) -> None:
        response.set_cookie(
            SESSION_COOKIE,
            token,
            max_age=int(SESSION_TTL.total_seconds()),
            httponly=True,
            secure=True,
            samesite="lax",
            path="/",
        )

    @app.post(
        "/auth/register",
        operation_id="register",
        response_model=CurrentUser,
        responses={
            409: {"description": "Username already exists", "model": CommandError},
            422: {"description": "Request schema validation failed", "model": CommandError},
        },
    )
    async def register(request: RegisterRequest, response: Response) -> CurrentUser:
        session = new_session()
        user = await resolved_repository.register_user(
            username=request.username,
            password_hash=hash_password(request.password),
            session_token_sha256=session.token_sha256,
            session_expires_at=session.expires_at,
        )
        set_session_cookie(response, session.token)
        return user

    @app.post(
        "/auth/login",
        operation_id="login",
        response_model=CurrentUser,
        responses={
            401: {"description": "Invalid credentials", "model": CommandError},
            422: {"description": "Request schema validation failed", "model": CommandError},
        },
    )
    async def login(request: LoginRequest, response: Response) -> CurrentUser:
        credential = await resolved_repository.get_user_credential(request.username)
        if credential is None or not verify_password(credential.password_hash, request.password):
            raise DomainError("invalid_credentials", "用户名或密码不正确。", 401)
        session = new_session()
        await resolved_repository.create_session(
            user_id=credential.user.user_id,
            token_sha256=session.token_sha256,
            expires_at=session.expires_at,
        )
        set_session_cookie(response, session.token)
        return credential.user

    @app.post("/auth/logout", operation_id="logout", status_code=204)
    async def logout(
        response: Response,
        session_token: Annotated[str | None, Cookie(alias=SESSION_COOKIE)] = None,
    ) -> None:
        if session_token:
            await resolved_repository.revoke_session(session_token_sha256(session_token))
        response.delete_cookie(
            SESSION_COOKIE,
            httponly=True,
            secure=True,
            samesite="lax",
            path="/",
        )

    @app.get("/me", operation_id="getCurrentUser", response_model=CurrentUser)
    async def get_current_user(user: AuthenticatedUser) -> CurrentUser:
        return user

    @app.get("/personas", operation_id="listPersonas", response_model=PersonaList)
    async def list_personas(_: AuthenticatedUser) -> PersonaList:
        return PersonaList(items=resolved_catalog.discover())

    @app.get("/creations", operation_id="listCreations", response_model=CreationList)
    async def list_creations(user: AuthenticatedUser) -> CreationList:
        return await resolved_repository.list_creations(user.user_id)

    @app.post(
        "/creations",
        operation_id="createCreation",
        status_code=202,
        response_model=CreationAccepted,
    )
    async def create_creation(
        request: CreateCreationRequest,
        idempotency_key: IdempotencyKey,
        user: AuthenticatedUser,
    ) -> CreationAccepted:
        key = scoped_key(user, idempotency_key)
        replay = await resolved_repository.replay_create_creation(
            key,
            request,
        )
        if replay is not None:
            return replay
        snapshot = await asyncio.to_thread(
            resolved_catalog.create_snapshot,
            request.persona_id,
        )
        return await resolved_repository.create_creation(
            idempotency_key=key,
            request=request,
            persona_snapshot=snapshot.summary,
            owner_id=user.user_id if authentication_enabled else None,
        )

    @app.get(
        "/creations/{creation_id}",
        operation_id="getCreation",
        response_model=CreationResource,
    )
    async def get_creation(creation_id: UUID, user: AuthenticatedUser) -> CreationResource:
        await require_owner(creation_id, user)
        resource = await resolved_repository.get_creation(creation_id)
        if resource is None:
            raise DomainError("creation_not_found", "Creation not found.", 404)
        return resource

    @app.delete(
        "/creations/{creation_id}",
        operation_id="deleteCreation",
        status_code=204,
        responses={
            404: {"description": "Creation not found", "model": CommandError},
            409: {
                "description": "The creation still has queued or running work",
                "model": CommandError,
            },
        },
    )
    async def delete_creation(creation_id: UUID, user: AuthenticatedUser) -> None:
        await require_owner(creation_id, user)
        await resolved_repository.delete_creation(creation_id)

    @app.get(
        "/creations/{creation_id}/runs/{run_kind}/presentation",
        operation_id="getDeliveryPresentation",
        response_model=DeliveryPresentation,
        summary="Read one succeeded run as a structured delivery presentation",
        responses={
            404: {"description": "Creation not found", "model": CommandError},
            409: {
                "description": "The requested run has no formal delivery",
                "model": CommandError,
            },
        },
    )
    async def get_delivery_presentation(
        creation_id: UUID,
        run_kind: Literal["initial", "revision"],
        user: AuthenticatedUser,
    ) -> DeliveryPresentation:
        await require_owner(creation_id, user)
        return await resolved_repository.get_delivery_presentation(creation_id, run_kind)

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
        user: AuthenticatedUser,
    ) -> RevisionAccepted:
        await require_owner(creation_id, user)
        return await resolved_repository.create_or_retry_revision(
            creation_id=creation_id,
            idempotency_key=scoped_key(user, idempotency_key),
            request=request,
        )

    @app.post(
        "/creations/{creation_id}/runs/{run_kind}/continue",
        operation_id="continueRun",
        status_code=202,
        response_model=RunControlAccepted,
    )
    async def continue_run(
        creation_id: UUID,
        run_kind: Literal["initial", "revision"],
        idempotency_key: IdempotencyKey,
        user: AuthenticatedUser,
    ) -> RunControlAccepted:
        await require_owner(creation_id, user)
        return await resolved_repository.continue_run(
            creation_id=creation_id,
            run_kind=run_kind,
            idempotency_key=scoped_key(user, idempotency_key),
        )

    @app.post(
        "/creations/{creation_id}/runs/{run_kind}/retry-final-review",
        operation_id="retryFinalReview",
        status_code=202,
        response_model=RunControlAccepted,
        summary="Repair the rejected evidence scope and re-run final review",
        description=(
            "Queues the same quality_rejected run unless that gate has exhausted its "
            "three-attempt limit. The worker binds the saved evidence to exact episode "
            "excerpts, applies one bounded immutable-batch repair, verifies the complete "
            "series, and then re-runs L0 (and L4 when the original rejection was L4). "
            "It never re-reviews unchanged drafts. An identical idempotency key replays "
            "its prior accepted response."
        ),
        responses={
            202: {"description": "Bounded repair queued or an identical accepted command replayed"},
            404: {"description": "Creation not found", "model": CommandError},
            409: {
                "description": "The requested run cannot retry final review",
                "model": CommandError,
            },
            422: {"description": "Request schema validation failed", "model": CommandError},
        },
    )
    async def retry_final_review(
        creation_id: UUID,
        run_kind: Literal["initial", "revision"],
        idempotency_key: IdempotencyKey,
        user: AuthenticatedUser,
    ) -> RunControlAccepted:
        await require_owner(creation_id, user)
        return await resolved_repository.retry_final_review(
            creation_id=creation_id,
            run_kind=run_kind,
            idempotency_key=scoped_key(user, idempotency_key),
        )

    @app.post(
        "/creations/{creation_id}/runs/{run_kind}/authorize-repair",
        operation_id="authorizeRepair",
        status_code=202,
        response_model=RunControlAccepted,
        summary="Authorize exactly one generation-plus-review cycle for a repair",
        description=(
            "Grants the pending repair authorization bound to the active lineage and "
            "queues the run for exactly one cycle. If a hard-constraint conflict remains, "
            "the run pauses again with the latest review evidence and no automatic "
            "repetition. Generic Continue cannot "
            "spend a content-repair budget."
        ),
        responses={
            202: {"description": "Repair authorized and queued"},
            404: {"description": "Creation not found", "model": CommandError},
            409: {"description": "The requested run cannot be authorized", "model": CommandError},
            422: {"description": "Request schema validation failed", "model": CommandError},
        },
    )
    async def authorize_repair(
        creation_id: UUID,
        run_kind: Literal["initial", "revision"],
        idempotency_key: IdempotencyKey,
        user: AuthenticatedUser,
    ) -> RunControlAccepted:
        await require_owner(creation_id, user)
        return await resolved_repository.authorize_repair(
            creation_id=creation_id,
            run_kind=run_kind,
            idempotency_key=scoped_key(user, idempotency_key),
        )

    @app.post(
        "/creations/{creation_id}/runs/{run_kind}/end",
        operation_id="endRun",
        status_code=202,
        response_model=RunControlAccepted,
    )
    async def end_run(
        creation_id: UUID,
        run_kind: Literal["initial", "revision"],
        idempotency_key: IdempotencyKey,
        user: AuthenticatedUser,
    ) -> RunControlAccepted:
        await require_owner(creation_id, user)
        return await resolved_repository.end_run(
            creation_id=creation_id,
            run_kind=run_kind,
            idempotency_key=scoped_key(user, idempotency_key),
        )

    @app.post(
        "/creations/{creation_id}/runs/{run_kind}/retry",
        operation_id="retryRun",
        status_code=202,
        response_model=RunControlAccepted,
    )
    async def retry_run(
        creation_id: UUID,
        run_kind: Literal["initial", "revision"],
        idempotency_key: IdempotencyKey,
        user: AuthenticatedUser,
    ) -> RunControlAccepted:
        """Revive a failed initial run whose failure was an operator-fixable relay error.

        Only a terminally failed initial run whose failure code is in the external
        relay allowlist and whose stage attempt budget is not exhausted can be
        retried. The run requeues with its original thread and approved business
        checkpoints, so already approved content is never regenerated. A failed
        revision run keeps its own resubmit-frozen-feedback semantics.
        """
        await require_owner(creation_id, user)
        return await resolved_repository.retry_run(
            creation_id=creation_id,
            run_kind=run_kind,
            idempotency_key=scoped_key(user, idempotency_key),
        )

    generated_openapi = app.openapi

    def openapi_with_authentication_errors() -> dict:
        schema = generated_openapi()
        authentication_response = {
            "description": "Authentication required",
            "content": {
                "application/json": {"schema": {"$ref": "#/components/schemas/CommandError"}}
            },
        }
        for path, methods in schema["paths"].items():
            if path.startswith("/auth/"):
                continue
            for method, operation in methods.items():
                if method in {"get", "post", "put", "patch", "delete"}:
                    operation.setdefault("responses", {})["401"] = authentication_response
        return schema

    app.openapi = openapi_with_authentication_errors  # type: ignore[method-assign]

    return app
