from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient
from persona_factory import create_persona_package

from pengine.api import create_app
from pengine.auth import SESSION_COOKIE, session_token_sha256
from pengine.config import Settings


def _app(tmp_path: Path):
    persona_root = tmp_path / "personas"
    create_persona_package(persona_root / "active")
    return create_app(settings=Settings(persona_root=persona_root, data_dir=tmp_path / "data"))


@pytest.mark.asyncio
async def test_registration_session_and_logout_are_secure_and_revocable(tmp_path: Path) -> None:
    app = _app(tmp_path)
    async with (
        app.router.lifespan_context(app),
        AsyncClient(transport=ASGITransport(app=app), base_url="https://test") as client,
    ):
        anonymous = await client.get("/me")
        anonymous_business = await client.get("/creations")
        registered = await client.post(
            "/auth/register",
            json={"username": "  alice  ", "password": "correct-horse"},
        )
        current = await client.get("/me")
        duplicate = await client.post(
            "/auth/register",
            json={"username": "alice", "password": "another-pass"},
        )
        logout = await client.post("/auth/logout")
        logged_out = await client.get("/me")
        repeated_logout = await client.post("/auth/logout")

        async with app.state.repository._connection() as connection:
            user_row = await (
                await connection.execute("SELECT password_hash FROM users WHERE username = 'alice'")
            ).fetchone()
            session_row = await (
                await connection.execute(
                    "SELECT token_sha256, revoked_at FROM sessions ORDER BY created_at LIMIT 1"
                )
            ).fetchone()

    assert anonymous.status_code == 401
    assert anonymous_business.status_code == 401
    assert registered.status_code == 200
    assert registered.json()["username"] == "alice"
    cookie = registered.headers["set-cookie"]
    assert f"{SESSION_COOKIE}=" in cookie
    assert "HttpOnly" in cookie
    assert "Secure" in cookie
    assert "SameSite=lax" in cookie
    assert "Max-Age=604800" in cookie
    assert current.status_code == 200
    assert duplicate.status_code == 409
    assert duplicate.json()["code"] == "username_taken"
    assert logout.status_code == 204
    assert logged_out.status_code == 401
    assert repeated_logout.status_code == 204
    assert user_row["password_hash"] != "correct-horse"
    assert user_row["password_hash"].startswith("$argon2id$")
    assert len(session_row["token_sha256"]) == 64
    assert session_row["revoked_at"] is not None


@pytest.mark.asyncio
async def test_login_uses_generic_failure_and_session_expiry(tmp_path: Path) -> None:
    app = _app(tmp_path)
    repository = app.state.repository
    async with (
        app.router.lifespan_context(app),
        AsyncClient(transport=ASGITransport(app=app), base_url="https://test") as client,
    ):
        registered = await client.post(
            "/auth/register",
            json={"username": "Alice", "password": "correct-horse"},
        )
        client.cookies.clear()
        wrong_password = await client.post(
            "/auth/login",
            json={"username": "Alice", "password": "incorrect-pass"},
        )
        unknown_user = await client.post(
            "/auth/login",
            json={"username": "nobody", "password": "incorrect-pass"},
        )
        login = await client.post(
            "/auth/login",
            json={"username": "Alice", "password": "correct-horse"},
        )

        token = client.cookies[SESSION_COOKIE]
        expired = await repository.resolve_session(
            session_token_sha256(token),
            now=datetime.now(UTC) + timedelta(days=8),
        )

    assert registered.status_code == 200
    assert wrong_password.status_code == 401
    assert unknown_user.status_code == 401
    assert wrong_password.json() == unknown_user.json()
    assert login.status_code == 200
    assert expired is None


@pytest.mark.asyncio
async def test_creation_listing_and_foreign_ids_are_account_scoped(tmp_path: Path) -> None:
    app = _app(tmp_path)
    transport = ASGITransport(app=app)
    request = {
        "persona_id": "test-persona",
        "story": "一个人回乡面对旧事。",
        "requirements": "生成完整短剧。",
    }
    async with app.router.lifespan_context(app):
        async with AsyncClient(transport=transport, base_url="https://test") as alice:
            await alice.post(
                "/auth/register",
                json={"username": "alice", "password": "correct-horse"},
            )
            created = await alice.post(
                "/creations",
                headers={"Idempotency-Key": "shared-key"},
                json=request,
            )
            creation_id = created.json()["creation_id"]
            own_list = await alice.get("/creations")
            own_detail = await alice.get(f"/creations/{creation_id}")

        async with AsyncClient(transport=transport, base_url="https://test") as bob:
            await bob.post(
                "/auth/register",
                json={"username": "bob", "password": "correct-horse"},
            )
            foreign_get = await bob.get(f"/creations/{creation_id}")
            foreign_control = await bob.post(
                f"/creations/{creation_id}/runs/initial/retry",
                headers={"Idempotency-Key": "foreign"},
            )
            empty_list = await bob.get("/creations")
            bob_created = await bob.post(
                "/creations",
                headers={"Idempotency-Key": "shared-key"},
                json=request,
            )

    assert created.status_code == 202
    assert own_list.status_code == 200
    assert [item["creation_id"] for item in own_list.json()["items"]] == [creation_id]
    assert own_list.json()["items"][0]["queue_position"] == 1
    assert own_detail.json()["initial"]["queue_position"] == 1
    assert foreign_get.status_code == 404
    assert foreign_control.status_code == 404
    assert foreign_get.json() == foreign_control.json()
    assert empty_list.json() == {"items": []}
    assert bob_created.status_code == 202
    assert bob_created.json()["creation_id"] != creation_id


@pytest.mark.asyncio
async def test_delete_creation_is_account_scoped(tmp_path: Path) -> None:
    app = _app(tmp_path)
    transport = ASGITransport(app=app)
    request = {
        "persona_id": "test-persona",
        "story": "一个人回乡面对旧事。",
        "requirements": "生成完整短剧。",
    }
    async with (
        app.router.lifespan_context(app),
        AsyncClient(transport=transport, base_url="https://test") as alice,
    ):
            await alice.post(
                "/auth/register",
                json={"username": "alice", "password": "correct-horse"},
            )
            created = await alice.post(
                "/creations",
                headers={"Idempotency-Key": "scoped-delete"},
                json=request,
            )
            creation_id = created.json()["creation_id"]

            async with AsyncClient(transport=transport, base_url="https://test") as anonymous:
                anonymous_delete = await anonymous.delete(f"/creations/{creation_id}")

            async with AsyncClient(transport=transport, base_url="https://test") as bob:
                await bob.post(
                    "/auth/register",
                    json={"username": "bob", "password": "correct-horse"},
                )
                foreign_delete = await bob.delete(f"/creations/{creation_id}")

            still_listed = await alice.get("/creations")

    assert anonymous_delete.status_code == 401
    assert anonymous_delete.json()["code"] == "authentication_required"
    assert foreign_delete.status_code == 404
    assert foreign_delete.json()["code"] == "creation_not_found"
    assert [item["creation_id"] for item in still_listed.json()["items"]] == [creation_id]
