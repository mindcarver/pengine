from __future__ import annotations

import hashlib
import secrets
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerificationError, VerifyMismatchError

SESSION_COOKIE = "pengine_session"
SESSION_TTL = timedelta(days=7)
_PASSWORD_HASHER = PasswordHasher()


@dataclass(frozen=True, slots=True)
class SessionCredential:
    token: str
    token_sha256: str
    expires_at: datetime


def hash_password(password: str) -> str:
    return _PASSWORD_HASHER.hash(password)


def verify_password(password_hash: str, password: str) -> bool:
    try:
        return _PASSWORD_HASHER.verify(password_hash, password)
    except (InvalidHashError, VerificationError, VerifyMismatchError):
        return False


def new_session(now: datetime | None = None) -> SessionCredential:
    issued_at = now or datetime.now(UTC)
    token = secrets.token_urlsafe(32)
    return SessionCredential(
        token=token,
        token_sha256=session_token_sha256(token),
        expires_at=issued_at + SESSION_TTL,
    )


def session_token_sha256(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()
