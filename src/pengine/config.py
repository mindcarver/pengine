from functools import lru_cache
from ipaddress import ip_address
from pathlib import Path
from urllib.parse import urlsplit

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_ALLOWED_REVIEW_MODELS = frozenset(
    {"deepseek-v4-flash", "gpt-5.5", "gpt-5.6-terra", "claude-opus-5"}
)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="PENGINE_",
        extra="ignore",
    )

    persona_root: Path = Path("personas")
    data_dir: Path = Path("data")
    host: str = "127.0.0.1"
    port: int = Field(default=8000, ge=1, le=65535)
    relay_base_url: str | None = None
    relay_api_key: SecretStr | None = None
    generation_model_id: str | None = None
    generation_max_output_tokens: int = Field(default=128_000, ge=1, le=128_000)
    generation_context_limit_tokens: int | None = Field(default=None, ge=1)
    review_model_id: str | None = None
    review_max_output_tokens: int | None = Field(default=None, ge=1)
    review_context_limit_tokens: int | None = Field(default=None, ge=1)
    model_timeout_seconds: float = Field(default=180.0, gt=0)
    run_timeout_seconds: float = Field(default=1800.0, gt=0)
    lease_seconds: int = Field(default=60, ge=5)
    worker_poll_seconds: float = Field(default=0.25, gt=0)
    agent_recursion_limit: int = Field(default=80, ge=4)
    retrieval_limit: int = Field(default=5, ge=1, le=20)

    @property
    def database_path(self) -> Path:
        return self.data_dir / "pengine.sqlite3"

    @property
    def snapshot_root(self) -> Path:
        return self.data_dir / "persona-snapshots"

    @property
    def relay_configured(self) -> bool:
        return bool(
            self.relay_base_url
            and self.relay_api_key
            and self.generation_model_id
            and self.review_model_id
        )

    @field_validator("generation_model_id")
    @classmethod
    def generation_model_must_be_opus_5(cls, value: str | None) -> str | None:
        if value is not None and value != "claude-opus-5":
            raise ValueError("generation_model_id must be claude-opus-5")
        return value

    @field_validator("review_model_id")
    @classmethod
    def review_model_must_be_allowed(cls, value: str | None) -> str | None:
        if value is not None and value not in _ALLOWED_REVIEW_MODELS:
            raise ValueError(
                f"review_model_id must be one of: {', '.join(sorted(_ALLOWED_REVIEW_MODELS))}"
            )
        return value

    @field_validator("host")
    @classmethod
    def host_must_be_loopback(cls, value: str) -> str:
        if value == "localhost":
            return value
        try:
            if ip_address(value).is_loopback:
                return value
        except ValueError:
            pass
        raise ValueError("Pengine V1 may bind only to a loopback address")

    @field_validator("relay_base_url")
    @classmethod
    def relay_base_url_must_be_safe(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if any(character.isspace() for character in value):
            raise ValueError("relay_base_url must be a valid HTTP URL")
        try:
            parsed = urlsplit(value)
            hostname = parsed.hostname
            _ = parsed.port
        except ValueError as exc:
            raise ValueError("relay_base_url must be a valid HTTP URL") from exc
        if parsed.scheme not in {"http", "https"} or not hostname:
            raise ValueError("relay_base_url must be a valid HTTP URL")
        if parsed.username is not None or parsed.password is not None:
            raise ValueError("relay_base_url must not contain credentials")
        if parsed.scheme == "https":
            return value
        if hostname == "localhost":
            return value
        try:
            if ip_address(hostname).is_loopback:
                return value
        except ValueError:
            pass
        raise ValueError("relay_base_url must use HTTPS unless the host is loopback")


@lru_cache
def get_settings() -> Settings:
    return Settings()
