from functools import lru_cache
from ipaddress import ip_address
from pathlib import Path

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


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
    relay_model_id: str | None = None
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
        return bool(self.relay_base_url and self.relay_api_key and self.relay_model_id)

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


@lru_cache
def get_settings() -> Settings:
    return Settings()
