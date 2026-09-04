from functools import lru_cache
from ipaddress import ip_address
from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

ANTHROPIC_MODEL_IDS = frozenset({"claude-opus-5", "claude-sonnet-5"})
# OpenRouter's Anthropic-compatible endpoint (base_url https://openrouter.ai/api)
# addresses Claude models with an ``anthropic/`` slug prefix. The prefixed slugs
# ride the same _SerialChatAnthropic route, so they share the anthropic profile.
OPENROUTER_ANTHROPIC_MODEL_IDS = frozenset(
    f"anthropic/{model_id}" for model_id in ANTHROPIC_MODEL_IDS
)
OPENROUTER_CHAT_COMPLETIONS_MODEL_IDS = frozenset(
    {"z-ai/glm-5.3-flash", "deepseek/deepseek-v4-flash"}
)
_ALLOWED_GENERATION_MODELS = (
    ANTHROPIC_MODEL_IDS | OPENROUTER_ANTHROPIC_MODEL_IDS | OPENROUTER_CHAT_COMPLETIONS_MODEL_IDS
)
_ALLOWED_REVIEW_MODELS = (
    frozenset({"deepseek-v4-flash", "gpt-5.5", "gpt-5.6-terra", *ANTHROPIC_MODEL_IDS})
    | OPENROUTER_ANTHROPIC_MODEL_IDS
    | OPENROUTER_CHAT_COMPLETIONS_MODEL_IDS
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
    # Tag the stable system prefix of Anthropic-route requests with an ephemeral
    # cache_control breakpoint: cache hits cut prefill TTFB to seconds, price the
    # cached input at 0.1x, and — on the relay's rotating upstream pool — skip the
    # cold-prefill path where its stalls live. Misses pay a 1.25x cache-write
    # premium on the tagged block.
    prompt_cache_enabled: bool = Field(default=True)
    # Streaming stall watchdog (Issue #266): a relay in a degraded window either
    # hangs a stream silently until its ~6-minute kill or trickles a few tokens
    # per minute. The watchdog aborts such generation streams early and the run
    # recovers through the bounded relay-interruption path. Defaults keep >=4x
    # margin against the slowest observed healthy window (~8 tok/s ≈ 8 chars/s).
    stream_watchdog_enabled: bool = Field(default=True)
    stream_stall_seconds: float = Field(default=120.0, gt=0)
    stream_crawl_window_seconds: float = Field(default=90.0, gt=0)
    stream_crawl_min_chars_per_second: float = Field(default=2.0, gt=0)
    lease_seconds: int = Field(default=60, ge=5)
    worker_poll_seconds: float = Field(default=0.25, gt=0)
    worker_concurrency: int = Field(default=5, ge=1, le=5)
    agent_recursion_limit: int = Field(default=200, ge=4)
    retrieval_limit: int = Field(default=5, ge=1, le=20)
    # Per-stage outbound-call guardrails. These are deliberately separate from the
    # graph recursion limit: one model call can consume several graph/tool steps.
    stage_model_call_limit: int = Field(default=48, ge=1)
    stage_review_call_limit: int = Field(default=32, ge=1)
    # Script generation spans multiple episodes: keep per-episode limits above while
    # retaining a bounded total for the whole script stage.
    script_stage_model_call_total_limit: int = Field(default=192, ge=1)
    script_stage_review_call_total_limit: int = Field(default=128, ge=1)
    # Content-review strictness tier. "standard" keeps every quality gate
    # blocking (paid repair rounds, content_rejected pauses). "mvp" trades
    # polish for throughput: per-episode deterministic continuity issues and
    # non-final milestone rejections become advisory (recorded, never
    # blocking), and outline-group semantic review is skipped entirely.
    # Structural integrity checks (schema, hashes, ordering, protocol
    # self-repair) and the binding final review stay enforcing in both tiers.
    content_review_tier: Literal["standard", "mvp"] = "standard"
    # Optional agent observability via a self-hosted Langfuse instance.
    # Disabled by default so pengine runs unchanged when tracing is off.
    langfuse_enabled: bool = Field(default=False)
    langfuse_host: str | None = None
    langfuse_public_key: SecretStr | None = None
    langfuse_secret_key: SecretStr | None = None

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

    @property
    def langfuse_configured(self) -> bool:
        return bool(
            self.langfuse_enabled
            and self.langfuse_host
            and self.langfuse_public_key
            and self.langfuse_secret_key
        )

    @field_validator("generation_model_id")
    @classmethod
    def generation_model_must_be_allowed(cls, value: str | None) -> str | None:
        if value is not None and value not in _ALLOWED_GENERATION_MODELS:
            raise ValueError(
                "generation_model_id must be one of: "
                f"{', '.join(sorted(_ALLOWED_GENERATION_MODELS))}"
            )
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
