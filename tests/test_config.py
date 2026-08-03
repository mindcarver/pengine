import pytest
from pydantic import ValidationError

from pengine.config import Settings


@pytest.mark.parametrize("host", ["127.0.0.1", "::1", "localhost"])
def test_loopback_hosts_are_allowed(host: str) -> None:
    assert Settings(host=host).host == host


@pytest.mark.parametrize(
    "relay_base_url",
    [
        "https://relay.example/anthropic",
        "http://localhost:8001/anthropic",
        "http://127.0.0.1:8001/anthropic",
        "http://[::1]:8001/anthropic",
    ],
)
def test_safe_relay_urls_are_allowed(relay_base_url: str) -> None:
    assert Settings(relay_base_url=relay_base_url).relay_base_url == relay_base_url


@pytest.mark.parametrize(
    "relay_base_url",
    [
        "http://relay.example/anthropic",
        "ftp://relay.example/anthropic",
        "not-a-url",
        "https://relay.example:not-a-port/anthropic",
        "https://relay example/anthropic",
        "https://user:password@relay.example/anthropic",
    ],
)
def test_unsafe_relay_urls_are_rejected(relay_base_url: str) -> None:
    with pytest.raises(ValidationError, match="relay_base_url"):
        Settings(relay_base_url=relay_base_url)


def test_non_loopback_host_is_rejected() -> None:
    with pytest.raises(ValidationError, match="loopback"):
        Settings(host="0.0.0.0")


def test_api_key_is_not_revealed_by_settings_repr() -> None:
    settings = Settings(relay_api_key="secret-value")

    assert "secret-value" not in repr(settings)


def test_role_specific_model_routes_are_required_by_default() -> None:
    settings = Settings(_env_file=None)

    assert settings.generation_max_output_tokens == 128_000
    assert settings.review_max_output_tokens is None
    assert settings.relay_configured is False


def test_role_specific_model_routes_load_from_environment(monkeypatch) -> None:
    monkeypatch.setenv("PENGINE_RELAY_BASE_URL", "https://relay.example/v1")
    monkeypatch.setenv("PENGINE_RELAY_API_KEY", "secret-value")
    monkeypatch.setenv("PENGINE_GENERATION_MODEL_ID", "claude-opus-5")
    monkeypatch.setenv("PENGINE_GENERATION_MAX_OUTPUT_TOKENS", "32000")
    monkeypatch.setenv("PENGINE_REVIEW_MODEL_ID", "deepseek-v4-flash")
    monkeypatch.setenv("PENGINE_REVIEW_MAX_OUTPUT_TOKENS", "12000")

    settings = Settings(_env_file=None)

    assert settings.generation_model_id == "claude-opus-5"
    assert settings.generation_max_output_tokens == 32000
    assert settings.review_model_id == "deepseek-v4-flash"
    assert settings.review_max_output_tokens == 12000
    assert settings.relay_configured is True


@pytest.mark.parametrize(
    ("generation_model_id", "review_model_id"),
    [(None, "deepseek-v4-flash"), ("claude-opus-5", None)],
)
def test_both_model_roles_are_required(
    generation_model_id: str | None,
    review_model_id: str | None,
) -> None:
    settings = Settings(
        _env_file=None,
        relay_base_url="https://relay.example/v1",
        relay_api_key="secret-value",
        generation_model_id=generation_model_id,
        review_model_id=review_model_id,
    )

    assert settings.relay_configured is False


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("generation_model_id", "claude-sonnet-5"),
        ("generation_model_id", "deepseek-v4-pro"),
        ("review_model_id", "claude-opus-5"),
    ],
)
def test_model_families_cannot_be_swapped(field: str, value: str) -> None:
    with pytest.raises(ValidationError):
        Settings(_env_file=None, **{field: value})


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("generation_max_output_tokens", 0),
        ("review_max_output_tokens", 0),
    ],
)
def test_invalid_relay_adapter_settings_are_rejected(field: str, value: object) -> None:
    with pytest.raises(ValidationError):
        Settings(_env_file=None, **{field: value})


def test_generation_output_cannot_exceed_opus_5_maximum() -> None:
    with pytest.raises(ValidationError):
        Settings(_env_file=None, generation_max_output_tokens=128_001)
