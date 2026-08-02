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


def test_anthropic_relay_adapter_is_the_default() -> None:
    settings = Settings(_env_file=None)

    assert settings.relay_adapter == "anthropic"
    assert settings.relay_max_output_tokens is None


def test_relay_adapter_and_output_token_override_load_from_environment(monkeypatch) -> None:
    monkeypatch.setenv("PENGINE_RELAY_ADAPTER", "deepseek")
    monkeypatch.setenv("PENGINE_RELAY_MAX_OUTPUT_TOKENS", "16384")

    settings = Settings(_env_file=None)

    assert settings.relay_adapter == "deepseek"
    assert settings.relay_max_output_tokens == 16384


@pytest.mark.parametrize(
    ("field", "value"),
    [("relay_adapter", "openai"), ("relay_max_output_tokens", 0)],
)
def test_invalid_relay_adapter_settings_are_rejected(field: str, value: object) -> None:
    with pytest.raises(ValidationError):
        Settings(_env_file=None, **{field: value})
