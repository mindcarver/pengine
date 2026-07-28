import pytest
from pydantic import ValidationError

from pengine.config import Settings


@pytest.mark.parametrize("host", ["127.0.0.1", "::1", "localhost"])
def test_loopback_hosts_are_allowed(host: str) -> None:
    assert Settings(host=host).host == host


def test_non_loopback_host_is_rejected() -> None:
    with pytest.raises(ValidationError, match="loopback"):
        Settings(host="0.0.0.0")


def test_api_key_is_not_revealed_by_settings_repr() -> None:
    settings = Settings(relay_api_key="secret-value")

    assert "secret-value" not in repr(settings)
