from pydantic import BaseModel, SecretStr

from pengine.config import Settings
from pengine.relay import RelayError, build_chat_model, classify_relay_exception


def test_build_chat_model_uses_only_operator_connection_fields() -> None:
    settings = Settings(
        relay_base_url="https://relay.example/anthropic",
        relay_api_key="secret-value",
        relay_model_id="model-id",
    )

    model = build_chat_model(settings)

    assert model.model == "model-id"
    assert model.anthropic_api_url == "https://relay.example/anthropic"
    assert model.max_retries == 0
    assert isinstance(model.anthropic_api_key, SecretStr)
    assert model.anthropic_api_key.get_secret_value() == "secret-value"
    assert "secret-value" not in repr(model)


def test_build_chat_model_disables_parallel_tool_calls() -> None:
    class ProbeTool(BaseModel):
        value: str

    settings = Settings(
        relay_base_url="https://relay.example/anthropic",
        relay_api_key="secret-value",
        relay_model_id="model-id",
    )

    bound = build_chat_model(settings).bind_tools([ProbeTool])

    assert bound.kwargs["tool_choice"]["disable_parallel_tool_use"] is True


def test_missing_relay_configuration_is_safe() -> None:
    settings = Settings(_env_file=None)

    try:
        build_chat_model(settings)
    except RelayError as exc:
        assert exc.code == "relay_unavailable"
        assert "key" not in exc.safe_message.lower()
    else:
        raise AssertionError("Expected missing relay configuration to fail")


def test_exception_mapping_does_not_echo_provider_body() -> None:
    class BadRequestError(Exception):
        pass

    raw = "vendor body with api-key=secret-value and full prompt"
    mapped = classify_relay_exception(BadRequestError(raw))

    assert mapped.code == "relay_incompatible"
    assert raw not in mapped.safe_message
    assert "secret-value" not in mapped.safe_message
