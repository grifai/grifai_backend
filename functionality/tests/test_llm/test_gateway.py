"""
Tests for LLMGateway.
All LLM calls are mocked — no real API keys needed.
"""

import pytest
import pytest_asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from grif.llm.gateway import LLMGateway, LLMError, LLMResponse


# ─── Fixtures ─────────────────────────────────────────────────────────────────

def _make_mock_response(content: str = "Hello", tokens: int = 50) -> MagicMock:
    """Build a mock LiteLLM response object."""
    mock = MagicMock()
    mock.choices[0].message.content = content
    mock.usage.prompt_tokens = tokens // 2
    mock.usage.completion_tokens = tokens // 2
    mock.usage.total_tokens = tokens
    return mock


# ─── LLMResponse ──────────────────────────────────────────────────────────────

def test_llm_response_parses_content() -> None:
    raw = _make_mock_response("Test content", tokens=100)
    resp = LLMResponse(raw=raw, model_used="claude-haiku-4-5-20251001")
    assert resp.content == "Test content"
    assert resp.total_tokens == 100
    assert resp.model_used == "claude-haiku-4-5-20251001"


# ─── Gateway completion ───────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_gateway_complete_success() -> None:
    """Gateway returns LLMResponse on success."""
    gateway = LLMGateway()
    mock_raw = _make_mock_response("Paris has many hotels", tokens=80)

    with patch("litellm.acompletion", new_callable=AsyncMock) as mock_completion:
        mock_completion.return_value = mock_raw
        response = await gateway.complete(
            messages=[{"role": "user", "content": "Tell me about Paris hotels"}],
            purpose="intent_classifier",
        )

    assert response.content == "Paris has many hotels"
    assert response.total_tokens == 80
    mock_completion.assert_called_once()


@pytest.mark.asyncio
async def test_gateway_uses_haiku_for_classifier() -> None:
    """intent_classifier purpose → Haiku model."""
    gateway = LLMGateway()
    mock_raw = _make_mock_response()
    called_with_model: list[str] = []

    async def fake_completion(model: str, **kwargs):
        called_with_model.append(model)
        return mock_raw

    with patch("litellm.acompletion", side_effect=fake_completion):
        await gateway.complete(
            messages=[{"role": "user", "content": "classify this"}],
            purpose="intent_classifier",
        )

    assert called_with_model[0] == "claude-haiku-4-5-20251001"


@pytest.mark.asyncio
async def test_gateway_uses_sonnet_for_config_generator() -> None:
    """config_generator purpose → Sonnet model."""
    gateway = LLMGateway()
    mock_raw = _make_mock_response()
    called_with_model: list[str] = []

    async def fake_completion(model: str, **kwargs):
        called_with_model.append(model)
        return mock_raw

    with patch("litellm.acompletion", side_effect=fake_completion):
        await gateway.complete(
            messages=[{"role": "user", "content": "generate config"}],
            purpose="config_generator",
        )

    assert called_with_model[0] == "claude-sonnet-4-6"


@pytest.mark.asyncio
async def test_gateway_fallback_on_primary_failure() -> None:
    """When primary model fails, gateway tries fallback."""
    gateway = LLMGateway()
    mock_raw = _make_mock_response("Fallback response")
    call_count = {"n": 0}

    async def fake_completion(model: str, **kwargs):
        call_count["n"] += 1
        if model == "claude-haiku-4-5-20251001":
            raise Exception("Anthropic unavailable")
        return mock_raw

    with patch("litellm.acompletion", side_effect=fake_completion):
        response = await gateway.complete(
            messages=[{"role": "user", "content": "test"}],
            purpose="intent_classifier",
        )

    assert response.content == "Fallback response"
    assert response.model_used == "gpt-4o-mini"
    assert call_count["n"] == 2  # primary + fallback


@pytest.mark.asyncio
async def test_gateway_raises_when_both_fail() -> None:
    """Raises LLMError when both primary and fallback fail."""
    gateway = LLMGateway()

    async def always_fail(model: str, **kwargs):
        raise Exception(f"Model {model} failed")

    with patch("litellm.acompletion", side_effect=always_fail):
        with pytest.raises(LLMError, match="failed"):
            await gateway.complete(
                messages=[{"role": "user", "content": "test"}],
                purpose="intent_classifier",
            )


@pytest.mark.asyncio
async def test_gateway_explicit_model_id_overrides_purpose() -> None:
    """Explicit model_id takes precedence over purpose."""
    gateway = LLMGateway()
    mock_raw = _make_mock_response()
    called_with: list[str] = []

    async def fake_completion(model: str, **kwargs):
        called_with.append(model)
        return mock_raw

    with patch("litellm.acompletion", side_effect=fake_completion):
        await gateway.complete(
            messages=[{"role": "user", "content": "test"}],
            purpose="intent_classifier",   # would be haiku
            model_id="gpt-4o",             # explicit override
        )

    assert called_with[0] == "gpt-4o"


@pytest.mark.asyncio
async def test_gateway_complete_json_appends_instruction() -> None:
    """complete_json appends JSON instruction to user message."""
    gateway = LLMGateway()
    mock_raw = _make_mock_response('{"key": "value"}')
    captured_messages: list = []

    async def fake_completion(model: str, messages: list, **kwargs):
        captured_messages.extend(messages)
        return mock_raw

    with patch("litellm.acompletion", side_effect=fake_completion):
        await gateway.complete_json(
            messages=[{"role": "user", "content": "Extract data"}],
            purpose="intent_classifier",
        )

    last_msg = captured_messages[-1]
    assert "json" in last_msg["content"].lower()
