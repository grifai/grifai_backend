"""Tests for pipeline/intent_classifier.py — mocked LLM."""
import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from grif.llm.gateway import LLMGateway, LLMResponse
from grif.models.enums import Complexity, TaskType, Urgency
from grif.pipeline.intent_classifier import IntentClassifier, _extract_json
from grif.pipeline.signal_parser import SignalParser


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _mock_response(content: str) -> LLMResponse:
    raw = MagicMock()
    raw.choices[0].message.content = content
    raw.usage.prompt_tokens = 50
    raw.usage.completion_tokens = 80
    raw.usage.total_tokens = 130
    return LLMResponse(raw=raw, model_used="gpt-4o-mini")


# ─── _extract_json ────────────────────────────────────────────────────────────

def test_extract_json_plain() -> None:
    text = '{"task_type": "search", "language": "ru"}'
    result = _extract_json(text)
    assert result["task_type"] == "search"


def test_extract_json_with_fences() -> None:
    text = '```json\n{"task_type": "research"}\n```'
    result = _extract_json(text)
    assert result["task_type"] == "research"


def test_extract_json_no_json_raises() -> None:
    with pytest.raises(ValueError, match="No JSON"):
        _extract_json("plain text without JSON")


# ─── IntentClassifier ─────────────────────────────────────────────────────────

@pytest.fixture
def gateway_mock() -> AsyncMock:
    gw = AsyncMock(spec=LLMGateway)
    return gw


@pytest.fixture
def classifier(gateway_mock: AsyncMock) -> IntentClassifier:
    return IntentClassifier(gateway=gateway_mock)


@pytest.fixture
def parser() -> SignalParser:
    return SignalParser()


@pytest.mark.asyncio
async def test_classify_hotel_search(
    classifier: IntentClassifier,
    gateway_mock: AsyncMock,
    parser: SignalParser,
) -> None:
    payload = {
        "task_type": "search",
        "entities": {"destination": "Париж"},
        "constraints": {"max_price_per_night": 150},
        "complexity": "simple",
        "urgency": "normal",
        "deadline": None,
        "domain": "travel",
        "language": "ru",
    }
    gateway_mock.complete_json.return_value = _mock_response(json.dumps(payload))

    signal = parser.parse_text("Найди отель в Париже до 150$", user_id="u1")
    intent = await classifier.classify(signal)

    assert intent.task_type == TaskType.SEARCH
    assert intent.complexity == Complexity.SIMPLE
    assert intent.language == "ru"
    assert intent.entities["destination"] == "Париж"
    gateway_mock.complete_json.assert_called_once()


@pytest.mark.asyncio
async def test_classify_research_task(
    classifier: IntentClassifier,
    gateway_mock: AsyncMock,
    parser: SignalParser,
) -> None:
    payload = {
        "task_type": "research",
        "entities": {"topic": "квантовые вычисления"},
        "constraints": {"pages": 30},
        "complexity": "multi_step",
        "urgency": "normal",
        "deadline": "2026-03-25",
        "domain": "research",
        "language": "ru",
    }
    gateway_mock.complete_json.return_value = _mock_response(json.dumps(payload))

    signal = parser.parse_text("Напиши курсовую по квантовым вычислениям", user_id="u1")
    intent = await classifier.classify(signal)

    assert intent.task_type == TaskType.RESEARCH
    assert intent.complexity == Complexity.MULTI_STEP
    assert intent.deadline is not None
    assert intent.entities["topic"] == "квантовые вычисления"


@pytest.mark.asyncio
async def test_classify_operate_task(
    classifier: IntentClassifier,
    gateway_mock: AsyncMock,
    parser: SignalParser,
) -> None:
    payload = {
        "task_type": "operate",
        "entities": {"channel": "telegram", "topic": "AI"},
        "constraints": {},
        "complexity": "multi_step",
        "urgency": "normal",
        "deadline": None,
        "domain": "content",
        "language": "ru",
    }
    gateway_mock.complete_json.return_value = _mock_response(json.dumps(payload))

    signal = parser.parse_text("Веди мой Telegram-канал каждый день про AI", user_id="u1")
    intent = await classifier.classify(signal)

    assert intent.task_type == TaskType.OPERATE
    assert intent.entities["channel"] == "telegram"


@pytest.mark.asyncio
async def test_classify_high_urgency(
    classifier: IntentClassifier,
    gateway_mock: AsyncMock,
    parser: SignalParser,
) -> None:
    payload = {
        "task_type": "coach",
        "entities": {"event": "переговоры"},
        "constraints": {},
        "complexity": "simple",
        "urgency": "high",
        "deadline": None,
        "domain": "negotiation",
        "language": "ru",
    }
    gateway_mock.complete_json.return_value = _mock_response(json.dumps(payload))

    signal = parser.parse_text("Срочно подготовь к переговорам через 2 часа", user_id="u1")
    intent = await classifier.classify(signal)

    assert intent.urgency == Urgency.HIGH
    assert intent.task_type == TaskType.COACH


@pytest.mark.asyncio
async def test_classify_unknown_task_type_defaults_to_search(
    classifier: IntentClassifier,
    gateway_mock: AsyncMock,
    parser: SignalParser,
) -> None:
    """Unknown task_type falls back to SEARCH safely."""
    payload = {
        "task_type": "unknown_xyz",
        "entities": {},
        "constraints": {},
        "complexity": "simple",
        "urgency": "normal",
        "deadline": None,
        "domain": "general",
        "language": "en",
    }
    gateway_mock.complete_json.return_value = _mock_response(json.dumps(payload))

    signal = parser.parse_text("do something", user_id="u1")
    intent = await classifier.classify(signal)
    assert intent.task_type == TaskType.SEARCH
