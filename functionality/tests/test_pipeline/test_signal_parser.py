"""Tests for pipeline/signal_parser.py"""
import pytest

from grif.models.enums import PerformanceChannel, SignalType
from grif.pipeline.signal_parser import SignalParser


# ── Helper ────────────────────────────────────────────────────────────────────

@pytest.fixture
def parser() -> SignalParser:
    return SignalParser()


# ── Text parsing ──────────────────────────────────────────────────────────────

def test_parse_text_basic(parser: SignalParser) -> None:
    signal = parser.parse_text("Найди отель в Париже", user_id="u1")
    assert signal.signal_type == SignalType.TEXT
    assert signal.text == "Найди отель в Париже"
    assert signal.user_id == "u1"


def test_parse_text_strips_whitespace(parser: SignalParser) -> None:
    signal = parser.parse_text("  hello   world  ", user_id="u1")
    assert signal.text == "hello world"


def test_parse_text_extracts_urls(parser: SignalParser) -> None:
    signal = parser.parse_text(
        "Check https://example.com for details", user_id="u1"
    )
    assert "urls_mentioned" in signal.metadata
    assert "https://example.com" in signal.metadata["urls_mentioned"]


def test_parse_text_no_urls(parser: SignalParser) -> None:
    signal = parser.parse_text("Simple text without links", user_id="u1")
    assert "urls_mentioned" not in signal.metadata


# ── Voice parsing ─────────────────────────────────────────────────────────────

def test_parse_voice(parser: SignalParser) -> None:
    signal = parser.parse_voice(
        transcript="Закажи пиццу с доставкой",
        user_id="u2",
        audio_duration_s=4.5,
    )
    assert signal.signal_type == SignalType.VOICE
    assert signal.text == "Закажи пиццу с доставкой"
    assert signal.metadata["audio_duration_s"] == 4.5


# ── Event parsing ─────────────────────────────────────────────────────────────

def test_parse_event_telegram(parser: SignalParser) -> None:
    signal = parser.parse_event(
        event_type="telegram_message",
        event_payload={"text": "Hello from Telegram", "chat_id": 123},
        user_id="u3",
    )
    assert signal.signal_type == SignalType.EVENT
    assert signal.text == "Hello from Telegram"
    assert signal.metadata["event_type"] == "telegram_message"


def test_parse_event_generic(parser: SignalParser) -> None:
    signal = parser.parse_event(
        event_type="crm_lead",
        event_payload={"lead_id": "L001", "status": "hot"},
        user_id="u3",
    )
    assert "crm_lead" in signal.text
    assert "L001" in signal.text


# ── Performance signal ────────────────────────────────────────────────────────

def test_parse_performance_signal(parser: SignalParser) -> None:
    signal = parser.parse_performance_signal(
        channel=PerformanceChannel.TELEGRAM,
        metrics={"views": 1500, "reactions": 42},
        user_id="u4",
        agent_id="agent_abc",
    )
    assert signal.signal_type == SignalType.PATTERN
    assert signal.channel == PerformanceChannel.TELEGRAM
    assert signal.metrics["views"] == 1500
    assert signal.metadata["agent_id"] == "agent_abc"


# ── Parsed_at timestamp ───────────────────────────────────────────────────────

def test_parsed_at_set_automatically(parser: SignalParser) -> None:
    signal = parser.parse_text("test", user_id="u1")
    assert signal.parsed_at is not None
