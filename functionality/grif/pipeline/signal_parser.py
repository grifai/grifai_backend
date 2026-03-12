"""
Step 1: Signal Parser.
Normalises raw input (text, voice transcript, event, performance signal)
into a uniform ParsedSignal for the Intent Classifier.
No LLM calls — purely deterministic transformation.
"""

import re
from datetime import datetime
from typing import Any

from pydantic import BaseModel

from grif.models.enums import PerformanceChannel, SignalType


class ParsedSignal(BaseModel):
    """Normalised input ready for Intent Classifier."""

    signal_type: SignalType
    text: str                           # Cleaned text content
    user_id: str
    metadata: dict[str, Any] = {}      # Source-specific extras
    parsed_at: datetime = None          # type: ignore[assignment]

    def model_post_init(self, __context: Any) -> None:
        if self.parsed_at is None:
            object.__setattr__(self, "parsed_at", datetime.utcnow())

    model_config = {"arbitrary_types_allowed": True}


class PerformanceSignal(ParsedSignal):
    """Signal from external metrics (Feedback Loop, mechanic #16)."""

    channel: PerformanceChannel
    metrics: dict[str, Any] = {}


# ─── Normalisation helpers ────────────────────────────────────────────────────

_WHITESPACE_RE = re.compile(r"\s+")
_URL_RE = re.compile(r"https?://\S+")


def _clean_text(raw: str) -> str:
    """Strip extra whitespace, normalise newlines."""
    text = raw.strip()
    text = _WHITESPACE_RE.sub(" ", text)
    return text


def _extract_urls(text: str) -> list[str]:
    return _URL_RE.findall(text)


# ─── Signal Parser ────────────────────────────────────────────────────────────

class SignalParser:
    """
    Converts raw input payloads into ParsedSignal objects.
    Supports: text message, voice transcript, webhook event, performance metric.
    """

    def parse_text(
        self,
        raw_text: str,
        user_id: str,
        metadata: dict[str, Any] | None = None,
    ) -> ParsedSignal:
        """Parse a plain text message from the user."""
        cleaned = _clean_text(raw_text)
        urls = _extract_urls(cleaned)
        extra = metadata or {}
        if urls:
            extra["urls_mentioned"] = urls

        return ParsedSignal(
            signal_type=SignalType.TEXT,
            text=cleaned,
            user_id=user_id,
            metadata=extra,
        )

    def parse_voice(
        self,
        transcript: str,
        user_id: str,
        audio_duration_s: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ParsedSignal:
        """Parse a voice message transcript (ASR output)."""
        cleaned = _clean_text(transcript)
        extra = metadata or {}
        if audio_duration_s is not None:
            extra["audio_duration_s"] = audio_duration_s

        return ParsedSignal(
            signal_type=SignalType.VOICE,
            text=cleaned,
            user_id=user_id,
            metadata=extra,
        )

    def parse_event(
        self,
        event_type: str,
        event_payload: dict[str, Any],
        user_id: str,
    ) -> ParsedSignal:
        """
        Parse a structured event (webhook, Telegram update, CRM trigger, etc.).
        Converts the event to a natural-language description for the classifier.
        """
        # Build a text description so the classifier can work uniformly
        text = self._event_to_text(event_type, event_payload)

        return ParsedSignal(
            signal_type=SignalType.EVENT,
            text=text,
            user_id=user_id,
            metadata={"event_type": event_type, "payload": event_payload},
        )

    def parse_performance_signal(
        self,
        channel: PerformanceChannel,
        metrics: dict[str, Any],
        user_id: str,
        agent_id: str | None = None,
    ) -> PerformanceSignal:
        """
        Parse an incoming performance signal from external metrics (mechanic #16).
        Used by Feedback Loop to update Production Memory.
        """
        text = self._metrics_to_text(channel, metrics)
        return PerformanceSignal(
            signal_type=SignalType.PATTERN,
            text=text,
            user_id=user_id,
            channel=channel,
            metrics=metrics,
            metadata={"agent_id": agent_id} if agent_id else {},
        )

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _event_to_text(
        self, event_type: str, payload: dict[str, Any]
    ) -> str:
        """Convert a structured event to a text description."""
        # Telegram message event
        if event_type == "telegram_message":
            return payload.get("text", f"Telegram event: {event_type}")

        # Generic fallback: key=value pairs
        parts = [f"{k}={v}" for k, v in list(payload.items())[:5]]
        return f"Event {event_type}: " + ", ".join(parts)

    def _metrics_to_text(
        self, channel: PerformanceChannel, metrics: dict[str, Any]
    ) -> str:
        """Describe performance metrics as human-readable text."""
        parts = [f"{k}={v}" for k, v in metrics.items()]
        return f"Performance update from {channel}: " + ", ".join(parts)
