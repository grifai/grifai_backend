"""Tests for grif/llm/fallback_map.py"""
import pytest

from grif.llm.fallback_map import (
    HAIKU,
    PURPOSE_MAP,
    SONNET,
    SUMMARIZER,
    ModelEntry,
    get_fallback,
    get_model_for_purpose,
)


def test_model_entries_have_fallback() -> None:
    assert SONNET.fallback == "gpt-4o"
    assert HAIKU.fallback == "gpt-4o-mini"
    assert SUMMARIZER.fallback == "claude-haiku-4-5-20251001"


def test_get_fallback_known_model() -> None:
    assert get_fallback("claude-sonnet-4-6") == "gpt-4o"
    assert get_fallback("claude-haiku-4-5-20251001") == "gpt-4o-mini"


def test_get_fallback_unknown_model() -> None:
    assert get_fallback("unknown-model-xyz") is None


def test_architecture_llm_call_points() -> None:
    """
    Per architecture: LLM called ONLY in 4 places.
    Verify that intent_classifier and self_evaluation use Haiku,
    config_generator and conflict_resolver use Sonnet.
    """
    assert get_model_for_purpose("intent_classifier") == HAIKU
    assert get_model_for_purpose("self_evaluation") == HAIKU
    assert get_model_for_purpose("config_generator") == SONNET
    assert get_model_for_purpose("conflict_resolver") == SONNET


def test_get_model_for_unknown_purpose_defaults_to_sonnet() -> None:
    entry = get_model_for_purpose("some_unknown_purpose")
    assert entry == SONNET


def test_all_purposes_have_entries() -> None:
    required_purposes = {
        "intent_classifier",
        "config_generator",
        "conflict_resolver",
        "self_evaluation",
        "clarification",
        "style_cloner",
        "explainer",
        "summarizer",
        "react_reasoning",
        "react_api_call",
        "react_comparison",
    }
    for purpose in required_purposes:
        entry = get_model_for_purpose(purpose)
        assert isinstance(entry, ModelEntry), f"Missing entry for purpose: {purpose}"
