"""Tests for Settings / config.py."""
import pytest
from unittest.mock import patch

from grif.config import Settings, get_settings


def test_settings_defaults() -> None:
    import os
    # Override env vars to test pure defaults (isolate from .env file)
    with patch.dict(os.environ, {"DEBUG": "false", "APP_ENV": "development"}, clear=False):
        s = Settings(_env_file=None)
        assert s.app_env == "development"
        assert s.debug is False
        assert s.react_max_cycles == 20
        assert s.router_jaccard_threshold == 0.80
        assert s.trust_escalation_approvals == 5


def test_settings_llm_models_configured() -> None:
    s = Settings()
    assert "haiku" in s.llm_classifier_model or "haiku" in s.llm_classifier_model
    assert "sonnet" in s.llm_generator_model


def test_settings_rate_limits() -> None:
    s = Settings()
    assert s.max_active_agents == 10
    assert s.max_tokens_per_day == 500_000
    assert s.max_recurring_agents == 5


def test_settings_ttls() -> None:
    s = Settings()
    assert s.agent_log_retention_days == 30
    assert s.production_memory_retention_days == 365
    assert s.sleeping_agent_archive_days == 90


def test_get_settings_cached() -> None:
    """get_settings() should return the same instance (lru_cache)."""
    s1 = get_settings()
    s2 = get_settings()
    assert s1 is s2
