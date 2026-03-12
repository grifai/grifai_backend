"""Conftest for API tests — skips DB lifespan by setting env to 'test'."""
import os
from unittest.mock import patch

import pytest


@pytest.fixture(autouse=True)
def patch_app_env(monkeypatch):
    """
    Set APP_ENV=test so the FastAPI lifespan skips automatic table creation.
    All API tests use mocked sessions; no real PostgreSQL is needed.
    """
    monkeypatch.setenv("APP_ENV", "test")
    # Also patch the already-cached settings object used by main.py
    with patch("grif.main.settings") as mock_settings:
        mock_settings.app_env = "test"
        mock_settings.debug = True
        mock_settings.app_version = "0.1.0"
        yield
