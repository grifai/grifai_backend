"""Tests for TokenTracker."""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import date

from grif.llm.token_tracker import TokenTracker, estimate_cost


def test_estimate_cost_known_model() -> None:
    cost = estimate_cost("claude-sonnet-4-6", 1_000_000)
    assert cost == pytest.approx(7.0)

    cost_haiku = estimate_cost("claude-haiku-4-5-20251001", 1_000_000)
    assert cost_haiku == pytest.approx(0.5)


def test_estimate_cost_unknown_model() -> None:
    # Unknown model → default rate of $5 / 1M
    cost = estimate_cost("unknown-model", 1_000_000)
    assert cost == pytest.approx(5.0)


def test_estimate_cost_zero_tokens() -> None:
    assert estimate_cost("gpt-4o", 0) == 0.0


@pytest.mark.asyncio
async def test_check_daily_limit_within() -> None:
    """Returns (True, remaining) when usage is below limit."""
    mock_session = AsyncMock()

    tracker = TokenTracker(session=mock_session)
    # Patch get_daily_usage to return 100_000
    tracker.get_daily_usage = AsyncMock(return_value=100_000)

    within, remaining = await tracker.check_daily_limit(
        user_id="user_1",
        max_tokens=500_000,
        buffer_tokens=50_000,
    )
    assert within is True
    assert remaining == 400_000


@pytest.mark.asyncio
async def test_check_daily_limit_exceeded() -> None:
    """Returns (False, 0) when usage + buffer exceeds limit."""
    mock_session = AsyncMock()
    tracker = TokenTracker(session=mock_session)
    tracker.get_daily_usage = AsyncMock(return_value=480_000)

    within, remaining = await tracker.check_daily_limit(
        user_id="user_1",
        max_tokens=500_000,
        buffer_tokens=50_000,  # 480k + 50k = 530k > 500k
    )
    assert within is False
    assert remaining == 20_000
