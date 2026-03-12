"""
Per-user, per-agent, per-day token usage tracking.
Integrates with the Priority Queue: when daily limit is exceeded,
low-priority agents are paused.
"""

from datetime import date, datetime
from typing import TYPE_CHECKING

import structlog
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from grif.models.db import TokenUsageDB

if TYPE_CHECKING:
    pass

log = structlog.get_logger(__name__)

# Approximate USD costs per 1M tokens (input+output blended)
_COST_PER_1M: dict[str, float] = {
    "claude-sonnet-4-6": 7.0,
    "claude-haiku-4-5-20251001": 0.5,
    "gpt-4o": 10.0,
    "gpt-4o-mini": 0.3,
}


def estimate_cost(model_id: str, total_tokens: int) -> float:
    rate = _COST_PER_1M.get(model_id, 5.0)
    return (total_tokens / 1_000_000) * rate


class TokenTracker:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def record(
        self,
        user_id: str,
        model_id: str,
        prompt_tokens: int,
        completion_tokens: int,
        agent_id: str | None = None,
    ) -> TokenUsageDB:
        total = prompt_tokens + completion_tokens
        cost = estimate_cost(model_id, total)

        entry = TokenUsageDB(
            user_id=user_id,
            agent_id=agent_id,
            model_id=model_id,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total,
            cost_usd=cost,
        )
        self._session.add(entry)
        await self._session.flush()

        log.debug(
            "token_usage_recorded",
            user_id=user_id,
            model_id=model_id,
            total_tokens=total,
            cost_usd=cost,
        )
        return entry

    async def get_daily_usage(self, user_id: str, day: date | None = None) -> int:
        """Return total tokens used by user today (UTC)."""
        target_day = day or date.today()
        start = datetime(target_day.year, target_day.month, target_day.day)
        end = datetime(target_day.year, target_day.month, target_day.day, 23, 59, 59)

        result = await self._session.execute(
            select(func.coalesce(func.sum(TokenUsageDB.total_tokens), 0)).where(
                TokenUsageDB.user_id == user_id,
                TokenUsageDB.created_at >= start,
                TokenUsageDB.created_at <= end,
            )
        )
        return int(result.scalar_one())

    async def get_agent_total(self, agent_id: str) -> int:
        """Return total tokens used by a specific agent across all time."""
        result = await self._session.execute(
            select(func.coalesce(func.sum(TokenUsageDB.total_tokens), 0)).where(
                TokenUsageDB.agent_id == agent_id
            )
        )
        return int(result.scalar_one())

    async def check_daily_limit(
        self, user_id: str, max_tokens: int, buffer_tokens: int = 0
    ) -> tuple[bool, int]:
        """
        Check whether user is within daily token limit.
        Returns (within_limit, tokens_remaining).
        buffer_tokens: projected tokens for the upcoming request.
        """
        used = await self.get_daily_usage(user_id)
        remaining = max_tokens - used
        within = (used + buffer_tokens) <= max_tokens
        return within, max(0, remaining)
