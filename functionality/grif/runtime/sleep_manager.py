"""
SleepManager — transitions an agent from ACTIVE to SLEEPING.

Responsibilities:
  1. Serialize the ReAct state (checkpoint) into AgentDB.checkpoint
  2. Update AgentDB.state → SLEEPING
  3. Create WakeQueueDB entry for each trigger
  4. Save context_summary for quick wake-up restoration
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import structlog
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from grif.models.agent_config import WakeTrigger
from grif.models.db import AgentDB, WakeQueueDB
from grif.models.enums import AgentState, WakeTriggerType

log = structlog.get_logger(__name__)


class SleepError(Exception):
    """Raised when an agent cannot be put to sleep."""


class SleepManager:
    """
    Puts an agent to sleep: checkpoints state and registers wake triggers.

    Usage:
        sm = SleepManager(session)
        await sm.sleep(
            agent_id="...",
            reason="Task complete for today",
            checkpoint=react_final_state,
            wake_triggers=[WakeTrigger(trigger_type=WakeTriggerType.SCHEDULE, ...)]
        )
    """

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def sleep(
        self,
        agent_id: str,
        reason: str,
        checkpoint: dict[str, Any] | None = None,
        wake_triggers: list[WakeTrigger] | None = None,
        context_summary: str | None = None,
    ) -> AgentDB:
        """
        Transition agent to SLEEPING state.
        Returns updated AgentDB row.

        Raises SleepError if agent not found or already archived.
        """
        result = await self._session.execute(
            select(AgentDB).where(AgentDB.id == agent_id)
        )
        agent = result.scalar_one_or_none()

        if agent is None:
            raise SleepError(f"Agent {agent_id} not found.")
        if agent.state == AgentState.ARCHIVED:
            raise SleepError(f"Agent {agent_id} is archived — cannot sleep.")

        # Update agent state
        agent.state = AgentState.SLEEPING
        if checkpoint is not None:
            agent.checkpoint = checkpoint
        if context_summary:
            agent.context_summary = context_summary

        # Register wake triggers
        triggers = wake_triggers or []
        for trigger in triggers:
            # value field may hold a datetime for SCHEDULE triggers
            scheduled_at = None
            if trigger.trigger_type == WakeTriggerType.SCHEDULE and isinstance(trigger.value, datetime):
                scheduled_at = trigger.value
            wake_entry = WakeQueueDB(
                agent_id=agent_id,
                user_id=agent.user_id,
                trigger_type=trigger.trigger_type,
                trigger_condition=trigger.condition,
                scheduled_at=scheduled_at,
                is_processed=False,
            )
            self._session.add(wake_entry)

        await self._session.flush()

        log.info(
            "agent_sleeping",
            agent_id=agent_id,
            reason=reason,
            triggers=len(triggers),
        )
        return agent

    async def get_due_wake_entries(
        self, limit: int = 50
    ) -> list[WakeQueueDB]:
        """
        Return unprocessed wake entries whose scheduled_at <= now.
        Used by the Celery wake-checker beat task.
        """
        now = datetime.utcnow()
        result = await self._session.execute(
            select(WakeQueueDB)
            .where(
                WakeQueueDB.is_processed == False,  # noqa: E712
                WakeQueueDB.trigger_type == WakeTriggerType.SCHEDULE,
                WakeQueueDB.scheduled_at <= now,
            )
            .limit(limit)
        )
        return list(result.scalars().all())

    async def get_sleeping_agents(self, user_id: str | None = None) -> list[AgentDB]:
        """Return all agents currently in SLEEPING state, optionally filtered by user."""
        stmt = select(AgentDB).where(AgentDB.state == AgentState.SLEEPING)
        if user_id:
            stmt = stmt.where(AgentDB.user_id == user_id)
        result = await self._session.execute(stmt)
        return list(result.scalars().all())

    async def mark_wake_entry_processed(self, entry: WakeQueueDB) -> None:
        """Mark a WakeQueueDB entry as fired."""
        entry.is_processed = True
        entry.fired_at = datetime.utcnow()
        await self._session.flush()
