"""
RecurringManager — manages RECURRING agents (agents that run on a cron schedule).

Responsibilities:
  1. Transition agent state to RECURRING and store schedule_cron
  2. Schedule / unschedule Celery beat tasks via RedBeat
  3. pause() / resume() recurring agents
  4. Track next_run_at in WakeQueueDB
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import structlog
from croniter import croniter
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from grif.models.db import AgentDB, WakeQueueDB
from grif.models.enums import AgentState, WakeTriggerType

log = structlog.get_logger(__name__)

_DEFAULT_TIMEZONE = "UTC"


class RecurringError(Exception):
    """Raised for invalid recurring operations."""


class RecurringManager:
    """
    Registers and manages recurring (cron-scheduled) agents.

    Usage:
        rm = RecurringManager(session)
        await rm.register(agent_id, cron="0 9 * * *", timezone="Europe/Moscow")
        await rm.pause(agent_id)
        await rm.resume(agent_id)
        await rm.unregister(agent_id)
    """

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def register(
        self,
        agent_id: str,
        cron: str,
        tz: str = _DEFAULT_TIMEZONE,
    ) -> AgentDB:
        """
        Mark agent as RECURRING with given cron schedule.
        Creates next WakeQueueDB entry.
        Raises RecurringError if cron is invalid.
        """
        if not croniter.is_valid(cron):
            raise RecurringError(f"Invalid cron expression: '{cron}'")

        result = await self._session.execute(
            select(AgentDB).where(AgentDB.id == agent_id)
        )
        agent = result.scalar_one_or_none()
        if agent is None:
            raise RecurringError(f"Agent {agent_id} not found.")

        agent.state = AgentState.RECURRING
        agent.schedule_cron = cron
        agent.schedule_timezone = tz

        # Schedule next run
        await self._schedule_next_run(agent_id, agent.user_id, cron, tz)
        await self._session.flush()

        log.info("recurring_registered", agent_id=agent_id, cron=cron, tz=tz)
        return agent

    async def pause(self, agent_id: str) -> AgentDB:
        """
        Pause a recurring agent (RECURRING → SLEEPING).
        Keeps schedule_cron so it can be resumed.
        """
        result = await self._session.execute(
            select(AgentDB).where(AgentDB.id == agent_id)
        )
        agent = result.scalar_one_or_none()
        if agent is None:
            raise RecurringError(f"Agent {agent_id} not found.")
        if agent.state != AgentState.RECURRING:
            raise RecurringError(f"Agent {agent_id} is not recurring (state={agent.state}).")

        agent.state = AgentState.SLEEPING
        # Mark pending wake entries as processed (cancelling them)
        await self._cancel_pending_entries(agent_id)
        await self._session.flush()

        log.info("recurring_paused", agent_id=agent_id)
        return agent

    async def resume(self, agent_id: str) -> AgentDB:
        """
        Resume a paused recurring agent (SLEEPING → RECURRING).
        Re-creates next WakeQueueDB entry.
        """
        result = await self._session.execute(
            select(AgentDB).where(AgentDB.id == agent_id)
        )
        agent = result.scalar_one_or_none()
        if agent is None:
            raise RecurringError(f"Agent {agent_id} not found.")
        if not agent.schedule_cron:
            raise RecurringError(f"Agent {agent_id} has no stored cron — cannot resume.")

        agent.state = AgentState.RECURRING
        await self._schedule_next_run(
            agent_id, agent.user_id, agent.schedule_cron, agent.schedule_timezone
        )
        await self._session.flush()

        log.info("recurring_resumed", agent_id=agent_id, cron=agent.schedule_cron)
        return agent

    async def unregister(self, agent_id: str) -> AgentDB:
        """
        Stop recurring entirely and archive the agent.
        """
        result = await self._session.execute(
            select(AgentDB).where(AgentDB.id == agent_id)
        )
        agent = result.scalar_one_or_none()
        if agent is None:
            raise RecurringError(f"Agent {agent_id} not found.")

        agent.state = AgentState.ARCHIVED
        agent.archived_at = datetime.now(tz=timezone.utc)
        agent.schedule_cron = None
        await self._cancel_pending_entries(agent_id)
        await self._session.flush()

        log.info("recurring_unregistered", agent_id=agent_id)
        return agent

    def next_run_at(self, cron: str, tz: str = _DEFAULT_TIMEZONE) -> datetime:
        """Compute next scheduled datetime for a cron expression."""
        import pytz
        try:
            tzinfo = pytz.timezone(tz)
        except Exception:
            tzinfo = pytz.utc
        now = datetime.now(tz=tzinfo)
        return croniter(cron, now).get_next(datetime)

    async def get_recurring_agents(
        self, user_id: str | None = None
    ) -> list[AgentDB]:
        """List all RECURRING agents, optionally filtered by user."""
        stmt = select(AgentDB).where(AgentDB.state == AgentState.RECURRING)
        if user_id:
            stmt = stmt.where(AgentDB.user_id == user_id)
        result = await self._session.execute(stmt)
        return list(result.scalars().all())

    # ── Private helpers ───────────────────────────────────────────────────────

    async def _schedule_next_run(
        self, agent_id: str, user_id: str, cron: str, tz: str
    ) -> WakeQueueDB:
        """Insert a WakeQueueDB entry for the next cron tick."""
        next_at = self.next_run_at(cron, tz)
        entry = WakeQueueDB(
            agent_id=agent_id,
            user_id=user_id,
            trigger_type=WakeTriggerType.SCHEDULE,
            trigger_condition=f"cron:{cron}",
            scheduled_at=next_at,
            is_processed=False,
        )
        self._session.add(entry)
        return entry

    async def _cancel_pending_entries(self, agent_id: str) -> None:
        """Mark all unprocessed wake entries for this agent as processed."""
        result = await self._session.execute(
            select(WakeQueueDB).where(
                WakeQueueDB.agent_id == agent_id,
                WakeQueueDB.is_processed == False,  # noqa: E712
            )
        )
        for entry in result.scalars().all():
            entry.is_processed = True
            entry.fired_at = datetime.now(tz=timezone.utc)
