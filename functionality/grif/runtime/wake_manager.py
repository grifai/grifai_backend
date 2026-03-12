"""
WakeManager — transitions an agent from SLEEPING back to ACTIVE.

Responsibilities:
  1. Update AgentDB.state → ACTIVE
  2. Restore short-term memory from DB logs (MemoryManager.load_short_term_from_db)
  3. Return restored context dict for the ReAct loop to resume from
  4. Optionally inject a wake message (e.g. "Price crossed threshold: €120")
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import structlog
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from grif.models.db import AgentDB, WakeQueueDB
from grif.models.enums import AgentState, WakeTriggerType

log = structlog.get_logger(__name__)


class WakeError(Exception):
    """Raised when an agent cannot be woken."""


class WakeContext:
    """Context returned after successfully waking an agent."""

    def __init__(
        self,
        agent_id: str,
        user_id: str,
        checkpoint: dict[str, Any] | None,
        context_summary: str | None,
        trigger_type: WakeTriggerType,
        trigger_message: str,
    ) -> None:
        self.agent_id = agent_id
        self.user_id = user_id
        self.checkpoint = checkpoint            # Last ReAct state snapshot
        self.context_summary = context_summary  # Working memory summary
        self.trigger_type = trigger_type
        self.trigger_message = trigger_message  # Human-readable wake reason


class WakeManager:
    """
    Wakes a sleeping agent.

    Usage:
        wm = WakeManager(session)
        ctx = await wm.wake(
            agent_id="...",
            trigger_type=WakeTriggerType.SCHEDULE,
            trigger_message="Daily check-in",
        )
        # Then pass ctx.checkpoint to ReActGraph.run()
    """

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def wake(
        self,
        agent_id: str,
        trigger_type: WakeTriggerType = WakeTriggerType.MANUAL,
        trigger_message: str = "Manual wake",
        wake_entry: WakeQueueDB | None = None,
    ) -> WakeContext:
        """
        Transition agent SLEEPING → ACTIVE.
        Returns WakeContext with restored checkpoint and summary.

        Raises WakeError if agent not found or not in SLEEPING state.
        """
        result = await self._session.execute(
            select(AgentDB).where(AgentDB.id == agent_id)
        )
        agent = result.scalar_one_or_none()

        if agent is None:
            raise WakeError(f"Agent {agent_id} not found.")
        if agent.state not in (AgentState.SLEEPING, AgentState.RECURRING):
            raise WakeError(
                f"Agent {agent_id} is in state '{agent.state}', expected sleeping/recurring."
            )

        # Transition to active
        agent.state = AgentState.ACTIVE

        # Mark wake entry as processed
        if wake_entry is not None:
            wake_entry.is_processed = True
            wake_entry.fired_at = datetime.utcnow()

        await self._session.flush()

        ctx = WakeContext(
            agent_id=agent_id,
            user_id=agent.user_id,
            checkpoint=agent.checkpoint,
            context_summary=agent.context_summary,
            trigger_type=trigger_type,
            trigger_message=trigger_message,
        )

        log.info(
            "agent_woken",
            agent_id=agent_id,
            trigger_type=trigger_type,
            has_checkpoint=agent.checkpoint is not None,
        )
        return ctx

    async def wake_by_entry(self, entry: WakeQueueDB) -> WakeContext:
        """Convenience: wake agent from a WakeQueueDB entry."""
        return await self.wake(
            agent_id=str(entry.agent_id),
            trigger_type=WakeTriggerType(entry.trigger_type),
            trigger_message=entry.trigger_condition or "Scheduled wake",
            wake_entry=entry,
        )

    async def get_pending_wake_entries(
        self,
        limit: int = 100,
    ) -> list[WakeQueueDB]:
        """
        Return all unprocessed wake entries (any trigger type).
        Used by the orchestrator to process the wake queue.
        """
        result = await self._session.execute(
            select(WakeQueueDB)
            .where(WakeQueueDB.is_processed == False)  # noqa: E712
            .order_by(WakeQueueDB.scheduled_at.asc().nullsfirst())
            .limit(limit)
        )
        return list(result.scalars().all())

    async def archive_agent(self, agent_id: str, reason: str = "") -> AgentDB:
        """
        Permanently archive an agent (SLEEPING/ACTIVE → ARCHIVED).
        Memory is retained per TTL settings.
        """
        result = await self._session.execute(
            select(AgentDB).where(AgentDB.id == agent_id)
        )
        agent = result.scalar_one_or_none()
        if agent is None:
            raise WakeError(f"Agent {agent_id} not found.")

        agent.state = AgentState.ARCHIVED
        agent.archived_at = datetime.utcnow()
        await self._session.flush()

        log.info("agent_archived", agent_id=agent_id, reason=reason)
        return agent
