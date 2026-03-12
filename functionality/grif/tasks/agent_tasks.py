"""
Celery tasks for GRIF agent lifecycle management.

Tasks:
  - wake_agent_task       : Wake a sleeping agent by ID
  - run_recurring_cycle   : Execute one ReAct cycle for a recurring agent
  - process_wake_queue    : Beat task — scan WakeQueueDB and fire due entries
  - cleanup_archived_agents: Beat task — purge expired archived agents
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from typing import Any

import structlog

from grif.tasks.celery_app import celery_app

log = structlog.get_logger(__name__)


def _run_async(coro: Any) -> Any:
    """Run an async coroutine from a sync Celery task."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(asyncio.run, coro)
                return future.result()
        return loop.run_until_complete(coro)
    except RuntimeError:
        return asyncio.run(coro)


# ── Wake task ─────────────────────────────────────────────────────────────────

@celery_app.task(
    name="grif.tasks.agent_tasks.wake_agent_task",
    bind=True,
    max_retries=3,
    default_retry_delay=30,
)
def wake_agent_task(self, agent_id: str, trigger_message: str = "Scheduled wake") -> dict:
    """
    Wake a sleeping agent and return its context.
    The actual ReAct loop is started by the API layer on receipt of this result.
    """
    async def _wake() -> dict:
        from grif.database import AsyncSessionFactory
        from grif.models.enums import WakeTriggerType
        from grif.runtime.wake_manager import WakeError, WakeManager

        async with AsyncSessionFactory() as session:
            try:
                wm = WakeManager(session)
                ctx = await wm.wake(
                    agent_id=agent_id,
                    trigger_type=WakeTriggerType.SCHEDULE,
                    trigger_message=trigger_message,
                )
                await session.commit()
                log.info("wake_task_done", agent_id=agent_id)
                return {
                    "agent_id": ctx.agent_id,
                    "user_id": ctx.user_id,
                    "trigger_message": ctx.trigger_message,
                    "has_checkpoint": ctx.checkpoint is not None,
                }
            except WakeError as e:
                log.warning("wake_task_failed", agent_id=agent_id, error=str(e))
                return {"agent_id": agent_id, "error": str(e)}

    try:
        return _run_async(_wake())
    except Exception as exc:
        log.error("wake_task_exception", agent_id=agent_id, error=str(exc))
        raise self.retry(exc=exc)


# ── Recurring cycle task ──────────────────────────────────────────────────────

@celery_app.task(
    name="grif.tasks.agent_tasks.run_recurring_cycle",
    bind=True,
    max_retries=2,
    default_retry_delay=60,
)
def run_recurring_cycle(self, agent_id: str, user_message: str = "Scheduled cycle") -> dict:
    """
    Execute one ReAct loop cycle for a recurring agent.
    Wakes the agent, runs the loop, then puts it back to RECURRING (sleeping until next cron).
    """
    async def _cycle() -> dict:
        from sqlalchemy import select

        from grif.database import AsyncSessionFactory
        from grif.llm.gateway import get_gateway
        from grif.models.agent_config import AgentConfig
        from grif.models.db import AgentDB
        from grif.models.enums import AgentState, WakeTriggerType
        from grif.runtime.memory_manager import MemoryManager
        from grif.runtime.react_loop import ReActGraph
        from grif.runtime.recurring_manager import RecurringManager
        from grif.runtime.wake_manager import WakeManager
        from grif.tools.registry import ToolRegistry

        async with AsyncSessionFactory() as session:
            # 1. Wake agent
            wm = WakeManager(session)
            ctx = await wm.wake(
                agent_id=agent_id,
                trigger_type=WakeTriggerType.SCHEDULE,
                trigger_message=user_message,
            )

            # 2. Load agent config
            result = await session.execute(select(AgentDB).where(AgentDB.id == agent_id))
            agent_db = result.scalar_one_or_none()
            if agent_db is None:
                return {"agent_id": agent_id, "error": "Agent not found"}

            config = AgentConfig.model_validate(agent_db.config)
            gateway = get_gateway()

            # 3. Restore memory
            memory = MemoryManager(
                session=session,
                gateway=gateway,
                agent_id=agent_id,
                user_id=ctx.user_id,
            )
            await memory.load_short_term_from_db()

            # 4. Build and run ReAct loop
            registry = ToolRegistry.from_config(config)
            graph = ReActGraph(
                gateway=gateway,
                registry=registry,
                memory=memory,
                agent_config=config,
                session=session,
            )
            final_state = await graph.run(user_message=user_message)

            # 5. Re-register next recurring run
            if agent_db.schedule_cron:
                rm = RecurringManager(session)
                await rm.register(
                    agent_id=agent_id,
                    cron=agent_db.schedule_cron,
                    tz=agent_db.schedule_timezone,
                )

            await session.commit()
            log.info(
                "recurring_cycle_done",
                agent_id=agent_id,
                decision=final_state.get("decision"),
                cycles=final_state.get("cycle_count"),
            )
            return {
                "agent_id": agent_id,
                "decision": final_state.get("decision"),
                "final_result": final_state.get("final_result", ""),
            }

    try:
        return _run_async(_cycle())
    except Exception as exc:
        log.error("recurring_cycle_exception", agent_id=agent_id, error=str(exc))
        raise self.retry(exc=exc)


# ── Process wake queue (beat task) ────────────────────────────────────────────

@celery_app.task(name="grif.tasks.agent_tasks.process_wake_queue")
def process_wake_queue() -> dict:
    """
    Beat task — runs every 60 seconds.
    Scans WakeQueueDB for due entries and dispatches wake_agent_task for each.
    """
    async def _process() -> dict:
        from grif.database import AsyncSessionFactory
        from grif.runtime.sleep_manager import SleepManager
        from grif.runtime.wake_manager import WakeManager

        async with AsyncSessionFactory() as session:
            sm = SleepManager(session)
            due_entries = await sm.get_due_wake_entries(limit=50)

            dispatched = 0
            for entry in due_entries:
                wake_agent_task.apply_async(
                    args=[str(entry.agent_id), entry.trigger_condition or "Scheduled wake"],
                    queue="wake",
                )
                await sm.mark_wake_entry_processed(entry)
                dispatched += 1

            await session.commit()
            log.info("wake_queue_processed", dispatched=dispatched)
            return {"dispatched": dispatched}

    return _run_async(_process())


# ── Cleanup archived agents (beat task) ───────────────────────────────────────

@celery_app.task(name="grif.tasks.agent_tasks.cleanup_archived_agents")
def cleanup_archived_agents() -> dict:
    """
    Beat task — runs every hour.
    Hard-deletes agents that have been archived longer than TTL.
    TTL is configured in settings.sleeping_agent_archive_days (default 90).
    """
    async def _cleanup() -> dict:
        from sqlalchemy import delete, select

        from grif.config import get_settings
        from grif.database import AsyncSessionFactory
        from grif.models.db import AgentDB
        from grif.models.enums import AgentState

        settings = get_settings()
        cutoff = datetime.now(tz=timezone.utc) - timedelta(
            days=settings.sleeping_agent_archive_days
        )

        async with AsyncSessionFactory() as session:
            result = await session.execute(
                select(AgentDB).where(
                    AgentDB.state == AgentState.ARCHIVED,
                    AgentDB.archived_at <= cutoff,
                )
            )
            stale = result.scalars().all()
            count = 0
            for agent in stale:
                await session.delete(agent)
                count += 1

            await session.commit()
            log.info("cleanup_done", deleted=count, cutoff=cutoff.isoformat())
            return {"deleted": count}

    return _run_async(_cleanup())
