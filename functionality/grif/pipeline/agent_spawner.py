"""
Step 7: Agent Spawner.
Creates an AgentDB record, persists AgentConfig, sets state = ACTIVE.
The actual LangGraph runtime is initialised in runtime/react_loop.py (Stage 3).
"""

import uuid
from datetime import datetime

import structlog
from sqlalchemy.ext.asyncio import AsyncSession

from grif.models.agent_config import AgentConfig
from grif.models.db import AgentDB, TaskDB
from grif.models.enums import AgentState

log = structlog.get_logger(__name__)


class AgentSpawner:
    """
    Step 7: Persist the fully configured agent to the database.
    Sets state=EMBRYO → ACTIVE after successful creation.
    """

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def spawn(
        self,
        config: AgentConfig,
        task_id: str | None = None,
    ) -> AgentDB:
        """
        Create and persist a new AgentDB record.
        Returns the persisted agent.
        """
        agent = AgentDB(
            id=config.id,
            user_id=config.user_id,
            state=AgentState.EMBRYO,
            task_type=config.task_type,
            blueprint_id=config.blueprint_id,
            config=config.model_dump(mode="json"),
            schedule_cron=config.schedule.cron_expression if config.schedule else None,
            schedule_timezone=config.schedule.timezone if config.schedule else "UTC",
            parent_agent_id=config.parent_agent_id,
        )

        self._session.add(agent)
        await self._session.flush()

        # Transition to ACTIVE
        agent.state = AgentState.ACTIVE
        agent.updated_at = datetime.utcnow()
        await self._session.flush()

        # Link task → agent if task_id provided
        if task_id:
            await self._link_task(task_id, agent.id)

        log.info(
            "agent_spawned",
            agent_id=str(agent.id),
            task_type=agent.task_type,
            blueprint_id=agent.blueprint_id,
            user_id=agent.user_id,
        )
        return agent

    async def spawn_group(
        self,
        configs: list[AgentConfig],
        task_id: str | None = None,
    ) -> list[AgentDB]:
        """Spawn multiple agents for a multi-agent execution plan."""
        agents = []
        for config in configs:
            agent = await self.spawn(config, task_id=task_id)
            agents.append(agent)
        log.info("agent_group_spawned", count=len(agents))
        return agents

    async def _link_task(self, task_id: str, agent_id: uuid.UUID) -> None:
        """Record which task spawned this agent (stored in task's execution_plan)."""
        try:
            from sqlalchemy import select
            result = await self._session.execute(
                select(TaskDB).where(TaskDB.id == task_id)
            )
            task = result.scalar_one_or_none()
            if task and task.execution_plan:
                plan = task.execution_plan or {}
                spawned = plan.setdefault("spawned_agents", [])
                spawned.append(str(agent_id))
                task.execution_plan = plan
                await self._session.flush()
        except Exception as exc:
            log.warning("agent_task_link_failed", error=str(exc))
