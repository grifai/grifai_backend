"""
Step 4: Blueprint Selector.
Deterministic selection with user-scored personalisation.
Personal blueprints > global blueprints.
"""

import structlog
from sqlalchemy.ext.asyncio import AsyncSession

from grif.blueprints.registry import Blueprint, BlueprintRegistry
from grif.models.intent import StructuredIntent

log = structlog.get_logger(__name__)


class BlueprintSelector:
    """Step 4: select the best blueprint for a classified intent."""

    def __init__(self, session: AsyncSession) -> None:
        self._registry = BlueprintRegistry(session)

    async def select(
        self,
        intent: StructuredIntent,
        user_id: str,
    ) -> Blueprint:
        """
        Select blueprint by task_type + domain.
        Priority: personal (highest scored) > global (best match).
        """
        blueprint = await self._registry.find_best(
            task_type=intent.task_type,
            domain=intent.domain,
            user_id=user_id,
        )
        log.info(
            "blueprint_selected",
            blueprint_id=blueprint.id,
            task_type=intent.task_type,
            domain=intent.domain,
            user_id=user_id,
        )
        return blueprint

    def get_registry(self) -> BlueprintRegistry:
        return self._registry
