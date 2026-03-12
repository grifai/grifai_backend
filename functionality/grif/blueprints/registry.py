"""
Blueprint Registry.
Loads JSON definitions from blueprints/definitions/.
Supports personal (per-user) blueprints with higher priority.
Implements per-user scoring for Agent Specialization Learning (mechanic #20).
"""

import json
from pathlib import Path
from typing import Any

import structlog
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from grif.models.db import BlueprintDB, BlueprintScoreDB

log = structlog.get_logger(__name__)

_DEFINITIONS_DIR = Path(__file__).parent / "definitions"


class Blueprint:
    """In-memory blueprint representation."""

    def __init__(self, data: dict[str, Any]) -> None:
        self.id: str = data["id"]
        self.name: str = data["name"]
        self.task_types: list[str] = data.get("task_types", [])
        self.domains: list[str] = data.get("domains", [])
        self.description: str = data.get("description", "")
        self.template: str = data.get("template", "generic_worker")
        self.default_tools: list[str] = data.get("default_tools", ["web_search"])
        self.default_model: str = data.get("default_model", "claude-sonnet-4-6")
        self.typical_complexity: str = data.get("typical_complexity", "simple")
        self.plan_pattern: str = data.get("plan_pattern", "pipeline")
        self.typical_agents: list[dict] = data.get("typical_agents", [])
        self.required_fields: list[str] = data.get("required_fields", [])
        self.optional_fields: list[str] = data.get("optional_fields", [])
        self.is_personal: bool = data.get("is_personal", False)
        self.owner_user_id: str | None = data.get("owner_user_id")
        self._raw = data

    def to_dict(self) -> dict[str, Any]:
        return self._raw


# ─── Global blueprint cache ────────────────────────────────────────────────────

_global_blueprints: dict[str, Blueprint] = {}


def _load_global_blueprints() -> dict[str, Blueprint]:
    """Load all JSON definitions from disk into memory. Called once at startup."""
    global _global_blueprints
    if _global_blueprints:
        return _global_blueprints

    for path in _DEFINITIONS_DIR.glob("*.json"):
        with path.open(encoding="utf-8") as f:
            data = json.load(f)
        bp = Blueprint(data)
        _global_blueprints[bp.id] = bp
        log.debug("blueprint_loaded", id=bp.id, task_types=bp.task_types)

    log.info("blueprints_loaded", count=len(_global_blueprints))
    return _global_blueprints


# ─── Registry class ────────────────────────────────────────────────────────────

class BlueprintRegistry:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session
        _load_global_blueprints()

    # ── Selection ──────────────────────────────────────────────────────────────

    async def find_best(
        self,
        task_type: str,
        domain: str | None,
        user_id: str,
    ) -> Blueprint:
        """
        Find the best matching blueprint.
        Priority: personal (per-user, highest score) > global (best match).
        """
        # 1. Try personal blueprints first
        personal = await self._get_personal_blueprints(user_id, task_type)
        if personal:
            best_personal = await self._rank_by_score(personal, user_id)
            if best_personal:
                log.info("blueprint_selected_personal", id=best_personal.id, user_id=user_id)
                return best_personal

        # 2. Fall back to global blueprints
        candidates = self._filter_global(task_type, domain)
        if not candidates:
            # Ultimate fallback
            return _global_blueprints["generic_worker"]

        best = await self._rank_by_score(candidates, user_id)
        log.info("blueprint_selected_global", id=best.id, task_type=task_type, domain=domain)
        return best

    def _filter_global(
        self,
        task_type: str,
        domain: str | None,
    ) -> list[Blueprint]:
        """
        Filter global blueprints by task_type and domain.
        Exact domain match > task_type only > generic_worker.
        """
        all_bp = list(_global_blueprints.values())
        type_matches = [bp for bp in all_bp if task_type in bp.task_types]

        if not type_matches:
            return [_global_blueprints["generic_worker"]]

        if domain and domain != "general":
            domain_matches = [bp for bp in type_matches if domain in bp.domains]
            if domain_matches:
                # Prefer specific (fewer domains) over generic
                non_generic = [bp for bp in domain_matches if bp.id != "generic_worker"]
                return non_generic or domain_matches

        # Exclude generic_worker if there are specific matches
        specific = [bp for bp in type_matches if bp.id != "generic_worker"]
        # Sort: blueprints whose id starts with the task_type are preferred first,
        # then by domain specificity (fewer domains = more specific)
        specific.sort(key=lambda bp: (0 if task_type in bp.id else 1, len(bp.domains)))
        return specific or type_matches

    async def _get_personal_blueprints(
        self,
        user_id: str,
        task_type: str,
    ) -> list[Blueprint]:
        """Load personal blueprints from DB for this user."""
        result = await self._session.execute(
            select(BlueprintDB).where(
                BlueprintDB.is_personal.is_(True),
                BlueprintDB.owner_user_id == user_id,
                BlueprintDB.task_type == task_type,
            )
        )
        rows = result.scalars().all()
        return [Blueprint(row.definition) for row in rows]

    async def _rank_by_score(
        self,
        candidates: list[Blueprint],
        user_id: str,
    ) -> Blueprint:
        """Return the highest-scored blueprint for this user. Falls back to first."""
        if len(candidates) == 1:
            return candidates[0]

        ids = [bp.id for bp in candidates]
        result = await self._session.execute(
            select(
                BlueprintScoreDB.blueprint_id,
                # avg score descending
                __import__("sqlalchemy").func.avg(BlueprintScoreDB.score).label("avg_score"),
            )
            .where(
                BlueprintScoreDB.user_id == user_id,
                BlueprintScoreDB.blueprint_id.in_(ids),
            )
            .group_by(BlueprintScoreDB.blueprint_id)
            .order_by(__import__("sqlalchemy").text("avg_score DESC"))
        )
        rows = result.all()
        if rows:
            best_id = rows[0].blueprint_id
            for bp in candidates:
                if bp.id == best_id:
                    return bp

        return candidates[0]

    # ── Scoring ────────────────────────────────────────────────────────────────

    async def record_score(
        self,
        blueprint_id: str,
        user_id: str,
        score: float,
        agent_id: str | None = None,
    ) -> None:
        """Record evaluation score for a blueprint (called from Self-Evaluation)."""
        entry = BlueprintScoreDB(
            blueprint_id=blueprint_id,
            user_id=user_id,
            score=score,
            agent_id=agent_id,
        )
        self._session.add(entry)
        await self._session.flush()

    async def get_avg_score(self, blueprint_id: str, user_id: str) -> float | None:
        """Get average score for a blueprint per user."""
        from sqlalchemy import func, select
        result = await self._session.execute(
            select(func.avg(BlueprintScoreDB.score)).where(
                BlueprintScoreDB.blueprint_id == blueprint_id,
                BlueprintScoreDB.user_id == user_id,
            )
        )
        val = result.scalar_one_or_none()
        return float(val) if val is not None else None

    # ── Lookup ─────────────────────────────────────────────────────────────────

    def get_by_id(self, blueprint_id: str) -> Blueprint | None:
        return _global_blueprints.get(blueprint_id)

    def list_all(self) -> list[Blueprint]:
        return list(_global_blueprints.values())
