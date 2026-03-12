"""
Step 3: Router.
Fully deterministic — NO LLM calls.
Decides: NEW | EXISTING | FORK | SKIP by checking active → sleeping → archived agents
using Jaccard similarity on keyword sets (threshold from config).
"""

from __future__ import annotations

import re
from typing import Any
from uuid import UUID

import structlog
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from grif.config import get_settings
from grif.models.db import AgentDB
from grif.models.enums import AgentState, RouterDecision
from grif.models.intent import StructuredIntent

log = structlog.get_logger(__name__)
settings = get_settings()

_STOP_WORDS = frozenset(
    "a an the and or but in on at to for of is are was were be been being "
    "have has had do does did will would could should may might must shall "
    "я ты он она мы вы они это эти те то что как где когда".split()
)


def _tokenize(text: str) -> set[str]:
    """Lowercase word tokens, remove stop words and short tokens."""
    tokens = re.findall(r"[a-zа-яё]+", text.lower())
    return {t for t in tokens if len(t) > 2 and t not in _STOP_WORDS}


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    intersection = len(a & b)
    union = len(a | b)
    return intersection / union if union else 0.0


def _intent_to_keywords(intent: StructuredIntent) -> set[str]:
    """Convert intent into a keyword set for similarity comparison."""
    parts: list[str] = [intent.raw_input, intent.task_type]
    if intent.domain:
        parts.append(intent.domain)
    for v in intent.entities.values():
        parts.append(str(v))
    for v in intent.constraints.values():
        parts.append(str(v))
    return _tokenize(" ".join(parts))


def _agent_to_keywords(agent: AgentDB) -> set[str]:
    """Extract keywords from stored AgentDB config."""
    config: dict[str, Any] = agent.config or {}
    parts = [agent.task_type]

    # Try to extract from the config's metadata / raw_input
    meta = config.get("metadata", {})
    if meta.get("raw_input"):
        parts.append(str(meta["raw_input"]))
    if meta.get("domain"):
        parts.append(str(meta["domain"]))

    # Entities stored in metadata
    for v in meta.get("entities", {}).values():
        parts.append(str(v))
    for v in meta.get("constraints", {}).values():
        parts.append(str(v))

    return _tokenize(" ".join(parts))


class RouterResult:
    def __init__(
        self,
        decision: RouterDecision,
        agent_id: UUID | None = None,
        similarity: float = 0.0,
    ) -> None:
        self.decision = decision
        self.agent_id = agent_id      # Set for EXISTING / FORK
        self.similarity = similarity


class Router:
    """
    Step 3: Route incoming intent to one of 4 actions.
    Priority: EXISTING (active) → EXISTING (sleeping) → FORK → NEW.

    EXISTING: similarity ≥ threshold AND same task_type AND same user
    FORK: 0.5 ≤ similarity < threshold — clone and adapt
    NEW: no similar agent found
    SKIP: exact duplicate already running (similarity == 1.0)
    """

    def __init__(self, session: AsyncSession) -> None:
        self._session = session
        self._threshold = settings.router_jaccard_threshold

    async def route(
        self,
        intent: StructuredIntent,
        user_id: str,
    ) -> RouterResult:
        incoming_kw = _intent_to_keywords(intent)
        log.debug("router_incoming_keywords", count=len(incoming_kw))

        # 1. Check ACTIVE agents
        result = await self._find_similar(
            intent, user_id, incoming_kw,
            states=[AgentState.ACTIVE],
        )
        if result:
            return result

        # 2. Check SLEEPING agents
        result = await self._find_similar(
            intent, user_id, incoming_kw,
            states=[AgentState.SLEEPING],
        )
        if result:
            return result

        # 3. Check ARCHIVED agents (for FORK)
        result = await self._find_similar(
            intent, user_id, incoming_kw,
            states=[AgentState.ARCHIVED],
            fork_only=True,
        )
        if result:
            return result

        log.info("router_decision", decision="new", user_id=user_id)
        return RouterResult(decision=RouterDecision.NEW)

    async def _find_similar(
        self,
        intent: StructuredIntent,
        user_id: str,
        incoming_kw: set[str],
        states: list[AgentState],
        fork_only: bool = False,
    ) -> RouterResult | None:
        """
        Query agents in given states, compute Jaccard, return best match.
        """
        result = await self._session.execute(
            select(AgentDB).where(
                AgentDB.user_id == user_id,
                AgentDB.task_type == intent.task_type,
                AgentDB.state.in_([s.value for s in states]),
            )
        )
        agents = result.scalars().all()

        best: tuple[float, AgentDB] | None = None
        for agent in agents:
            agent_kw = _agent_to_keywords(agent)
            sim = _jaccard(incoming_kw, agent_kw)
            if best is None or sim > best[0]:
                best = (sim, agent)

        if best is None:
            return None

        sim, agent = best
        log.debug(
            "router_similarity",
            agent_id=str(agent.id),
            sim=round(sim, 3),
            state=agent.state,
        )

        if fork_only:
            if sim >= 0.5:
                log.info("router_decision", decision="fork", similarity=round(sim, 3))
                return RouterResult(RouterDecision.FORK, agent.id, sim)
            return None

        if sim >= 1.0:
            log.info("router_decision", decision="skip", agent_id=str(agent.id))
            return RouterResult(RouterDecision.SKIP, agent.id, sim)

        if sim >= self._threshold:
            log.info(
                "router_decision",
                decision="existing",
                agent_id=str(agent.id),
                similarity=round(sim, 3),
            )
            return RouterResult(RouterDecision.EXISTING, agent.id, sim)

        if sim >= 0.5:
            log.info("router_decision", decision="fork", similarity=round(sim, 3))
            return RouterResult(RouterDecision.FORK, agent.id, sim)

        return None
