"""
Three-tier Memory Manager.

Short-term  : last N ReAct cycle logs (in-process list, no DB)
Working     : LLM-summarised context from short-term (1 Summarizer call)
Long-term   : RAG via pgvector — stored in agent_memory table

The working memory is the "injected context" that goes into every LLM call
in the ReAct loop as an additional system message.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import structlog
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from grif.llm.gateway import LLMGateway
from grif.models.db import AgentMemoryDB
from grif.models.enums import MemoryType
from grif.models.memory import (
    DecisionMemory,
    FactMemory,
    PreferenceMemory,
    ProductionMemory,
    ReActCycleLog,
)

if TYPE_CHECKING:
    pass

log = structlog.get_logger(__name__)

_SHORT_TERM_SIZE = 5     # Keep last N cycles in-memory
_WORKING_THRESHOLD = 10  # Summarise when short-term exceeds this


class MemoryManager:
    """
    Manages all three memory tiers for one agent instance.

    Usage:
        mm = MemoryManager(session, gateway, agent_id, user_id)
        context = await mm.get_working_context()   # inject into LLM
        await mm.add_cycle_log(log_entry)
        await mm.store_fact(fact)
        facts = await mm.retrieve_relevant(query_text)
    """

    def __init__(
        self,
        session: AsyncSession,
        gateway: LLMGateway,
        agent_id: str,
        user_id: str,
    ) -> None:
        self._session = session
        self._gateway = gateway
        self._agent_id = agent_id
        self._user_id = user_id

        # Short-term: in-process list
        self._short_term: list[ReActCycleLog] = []
        # Working: cached summary string
        self._working_memory: str = ""

    # ── Short-term ────────────────────────────────────────────────────────────

    async def add_cycle_log(self, log_entry: ReActCycleLog) -> None:
        """Add a ReAct cycle to short-term memory and persist to DB."""
        self._short_term.append(log_entry)
        # Keep only last N in memory
        if len(self._short_term) > _SHORT_TERM_SIZE:
            self._short_term = self._short_term[-_SHORT_TERM_SIZE:]

        # Persist to agent_logs (audit trail) — done by react_loop.py separately
        # If short-term exceeded threshold: refresh working memory
        if len(self._short_term) >= _WORKING_THRESHOLD:
            await self._refresh_working_memory()

    def get_recent_cycles(self, n: int = _SHORT_TERM_SIZE) -> list[ReActCycleLog]:
        return self._short_term[-n:]

    # ── Working memory (LLM summarisation) ───────────────────────────────────

    async def get_working_context(self) -> str:
        """
        Return the current working memory context string.
        If empty, build it from short-term + long-term retrieval.
        """
        if not self._working_memory:
            await self._refresh_working_memory()
        return self._working_memory

    async def _refresh_working_memory(self) -> None:
        """
        Summarise short-term cycles into a compact working memory string.
        Uses SUMMARIZER model (GPT-4o-mini). Only called when needed.
        """
        if not self._short_term:
            self._working_memory = ""
            return

        cycles_text = "\n".join(
            f"Cycle {c.cycle_number}: thought={c.thought[:100]} | "
            f"action={c.action} | obs={c.observation[:100]} | decision={c.decision}"
            for c in self._short_term
        )

        try:
            response = await self._gateway.complete(
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Summarise the following agent activity into a compact memory context "
                            "(max 200 words). Focus on: what was found, what decisions were made, "
                            "what is still pending."
                        ),
                    },
                    {"role": "user", "content": cycles_text},
                ],
                purpose="summarizer",
                temperature=0.1,
                max_tokens=300,
            )
            self._working_memory = response.content
            log.debug("working_memory_refreshed", length=len(self._working_memory))
        except Exception as exc:
            log.warning("working_memory_refresh_failed", error=str(exc))
            # Fallback: use raw cycles text
            self._working_memory = cycles_text[:500]

    def inject_working_memory(self, messages: list[dict]) -> list[dict]:
        """
        Prepend working memory context as a system message if non-empty.
        Mutates and returns the messages list.
        """
        if self._working_memory:
            return [
                {
                    "role": "system",
                    "content": f"[Memory Context]\n{self._working_memory}",
                },
                *messages,
            ]
        return messages

    # ── Long-term (pgvector RAG) ──────────────────────────────────────────────

    async def store_fact(self, fact: FactMemory) -> None:
        await self._store_memory(MemoryType.FACT, fact.model_dump(mode="json"))

    async def store_decision(self, decision: DecisionMemory) -> None:
        await self._store_memory(MemoryType.DECISION, decision.model_dump(mode="json"))

    async def store_preference(self, pref: PreferenceMemory) -> None:
        await self._store_memory(MemoryType.PREFERENCE, pref.model_dump(mode="json"))

    async def update_production_memory(self, production: ProductionMemory) -> None:
        """Upsert production memory for this agent (only one per agent)."""
        result = await self._session.execute(
            select(AgentMemoryDB).where(
                AgentMemoryDB.agent_id == self._agent_id,
                AgentMemoryDB.memory_type == MemoryType.PRODUCTION,
            )
        )
        existing = result.scalar_one_or_none()
        if existing:
            existing.content = production.model_dump(mode="json")
        else:
            await self._store_memory(
                MemoryType.PRODUCTION, production.model_dump(mode="json")
            )
        await self._session.flush()

    async def get_production_memory(self) -> ProductionMemory | None:
        result = await self._session.execute(
            select(AgentMemoryDB).where(
                AgentMemoryDB.agent_id == self._agent_id,
                AgentMemoryDB.memory_type == MemoryType.PRODUCTION,
            )
        )
        row = result.scalar_one_or_none()
        if row:
            return ProductionMemory.model_validate(row.content)
        return None

    async def retrieve_relevant(
        self,
        query_text: str,
        limit: int = 5,
        memory_types: list[MemoryType] | None = None,
    ) -> list[dict]:
        """
        Retrieve relevant memories using text similarity (exact match fallback).
        Full vector search requires embedding generation — deferred to production.
        """
        types = [m.value for m in (memory_types or [MemoryType.FACT, MemoryType.DECISION])]

        result = await self._session.execute(
            select(AgentMemoryDB)
            .where(
                AgentMemoryDB.agent_id == self._agent_id,
                AgentMemoryDB.memory_type.in_(types),
            )
            .order_by(AgentMemoryDB.created_at.desc())
            .limit(limit)
        )
        rows = result.scalars().all()
        return [r.content for r in rows]

    async def _store_memory(
        self, memory_type: MemoryType, content: dict
    ) -> AgentMemoryDB:
        entry = AgentMemoryDB(
            agent_id=self._agent_id,
            user_id=self._user_id,
            memory_type=memory_type.value,
            content=content,
        )
        self._session.add(entry)
        await self._session.flush()
        return entry

    async def load_short_term_from_db(self, last_n: int = _SHORT_TERM_SIZE) -> None:
        """
        Restore short-term memory from DB logs (used after agent wake-up).
        """
        from grif.models.db import AgentLogDB
        result = await self._session.execute(
            select(AgentLogDB)
            .where(AgentLogDB.agent_id == self._agent_id)
            .order_by(AgentLogDB.created_at.desc())
            .limit(last_n)
        )
        rows = list(reversed(result.scalars().all()))
        self._short_term = [
            ReActCycleLog(
                cycle_number=r.cycle_number,
                thought=r.thought,
                action=r.action,
                action_input=r.action_input or {},
                observation=r.observation,
                decision=r.decision,
                tokens_used=r.tokens_used,
            )
            for r in rows
        ]
        log.debug("short_term_restored_from_db", count=len(self._short_term))
