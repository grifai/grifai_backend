"""
PriorityQueue — orders agents for execution based on urgency, cost, and state.

No LLM calls — deterministic scoring.

Priority score formula:
    score = urgency_weight + recency_bonus - cost_penalty - queue_age_bonus

Agents with higher scores run first.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from grif.models.enums import AgentState, TaskType, Urgency

# Urgency → base weight
_URGENCY_WEIGHTS: dict[str, float] = {
    Urgency.HIGH: 100.0,
    Urgency.NORMAL: 50.0,
    Urgency.LOW: 10.0,
}

# Task types that get extra priority boost
_PRIORITY_TASK_TYPES = {TaskType.REMIND, TaskType.MONITOR}
_BOOST_PRIORITY = 30.0

# Cost penalty: more expensive agents get bumped down slightly
_COST_PENALTY_FACTOR = 5.0  # per $0.10 of estimated cost


@dataclass(order=True)
class QueueEntry:
    """
    An agent waiting to be executed.
    Negative priority for min-heap → highest priority runs first.
    """
    priority: float = field(compare=True)
    agent_id: str = field(compare=False)
    user_id: str = field(compare=False)
    task_type: str = field(compare=False)
    urgency: str = field(compare=False, default=Urgency.NORMAL)
    estimated_cost_usd: float = field(compare=False, default=0.0)
    enqueued_at: datetime = field(compare=False, default_factory=lambda: datetime.now(tz=timezone.utc))
    metadata: dict[str, Any] = field(compare=False, default_factory=dict)


class AgentPriorityQueue:
    """
    In-process priority queue for agent execution scheduling.

    Usage:
        q = AgentPriorityQueue()
        q.push(agent_id="...", user_id="u1", task_type=TaskType.REMIND, urgency=Urgency.HIGH)
        entry = q.pop()  # highest priority agent
    """

    def __init__(self, max_concurrent: int = 10) -> None:
        self._heap: list[QueueEntry] = []
        self._max_concurrent = max_concurrent
        self._running: set[str] = set()

    def push(
        self,
        agent_id: str,
        user_id: str,
        task_type: TaskType | str,
        urgency: Urgency | str = Urgency.NORMAL,
        estimated_cost_usd: float = 0.0,
        metadata: dict[str, Any] | None = None,
    ) -> QueueEntry:
        """Add an agent to the priority queue. Returns the created entry."""
        score = self._compute_score(task_type, urgency, estimated_cost_usd)
        # Negate for min-heap (higher score = lower heap value = pops first)
        entry = QueueEntry(
            priority=-score,
            agent_id=agent_id,
            user_id=user_id,
            task_type=str(task_type),
            urgency=str(urgency),
            estimated_cost_usd=estimated_cost_usd,
            metadata=metadata or {},
        )
        heapq.heappush(self._heap, entry)
        return entry

    def pop(self) -> QueueEntry | None:
        """Pop the highest-priority agent. Returns None if queue is empty."""
        if not self._heap:
            return None
        entry = heapq.heappop(self._heap)
        self._running.add(entry.agent_id)
        return entry

    def complete(self, agent_id: str) -> None:
        """Mark an agent as finished running (frees up a concurrent slot)."""
        self._running.discard(agent_id)

    def can_run(self) -> bool:
        """True if there's a free concurrent execution slot."""
        return len(self._running) < self._max_concurrent

    def peek(self) -> QueueEntry | None:
        """Peek at the next entry without removing it."""
        return self._heap[0] if self._heap else None

    def size(self) -> int:
        return len(self._heap)

    def running_count(self) -> int:
        return len(self._running)

    def drain_ready(self) -> list[QueueEntry]:
        """
        Pop all agents that can start now (up to max_concurrent limit).
        Used by the orchestrator's tick loop.
        """
        ready: list[QueueEntry] = []
        while self._heap and self.can_run():
            entry = self.pop()
            if entry:
                ready.append(entry)
        return ready

    def _compute_score(
        self,
        task_type: TaskType | str,
        urgency: Urgency | str,
        estimated_cost_usd: float,
    ) -> float:
        score = _URGENCY_WEIGHTS.get(str(urgency), 50.0)

        # Boost high-priority task types
        if str(task_type) in {str(t) for t in _PRIORITY_TASK_TYPES}:
            score += _BOOST_PRIORITY

        # Penalty for expensive agents (prefer cheap quick ones)
        cost_penalty = (estimated_cost_usd / 0.10) * _COST_PENALTY_FACTOR
        score -= cost_penalty

        return max(0.0, score)
