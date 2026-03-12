"""
ConflictResolver — resolves output conflicts between parallel agents.

Uses 1 Sonnet LLM call (authorised LLM call point in architecture).
Called only when two parallel agents produce contradictory results.
"""

from __future__ import annotations

import json
from typing import Any

import structlog

from grif.llm.gateway import LLMGateway

log = structlog.get_logger(__name__)


class ConflictResolution:
    """Result of conflict resolution."""

    def __init__(
        self,
        winning_result: str,
        explanation: str,
        agent_a_role: str,
        agent_b_role: str,
        merged: bool = False,
    ) -> None:
        self.winning_result = winning_result
        self.explanation = explanation
        self.agent_a_role = agent_a_role
        self.agent_b_role = agent_b_role
        self.merged = merged  # True if both results were merged rather than one chosen


class ConflictResolver:
    """
    Resolves conflicts between two parallel agents that produced different results.

    One Sonnet call is made to:
    1. Understand the nature of the conflict
    2. Select the better result (or merge both)
    3. Explain the decision

    Usage:
        resolver = ConflictResolver(gateway)
        resolution = await resolver.resolve(
            task_description="Compare hotel prices in Paris",
            agent_a_role="researcher_a",
            agent_a_result="Hotel A: €150/night, excellent location",
            agent_b_role="researcher_b",
            agent_b_result="Hotel A: €180/night, good location",
        )
    """

    _SYSTEM_PROMPT = (
        "You are a conflict resolver for a multi-agent AI system. "
        "Two agents have produced results that contradict each other. "
        "Your job is to:\n"
        "1. Identify why the results differ\n"
        "2. Select the more accurate / complete result, OR merge both if complementary\n"
        "3. Provide a brief explanation\n\n"
        "Respond in JSON:\n"
        '{"winning_result": "<final result>", "explanation": "<why>", "merged": <true|false>}'
    )

    def __init__(self, gateway: LLMGateway) -> None:
        self._gateway = gateway

    async def resolve(
        self,
        task_description: str,
        agent_a_role: str,
        agent_a_result: str,
        agent_b_role: str,
        agent_b_result: str,
        user_id: str = "system",
    ) -> ConflictResolution:
        """
        Resolve a conflict between two agents. Makes 1 Sonnet LLM call.
        Falls back to agent_a result on LLM error.
        """
        user_message = (
            f"Task: {task_description}\n\n"
            f"Agent '{agent_a_role}' result:\n{agent_a_result}\n\n"
            f"Agent '{agent_b_role}' result:\n{agent_b_result}\n\n"
            "Resolve this conflict."
        )

        try:
            response = await self._gateway.complete_json(
                messages=[
                    {"role": "system", "content": self._SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ],
                purpose="conflict_resolver",
                user_id=user_id,
            )
            data: dict[str, Any] = response.parsed or {}
            winning = str(data.get("winning_result", agent_a_result))
            explanation = str(data.get("explanation", "Selected based on completeness."))
            merged = bool(data.get("merged", False))

            log.info(
                "conflict_resolved",
                agent_a=agent_a_role,
                agent_b=agent_b_role,
                merged=merged,
            )
            return ConflictResolution(
                winning_result=winning,
                explanation=explanation,
                agent_a_role=agent_a_role,
                agent_b_role=agent_b_role,
                merged=merged,
            )

        except Exception as exc:
            log.warning("conflict_resolver_llm_failed", error=str(exc))
            # Fallback: prefer agent_a result
            return ConflictResolution(
                winning_result=agent_a_result,
                explanation=f"LLM unavailable; defaulting to {agent_a_role}.",
                agent_a_role=agent_a_role,
                agent_b_role=agent_b_role,
                merged=False,
            )

    async def resolve_multi(
        self,
        task_description: str,
        results: dict[str, str],
        user_id: str = "system",
    ) -> ConflictResolution:
        """
        Resolve conflicts among more than two agents.
        Iterates pairwise and returns final merged result.
        """
        roles = list(results.keys())
        if len(roles) < 2:
            role = roles[0] if roles else "unknown"
            return ConflictResolution(
                winning_result=results.get(role, ""),
                explanation="Only one agent — no conflict.",
                agent_a_role=role,
                agent_b_role=role,
            )

        current_winner = roles[0]
        current_result = results[roles[0]]

        for i in range(1, len(roles)):
            next_role = roles[i]
            resolution = await self.resolve(
                task_description=task_description,
                agent_a_role=current_winner,
                agent_a_result=current_result,
                agent_b_role=next_role,
                agent_b_result=results[next_role],
                user_id=user_id,
            )
            current_result = resolution.winning_result
            current_winner = f"merged({current_winner},{next_role})" if resolution.merged else current_winner

        return ConflictResolution(
            winning_result=current_result,
            explanation="Multi-agent conflict resolved iteratively.",
            agent_a_role=roles[0],
            agent_b_role=roles[-1],
            merged=True,
        )
