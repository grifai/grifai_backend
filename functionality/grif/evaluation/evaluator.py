"""
SelfEvaluator — scores agent task results after completion.

One authorised Haiku LLM call (4th LLM call point in architecture).
Scores 1-5 across three dimensions: accuracy, completeness, efficiency.
Updates AgentDB.avg_score and AgentDB.eval_count.
Updates BlueprintScoreDB for blueprint improvement feedback loop.
"""

from __future__ import annotations

from typing import Any

import structlog
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from grif.llm.gateway import LLMGateway
from grif.models.db import AgentDB, BlueprintScoreDB

log = structlog.get_logger(__name__)

_SYSTEM_PROMPT = (
    "You are a strict evaluator of AI agent outputs. "
    "Score the agent's result on three criteria (1-5 each):\n"
    "- accuracy: Is the result factually correct and on-target?\n"
    "- completeness: Does it fully address the user's request?\n"
    "- efficiency: Did the agent accomplish this without wasted steps?\n\n"
    "Respond in JSON ONLY:\n"
    '{"accuracy": <1-5>, "completeness": <1-5>, "efficiency": <1-5>, "comment": "<one sentence>"}'
)


class EvaluationResult:
    def __init__(
        self,
        accuracy: float,
        completeness: float,
        efficiency: float,
        comment: str,
        agent_id: str,
    ) -> None:
        self.accuracy = accuracy
        self.completeness = completeness
        self.efficiency = efficiency
        self.comment = comment
        self.agent_id = agent_id

    @property
    def overall(self) -> float:
        return round((self.accuracy + self.completeness + self.efficiency) / 3, 2)

    def __repr__(self) -> str:
        return (
            f"EvaluationResult(overall={self.overall}, "
            f"accuracy={self.accuracy}, completeness={self.completeness}, "
            f"efficiency={self.efficiency})"
        )


class SelfEvaluator:
    """
    Evaluates a completed agent task.
    Makes 1 Haiku LLM call; updates DB score records.

    Usage:
        evaluator = SelfEvaluator(gateway, session)
        result = await evaluator.evaluate(
            agent_id="...",
            user_request="Find hotels in Paris under €150",
            agent_result="Found 5 hotels: ...",
        )
    """

    def __init__(self, gateway: LLMGateway, session: AsyncSession) -> None:
        self._gateway = gateway
        self._session = session

    async def evaluate(
        self,
        agent_id: str,
        user_request: str,
        agent_result: str,
        user_id: str = "system",
    ) -> EvaluationResult:
        """
        Score agent result. Makes 1 Haiku call.
        Falls back to score=3 on LLM failure.
        Persists scores to AgentDB and BlueprintScoreDB.
        """
        user_message = (
            f"User request: {user_request}\n\n"
            f"Agent result:\n{agent_result[:2000]}"
        )

        scores: dict[str, Any] = {}
        try:
            response = await self._gateway.complete_json(
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ],
                purpose="self_evaluation",
                user_id=user_id,
            )
            scores = response.parsed or {}
        except Exception as exc:
            log.warning("self_evaluator_llm_failed", agent_id=agent_id, error=str(exc))

        eval_result = EvaluationResult(
            accuracy=float(scores.get("accuracy", 3)),
            completeness=float(scores.get("completeness", 3)),
            efficiency=float(scores.get("efficiency", 3)),
            comment=str(scores.get("comment", "Evaluated without LLM (fallback).")),
            agent_id=agent_id,
        )

        # Clamp to [1, 5]
        eval_result.accuracy = max(1.0, min(5.0, eval_result.accuracy))
        eval_result.completeness = max(1.0, min(5.0, eval_result.completeness))
        eval_result.efficiency = max(1.0, min(5.0, eval_result.efficiency))

        await self._persist_scores(agent_id, eval_result, user_id)

        log.info(
            "agent_evaluated",
            agent_id=agent_id,
            overall=eval_result.overall,
            comment=eval_result.comment,
        )
        return eval_result

    async def _persist_scores(
        self,
        agent_id: str,
        result: EvaluationResult,
        user_id: str,
    ) -> None:
        try:
            # Update AgentDB rolling average
            db_result = await self._session.execute(
                select(AgentDB).where(AgentDB.id == agent_id)
            )
            agent = db_result.scalar_one_or_none()
            if agent:
                prev_avg = agent.avg_score or 0.0
                prev_count = agent.eval_count or 0
                new_count = prev_count + 1
                agent.avg_score = round(
                    (prev_avg * prev_count + result.overall) / new_count, 2
                )
                agent.eval_count = new_count

                # Persist to BlueprintScoreDB for blueprint feedback loop
                if agent.blueprint_id:
                    score_entry = BlueprintScoreDB(
                        blueprint_id=agent.blueprint_id,
                        user_id=user_id,
                        score=result.overall,
                        agent_id=agent.id,
                    )
                    self._session.add(score_entry)

            await self._session.flush()
        except Exception as exc:
            log.warning("eval_score_persist_failed", agent_id=agent_id, error=str(exc))
