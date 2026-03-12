"""
AuditLogger — structured audit trail for all agent actions.

Wraps structlog with domain-specific log events.
Every significant event is logged with: user_id, agent_id, action, metadata.

Also provides human-readable explanations of agent decisions
(uses 1 Haiku call — authorised LLM point under 'explainer' purpose).
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import structlog

log = structlog.get_logger(__name__)


class AuditEvent:
    """Standardised audit event constants."""
    # Pipeline
    TASK_RECEIVED = "task_received"
    INTENT_CLASSIFIED = "intent_classified"
    CLARIFICATION_REQUESTED = "clarification_requested"
    AGENT_ROUTED = "agent_routed"
    PLAN_CREATED = "plan_created"
    AGENT_SPAWNED = "agent_spawned"
    # Runtime
    REACT_STARTED = "react_started"
    REACT_CYCLE = "react_cycle"
    TOOL_CALLED = "tool_called"
    TOOL_BLOCKED = "tool_blocked"
    PERMISSION_ESCALATED = "permission_escalated"
    REACT_FINISHED = "react_finished"
    # State transitions
    AGENT_SLEPT = "agent_slept"
    AGENT_WOKE = "agent_woke"
    AGENT_ARCHIVED = "agent_archived"
    # Evaluation
    AGENT_EVALUATED = "agent_evaluated"
    # Error
    AGENT_ESCALATED = "agent_escalated"
    TASK_FAILED = "task_failed"


class AuditLogger:
    """
    Structured event logger for the GRIF agent lifecycle.

    Usage:
        audit = AuditLogger(user_id="u1", agent_id="agent-123")
        audit.log(AuditEvent.TOOL_CALLED, tool="web_search", query="hotels Paris")
        audit.error(AuditEvent.TASK_FAILED, reason="LLM timeout")
    """

    def __init__(
        self,
        user_id: str,
        agent_id: str | None = None,
        task_id: str | None = None,
    ) -> None:
        self._base = {
            "user_id": user_id,
            "agent_id": agent_id or "",
            "task_id": task_id or "",
        }

    def log(self, event: str, **kwargs: Any) -> None:
        log.info(event, **self._base, **kwargs, ts=_now())

    def debug(self, event: str, **kwargs: Any) -> None:
        log.debug(event, **self._base, **kwargs, ts=_now())

    def warning(self, event: str, **kwargs: Any) -> None:
        log.warning(event, **self._base, **kwargs, ts=_now())

    def error(self, event: str, **kwargs: Any) -> None:
        log.error(event, **self._base, **kwargs, ts=_now())

    def tool_called(self, tool_name: str, args: dict[str, Any]) -> None:
        self.log(AuditEvent.TOOL_CALLED, tool=tool_name, args=_truncate_args(args))

    def tool_blocked(self, tool_name: str, reason: str) -> None:
        self.warning(AuditEvent.TOOL_BLOCKED, tool=tool_name, reason=reason)

    def react_cycle(
        self,
        cycle: int,
        action: str,
        decision: str,
        tokens_used: int = 0,
    ) -> None:
        self.log(
            AuditEvent.REACT_CYCLE,
            cycle=cycle,
            action=action,
            decision=decision,
            tokens_used=tokens_used,
        )

    def agent_evaluated(self, overall_score: float, comment: str) -> None:
        self.log(AuditEvent.AGENT_EVALUATED, score=overall_score, comment=comment)


# ── Explainer (LLM-based human-readable explanation) ──────────────────────────

class AgentExplainer:
    """
    Generates human-readable explanations of agent decisions.
    One Haiku call per explanation (authorised 'explainer' purpose).

    Usage:
        explainer = AgentExplainer(gateway)
        text = await explainer.explain(
            action="web_search",
            context="User asked for hotels in Paris under €150",
            result="Found 5 hotels: ...",
        )
    """

    _SYSTEM_PROMPT = (
        "You are an assistant explaining AI agent actions in plain language. "
        "Given an agent action and its result, write a brief (1-2 sentence) "
        "human-readable explanation of what happened and why. "
        "Be concise and non-technical."
    )

    def __init__(self, gateway: Any) -> None:
        self._gateway = gateway

    async def explain(
        self,
        action: str,
        context: str,
        result: str,
        user_id: str = "system",
        language: str = "ru",
    ) -> str:
        """Generate explanation. Falls back to a template string on error."""
        prompt = (
            f"Action: {action}\n"
            f"Context: {context[:300]}\n"
            f"Result: {result[:500]}\n\n"
            f"Explain in {language}."
        )
        try:
            response = await self._gateway.complete(
                messages=[
                    {"role": "system", "content": self._SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                purpose="explainer",
                temperature=0.2,
                max_tokens=100,
                user_id=user_id,
            )
            return response.content.strip()
        except Exception as exc:
            log.warning("explainer_llm_failed", error=str(exc))
            return f"Agent performed '{action}' and produced a result."


# ── Helpers ───────────────────────────────────────────────────────────────────

def _now() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


def _truncate_args(args: dict[str, Any], max_len: int = 200) -> dict[str, Any]:
    return {
        k: (str(v)[:max_len] + "…" if len(str(v)) > max_len else v)
        for k, v in args.items()
    }
