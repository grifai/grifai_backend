"""Request and response Pydantic schemas for Stage 6 API."""
from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


# ─── Tasks ────────────────────────────────────────────────────────────────────

class SubmitTaskRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=8000)
    signal_type: str = Field(default="text")
    metadata: dict[str, Any] = Field(default_factory=dict)


class ClarificationQuestionSchema(BaseModel):
    field_name: str
    question: str
    options: list[str] | None = None


class TaskResponse(BaseModel):
    task_id: str
    status: str  # pending | clarifying | planning | running | done | failed
    agent_ids: list[str] = Field(default_factory=list)
    clarification_questions: list[ClarificationQuestionSchema] | None = None
    estimated_cost: dict[str, Any] | None = None
    message: str | None = None


class ClarificationAnswerItem(BaseModel):
    field_name: str
    value: Any


class AnswerClarificationRequest(BaseModel):
    answers: list[ClarificationAnswerItem]


# ─── Agents ───────────────────────────────────────────────────────────────────

class AgentSummary(BaseModel):
    agent_id: str
    user_id: str
    state: str
    task_type: str
    blueprint_id: str | None = None
    avg_score: float | None = None
    eval_count: int = 0
    created_at: datetime
    updated_at: datetime

    @classmethod
    def from_db(cls, agent: Any) -> "AgentSummary":
        return cls(
            agent_id=str(agent.id),
            user_id=agent.user_id,
            state=agent.state,
            task_type=agent.task_type,
            blueprint_id=agent.blueprint_id,
            avg_score=agent.avg_score,
            eval_count=agent.eval_count or 0,
            created_at=agent.created_at,
            updated_at=agent.updated_at,
        )


class WakeAgentRequest(BaseModel):
    message: str | None = None


class WakeAgentResponse(BaseModel):
    agent_id: str
    previous_state: str
    new_state: str
    context_summary: str | None = None


class ApproveToolRequest(BaseModel):
    tool_name: str
    approved: bool


class ApproveToolResponse(BaseModel):
    agent_id: str
    tool_name: str
    approved: bool
    message: str


class AgentExplanationResponse(BaseModel):
    agent_id: str
    explanation: str
    last_action: str | None = None
    last_cycle: int | None = None


# ─── WebSocket events ─────────────────────────────────────────────────────────

class ProgressEvent(BaseModel):
    task_id: str
    event_type: str
    data: dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
