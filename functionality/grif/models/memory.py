from datetime import datetime
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


# ─── Atomic memory entries ────────────────────────────────────────────────────

class FactMemory(BaseModel):
    """A factual piece of information the agent has learned."""

    id: UUID = Field(default_factory=uuid4)
    key: str
    value: Any
    source: str | None = None
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    created_at: datetime = Field(default_factory=datetime.utcnow)


class DecisionMemory(BaseModel):
    """A decision the agent made, with the reasoning behind it."""

    id: UUID = Field(default_factory=uuid4)
    decision: str
    reasoning: str
    alternatives_considered: list[str] = Field(default_factory=list)
    context: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)


class PreferenceMemory(BaseModel):
    """User preference inferred or explicitly stated."""

    id: UUID = Field(default_factory=uuid4)
    category: str = Field(description="E.g. 'communication_style', 'budget', 'time'")
    preference: str
    strength: float = Field(default=0.5, ge=0.0, le=1.0, description="0=weak, 1=strong")
    created_at: datetime = Field(default_factory=datetime.utcnow)


# ─── Production Memory (for Recurring / OPERATE agents) ──────────────────────

class ReleaseRecord(BaseModel):
    """One unit of published content or executed action."""

    id: UUID = Field(default_factory=uuid4)
    title: str
    content_type: str  # post, email, report, action, etc.
    channel: str | None = None
    published_at: datetime = Field(default_factory=datetime.utcnow)
    metrics: dict[str, Any] = Field(
        default_factory=dict,
        description="Views, opens, conversions — filled by Feedback Loop.",
    )


class ContentPlanItem(BaseModel):
    topic: str
    planned_date: datetime | None = None
    status: str = "planned"  # planned, in_progress, published, cancelled


class StyleEvolutionEntry(BaseModel):
    changed_at: datetime = Field(default_factory=datetime.utcnow)
    parameter: str
    old_value: Any
    new_value: Any
    reason: str | None = None


class EffectivenessMetrics(BaseModel):
    """Aggregated performance metrics for self-improvement."""

    total_published: int = 0
    avg_engagement: float = 0.0
    best_topics: list[str] = Field(default_factory=list)
    worst_topics: list[str] = Field(default_factory=list)
    last_updated: datetime = Field(default_factory=datetime.utcnow)


class ProductionMemory(BaseModel):
    """
    Extended memory for Recurring / OPERATE agents.
    Tracks release history, content plan, style evolution, effectiveness.
    Stored alongside Fact/Decision/Preference in agent_memory table (memory_type=production).
    """

    release_history: list[ReleaseRecord] = Field(default_factory=list)
    content_plan: list[ContentPlanItem] = Field(default_factory=list)
    style_evolution: list[StyleEvolutionEntry] = Field(default_factory=list)
    effectiveness_metrics: EffectivenessMetrics = Field(
        default_factory=EffectivenessMetrics
    )

    def latest_releases(self, n: int = 10) -> list[ReleaseRecord]:
        return sorted(self.release_history, key=lambda r: r.published_at, reverse=True)[:n]

    def upcoming_plan(self, n: int = 5) -> list[ContentPlanItem]:
        now = datetime.utcnow()
        planned = [
            i for i in self.content_plan
            if i.status == "planned" and (i.planned_date is None or i.planned_date >= now)
        ]
        return planned[:n]


# ─── ReAct cycle log entry ────────────────────────────────────────────────────

class ReActCycleLog(BaseModel):
    """One full Thought→Action→Observation→Decision cycle."""

    cycle_number: int
    thought: str
    action: str
    action_input: dict[str, Any] = Field(default_factory=dict)
    observation: str
    decision: str  # ReactDecision value
    tokens_used: int = 0
    created_at: datetime = Field(default_factory=datetime.utcnow)
