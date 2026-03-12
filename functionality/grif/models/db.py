"""
SQLAlchemy ORM models for PostgreSQL.
All tables use UUID primary keys and store JSON blobs for nested structures.
pgvector is used for memory embeddings (semantic search / RAG).
"""

import uuid
from datetime import datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    BigInteger,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy.sql import func


class Base(DeclarativeBase):
    pass


# ─── Agents ──────────────────────────────────────────────────────────────────

class AgentDB(Base):
    __tablename__ = "agents"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    user_id: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    state: Mapped[str] = mapped_column(String(32), nullable=False, default="embryo")
    task_type: Mapped[str] = mapped_column(String(32), nullable=False)
    blueprint_id: Mapped[str | None] = mapped_column(String(128), nullable=True)

    # Full AgentConfig as JSON
    config: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)

    # LangGraph checkpoint (serialized state)
    checkpoint: Mapped[dict | None] = mapped_column(JSONB, nullable=True)

    # Working context summary (refreshed on wake)
    context_summary: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Recurring schedule (cron expression, null for one-shot agents)
    schedule_cron: Mapped[str | None] = mapped_column(String(64), nullable=True)
    schedule_timezone: Mapped[str] = mapped_column(String(64), default="UTC")

    # Parent for FORK lineage
    parent_agent_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("agents.id", ondelete="SET NULL"), nullable=True
    )

    # Evaluation score (1-5, average of Self-Evaluation calls)
    avg_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    eval_count: Mapped[int] = mapped_column(Integer, default=0)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )
    archived_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    # Relationships
    memory_entries: Mapped[list["AgentMemoryDB"]] = relationship(
        "AgentMemoryDB", back_populates="agent", cascade="all, delete-orphan"
    )
    logs: Mapped[list["AgentLogDB"]] = relationship(
        "AgentLogDB", back_populates="agent", cascade="all, delete-orphan"
    )
    wake_queue_entries: Mapped[list["WakeQueueDB"]] = relationship(
        "WakeQueueDB", back_populates="agent", cascade="all, delete-orphan"
    )

    __table_args__ = (
        Index("ix_agents_user_state", "user_id", "state"),
        Index("ix_agents_task_type", "user_id", "task_type"),
    )


# ─── Agent Memory ─────────────────────────────────────────────────────────────

class AgentMemoryDB(Base):
    __tablename__ = "agent_memory"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    agent_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("agents.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    user_id: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    memory_type: Mapped[str] = mapped_column(
        String(32), nullable=False
    )  # MemoryType enum value

    # Serialized Pydantic model (FactMemory, DecisionMemory, etc.)
    content: Mapped[dict] = mapped_column(JSONB, nullable=False)

    # pgvector embedding for semantic search (1536 dims = text-embedding-3-small)
    embedding: Mapped[list[float] | None] = mapped_column(
        Vector(1536), nullable=True
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    expires_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    agent: Mapped["AgentDB"] = relationship("AgentDB", back_populates="memory_entries")

    __table_args__ = (
        Index("ix_agent_memory_type", "agent_id", "memory_type"),
        Index(
            "ix_agent_memory_embedding",
            "embedding",
            postgresql_using="ivfflat",
            postgresql_with={"lists": 100},
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
    )


# ─── Agent Logs (Audit Trail) ─────────────────────────────────────────────────

class AgentLogDB(Base):
    __tablename__ = "agent_logs"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    agent_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("agents.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    user_id: Mapped[str] = mapped_column(String(128), nullable=False, index=True)

    cycle_number: Mapped[int] = mapped_column(Integer, nullable=False)
    thought: Mapped[str] = mapped_column(Text, nullable=False)
    action: Mapped[str] = mapped_column(String(128), nullable=False)
    action_input: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)
    observation: Mapped[str] = mapped_column(Text, nullable=False)
    decision: Mapped[str] = mapped_column(String(32), nullable=False)
    tokens_used: Mapped[int] = mapped_column(Integer, default=0)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), index=True
    )

    agent: Mapped["AgentDB"] = relationship("AgentDB", back_populates="logs")

    __table_args__ = (
        Index("ix_agent_logs_created", "agent_id", "created_at"),
    )


# ─── User Profiles ────────────────────────────────────────────────────────────

class UserProfileDB(Base):
    __tablename__ = "user_profiles"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    user_id: Mapped[str] = mapped_column(
        String(128), nullable=False, unique=True, index=True
    )

    # Nested JSON structures
    preferences: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)
    contacts: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)
    channels: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)
    biorhythms: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)

    # Style Guide (200-300 tokens, generated by StyleCloner)
    style_guide: Mapped[str | None] = mapped_column(Text, nullable=True)
    style_guide_updated_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )


# ─── Wake Queue ──────────────────────────────────────────────────────────────

class WakeQueueDB(Base):
    __tablename__ = "wake_queue"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    agent_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("agents.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    user_id: Mapped[str] = mapped_column(String(128), nullable=False)
    trigger_type: Mapped[str] = mapped_column(String(32), nullable=False)
    trigger_condition: Mapped[str | None] = mapped_column(Text, nullable=True)
    scheduled_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True, index=True
    )
    fired_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    is_processed: Mapped[bool] = mapped_column(Boolean, default=False, index=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    agent: Mapped["AgentDB"] = relationship("AgentDB", back_populates="wake_queue_entries")


# ─── Blueprints ───────────────────────────────────────────────────────────────

class BlueprintDB(Base):
    __tablename__ = "blueprints"

    id: Mapped[str] = mapped_column(String(128), primary_key=True)
    task_type: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    domain: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    name: Mapped[str] = mapped_column(String(256), nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Full blueprint definition as JSON (mirrors JSON files in blueprints/definitions/)
    definition: Mapped[dict] = mapped_column(JSONB, nullable=False)

    is_personal: Mapped[bool] = mapped_column(Boolean, default=False)
    # If personal, which user it belongs to
    owner_user_id: Mapped[str | None] = mapped_column(String(128), nullable=True, index=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    __table_args__ = (
        Index("ix_blueprints_task_domain", "task_type", "domain"),
    )


# ─── Blueprint Scores ─────────────────────────────────────────────────────────

class BlueprintScoreDB(Base):
    __tablename__ = "blueprint_scores"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    blueprint_id: Mapped[str] = mapped_column(
        String(128),
        ForeignKey("blueprints.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    user_id: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    score: Mapped[float] = mapped_column(Float, nullable=False)  # 1.0–5.0
    agent_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    __table_args__ = (
        Index("ix_blueprint_scores_user_blueprint", "user_id", "blueprint_id"),
    )


# ─── Token Usage ──────────────────────────────────────────────────────────────

class TokenUsageDB(Base):
    __tablename__ = "token_usage"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    user_id: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    agent_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), nullable=True, index=True
    )
    model_id: Mapped[str] = mapped_column(String(128), nullable=False)
    prompt_tokens: Mapped[int] = mapped_column(Integer, default=0)
    completion_tokens: Mapped[int] = mapped_column(Integer, default=0)
    total_tokens: Mapped[int] = mapped_column(Integer, default=0)
    cost_usd: Mapped[float] = mapped_column(Float, default=0.0)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), index=True
    )

    __table_args__ = (
        Index("ix_token_usage_user_date", "user_id", "created_at"),
    )


# ─── Tasks (top-level user requests) ─────────────────────────────────────────

class TaskDB(Base):
    """
    Represents a top-level user request flowing through the pipeline.
    One task can spawn multiple agents (via ExecutionPlan).
    """

    __tablename__ = "tasks"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    user_id: Mapped[str] = mapped_column(String(128), nullable=False, index=True)

    # Raw user input
    raw_input: Mapped[str] = mapped_column(Text, nullable=False)
    signal_type: Mapped[str] = mapped_column(String(32), default="text")

    # Pipeline stages stored as JSON
    classified_intent: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    execution_plan: Mapped[dict | None] = mapped_column(JSONB, nullable=True)

    status: Mapped[str] = mapped_column(
        String(32), default="pending", index=True
    )  # pending, clarifying, planning, running, done, failed

    # Clarification state
    pending_clarification: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    clarification_answers: Mapped[dict | None] = mapped_column(JSONB, nullable=True)

    # Budget approval
    budget_approved: Mapped[bool] = mapped_column(Boolean, default=False)
    budget_level: Mapped[str | None] = mapped_column(
        String(32), nullable=True
    )  # deep, fast, minimum

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )
