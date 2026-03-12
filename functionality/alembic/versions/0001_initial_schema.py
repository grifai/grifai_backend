"""Initial schema: agents, memory, logs, users, wake_queue, blueprints, tokens, tasks

Revision ID: 0001
Revises:
Create Date: 2026-03-11 00:00:00.000000
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects.postgresql import JSONB, UUID

revision: str = "0001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Enable pgvector extension
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    # ── agents ──────────────────────────────────────────────────────────────
    op.create_table(
        "agents",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("user_id", sa.String(128), nullable=False),
        sa.Column("state", sa.String(32), nullable=False, server_default="embryo"),
        sa.Column("task_type", sa.String(32), nullable=False),
        sa.Column("blueprint_id", sa.String(128), nullable=True),
        sa.Column("config", JSONB, nullable=False, server_default="{}"),
        sa.Column("checkpoint", JSONB, nullable=True),
        sa.Column("context_summary", sa.Text, nullable=True),
        sa.Column("schedule_cron", sa.String(64), nullable=True),
        sa.Column("schedule_timezone", sa.String(64), server_default="UTC"),
        sa.Column(
            "parent_agent_id",
            UUID(as_uuid=True),
            sa.ForeignKey("agents.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("avg_score", sa.Float, nullable=True),
        sa.Column("eval_count", sa.Integer, server_default="0"),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
        ),
        sa.Column("archived_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("ix_agents_user_state", "agents", ["user_id", "state"])
    op.create_index("ix_agents_task_type", "agents", ["user_id", "task_type"])

    # ── agent_memory ─────────────────────────────────────────────────────────
    op.create_table(
        "agent_memory",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "agent_id",
            UUID(as_uuid=True),
            sa.ForeignKey("agents.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("user_id", sa.String(128), nullable=False),
        sa.Column("memory_type", sa.String(32), nullable=False),
        sa.Column("content", JSONB, nullable=False),
        sa.Column("embedding", Vector(1536), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
        ),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("ix_agent_memory_type", "agent_memory", ["agent_id", "memory_type"])
    # IVFFlat index for vector similarity search
    op.execute(
        "CREATE INDEX ix_agent_memory_embedding ON agent_memory "
        "USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100)"
        " WHERE embedding IS NOT NULL"
    )

    # ── agent_logs ────────────────────────────────────────────────────────────
    op.create_table(
        "agent_logs",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "agent_id",
            UUID(as_uuid=True),
            sa.ForeignKey("agents.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("user_id", sa.String(128), nullable=False),
        sa.Column("cycle_number", sa.Integer, nullable=False),
        sa.Column("thought", sa.Text, nullable=False),
        sa.Column("action", sa.String(128), nullable=False),
        sa.Column("action_input", JSONB, nullable=False, server_default="{}"),
        sa.Column("observation", sa.Text, nullable=False),
        sa.Column("decision", sa.String(32), nullable=False),
        sa.Column("tokens_used", sa.Integer, server_default="0"),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
        ),
    )
    op.create_index("ix_agent_logs_agent", "agent_logs", ["agent_id"])
    op.create_index("ix_agent_logs_created", "agent_logs", ["agent_id", "created_at"])
    op.create_index("ix_agent_logs_user", "agent_logs", ["user_id"])

    # ── user_profiles ─────────────────────────────────────────────────────────
    op.create_table(
        "user_profiles",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("user_id", sa.String(128), nullable=False, unique=True),
        sa.Column("preferences", JSONB, nullable=False, server_default="{}"),
        sa.Column("contacts", JSONB, nullable=False, server_default="{}"),
        sa.Column("channels", JSONB, nullable=False, server_default="{}"),
        sa.Column("biorhythms", JSONB, nullable=False, server_default="{}"),
        sa.Column("style_guide", sa.Text, nullable=True),
        sa.Column("style_guide_updated_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
        ),
    )
    op.create_index("ix_user_profiles_user", "user_profiles", ["user_id"])

    # ── wake_queue ────────────────────────────────────────────────────────────
    op.create_table(
        "wake_queue",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "agent_id",
            UUID(as_uuid=True),
            sa.ForeignKey("agents.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("user_id", sa.String(128), nullable=False),
        sa.Column("trigger_type", sa.String(32), nullable=False),
        sa.Column("trigger_condition", sa.Text, nullable=True),
        sa.Column("scheduled_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("fired_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("is_processed", sa.Boolean, server_default="false"),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
        ),
    )
    op.create_index("ix_wake_queue_agent", "wake_queue", ["agent_id"])
    op.create_index("ix_wake_queue_scheduled", "wake_queue", ["scheduled_at"])
    op.create_index("ix_wake_queue_pending", "wake_queue", ["is_processed"])

    # ── blueprints ────────────────────────────────────────────────────────────
    op.create_table(
        "blueprints",
        sa.Column("id", sa.String(128), primary_key=True),
        sa.Column("task_type", sa.String(32), nullable=False),
        sa.Column("domain", sa.String(64), nullable=True),
        sa.Column("name", sa.String(256), nullable=False),
        sa.Column("description", sa.Text, nullable=True),
        sa.Column("definition", JSONB, nullable=False),
        sa.Column("is_personal", sa.Boolean, server_default="false"),
        sa.Column("owner_user_id", sa.String(128), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
        ),
    )
    op.create_index("ix_blueprints_task_domain", "blueprints", ["task_type", "domain"])
    op.create_index("ix_blueprints_owner", "blueprints", ["owner_user_id"])

    # ── blueprint_scores ──────────────────────────────────────────────────────
    op.create_table(
        "blueprint_scores",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "blueprint_id",
            sa.String(128),
            sa.ForeignKey("blueprints.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("user_id", sa.String(128), nullable=False),
        sa.Column("score", sa.Float, nullable=False),
        sa.Column("agent_id", UUID(as_uuid=True), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
        ),
    )
    op.create_index(
        "ix_blueprint_scores_user_blueprint",
        "blueprint_scores",
        ["user_id", "blueprint_id"],
    )

    # ── token_usage ───────────────────────────────────────────────────────────
    op.create_table(
        "token_usage",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("user_id", sa.String(128), nullable=False),
        sa.Column("agent_id", UUID(as_uuid=True), nullable=True),
        sa.Column("model_id", sa.String(128), nullable=False),
        sa.Column("prompt_tokens", sa.Integer, server_default="0"),
        sa.Column("completion_tokens", sa.Integer, server_default="0"),
        sa.Column("total_tokens", sa.Integer, server_default="0"),
        sa.Column("cost_usd", sa.Float, server_default="0"),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            index=True,
        ),
    )
    op.create_index("ix_token_usage_user", "token_usage", ["user_id"])
    op.create_index("ix_token_usage_user_date", "token_usage", ["user_id", "created_at"])

    # ── tasks ─────────────────────────────────────────────────────────────────
    op.create_table(
        "tasks",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("user_id", sa.String(128), nullable=False),
        sa.Column("raw_input", sa.Text, nullable=False),
        sa.Column("signal_type", sa.String(32), server_default="text"),
        sa.Column("classified_intent", JSONB, nullable=True),
        sa.Column("execution_plan", JSONB, nullable=True),
        sa.Column("status", sa.String(32), server_default="pending"),
        sa.Column("pending_clarification", JSONB, nullable=True),
        sa.Column("clarification_answers", JSONB, nullable=True),
        sa.Column("budget_approved", sa.Boolean, server_default="false"),
        sa.Column("budget_level", sa.String(32), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
        ),
    )
    op.create_index("ix_tasks_user", "tasks", ["user_id"])
    op.create_index("ix_tasks_status", "tasks", ["user_id", "status"])


def downgrade() -> None:
    op.drop_table("tasks")
    op.drop_table("token_usage")
    op.drop_table("blueprint_scores")
    op.drop_table("blueprints")
    op.drop_table("wake_queue")
    op.drop_table("user_profiles")
    op.drop_table("agent_logs")
    op.drop_table("agent_memory")
    op.drop_table("agents")
    op.execute("DROP EXTENSION IF EXISTS vector")
