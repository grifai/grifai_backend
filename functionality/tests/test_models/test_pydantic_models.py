"""Tests for Pydantic model construction and validation."""
import uuid
from datetime import datetime

import pytest

from grif.models.agent_config import (
    AgentConfig,
    CommunicationConfig,
    ModelConfig,
    PromptLayers,
    Schedule,
    ToolPermission,
    WakeTrigger,
)
from grif.models.enums import (
    AgentState,
    TaskType,
    ToolCategory,
    WakeTriggerType,
)
from grif.models.execution_plan import AgentRole, ExecutionPlan, Phase
from grif.models.intent import (
    ClassifiedIntent,
    ClarificationQuestion,
    ClarificationRequest,
    StructuredIntent,
)
from grif.models.memory import (
    DecisionMemory,
    EffectivenessMetrics,
    FactMemory,
    PreferenceMemory,
    ProductionMemory,
    ReActCycleLog,
    ReleaseRecord,
)


# ─── StructuredIntent ─────────────────────────────────────────────────────────

def test_structured_intent_defaults() -> None:
    intent = StructuredIntent(
        task_type=TaskType.SEARCH,
        raw_input="Find a hotel in Paris",
    )
    assert intent.task_type == TaskType.SEARCH
    assert intent.language == "ru"
    assert intent.entities == {}
    assert intent.constraints == {}


def test_structured_intent_with_entities() -> None:
    intent = StructuredIntent(
        task_type=TaskType.RESEARCH,
        entities={"topic": "AI agents", "depth": "deep"},
        constraints={"max_pages": 50},
        raw_input="Research AI agents deeply",
        language="en",
    )
    assert intent.entities["topic"] == "AI agents"
    assert intent.constraints["max_pages"] == 50


# ─── ClassifiedIntent ─────────────────────────────────────────────────────────

def test_classified_intent_no_clarification() -> None:
    intent = StructuredIntent(task_type=TaskType.MONITOR, raw_input="Watch BTC price")
    classified = ClassifiedIntent(structured_intent=intent)
    assert not classified.clarification_needed
    assert classified.clarification_request is None
    assert classified.missing_fields == []


def test_classified_intent_with_clarification() -> None:
    intent = StructuredIntent(task_type=TaskType.RESEARCH, raw_input="Research something")
    req = ClarificationRequest(
        mode="structured_interview",
        questions=[
            ClarificationQuestion(field_name="topic", question="What topic?"),
            ClarificationQuestion(
                field_name="depth",
                question="How deep?",
                options=["quick", "thorough"],
            ),
        ],
        context_summary="Need to know the topic",
    )
    classified = ClassifiedIntent(
        structured_intent=intent,
        clarification_needed=True,
        clarification_request=req,
        missing_fields=["topic", "depth"],
    )
    assert classified.clarification_needed
    assert len(classified.clarification_request.questions) == 2
    assert classified.missing_fields == ["topic", "depth"]


# ─── AgentConfig ──────────────────────────────────────────────────────────────

def _make_prompt_layers() -> PromptLayers:
    return PromptLayers(
        layer_1_core_identity="You are GRIF agent.",
        layer_2_role_template="You are a researcher.",
        layer_3_task_context="Find info about X.",
        layer_4_user_persona="User prefers brevity.",
    )


def test_agent_config_construction() -> None:
    layers = _make_prompt_layers()
    config = AgentConfig(
        user_id="user_123",
        task_type=TaskType.RESEARCH,
        blueprint_id="research_analyst",
        prompt_layers=layers,
        tools=["web_search", "fetch"],
    )
    assert config.user_id == "user_123"
    assert config.task_type == TaskType.RESEARCH
    assert isinstance(config.id, uuid.UUID)


def test_prompt_layers_assembly() -> None:
    layers = _make_prompt_layers()
    assembled = layers.assemble()
    assert "You are GRIF agent." in assembled
    assert "You are a researcher." in assembled
    assert "Find info about X." in assembled
    assert "User prefers brevity." in assembled


def test_prompt_layers_without_persona() -> None:
    layers = PromptLayers(
        layer_1_core_identity="Core",
        layer_2_role_template="Role",
        layer_3_task_context="Task",
    )
    assembled = layers.assemble()
    # Layer 4 is empty — should not add empty separator
    assert assembled.count("---") == 2  # 3 layers = 2 separators


def test_tool_permission_defaults() -> None:
    perm = ToolPermission(
        tool_name="post_telegram",
        category=ToolCategory.WRITE_PUBLIC,
    )
    assert not perm.auto_approved
    assert perm.approval_count == 0
    assert perm.trust_threshold == 5


def test_wake_trigger() -> None:
    trigger = WakeTrigger(
        trigger_type=WakeTriggerType.CONDITION,
        condition="price",
        value=5000,
    )
    assert trigger.trigger_type == WakeTriggerType.CONDITION
    assert trigger.value == 5000


# ─── ExecutionPlan ────────────────────────────────────────────────────────────

def test_execution_plan_pipeline() -> None:
    plan = ExecutionPlan(
        task_id="task_abc",
        user_id="user_123",
        pattern="pipeline",
        phases=[
            Phase(
                name="main",
                mode="one_shot",
                agents=[
                    AgentRole(role="researcher", goal="Find data", order=1),
                    AgentRole(role="writer", goal="Write report", order=2),
                ],
            )
        ],
    )
    assert plan.pattern == "pipeline"
    assert len(plan.phases[0].agents) == 2
    assert plan.phases[0].max_order == 2


def test_execution_plan_parallel_agents() -> None:
    phase = Phase(
        name="research",
        mode="one_shot",
        agents=[
            AgentRole(role="researcher_1", goal="Find EN sources", order=1),
            AgentRole(role="researcher_2", goal="Find RU sources", order=1),
            AgentRole(role="writer", goal="Write", order=2),
        ],
    )
    parallel = phase.agents_at_order(1)
    assert len(parallel) == 2
    assert {a.role for a in parallel} == {"researcher_1", "researcher_2"}


def test_execution_plan_all_roles() -> None:
    plan = ExecutionPlan(
        task_id="t1",
        user_id="u1",
        pattern="pipeline",
        phases=[
            Phase(name="setup", agents=[AgentRole(role="planner", goal="Plan")]),
            Phase(
                name="work",
                agents=[
                    AgentRole(role="worker_a", goal="A"),
                    AgentRole(role="worker_b", goal="B"),
                ],
            ),
        ],
    )
    assert len(plan.all_agent_roles()) == 3


# ─── Memory models ────────────────────────────────────────────────────────────

def test_fact_memory() -> None:
    fact = FactMemory(key="hotel_name", value="Grand Paris", source="booking.com")
    assert fact.key == "hotel_name"
    assert fact.confidence == 1.0
    assert isinstance(fact.id, uuid.UUID)


def test_decision_memory() -> None:
    dec = DecisionMemory(
        decision="Choose hotel Grand Paris",
        reasoning="Price $138 < $150, rating 4.7, breakfast included",
    )
    assert "Grand Paris" in dec.decision


def test_production_memory_latest_releases() -> None:
    pm = ProductionMemory()
    for i in range(15):
        pm.release_history.append(
            ReleaseRecord(
                title=f"Post {i}",
                content_type="post",
                channel="telegram",
            )
        )
    latest = pm.latest_releases(10)
    assert len(latest) == 10


def test_react_cycle_log() -> None:
    log_entry = ReActCycleLog(
        cycle_number=1,
        thought="I need to search for hotels",
        action="web_search",
        action_input={"query": "hotels in Paris"},
        observation="Found 10 hotels",
        decision="continue",
        tokens_used=350,
    )
    assert log_entry.cycle_number == 1
    assert log_entry.decision == "continue"
