"""Tests for grif/models/enums.py"""
import pytest

from grif.models.enums import (
    AgentState,
    Complexity,
    MemoryType,
    PlanPattern,
    PhaseMode,
    ReactDecision,
    RouterDecision,
    TaskType,
    ToolCategory,
    Urgency,
    WakeTriggerType,
)


def test_task_type_values() -> None:
    """All 9 TaskTypes from the architecture spec must be present."""
    expected = {
        "search", "monitor", "research", "coach",
        "compare", "execute", "remind", "generate", "operate",
    }
    assert set(TaskType) == expected


def test_agent_state_values() -> None:
    expected = {"embryo", "active", "sleeping", "recurring", "archived"}
    assert set(AgentState) == expected


def test_tool_category_values() -> None:
    expected = {"read", "write_safe", "write_public", "write_irreversible"}
    assert set(ToolCategory) == expected


def test_react_decision_values() -> None:
    expected = {"continue", "report", "wait", "sleep", "escalate"}
    assert set(ReactDecision) == expected


def test_router_decision_values() -> None:
    expected = {"new", "existing", "fork", "skip"}
    assert set(RouterDecision) == expected


def test_plan_pattern_values() -> None:
    expected = {"pipeline", "parallel_merge", "pipeline_review", "hypothesis_testing"}
    assert set(PlanPattern) == expected


def test_phase_mode_values() -> None:
    assert set(PhaseMode) == {"one_shot", "recurring"}


def test_memory_type_values() -> None:
    expected = {"fact", "decision", "preference", "production"}
    assert set(MemoryType) == expected


def test_str_enum_equality() -> None:
    """StrEnum values should compare equal to plain strings."""
    assert TaskType.SEARCH == "search"
    assert AgentState.ACTIVE == "active"
    assert ToolCategory.WRITE_IRREVERSIBLE == "write_irreversible"


def test_enum_membership() -> None:
    assert "operate" in TaskType
    assert "embryo" in AgentState
    assert "production" in MemoryType
