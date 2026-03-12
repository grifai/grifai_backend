"""Tests for orchestrator/replanner.py — deterministic, no LLM."""
import pytest

from grif.models.enums import PlanPattern, TaskType
from grif.models.execution_plan import AgentRole, ExecutionPlan, Phase
from grif.orchestrator.replanner import DynamicReplanner, _MAX_FAILURES


# ─── Fixtures ─────────────────────────────────────────────────────────────────

def _make_plan(roles: list[tuple[str, int]] | None = None) -> ExecutionPlan:
    """Create a simple plan. roles = [(role_name, order), ...]"""
    if roles is None:
        roles = [("executor_a", 1), ("executor_b", 1), ("selector", 2)]
    return ExecutionPlan(
        task_id="task-1",
        user_id="u1",
        pattern=PlanPattern.HYPOTHESIS_TESTING,
        phases=[
            Phase(
                name="test",
                agents=[
                    AgentRole(role=r, goal=f"Goal of {r}", order=o)
                    for r, o in roles
                ],
            )
        ],
    )


# ─── No failure ───────────────────────────────────────────────────────────────

def test_replan_no_failures_returns_same_plan() -> None:
    replanner = DynamicReplanner()
    plan = _make_plan()
    result = replanner.replan(plan, failed_roles=[])
    assert not result.should_escalate
    assert result.plan is plan


# ─── Non-critical failure ──────────────────────────────────────────────────────

def test_replan_removes_failed_non_critical_role() -> None:
    replanner = DynamicReplanner()
    plan = _make_plan([("researcher_a", 1), ("researcher_b", 1), ("comparator", 2)])
    # researcher_b fails but researcher_a survives → non-critical (another order-1 survives)
    result = replanner.replan(plan, failed_roles=["researcher_b"], failure_reason="timeout")
    assert not result.should_escalate
    assert result.plan is not None
    roles = {r.role for r in result.plan.all_agent_roles()}
    assert "researcher_b" not in roles
    assert "researcher_a" in roles
    assert "comparator" in roles


def test_replan_increments_version() -> None:
    replanner = DynamicReplanner()
    plan = _make_plan([("researcher_a", 1), ("researcher_b", 1), ("comparator", 2)])
    result = replanner.replan(plan, failed_roles=["researcher_b"])
    assert result.plan is not None
    assert result.plan.version == plan.version + 1


def test_replan_sets_replan_reason() -> None:
    replanner = DynamicReplanner()
    plan = _make_plan([("researcher_a", 1), ("researcher_b", 1), ("comparator", 2)])
    result = replanner.replan(plan, failed_roles=["researcher_b"], failure_reason="OOM")
    assert result.plan is not None
    assert "researcher_b" in result.plan.replan_reason


# ─── Critical failure ─────────────────────────────────────────────────────────

def test_replan_escalates_when_all_order1_fail() -> None:
    """If ALL order-1 agents in first phase fail → escalate."""
    replanner = DynamicReplanner()
    plan = _make_plan([("executor_a", 1), ("selector", 2)])
    # executor_a is the ONLY order-1 agent → critical
    result = replanner.replan(plan, failed_roles=["executor_a"])
    assert result.should_escalate
    assert result.plan is None


def test_replan_escalates_when_phase_becomes_empty() -> None:
    """Remove all agents from a phase → escalate."""
    replanner = DynamicReplanner()
    plan = _make_plan([("executor", 1)])
    result = replanner.replan(plan, failed_roles=["executor"])
    assert result.should_escalate


# ─── Max failures ──────────────────────────────────────────────────────────────

def test_replan_escalates_at_max_failures() -> None:
    replanner = DynamicReplanner()
    # Set version to simulate accumulated failures
    plan = _make_plan([("researcher_a", 1), ("researcher_b", 1), ("comparator", 2)])
    plan = plan.model_copy(update={"version": _MAX_FAILURES})  # already had 2 failures
    result = replanner.replan(plan, failed_roles=["researcher_b"])
    assert result.should_escalate


def test_replan_does_not_escalate_below_max() -> None:
    replanner = DynamicReplanner()
    plan = _make_plan([("researcher_a", 1), ("researcher_b", 1), ("comparator", 2)])
    plan = plan.model_copy(update={"version": 1})  # 0 accumulated failures
    result = replanner.replan(plan, failed_roles=["researcher_b"])
    # 0 + 1 = 1 failure, below MAX_FAILURES=3
    assert not result.should_escalate


# ─── retry_role ───────────────────────────────────────────────────────────────

def test_retry_role_resets_agent_id() -> None:
    replanner = DynamicReplanner()
    import uuid
    plan = _make_plan([("executor_a", 1), ("selector", 2)])
    plan.phases[0].agents[0].agent_id = uuid.uuid4()

    updated = replanner.retry_role(plan, "executor_a")
    executor = next(r for r in updated.all_agent_roles() if r.role == "executor_a")
    assert executor.agent_id is None


def test_retry_role_updates_goal() -> None:
    replanner = DynamicReplanner()
    plan = _make_plan([("executor_a", 1)])
    updated = replanner.retry_role(plan, "executor_a", new_goal="Revised: try simpler approach")
    executor = next(r for r in updated.all_agent_roles() if r.role == "executor_a")
    assert "simpler approach" in executor.goal


def test_retry_role_increments_version() -> None:
    replanner = DynamicReplanner()
    plan = _make_plan([("executor_a", 1)])
    updated = replanner.retry_role(plan, "executor_a")
    assert updated.version == plan.version + 1


def test_retry_role_leaves_other_roles_unchanged() -> None:
    replanner = DynamicReplanner()
    plan = _make_plan([("executor_a", 1), ("executor_b", 1), ("selector", 2)])
    updated = replanner.retry_role(plan, "executor_a", new_goal="New goal")
    # executor_b and selector should be untouched
    executor_b = next(r for r in updated.all_agent_roles() if r.role == "executor_b")
    assert executor_b.goal == "Goal of executor_b"
