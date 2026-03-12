"""Tests for orchestrator/planner.py — deterministic, no LLM."""
import pytest

from grif.models.agent_config import AgentConfig, ModelConfig, PromptLayers
from grif.models.enums import PlanPattern, TaskType
from grif.models.intent import StructuredIntent
from grif.orchestrator.planner import MultiAgentPlanner


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _intent(task_type: TaskType, raw: str = "Test task", entities: dict | None = None) -> StructuredIntent:
    return StructuredIntent(
        task_type=task_type,
        entities=entities or {},
        raw_input=raw,
    )


def _config(task_type: TaskType = TaskType.EXECUTE, tools: list[str] | None = None) -> AgentConfig:
    return AgentConfig(
        user_id="u1",
        task_type=task_type,
        blueprint_id="generic_worker",
        prompt_layers=PromptLayers(
            layer_1_core_identity="Core",
            layer_2_role_template="Role",
            layer_3_task_context="Task",
        ),
        tools=tools or [],
        model_config=ModelConfig(),
        tool_permissions=[],
    )


# ─── Pattern selection ────────────────────────────────────────────────────────

def test_search_produces_parallel_merge() -> None:
    planner = MultiAgentPlanner()
    plan = planner.plan(_intent(TaskType.SEARCH), _config(TaskType.SEARCH), "task-1")
    assert plan.pattern == PlanPattern.PARALLEL_MERGE


def test_compare_produces_parallel_merge() -> None:
    planner = MultiAgentPlanner()
    plan = planner.plan(_intent(TaskType.COMPARE), _config(TaskType.COMPARE), "task-2")
    assert plan.pattern == PlanPattern.PARALLEL_MERGE


def test_generate_produces_pipeline_review() -> None:
    planner = MultiAgentPlanner()
    plan = planner.plan(_intent(TaskType.GENERATE), _config(TaskType.GENERATE), "task-3")
    assert plan.pattern == PlanPattern.PIPELINE_REVIEW


def test_execute_produces_hypothesis_testing() -> None:
    planner = MultiAgentPlanner()
    plan = planner.plan(_intent(TaskType.EXECUTE), _config(TaskType.EXECUTE), "task-4")
    assert plan.pattern == PlanPattern.HYPOTHESIS_TESTING


def test_operate_produces_hypothesis_testing() -> None:
    planner = MultiAgentPlanner()
    plan = planner.plan(_intent(TaskType.OPERATE), _config(TaskType.OPERATE), "task-5")
    assert plan.pattern == PlanPattern.HYPOTHESIS_TESTING


def test_research_produces_pipeline() -> None:
    planner = MultiAgentPlanner()
    plan = planner.plan(_intent(TaskType.RESEARCH), _config(TaskType.RESEARCH), "task-6")
    assert plan.pattern == PlanPattern.PIPELINE


def test_coach_produces_pipeline() -> None:
    planner = MultiAgentPlanner()
    plan = planner.plan(_intent(TaskType.COACH), _config(TaskType.COACH), "task-7")
    assert plan.pattern == PlanPattern.PIPELINE


def test_monitor_produces_pipeline() -> None:
    planner = MultiAgentPlanner()
    plan = planner.plan(_intent(TaskType.MONITOR), _config(TaskType.MONITOR), "task-8")
    assert plan.pattern == PlanPattern.PIPELINE


# ─── Plan structure ───────────────────────────────────────────────────────────

def test_pipeline_review_has_generator_and_critic() -> None:
    planner = MultiAgentPlanner()
    plan = planner.plan(_intent(TaskType.GENERATE, "Write a blog post"), _config(TaskType.GENERATE), "t")
    roles = {r.role for r in plan.all_agent_roles()}
    assert "generator" in roles
    assert "critic" in roles


def test_parallel_merge_has_two_researchers_and_comparator() -> None:
    planner = MultiAgentPlanner()
    plan = planner.plan(
        _intent(TaskType.COMPARE, "Compare iPhone vs Android", {"a": "iPhone", "b": "Android"}),
        _config(TaskType.COMPARE),
        "t",
    )
    roles = {r.role for r in plan.all_agent_roles()}
    assert "researcher_a" in roles
    assert "researcher_b" in roles
    assert "comparator" in roles


def test_hypothesis_testing_has_two_executors_and_selector() -> None:
    planner = MultiAgentPlanner()
    plan = planner.plan(_intent(TaskType.EXECUTE), _config(TaskType.EXECUTE), "t")
    roles = {r.role for r in plan.all_agent_roles()}
    assert "executor_a" in roles
    assert "executor_b" in roles
    assert "selector" in roles


def test_pipeline_research_has_researcher_and_writer() -> None:
    planner = MultiAgentPlanner()
    plan = planner.plan(_intent(TaskType.RESEARCH, "Research quantum computing"), _config(TaskType.RESEARCH), "t")
    roles = {r.role for r in plan.all_agent_roles()}
    assert "researcher" in roles
    assert "writer" in roles


# ─── Parallelism via order ────────────────────────────────────────────────────

def test_parallel_merge_order1_agents_are_parallel() -> None:
    planner = MultiAgentPlanner()
    plan = planner.plan(_intent(TaskType.COMPARE), _config(TaskType.COMPARE), "t")
    phase = plan.phases[0]
    order1 = phase.agents_at_order(1)
    assert len(order1) == 2  # researcher_a and researcher_b run in parallel


def test_parallel_merge_comparator_is_order2() -> None:
    planner = MultiAgentPlanner()
    plan = planner.plan(_intent(TaskType.COMPARE), _config(TaskType.COMPARE), "t")
    phase = plan.phases[0]
    order2 = phase.agents_at_order(2)
    assert len(order2) == 1
    assert order2[0].role == "comparator"


def test_pipeline_review_sequential_orders() -> None:
    planner = MultiAgentPlanner()
    plan = planner.plan(_intent(TaskType.GENERATE), _config(TaskType.GENERATE), "t")
    phase = plan.phases[0]
    assert phase.max_order == 2
    assert len(phase.agents_at_order(1)) == 1
    assert len(phase.agents_at_order(2)) == 1


# ─── Plan metadata ────────────────────────────────────────────────────────────

def test_plan_has_correct_user_id() -> None:
    planner = MultiAgentPlanner()
    config = _config()
    plan = planner.plan(_intent(TaskType.EXECUTE), config, "task-99")
    assert plan.user_id == "u1"


def test_plan_has_correct_task_id() -> None:
    planner = MultiAgentPlanner()
    plan = planner.plan(_intent(TaskType.EXECUTE), _config(), "my-task-id")
    assert plan.task_id == "my-task-id"


def test_plan_version_starts_at_1() -> None:
    planner = MultiAgentPlanner()
    plan = planner.plan(_intent(TaskType.EXECUTE), _config(), "t")
    assert plan.version == 1


def test_tools_propagated_to_roles() -> None:
    planner = MultiAgentPlanner()
    plan = planner.plan(
        _intent(TaskType.EXECUTE),
        _config(TaskType.EXECUTE, tools=["web_search", "email_client"]),
        "t",
    )
    # All non-merger/selector roles should have tools
    roles_with_tools = [r for r in plan.all_agent_roles() if r.tools]
    assert len(roles_with_tools) >= 1
