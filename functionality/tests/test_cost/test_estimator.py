"""Tests for cost/estimator.py and cost/priority_queue.py."""
import pytest

from grif.cost.estimator import CostEstimate, CostEstimator
from grif.cost.priority_queue import AgentPriorityQueue
from grif.models.agent_config import AgentConfig, ModelConfig, PromptLayers
from grif.models.enums import PlanPattern, TaskType, Urgency
from grif.models.execution_plan import AgentRole, ExecutionPlan, Phase
from grif.models.intent import StructuredIntent
from grif.orchestrator.planner import MultiAgentPlanner


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _config(task_type: TaskType = TaskType.SEARCH) -> AgentConfig:
    return AgentConfig(
        user_id="u1",
        task_type=task_type,
        blueprint_id="generic_worker",
        prompt_layers=PromptLayers(
            layer_1_core_identity="Core",
            layer_2_role_template="Role",
            layer_3_task_context="Task",
        ),
        tools=[],
        model_config=ModelConfig(),
        tool_permissions=[],
    )


def _intent(task_type: TaskType) -> StructuredIntent:
    return StructuredIntent(task_type=task_type, entities={}, raw_input="Test task")


# ═══════════════════════════════════════════════════════════════════════════════
# CostEstimator
# ═══════════════════════════════════════════════════════════════════════════════

def test_estimate_plan_returns_estimate() -> None:
    estimator = CostEstimator()
    planner = MultiAgentPlanner()
    config = _config(TaskType.SEARCH)
    plan = planner.plan(_intent(TaskType.SEARCH), config, "t")
    estimate = estimator.estimate_plan(plan, config)
    assert isinstance(estimate, CostEstimate)


def test_estimate_tokens_positive() -> None:
    estimator = CostEstimator()
    planner = MultiAgentPlanner()
    config = _config(TaskType.RESEARCH)
    plan = planner.plan(_intent(TaskType.RESEARCH), config, "t")
    estimate = estimator.estimate_plan(plan, config)
    assert estimate.estimated_tokens > 0


def test_estimate_cost_positive() -> None:
    estimator = CostEstimator()
    planner = MultiAgentPlanner()
    config = _config(TaskType.GENERATE)
    plan = planner.plan(_intent(TaskType.GENERATE), config, "t")
    estimate = estimator.estimate_plan(plan, config)
    assert estimate.estimated_cost_usd > 0.0


def test_parallel_merge_costs_more_than_pipeline() -> None:
    estimator = CostEstimator()
    planner = MultiAgentPlanner()
    config_search = _config(TaskType.COMPARE)  # PARALLEL_MERGE
    config_research = _config(TaskType.COACH)  # PIPELINE
    plan_parallel = planner.plan(_intent(TaskType.COMPARE), config_search, "t")
    plan_pipeline = planner.plan(_intent(TaskType.COACH), config_research, "t")
    est_parallel = estimator.estimate_plan(plan_parallel, config_search)
    est_pipeline = estimator.estimate_plan(plan_pipeline, config_research)
    # Parallel merge should use more tokens overall
    assert est_parallel.estimated_tokens > est_pipeline.estimated_tokens


def test_estimate_breakdown_has_all_roles() -> None:
    estimator = CostEstimator()
    planner = MultiAgentPlanner()
    config = _config(TaskType.COMPARE)
    plan = planner.plan(_intent(TaskType.COMPARE), config, "t")
    estimate = estimator.estimate_plan(plan, config)
    roles = {r.role for r in plan.all_agent_roles()}
    assert set(estimate.breakdown.keys()) == roles


def test_estimate_single_returns_estimate() -> None:
    estimator = CostEstimator()
    config = _config(TaskType.REMIND)
    estimate = estimator.estimate_single(config)
    assert estimate.estimated_tokens > 0
    assert estimate.estimated_cost_usd > 0.0


def test_estimate_single_custom_cycles() -> None:
    estimator = CostEstimator()
    config = _config(TaskType.SEARCH)
    e1 = estimator.estimate_single(config, max_cycles=1)
    e5 = estimator.estimate_single(config, max_cycles=5)
    assert e5.estimated_tokens == e1.estimated_tokens * 5


def test_annotate_plan_fills_plan_fields() -> None:
    estimator = CostEstimator()
    planner = MultiAgentPlanner()
    config = _config(TaskType.EXECUTE)
    plan = planner.plan(_intent(TaskType.EXECUTE), config, "t")
    assert plan.estimated_tokens is None
    assert plan.estimated_cost_usd is None

    annotated = estimator.annotate_plan(plan, config)
    assert annotated.estimated_tokens is not None
    assert annotated.estimated_cost_usd is not None


def test_budget_level_minimum() -> None:
    e = CostEstimate(estimated_tokens=100, estimated_cost_usd=0.01, breakdown={})
    assert e.budget_level == "minimum"


def test_budget_level_fast() -> None:
    e = CostEstimate(estimated_tokens=5000, estimated_cost_usd=0.20, breakdown={})
    assert e.budget_level == "fast"


def test_budget_level_deep() -> None:
    e = CostEstimate(estimated_tokens=50000, estimated_cost_usd=1.50, breakdown={})
    assert e.budget_level == "deep"


# ═══════════════════════════════════════════════════════════════════════════════
# AgentPriorityQueue
# ═══════════════════════════════════════════════════════════════════════════════

def test_push_and_pop_returns_entry() -> None:
    q = AgentPriorityQueue()
    q.push("agent-1", "u1", TaskType.SEARCH, Urgency.NORMAL)
    entry = q.pop()
    assert entry is not None
    assert entry.agent_id == "agent-1"


def test_high_urgency_pops_before_low() -> None:
    q = AgentPriorityQueue()
    q.push("low", "u1", TaskType.RESEARCH, Urgency.LOW)
    q.push("high", "u1", TaskType.RESEARCH, Urgency.HIGH)
    entry = q.pop()
    assert entry is not None
    assert entry.agent_id == "high"


def test_remind_gets_priority_boost() -> None:
    q = AgentPriorityQueue()
    q.push("research", "u1", TaskType.RESEARCH, Urgency.NORMAL)
    q.push("remind", "u1", TaskType.REMIND, Urgency.NORMAL)
    entry = q.pop()
    assert entry is not None
    assert entry.agent_id == "remind"


def test_expensive_agent_gets_penalty() -> None:
    q = AgentPriorityQueue()
    q.push("cheap", "u1", TaskType.SEARCH, Urgency.NORMAL, estimated_cost_usd=0.01)
    q.push("expensive", "u1", TaskType.SEARCH, Urgency.NORMAL, estimated_cost_usd=2.00)
    entry = q.pop()
    assert entry is not None
    assert entry.agent_id == "cheap"


def test_pop_empty_returns_none() -> None:
    q = AgentPriorityQueue()
    assert q.pop() is None


def test_size_and_running() -> None:
    q = AgentPriorityQueue(max_concurrent=2)
    q.push("a1", "u1", TaskType.SEARCH, Urgency.NORMAL)
    q.push("a2", "u1", TaskType.SEARCH, Urgency.NORMAL)
    assert q.size() == 2
    q.pop()
    assert q.size() == 1
    assert q.running_count() == 1


def test_can_run_respects_limit() -> None:
    q = AgentPriorityQueue(max_concurrent=1)
    q.push("a1", "u1", TaskType.SEARCH, Urgency.NORMAL)
    q.push("a2", "u1", TaskType.SEARCH, Urgency.NORMAL)
    assert q.can_run() is True
    q.pop()  # Running: 1
    assert q.can_run() is False  # At limit


def test_complete_frees_slot() -> None:
    q = AgentPriorityQueue(max_concurrent=1)
    q.push("a1", "u1", TaskType.SEARCH, Urgency.NORMAL)
    q.pop()
    assert q.can_run() is False
    q.complete("a1")
    assert q.can_run() is True


def test_drain_ready() -> None:
    q = AgentPriorityQueue(max_concurrent=3)
    for i in range(5):
        q.push(f"agent-{i}", "u1", TaskType.SEARCH, Urgency.NORMAL)
    ready = q.drain_ready()
    assert len(ready) == 3  # limited by max_concurrent
    assert q.running_count() == 3
    assert q.size() == 2


def test_peek_does_not_remove() -> None:
    q = AgentPriorityQueue()
    q.push("a1", "u1", TaskType.SEARCH, Urgency.NORMAL)
    entry = q.peek()
    assert entry is not None
    assert q.size() == 1  # Still in queue
