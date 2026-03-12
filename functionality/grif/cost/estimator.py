"""
CostEstimator — deterministic token/cost estimation before running an agent.

No LLM calls — uses historical averages and heuristics.
Fills ExecutionPlan.estimated_tokens and ExecutionPlan.estimated_cost_usd.
"""

from __future__ import annotations

from grif.llm.fallback_map import HAIKU, SONNET, SUMMARIZER
from grif.models.agent_config import AgentConfig
from grif.models.enums import PlanPattern, TaskType
from grif.models.execution_plan import ExecutionPlan

# Token cost per 1K tokens (USD) — approximate mid-2025 pricing
_COST_PER_1K: dict[str, float] = {
    SONNET.primary: 0.015,   # claude-sonnet-4-6 output
    HAIKU.primary: 0.00125,  # claude-haiku output
    SUMMARIZER.primary: 0.0006,  # gpt-4o-mini output
    "gpt-4o": 0.015,
    "default": 0.01,
}

# Average tokens per ReAct cycle by task complexity
_TOKENS_PER_CYCLE: dict[TaskType, int] = {
    TaskType.SEARCH: 800,
    TaskType.MONITOR: 400,
    TaskType.REMIND: 300,
    TaskType.RESEARCH: 1500,
    TaskType.COACH: 1200,
    TaskType.COMPARE: 1000,
    TaskType.GENERATE: 2000,
    TaskType.EXECUTE: 1000,
    TaskType.OPERATE: 1500,
}

# Expected max cycles by task type
_EXPECTED_CYCLES: dict[TaskType, int] = {
    TaskType.SEARCH: 3,
    TaskType.MONITOR: 2,
    TaskType.REMIND: 1,
    TaskType.RESEARCH: 8,
    TaskType.COACH: 5,
    TaskType.COMPARE: 5,
    TaskType.GENERATE: 6,
    TaskType.EXECUTE: 6,
    TaskType.OPERATE: 10,
}

# Per-pattern overhead multiplier
_PATTERN_MULTIPLIER: dict[PlanPattern, float] = {
    PlanPattern.PIPELINE: 1.0,
    PlanPattern.PARALLEL_MERGE: 1.8,   # 2 parallel agents
    PlanPattern.PIPELINE_REVIEW: 1.6,  # generator + critic
    PlanPattern.HYPOTHESIS_TESTING: 2.0,  # 2 tracks + selector
}


class CostEstimate:
    def __init__(
        self,
        estimated_tokens: int,
        estimated_cost_usd: float,
        breakdown: dict[str, float],
    ) -> None:
        self.estimated_tokens = estimated_tokens
        self.estimated_cost_usd = estimated_cost_usd
        self.breakdown = breakdown  # role → tokens

    @property
    def budget_level(self) -> str:
        """Classify cost as 'minimum', 'fast', or 'deep'."""
        if self.estimated_cost_usd < 0.05:
            return "minimum"
        if self.estimated_cost_usd < 0.50:
            return "fast"
        return "deep"

    def __repr__(self) -> str:
        return (
            f"CostEstimate(tokens={self.estimated_tokens}, "
            f"cost=${self.estimated_cost_usd:.4f}, level={self.budget_level})"
        )


class CostEstimator:
    """
    Pre-run token and cost estimator.
    Deterministic — no LLM calls.

    Usage:
        estimator = CostEstimator()
        estimate = estimator.estimate_plan(plan, agent_config)
        plan = estimator.annotate_plan(plan, agent_config)  # fills plan fields
    """

    def estimate_plan(
        self,
        plan: ExecutionPlan,
        agent_config: AgentConfig,
    ) -> CostEstimate:
        task_type = agent_config.task_type
        tokens_per_cycle = _TOKENS_PER_CYCLE.get(task_type, 1000)
        expected_cycles = _EXPECTED_CYCLES.get(task_type, 5)
        pattern_mult = _PATTERN_MULTIPLIER.get(plan.pattern, 1.0)
        num_roles = len(plan.all_agent_roles())

        tokens_per_role = tokens_per_cycle * expected_cycles
        total_tokens = int(tokens_per_role * num_roles * pattern_mult)

        model_id = agent_config.get_model_config().model_id
        cost_per_1k = _COST_PER_1K.get(model_id, _COST_PER_1K["default"])
        total_cost = round((total_tokens / 1000) * cost_per_1k, 6)

        breakdown = {
            r.role: int(tokens_per_role * pattern_mult)
            for r in plan.all_agent_roles()
        }

        return CostEstimate(
            estimated_tokens=total_tokens,
            estimated_cost_usd=total_cost,
            breakdown=breakdown,
        )

    def estimate_single(
        self,
        agent_config: AgentConfig,
        max_cycles: int | None = None,
    ) -> CostEstimate:
        """Estimate cost for a single agent without a full plan."""
        task_type = agent_config.task_type
        tokens_per_cycle = _TOKENS_PER_CYCLE.get(task_type, 1000)
        cycles = max_cycles or _EXPECTED_CYCLES.get(task_type, 5)
        total_tokens = tokens_per_cycle * cycles

        model_id = agent_config.get_model_config().model_id
        cost_per_1k = _COST_PER_1K.get(model_id, _COST_PER_1K["default"])
        total_cost = round((total_tokens / 1000) * cost_per_1k, 6)

        return CostEstimate(
            estimated_tokens=total_tokens,
            estimated_cost_usd=total_cost,
            breakdown={"main": total_tokens},
        )

    def annotate_plan(
        self,
        plan: ExecutionPlan,
        agent_config: AgentConfig,
    ) -> ExecutionPlan:
        """Fill plan.estimated_tokens and plan.estimated_cost_usd in-place."""
        estimate = self.estimate_plan(plan, agent_config)
        plan.estimated_tokens = estimate.estimated_tokens
        plan.estimated_cost_usd = estimate.estimated_cost_usd
        return plan
