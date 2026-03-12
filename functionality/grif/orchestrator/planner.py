"""
MultiAgentPlanner — builds ExecutionPlan from a single user intent.

No LLM calls — pure deterministic logic.

Pattern selection rules:
  SIMPLE / single-step          → PIPELINE (1 agent)
  RESEARCH / COACH / MONITOR    → PIPELINE (researcher → writer)
  COMPARE                       → PARALLEL_MERGE (2 researchers → comparator)
  GENERATE / OPERATE (content)  → PIPELINE_REVIEW (generator → critic)
  EXECUTE (complex) / OPERATE   → HYPOTHESIS_TESTING (two parallel tracks → select)
"""

from __future__ import annotations

import structlog

from grif.models.agent_config import AgentConfig
from grif.models.enums import PlanPattern, TaskType
from grif.models.execution_plan import AgentRole, ExecutionPlan, Phase, PhaseMode
from grif.models.intent import StructuredIntent

log = structlog.get_logger(__name__)

# ── Pattern selection ─────────────────────────────────────────────────────────

_PIPELINE_TYPES = {TaskType.RESEARCH, TaskType.COACH, TaskType.MONITOR, TaskType.REMIND}
_PARALLEL_MERGE_TYPES = {TaskType.COMPARE, TaskType.SEARCH}
_PIPELINE_REVIEW_TYPES = {TaskType.GENERATE}
_HYPOTHESIS_TYPES = {TaskType.EXECUTE, TaskType.OPERATE}


def _select_pattern(intent: StructuredIntent) -> PlanPattern:
    """Deterministically select plan pattern from task type."""
    tt = intent.task_type
    if tt in _PARALLEL_MERGE_TYPES:
        return PlanPattern.PARALLEL_MERGE
    if tt in _PIPELINE_REVIEW_TYPES:
        return PlanPattern.PIPELINE_REVIEW
    if tt in _HYPOTHESIS_TYPES:
        return PlanPattern.HYPOTHESIS_TESTING
    if tt in _PIPELINE_TYPES:
        return PlanPattern.PIPELINE
    return PlanPattern.PIPELINE  # default


# ── Phase builders ────────────────────────────────────────────────────────────

def _build_pipeline(intent: StructuredIntent, config: AgentConfig) -> list[Phase]:
    """A → B sequential plan."""
    tools = config.tools or []
    return [
        Phase(
            name="execute",
            mode=PhaseMode.ONE_SHOT,
            agents=[
                AgentRole(
                    role="executor",
                    goal=intent.raw_input,
                    order=1,
                    blueprint_id=config.blueprint_id,
                    tools=tools,
                )
            ],
        )
    ]


def _build_pipeline_research(intent: StructuredIntent, config: AgentConfig) -> list[Phase]:
    """researcher (order=1) → writer (order=2) sequentially."""
    tools = config.tools or []
    read_tools = [t for t in tools if "search" in t or "fetch" in t or "read" in t]
    write_tools = [t for t in tools if t not in read_tools]
    return [
        Phase(
            name="research_and_write",
            mode=PhaseMode.ONE_SHOT,
            agents=[
                AgentRole(
                    role="researcher",
                    goal=f"Research and gather information: {intent.raw_input}",
                    order=1,
                    blueprint_id=config.blueprint_id,
                    tools=read_tools or tools,
                ),
                AgentRole(
                    role="writer",
                    goal=f"Synthesize research findings and produce final output for: {intent.raw_input}",
                    order=2,
                    blueprint_id=config.blueprint_id,
                    tools=write_tools or tools,
                ),
            ],
        )
    ]


def _build_parallel_merge(intent: StructuredIntent, config: AgentConfig) -> list[Phase]:
    """researcher_A ∥ researcher_B (order=1) → comparator (order=2)."""
    tools = config.tools or []
    entities = list(intent.entities.values())
    subject_a = str(entities[0]) if len(entities) >= 1 else "option A"
    subject_b = str(entities[1]) if len(entities) >= 2 else "option B"
    return [
        Phase(
            name="compare",
            mode=PhaseMode.ONE_SHOT,
            agents=[
                AgentRole(
                    role="researcher_a",
                    goal=f"Research {subject_a}: {intent.raw_input}",
                    order=1,
                    blueprint_id=config.blueprint_id,
                    tools=tools,
                ),
                AgentRole(
                    role="researcher_b",
                    goal=f"Research {subject_b}: {intent.raw_input}",
                    order=1,
                    blueprint_id=config.blueprint_id,
                    tools=tools,
                ),
                AgentRole(
                    role="comparator",
                    goal=f"Compare and summarize findings: {intent.raw_input}",
                    order=2,
                    blueprint_id=config.blueprint_id,
                    tools=[],
                ),
            ],
        )
    ]


def _build_pipeline_review(intent: StructuredIntent, config: AgentConfig) -> list[Phase]:
    """generator (order=1) → critic (order=2, review loop)."""
    tools = config.tools or []
    return [
        Phase(
            name="generate_and_review",
            mode=PhaseMode.ONE_SHOT,
            agents=[
                AgentRole(
                    role="generator",
                    goal=f"Generate content for: {intent.raw_input}",
                    order=1,
                    blueprint_id=config.blueprint_id,
                    tools=tools,
                ),
                AgentRole(
                    role="critic",
                    goal=f"Review and improve generated content for: {intent.raw_input}",
                    order=2,
                    blueprint_id=config.blueprint_id,
                    tools=[],
                ),
            ],
        )
    ]


def _build_hypothesis_testing(intent: StructuredIntent, config: AgentConfig) -> list[Phase]:
    """track_A ∥ track_B (order=1) → selector (order=2)."""
    tools = config.tools or []
    return [
        Phase(
            name="hypothesis_test",
            mode=PhaseMode.ONE_SHOT,
            agents=[
                AgentRole(
                    role="executor_a",
                    goal=f"Approach A for: {intent.raw_input}",
                    order=1,
                    blueprint_id=config.blueprint_id,
                    tools=tools,
                ),
                AgentRole(
                    role="executor_b",
                    goal=f"Approach B (alternative) for: {intent.raw_input}",
                    order=1,
                    blueprint_id=config.blueprint_id,
                    tools=tools,
                ),
                AgentRole(
                    role="selector",
                    goal=f"Select the best result for: {intent.raw_input}",
                    order=2,
                    blueprint_id=config.blueprint_id,
                    tools=[],
                ),
            ],
        )
    ]


# ── Builder map ───────────────────────────────────────────────────────────────

_BUILDERS = {
    PlanPattern.PIPELINE: _build_pipeline,
    PlanPattern.PARALLEL_MERGE: _build_parallel_merge,
    PlanPattern.PIPELINE_REVIEW: _build_pipeline_review,
    PlanPattern.HYPOTHESIS_TESTING: _build_hypothesis_testing,
}

_RESEARCH_PIPELINES = _PIPELINE_TYPES


# ── Public class ──────────────────────────────────────────────────────────────

class MultiAgentPlanner:
    """
    Builds an ExecutionPlan for a user intent.
    No LLM calls — deterministic pattern + role assignment.

    Usage:
        planner = MultiAgentPlanner()
        plan = planner.plan(intent, agent_config, task_id)
    """

    def plan(
        self,
        intent: StructuredIntent,
        config: AgentConfig,
        task_id: str,
    ) -> ExecutionPlan:
        pattern = _select_pattern(intent)

        if pattern == PlanPattern.PIPELINE and intent.task_type in _RESEARCH_PIPELINES:
            phases = _build_pipeline_research(intent, config)
        else:
            builder = _BUILDERS.get(pattern, _build_pipeline)
            phases = builder(intent, config)

        plan = ExecutionPlan(
            task_id=task_id,
            user_id=config.user_id,
            pattern=pattern,
            phases=phases,
        )
        log.info(
            "plan_built",
            task_id=task_id,
            pattern=pattern,
            phases=len(phases),
            total_roles=len(plan.all_agent_roles()),
        )
        return plan
