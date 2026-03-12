"""
DynamicReplanner — rebuilds an ExecutionPlan when one or more agents fail.

No LLM calls — deterministic recovery logic.

Recovery strategies:
  1. Agent failure (ESCALATE decision):
     - If a non-critical role fails → remove it, increment version
     - If a critical role fails (order=1 in first phase) → escalate to user
  2. Too many failures (>= MAX_FAILURES) → mark plan as failed
  3. Partial success (some order=1 parallel agents succeed) → continue with winners
"""

from __future__ import annotations

import structlog

from grif.models.execution_plan import AgentRole, ExecutionPlan, Phase

log = structlog.get_logger(__name__)

_MAX_FAILURES = 3


class ReplanResult:
    """Result of a replan attempt."""

    def __init__(
        self,
        plan: ExecutionPlan | None,
        should_escalate: bool,
        escalation_reason: str = "",
    ) -> None:
        self.plan = plan                    # Updated plan (None if escalate)
        self.should_escalate = should_escalate
        self.escalation_reason = escalation_reason


class DynamicReplanner:
    """
    Dynamically adjusts the ExecutionPlan when agents fail.

    Usage:
        replanner = DynamicReplanner()
        result = replanner.replan(
            plan=current_plan,
            failed_roles=["researcher_a"],
            failure_reason="LLM timeout on cycle 20",
        )
        if result.should_escalate:
            # Notify user
        else:
            # Continue with result.plan
    """

    def replan(
        self,
        plan: ExecutionPlan,
        failed_roles: list[str],
        failure_reason: str = "",
    ) -> ReplanResult:
        """
        Attempt to recover from agent failures.
        Returns updated plan or escalation signal.
        """
        if not failed_roles:
            return ReplanResult(plan=plan, should_escalate=False)

        # Check failure count threshold
        total_failures = (plan.version - 1) + len(failed_roles)
        if total_failures >= _MAX_FAILURES:
            reason = (
                f"Too many failures ({total_failures}). "
                f"Latest: {failure_reason or 'unknown'}. "
                f"Failed roles: {', '.join(failed_roles)}"
            )
            log.warning("replan_escalate_max_failures", plan_id=str(plan.plan_id), failures=total_failures)
            return ReplanResult(plan=None, should_escalate=True, escalation_reason=reason)

        failed_set = set(failed_roles)

        # Escalate only if ALL order-1 agents in first phase fail (no survivors)
        first_phase_order1 = {
            r.role for r in plan.phases[0].agents if r.order == 1
        } if plan.phases else set()

        surviving_order1 = first_phase_order1 - failed_set
        if first_phase_order1 and not surviving_order1:
            # Every order-1 agent in phase-1 failed — nothing to continue with
            reason = (
                f"All first-phase order-1 agents failed: "
                f"{', '.join(first_phase_order1)}. "
                f"Reason: {failure_reason}"
            )
            log.warning("replan_escalate_critical_failure", plan_id=str(plan.plan_id), roles=list(first_phase_order1))
            return ReplanResult(plan=None, should_escalate=True, escalation_reason=reason)

        # Non-critical failure: remove failed roles and adjust plan
        new_phases = []
        removed: list[str] = []

        for phase in plan.phases:
            surviving_agents = [a for a in phase.agents if a.role not in failed_set]

            if not surviving_agents:
                # Entire phase removed → escalate
                reason = f"Phase '{phase.name}' lost all agents to failures."
                log.warning("replan_escalate_empty_phase", phase=phase.name)
                return ReplanResult(plan=None, should_escalate=True, escalation_reason=reason)

            removed.extend([a.role for a in phase.agents if a.role in failed_set])
            new_phases.append(
                Phase(
                    name=phase.name,
                    mode=phase.mode,
                    schedule=phase.schedule,
                    timezone=phase.timezone,
                    agents=surviving_agents,
                )
            )

        updated = ExecutionPlan(
            plan_id=plan.plan_id,
            task_id=plan.task_id,
            user_id=plan.user_id,
            pattern=plan.pattern,
            phases=new_phases,
            version=plan.version + 1,
            replan_reason=f"Removed failed roles: {', '.join(removed)}. {failure_reason}",
        )

        log.info(
            "replanned",
            plan_id=str(plan.plan_id),
            version=updated.version,
            removed=removed,
        )
        return ReplanResult(plan=updated, should_escalate=False)

    def retry_role(
        self,
        plan: ExecutionPlan,
        role_name: str,
        new_goal: str | None = None,
    ) -> ExecutionPlan:
        """
        Clone a failed role with an optional revised goal for retry.
        Bumps plan version.
        """
        new_phases = []
        for phase in plan.phases:
            new_agents = []
            for agent in phase.agents:
                if agent.role == role_name:
                    new_agents.append(
                        AgentRole(
                            role=agent.role,
                            goal=new_goal or agent.goal,
                            order=agent.order,
                            blueprint_id=agent.blueprint_id,
                            model_config_override=agent.model_config_override,
                            tools=agent.tools,
                            agent_id=None,  # Reset: needs re-spawn
                        )
                    )
                else:
                    new_agents.append(agent)
            new_phases.append(
                Phase(
                    name=phase.name,
                    mode=phase.mode,
                    schedule=phase.schedule,
                    timezone=phase.timezone,
                    agents=new_agents,
                )
            )

        return ExecutionPlan(
            plan_id=plan.plan_id,
            task_id=plan.task_id,
            user_id=plan.user_id,
            pattern=plan.pattern,
            phases=new_phases,
            version=plan.version + 1,
            replan_reason=f"Retrying role '{role_name}'" + (f" with new goal" if new_goal else ""),
        )
