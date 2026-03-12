from uuid import UUID, uuid4

from pydantic import BaseModel, Field

from grif.models.agent_config import ModelConfig
from grif.models.enums import PhaseMode, PlanPattern


class AgentRole(BaseModel):
    """
    One agent's role inside a multi-agent execution plan.
    Agents with the same `order` within a phase run in parallel
    via asyncio.gather().
    """

    role: str = Field(description="Role name: researcher, writer, critic, etc.")
    goal: str = Field(description="What this agent is responsible for producing.")
    order: int = Field(
        default=1,
        description="Execution order within the phase. Same order = parallel.",
    )
    blueprint_id: str | None = None
    model_config_override: ModelConfig | None = None
    tools: list[str] = Field(default_factory=list)
    # Set at spawn time
    agent_id: UUID | None = None


class Phase(BaseModel):
    """
    A phase in the execution plan.
    Supports one_shot (runs once at setup) and recurring (runs on schedule).
    """

    name: str
    mode: PhaseMode = PhaseMode.ONE_SHOT
    # Only used when mode == recurring
    schedule: str | None = Field(
        default=None,
        description="Cron expression for recurring phases.",
        examples=["0 9 * * *"],
    )
    timezone: str = "UTC"
    agents: list[AgentRole]

    @property
    def max_order(self) -> int:
        return max((a.order for a in self.agents), default=1)

    def agents_at_order(self, order: int) -> list[AgentRole]:
        return [a for a in self.agents if a.order == order]


class ExecutionPlan(BaseModel):
    """
    Full multi-agent execution plan output from MultiAgentPlanner.
    Phases run sequentially; agents within each order-level run in parallel.
    """

    plan_id: UUID = Field(default_factory=uuid4)
    task_id: str
    user_id: str
    pattern: PlanPattern
    phases: list[Phase]

    # Cost estimation context (filled by CostEstimator)
    estimated_tokens: int | None = None
    estimated_cost_usd: float | None = None

    # Dynamic replanning: increment on each replan
    version: int = 1
    replan_reason: str | None = None

    def all_agent_roles(self) -> list[AgentRole]:
        return [agent for phase in self.phases for agent in phase.agents]
