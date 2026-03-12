"""
GET  /agents                           — List agents for the authenticated user.
POST /agents/{agent_id}/wake           — Manually wake a sleeping agent.
POST /agents/{agent_id}/approve-tool   — Approve or deny a tool action.
GET  /agents/{agent_id}/explain        — Human-readable explanation of last action.
"""
from __future__ import annotations

import uuid

import structlog
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from grif.api.deps import get_gateway, get_session, get_user_id
from grif.api.schemas import (
    AgentExplanationResponse,
    AgentSummary,
    ApproveToolRequest,
    ApproveToolResponse,
    WakeAgentRequest,
    WakeAgentResponse,
)
from grif.audit.logger import AgentExplainer
from grif.llm.gateway import LLMGateway
from grif.models.db import AgentDB, AgentLogDB
from grif.models.enums import AgentState, WakeTriggerType
from grif.runtime.wake_manager import WakeManager

log = structlog.get_logger(__name__)
router = APIRouter()

_WAKEABLE_STATES = {AgentState.SLEEPING, AgentState.RECURRING}


# ─── GET /agents ──────────────────────────────────────────────────────────────

@router.get("", response_model=list[AgentSummary])
async def list_agents(
    state: str | None = None,
    task_type: str | None = None,
    session: AsyncSession = Depends(get_session),
    user_id: str = Depends(get_user_id),
) -> list[AgentSummary]:
    """List agents for the authenticated user, excluding archived ones."""
    query = select(AgentDB).where(
        AgentDB.user_id == user_id,
        AgentDB.state != AgentState.ARCHIVED,
    )
    if state:
        query = query.where(AgentDB.state == state)
    if task_type:
        query = query.where(AgentDB.task_type == task_type)

    result = await session.execute(query.order_by(AgentDB.created_at.desc()))
    agents = result.scalars().all()
    return [AgentSummary.from_db(a) for a in agents]


# ─── POST /agents/{agent_id}/wake ─────────────────────────────────────────────

@router.post("/{agent_id}/wake", response_model=WakeAgentResponse)
async def wake_agent(
    agent_id: str,
    req: WakeAgentRequest,
    session: AsyncSession = Depends(get_session),
    user_id: str = Depends(get_user_id),
) -> WakeAgentResponse:
    """Manually wake a sleeping or recurring agent."""
    agent = await _get_agent_or_404(agent_id, user_id, session)

    if agent.state not in _WAKEABLE_STATES:
        raise HTTPException(
            status_code=400,
            detail=f"Agent is in state '{agent.state}'; can only wake sleeping/recurring agents",
        )

    previous_state = agent.state
    wake_mgr = WakeManager(session)
    ctx = await wake_mgr.wake(
        agent_id=agent_id,
        trigger_type=WakeTriggerType.MANUAL,
        trigger_message=req.message or "Manual wake via API",
    )

    return WakeAgentResponse(
        agent_id=agent_id,
        previous_state=previous_state,
        new_state=AgentState.ACTIVE,
        context_summary=ctx.context_summary,
    )


# ─── POST /agents/{agent_id}/approve-tool ─────────────────────────────────────

@router.post("/{agent_id}/approve-tool", response_model=ApproveToolResponse)
async def approve_tool(
    agent_id: str,
    req: ApproveToolRequest,
    session: AsyncSession = Depends(get_session),
    user_id: str = Depends(get_user_id),
) -> ApproveToolResponse:
    """Approve or deny a specific tool for this agent."""
    agent = await _get_agent_or_404(agent_id, user_id, session)

    # Mutate the tool_permissions list inside the JSONB config
    config = dict(agent.config)
    perms: list[dict] = list(config.get("tool_permissions", []))

    found = False
    for perm in perms:
        if perm.get("tool_name") == req.tool_name:
            perm["auto_approved"] = req.approved
            if req.approved:
                perm["approval_count"] = perm.get("approval_count", 0) + 1
            found = True
            break

    if not found:
        perms.append({
            "tool_name": req.tool_name,
            "auto_approved": req.approved,
            "approval_count": 1 if req.approved else 0,
            "trust_threshold": 3,
        })

    config["tool_permissions"] = perms
    agent.config = config
    await session.flush()

    action = "approved" if req.approved else "denied"
    log.info(
        "tool_approval_updated",
        agent_id=agent_id,
        tool=req.tool_name,
        approved=req.approved,
    )
    return ApproveToolResponse(
        agent_id=agent_id,
        tool_name=req.tool_name,
        approved=req.approved,
        message=f"Tool '{req.tool_name}' {action} for agent {agent_id}",
    )


# ─── GET /agents/{agent_id}/explain ───────────────────────────────────────────

@router.get("/{agent_id}/explain", response_model=AgentExplanationResponse)
async def explain_agent(
    agent_id: str,
    session: AsyncSession = Depends(get_session),
    gateway: LLMGateway = Depends(get_gateway),
    user_id: str = Depends(get_user_id),
) -> AgentExplanationResponse:
    """Return a human-readable explanation of the agent's last ReAct cycle."""
    await _get_agent_or_404(agent_id, user_id, session)

    # Fetch last log entry
    logs_result = await session.execute(
        select(AgentLogDB)
        .where(AgentLogDB.agent_id == uuid.UUID(agent_id))
        .order_by(AgentLogDB.created_at.desc())
        .limit(1)
    )
    last_log = logs_result.scalar_one_or_none()

    if not last_log:
        return AgentExplanationResponse(
            agent_id=agent_id,
            explanation="Agent has not started any work yet.",
            last_action=None,
            last_cycle=None,
        )

    explainer = AgentExplainer(gateway)
    explanation = await explainer.explain(
        action=last_log.action,
        context=last_log.thought,
        result=last_log.observation,
        user_id=user_id,
    )

    return AgentExplanationResponse(
        agent_id=agent_id,
        explanation=explanation,
        last_action=last_log.action,
        last_cycle=last_log.cycle_number,
    )


# ─── Helper ───────────────────────────────────────────────────────────────────

async def _get_agent_or_404(
    agent_id: str,
    user_id: str,
    session: AsyncSession,
) -> AgentDB:
    try:
        agent_uuid = uuid.UUID(agent_id)
    except ValueError:
        raise HTTPException(status_code=404, detail="Agent not found")

    result = await session.execute(
        select(AgentDB).where(
            AgentDB.id == agent_uuid,
            AgentDB.user_id == user_id,
        )
    )
    agent = result.scalar_one_or_none()
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")
    return agent
