"""
POST /tasks                           — Submit a user task, run full GRIF pipeline.
POST /clarification/{task_id}/answer  — Provide answers to clarification questions.
"""
from __future__ import annotations

import uuid
from typing import Any

import structlog
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from grif.api import progress
from grif.api.deps import get_gateway, get_session, get_user_id
from grif.api.schemas import (
    AnswerClarificationRequest,
    ClarificationQuestionSchema,
    SubmitTaskRequest,
    TaskResponse,
)
from grif.cost.estimator import CostEstimator
from grif.database import AsyncSessionFactory
from grif.llm.gateway import LLMGateway
from grif.models.db import AgentDB, TaskDB
from grif.models.intent import ClassifiedIntent, StructuredIntent
from grif.orchestrator.planner import MultiAgentPlanner
from grif.pipeline.agent_spawner import AgentSpawner
from grif.pipeline.blueprint_selector import BlueprintSelector
from grif.pipeline.clarification import ClarificationPhase
from grif.pipeline.config_generator import ConfigGenerator
from grif.pipeline.intent_classifier import IntentClassifier
from grif.pipeline.signal_parser import SignalParser
from grif.runtime.memory_manager import MemoryManager
from grif.runtime.react_loop import ReActGraph
from grif.tools.registry import ToolRegistry

log = structlog.get_logger(__name__)
router = APIRouter()

_planner = MultiAgentPlanner()
_estimator = CostEstimator()
_signal_parser = SignalParser()


# ─── POST /tasks ──────────────────────────────────────────────────────────────

@router.post("", status_code=status.HTTP_201_CREATED, response_model=TaskResponse)
async def submit_task(
    req: SubmitTaskRequest,
    background_tasks: BackgroundTasks,
    session: AsyncSession = Depends(get_session),
    gateway: LLMGateway = Depends(get_gateway),
    user_id: str = Depends(get_user_id),
) -> TaskResponse:
    """
    Full GRIF pipeline — Steps 1-7:
    Parse signal → Classify intent → Clarification? → Blueprint →
    Config → Plan → Spawn → ReAct (background).
    """
    task_id = str(uuid.uuid4())

    # Persist task record immediately so /ws can start subscribing
    task = TaskDB(
        id=uuid.UUID(task_id),
        user_id=user_id,
        raw_input=req.text,
        signal_type=req.signal_type,
        status="pending",
    )
    session.add(task)
    await session.flush()

    # Step 1: Parse signal
    signal = _signal_parser.parse_text(req.text, user_id=user_id, metadata=req.metadata)
    await progress.publish(task_id, {"event_type": "task_received", "task_id": task_id})

    # Step 2: Classify intent (1 Haiku LLM call)
    classifier = IntentClassifier(gateway)
    try:
        intent = await classifier.classify(signal)
    except Exception as exc:
        log.error("intent_classify_failed", task_id=task_id, error=str(exc))
        task.status = "failed"
        await session.flush()
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Intent classification failed: {exc}",
        )

    task.classified_intent = intent.model_dump(mode="json")
    await progress.publish(task_id, {
        "event_type": "intent_classified",
        "task_id": task_id,
        "data": {"task_type": intent.task_type, "complexity": intent.complexity},
    })

    # Step 2.5: Clarification check
    clarification_phase = ClarificationPhase(gateway)
    classified = await clarification_phase.process(intent)

    if classified.clarification_needed and classified.clarification_request:
        task.status = "clarifying"
        task.pending_clarification = classified.model_dump(mode="json")
        await session.flush()

        questions = [
            ClarificationQuestionSchema(
                field_name=q.field_name,
                question=q.question,
                options=q.options,
            )
            for q in (classified.clarification_request.questions or [])
        ]
        return TaskResponse(
            task_id=task_id,
            status="clarifying",
            clarification_questions=questions,
            message="Please answer the following questions to proceed.",
        )

    # Steps 3-7: Blueprint → Config → Plan → Spawn agents
    task.status = "planning"
    await session.flush()

    try:
        agent_ids, cost_info = await _run_planning_phase(
            task_id=task_id,
            intent=intent,
            user_id=user_id,
            session=session,
            gateway=gateway,
        )
    except Exception as exc:
        log.error("planning_phase_failed", task_id=task_id, error=str(exc))
        task.status = "failed"
        await session.flush()
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Planning failed: {exc}",
        )

    task.status = "running"
    await session.flush()

    # Launch ReAct loop in background (non-blocking)
    for agent_id in agent_ids:
        background_tasks.add_task(
            _run_agent_background,
            task_id=task_id,
            agent_id=agent_id,
            user_message=req.text,
        )

    return TaskResponse(
        task_id=task_id,
        status="running",
        agent_ids=agent_ids,
        estimated_cost=cost_info,
    )


# ─── POST /clarification/{task_id}/answer ─────────────────────────────────────

@router.post("/clarification/{task_id}/answer", response_model=TaskResponse)
async def answer_clarification(
    task_id: str,
    req: AnswerClarificationRequest,
    background_tasks: BackgroundTasks,
    session: AsyncSession = Depends(get_session),
    gateway: LLMGateway = Depends(get_gateway),
    user_id: str = Depends(get_user_id),
) -> TaskResponse:
    """
    Submit answers to clarification questions and continue the pipeline.
    Task must be in 'clarifying' status.
    """
    result = await session.execute(
        select(TaskDB).where(
            TaskDB.id == uuid.UUID(task_id),
            TaskDB.user_id == user_id,
        )
    )
    task = result.scalar_one_or_none()
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    if task.status != "clarifying":
        raise HTTPException(
            status_code=400,
            detail=f"Task is in status '{task.status}', expected 'clarifying'",
        )

    if not task.pending_clarification:
        raise HTTPException(status_code=400, detail="No pending clarification data found")

    # Restore classified intent and merge answers
    classified = ClassifiedIntent.model_validate(task.pending_clarification)
    clarification_phase = ClarificationPhase(gateway)
    answers_raw = [
        {"field_name": a.field_name, "value": a.value}
        for a in req.answers
    ]
    updated = clarification_phase.apply_answers(classified, answers_raw)
    intent = updated.structured_intent

    task.clarification_answers = answers_raw
    task.status = "planning"
    await session.flush()

    try:
        agent_ids, cost_info = await _run_planning_phase(
            task_id=task_id,
            intent=intent,
            user_id=user_id,
            session=session,
            gateway=gateway,
        )
    except Exception as exc:
        log.error("planning_phase_failed_after_clarification", task_id=task_id, error=str(exc))
        task.status = "failed"
        await session.flush()
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Planning failed: {exc}",
        )

    task.status = "running"
    await session.flush()

    for agent_id in agent_ids:
        background_tasks.add_task(
            _run_agent_background,
            task_id=task_id,
            agent_id=agent_id,
            user_message=task.raw_input,
        )

    return TaskResponse(
        task_id=task_id,
        status="running",
        agent_ids=agent_ids,
        estimated_cost=cost_info,
    )


# ─── Pipeline helper (steps 3-7) ──────────────────────────────────────────────

async def _run_planning_phase(
    task_id: str,
    intent: StructuredIntent,
    user_id: str,
    session: AsyncSession,
    gateway: LLMGateway,
) -> tuple[list[str], dict[str, Any]]:
    """
    Steps 3-7: Blueprint → Config (1 Sonnet call) → Plan → Cost → Spawn.
    Returns (agent_ids, cost_info).
    """
    # Step 4: Blueprint selection (deterministic)
    selector = BlueprintSelector(session)
    blueprint = await selector.select(intent, user_id=user_id)

    # Step 5: Config generation (1 Sonnet LLM call)
    config_gen = ConfigGenerator(gateway)
    config = await config_gen.generate(intent, blueprint, user_id=user_id)

    # Step 6: Build execution plan + estimate cost
    plan = _planner.plan(intent, config, task_id)
    estimate = _estimator.estimate_plan(plan, config)
    cost_info: dict[str, Any] = {
        "estimated_tokens": estimate.estimated_tokens,
        "estimated_cost_usd": round(estimate.estimated_cost_usd, 4),
        "budget_level": estimate.budget_level,
    }

    # Step 7: Spawn agent(s)
    spawner = AgentSpawner(session)
    agent = await spawner.spawn(config, task_id=task_id)

    await progress.publish(task_id, {
        "event_type": "agent_spawned",
        "task_id": task_id,
        "data": {
            "agent_id": str(agent.id),
            "blueprint": blueprint.id,
            "budget_level": estimate.budget_level,
        },
    })

    return [str(agent.id)], cost_info


# ─── Background ReAct runner ──────────────────────────────────────────────────

async def _run_agent_background(
    task_id: str,
    agent_id: str,
    user_message: str,
) -> None:
    """Run ReAct loop for a spawned agent. Creates its own DB session."""
    async with AsyncSessionFactory() as session:
        async with session.begin():
            try:
                result = await session.execute(
                    select(AgentDB).where(AgentDB.id == uuid.UUID(agent_id))
                )
                agent_db = result.scalar_one_or_none()
                if not agent_db:
                    log.error("agent_not_found_for_run", agent_id=agent_id)
                    return

                from grif.models.agent_config import AgentConfig
                config = AgentConfig.model_validate(agent_db.config)

                gateway = LLMGateway()
                registry = ToolRegistry()
                memory = MemoryManager(
                    agent_id=agent_id,
                    user_id=agent_db.user_id,
                    session=session,
                    gateway=gateway,
                )
                graph = ReActGraph(
                    gateway=gateway,
                    registry=registry,
                    memory=memory,
                    agent_config=config,
                    session=session,
                )

                await progress.publish(task_id, {
                    "event_type": "react_started",
                    "task_id": task_id,
                    "data": {"agent_id": agent_id},
                })

                final_state = await graph.run(user_message=user_message)

                # Update task status
                task_result = await session.execute(
                    select(TaskDB).where(TaskDB.id == uuid.UUID(task_id))
                )
                task = task_result.scalar_one_or_none()
                if task:
                    task.status = "done" if final_state.get("final_result") else "failed"

                await progress.publish(task_id, {
                    "event_type": "task_complete",
                    "task_id": task_id,
                    "data": {
                        "agent_id": agent_id,
                        "decision": final_state.get("decision", ""),
                        "result": (final_state.get("final_result") or "")[:500],
                    },
                })

            except Exception as exc:
                log.error("agent_background_run_failed", agent_id=agent_id, error=str(exc))
                await progress.publish(task_id, {
                    "event_type": "error",
                    "task_id": task_id,
                    "data": {"agent_id": agent_id, "error": str(exc)},
                })
