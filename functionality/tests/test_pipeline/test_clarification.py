"""Tests for pipeline/clarification.py — gap analysis is deterministic, questions mocked."""
import json
import pytest
from unittest.mock import AsyncMock, MagicMock

from grif.llm.gateway import LLMGateway, LLMResponse
from grif.models.enums import ClarificationMode, Complexity, TaskType
from grif.models.intent import StructuredIntent
from grif.pipeline.clarification import ClarificationPhase


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _mock_response(content: str) -> LLMResponse:
    raw = MagicMock()
    raw.choices[0].message.content = content
    raw.usage.prompt_tokens = 20
    raw.usage.completion_tokens = 40
    raw.usage.total_tokens = 60
    return LLMResponse(raw=raw, model_used="gpt-4o-mini")


def _make_intent(
    task_type: TaskType,
    entities: dict | None = None,
    constraints: dict | None = None,
    complexity: Complexity = Complexity.SIMPLE,
) -> StructuredIntent:
    return StructuredIntent(
        task_type=task_type,
        entities=entities or {},
        constraints=constraints or {},
        complexity=complexity,
        raw_input="test input",
    )


# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def gateway_mock() -> AsyncMock:
    gw = AsyncMock(spec=LLMGateway)
    questions_json = json.dumps([
        {"field_name": "topic", "question": "What topic should I research?", "options": None}
    ])
    gw.complete_json.return_value = _mock_response(questions_json)
    return gw


@pytest.fixture
def phase(gateway_mock: AsyncMock) -> ClarificationPhase:
    return ClarificationPhase(gateway=gateway_mock)


# ─── Gap Analysis (deterministic) ─────────────────────────────────────────────

@pytest.mark.asyncio
async def test_no_clarification_when_fields_present(phase: ClarificationPhase) -> None:
    """SEARCH with topic present → no clarification needed."""
    intent = _make_intent(
        TaskType.SEARCH,
        entities={"topic": "Paris hotels"},
    )
    classified = await phase.process(intent)
    assert not classified.clarification_needed
    assert classified.clarification_request is None


@pytest.mark.asyncio
async def test_clarification_needed_for_monitor(
    phase: ClarificationPhase, gateway_mock: AsyncMock
) -> None:
    """MONITOR without item/condition/threshold → needs clarification."""
    gateway_mock.complete_json.return_value = _mock_response(
        json.dumps([
            {"field_name": "item", "question": "What to monitor?"},
            {"field_name": "condition", "question": "What condition?"},
            {"field_name": "threshold", "question": "What threshold?"},
        ])
    )
    intent = _make_intent(TaskType.MONITOR)
    classified = await phase.process(intent)
    assert classified.clarification_needed
    assert len(classified.missing_fields) == 3


@pytest.mark.asyncio
async def test_clarification_needed_for_research_no_topic(
    phase: ClarificationPhase,
) -> None:
    intent = _make_intent(TaskType.RESEARCH)
    classified = await phase.process(intent)
    assert classified.clarification_needed
    assert "topic" in classified.missing_fields


@pytest.mark.asyncio
async def test_no_clarification_for_execute_with_required(
    phase: ClarificationPhase,
) -> None:
    intent = _make_intent(
        TaskType.EXECUTE,
        entities={"action": "send_email", "target": "boss@example.com"},
    )
    classified = await phase.process(intent)
    assert not classified.clarification_needed


# ─── Mode selection ───────────────────────────────────────────────────────────

def test_mode_simple_task(phase: ClarificationPhase) -> None:
    intent = _make_intent(TaskType.SEARCH, complexity=Complexity.SIMPLE)
    assert phase._choose_mode(intent) == ClarificationMode.QUICK_CONFIRM


def test_mode_multi_step_task(phase: ClarificationPhase) -> None:
    intent = _make_intent(TaskType.RESEARCH, complexity=Complexity.MULTI_STEP)
    assert phase._choose_mode(intent) == ClarificationMode.STRUCTURED_INTERVIEW


def test_mode_operate_task(phase: ClarificationPhase) -> None:
    intent = _make_intent(TaskType.OPERATE)
    assert phase._choose_mode(intent) == ClarificationMode.PROGRESSIVE


# ─── Question limit per mode ──────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_quick_confirm_max_2_questions(
    phase: ClarificationPhase, gateway_mock: AsyncMock
) -> None:
    gateway_mock.complete_json.return_value = _mock_response(
        json.dumps([
            {"field_name": "item", "question": "Q1?"},
            {"field_name": "condition", "question": "Q2?"},
        ])
    )
    intent = _make_intent(TaskType.MONITOR, complexity=Complexity.SIMPLE)
    classified = await phase.process(intent)
    # Quick Confirm limits to 2
    assert len(classified.clarification_request.questions) <= 2


# ─── apply_answers ────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_apply_answers_merges_entities(phase: ClarificationPhase) -> None:
    intent = _make_intent(TaskType.RESEARCH)
    classified = await phase.process(intent)

    answers = [{"field_name": "topic", "value": "quantum computing"}]
    updated = phase.apply_answers(classified, answers)

    assert not updated.clarification_needed
    assert updated.structured_intent.entities["topic"] == "quantum computing"


@pytest.mark.asyncio
async def test_apply_answers_puts_limit_in_constraints(phase: ClarificationPhase) -> None:
    intent = _make_intent(TaskType.SEARCH)
    classified = await phase.process(intent)

    answers = [
        {"field_name": "topic", "value": "hotels"},
        {"field_name": "max_price", "value": 200},
    ]
    updated = phase.apply_answers(classified, answers)
    assert updated.structured_intent.constraints.get("max_price") == 200
    assert updated.structured_intent.entities.get("topic") == "hotels"
