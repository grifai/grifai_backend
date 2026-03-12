"""Tests for evaluation/evaluator.py — mocked LLM and DB."""
import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest

from grif.evaluation.evaluator import EvaluationResult, SelfEvaluator


# ─── Fixtures ─────────────────────────────────────────────────────────────────

def _make_evaluator(
    accuracy: float = 4.0,
    completeness: float = 5.0,
    efficiency: float = 3.0,
    comment: str = "Good result",
) -> tuple[SelfEvaluator, AsyncMock, AsyncMock]:
    gateway = AsyncMock()
    llm_response = MagicMock()
    llm_response.parsed = {
        "accuracy": accuracy,
        "completeness": completeness,
        "efficiency": efficiency,
        "comment": comment,
    }
    gateway.complete_json = AsyncMock(return_value=llm_response)

    # Agent with blueprint
    agent = MagicMock()
    agent.id = uuid.uuid4()
    agent.avg_score = None
    agent.eval_count = 0
    agent.blueprint_id = "generic_worker"

    session = AsyncMock()
    session.add = MagicMock()
    session.flush = AsyncMock()
    result_mock = MagicMock()
    result_mock.scalar_one_or_none.return_value = agent
    session.execute = AsyncMock(return_value=result_mock)

    evaluator = SelfEvaluator(gateway=gateway, session=session)
    return evaluator, gateway, session


# ─── Basic evaluation ──────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_evaluate_returns_result() -> None:
    evaluator, _, _ = _make_evaluator()
    result = await evaluator.evaluate(
        agent_id=str(uuid.uuid4()),
        user_request="Find hotels in Paris",
        agent_result="Found 5 hotels",
    )
    assert isinstance(result, EvaluationResult)


@pytest.mark.asyncio
async def test_evaluate_calls_llm_once() -> None:
    evaluator, gateway, _ = _make_evaluator()
    await evaluator.evaluate(
        agent_id=str(uuid.uuid4()),
        user_request="Find hotels",
        agent_result="Hotels: ...",
    )
    gateway.complete_json.assert_called_once()


@pytest.mark.asyncio
async def test_evaluate_scores_correct() -> None:
    evaluator, _, _ = _make_evaluator(accuracy=4, completeness=5, efficiency=3)
    result = await evaluator.evaluate(
        agent_id=str(uuid.uuid4()),
        user_request="Task",
        agent_result="Result",
    )
    assert result.accuracy == 4.0
    assert result.completeness == 5.0
    assert result.efficiency == 3.0


@pytest.mark.asyncio
async def test_evaluate_overall_is_average() -> None:
    evaluator, _, _ = _make_evaluator(accuracy=3, completeness=3, efficiency=3)
    result = await evaluator.evaluate(
        agent_id=str(uuid.uuid4()),
        user_request="Task",
        agent_result="Result",
    )
    assert result.overall == 3.0


@pytest.mark.asyncio
async def test_evaluate_comment_preserved() -> None:
    evaluator, _, _ = _make_evaluator(comment="Excellent result, very complete")
    result = await evaluator.evaluate(
        agent_id=str(uuid.uuid4()),
        user_request="Task",
        agent_result="Result",
    )
    assert result.comment == "Excellent result, very complete"


# ─── Score clamping ───────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_evaluate_clamps_scores_above_5() -> None:
    evaluator, _, _ = _make_evaluator(accuracy=10, completeness=7, efficiency=6)
    result = await evaluator.evaluate(
        agent_id=str(uuid.uuid4()),
        user_request="Task",
        agent_result="Result",
    )
    assert result.accuracy <= 5.0
    assert result.completeness <= 5.0
    assert result.efficiency <= 5.0


@pytest.mark.asyncio
async def test_evaluate_clamps_scores_below_1() -> None:
    evaluator, _, _ = _make_evaluator(accuracy=0, completeness=-1, efficiency=0.5)
    result = await evaluator.evaluate(
        agent_id=str(uuid.uuid4()),
        user_request="Task",
        agent_result="Result",
    )
    assert result.accuracy >= 1.0
    assert result.completeness >= 1.0


# ─── LLM failure fallback ─────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_evaluate_fallback_on_llm_error() -> None:
    evaluator, gateway, _ = _make_evaluator()
    gateway.complete_json = AsyncMock(side_effect=RuntimeError("LLM down"))
    result = await evaluator.evaluate(
        agent_id=str(uuid.uuid4()),
        user_request="Task",
        agent_result="Result",
    )
    # Falls back to score=3
    assert result.accuracy == 3.0
    assert result.completeness == 3.0
    assert result.efficiency == 3.0


@pytest.mark.asyncio
async def test_evaluate_fallback_on_empty_parsed() -> None:
    evaluator, gateway, _ = _make_evaluator()
    llm_response = MagicMock()
    llm_response.parsed = {}
    gateway.complete_json = AsyncMock(return_value=llm_response)
    result = await evaluator.evaluate(
        agent_id=str(uuid.uuid4()),
        user_request="Task",
        agent_result="Result",
    )
    assert result.overall == 3.0


# ─── DB persistence ───────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_evaluate_updates_agent_avg_score() -> None:
    agent_id = str(uuid.uuid4())
    evaluator, _, session = _make_evaluator(accuracy=5, completeness=5, efficiency=5)
    # Agent has no prior score
    agent = session.execute.return_value.scalar_one_or_none.return_value
    agent.avg_score = None
    agent.eval_count = 0

    await evaluator.evaluate(
        agent_id=agent_id,
        user_request="Task",
        agent_result="Result",
    )
    assert agent.avg_score == 5.0
    assert agent.eval_count == 1


@pytest.mark.asyncio
async def test_evaluate_rolling_average() -> None:
    agent_id = str(uuid.uuid4())
    evaluator, _, session = _make_evaluator(accuracy=5, completeness=5, efficiency=5)
    agent = session.execute.return_value.scalar_one_or_none.return_value
    agent.avg_score = 3.0
    agent.eval_count = 1  # Already had one eval

    await evaluator.evaluate(
        agent_id=agent_id,
        user_request="Task",
        agent_result="Result",
    )
    # Rolling avg: (3.0 * 1 + 5.0) / 2 = 4.0
    assert agent.avg_score == 4.0
    assert agent.eval_count == 2


@pytest.mark.asyncio
async def test_evaluate_creates_blueprint_score() -> None:
    agent_id = str(uuid.uuid4())
    evaluator, _, session = _make_evaluator()
    await evaluator.evaluate(
        agent_id=agent_id,
        user_request="Task",
        agent_result="Result",
    )
    # BlueprintScoreDB entry added (agent has blueprint_id)
    session.add.assert_called_once()


@pytest.mark.asyncio
async def test_evaluate_persist_failure_does_not_crash() -> None:
    """Score persist failure should be swallowed."""
    agent_id = str(uuid.uuid4())
    evaluator, _, session = _make_evaluator()
    session.flush = AsyncMock(side_effect=RuntimeError("DB error"))
    # Should not raise
    result = await evaluator.evaluate(
        agent_id=agent_id,
        user_request="Task",
        agent_result="Result",
    )
    assert isinstance(result, EvaluationResult)


# ─── EvaluationResult ─────────────────────────────────────────────────────────

def test_evaluation_result_overall() -> None:
    r = EvaluationResult(accuracy=4, completeness=5, efficiency=3, comment="ok", agent_id="a")
    assert r.overall == 4.0


def test_evaluation_result_repr() -> None:
    r = EvaluationResult(accuracy=4, completeness=4, efficiency=4, comment="ok", agent_id="a")
    assert "overall=4.0" in repr(r)
