"""Tests for orchestrator/conflict_resolver.py — mocked LLM."""
import pytest
from unittest.mock import AsyncMock, MagicMock

from grif.orchestrator.conflict_resolver import ConflictResolution, ConflictResolver


# ─── Fixtures ─────────────────────────────────────────────────────────────────

def _make_resolver(
    winning_result: str = "Hotel A at €150",
    explanation: str = "More accurate price",
    merged: bool = False,
) -> tuple[ConflictResolver, AsyncMock]:
    gateway = AsyncMock()
    response = MagicMock()
    response.parsed = {
        "winning_result": winning_result,
        "explanation": explanation,
        "merged": merged,
    }
    gateway.complete_json = AsyncMock(return_value=response)
    resolver = ConflictResolver(gateway=gateway)
    return resolver, gateway


# ─── Basic resolution ─────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_resolve_returns_conflict_resolution() -> None:
    resolver, _ = _make_resolver()
    result = await resolver.resolve(
        task_description="Compare hotel prices",
        agent_a_role="researcher_a",
        agent_a_result="Hotel A: €150",
        agent_b_role="researcher_b",
        agent_b_result="Hotel A: €180",
    )
    assert isinstance(result, ConflictResolution)
    assert result.winning_result == "Hotel A at €150"


@pytest.mark.asyncio
async def test_resolve_calls_llm_once() -> None:
    resolver, gateway = _make_resolver()
    await resolver.resolve(
        task_description="Compare",
        agent_a_role="a",
        agent_a_result="Result A",
        agent_b_role="b",
        agent_b_result="Result B",
    )
    gateway.complete_json.assert_called_once()


@pytest.mark.asyncio
async def test_resolve_preserves_explanation() -> None:
    resolver, _ = _make_resolver(explanation="More complete data from researcher_a")
    result = await resolver.resolve(
        task_description="Compare",
        agent_a_role="a",
        agent_a_result="Result A",
        agent_b_role="b",
        agent_b_result="Result B",
    )
    assert result.explanation == "More complete data from researcher_a"


@pytest.mark.asyncio
async def test_resolve_merged_flag() -> None:
    resolver, _ = _make_resolver(merged=True, winning_result="Combined: A+B")
    result = await resolver.resolve(
        task_description="Research",
        agent_a_role="a",
        agent_a_result="Part 1",
        agent_b_role="b",
        agent_b_result="Part 2",
    )
    assert result.merged is True
    assert result.winning_result == "Combined: A+B"


# ─── LLM failure fallback ──────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_resolve_fallback_on_llm_error() -> None:
    gateway = AsyncMock()
    gateway.complete_json = AsyncMock(side_effect=RuntimeError("LLM down"))
    resolver = ConflictResolver(gateway=gateway)

    result = await resolver.resolve(
        task_description="Compare",
        agent_a_role="researcher_a",
        agent_a_result="Fallback result",
        agent_b_role="researcher_b",
        agent_b_result="Other result",
    )
    # Fallback: prefers agent_a
    assert result.winning_result == "Fallback result"
    assert result.merged is False


@pytest.mark.asyncio
async def test_resolve_fallback_when_parsed_empty() -> None:
    gateway = AsyncMock()
    response = MagicMock()
    response.parsed = {}  # empty JSON
    gateway.complete_json = AsyncMock(return_value=response)
    resolver = ConflictResolver(gateway=gateway)

    result = await resolver.resolve(
        task_description="Compare",
        agent_a_role="a",
        agent_a_result="Default A",
        agent_b_role="b",
        agent_b_result="Default B",
    )
    # With empty parsed, winning_result falls back to agent_a_result
    assert result.winning_result == "Default A"


# ─── Multi-agent resolution ───────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_resolve_multi_single_agent_no_llm_call() -> None:
    resolver, gateway = _make_resolver()
    result = await resolver.resolve_multi(
        task_description="Single result",
        results={"agent_a": "Only result"},
    )
    # Should NOT call LLM — only one agent
    gateway.complete_json.assert_not_called()
    assert result.winning_result == "Only result"


@pytest.mark.asyncio
async def test_resolve_multi_two_agents_calls_llm_once() -> None:
    resolver, gateway = _make_resolver(winning_result="Merged AB")
    result = await resolver.resolve_multi(
        task_description="Merge two",
        results={"a": "Result A", "b": "Result B"},
    )
    gateway.complete_json.assert_called_once()
    assert result.winning_result == "Merged AB"


@pytest.mark.asyncio
async def test_resolve_multi_three_agents_calls_llm_twice() -> None:
    resolver, gateway = _make_resolver(winning_result="Merged")
    await resolver.resolve_multi(
        task_description="Three-way merge",
        results={"a": "A", "b": "B", "c": "C"},
    )
    assert gateway.complete_json.call_count == 2
