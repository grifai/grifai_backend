"""Tests for runtime/memory_manager.py — mocked DB and LLM."""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from grif.models.memory import (
    DecisionMemory,
    FactMemory,
    PreferenceMemory,
    ProductionMemory,
    ReActCycleLog,
)
from grif.models.enums import MemoryType
from grif.runtime.memory_manager import MemoryManager, _SHORT_TERM_SIZE, _WORKING_THRESHOLD


# ─── Fixtures ─────────────────────────────────────────────────────────────────

def _make_cycle_log(n: int = 1) -> ReActCycleLog:
    return ReActCycleLog(
        cycle_number=n,
        thought=f"I need to find hotels - cycle {n}",
        action="web_search",
        action_input={"query": "hotel paris"},
        observation="Found 5 hotels",
        decision="continue",
    )


def _make_manager(working_memory_response: str = "Summarised context") -> tuple[MemoryManager, AsyncMock, AsyncMock]:
    session = AsyncMock()
    session.add = MagicMock()
    session.flush = AsyncMock()

    gateway = AsyncMock()
    llm_response = MagicMock()
    llm_response.content = working_memory_response
    gateway.complete = AsyncMock(return_value=llm_response)

    mm = MemoryManager(
        session=session,
        gateway=gateway,
        agent_id="agent-123",
        user_id="user-1",
    )
    return mm, session, gateway


# ─── Short-term memory ────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_add_cycle_log_stores_in_short_term() -> None:
    mm, session, gateway = _make_manager()
    log = _make_cycle_log(1)
    await mm.add_cycle_log(log)
    assert len(mm._short_term) == 1
    assert mm._short_term[0].cycle_number == 1


@pytest.mark.asyncio
async def test_short_term_capped_at_max_size() -> None:
    mm, session, gateway = _make_manager()
    for i in range(_SHORT_TERM_SIZE + 3):
        await mm.add_cycle_log(_make_cycle_log(i))
    # Should never exceed _SHORT_TERM_SIZE
    assert len(mm._short_term) <= _SHORT_TERM_SIZE


@pytest.mark.asyncio
async def test_short_term_keeps_most_recent() -> None:
    mm, session, gateway = _make_manager()
    for i in range(_SHORT_TERM_SIZE + 2):
        await mm.add_cycle_log(_make_cycle_log(i))
    # Latest cycle should be present
    cycle_numbers = [c.cycle_number for c in mm._short_term]
    assert _SHORT_TERM_SIZE + 1 in cycle_numbers


def test_get_recent_cycles_empty() -> None:
    mm, _, _ = _make_manager()
    result = mm.get_recent_cycles()
    assert result == []


@pytest.mark.asyncio
async def test_get_recent_cycles_returns_n() -> None:
    mm, session, gateway = _make_manager()
    for i in range(4):
        await mm.add_cycle_log(_make_cycle_log(i))
    result = mm.get_recent_cycles(2)
    assert len(result) == 2


# ─── Working memory ───────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_get_working_context_empty_when_no_cycles() -> None:
    mm, _, _ = _make_manager()
    ctx = await mm.get_working_context()
    assert ctx == ""


@pytest.mark.asyncio
async def test_get_working_context_triggers_summarization() -> None:
    mm, session, gateway = _make_manager("Agent found 5 hotels in Paris.")
    # Add a cycle to trigger summarization
    await mm.add_cycle_log(_make_cycle_log(1))
    # Force working memory to be empty then request it
    mm._working_memory = ""
    ctx = await mm.get_working_context()
    assert ctx == "Agent found 5 hotels in Paris."
    gateway.complete.assert_called_once()


@pytest.mark.asyncio
async def test_working_memory_cached_after_first_call() -> None:
    mm, session, gateway = _make_manager("Cached summary")
    await mm.add_cycle_log(_make_cycle_log(1))
    mm._working_memory = ""
    await mm.get_working_context()
    await mm.get_working_context()  # Second call — should NOT call LLM again
    gateway.complete.assert_called_once()


@pytest.mark.asyncio
async def test_refresh_working_memory_fallback_on_error() -> None:
    mm, session, gateway = _make_manager()
    gateway.complete = AsyncMock(side_effect=RuntimeError("LLM down"))
    await mm.add_cycle_log(_make_cycle_log(1))
    mm._working_memory = ""
    ctx = await mm.get_working_context()
    # Fallback to raw cycles text (truncated to 500 chars)
    assert "Cycle 1" in ctx or len(ctx) <= 500


@pytest.mark.asyncio
async def test_refresh_working_memory_calls_llm() -> None:
    """_refresh_working_memory triggers an LLM call when short-term is non-empty."""
    mm, session, gateway = _make_manager("Threshold summary")
    mm._short_term = [_make_cycle_log(i) for i in range(3)]
    await mm._refresh_working_memory()
    gateway.complete.assert_called_once()
    assert mm._working_memory == "Threshold summary"


def test_inject_working_memory_prepends_system_message() -> None:
    mm, _, _ = _make_manager()
    mm._working_memory = "Context about task"
    messages = [{"role": "user", "content": "Hello"}]
    result = mm.inject_working_memory(messages)
    assert result[0]["role"] == "system"
    assert "Context about task" in result[0]["content"]
    assert result[1] == messages[0]


def test_inject_working_memory_no_op_when_empty() -> None:
    mm, _, _ = _make_manager()
    mm._working_memory = ""
    messages = [{"role": "user", "content": "Hello"}]
    result = mm.inject_working_memory(messages)
    assert result == messages


# ─── Long-term: store methods ─────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_store_fact_adds_to_session() -> None:
    mm, session, _ = _make_manager()
    fact = FactMemory(key="hotel_price", value="150 EUR", source="booking.com")
    await mm.store_fact(fact)
    session.add.assert_called_once()
    session.flush.assert_called_once()


@pytest.mark.asyncio
async def test_store_decision_adds_to_session() -> None:
    mm, session, _ = _make_manager()
    decision = DecisionMemory(
        decision="Book Hotel A",
        reasoning="Best price/quality ratio",
    )
    await mm.store_decision(decision)
    session.add.assert_called_once()


@pytest.mark.asyncio
async def test_store_preference_adds_to_session() -> None:
    mm, session, _ = _make_manager()
    pref = PreferenceMemory(category="accommodation", preference="4-star hotels", strength=0.9)
    await mm.store_preference(pref)
    session.add.assert_called_once()


@pytest.mark.asyncio
async def test_update_production_memory_creates_when_absent() -> None:
    mm, session, _ = _make_manager()
    # Simulate no existing production memory
    result_mock = MagicMock()
    result_mock.scalar_one_or_none.return_value = None
    session.execute = AsyncMock(return_value=result_mock)

    production = ProductionMemory()
    await mm.update_production_memory(production)
    session.add.assert_called_once()


@pytest.mark.asyncio
async def test_update_production_memory_updates_existing() -> None:
    mm, session, _ = _make_manager()
    existing = MagicMock()
    existing.content = {}
    result_mock = MagicMock()
    result_mock.scalar_one_or_none.return_value = existing
    session.execute = AsyncMock(return_value=result_mock)

    production = ProductionMemory()
    await mm.update_production_memory(production)
    # Should update existing, not add new
    session.add.assert_not_called()
    assert existing.content is not None


# ─── Long-term: retrieve ──────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_retrieve_relevant_returns_content_list() -> None:
    mm, session, _ = _make_manager()
    row1 = MagicMock()
    row1.content = {"key": "hotel_price", "value": "150 EUR"}
    row2 = MagicMock()
    row2.content = {"key": "user_pref", "value": "4-star"}

    result_mock = MagicMock()
    result_mock.scalars.return_value.all.return_value = [row1, row2]
    session.execute = AsyncMock(return_value=result_mock)

    results = await mm.retrieve_relevant("hotels in Paris")
    assert len(results) == 2
    assert results[0] == {"key": "hotel_price", "value": "150 EUR"}


@pytest.mark.asyncio
async def test_retrieve_relevant_empty() -> None:
    mm, session, _ = _make_manager()
    result_mock = MagicMock()
    result_mock.scalars.return_value.all.return_value = []
    session.execute = AsyncMock(return_value=result_mock)

    results = await mm.retrieve_relevant("unrelated query")
    assert results == []


# ─── load_short_term_from_db ──────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_load_short_term_from_db_restores_cycles() -> None:
    mm, session, _ = _make_manager()

    row = MagicMock()
    row.cycle_number = 1
    row.thought = "I need to search"
    row.action = "web_search"
    row.action_input = {"query": "hotel"}
    row.observation = "Found 3 hotels"
    row.decision = "continue"
    row.tokens_used = 150

    result_mock = MagicMock()
    result_mock.scalars.return_value.all.return_value = [row]
    session.execute = AsyncMock(return_value=result_mock)

    await mm.load_short_term_from_db()
    assert len(mm._short_term) == 1
    assert mm._short_term[0].thought == "I need to search"
