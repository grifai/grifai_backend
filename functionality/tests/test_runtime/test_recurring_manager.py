"""Tests for runtime/recurring_manager.py."""
import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from grif.models.enums import AgentState, WakeTriggerType
from grif.runtime.recurring_manager import RecurringError, RecurringManager


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _make_agent(
    state: AgentState = AgentState.ACTIVE,
    cron: str | None = None,
) -> MagicMock:
    agent = MagicMock()
    agent.id = str(uuid.uuid4())
    agent.user_id = "u1"
    agent.state = state
    agent.schedule_cron = cron
    agent.schedule_timezone = "UTC"
    agent.archived_at = None
    return agent


def _make_session(agent: MagicMock | None = None) -> AsyncMock:
    session = AsyncMock()
    session.add = MagicMock()
    session.flush = AsyncMock()

    result_mock = MagicMock()
    result_mock.scalar_one_or_none.return_value = agent
    result_mock.scalars.return_value.all.return_value = []
    session.execute = AsyncMock(return_value=result_mock)
    return session


# ─── register ─────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_register_sets_recurring_state() -> None:
    agent = _make_agent(state=AgentState.ACTIVE)
    session = _make_session(agent)
    rm = RecurringManager(session)
    await rm.register(agent.id, cron="0 9 * * *")
    assert agent.state == AgentState.RECURRING


@pytest.mark.asyncio
async def test_register_stores_cron() -> None:
    agent = _make_agent(state=AgentState.ACTIVE)
    session = _make_session(agent)
    rm = RecurringManager(session)
    await rm.register(agent.id, cron="0 9 * * *", tz="Europe/Moscow")
    assert agent.schedule_cron == "0 9 * * *"
    assert agent.schedule_timezone == "Europe/Moscow"


@pytest.mark.asyncio
async def test_register_creates_wake_queue_entry() -> None:
    agent = _make_agent(state=AgentState.ACTIVE)
    session = _make_session(agent)
    rm = RecurringManager(session)
    await rm.register(agent.id, cron="0 9 * * *")
    # WakeQueueDB entry was added
    session.add.assert_called_once()


@pytest.mark.asyncio
async def test_register_invalid_cron_raises() -> None:
    agent = _make_agent()
    session = _make_session(agent)
    rm = RecurringManager(session)
    with pytest.raises(RecurringError, match="Invalid cron"):
        await rm.register(agent.id, cron="not a cron")


@pytest.mark.asyncio
async def test_register_agent_not_found_raises() -> None:
    session = _make_session(agent=None)
    rm = RecurringManager(session)
    with pytest.raises(RecurringError, match="not found"):
        await rm.register("bad-id", cron="0 9 * * *")


# ─── pause ────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_pause_sets_sleeping_state() -> None:
    agent = _make_agent(state=AgentState.RECURRING, cron="0 9 * * *")
    session = _make_session(agent)
    # _cancel_pending_entries needs to return empty list
    cancel_result = MagicMock()
    cancel_result.scalars.return_value.all.return_value = []
    # First execute (select agent) → agent; second (select wake entries) → empty
    session.execute = AsyncMock(side_effect=[
        MagicMock(**{"scalar_one_or_none.return_value": agent}),
        MagicMock(**{"scalars.return_value.all.return_value": []}),
    ])
    rm = RecurringManager(session)
    await rm.pause(agent.id)
    assert agent.state == AgentState.SLEEPING


@pytest.mark.asyncio
async def test_pause_non_recurring_raises() -> None:
    agent = _make_agent(state=AgentState.ACTIVE)
    session = _make_session(agent)
    rm = RecurringManager(session)
    with pytest.raises(RecurringError, match="not recurring"):
        await rm.pause(agent.id)


# ─── resume ───────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_resume_sets_recurring_state() -> None:
    agent = _make_agent(state=AgentState.SLEEPING, cron="0 9 * * *")
    session = _make_session(agent)
    rm = RecurringManager(session)
    await rm.resume(agent.id)
    assert agent.state == AgentState.RECURRING


@pytest.mark.asyncio
async def test_resume_without_cron_raises() -> None:
    agent = _make_agent(state=AgentState.SLEEPING, cron=None)
    session = _make_session(agent)
    rm = RecurringManager(session)
    with pytest.raises(RecurringError, match="no stored cron"):
        await rm.resume(agent.id)


# ─── unregister ───────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_unregister_archives_agent() -> None:
    agent = _make_agent(state=AgentState.RECURRING, cron="0 9 * * *")
    session = _make_session(agent)
    # First execute: get agent; second: cancel entries (empty)
    session.execute = AsyncMock(side_effect=[
        MagicMock(**{"scalar_one_or_none.return_value": agent}),
        MagicMock(**{"scalars.return_value.all.return_value": []}),
    ])
    rm = RecurringManager(session)
    await rm.unregister(agent.id)
    assert agent.state == AgentState.ARCHIVED
    assert agent.schedule_cron is None
    assert agent.archived_at is not None


# ─── next_run_at ──────────────────────────────────────────────────────────────

def test_next_run_at_returns_future_datetime() -> None:
    rm = RecurringManager(AsyncMock())
    next_run = rm.next_run_at("0 9 * * *", tz="UTC")
    assert isinstance(next_run, datetime)
    # Should be in the future
    assert next_run > datetime.now(tz=timezone.utc)


def test_next_run_at_invalid_tz_defaults_to_utc() -> None:
    rm = RecurringManager(AsyncMock())
    next_run = rm.next_run_at("0 9 * * *", tz="Invalid/Timezone")
    assert isinstance(next_run, datetime)


# ─── get_recurring_agents ─────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_get_recurring_agents_returns_list() -> None:
    agent = _make_agent(state=AgentState.RECURRING)
    session = AsyncMock()
    result_mock = MagicMock()
    result_mock.scalars.return_value.all.return_value = [agent]
    session.execute = AsyncMock(return_value=result_mock)

    rm = RecurringManager(session)
    agents = await rm.get_recurring_agents()
    assert len(agents) == 1


@pytest.mark.asyncio
async def test_get_recurring_agents_filtered_by_user() -> None:
    session = AsyncMock()
    result_mock = MagicMock()
    result_mock.scalars.return_value.all.return_value = []
    session.execute = AsyncMock(return_value=result_mock)

    rm = RecurringManager(session)
    agents = await rm.get_recurring_agents(user_id="u1")
    assert agents == []
    session.execute.assert_called_once()
