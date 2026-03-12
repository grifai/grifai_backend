"""Tests for runtime/sleep_manager.py and wake_manager.py."""
import uuid
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from grif.models.enums import AgentState, WakeTriggerType
from grif.runtime.sleep_manager import SleepError, SleepManager
from grif.runtime.wake_manager import WakeError, WakeManager


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _make_agent_db(
    agent_id: str | None = None,
    state: AgentState = AgentState.ACTIVE,
    user_id: str = "u1",
    cron: str | None = None,
) -> MagicMock:
    agent = MagicMock()
    agent.id = agent_id or str(uuid.uuid4())
    agent.state = state
    agent.user_id = user_id
    agent.schedule_cron = cron
    agent.schedule_timezone = "UTC"
    agent.checkpoint = None
    agent.context_summary = None
    return agent


def _make_session(agent: MagicMock | None = None) -> AsyncMock:
    session = AsyncMock()
    session.add = MagicMock()
    session.flush = AsyncMock()
    session.delete = AsyncMock()

    result_mock = MagicMock()
    result_mock.scalar_one_or_none.return_value = agent
    session.execute = AsyncMock(return_value=result_mock)
    return session


# ═══════════════════════════════════════════════════════════════════════════════
# SleepManager
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_sleep_transitions_to_sleeping() -> None:
    agent = _make_agent_db(state=AgentState.ACTIVE)
    session = _make_session(agent)
    sm = SleepManager(session)
    await sm.sleep(agent.id, reason="Task done")
    assert agent.state == AgentState.SLEEPING


@pytest.mark.asyncio
async def test_sleep_saves_checkpoint() -> None:
    agent = _make_agent_db(state=AgentState.ACTIVE)
    session = _make_session(agent)
    sm = SleepManager(session)
    checkpoint = {"messages": [], "cycle_count": 5}
    await sm.sleep(agent.id, reason="Done", checkpoint=checkpoint)
    assert agent.checkpoint == checkpoint


@pytest.mark.asyncio
async def test_sleep_saves_context_summary() -> None:
    agent = _make_agent_db(state=AgentState.ACTIVE)
    session = _make_session(agent)
    sm = SleepManager(session)
    await sm.sleep(agent.id, reason="Done", context_summary="Found 5 hotels")
    assert agent.context_summary == "Found 5 hotels"


@pytest.mark.asyncio
async def test_sleep_creates_wake_entries() -> None:
    from grif.models.agent_config import WakeTrigger
    agent = _make_agent_db(state=AgentState.ACTIVE)
    session = _make_session(agent)
    sm = SleepManager(session)

    triggers = [
        WakeTrigger(
            trigger_type=WakeTriggerType.SCHEDULE,
            condition="0 9 * * *",
            value=datetime.now(tz=timezone.utc) + timedelta(hours=24),
        )
    ]
    await sm.sleep(agent.id, reason="Done", wake_triggers=triggers)
    # WakeQueueDB entry added
    session.add.assert_called_once()


@pytest.mark.asyncio
async def test_sleep_raises_if_agent_not_found() -> None:
    session = _make_session(agent=None)
    sm = SleepManager(session)
    with pytest.raises(SleepError, match="not found"):
        await sm.sleep("nonexistent", reason="test")


@pytest.mark.asyncio
async def test_sleep_raises_if_already_archived() -> None:
    agent = _make_agent_db(state=AgentState.ARCHIVED)
    session = _make_session(agent)
    sm = SleepManager(session)
    with pytest.raises(SleepError, match="archived"):
        await sm.sleep(agent.id, reason="test")


@pytest.mark.asyncio
async def test_get_sleeping_agents_returns_list() -> None:
    agent = _make_agent_db(state=AgentState.SLEEPING)
    session = AsyncMock()
    result_mock = MagicMock()
    result_mock.scalars.return_value.all.return_value = [agent]
    session.execute = AsyncMock(return_value=result_mock)

    sm = SleepManager(session)
    agents = await sm.get_sleeping_agents()
    assert len(agents) == 1


@pytest.mark.asyncio
async def test_get_due_wake_entries_returns_scheduled() -> None:
    entry = MagicMock()
    entry.agent_id = uuid.uuid4()
    entry.trigger_type = WakeTriggerType.SCHEDULE
    entry.scheduled_at = datetime.utcnow() - timedelta(minutes=5)

    session = AsyncMock()
    result_mock = MagicMock()
    result_mock.scalars.return_value.all.return_value = [entry]
    session.execute = AsyncMock(return_value=result_mock)

    sm = SleepManager(session)
    entries = await sm.get_due_wake_entries()
    assert len(entries) == 1


@pytest.mark.asyncio
async def test_mark_wake_entry_processed() -> None:
    entry = MagicMock()
    entry.is_processed = False
    entry.fired_at = None
    session = AsyncMock()
    session.flush = AsyncMock()

    sm = SleepManager(session)
    await sm.mark_wake_entry_processed(entry)
    assert entry.is_processed is True
    assert entry.fired_at is not None


# ═══════════════════════════════════════════════════════════════════════════════
# WakeManager
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_wake_transitions_to_active() -> None:
    agent = _make_agent_db(state=AgentState.SLEEPING)
    session = _make_session(agent)
    wm = WakeManager(session)
    await wm.wake(agent.id, trigger_type=WakeTriggerType.MANUAL, trigger_message="Test wake")
    assert agent.state == AgentState.ACTIVE


@pytest.mark.asyncio
async def test_wake_returns_context_with_checkpoint() -> None:
    agent = _make_agent_db(state=AgentState.SLEEPING)
    agent.checkpoint = {"messages": [], "cycle_count": 3}
    agent.context_summary = "Previous: found 5 hotels"
    session = _make_session(agent)
    wm = WakeManager(session)

    ctx = await wm.wake(agent.id, trigger_message="Scheduled")
    assert ctx.checkpoint == {"messages": [], "cycle_count": 3}
    assert ctx.context_summary == "Previous: found 5 hotels"


@pytest.mark.asyncio
async def test_wake_raises_if_not_found() -> None:
    session = _make_session(agent=None)
    wm = WakeManager(session)
    with pytest.raises(WakeError, match="not found"):
        await wm.wake("bad-id")


@pytest.mark.asyncio
async def test_wake_raises_if_not_sleeping() -> None:
    agent = _make_agent_db(state=AgentState.ACTIVE)
    session = _make_session(agent)
    wm = WakeManager(session)
    with pytest.raises(WakeError, match="sleeping"):
        await wm.wake(agent.id)


@pytest.mark.asyncio
async def test_wake_marks_entry_processed() -> None:
    agent = _make_agent_db(state=AgentState.SLEEPING)
    session = _make_session(agent)
    wm = WakeManager(session)

    wake_entry = MagicMock()
    wake_entry.is_processed = False
    wake_entry.fired_at = None

    await wm.wake(agent.id, wake_entry=wake_entry)
    assert wake_entry.is_processed is True
    assert wake_entry.fired_at is not None


@pytest.mark.asyncio
async def test_wake_recurring_agent() -> None:
    """Recurring agents can also be woken."""
    agent = _make_agent_db(state=AgentState.RECURRING)
    session = _make_session(agent)
    wm = WakeManager(session)
    ctx = await wm.wake(agent.id, trigger_type=WakeTriggerType.SCHEDULE)
    assert agent.state == AgentState.ACTIVE
    assert ctx.trigger_type == WakeTriggerType.SCHEDULE


@pytest.mark.asyncio
async def test_archive_agent() -> None:
    agent = _make_agent_db(state=AgentState.SLEEPING)
    session = _make_session(agent)
    wm = WakeManager(session)
    await wm.archive_agent(agent.id, reason="User requested")
    assert agent.state == AgentState.ARCHIVED
    assert agent.archived_at is not None


@pytest.mark.asyncio
async def test_archive_raises_if_not_found() -> None:
    session = _make_session(agent=None)
    wm = WakeManager(session)
    with pytest.raises(WakeError):
        await wm.archive_agent("bad-id")


@pytest.mark.asyncio
async def test_get_pending_wake_entries() -> None:
    entry = MagicMock()
    session = AsyncMock()
    result_mock = MagicMock()
    result_mock.scalars.return_value.all.return_value = [entry]
    session.execute = AsyncMock(return_value=result_mock)

    wm = WakeManager(session)
    entries = await wm.get_pending_wake_entries()
    assert len(entries) == 1
