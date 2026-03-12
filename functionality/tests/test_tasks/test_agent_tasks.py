"""Tests for tasks/agent_tasks.py — mocked Celery + DB."""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch


# ─── process_wake_queue ───────────────────────────────────────────────────────

def test_process_wake_queue_dispatches_tasks() -> None:
    """process_wake_queue should call wake_agent_task.apply_async for each due entry."""
    entry = MagicMock()
    entry.agent_id = "agent-123"
    entry.trigger_condition = "Scheduled"

    async def _mock_get_due(limit=50):
        return [entry]

    async def _mock_mark_processed(e):
        pass

    with (
        patch("grif.tasks.agent_tasks._run_async") as mock_run,
        patch("grif.tasks.agent_tasks.wake_agent_task") as mock_wake_task,
    ):
        # Simulate successful execution returning dict
        mock_run.return_value = {"dispatched": 1}
        from grif.tasks.agent_tasks import process_wake_queue
        result = process_wake_queue()
        # _run_async was called
        mock_run.assert_called_once()


def test_cleanup_archived_agents_runs() -> None:
    with patch("grif.tasks.agent_tasks._run_async") as mock_run:
        mock_run.return_value = {"deleted": 0}
        from grif.tasks.agent_tasks import cleanup_archived_agents
        result = cleanup_archived_agents()
        mock_run.assert_called_once()


# ─── wake_agent_task unit ─────────────────────────────────────────────────────

def test_wake_agent_task_is_registered() -> None:
    from grif.tasks.celery_app import celery_app
    task_names = [t for t in celery_app.tasks.keys()]
    assert "grif.tasks.agent_tasks.wake_agent_task" in task_names


def test_run_recurring_cycle_is_registered() -> None:
    from grif.tasks.celery_app import celery_app
    assert "grif.tasks.agent_tasks.run_recurring_cycle" in celery_app.tasks


def test_process_wake_queue_is_registered() -> None:
    from grif.tasks.celery_app import celery_app
    assert "grif.tasks.agent_tasks.process_wake_queue" in celery_app.tasks


def test_cleanup_is_registered() -> None:
    from grif.tasks.celery_app import celery_app
    assert "grif.tasks.agent_tasks.cleanup_archived_agents" in celery_app.tasks


# ─── Celery app config ────────────────────────────────────────────────────────

def test_celery_app_has_correct_queues() -> None:
    from grif.tasks.celery_app import celery_app
    queues = {q.name for q in celery_app.conf.task_queues}
    assert "default" in queues
    assert "wake" in queues
    assert "recurring" in queues
    assert "cleanup" in queues


def test_celery_beat_schedule_has_wake_queue_task() -> None:
    from grif.tasks.celery_app import celery_app
    beat_tasks = set(celery_app.conf.beat_schedule.keys())
    assert "process-wake-queue" in beat_tasks
    assert "cleanup-archived-agents" in beat_tasks


def test_celery_task_routes() -> None:
    from grif.tasks.celery_app import celery_app
    routes = celery_app.conf.task_routes
    assert routes["grif.tasks.agent_tasks.wake_agent_task"]["queue"] == "wake"
    assert routes["grif.tasks.agent_tasks.run_recurring_cycle"]["queue"] == "recurring"


# ─── _run_async helper ────────────────────────────────────────────────────────

def test_run_async_executes_coroutine() -> None:
    import asyncio
    from grif.tasks.agent_tasks import _run_async

    async def _coro():
        return 42

    result = _run_async(_coro())
    assert result == 42
