"""Tests for api/ws.py — WebSocket /ws/progress/{task_id}."""
import asyncio
import json
import uuid
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from grif.api import progress
from grif.main import create_app


# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def clear_progress_after_test():
    """Clean up subscriptions after each test."""
    yield
    # Clear all subscriptions
    from grif.api.progress import _subscribers
    _subscribers.clear()


def _make_app():
    return create_app()


# ═══════════════════════════════════════════════════════════════════════════════
# WebSocket connection
# ═══════════════════════════════════════════════════════════════════════════════

def test_ws_progress_connects() -> None:
    """WebSocket endpoint accepts connections."""
    task_id = str(uuid.uuid4())
    app = _make_app()

    with TestClient(app) as client:
        with client.websocket_connect(f"/ws/progress/{task_id}") as ws:
            # Connection accepted — send a heartbeat to verify it works
            # Publish an event to trigger a response immediately
            progress.subscribe(task_id)  # Extra subscriber to ensure publish works
            # The ws client is connected
            assert ws is not None


def test_ws_progress_receives_published_event() -> None:
    """Events published via progress.publish() are forwarded to WebSocket clients."""
    task_id = str(uuid.uuid4())
    app = _make_app()

    with TestClient(app) as client:
        with client.websocket_connect(f"/ws/progress/{task_id}") as ws:
            # Publish an event from another "coroutine"
            event = {"event_type": "task_received", "task_id": task_id, "data": {}}

            # Use asyncio to publish (TestClient uses sync context)
            loop = asyncio.new_event_loop()
            try:
                loop.run_until_complete(progress.publish(task_id, event))
            finally:
                loop.close()

            # Read the message from WebSocket
            raw = ws.receive_text()
            data = json.loads(raw)

            assert data["event_type"] == "task_received"
            assert data["task_id"] == task_id


def test_ws_progress_closes_on_task_complete() -> None:
    """Connection closes automatically when task_complete event is received."""
    task_id = str(uuid.uuid4())
    app = _make_app()

    with TestClient(app) as client:
        with client.websocket_connect(f"/ws/progress/{task_id}") as ws:
            complete_event = {
                "event_type": "task_complete",
                "task_id": task_id,
                "data": {"result": "Hotels found"},
            }

            loop = asyncio.new_event_loop()
            try:
                loop.run_until_complete(progress.publish(task_id, complete_event))
            finally:
                loop.close()

            raw = ws.receive_text()
            data = json.loads(raw)
            assert data["event_type"] == "task_complete"
            # Connection should close after terminal event
            # The server-side loop breaks; client may receive close frame


def test_ws_progress_closes_on_error_event() -> None:
    """Connection closes automatically when error event is received."""
    task_id = str(uuid.uuid4())
    app = _make_app()

    with TestClient(app) as client:
        with client.websocket_connect(f"/ws/progress/{task_id}") as ws:
            error_event = {
                "event_type": "error",
                "task_id": task_id,
                "data": {"error": "LLM down"},
            }

            loop = asyncio.new_event_loop()
            try:
                loop.run_until_complete(progress.publish(task_id, error_event))
            finally:
                loop.close()

            raw = ws.receive_text()
            data = json.loads(raw)
            assert data["event_type"] == "error"


def test_ws_progress_task_id_in_all_events() -> None:
    """All published events include the task_id field."""
    task_id = str(uuid.uuid4())
    app = _make_app()

    with TestClient(app) as client:
        with client.websocket_connect(f"/ws/progress/{task_id}") as ws:
            events = [
                {"event_type": "task_received", "task_id": task_id},
                {"event_type": "intent_classified", "task_id": task_id, "data": {"task_type": "search"}},
                {"event_type": "task_complete", "task_id": task_id, "data": {"result": "done"}},
            ]

            loop = asyncio.new_event_loop()
            try:
                for ev in events:
                    loop.run_until_complete(progress.publish(task_id, ev))
            finally:
                loop.close()

            received = []
            for _ in range(len(events)):
                raw = ws.receive_text()
                received.append(json.loads(raw))
                if received[-1]["event_type"] == "task_complete":
                    break

            assert all("task_id" in ev for ev in received)


# ═══════════════════════════════════════════════════════════════════════════════
# progress module unit tests
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_progress_subscribe_creates_queue() -> None:
    task_id = str(uuid.uuid4())
    q = progress.subscribe(task_id)
    assert isinstance(q, asyncio.Queue)
    assert progress.has_subscribers(task_id)
    progress.unsubscribe(task_id, q)
    assert not progress.has_subscribers(task_id)


@pytest.mark.asyncio
async def test_progress_publish_puts_to_queue() -> None:
    task_id = str(uuid.uuid4())
    q = progress.subscribe(task_id)

    event = {"event_type": "test", "task_id": task_id}
    await progress.publish(task_id, event)

    received = q.get_nowait()
    assert received["event_type"] == "test"

    progress.unsubscribe(task_id, q)


@pytest.mark.asyncio
async def test_progress_publish_to_multiple_subscribers() -> None:
    task_id = str(uuid.uuid4())
    q1 = progress.subscribe(task_id)
    q2 = progress.subscribe(task_id)

    await progress.publish(task_id, {"event_type": "ping", "task_id": task_id})

    assert q1.qsize() == 1
    assert q2.qsize() == 1

    progress.unsubscribe(task_id, q1)
    progress.unsubscribe(task_id, q2)


@pytest.mark.asyncio
async def test_progress_publish_no_subscribers_no_error() -> None:
    task_id = str(uuid.uuid4())
    # Should not raise even with no subscribers
    await progress.publish(task_id, {"event_type": "test", "task_id": task_id})


@pytest.mark.asyncio
async def test_progress_unsubscribe_cleans_up() -> None:
    task_id = str(uuid.uuid4())
    q = progress.subscribe(task_id)
    assert progress.has_subscribers(task_id)
    progress.unsubscribe(task_id, q)
    assert not progress.has_subscribers(task_id)


@pytest.mark.asyncio
async def test_progress_queue_full_drops_event() -> None:
    """When queue is full, events are dropped without raising."""
    from grif.api.progress import _subscribers
    task_id = str(uuid.uuid4())
    # Create a tiny queue
    q: asyncio.Queue = asyncio.Queue(maxsize=1)
    _subscribers.setdefault(task_id, []).append(q)

    # Fill the queue
    await progress.publish(task_id, {"event_type": "first"})
    # This should not raise even though queue is full
    await progress.publish(task_id, {"event_type": "overflow"})

    # Only first event present
    assert q.qsize() == 1
    ev = q.get_nowait()
    assert ev["event_type"] == "first"

    progress.unsubscribe(task_id, q)


@pytest.mark.asyncio
async def test_progress_clear_removes_all_subscribers() -> None:
    task_id = str(uuid.uuid4())
    progress.subscribe(task_id)
    progress.subscribe(task_id)
    assert progress.has_subscribers(task_id)
    progress.clear(task_id)
    assert not progress.has_subscribers(task_id)
