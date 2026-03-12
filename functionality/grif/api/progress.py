"""In-memory event bus for WebSocket progress streaming.

Architecture:
    publisher (pipeline background task) → put_nowait → asyncio.Queue
    subscriber (WebSocket handler)       ← get()       asyncio.Queue

One Queue per subscriber per task_id. Multiple WebSocket clients can
subscribe to the same task.
"""

import asyncio
from typing import Any

# Global registry: task_id → list of subscriber queues
_subscribers: dict[str, list[asyncio.Queue]] = {}


def subscribe(task_id: str) -> asyncio.Queue:
    """Register a new subscriber queue for this task_id."""
    queue: asyncio.Queue = asyncio.Queue(maxsize=200)
    _subscribers.setdefault(task_id, []).append(queue)
    return queue


def unsubscribe(task_id: str, queue: asyncio.Queue) -> None:
    """Remove subscriber queue. Cleans up empty entry."""
    subs = _subscribers.get(task_id, [])
    if queue in subs:
        subs.remove(queue)
    if not subs:
        _subscribers.pop(task_id, None)


async def publish(task_id: str, event: dict[str, Any]) -> None:
    """Publish event to all live subscribers of a task (non-blocking)."""
    for queue in list(_subscribers.get(task_id, [])):
        try:
            queue.put_nowait(event)
        except asyncio.QueueFull:
            pass  # Slow subscriber — drop rather than block


def has_subscribers(task_id: str) -> bool:
    """True if at least one WebSocket subscriber is active for task_id."""
    return bool(_subscribers.get(task_id))


def clear(task_id: str) -> None:
    """Remove all subscribers for a task (used in tests / teardown)."""
    _subscribers.pop(task_id, None)
