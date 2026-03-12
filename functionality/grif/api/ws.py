"""WebSocket /ws/progress/{task_id} — real-time task progress streaming.

Clients connect before or after POST /tasks. Events are buffered in an
asyncio.Queue so nothing is lost if the client connects a moment late.

Event types emitted by the pipeline:
    task_received      — task persisted, signal parsed
    intent_classified  — Haiku classified the intent
    agent_spawned      — agent created in DB
    react_started      — ReAct loop began
    react_cycle        — one Thought→Action→Observation cycle completed
    tool_called        — tool executed inside the loop
    task_complete      — final result available
    error              — unrecoverable error
    heartbeat          — keep-alive (every 30 s of silence)
"""

import asyncio
import json

import structlog
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from grif.api import progress

log = structlog.get_logger(__name__)
router = APIRouter()

_HEARTBEAT_INTERVAL = 30  # seconds
_TERMINAL_EVENTS = {"task_complete", "error"}


@router.websocket("/progress/{task_id}")
async def ws_progress(websocket: WebSocket, task_id: str) -> None:
    """
    Stream progress events for *task_id*.

    - Sends a JSON object for every event published via progress.publish().
    - Sends a heartbeat every 30 s if no events arrive.
    - Closes the connection automatically on task_complete or error.
    """
    await websocket.accept()
    queue = progress.subscribe(task_id)
    log.info("ws_progress_connected", task_id=task_id)

    try:
        while True:
            try:
                event = await asyncio.wait_for(
                    queue.get(), timeout=_HEARTBEAT_INTERVAL
                )
                await websocket.send_text(json.dumps(event, default=str))

                if event.get("event_type") in _TERMINAL_EVENTS:
                    break  # Pipeline done — close cleanly

            except asyncio.TimeoutError:
                # Keep-alive heartbeat
                heartbeat = {"event_type": "heartbeat", "task_id": task_id}
                await websocket.send_text(json.dumps(heartbeat))

    except WebSocketDisconnect:
        log.info("ws_progress_client_disconnected", task_id=task_id)
    except Exception as exc:
        log.error("ws_progress_error", task_id=task_id, error=str(exc))
        try:
            await websocket.close(code=1011)
        except Exception:
            pass
    finally:
        progress.unsubscribe(task_id, queue)
        log.info("ws_progress_closed", task_id=task_id)
