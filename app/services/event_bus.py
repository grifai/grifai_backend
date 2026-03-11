"""Redis Streams event bus for inter-component communication."""

import asyncio
import json
from typing import Awaitable, Callable

import redis.asyncio as aioredis

# ── Stream names ───────────────────────────────────────────────────────────────

STREAMS = {
    "INCOMING": "incoming_messages",  # new batched message from a contact
    "DRAFTS": "reply_drafts",  # generated reply draft
    "APPROVED": "approved_replies",  # user-approved reply that was sent
    "PROFILE_UPDATES": "profile_updates",  # request to rebuild a contact profile
    "NOTIFICATIONS": "notifications",  # UI/WebSocket notifications
}

# Callback signature: (event_type, data, msg_id) -> None
ConsumerCallback = Callable[[str, dict, str], Awaitable[None]]


# ── EventBus ──────────────────────────────────────────────────────────────────


class EventBus:
    """Thin async wrapper around Redis Streams."""

    def __init__(self, redis_url: str):
        self._url = redis_url
        self._redis: aioredis.Redis | None = None

    async def connect(self) -> None:
        self._redis = aioredis.from_url(self._url, decode_responses=True)

    async def close(self) -> None:
        if self._redis:
            await self._redis.aclose()
            self._redis = None

    # ── Write ──────────────────────────────────────────────────────────────────

    async def publish(self, stream: str, event_type: str, data: dict) -> str:
        """Append event to stream. Returns the Redis message ID."""
        assert self._redis, "Call connect() first"
        msg_id = await self._redis.xadd(
            stream,
            {"type": event_type, "data": json.dumps(data, ensure_ascii=False)},
        )
        return msg_id

    # ── Consumer groups ────────────────────────────────────────────────────────

    async def ensure_group(self, stream: str, group: str) -> None:
        """Create consumer group (mkstream=True so the stream is created if missing)."""
        assert self._redis, "Call connect() first"
        try:
            await self._redis.xgroup_create(stream, group, id="0", mkstream=True)
        except aioredis.ResponseError as exc:
            if "BUSYGROUP" not in str(exc):
                raise

    async def ack(self, stream: str, group: str, message_id: str) -> None:
        """Acknowledge that a message has been processed."""
        assert self._redis, "Call connect() first"
        await self._redis.xack(stream, group, message_id)

    # ── Subscribe loop ─────────────────────────────────────────────────────────

    async def subscribe(
        self,
        stream: str,
        group: str,
        consumer: str,
        callback: ConsumerCallback,
        batch_size: int = 10,
    ) -> None:
        """
        Blocking read loop using XREADGROUP.

        Calls callback(event_type, data, msg_id) for each message.
        Auto-acks on success. On callback exception the message stays
        pending (will be redelivered after PEL timeout).
        Exits cleanly on asyncio.CancelledError.
        """
        assert self._redis, "Call connect() first"
        await self.ensure_group(stream, group)

        while True:
            try:
                results = await self._redis.xreadgroup(
                    groupname=group,
                    consumername=consumer,
                    streams={stream: ">"},
                    count=batch_size,
                    block=1000,  # ms; yields to event loop on timeout
                )
                if not results:
                    await asyncio.sleep(0)
                    continue

                for _stream_name, messages in results:
                    for msg_id, fields in messages:
                        event_type = fields.get("type", "")
                        try:
                            data = json.loads(fields.get("data", "{}"))
                        except json.JSONDecodeError:
                            data = {}
                        try:
                            await callback(event_type, data, msg_id)
                            await self.ack(stream, group, msg_id)
                        except Exception as exc:
                            print(f"[EventBus] {stream}/{group} callback error: {exc}")

            except asyncio.CancelledError:
                break
            except Exception as exc:
                print(f"[EventBus] Subscribe error ({stream}/{group}): {exc}")
                await asyncio.sleep(1)
