from typing import Any, Callable, Dict

import aioredis

REDIS_URL = "redis://redis:6379"


class RedisEventBus:
    def __init__(self, redis_url: str = REDIS_URL):
        self.redis_url = redis_url
        self.redis = None

    async def connect(self):
        self.redis = await aioredis.from_url(self.redis_url, decode_responses=True)

    async def publish(self, stream: str, data: Dict[str, Any]):
        await self.redis.xadd(stream, data)

    async def subscribe(
        self,
        stream: str,
        group: str,
        consumer: str,
        handler: Callable[[Dict[str, Any]], None],
    ):
        try:
            await self.redis.xgroup_create(stream, group, id="$", mkstream=True)
        except aioredis.exceptions.ResponseError as e:
            if "BUSYGROUP" not in str(e):
                raise
        while True:
            resp = await self.redis.xreadgroup(group, consumer, streams={stream: ">"}, count=10, block=1000)
            for s, messages in resp:
                for msg_id, msg in messages:
                    await handler(msg)
                    await self.redis.xack(stream, group, msg_id)

    async def close(self):
        if self.redis:
            await self.redis.close()


# Пример использования:
# bus = RedisEventBus()
# await bus.connect()
# await bus.publish("events", {"type": "test", "payload": "hello"})
# await bus.subscribe("events", "group1", "consumer1", print)
