import os
import time
from typing import Any

import httpx

from .base import BaseLLM

CLAUDE_API_KEY = os.getenv("CLAUDE_API_KEY")
CLAUDE_API_URL = os.getenv("CLAUDE_API_URL", "https://api.anthropic.com/v1/messages")
CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-3-opus-20240229")


class ClaudeLLM(BaseLLM):
    def __init__(self):
        self.api_key = CLAUDE_API_KEY
        self.api_url = CLAUDE_API_URL
        self.model = CLAUDE_MODEL

    async def generate(self, prompt: str, stream: bool = False, **kwargs) -> Any:
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        payload = {
            "model": self.model,
            "max_tokens": kwargs.get("max_tokens", 1024),
            "messages": [{"role": "user", "content": prompt}],
            "stream": stream,
        }
        start = time.monotonic()
        async with httpx.AsyncClient(timeout=60) as client:
            try:
                resp = await client.post(self.api_url, headers=headers, json=payload)
                resp.raise_for_status()
                data = resp.json()
                latency = time.monotonic() - start
                # Логирование latency и token count
                print(f"[ClaudeLLM] latency={latency:.2f}s tokens={data.get('usage', {}).get('output_tokens', 0)}")
                return data["content"] if not stream else data  # stream поддержка — доработать под SSE
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 429:
                    raise RuntimeError("rate_limit")
                raise

    def count_tokens(self, prompt: str) -> int:
        # Примерная оценка (реализация зависит от модели)
        return len(prompt.split())
