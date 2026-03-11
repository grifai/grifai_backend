import os
import httpx
import asyncio
import time
from .base import BaseLLM
from typing import Any, AsyncGenerator

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_API_URL = os.getenv("OPENAI_API_URL", "https://api.openai.com/v1/chat/completions")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4")

class OpenAILLM(BaseLLM):
    def __init__(self):
        self.api_key = OPENAI_API_KEY
        self.api_url = OPENAI_API_URL
        self.model = OPENAI_MODEL

    async def generate(self, prompt: str, stream: bool = False, **kwargs) -> Any:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": self.model,
            "max_tokens": kwargs.get("max_tokens", 1024),
            "messages": [{"role": "user", "content": prompt}],
            "stream": stream
        }
        start = time.monotonic()
        async with httpx.AsyncClient(timeout=60) as client:
            try:
                resp = await client.post(self.api_url, headers=headers, json=payload)
                resp.raise_for_status()
                data = resp.json()
                latency = time.monotonic() - start
                print(f"[OpenAILLM] latency={latency:.2f}s tokens={data.get('usage', {}).get('total_tokens', 0)}")
                return data["choices"][0]["message"]["content"] if not stream else data  # stream поддержка — доработать
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 429:
                    raise RuntimeError("rate_limit")
                raise

    def count_tokens(self, prompt: str) -> int:
        return len(prompt.split())
