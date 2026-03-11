import asyncio
import random
import time
from .claude import ClaudeLLM
from .openai import OpenAILLM
from typing import Any

class LLMService:
    def __init__(self):
        self.claude = ClaudeLLM()
        self.openai = OpenAILLM()

    async def generate(self, prompt: str, stream: bool = False, max_retries: int = 3, **kwargs) -> Any:
        backoff = 1
        for attempt in range(max_retries):
            try:
                return await self.claude.generate(prompt, stream=stream, **kwargs)
            except RuntimeError as e:
                if str(e) == "rate_limit":
                    print(f"[LLMService] Claude rate limit, retrying in {backoff}s...")
                    await asyncio.sleep(backoff + random.uniform(0, 0.5))
                    backoff *= 2
                    continue
                print(f"[LLMService] Claude error: {e}, fallback to OpenAI")
            except Exception as e:
                print(f"[LLMService] Claude error: {e}, fallback to OpenAI")
            # fallback на OpenAI
            try:
                return await self.openai.generate(prompt, stream=stream, **kwargs)
            except RuntimeError as e:
                if str(e) == "rate_limit" and attempt < max_retries - 1:
                    print(f"[LLMService] OpenAI rate limit, retrying in {backoff}s...")
                    await asyncio.sleep(backoff + random.uniform(0, 0.5))
                    backoff *= 2
                    continue
                raise
        raise RuntimeError("All LLM providers failed after retries")
