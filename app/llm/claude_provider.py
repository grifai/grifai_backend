from .base import LLMProvider

try:
    from anthropic import Anthropic

    _HAS_ANTHROPIC = True
except ImportError:
    _HAS_ANTHROPIC = False


class ClaudeProvider(LLMProvider):
    """Заглушка. Для активации: pip install anthropic + ANTHROPIC_KEY в .env"""

    def __init__(self, api_key: str, model: str = "claude-sonnet-4-20250514"):
        if not _HAS_ANTHROPIC:
            raise ImportError("pip install anthropic")
        self.client = Anthropic(api_key=api_key)
        self.model = model

    def generate(
        self,
        system_prompt,
        user_message,
        temperature=0.7,
        max_tokens=1000,
        response_format="text",
    ):
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system_prompt,
            messages=[{"role": "user", "content": user_message}],
        )
        return resp.content[0].text

    def generate_with_history(
        self, system_prompt, messages, temperature=0.7, max_tokens=1000
    ):
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system_prompt,
            messages=messages,
        )
        return resp.content[0].text
