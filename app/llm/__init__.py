from .base import LLMProvider
from .claude_provider import ClaudeProvider
from .embeddings import EmbeddingProvider
from .openai_provider import OpenAIProvider

__all__ = [
    "LLMProvider",
    "OpenAIProvider",
    "ClaudeProvider",
    "EmbeddingProvider",
    "get_llm",
    "get_embeddings",
]

_llm_cache: dict[str, LLMProvider] = {}
_embeddings_cache: EmbeddingProvider | None = None


def get_llm(
    provider: str = "openai", api_key: str = "", model: str = ""
) -> LLMProvider:
    """Получить LLM-провайдер. Кеширует инстанс."""
    if provider in _llm_cache:
        return _llm_cache[provider]

    if provider == "claude":
        key = api_key or _get_env("ANTHROPIC_KEY")
        mdl = model or "claude-sonnet-4-20250514"
        instance = ClaudeProvider(api_key=key, model=mdl)
    else:
        key = api_key or _get_env("OPENAI_KEY")
        mdl = model or _get_env("MODEL", "gpt-4o-mini")
        instance = OpenAIProvider(api_key=key, model=mdl)

    _llm_cache[provider] = instance
    return instance


def get_embeddings(api_key: str = "", model: str = "") -> EmbeddingProvider:
    """Эмбеддинги — всегда OpenAI."""
    global _embeddings_cache
    if _embeddings_cache is not None:
        return _embeddings_cache
    key = api_key or _get_env("OPENAI_KEY")
    mdl = model or "text-embedding-3-small"
    _embeddings_cache = EmbeddingProvider(api_key=key, model=mdl)
    return _embeddings_cache


def _get_env(name: str, default: str = "") -> str:
    import os

    from dotenv import load_dotenv

    load_dotenv()
    val = os.getenv(name, default)
    if not val:
        raise ValueError(f"Переменная {name} не задана в .env")
    return val
