"""
LiteLLM async gateway with:
- Unified interface to Anthropic + OpenAI
- Automatic fallback (primary ↔ fallback) via LiteLLM router
- Exponential backoff retry
- Per-call token logging
- Structured logging
"""

import asyncio
from typing import Any

import litellm
import structlog
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from grif.config import get_settings
from grif.llm.fallback_map import ModelEntry, get_fallback, get_model_for_purpose

log = structlog.get_logger(__name__)

settings = get_settings()

# Suppress LiteLLM verbose output in non-debug mode
litellm.set_verbose = settings.debug


class LLMError(Exception):
    """Raised when all LLM attempts (primary + fallback) are exhausted."""


class LLMResponse:
    """Normalised response from LiteLLM."""

    def __init__(self, raw: Any, model_used: str) -> None:
        self._raw = raw
        self.model_used = model_used
        self.content: str = raw.choices[0].message.content or ""
        usage = getattr(raw, "usage", None)
        self.prompt_tokens: int = getattr(usage, "prompt_tokens", 0)
        self.completion_tokens: int = getattr(usage, "completion_tokens", 0)
        self.total_tokens: int = getattr(usage, "total_tokens", 0)

    def __repr__(self) -> str:
        return (
            f"LLMResponse(model={self.model_used}, "
            f"tokens={self.total_tokens}, content_len={len(self.content)})"
        )


class LLMGateway:
    """
    Single entry point for all LLM calls in GRIF.

    Usage:
        gateway = LLMGateway()
        response = await gateway.complete(
            messages=[{"role": "user", "content": "Hello"}],
            purpose="intent_classifier",
        )
    """

    def __init__(self) -> None:
        self._settings = get_settings()
        self._configure_litellm()

    def _configure_litellm(self) -> None:
        if self._settings.anthropic_api_key:
            litellm.anthropic_key = self._settings.anthropic_api_key
        if self._settings.openai_api_key:
            litellm.openai_key = self._settings.openai_api_key

    async def complete(
        self,
        messages: list[dict[str, str]],
        *,
        purpose: str | None = None,
        model_id: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        tools: list[dict] | None = None,
        tool_choice: str | dict | None = None,
        user_id: str | None = None,
        agent_id: str | None = None,
    ) -> LLMResponse:
        """
        Make an async LLM completion call with automatic fallback.

        Priority for model selection:
        1. Explicit `model_id` parameter
        2. `purpose` → ModelEntry lookup from fallback_map
        3. Default: SONNET
        """
        if model_id:
            primary = model_id
            fallback = get_fallback(model_id)
        elif purpose:
            entry: ModelEntry = get_model_for_purpose(purpose)
            primary = entry.primary
            fallback = entry.fallback
        else:
            entry = get_model_for_purpose("react_reasoning")
            primary = entry.primary
            fallback = entry.fallback

        call_kwargs: dict[str, Any] = {
            "messages": messages,
            "temperature": temperature if temperature is not None else 0.3,
            "max_tokens": max_tokens or 4096,
            "timeout": self._settings.llm_timeout,
        }
        if tools:
            call_kwargs["tools"] = tools
        if tool_choice:
            call_kwargs["tool_choice"] = tool_choice

        return await self._call_with_fallback(
            primary=primary,
            fallback=fallback,
            call_kwargs=call_kwargs,
            user_id=user_id,
            agent_id=agent_id,
        )

    async def _call_with_fallback(
        self,
        primary: str,
        fallback: str | None,
        call_kwargs: dict[str, Any],
        user_id: str | None,
        agent_id: str | None,
    ) -> LLMResponse:
        # Try primary model first
        try:
            return await self._call_single(
                model=primary,
                call_kwargs=call_kwargs,
                user_id=user_id,
                agent_id=agent_id,
            )
        except Exception as primary_err:
            log.warning(
                "llm_primary_failed",
                primary=primary,
                fallback=fallback,
                error=str(primary_err),
            )

        # Try fallback model if available
        if fallback:
            try:
                return await self._call_single(
                    model=fallback,
                    call_kwargs=call_kwargs,
                    user_id=user_id,
                    agent_id=agent_id,
                )
            except Exception as fallback_err:
                log.error(
                    "llm_fallback_failed",
                    primary=primary,
                    fallback=fallback,
                    error=str(fallback_err),
                )
                raise LLMError(
                    f"Both primary ({primary}) and fallback ({fallback}) models failed. "
                    f"Last error: {fallback_err}"
                ) from fallback_err

        raise LLMError(f"Primary model {primary} failed and no fallback configured.")

    async def _call_single(
        self,
        model: str,
        call_kwargs: dict[str, Any],
        user_id: str | None,
        agent_id: str | None,
    ) -> LLMResponse:
        """Single model call with exponential-backoff retry."""
        last_error: Exception | None = None

        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(self._settings.llm_max_retries),
            wait=wait_exponential(
                multiplier=self._settings.llm_retry_base_delay,
                min=1,
                max=30,
            ),
            retry=retry_if_exception_type(
                (litellm.APIConnectionError, litellm.Timeout, litellm.RateLimitError)
            ),
            reraise=True,
        ):
            with attempt:
                log.debug(
                    "llm_call_attempt",
                    model=model,
                    attempt_number=attempt.retry_state.attempt_number,
                    user_id=user_id,
                )

                raw = await litellm.acompletion(model=model, **call_kwargs)
                response = LLMResponse(raw=raw, model_used=model)

                log.info(
                    "llm_call_success",
                    model=model,
                    total_tokens=response.total_tokens,
                    user_id=user_id,
                    agent_id=agent_id,
                )
                return response

        # Should never reach here due to reraise=True
        raise LLMError(f"All retries exhausted for model {model}")

    async def complete_json(
        self,
        messages: list[dict[str, str]],
        *,
        purpose: str | None = None,
        model_id: str | None = None,
        temperature: float = 0.1,
        max_tokens: int = 4096,
        user_id: str | None = None,
        agent_id: str | None = None,
    ) -> LLMResponse:
        """
        Convenience wrapper that requests JSON output.
        Appends a JSON instruction to the last user message if not present.
        """
        msgs = list(messages)
        last = msgs[-1]
        if last.get("role") == "user" and "json" not in last.get("content", "").lower():
            msgs[-1] = {
                **last,
                "content": last["content"] + "\n\nRespond with valid JSON only.",
            }

        return await self.complete(
            messages=msgs,
            purpose=purpose,
            model_id=model_id,
            temperature=temperature,
            max_tokens=max_tokens,
            user_id=user_id,
            agent_id=agent_id,
        )

    async def complete_with_tools(
        self,
        messages: list[dict[str, str]],
        tools: list[dict],
        *,
        purpose: str | None = None,
        model_id: str | None = None,
        tool_choice: str | dict = "auto",
        temperature: float = 0.1,
        max_tokens: int = 4096,
        user_id: str | None = None,
        agent_id: str | None = None,
    ) -> LLMResponse:
        """Completion with function-calling tools schema."""
        return await self.complete(
            messages=messages,
            purpose=purpose,
            model_id=model_id,
            temperature=temperature,
            max_tokens=max_tokens,
            tools=tools,
            tool_choice=tool_choice,
            user_id=user_id,
            agent_id=agent_id,
        )


# Module-level singleton (lazy init)
_gateway: LLMGateway | None = None


def get_gateway() -> LLMGateway:
    global _gateway
    if _gateway is None:
        _gateway = LLMGateway()
    return _gateway
