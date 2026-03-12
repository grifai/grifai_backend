"""
Primary → fallback model mapping.
When the primary provider (Anthropic) is unavailable, LiteLLM
automatically retries with the fallback (OpenAI) and vice-versa.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelEntry:
    primary: str
    fallback: str
    provider: str  # primary provider name


# Canonical model entries
SONNET = ModelEntry(
    primary="claude-sonnet-4-6",
    fallback="gpt-4o",
    provider="anthropic",
)

HAIKU = ModelEntry(
    primary="claude-haiku-4-5-20251001",
    fallback="gpt-4o-mini",
    provider="anthropic",
)

SUMMARIZER = ModelEntry(
    primary="gpt-4o-mini",
    fallback="claude-haiku-4-5-20251001",
    provider="openai",
)

# Purpose → ModelEntry mapping (used by model_router.py)
PURPOSE_MAP: dict[str, ModelEntry] = {
    # LLM calls per architecture:
    "intent_classifier": HAIKU,      # Step 2: classify intent
    "config_generator": SONNET,      # Step 5: generate AgentConfig
    "conflict_resolver": SONNET,     # Orchestrator: resolve conflicts
    "self_evaluation": HAIKU,        # Evaluation: score after completion
    "clarification": HAIKU,          # Step 2.5: generate clarifying questions
    "style_cloner": SONNET,          # User: generate Style Guide
    "explainer": HAIKU,              # Audit: human-readable explanation
    "summarizer": SUMMARIZER,        # Memory: working memory summarisation
    # ReAct loop defaults (overridden by AgentConfig.model_config_)
    "react_reasoning": SONNET,       # Heavy reasoning / decision steps
    "react_api_call": HAIKU,         # Simple API call steps
    "react_comparison": HAIKU,       # Comparison / dedup
}

# Full model id → ModelEntry (for reverse lookup)
ALL_MODELS: dict[str, ModelEntry] = {
    SONNET.primary: SONNET,
    HAIKU.primary: HAIKU,
    SUMMARIZER.primary: SUMMARIZER,
    "gpt-4o": ModelEntry("gpt-4o", "claude-sonnet-4-6", "openai"),
}


def get_fallback(model_id: str) -> str | None:
    """Return fallback model id for a given primary model id."""
    entry = ALL_MODELS.get(model_id)
    return entry.fallback if entry else None


def get_model_for_purpose(purpose: str) -> ModelEntry:
    """Return ModelEntry for a given purpose key. Defaults to SONNET."""
    return PURPOSE_MAP.get(purpose, SONNET)
