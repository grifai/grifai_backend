"""
Model Router — deterministic model selection for ReAct loop.
No LLM calls here — pure logic.

Cascade logic (from architecture):
  - Simple API call / comparison / dedup → Haiku  (fast + cheap)
  - Reasoning / decision / generation      → Sonnet (agent's configured model)
  - Summarisation                          → GPT-4o-mini

The agent's AgentConfig.model_config_ can always override the default.
"""

from grif.llm.fallback_map import HAIKU, SONNET, SUMMARIZER, ModelEntry
from grif.models.agent_config import AgentConfig
from grif.models.enums import TaskType

# Keywords that suggest a heavy reasoning step
_HEAVY_KEYWORDS = frozenset(
    [
        "analyz", "reason", "plan", "strateg", "synthesiz", "evaluat",
        "generat", "writ", "creat", "compos", "summar", "draft",
        "анализ", "стратег", "создай", "напиши", "сгенерир",
    ]
)

# Keywords that suggest a simple lookup / comparison step
_LIGHT_KEYWORDS = frozenset(
    [
        "search", "find", "get", "fetch", "check", "look", "compare",
        "list", "count", "extract", "parse",
        "найди", "получи", "проверь", "извлеки",
    ]
)


def select_model(
    step_description: str,
    agent_config: AgentConfig | None = None,
    force_purpose: str | None = None,
) -> ModelEntry:
    """
    Deterministically select a model for the current ReAct step.

    Priority:
    1. force_purpose → direct lookup in PURPOSE_MAP
    2. agent_config.model_config_ override (if explicitly set by config_generator)
    3. Heuristic from step_description keywords
    4. Default: SONNET
    """
    from grif.llm.fallback_map import PURPOSE_MAP

    # 1. Explicit purpose override
    if force_purpose and force_purpose in PURPOSE_MAP:
        return PURPOSE_MAP[force_purpose]

    # 2. Agent config override (if the agent was configured with a specific model)
    if agent_config:
        model_id = agent_config.get_model_config().model_id
        # If it's a Haiku-class model, use Haiku
        if "haiku" in model_id.lower() or "mini" in model_id.lower():
            return HAIKU
        # If it's a Sonnet-class or GPT-4 model, use Sonnet
        if "sonnet" in model_id.lower() or "gpt-4o" in model_id.lower().replace("-mini", ""):
            return SONNET

    # 3. Heuristic on step text
    desc_lower = step_description.lower()

    # Summarisation is explicit
    if "summar" in desc_lower or "compress" in desc_lower or "суммаризи" in desc_lower:
        return SUMMARIZER

    # Count keyword matches
    heavy = sum(1 for kw in _HEAVY_KEYWORDS if kw in desc_lower)
    light = sum(1 for kw in _LIGHT_KEYWORDS if kw in desc_lower)

    if light > heavy:
        return HAIKU
    if heavy > 0:
        return SONNET

    # 4. Task-type heuristic
    if agent_config:
        lightweight_tasks = {TaskType.MONITOR, TaskType.REMIND, TaskType.SEARCH}
        if agent_config.task_type in lightweight_tasks:
            return HAIKU

    return SONNET


def select_model_id(
    step_description: str,
    agent_config: AgentConfig | None = None,
    force_purpose: str | None = None,
) -> str:
    """Convenience wrapper — returns primary model ID string."""
    return select_model(step_description, agent_config, force_purpose).primary
