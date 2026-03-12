"""Tests for runtime/model_router.py — fully deterministic, no LLM calls."""
import pytest

from grif.llm.fallback_map import HAIKU, SONNET, SUMMARIZER
from grif.models.agent_config import AgentConfig, ModelConfig, PromptLayers
from grif.models.enums import TaskType
from grif.runtime.model_router import select_model, select_model_id


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _make_config(task_type: TaskType = TaskType.EXECUTE, model_id: str = "claude-sonnet-4-6") -> AgentConfig:
    return AgentConfig(
        user_id="u1",
        task_type=task_type,
        blueprint_id="generic_worker",
        prompt_layers=PromptLayers(
            layer_1_core_identity="Core",
            layer_2_role_template="Role",
            layer_3_task_context="Task",
        ),
        tools=[],
        model_config=ModelConfig(model_id=model_id),
        tool_permissions=[],
    )


# ─── force_purpose ────────────────────────────────────────────────────────────

def test_force_purpose_intent_classifier_returns_haiku() -> None:
    result = select_model("anything", force_purpose="intent_classifier")
    assert result == HAIKU


def test_force_purpose_config_generator_returns_sonnet() -> None:
    result = select_model("anything", force_purpose="config_generator")
    assert result == SONNET


def test_force_purpose_summarizer() -> None:
    result = select_model("anything", force_purpose="summarizer")
    assert result == SUMMARIZER


def test_force_purpose_self_evaluation_returns_haiku() -> None:
    result = select_model("anything", force_purpose="self_evaluation")
    assert result == HAIKU


def test_force_purpose_unknown_falls_through_to_heuristic() -> None:
    # Unknown purpose → falls through to heuristic
    result = select_model("search find get", force_purpose="nonexistent_purpose")
    assert result == HAIKU  # keyword heuristic: light > heavy


# ─── agent_config model override ──────────────────────────────────────────────

def test_agent_config_haiku_model_returns_haiku() -> None:
    config = _make_config(model_id="claude-haiku-4-5-20251001")
    result = select_model("reason and plan", agent_config=config)
    assert result == HAIKU


def test_agent_config_mini_model_returns_haiku() -> None:
    config = _make_config(model_id="gpt-4o-mini")
    result = select_model("analyze and synthesize", agent_config=config)
    assert result == HAIKU


def test_agent_config_sonnet_model_returns_sonnet() -> None:
    config = _make_config(model_id="claude-sonnet-4-6")
    result = select_model("search find", agent_config=config)
    assert result == SONNET  # config override wins over keyword heuristic


# ─── Keyword heuristics ───────────────────────────────────────────────────────

def test_summarize_keyword_returns_summarizer() -> None:
    result = select_model("summarize the results")
    assert result == SUMMARIZER


def test_compress_keyword_returns_summarizer() -> None:
    result = select_model("compress and distill content")
    assert result == SUMMARIZER


def test_light_keywords_return_haiku() -> None:
    result = select_model("search find get fetch check")
    assert result == HAIKU


def test_heavy_keywords_return_sonnet() -> None:
    result = select_model("analyze reason plan strategy")
    assert result == SONNET


def test_more_light_than_heavy_returns_haiku() -> None:
    # 3 light, 1 heavy → haiku
    result = select_model("search find get analyze")
    assert result == HAIKU


def test_equal_light_and_heavy_returns_sonnet_or_haiku() -> None:
    # 1 light, 1 heavy → not light > heavy, so check heavy > 0 → SONNET
    result = select_model("search analyze")
    # heavy=1, light=1 → light not > heavy, heavy > 0 → SONNET
    assert result == SONNET


def test_no_keywords_empty_string() -> None:
    result = select_model("")
    assert result == SONNET  # default


def test_russian_light_keywords() -> None:
    result = select_model("найди отель и получи список")
    assert result == HAIKU


def test_russian_heavy_keywords() -> None:
    result = select_model("анализ данных и стратегия")
    assert result == SONNET


# ─── Task-type heuristic ──────────────────────────────────────────────────────

def test_monitor_task_type_returns_haiku() -> None:
    config = _make_config(task_type=TaskType.MONITOR, model_id="unknown-model-xyz")
    result = select_model("check status", agent_config=config)
    # No heavy/light keywords → falls to task-type check
    assert result == HAIKU


def test_remind_task_type_returns_haiku() -> None:
    config = _make_config(task_type=TaskType.REMIND, model_id="unknown-model-xyz")
    result = select_model("notify user", agent_config=config)
    assert result == HAIKU


def test_search_task_type_returns_haiku() -> None:
    config = _make_config(task_type=TaskType.SEARCH, model_id="unknown-model-xyz")
    result = select_model("process results", agent_config=config)
    assert result == HAIKU


def test_execute_task_type_default_sonnet() -> None:
    config = _make_config(task_type=TaskType.EXECUTE, model_id="unknown-model-xyz")
    result = select_model("process data", agent_config=config)
    assert result == SONNET


# ─── select_model_id convenience wrapper ──────────────────────────────────────

def test_select_model_id_returns_string() -> None:
    model_id = select_model_id("search find get")
    assert isinstance(model_id, str)
    assert len(model_id) > 0


def test_select_model_id_haiku_path() -> None:
    model_id = select_model_id("search find get fetch")
    assert "haiku" in model_id.lower() or "mini" in model_id.lower()


def test_select_model_id_force_purpose() -> None:
    model_id = select_model_id("anything", force_purpose="config_generator")
    assert "sonnet" in model_id.lower() or "gpt-4" in model_id.lower()
