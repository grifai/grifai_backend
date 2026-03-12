"""Tests for audit/logger.py."""
import pytest
from unittest.mock import AsyncMock, MagicMock

from grif.audit.logger import AgentExplainer, AuditEvent, AuditLogger, _truncate_args


# ─── AuditLogger ──────────────────────────────────────────────────────────────

def test_audit_logger_creates_with_required_fields() -> None:
    audit = AuditLogger(user_id="u1", agent_id="agent-1", task_id="task-1")
    assert audit._base["user_id"] == "u1"
    assert audit._base["agent_id"] == "agent-1"
    assert audit._base["task_id"] == "task-1"


def test_audit_logger_defaults_optional_fields() -> None:
    audit = AuditLogger(user_id="u1")
    assert audit._base["agent_id"] == ""
    assert audit._base["task_id"] == ""


def test_audit_logger_log_does_not_raise() -> None:
    audit = AuditLogger(user_id="u1", agent_id="agent-1")
    # Should not raise
    audit.log(AuditEvent.TASK_RECEIVED, raw_input="Find hotels")


def test_audit_logger_debug_does_not_raise() -> None:
    audit = AuditLogger(user_id="u1")
    audit.debug(AuditEvent.REACT_CYCLE, cycle=1)


def test_audit_logger_warning_does_not_raise() -> None:
    audit = AuditLogger(user_id="u1")
    audit.warning(AuditEvent.TOOL_BLOCKED, tool="telegram", reason="No approval")


def test_audit_logger_error_does_not_raise() -> None:
    audit = AuditLogger(user_id="u1")
    audit.error(AuditEvent.TASK_FAILED, reason="LLM timeout")


def test_tool_called_convenience() -> None:
    audit = AuditLogger(user_id="u1", agent_id="a1")
    audit.tool_called("web_search", {"query": "hotels paris"})  # should not raise


def test_tool_blocked_convenience() -> None:
    audit = AuditLogger(user_id="u1", agent_id="a1")
    audit.tool_blocked("telegram_post", "Requires approval")


def test_react_cycle_convenience() -> None:
    audit = AuditLogger(user_id="u1", agent_id="a1")
    audit.react_cycle(cycle=3, action="web_search", decision="continue", tokens_used=500)


def test_agent_evaluated_convenience() -> None:
    audit = AuditLogger(user_id="u1", agent_id="a1")
    audit.agent_evaluated(overall_score=4.2, comment="Good result")


# ─── AuditEvent constants ─────────────────────────────────────────────────────

def test_audit_event_constants_are_strings() -> None:
    assert isinstance(AuditEvent.TASK_RECEIVED, str)
    assert isinstance(AuditEvent.TOOL_CALLED, str)
    assert isinstance(AuditEvent.AGENT_EVALUATED, str)
    assert isinstance(AuditEvent.REACT_CYCLE, str)


# ─── _truncate_args ───────────────────────────────────────────────────────────

def test_truncate_args_short_values_unchanged() -> None:
    args = {"query": "hotels", "limit": 5}
    result = _truncate_args(args)
    assert result["query"] == "hotels"
    assert result["limit"] == 5


def test_truncate_args_long_values_truncated() -> None:
    long_val = "x" * 500
    args = {"data": long_val}
    result = _truncate_args(args, max_len=200)
    assert len(result["data"]) <= 204  # 200 + "…"
    assert result["data"].endswith("…")


# ─── AgentExplainer ───────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_explainer_calls_llm() -> None:
    gateway = AsyncMock()
    response = MagicMock()
    response.content = "The agent searched for hotels in Paris."
    gateway.complete = AsyncMock(return_value=response)

    explainer = AgentExplainer(gateway)
    text = await explainer.explain(
        action="web_search",
        context="User asked for hotels in Paris",
        result="Found 5 hotels",
    )
    gateway.complete.assert_called_once()
    assert text == "The agent searched for hotels in Paris."


@pytest.mark.asyncio
async def test_explainer_fallback_on_llm_error() -> None:
    gateway = AsyncMock()
    gateway.complete = AsyncMock(side_effect=RuntimeError("LLM down"))

    explainer = AgentExplainer(gateway)
    text = await explainer.explain(
        action="web_search",
        context="User asked for hotels",
        result="Error",
    )
    assert "web_search" in text
    assert len(text) > 5


@pytest.mark.asyncio
async def test_explainer_trims_long_inputs() -> None:
    gateway = AsyncMock()
    response = MagicMock()
    response.content = "Explanation."
    gateway.complete = AsyncMock(return_value=response)

    explainer = AgentExplainer(gateway)
    long_context = "x" * 1000
    long_result = "y" * 1000
    await explainer.explain(
        action="action",
        context=long_context,
        result=long_result,
    )
    # Should not crash and LLM should be called
    gateway.complete.assert_called_once()
    call_args = gateway.complete.call_args
    user_msg = call_args[1]["messages"][1]["content"]
    # Context is trimmed to 300, result to 500
    assert len(user_msg) < 1500
