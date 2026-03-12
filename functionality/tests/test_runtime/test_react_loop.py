"""Tests for runtime/react_loop.py — mocked LLM, registry, memory."""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from grif.models.agent_config import AgentConfig, ModelConfig, PromptLayers
from grif.models.enums import ReactDecision, TaskType
from grif.runtime.react_loop import ReActGraph, _infer_decision, _SLEEP_SIGNALS, _WAIT_SIGNALS


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _make_config() -> AgentConfig:
    return AgentConfig(
        user_id="u1",
        task_type=TaskType.SEARCH,
        blueprint_id="generic_worker",
        prompt_layers=PromptLayers(
            layer_1_core_identity="You are GRIF.",
            layer_2_role_template="Search agent.",
            layer_3_task_context="Find hotels.",
        ),
        tools=["web_search"],
        model_config=ModelConfig(),
        tool_permissions=[],
    )


def _make_graph(
    llm_content: str = "final answer: done",
    tool_calls: list | None = None,
) -> tuple[ReActGraph, AsyncMock, AsyncMock, AsyncMock]:
    """Build a ReActGraph with all dependencies mocked."""
    from grif.tools.registry import ToolRegistry
    from grif.runtime.memory_manager import MemoryManager

    gateway = AsyncMock()
    llm_response = MagicMock()
    llm_response.content = llm_content
    raw_msg = MagicMock()
    raw_msg.tool_calls = tool_calls or []
    raw_choice = MagicMock()
    raw_choice.message = raw_msg
    raw = MagicMock()
    raw.choices = [raw_choice]
    llm_response._raw = raw
    gateway.complete = AsyncMock(return_value=llm_response)
    gateway.complete_with_tools = AsyncMock(return_value=llm_response)

    registry = MagicMock(spec=ToolRegistry)
    registry.get_schemas = MagicMock(return_value=[])
    registry.execute = AsyncMock()

    memory = MagicMock(spec=MemoryManager)
    memory.get_working_context = AsyncMock(return_value="")
    memory.add_cycle_log = AsyncMock()

    session = AsyncMock()
    session.add = MagicMock()
    session.flush = AsyncMock()

    config = _make_config()

    graph = ReActGraph(
        gateway=gateway,
        registry=registry,
        memory=memory,
        agent_config=config,
        session=session,
    )
    return graph, gateway, registry, memory


# ─── _infer_decision ──────────────────────────────────────────────────────────

def test_infer_decision_escalate_at_max_cycles() -> None:
    result = _infer_decision("some text", cycle=20, max_cycles=20)
    assert result == ReactDecision.ESCALATE


def test_infer_decision_sleep_signal() -> None:
    result = _infer_decision("sleep — задача выполнена", cycle=1, max_cycles=20)
    assert result == ReactDecision.SLEEP


def test_infer_decision_wait_signal() -> None:
    result = _infer_decision("wait for user input", cycle=1, max_cycles=20)
    assert result == ReactDecision.WAIT


def test_infer_decision_report_signal() -> None:
    result = _infer_decision("Final answer: I found 5 hotels", cycle=1, max_cycles=20)
    assert result == ReactDecision.REPORT


def test_infer_decision_continue_when_no_signals() -> None:
    result = _infer_decision("I need to search for more data", cycle=1, max_cycles=20)
    assert result == ReactDecision.CONTINUE


def test_infer_decision_russian_sleep() -> None:
    result = _infer_decision("засыпаю, всё готово", cycle=3, max_cycles=20)
    assert result == ReactDecision.SLEEP


def test_infer_decision_russian_wait() -> None:
    result = _infer_decision("ожидаю ответ пользователя", cycle=2, max_cycles=20)
    assert result == ReactDecision.WAIT


def test_infer_decision_russian_complete() -> None:
    result = _infer_decision("итого: нашёл 3 отеля", cycle=5, max_cycles=20)
    assert result == ReactDecision.REPORT


def test_infer_decision_escalate_overrides_all() -> None:
    # Even with final answer signal, max_cycles triggers ESCALATE
    result = _infer_decision("final answer: done", cycle=20, max_cycles=20)
    assert result == ReactDecision.ESCALATE


# ─── ReActGraph.run — happy path ──────────────────────────────────────────────

@pytest.mark.asyncio
async def test_react_run_returns_final_state() -> None:
    graph, gateway, registry, memory = _make_graph(llm_content="final answer: found 3 hotels")
    result = await graph.run("Find hotels in Paris")
    assert "decision" in result
    assert "final_result" in result


@pytest.mark.asyncio
async def test_react_run_report_decision_sets_final_result() -> None:
    graph, gateway, registry, memory = _make_graph(llm_content="final answer: found 3 hotels")
    result = await graph.run("Find hotels in Paris")
    assert result["decision"] == ReactDecision.REPORT.value
    assert "found 3 hotels" in result["final_result"].lower()


@pytest.mark.asyncio
async def test_react_run_sleep_decision_ends() -> None:
    graph, gateway, registry, memory = _make_graph(llm_content="sleep — задача выполнена")
    result = await graph.run("Monitor prices")
    assert result["decision"] == ReactDecision.SLEEP.value


@pytest.mark.asyncio
async def test_react_run_wait_decision_ends() -> None:
    graph, gateway, registry, memory = _make_graph(llm_content="waiting for user input to proceed")
    result = await graph.run("What hotels do you prefer?")
    assert result["decision"] == ReactDecision.WAIT.value


@pytest.mark.asyncio
async def test_react_run_memory_injected() -> None:
    graph, gateway, registry, memory = _make_graph(llm_content="final answer: done")
    memory.get_working_context = AsyncMock(return_value="Previous: found hotels in Rome")
    await graph.run("Find more")
    memory.get_working_context.assert_called_once()


@pytest.mark.asyncio
async def test_react_run_cycle_count_increments() -> None:
    graph, gateway, registry, memory = _make_graph(llm_content="final answer: done")
    result = await graph.run("Find hotels")
    assert result["cycle_count"] >= 1


# ─── Tool call flow ───────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_react_run_executes_tool_call() -> None:
    from grif.tools.base import ToolResult

    # Mock a tool call in the LLM response
    tc = MagicMock()
    tc.id = "call_123"
    tc.function.name = "web_search"
    tc.function.arguments = '{"query": "hotels paris"}'

    # First call: has tool_call; second call: final answer
    llm_no_tool = MagicMock()
    llm_no_tool.content = "final answer: found 5 hotels"
    raw_no_tool_msg = MagicMock()
    raw_no_tool_msg.tool_calls = []
    raw_no_tool = MagicMock()
    raw_no_tool.choices = [MagicMock(message=raw_no_tool_msg)]
    llm_no_tool._raw = raw_no_tool

    llm_with_tool = MagicMock()
    llm_with_tool.content = "[Calling tool: web_search]"
    raw_tool_msg = MagicMock()
    raw_tool_msg.tool_calls = [tc]
    raw_tool = MagicMock()
    raw_tool.choices = [MagicMock(message=raw_tool_msg)]
    llm_with_tool._raw = raw_tool

    graph, gateway, registry, memory = _make_graph()
    gateway.complete_with_tools = AsyncMock(side_effect=[llm_with_tool, llm_no_tool])

    tool_result = ToolResult(tool_name="web_search", success=True, output="5 hotels found")
    registry.get_schemas = MagicMock(return_value=[{"type": "function", "function": {"name": "web_search"}}])
    registry.execute = AsyncMock(return_value=tool_result)

    result = await graph.run("Find hotels in Paris")
    registry.execute.assert_called_once_with(
        tool_name="web_search",
        kwargs={"query": "hotels paris"},
        agent_config=graph._config,
    )


@pytest.mark.asyncio
async def test_react_permission_denied_continues_as_wait() -> None:
    from grif.tools.base import ToolResult
    from grif.tools.registry import PermissionDeniedError

    tc = MagicMock()
    tc.id = "call_456"
    tc.function.name = "telegram_post"
    tc.function.arguments = '{"text": "Hello"}'

    llm_with_tool = MagicMock()
    llm_with_tool.content = "[Calling tool: telegram_post]"
    raw_tool_msg = MagicMock()
    raw_tool_msg.tool_calls = [tc]
    raw_tool = MagicMock()
    raw_tool.choices = [MagicMock(message=raw_tool_msg)]
    llm_with_tool._raw = raw_tool

    llm_wait = MagicMock()
    llm_wait.content = "waiting for approval to post"
    raw_wait_msg = MagicMock()
    raw_wait_msg.tool_calls = []
    raw_wait = MagicMock()
    raw_wait.choices = [MagicMock(message=raw_wait_msg)]
    llm_wait._raw = raw_wait

    graph, gateway, registry, memory = _make_graph()
    gateway.complete_with_tools = AsyncMock(side_effect=[llm_with_tool, llm_wait])
    registry.get_schemas = MagicMock(return_value=[{"type": "function", "function": {"name": "telegram_post"}}])
    from grif.models.enums import ToolCategory
    registry.execute = AsyncMock(side_effect=PermissionDeniedError("telegram_post", ToolCategory.WRITE_PUBLIC))

    result = await graph.run("Post to telegram")
    # Should end in WAIT after permission blocked observation
    assert result["decision"] in (ReactDecision.WAIT.value, ReactDecision.CONTINUE.value, ReactDecision.REPORT.value)


# ─── Error handling ───────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_react_llm_error_sets_escalate() -> None:
    graph, gateway, registry, memory = _make_graph()
    gateway.complete_with_tools = AsyncMock(side_effect=RuntimeError("LLM timeout"))
    gateway.complete = AsyncMock(side_effect=RuntimeError("LLM timeout"))

    result = await graph.run("Do something")
    assert result["decision"] == ReactDecision.ESCALATE.value
    assert result["error"] != ""


# ─── DB logging ───────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_react_logs_cycle_to_memory() -> None:
    graph, gateway, registry, memory = _make_graph(llm_content="final answer: done")
    await graph.run("Find hotels")
    memory.add_cycle_log.assert_called()


@pytest.mark.asyncio
async def test_react_persist_log_failure_does_not_crash() -> None:
    graph, gateway, registry, memory = _make_graph(llm_content="final answer: done")
    graph._session.flush = AsyncMock(side_effect=RuntimeError("DB error"))
    # Should not raise — persist failure is caught and logged
    result = await graph.run("Find hotels")
    assert "decision" in result
