"""
ReAct Loop — LangGraph StateGraph implementation.

Graph topology:
  START → inject_memory → reason → [tool_call?] → act → observe → decide
                                       ↑_____________________|  (CONTINUE)
                                   END ◄───────────────────── (REPORT | SLEEP | WAIT | ESCALATE)

Architecture rules:
- Max 20 cycles (config.react_max_cycles)
- LLM called ONLY in `reason` node (this is the 3rd authorised LLM call point)
- Model selected by ModelRouter (cascading: Haiku for lookups, Sonnet for reasoning)
- Every cycle is logged to AgentLogDB (Audit Trail)
- On max cycles exceeded → decision = ESCALATE
"""

from __future__ import annotations

import json
from typing import Annotated, Any, TypedDict

import structlog
from langchain_core.messages import BaseMessage
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from sqlalchemy.ext.asyncio import AsyncSession

from grif.config import get_settings
from grif.llm.gateway import LLMGateway
from grif.models.agent_config import AgentConfig
from grif.models.db import AgentLogDB
from grif.models.enums import ReactDecision
from grif.models.memory import ReActCycleLog
from grif.runtime.memory_manager import MemoryManager
from grif.runtime.model_router import select_model_id
from grif.tools.registry import PermissionDeniedError, ToolRegistry

log = structlog.get_logger(__name__)
settings = get_settings()


# ── State ─────────────────────────────────────────────────────────────────────

class ReActState(TypedDict):
    messages: Annotated[list[dict], add_messages]
    cycle_count: int
    decision: str                  # ReactDecision value
    agent_id: str
    user_id: str
    pending_tool_call: dict | None # Populated by reason, consumed by act
    last_observation: str
    final_result: str
    error: str


# ── Decision detection helpers ─────────────────────────────────────────────────

def _msg_content(msg: Any) -> str:
    """Extract text content from either a dict or a LangChain BaseMessage."""
    if isinstance(msg, dict):
        return msg.get("content", "") or ""
    return getattr(msg, "content", "") or ""


def _msg_role(msg: Any) -> str:
    """Extract role from either a dict or a LangChain BaseMessage."""
    if isinstance(msg, dict):
        return msg.get("role", "") or ""
    # LangChain BaseMessage types: HumanMessage→human, AIMessage→ai, SystemMessage→system, ToolMessage→tool
    type_name = type(msg).__name__.lower()
    if "human" in type_name:
        return "user"
    if "ai" in type_name or "assistant" in type_name:
        return "assistant"
    if "system" in type_name:
        return "system"
    if "tool" in type_name:
        return "tool"
    return getattr(msg, "role", "") or ""


_SLEEP_SIGNALS = ["sleep", "спать", "засыпаю", "done for now", "задача выполнена", "monitor set"]
_WAIT_SIGNALS = ["wait", "waiting for", "need user input", "ожидаю", "нужен ответ пользователя"]
_COMPLETE_SIGNALS = ["final answer:", "result:", "итого:", "готово:", "вот результат"]


def _infer_decision(text: str, cycle: int, max_cycles: int) -> ReactDecision:
    """Deterministically infer decision from LLM response text."""
    lower = text.lower()
    if cycle >= max_cycles:
        return ReactDecision.ESCALATE
    if any(s in lower for s in _SLEEP_SIGNALS):
        return ReactDecision.SLEEP
    if any(s in lower for s in _WAIT_SIGNALS):
        return ReactDecision.WAIT
    if any(s in lower for s in _COMPLETE_SIGNALS):
        return ReactDecision.REPORT
    return ReactDecision.CONTINUE


# ── Graph nodes ───────────────────────────────────────────────────────────────

class ReActGraph:
    """
    Builds and runs the LangGraph ReAct state machine for one agent.

    Usage:
        graph = ReActGraph(gateway, registry, memory_manager, agent_config, session)
        result = await graph.run(user_message="Find hotels in Paris")
    """

    def __init__(
        self,
        gateway: LLMGateway,
        registry: ToolRegistry,
        memory: MemoryManager,
        agent_config: AgentConfig,
        session: AsyncSession,
    ) -> None:
        self._gateway = gateway
        self._registry = registry
        self._memory = memory
        self._config = agent_config
        self._session = session
        self._graph = self._build_graph()

    def _build_graph(self):
        g = StateGraph(ReActState)

        g.add_node("inject_memory", self._node_inject_memory)
        g.add_node("reason", self._node_reason)
        g.add_node("act", self._node_act)
        g.add_node("observe", self._node_observe)
        g.add_node("decide", self._node_decide)

        g.set_entry_point("inject_memory")
        g.add_edge("inject_memory", "reason")
        g.add_edge("reason", "act")
        g.add_edge("act", "observe")
        g.add_edge("observe", "decide")

        g.add_conditional_edges(
            "decide",
            self._route_after_decide,
            {
                "continue": "reason",
                "end": END,
            },
        )

        return g.compile()

    # ── Node: inject_memory ───────────────────────────────────────────────────

    async def _node_inject_memory(self, state: ReActState) -> dict:
        ctx = await self._memory.get_working_context()
        messages = list(state["messages"])

        # Inject working memory as system message before user message
        if ctx:
            messages = [{"role": "system", "content": f"[Memory]\n{ctx}"}, *messages]

        return {
            "messages": messages,
            "cycle_count": state.get("cycle_count", 0),
            "decision": "",
            "pending_tool_call": None,
            "last_observation": "",
            "final_result": "",
            "error": "",
        }

    # ── Node: reason ──────────────────────────────────────────────────────────

    async def _node_reason(self, state: ReActState) -> dict:
        cycle = state["cycle_count"] + 1
        log.debug("react_reason_start", cycle=cycle, agent_id=state["agent_id"])

        tool_schemas = self._registry.get_schemas(self._config.tools)
        model_id = select_model_id(
            step_description=_msg_content(state["messages"][-1]) if state["messages"] else "",
            agent_config=self._config,
        )

        messages = [{"role": "system", "content": self._config.system_prompt}] + list(
            state["messages"]
        )

        try:
            if tool_schemas:
                response = await self._gateway.complete_with_tools(
                    messages=messages,
                    tools=tool_schemas,
                    model_id=model_id,
                    temperature=self._config.get_model_config().temperature,
                    max_tokens=self._config.get_model_config().max_tokens,
                    user_id=state["user_id"],
                    agent_id=state["agent_id"],
                )
            else:
                response = await self._gateway.complete(
                    messages=messages,
                    model_id=model_id,
                    temperature=self._config.get_model_config().temperature,
                    max_tokens=self._config.get_model_config().max_tokens,
                    user_id=state["user_id"],
                    agent_id=state["agent_id"],
                )
        except Exception as exc:
            log.error("react_reason_failed", cycle=cycle, error=str(exc))
            return {
                "messages": [{"role": "assistant", "content": f"Error: {exc}"}],
                "cycle_count": cycle,
                "decision": ReactDecision.ESCALATE,
                "error": str(exc),
            }

        # Check for tool call in the response
        raw = response._raw
        pending_tool: dict | None = None

        tool_calls = getattr(raw.choices[0].message, "tool_calls", None)
        if tool_calls:
            tc = tool_calls[0]
            pending_tool = {
                "id": tc.id,
                "name": tc.function.name,
                "arguments": json.loads(tc.function.arguments),
            }
            content = f"[Calling tool: {tc.function.name}]"
        else:
            content = response.content

        new_msg = {"role": "assistant", "content": content}
        return {
            "messages": [new_msg],
            "cycle_count": cycle,
            "pending_tool_call": pending_tool,
            "decision": "",
        }

    # ── Node: act ─────────────────────────────────────────────────────────────

    async def _node_act(self, state: ReActState) -> dict:
        tc = state.get("pending_tool_call")
        if not tc:
            return {}

        tool_name = tc["name"]
        kwargs = tc["arguments"]
        log.debug("react_act", tool=tool_name, cycle=state["cycle_count"])

        try:
            result = await self._registry.execute(
                tool_name=tool_name,
                kwargs=kwargs,
                agent_config=self._config,
            )
            observation = result.to_observation()
        except PermissionDeniedError as e:
            observation = f"[BLOCKED] {e}"
            # Tool needs approval → will surface as WAIT decision
        except Exception as exc:
            observation = f"[ERROR] {exc}"

        tool_msg = {
            "role": "tool",
            "tool_call_id": tc.get("id", ""),
            "name": tool_name,
            "content": observation,
        }
        return {"messages": [tool_msg], "last_observation": observation}

    # ── Node: observe ─────────────────────────────────────────────────────────

    async def _node_observe(self, state: ReActState) -> dict:
        # Observation already added to messages in act node
        # Log the cycle to DB
        cycle = state["cycle_count"]
        messages = state["messages"]

        thought = ""
        action = "none"
        action_input: dict = {}
        observation = state.get("last_observation", "")

        # Extract thought from last assistant message
        for msg in reversed(messages):
            if _msg_role(msg) == "assistant":
                thought = _msg_content(msg)
                break

        tc = state.get("pending_tool_call")
        if tc:
            action = tc.get("name", "none")
            action_input = tc.get("arguments", {})

        log_entry = ReActCycleLog(
            cycle_number=cycle,
            thought=thought[:1000],
            action=action,
            action_input=action_input,
            observation=observation[:1000],
            decision="",  # Filled by decide node
        )
        await self._memory.add_cycle_log(log_entry)

        return {}

    # ── Node: decide ─────────────────────────────────────────────────────────

    async def _node_decide(self, state: ReActState) -> dict:
        cycle = state["cycle_count"]
        messages = state["messages"]

        # Get last assistant message
        last_text = ""
        for msg in reversed(messages):
            if _msg_role(msg) == "assistant":
                last_text = _msg_content(msg)
                break

        # No pending tool call + we got a text response = done or deciding
        has_tool_call = bool(state.get("pending_tool_call"))
        if has_tool_call:
            # Still executing tools → continue
            decision = ReactDecision.CONTINUE
        else:
            decision = _infer_decision(last_text, cycle, settings.react_max_cycles)

        # Persist decision to DB log
        await self._persist_log(
            state=state,
            thought=last_text[:500],
            action=state.get("pending_tool_call", {}).get("name", "none") if state.get("pending_tool_call") else "none",
            observation=state.get("last_observation", ""),
            decision=decision,
        )

        final_result = last_text if decision == ReactDecision.REPORT else ""

        return {
            "decision": decision.value,
            "final_result": final_result,
            "pending_tool_call": None,  # Clear after use
        }

    # ── Routing ───────────────────────────────────────────────────────────────

    def _route_after_decide(self, state: ReActState) -> str:
        decision = state.get("decision", "")
        if decision == ReactDecision.CONTINUE:
            return "continue"
        return "end"

    # ── DB logging ────────────────────────────────────────────────────────────

    async def _persist_log(
        self,
        state: ReActState,
        thought: str,
        action: str,
        observation: str,
        decision: ReactDecision,
    ) -> None:
        try:
            entry = AgentLogDB(
                agent_id=state["agent_id"],
                user_id=state["user_id"],
                cycle_number=state["cycle_count"],
                thought=thought,
                action=action,
                action_input=state.get("pending_tool_call", {}).get("arguments", {}) or {},
                observation=observation,
                decision=decision.value,
            )
            self._session.add(entry)
            await self._session.flush()
        except Exception as exc:
            log.warning("react_log_persist_failed", error=str(exc))

    # ── Public run ────────────────────────────────────────────────────────────

    async def run(
        self,
        user_message: str,
        initial_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Run the ReAct loop for a single user request.
        Returns final state dict with `final_result` and `decision`.
        """
        initial_state: ReActState = {
            "messages": [{"role": "user", "content": user_message}],
            "cycle_count": 0,
            "decision": "",
            "agent_id": str(self._config.id),
            "user_id": self._config.user_id,
            "pending_tool_call": None,
            "last_observation": "",
            "final_result": "",
            "error": "",
        }

        log.info(
            "react_loop_start",
            agent_id=str(self._config.id),
            task_type=self._config.task_type,
        )

        final_state = await self._graph.ainvoke(initial_state)

        log.info(
            "react_loop_end",
            agent_id=str(self._config.id),
            decision=final_state.get("decision"),
            cycles=final_state.get("cycle_count"),
        )
        return final_state
