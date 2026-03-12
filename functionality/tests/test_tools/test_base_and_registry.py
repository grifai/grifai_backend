"""Tests for tools/base.py and tools/registry.py."""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Any

from grif.models.agent_config import AgentConfig, ModelConfig, PromptLayers, ToolPermission
from grif.models.enums import TaskType, ToolCategory
from grif.tools.base import BaseTool, ToolResult
from grif.tools.registry import PermissionDeniedError, ToolRegistry


# ─── Fixtures ─────────────────────────────────────────────────────────────────

class MockReadTool(BaseTool):
    name = "mock_read"
    description = "A mock read tool for testing."
    category = ToolCategory.READ

    async def execute(self, query: str, **kwargs: Any) -> ToolResult:
        return ToolResult(tool_name=self.name, success=True, output=f"Result for: {query}")


class MockWritePublicTool(BaseTool):
    name = "mock_post"
    description = "A mock WRITE_PUBLIC tool."
    category = ToolCategory.WRITE_PUBLIC

    async def execute(self, text: str, **kwargs: Any) -> ToolResult:
        return ToolResult(tool_name=self.name, success=True, output=f"Posted: {text}")


class MockIrreversibleTool(BaseTool):
    name = "mock_delete"
    description = "A mock WRITE_IRREVERSIBLE tool."
    category = ToolCategory.WRITE_IRREVERSIBLE

    async def execute(self, target: str, **kwargs: Any) -> ToolResult:
        return ToolResult(tool_name=self.name, success=True, output=f"Deleted: {target}")


def _make_config(permissions: list[ToolPermission] | None = None) -> AgentConfig:
    return AgentConfig(
        user_id="u1",
        task_type=TaskType.EXECUTE,
        blueprint_id="generic_worker",
        prompt_layers=PromptLayers(
            layer_1_core_identity="Core",
            layer_2_role_template="Role",
            layer_3_task_context="Task",
        ),
        tools=[],
        model_config=ModelConfig(),
        tool_permissions=permissions or [],
    )


# ─── BaseTool ─────────────────────────────────────────────────────────────────

def test_base_tool_schema() -> None:
    tool = MockReadTool()
    schema = tool.to_function_schema()
    assert schema["type"] == "function"
    assert schema["function"]["name"] == "mock_read"
    assert "parameters" in schema["function"]


def test_tool_result_to_observation_success() -> None:
    r = ToolResult(tool_name="web_search", success=True, output="Found 5 hotels")
    obs = r.to_observation()
    assert "Found 5 hotels" in obs
    assert "web_search" in obs


def test_tool_result_to_observation_error() -> None:
    r = ToolResult(tool_name="web_search", success=False, error="API timeout")
    obs = r.to_observation()
    assert "ERROR" in obs
    assert "API timeout" in obs


# ─── ToolRegistry: READ ───────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_registry_read_tool_executes_without_approval() -> None:
    registry = ToolRegistry()
    tool = MockReadTool()
    registry.register(tool)

    config = _make_config([
        ToolPermission(tool_name="mock_read", category=ToolCategory.READ, auto_approved=True)
    ])
    result = await registry.execute("mock_read", {"query": "test"}, config)
    assert result.success
    assert "test" in result.output


# ─── ToolRegistry: WRITE_IRREVERSIBLE ─────────────────────────────────────────

@pytest.mark.asyncio
async def test_registry_irreversible_always_raises() -> None:
    registry = ToolRegistry()
    registry.register(MockIrreversibleTool())

    config = _make_config([
        ToolPermission(
            tool_name="mock_delete",
            category=ToolCategory.WRITE_IRREVERSIBLE,
            auto_approved=True,  # Even auto_approved cannot bypass IRREVERSIBLE
        )
    ])
    with pytest.raises(PermissionDeniedError):
        await registry.execute("mock_delete", {"target": "file.txt"}, config)


# ─── ToolRegistry: WRITE_PUBLIC ──────────────────────────────────────────────

@pytest.mark.asyncio
async def test_registry_write_public_raises_before_approval() -> None:
    registry = ToolRegistry()
    registry.register(MockWritePublicTool())

    config = _make_config([
        ToolPermission(
            tool_name="mock_post",
            category=ToolCategory.WRITE_PUBLIC,
            auto_approved=False,
        )
    ])
    with pytest.raises(PermissionDeniedError):
        await registry.execute("mock_post", {"text": "Hello"}, config)


@pytest.mark.asyncio
async def test_registry_write_public_executes_after_escalation() -> None:
    registry = ToolRegistry()
    registry.register(MockWritePublicTool())

    config = _make_config([
        ToolPermission(
            tool_name="mock_post",
            category=ToolCategory.WRITE_PUBLIC,
            auto_approved=True,  # Already escalated
        )
    ])
    result = await registry.execute("mock_post", {"text": "Hello!"}, config)
    assert result.success


@pytest.mark.asyncio
async def test_registry_write_public_draft_mode() -> None:
    registry = ToolRegistry()
    registry.register(MockWritePublicTool())

    config = _make_config([
        ToolPermission(tool_name="mock_post", category=ToolCategory.WRITE_PUBLIC)
    ])
    # Draft mode — returns preview without executing
    result = await registry.execute("mock_post", {"text": "Hello"}, config, draft_mode=True)
    assert result.success
    assert result.metadata.get("is_draft") is True


# ─── Trust Escalation ─────────────────────────────────────────────────────────

def test_trust_escalation_after_n_approvals() -> None:
    registry = ToolRegistry()
    config = _make_config([
        ToolPermission(
            tool_name="mock_post",
            category=ToolCategory.WRITE_PUBLIC,
            auto_approved=False,
            approval_count=0,
            trust_threshold=3,
        )
    ])

    # Not escalated yet
    assert not registry.record_approval("mock_post", config)
    assert not registry.record_approval("mock_post", config)
    # Third approval → escalated
    escalated = registry.record_approval("mock_post", config)
    assert escalated
    perm = config.tool_permissions[0]
    assert perm.auto_approved is True
    assert perm.approval_count == 3


def test_trust_escalation_irreversible_not_affected() -> None:
    """WRITE_IRREVERSIBLE tools are never escalated."""
    registry = ToolRegistry()
    config = _make_config([
        ToolPermission(
            tool_name="mock_delete",
            category=ToolCategory.WRITE_IRREVERSIBLE,
        )
    ])
    # Record many approvals — should never flip to auto
    for _ in range(10):
        registry.record_approval("mock_delete", config)
    perm = config.tool_permissions[0]
    assert perm.auto_approved is False


# ─── ToolRegistry: unknown tool ───────────────────────────────────────────────

@pytest.mark.asyncio
async def test_registry_unknown_tool_returns_error() -> None:
    registry = ToolRegistry()
    config = _make_config()
    result = await registry.execute("nonexistent_tool", {}, config)
    assert not result.success
    assert "not found" in result.error


# ─── get_schemas ─────────────────────────────────────────────────────────────

def test_registry_get_schemas() -> None:
    registry = ToolRegistry()
    registry.register(MockReadTool())
    registry.register(MockWritePublicTool())

    schemas = registry.get_schemas()
    assert len(schemas) == 2
    names = {s["function"]["name"] for s in schemas}
    assert "mock_read" in names
    assert "mock_post" in names


def test_registry_get_schemas_filtered() -> None:
    registry = ToolRegistry()
    registry.register(MockReadTool())
    registry.register(MockWritePublicTool())

    schemas = registry.get_schemas(["mock_read"])
    assert len(schemas) == 1
    assert schemas[0]["function"]["name"] == "mock_read"
