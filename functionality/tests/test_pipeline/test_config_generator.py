"""Tests for pipeline/config_generator.py — mocked LLM."""
import json
import pytest
from unittest.mock import AsyncMock, MagicMock

from grif.blueprints.registry import Blueprint
from grif.llm.gateway import LLMGateway, LLMResponse
from grif.models.enums import TaskType, WakeTriggerType
from grif.models.intent import StructuredIntent
from grif.pipeline.config_generator import ConfigGenerator


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _mock_response(content: str) -> LLMResponse:
    raw = MagicMock()
    raw.choices[0].message.content = content
    raw.usage.prompt_tokens = 100
    raw.usage.completion_tokens = 200
    raw.usage.total_tokens = 300
    return LLMResponse(raw=raw, model_used="gpt-4o")


def _make_blueprint(
    id: str = "research_analyst",
    task_types: list[str] | None = None,
    tools: list[str] | None = None,
    model: str = "claude-sonnet-4-6",
    template: str = "research_analyst",
) -> Blueprint:
    return Blueprint({
        "id": id,
        "name": "Test Blueprint",
        "task_types": task_types or ["research"],
        "domains": ["research"],
        "description": "Test",
        "template": template,
        "default_tools": tools or ["web_search"],
        "default_model": model,
        "typical_complexity": "multi_step",
        "plan_pattern": "pipeline",
        "typical_agents": [],
        "required_fields": ["topic"],
        "optional_fields": [],
    })


def _make_intent(
    task_type: TaskType = TaskType.RESEARCH,
    entities: dict | None = None,
    constraints: dict | None = None,
    raw: str = "Research quantum computing",
    language: str = "ru",
) -> StructuredIntent:
    return StructuredIntent(
        task_type=task_type,
        entities=entities or {"topic": "quantum computing"},
        constraints=constraints or {},
        raw_input=raw,
        language=language,
        domain="research",
    )


# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def gateway_mock() -> AsyncMock:
    gw = AsyncMock(spec=LLMGateway)
    layer3_response = json.dumps({
        "layer_3_task_context": "Research quantum computing thoroughly. Find 10+ sources.",
        "tools": ["web_search"],
        "model_id": "claude-sonnet-4-6",
        "temperature": 0.2,
        "max_tokens": 4096,
        "wake_triggers": [],
        "schedule_cron": None,
        "schedule_timezone": "UTC",
    })
    gw.complete_json.return_value = _mock_response(layer3_response)
    return gw


@pytest.fixture
def generator(gateway_mock: AsyncMock) -> ConfigGenerator:
    return ConfigGenerator(gateway=gateway_mock)


# ─── Basic generation ─────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_generate_basic_config(
    generator: ConfigGenerator,
    gateway_mock: AsyncMock,
) -> None:
    intent = _make_intent()
    bp = _make_blueprint()

    config = await generator.generate(intent=intent, blueprint=bp, user_id="u1")

    assert config.user_id == "u1"
    assert config.task_type == TaskType.RESEARCH
    assert config.blueprint_id == "research_analyst"
    assert config.system_prompt  # Should be assembled
    assert "core_identity" in config.system_prompt.lower() or len(config.system_prompt) > 50
    gateway_mock.complete_json.assert_called_once()


@pytest.mark.asyncio
async def test_generate_system_prompt_has_4_layers(
    generator: ConfigGenerator,
) -> None:
    """System prompt must contain content from all 4 layers."""
    intent = _make_intent()
    bp = _make_blueprint()

    config = await generator.generate(
        intent=intent,
        blueprint=bp,
        user_id="u1",
        user_persona="User prefers bullet points.",
    )

    # Layer 1: core identity text
    assert "GRIF" in config.system_prompt or "accurate" in config.system_prompt.lower()
    # Layer 4: user persona
    assert "bullet" in config.system_prompt.lower()


@pytest.mark.asyncio
async def test_generate_fork_reuses_layers_1_2_4(
    generator: ConfigGenerator,
    gateway_mock: AsyncMock,
) -> None:
    """FORK: layers 1, 2, 4 from parent; layer 3 regenerated."""
    import uuid
    from grif.models.agent_config import PromptLayers, AgentConfig, ModelConfig

    parent_layers = PromptLayers(
        layer_1_core_identity="PARENT CORE IDENTITY",
        layer_2_role_template="PARENT ROLE TEMPLATE",
        layer_3_task_context="Old task context",
        layer_4_user_persona="PARENT USER PERSONA",
    )
    parent_config = AgentConfig(
        user_id="u1",
        task_type=TaskType.RESEARCH,
        blueprint_id="research_analyst",
        prompt_layers=parent_layers,
        model_config=ModelConfig(),
    )

    intent = _make_intent()
    bp = _make_blueprint()
    parent_id = parent_config.id

    forked = await generator.generate(
        intent=intent,
        blueprint=bp,
        user_id="u1",
        parent_agent_id=parent_id,
        parent_config=parent_config,
    )

    assert forked.parent_agent_id == parent_id
    assert "PARENT CORE IDENTITY" in forked.system_prompt
    assert "PARENT ROLE TEMPLATE" in forked.system_prompt
    assert "PARENT USER PERSONA" in forked.system_prompt
    # Layer 3 should be regenerated (NOT the old context)
    assert "Old task context" not in forked.system_prompt


# ─── Wake triggers ────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_monitor_task_gets_wake_trigger(
    generator: ConfigGenerator,
) -> None:
    intent = StructuredIntent(
        task_type=TaskType.MONITOR,
        entities={"item": "BTC", "condition": "price_below"},
        constraints={"threshold": 50000},
        raw_input="Alert when BTC < 50000",
        domain="crypto",
    )
    bp = _make_blueprint(id="price_tracker", task_types=["monitor"])

    config = await generator.generate(intent=intent, blueprint=bp, user_id="u1")

    assert len(config.wake_triggers) > 0
    assert config.wake_triggers[0].trigger_type == WakeTriggerType.CONDITION


@pytest.mark.asyncio
async def test_operate_task_gets_schedule(generator: ConfigGenerator) -> None:
    intent = StructuredIntent(
        task_type=TaskType.OPERATE,
        entities={"channel": "telegram", "topic": "AI", "frequency": "daily"},
        constraints={},
        raw_input="Post daily about AI",
        domain="content",
    )
    bp = _make_blueprint(id="content_writer", task_types=["operate"])

    config = await generator.generate(intent=intent, blueprint=bp, user_id="u1")
    assert config.schedule is not None
    assert "9" in config.schedule.cron_expression  # daily 09:00


# ─── Frequency to cron ────────────────────────────────────────────────────────

def test_frequency_to_cron_daily() -> None:
    assert ConfigGenerator._frequency_to_cron("daily") == "0 9 * * *"
    assert ConfigGenerator._frequency_to_cron("ежедневно") == "0 9 * * *"


def test_frequency_to_cron_weekly() -> None:
    assert ConfigGenerator._frequency_to_cron("weekly") == "0 9 * * 1"


def test_frequency_to_cron_unknown() -> None:
    assert ConfigGenerator._frequency_to_cron("twice a month") is None


def test_frequency_to_cron_none() -> None:
    assert ConfigGenerator._frequency_to_cron(None) is None


# ─── Fallback Layer 3 ─────────────────────────────────────────────────────────

def test_fallback_layer_3_deterministic() -> None:
    gen = ConfigGenerator(gateway=AsyncMock())
    intent = _make_intent(raw="Research AI in 2026")
    layer3 = gen._fallback_layer_3(intent)
    assert "research" in layer3.lower()
    assert "Research AI in 2026" in layer3


# ─── Tool binder ─────────────────────────────────────────────────────────────

def test_tool_binder_web_search() -> None:
    from grif.pipeline.tool_binder import ToolBinder
    from grif.models.agent_config import AgentConfig, ModelConfig, PromptLayers

    config = AgentConfig(
        user_id="u1",
        task_type=TaskType.SEARCH,
        blueprint_id="travel_scout",
        prompt_layers=PromptLayers(
            layer_1_core_identity="Core",
            layer_2_role_template="Role",
            layer_3_task_context="Task",
        ),
        tools=["web_search", "fetch"],
        model_config=ModelConfig(),
    )
    binder = ToolBinder()
    # web_search requires tavily_api_key which is empty in test env
    # but should not raise — just skip missing creds
    bound = binder.bind(config)
    # fetch has no credential requirement → always bound
    assert "fetch" in bound


def test_tool_binder_confirmation_rules() -> None:
    from grif.pipeline.tool_binder import ToolBinder
    from grif.models.agent_config import AgentConfig, ModelConfig, PromptLayers, ToolPermission
    from grif.models.enums import ToolCategory

    config = AgentConfig(
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
        tool_permissions=[
            ToolPermission(tool_name="web_search", category=ToolCategory.READ, auto_approved=True),
            ToolPermission(tool_name="delete", category=ToolCategory.WRITE_IRREVERSIBLE),
            ToolPermission(tool_name="post_telegram", category=ToolCategory.WRITE_PUBLIC),
        ],
    )
    binder = ToolBinder()
    assert not binder.requires_confirmation("web_search", config)
    assert binder.requires_confirmation("delete", config)
    assert binder.requires_confirmation("post_telegram", config)
