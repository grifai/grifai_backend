"""Tests for blueprints/registry.py and pipeline/blueprint_selector.py."""
import pytest
from unittest.mock import AsyncMock, MagicMock

from grif.blueprints.registry import BlueprintRegistry, Blueprint, _load_global_blueprints
from grif.models.enums import TaskType
from grif.models.intent import StructuredIntent
from grif.pipeline.blueprint_selector import BlueprintSelector


# ─── BlueprintRegistry (global) ───────────────────────────────────────────────

@pytest.fixture
def session_no_personal() -> AsyncMock:
    session = AsyncMock()
    result = MagicMock()
    result.scalars.return_value.all.return_value = []
    session.execute.return_value = result
    return session


@pytest.mark.asyncio
async def test_registry_loads_global_blueprints(session_no_personal: AsyncMock) -> None:
    registry = BlueprintRegistry(session=session_no_personal)
    blueprints = registry.list_all()
    assert len(blueprints) >= 7
    ids = {bp.id for bp in blueprints}
    assert "generic_worker" in ids
    assert "research_analyst" in ids
    assert "travel_scout" in ids


@pytest.mark.asyncio
async def test_registry_find_travel_by_domain(session_no_personal: AsyncMock) -> None:
    registry = BlueprintRegistry(session=session_no_personal)

    # Patch scores to return empty
    result = MagicMock()
    result.all.return_value = []
    session_no_personal.execute.return_value = result

    bp = await registry.find_best(
        task_type="search",
        domain="travel",
        user_id="u1",
    )
    assert bp.id == "travel_scout"


@pytest.mark.asyncio
async def test_registry_find_research_by_type(session_no_personal: AsyncMock) -> None:
    registry = BlueprintRegistry(session=session_no_personal)
    result = MagicMock()
    result.all.return_value = []
    session_no_personal.execute.return_value = result

    bp = await registry.find_best(
        task_type="research",
        domain="general",
        user_id="u1",
    )
    assert bp.id == "research_analyst"


@pytest.mark.asyncio
async def test_registry_fallback_generic_worker(session_no_personal: AsyncMock) -> None:
    """Unknown domain + task_type with no specific blueprint → generic_worker."""
    registry = BlueprintRegistry(session=session_no_personal)
    result = MagicMock()
    result.all.return_value = []
    session_no_personal.execute.return_value = result

    bp = await registry.find_best(
        task_type="execute",
        domain="unknown_domain",
        user_id="u1",
    )
    assert bp.id == "generic_worker"


@pytest.mark.asyncio
async def test_registry_get_by_id(session_no_personal: AsyncMock) -> None:
    registry = BlueprintRegistry(session=session_no_personal)
    bp = registry.get_by_id("content_writer")
    assert bp is not None
    assert bp.id == "content_writer"


@pytest.mark.asyncio
async def test_registry_get_by_id_missing(session_no_personal: AsyncMock) -> None:
    registry = BlueprintRegistry(session=session_no_personal)
    assert registry.get_by_id("nonexistent") is None


# ─── BlueprintSelector ────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_blueprint_selector_returns_blueprint(session_no_personal: AsyncMock) -> None:
    result = MagicMock()
    result.all.return_value = []
    session_no_personal.execute.return_value = result

    selector = BlueprintSelector(session=session_no_personal)
    intent = StructuredIntent(
        task_type=TaskType.RESEARCH,
        raw_input="Research quantum computing",
        domain="research",
    )
    bp = await selector.select(intent, user_id="u1")
    assert isinstance(bp, Blueprint)
    assert bp.id == "research_analyst"


# ─── Blueprint structure ──────────────────────────────────────────────────────

def test_blueprint_fields_present() -> None:
    blueprints = _load_global_blueprints()
    for bp in blueprints.values():
        assert bp.id
        assert bp.name
        assert isinstance(bp.task_types, list)
        assert isinstance(bp.default_tools, list)
        assert bp.template  # Must map to a YAML file
