"""Tests for pipeline/router.py — fully deterministic, no LLM, mocked DB."""
import pytest
from unittest.mock import AsyncMock, MagicMock

from grif.models.enums import RouterDecision, TaskType
from grif.models.intent import StructuredIntent
from grif.pipeline.router import Router, _jaccard, _tokenize


# ─── Unit: helpers ────────────────────────────────────────────────────────────

def test_tokenize_basic() -> None:
    tokens = _tokenize("Find a hotel in Paris")
    assert "hotel" in tokens
    assert "paris" in tokens
    # "a" is a stop word (length ≤ 2)
    assert "a" not in tokens


def test_tokenize_russian() -> None:
    tokens = _tokenize("Найди отель в Париже")
    assert "найди" in tokens
    assert "отель" in tokens


def test_jaccard_identical() -> None:
    a = {"hotel", "paris", "budget"}
    assert _jaccard(a, a) == 1.0


def test_jaccard_disjoint() -> None:
    a = {"hotel", "paris"}
    b = {"flight", "london"}
    assert _jaccard(a, b) == 0.0


def test_jaccard_partial() -> None:
    a = {"hotel", "paris", "cheap"}
    b = {"hotel", "paris", "luxury"}
    # intersection=2, union=4 → 0.5
    assert _jaccard(a, b) == pytest.approx(0.5)


def test_jaccard_empty() -> None:
    assert _jaccard(set(), {"a"}) == 0.0


# ─── Router: no existing agents → NEW ─────────────────────────────────────────

@pytest.fixture
def empty_session() -> AsyncMock:
    session = AsyncMock()
    result = MagicMock()
    result.scalars.return_value.all.return_value = []
    session.execute.return_value = result
    return session


@pytest.fixture
def router(empty_session: AsyncMock) -> Router:
    return Router(session=empty_session)


def _make_intent(task_type: TaskType = TaskType.SEARCH, raw: str = "Find hotel Paris") -> StructuredIntent:
    return StructuredIntent(
        task_type=task_type,
        entities={"destination": "Paris"},
        raw_input=raw,
    )


@pytest.mark.asyncio
async def test_router_new_when_no_agents(router: Router) -> None:
    intent = _make_intent()
    result = await router.route(intent, user_id="u1")
    assert result.decision == RouterDecision.NEW
    assert result.agent_id is None


# ─── Router: existing active agent (EXISTING) ─────────────────────────────────

@pytest.fixture
def session_with_active_agent() -> AsyncMock:
    import uuid
    agent_id = uuid.uuid4()

    agent = MagicMock()
    agent.id = agent_id
    agent.state = "active"
    agent.task_type = "search"
    agent.config = {
        "metadata": {
            "raw_input": "Find hotel Paris cheap",
            "domain": "travel",
            "entities": {"destination": "Paris"},
            "constraints": {},
        }
    }

    session = AsyncMock()
    result = MagicMock()
    result.scalars.return_value.all.side_effect = [
        [agent],   # active check
        [],        # sleeping check
        [],        # archived check
    ]
    session.execute.return_value = result
    return session


@pytest.mark.asyncio
async def test_router_existing_for_similar_active(session_with_active_agent: AsyncMock) -> None:
    router = Router(session=session_with_active_agent)
    intent = _make_intent(TaskType.SEARCH, "Find hotel Paris budget")
    result = await router.route(intent, user_id="u1")
    # Should be EXISTING or FORK depending on similarity — at least not NEW
    assert result.decision in (RouterDecision.EXISTING, RouterDecision.FORK)
    assert result.agent_id is not None


# ─── Router: different task type → NEW ────────────────────────────────────────

@pytest.fixture
def session_with_wrong_type_agent() -> AsyncMock:
    import uuid
    agent = MagicMock()
    agent.id = uuid.uuid4()
    agent.state = "active"
    agent.task_type = "research"  # Different from SEARCH
    agent.config = {"metadata": {"raw_input": "Research quantum computing"}}

    session = AsyncMock()
    result = MagicMock()
    # Returns empty for all states because task_type filter is in the query
    result.scalars.return_value.all.return_value = []
    session.execute.return_value = result
    return session


@pytest.mark.asyncio
async def test_router_new_for_different_task_type(
    session_with_wrong_type_agent: AsyncMock,
) -> None:
    router = Router(session=session_with_wrong_type_agent)
    intent = _make_intent(TaskType.SEARCH, "Find hotel Paris")
    result = await router.route(intent, user_id="u1")
    assert result.decision == RouterDecision.NEW


# ─── Jaccard threshold boundary ───────────────────────────────────────────────

def test_jaccard_threshold_boundary() -> None:
    """Verify threshold math: 0.8 means 4/5 shared tokens → EXISTING."""
    a = {"hotel", "paris", "cheap", "budget", "travel"}
    b = {"hotel", "paris", "cheap", "budget", "luxury"}  # 4/6 = 0.666 → FORK or below
    sim = _jaccard(a, b)
    assert sim < 0.80

    c = {"hotel", "paris", "cheap", "budget", "travel"}
    d = {"hotel", "paris", "cheap", "budget", "travel"}  # 5/5 = 1.0 → SKIP
    assert _jaccard(c, d) == 1.0
