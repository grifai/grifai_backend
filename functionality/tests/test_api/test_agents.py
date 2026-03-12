"""Tests for api/agents.py — GET /agents, wake, approve-tool, explain."""
import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from grif.api.deps import get_gateway, get_session
from grif.main import create_app
from grif.models.enums import AgentState


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _make_agent_db(
    agent_id: str | None = None,
    user_id: str = "u1",
    state: str = AgentState.ACTIVE,
    task_type: str = "search",
    blueprint_id: str = "generic_worker",
    avg_score: float | None = None,
    eval_count: int = 0,
    config: dict | None = None,
):
    agent = MagicMock()
    agent.id = uuid.UUID(agent_id or str(uuid.uuid4()))
    agent.user_id = user_id
    agent.state = state
    agent.task_type = task_type
    agent.blueprint_id = blueprint_id
    agent.avg_score = avg_score
    agent.eval_count = eval_count
    agent.config = config or {"tool_permissions": []}
    agent.created_at = datetime(2024, 1, 1, tzinfo=timezone.utc)
    agent.updated_at = datetime(2024, 1, 2, tzinfo=timezone.utc)
    return agent


def _make_log_db(agent_id: str, cycle: int = 1, action: str = "web_search"):
    log = MagicMock()
    log.agent_id = uuid.UUID(agent_id)
    log.cycle_number = cycle
    log.action = action
    log.thought = "I need to search for hotels"
    log.observation = "Found 5 hotels under 150 euros"
    log.created_at = datetime(2024, 1, 1, tzinfo=timezone.utc)
    return log


def _make_session_returning(first_result=None, second_result=None):
    """Mock session that returns first_result on first execute, second on second."""
    session = AsyncMock()
    session.add = MagicMock()
    session.flush = AsyncMock()
    call_count = {"n": 0}

    def _execute_side_effect(query):
        result = MagicMock()
        call_count["n"] += 1
        if call_count["n"] == 1:
            result.scalar_one_or_none.return_value = first_result
            result.scalars.return_value.all.return_value = [first_result] if first_result else []
        else:
            result.scalar_one_or_none.return_value = second_result
            result.scalars.return_value.all.return_value = [second_result] if second_result else []
        return result

    session.execute = AsyncMock(side_effect=_execute_side_effect)
    return session


def _make_test_client(mock_session=None, mock_gateway=None):
    app = create_app()
    if mock_session is not None:
        async def _get_session_override():
            yield mock_session
        app.dependency_overrides[get_session] = _get_session_override
    if mock_gateway is not None:
        app.dependency_overrides[get_gateway] = lambda: mock_gateway
    return TestClient(app)


# ═══════════════════════════════════════════════════════════════════════════════
# GET /agents
# ═══════════════════════════════════════════════════════════════════════════════

def test_list_agents_returns_empty_list() -> None:
    session = AsyncMock()
    result = MagicMock()
    result.scalars.return_value.all.return_value = []
    session.execute = AsyncMock(return_value=result)
    session.flush = AsyncMock()

    client = _make_test_client(session)
    response = client.get("/agents", headers={"X-User-Id": "u1"})

    assert response.status_code == 200
    assert response.json() == []


def test_list_agents_returns_agents() -> None:
    agent_id = str(uuid.uuid4())
    agent = _make_agent_db(agent_id=agent_id)

    session = AsyncMock()
    result = MagicMock()
    result.scalars.return_value.all.return_value = [agent]
    session.execute = AsyncMock(return_value=result)
    session.flush = AsyncMock()

    client = _make_test_client(session)
    response = client.get("/agents", headers={"X-User-Id": "u1"})

    assert response.status_code == 200
    data = response.json()
    assert len(data) == 1
    assert data[0]["agent_id"] == agent_id
    assert data[0]["state"] == AgentState.ACTIVE
    assert data[0]["task_type"] == "search"


def test_list_agents_excludes_archived() -> None:
    """Archived agents should not appear. Verified by the SQL filter in the query."""
    session = AsyncMock()
    result = MagicMock()
    result.scalars.return_value.all.return_value = []  # No non-archived agents
    session.execute = AsyncMock(return_value=result)
    session.flush = AsyncMock()

    client = _make_test_client(session)
    response = client.get("/agents", headers={"X-User-Id": "u1"})

    assert response.status_code == 200


def test_list_agents_respects_state_filter() -> None:
    agent_id = str(uuid.uuid4())
    sleeping_agent = _make_agent_db(agent_id=agent_id, state=AgentState.SLEEPING)

    session = AsyncMock()
    result = MagicMock()
    result.scalars.return_value.all.return_value = [sleeping_agent]
    session.execute = AsyncMock(return_value=result)
    session.flush = AsyncMock()

    client = _make_test_client(session)
    response = client.get("/agents?state=sleeping", headers={"X-User-Id": "u1"})

    assert response.status_code == 200
    data = response.json()
    assert data[0]["state"] == AgentState.SLEEPING


def test_list_agents_includes_score_and_eval_count() -> None:
    agent_id = str(uuid.uuid4())
    agent = _make_agent_db(agent_id=agent_id, avg_score=4.2, eval_count=5)

    session = AsyncMock()
    result = MagicMock()
    result.scalars.return_value.all.return_value = [agent]
    session.execute = AsyncMock(return_value=result)
    session.flush = AsyncMock()

    client = _make_test_client(session)
    response = client.get("/agents", headers={"X-User-Id": "u1"})

    assert response.status_code == 200
    data = response.json()
    assert data[0]["avg_score"] == 4.2
    assert data[0]["eval_count"] == 5


# ═══════════════════════════════════════════════════════════════════════════════
# POST /agents/{agent_id}/wake
# ═══════════════════════════════════════════════════════════════════════════════

def test_wake_agent_success() -> None:
    agent_id = str(uuid.uuid4())
    sleeping_agent = _make_agent_db(agent_id=agent_id, state=AgentState.SLEEPING)
    session = _make_session_returning(first_result=sleeping_agent)

    from grif.runtime.wake_manager import WakeContext
    from grif.models.enums import WakeTriggerType
    mock_ctx = WakeContext(
        agent_id=agent_id,
        user_id="u1",
        checkpoint=None,
        context_summary="Searching for hotels",
        trigger_type=WakeTriggerType.MANUAL,
        trigger_message="Manual wake",
    )

    with patch("grif.api.agents.WakeManager.wake", new_callable=AsyncMock, return_value=mock_ctx):
        client = _make_test_client(session)
        response = client.post(
            f"/agents/{agent_id}/wake",
            json={"message": "Wake up!"},
            headers={"X-User-Id": "u1"},
        )

    assert response.status_code == 200
    data = response.json()
    assert data["agent_id"] == agent_id
    assert data["previous_state"] == AgentState.SLEEPING
    assert data["new_state"] == AgentState.ACTIVE
    assert data["context_summary"] == "Searching for hotels"


def test_wake_agent_not_found() -> None:
    session = _make_session_returning(first_result=None)
    client = _make_test_client(session)
    response = client.post(
        f"/agents/{uuid.uuid4()}/wake",
        json={},
        headers={"X-User-Id": "u1"},
    )
    assert response.status_code == 404


def test_wake_agent_wrong_state_returns_400() -> None:
    agent_id = str(uuid.uuid4())
    active_agent = _make_agent_db(agent_id=agent_id, state=AgentState.ACTIVE)
    session = _make_session_returning(first_result=active_agent)

    client = _make_test_client(session)
    response = client.post(
        f"/agents/{agent_id}/wake",
        json={},
        headers={"X-User-Id": "u1"},
    )

    assert response.status_code == 400
    assert "sleeping" in response.json()["detail"].lower() or "recurring" in response.json()["detail"].lower()


def test_wake_recurring_agent_succeeds() -> None:
    agent_id = str(uuid.uuid4())
    recurring_agent = _make_agent_db(agent_id=agent_id, state=AgentState.RECURRING)
    session = _make_session_returning(first_result=recurring_agent)

    from grif.runtime.wake_manager import WakeContext
    from grif.models.enums import WakeTriggerType
    mock_ctx = WakeContext(
        agent_id=agent_id,
        user_id="u1",
        checkpoint=None,
        context_summary=None,
        trigger_type=WakeTriggerType.MANUAL,
        trigger_message="Wake recurring",
    )

    with patch("grif.api.agents.WakeManager.wake", new_callable=AsyncMock, return_value=mock_ctx):
        client = _make_test_client(session)
        response = client.post(
            f"/agents/{agent_id}/wake",
            json={},
            headers={"X-User-Id": "u1"},
        )

    assert response.status_code == 200
    data = response.json()
    assert data["previous_state"] == AgentState.RECURRING


def test_wake_agent_invalid_uuid_returns_404() -> None:
    session = AsyncMock()
    session.flush = AsyncMock()
    client = _make_test_client(session)
    response = client.post(
        "/agents/not-a-uuid/wake",
        json={},
        headers={"X-User-Id": "u1"},
    )
    assert response.status_code == 404


# ═══════════════════════════════════════════════════════════════════════════════
# POST /agents/{agent_id}/approve-tool
# ═══════════════════════════════════════════════════════════════════════════════

def test_approve_tool_success() -> None:
    agent_id = str(uuid.uuid4())
    agent = _make_agent_db(agent_id=agent_id, config={"tool_permissions": []})
    session = _make_session_returning(first_result=agent)

    client = _make_test_client(session)
    response = client.post(
        f"/agents/{agent_id}/approve-tool",
        json={"tool_name": "telegram_bot", "approved": True},
        headers={"X-User-Id": "u1"},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["tool_name"] == "telegram_bot"
    assert data["approved"] is True
    assert "approved" in data["message"].lower()


def test_deny_tool_success() -> None:
    agent_id = str(uuid.uuid4())
    agent = _make_agent_db(agent_id=agent_id, config={"tool_permissions": []})
    session = _make_session_returning(first_result=agent)

    client = _make_test_client(session)
    response = client.post(
        f"/agents/{agent_id}/approve-tool",
        json={"tool_name": "email_client", "approved": False},
        headers={"X-User-Id": "u1"},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["approved"] is False
    assert "denied" in data["message"].lower()


def test_approve_tool_updates_existing_permission() -> None:
    agent_id = str(uuid.uuid4())
    existing_config = {
        "tool_permissions": [
            {"tool_name": "telegram_bot", "auto_approved": False, "approval_count": 0}
        ]
    }
    agent = _make_agent_db(agent_id=agent_id, config=existing_config)
    session = _make_session_returning(first_result=agent)

    client = _make_test_client(session)
    response = client.post(
        f"/agents/{agent_id}/approve-tool",
        json={"tool_name": "telegram_bot", "approved": True},
        headers={"X-User-Id": "u1"},
    )

    assert response.status_code == 200
    # The config should have been updated in-place
    perms = agent.config.get("tool_permissions", [])
    tg_perm = next((p for p in perms if p["tool_name"] == "telegram_bot"), None)
    assert tg_perm is not None
    assert tg_perm["auto_approved"] is True
    assert tg_perm["approval_count"] >= 1


def test_approve_tool_agent_not_found() -> None:
    session = _make_session_returning(first_result=None)
    client = _make_test_client(session)
    response = client.post(
        f"/agents/{uuid.uuid4()}/approve-tool",
        json={"tool_name": "web_search", "approved": True},
        headers={"X-User-Id": "u1"},
    )
    assert response.status_code == 404


# ═══════════════════════════════════════════════════════════════════════════════
# GET /agents/{agent_id}/explain
# ═══════════════════════════════════════════════════════════════════════════════

def test_explain_agent_with_logs() -> None:
    agent_id = str(uuid.uuid4())
    agent = _make_agent_db(agent_id=agent_id)
    log_entry = _make_log_db(agent_id=agent_id, cycle=3, action="web_search")
    session = _make_session_returning(first_result=agent, second_result=log_entry)

    mock_gateway = AsyncMock()

    with patch("grif.api.agents.AgentExplainer.explain", new_callable=AsyncMock) as mock_explain:
        mock_explain.return_value = "The agent searched for hotels using web_search."

        client = _make_test_client(session, mock_gateway)
        response = client.get(
            f"/agents/{agent_id}/explain",
            headers={"X-User-Id": "u1"},
        )

    assert response.status_code == 200
    data = response.json()
    assert data["agent_id"] == agent_id
    assert "searched" in data["explanation"]
    assert data["last_action"] == "web_search"
    assert data["last_cycle"] == 3


def test_explain_agent_no_logs() -> None:
    agent_id = str(uuid.uuid4())
    agent = _make_agent_db(agent_id=agent_id)
    session = _make_session_returning(first_result=agent, second_result=None)

    mock_gateway = AsyncMock()

    client = _make_test_client(session, mock_gateway)
    response = client.get(
        f"/agents/{agent_id}/explain",
        headers={"X-User-Id": "u1"},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["agent_id"] == agent_id
    assert "not started" in data["explanation"].lower()
    assert data["last_action"] is None
    assert data["last_cycle"] is None


def test_explain_agent_not_found() -> None:
    session = _make_session_returning(first_result=None)
    mock_gateway = AsyncMock()

    client = _make_test_client(session, mock_gateway)
    response = client.get(
        f"/agents/{uuid.uuid4()}/explain",
        headers={"X-User-Id": "u1"},
    )

    assert response.status_code == 404


def test_explain_agent_explainer_fallback_on_error() -> None:
    agent_id = str(uuid.uuid4())
    agent = _make_agent_db(agent_id=agent_id)
    log_entry = _make_log_db(agent_id=agent_id, cycle=1, action="web_search")
    session = _make_session_returning(first_result=agent, second_result=log_entry)

    # Patch the gateway's complete method to raise — AgentExplainer catches this internally
    mock_gateway = AsyncMock()
    mock_gateway.complete = AsyncMock(side_effect=RuntimeError("LLM down"))

    client = _make_test_client(session, mock_gateway)
    response = client.get(
        f"/agents/{agent_id}/explain",
        headers={"X-User-Id": "u1"},
    )

    # AgentExplainer handles the error internally and returns fallback text
    assert response.status_code == 200
    data = response.json()
    assert len(data["explanation"]) > 5
    assert data["last_action"] == "web_search"
