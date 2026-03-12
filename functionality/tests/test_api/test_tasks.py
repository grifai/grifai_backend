"""Tests for api/tasks.py — POST /tasks, POST /clarification/{task_id}/answer."""
import uuid
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from grif.api.deps import get_gateway, get_session
from grif.main import create_app
from grif.models.enums import Complexity, TaskType, Urgency


# ─── Fixtures ─────────────────────────────────────────────────────────────────

def _make_task_db(task_id: str, status: str = "pending", pending_clarification=None):
    task = MagicMock()
    task.id = uuid.UUID(task_id)
    task.status = status
    task.raw_input = "Find hotels in Paris"
    task.pending_clarification = pending_clarification
    task.classified_intent = None
    task.clarification_answers = None
    return task


def _make_agent_db(agent_id: str):
    agent = MagicMock()
    agent.id = uuid.UUID(agent_id)
    agent.user_id = "u1"
    agent.state = "active"
    agent.task_type = "search"
    agent.blueprint_id = "generic_worker"
    agent.config = {"task_type": "search", "blueprint_id": "generic_worker", "tools": [], "tool_permissions": [], "prompt_layers": {"layer_1_core_identity": "Core", "layer_2_role_template": "Role", "layer_3_task_context": "Task", "layer_4_user_persona": ""}, "model_config": {"model_id": "claude-sonnet-4-6", "temperature": 0.3, "max_tokens": 4096}, "user_id": "u1", "id": agent_id}
    return agent


def _make_intent():
    from grif.models.intent import StructuredIntent
    return StructuredIntent(
        task_type=TaskType.SEARCH,
        entities={"topic": "hotels"},
        constraints={},
        complexity=Complexity.SIMPLE,
        urgency=Urgency.NORMAL,
        domain="travel",
        raw_input="Find hotels in Paris",
        language="en",
    )


def _make_classified_intent(clarification_needed: bool = False):
    from grif.models.intent import ClassifiedIntent, ClarificationRequest, ClarificationQuestion
    from grif.models.enums import ClarificationMode
    intent = _make_intent()
    if clarification_needed:
        request = ClarificationRequest(
            mode=ClarificationMode.QUICK_CONFIRM,
            questions=[
                ClarificationQuestion(field_name="budget", question="What is your budget?")
            ],
            context_summary="Task: search",
        )
        return ClassifiedIntent(
            structured_intent=intent,
            clarification_needed=True,
            clarification_request=request,
            missing_fields=["budget"],
        )
    return ClassifiedIntent(structured_intent=intent, clarification_needed=False)


def _make_mock_session(task=None, agent=None):
    session = AsyncMock()
    session.add = MagicMock()
    session.flush = AsyncMock()

    def _execute_side_effect(query):
        result = MagicMock()
        # Try to figure out which model is being queried
        # Return task or agent based on what's set
        result.scalar_one_or_none.return_value = task if task is not None else agent
        return result

    session.execute = AsyncMock(side_effect=_execute_side_effect)
    return session


def _make_mock_gateway():
    gateway = AsyncMock()
    # Intent classifier response
    response = MagicMock()
    response.content = '{"task_type": "search", "entities": {"topic": "hotels"}, "constraints": {}, "complexity": "simple", "urgency": "normal", "deadline": null, "domain": "travel", "language": "en"}'
    response.total_tokens = 80
    # Config generator response
    config_response = MagicMock()
    config_response.content = '{"layer_3_task_context": "Search for hotels in Paris under 150 euros.", "tools": ["web_search"], "model_id": "claude-sonnet-4-6", "temperature": 0.3, "max_tokens": 4096}'
    gateway.complete_json = AsyncMock(return_value=response)
    return gateway


# ─── App factory with dependency overrides ────────────────────────────────────

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
# POST /tasks
# ═══════════════════════════════════════════════════════════════════════════════

def test_submit_task_returns_201_with_running_status() -> None:
    task_id = str(uuid.uuid4())
    agent_id = str(uuid.uuid4())
    mock_session = _make_mock_session()
    mock_gateway = _make_mock_gateway()

    with (
        patch("grif.api.tasks.IntentClassifier.classify", new_callable=AsyncMock) as mock_classify,
        patch("grif.api.tasks.ClarificationPhase.process", new_callable=AsyncMock) as mock_clarify,
        patch("grif.api.tasks._run_planning_phase", new_callable=AsyncMock) as mock_plan,
    ):
        mock_classify.return_value = _make_intent()
        mock_clarify.return_value = _make_classified_intent(clarification_needed=False)
        mock_plan.return_value = ([agent_id], {"estimated_tokens": 1000, "estimated_cost_usd": 0.015, "budget_level": "minimum"})

        client = _make_test_client(mock_session, mock_gateway)
        response = client.post(
            "/tasks",
            json={"text": "Find cheap hotels in Paris"},
            headers={"X-User-Id": "u1"},
        )

    assert response.status_code == 201
    data = response.json()
    assert data["status"] == "running"
    assert len(data["agent_ids"]) == 1
    assert data["estimated_cost"]["budget_level"] == "minimum"


def test_submit_task_clarification_needed() -> None:
    mock_session = _make_mock_session()
    mock_gateway = _make_mock_gateway()

    with (
        patch("grif.api.tasks.IntentClassifier.classify", new_callable=AsyncMock) as mock_classify,
        patch("grif.api.tasks.ClarificationPhase.process", new_callable=AsyncMock) as mock_clarify,
    ):
        mock_classify.return_value = _make_intent()
        mock_clarify.return_value = _make_classified_intent(clarification_needed=True)

        client = _make_test_client(mock_session, mock_gateway)
        response = client.post(
            "/tasks",
            json={"text": "Monitor something"},
            headers={"X-User-Id": "u1"},
        )

    assert response.status_code == 201
    data = response.json()
    assert data["status"] == "clarifying"
    assert data["clarification_questions"] is not None
    assert len(data["clarification_questions"]) >= 1
    assert data["clarification_questions"][0]["field_name"] == "budget"


def test_submit_task_uses_header_user_id() -> None:
    agent_id = str(uuid.uuid4())
    mock_session = _make_mock_session()
    mock_gateway = _make_mock_gateway()

    captured = {}

    async def _mock_plan(task_id, intent, user_id, session, gateway):
        captured["user_id"] = user_id
        return [agent_id], {"estimated_tokens": 100, "estimated_cost_usd": 0.001, "budget_level": "minimum"}

    with (
        patch("grif.api.tasks.IntentClassifier.classify", new_callable=AsyncMock) as mock_classify,
        patch("grif.api.tasks.ClarificationPhase.process", new_callable=AsyncMock) as mock_clarify,
        patch("grif.api.tasks._run_planning_phase", side_effect=_mock_plan),
    ):
        mock_classify.return_value = _make_intent()
        mock_clarify.return_value = _make_classified_intent(clarification_needed=False)

        client = _make_test_client(mock_session, mock_gateway)
        client.post(
            "/tasks",
            json={"text": "Search for flights"},
            headers={"X-User-Id": "custom-user-42"},
        )

    assert captured.get("user_id") == "custom-user-42"


def test_submit_task_defaults_to_anonymous_user_id() -> None:
    agent_id = str(uuid.uuid4())
    mock_session = _make_mock_session()
    mock_gateway = _make_mock_gateway()

    captured = {}

    async def _mock_plan(task_id, intent, user_id, session, gateway):
        captured["user_id"] = user_id
        return [agent_id], {"estimated_tokens": 100, "estimated_cost_usd": 0.001, "budget_level": "minimum"}

    with (
        patch("grif.api.tasks.IntentClassifier.classify", new_callable=AsyncMock) as mock_classify,
        patch("grif.api.tasks.ClarificationPhase.process", new_callable=AsyncMock) as mock_clarify,
        patch("grif.api.tasks._run_planning_phase", side_effect=_mock_plan),
    ):
        mock_classify.return_value = _make_intent()
        mock_clarify.return_value = _make_classified_intent(clarification_needed=False)

        client = _make_test_client(mock_session, mock_gateway)
        client.post("/tasks", json={"text": "Find me a restaurant"})

    assert captured.get("user_id") == "anonymous"


def test_submit_task_intent_classifier_failure_returns_503() -> None:
    mock_session = _make_mock_session()
    mock_gateway = _make_mock_gateway()

    with patch(
        "grif.api.tasks.IntentClassifier.classify",
        new_callable=AsyncMock,
        side_effect=RuntimeError("LLM unavailable"),
    ):
        client = _make_test_client(mock_session, mock_gateway)
        response = client.post(
            "/tasks",
            json={"text": "Book a flight"},
            headers={"X-User-Id": "u1"},
        )

    assert response.status_code == 503
    assert "Intent classification failed" in response.json()["detail"]


def test_submit_task_returns_task_id() -> None:
    agent_id = str(uuid.uuid4())
    mock_session = _make_mock_session()
    mock_gateway = _make_mock_gateway()

    with (
        patch("grif.api.tasks.IntentClassifier.classify", new_callable=AsyncMock) as mock_classify,
        patch("grif.api.tasks.ClarificationPhase.process", new_callable=AsyncMock) as mock_clarify,
        patch("grif.api.tasks._run_planning_phase", new_callable=AsyncMock) as mock_plan,
    ):
        mock_classify.return_value = _make_intent()
        mock_clarify.return_value = _make_classified_intent(clarification_needed=False)
        mock_plan.return_value = ([agent_id], {"estimated_tokens": 500, "estimated_cost_usd": 0.01, "budget_level": "minimum"})

        client = _make_test_client(mock_session, mock_gateway)
        response = client.post(
            "/tasks",
            json={"text": "Research AI trends"},
            headers={"X-User-Id": "u1"},
        )

    assert response.status_code == 201
    data = response.json()
    # task_id must be a valid UUID
    uuid.UUID(data["task_id"])


def test_submit_task_empty_text_returns_422() -> None:
    client = _make_test_client()
    response = client.post("/tasks", json={"text": ""})
    assert response.status_code == 422


def test_submit_task_planning_failure_returns_503() -> None:
    mock_session = _make_mock_session()
    mock_gateway = _make_mock_gateway()

    with (
        patch("grif.api.tasks.IntentClassifier.classify", new_callable=AsyncMock) as mock_classify,
        patch("grif.api.tasks.ClarificationPhase.process", new_callable=AsyncMock) as mock_clarify,
        patch(
            "grif.api.tasks._run_planning_phase",
            new_callable=AsyncMock,
            side_effect=RuntimeError("DB down"),
        ),
    ):
        mock_classify.return_value = _make_intent()
        mock_clarify.return_value = _make_classified_intent(clarification_needed=False)

        client = _make_test_client(mock_session, mock_gateway)
        response = client.post(
            "/tasks",
            json={"text": "Do something complex"},
            headers={"X-User-Id": "u1"},
        )

    assert response.status_code == 503
    assert "Planning failed" in response.json()["detail"]


# ═══════════════════════════════════════════════════════════════════════════════
# POST /clarification/{task_id}/answer
# ═══════════════════════════════════════════════════════════════════════════════

def _make_pending_clarification_json():
    from grif.models.intent import ClassifiedIntent, ClarificationRequest, ClarificationQuestion
    from grif.models.enums import ClarificationMode
    intent = _make_intent()
    request = ClarificationRequest(
        mode=ClarificationMode.QUICK_CONFIRM,
        questions=[
            ClarificationQuestion(field_name="budget", question="What is your budget?")
        ],
        context_summary="Task: search",
    )
    classified = ClassifiedIntent(
        structured_intent=intent,
        clarification_needed=True,
        clarification_request=request,
        missing_fields=["budget"],
    )
    return classified.model_dump(mode="json")


def test_answer_clarification_continues_pipeline() -> None:
    task_id = str(uuid.uuid4())
    agent_id = str(uuid.uuid4())
    task = _make_task_db(task_id, status="clarifying", pending_clarification=_make_pending_clarification_json())
    mock_session = _make_mock_session(task=task)
    mock_gateway = _make_mock_gateway()

    with patch("grif.api.tasks._run_planning_phase", new_callable=AsyncMock) as mock_plan:
        mock_plan.return_value = ([agent_id], {"estimated_tokens": 500, "estimated_cost_usd": 0.01, "budget_level": "minimum"})

        client = _make_test_client(mock_session, mock_gateway)
        response = client.post(
            f"/tasks/clarification/{task_id}/answer",
            json={"answers": [{"field_name": "budget", "value": "150 euros"}]},
            headers={"X-User-Id": "u1"},
        )

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "running"
    assert len(data["agent_ids"]) == 1


def test_answer_clarification_task_not_found() -> None:
    mock_session = _make_mock_session(task=None)
    mock_gateway = _make_mock_gateway()

    client = _make_test_client(mock_session, mock_gateway)
    response = client.post(
        f"/tasks/clarification/{uuid.uuid4()}/answer",
        json={"answers": []},
        headers={"X-User-Id": "u1"},
    )

    assert response.status_code == 404


def test_answer_clarification_wrong_status() -> None:
    task_id = str(uuid.uuid4())
    task = _make_task_db(task_id, status="running")  # Not "clarifying"
    mock_session = _make_mock_session(task=task)
    mock_gateway = _make_mock_gateway()

    client = _make_test_client(mock_session, mock_gateway)
    response = client.post(
        f"/tasks/clarification/{task_id}/answer",
        json={"answers": [{"field_name": "budget", "value": "100"}]},
        headers={"X-User-Id": "u1"},
    )

    assert response.status_code == 400
    assert "running" in response.json()["detail"]


def test_answer_clarification_no_pending_data() -> None:
    task_id = str(uuid.uuid4())
    task = _make_task_db(task_id, status="clarifying", pending_clarification=None)
    mock_session = _make_mock_session(task=task)
    mock_gateway = _make_mock_gateway()

    client = _make_test_client(mock_session, mock_gateway)
    response = client.post(
        f"/tasks/clarification/{task_id}/answer",
        json={"answers": [{"field_name": "budget", "value": "100"}]},
        headers={"X-User-Id": "u1"},
    )

    assert response.status_code == 400
    assert "clarification" in response.json()["detail"].lower()


def test_answer_clarification_merges_answers_into_entities() -> None:
    """Answers should be merged into intent entities and pipeline called."""
    task_id = str(uuid.uuid4())
    agent_id = str(uuid.uuid4())
    task = _make_task_db(task_id, status="clarifying", pending_clarification=_make_pending_clarification_json())
    mock_session = _make_mock_session(task=task)
    mock_gateway = _make_mock_gateway()

    captured = {}

    async def _mock_plan(task_id, intent, user_id, session, gateway):
        captured["entities"] = dict(intent.entities)
        return [agent_id], {"estimated_tokens": 100, "estimated_cost_usd": 0.001, "budget_level": "minimum"}

    with patch("grif.api.tasks._run_planning_phase", side_effect=_mock_plan):
        client = _make_test_client(mock_session, mock_gateway)
        client.post(
            f"/tasks/clarification/{task_id}/answer",
            json={"answers": [{"field_name": "topic", "value": "Paris luxury hotels"}]},
            headers={"X-User-Id": "u1"},
        )

    assert "topic" in captured.get("entities", {})
    assert captured["entities"]["topic"] == "Paris luxury hotels"
