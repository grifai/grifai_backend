"""
Step 2: Intent Classifier.
ONE LLM call (Haiku) → extracts TaskType, entities, constraints, complexity, urgency.
This is one of the 4 authorised LLM call points in GRIF.
"""

import json
import re
from datetime import datetime
from typing import Any

import structlog

from grif.llm.gateway import LLMGateway
from grif.models.enums import Complexity, TaskType, Urgency
from grif.models.intent import StructuredIntent
from grif.pipeline.signal_parser import ParsedSignal
from grif.prompts.layers import load_classifier_prompt

log = structlog.get_logger(__name__)

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def _extract_json(text: str) -> dict[str, Any]:
    """Extract JSON object from LLM response (handles markdown code fences)."""
    # Remove ```json ... ``` fences if present
    clean = re.sub(r"```(?:json)?\s*", "", text).strip().rstrip("`").strip()
    match = _JSON_RE.search(clean)
    if not match:
        raise ValueError(f"No JSON object found in LLM response: {text[:200]}")
    return json.loads(match.group())


def _parse_deadline(value: Any) -> datetime | None:
    if not value or value == "null":
        return None
    try:
        return datetime.fromisoformat(str(value))
    except (ValueError, TypeError):
        return None


def _safe_task_type(value: str) -> TaskType:
    try:
        return TaskType(value.lower())
    except ValueError:
        log.warning("unknown_task_type", value=value)
        return TaskType.SEARCH


def _safe_complexity(value: str) -> Complexity:
    try:
        return Complexity(value.lower())
    except ValueError:
        return Complexity.SIMPLE


def _safe_urgency(value: str) -> Urgency:
    try:
        return Urgency(value.lower())
    except ValueError:
        return Urgency.NORMAL


class IntentClassifier:
    """
    Classifies user intent using a single Haiku LLM call.
    Output: StructuredIntent with all fields filled.
    """

    def __init__(self, gateway: LLMGateway) -> None:
        self._gateway = gateway
        self._system_prompt = load_classifier_prompt()

    async def classify(self, signal: ParsedSignal) -> StructuredIntent:
        """
        Step 2: classify a ParsedSignal into StructuredIntent.
        Makes exactly 1 LLM call (Haiku).
        """
        log.info("intent_classify_start", user_id=signal.user_id, text_len=len(signal.text))

        response = await self._gateway.complete_json(
            messages=[
                {"role": "system", "content": self._system_prompt},
                {"role": "user", "content": signal.text},
            ],
            purpose="intent_classifier",
            temperature=0.1,
            max_tokens=512,
            user_id=signal.user_id,
        )

        raw = _extract_json(response.content)
        intent = self._build_intent(raw, signal)

        log.info(
            "intent_classified",
            task_type=intent.task_type,
            complexity=intent.complexity,
            urgency=intent.urgency,
            domain=intent.domain,
            tokens=response.total_tokens,
        )
        return intent

    def _build_intent(
        self, raw: dict[str, Any], signal: ParsedSignal
    ) -> StructuredIntent:
        return StructuredIntent(
            task_type=_safe_task_type(raw.get("task_type", "search")),
            entities=raw.get("entities") or {},
            constraints=raw.get("constraints") or {},
            complexity=_safe_complexity(raw.get("complexity", "simple")),
            urgency=_safe_urgency(raw.get("urgency", "normal")),
            deadline=_parse_deadline(raw.get("deadline")),
            domain=raw.get("domain") or "general",
            raw_input=signal.text,
            language=raw.get("language") or "ru",
        )
