"""
Step 2.5: Clarification Phase.
Gap Analysis is DETERMINISTIC (no LLM).
Question text is generated with ONE Haiku call only if questions are needed.
Three modes: Quick Confirm, Structured Interview, Progressive.
"""

import json
from typing import Any

import structlog

from grif.llm.gateway import LLMGateway
from grif.models.enums import ClarificationMode, Complexity, TaskType, Urgency
from grif.models.intent import (
    ClassifiedIntent,
    ClarificationQuestion,
    ClarificationRequest,
    StructuredIntent,
)

log = structlog.get_logger(__name__)

# ─── Required fields per TaskType ─────────────────────────────────────────────

_REQUIRED_FIELDS: dict[TaskType, list[str]] = {
    TaskType.SEARCH: ["topic"],
    TaskType.MONITOR: ["item", "condition", "threshold"],
    TaskType.RESEARCH: ["topic"],
    TaskType.COACH: ["event"],
    TaskType.COMPARE: ["items"],
    TaskType.EXECUTE: ["action", "target"],
    TaskType.REMIND: ["what", "when"],
    TaskType.GENERATE: ["content_type", "topic"],
    TaskType.OPERATE: ["operation_type", "frequency"],
}

# Fields that can be inferred from UserProfile (no need to ask)
_INFER_FROM_PROFILE: set[str] = {
    "language", "timezone", "currency", "notification_channel"
}


class ClarificationPhase:
    """
    Implements Gap Analysis + optional question generation.

    Gap Analysis (deterministic):
    1. Check required fields for the task_type.
    2. Check if any required field is missing from entities/constraints.
    3. Subtract fields that can be inferred from User Profile.

    Question generation (Haiku, 1 call, only if gaps exist):
    - Quick Confirm: 1-2 questions for simple tasks.
    - Structured Interview: 3-5 questions at once for complex tasks.
    - Progressive: single most important question (proactive agents).
    """

    def __init__(self, gateway: LLMGateway) -> None:
        self._gateway = gateway

    async def process(
        self,
        intent: StructuredIntent,
        user_profile: dict[str, Any] | None = None,
    ) -> ClassifiedIntent:
        """
        Analyse gaps and produce ClassifiedIntent.
        If no gaps: returns immediately (no LLM call).
        If gaps: makes 1 Haiku call to generate friendly question text.
        """
        missing = self._gap_analysis(intent, user_profile)

        if not missing:
            return ClassifiedIntent(
                structured_intent=intent,
                clarification_needed=False,
            )

        mode = self._choose_mode(intent)
        # Limit questions per mode, but keep full missing_fields list
        if mode == ClarificationMode.QUICK_CONFIRM:
            fields_to_ask = missing[:2]
        elif mode == ClarificationMode.STRUCTURED_INTERVIEW:
            fields_to_ask = missing[:5]
        else:  # Progressive
            fields_to_ask = missing[:1]

        questions = await self._generate_questions(intent, fields_to_ask, mode)
        request = ClarificationRequest(
            mode=mode,
            questions=questions,
            context_summary=self._context_summary(intent),
        )

        return ClassifiedIntent(
            structured_intent=intent,
            clarification_needed=True,
            clarification_request=request,
            missing_fields=missing,  # Full list, not truncated
        )

    # ── Gap Analysis (deterministic) ──────────────────────────────────────────

    def _gap_analysis(
        self,
        intent: StructuredIntent,
        user_profile: dict[str, Any] | None,
    ) -> list[str]:
        """
        Returns list of missing required field names.
        Subtracts fields already in entities/constraints or inferable from profile.
        """
        required = _REQUIRED_FIELDS.get(intent.task_type, [])
        known = set(intent.entities.keys()) | set(intent.constraints.keys())

        # Fields available from user profile
        profile_fields: set[str] = set()
        if user_profile:
            profile_fields = _INFER_FROM_PROFILE & set(user_profile.keys())

        missing = [
            f for f in required
            if f not in known and f not in profile_fields
        ]
        return missing

    # ── Mode selection (deterministic) ────────────────────────────────────────

    def _choose_mode(self, intent: StructuredIntent) -> ClarificationMode:
        if intent.complexity == Complexity.MULTI_STEP:
            return ClarificationMode.STRUCTURED_INTERVIEW
        if intent.task_type == TaskType.OPERATE:
            return ClarificationMode.PROGRESSIVE
        return ClarificationMode.QUICK_CONFIRM

    # ── Question generation (Haiku, 1 call) ───────────────────────────────────

    async def _generate_questions(
        self,
        intent: StructuredIntent,
        missing_fields: list[str],
        mode: ClarificationMode,
    ) -> list[ClarificationQuestion]:
        """
        Generate friendly question text for each missing field.
        Uses 1 Haiku call with structured JSON output.
        """
        system = (
            "You are a helpful assistant gathering information before starting a task. "
            "Generate concise, friendly questions to ask the user. "
            "Respond ONLY with a JSON array: "
            '[{"field_name": "...", "question": "...", "options": [...] or null}]'
        )
        user_msg = (
            f"Task type: {intent.task_type}\n"
            f"Already known: {json.dumps({**intent.entities, **intent.constraints}, ensure_ascii=False)}\n"
            f"Missing fields to ask about: {', '.join(missing_fields)}\n"
            f"Mode: {mode} — "
            + (
                "ask all at once" if mode == ClarificationMode.STRUCTURED_INTERVIEW
                else "ask the most important one"
            )
            + "\nGenerate questions in the language: "
            + intent.language
        )

        try:
            response = await self._gateway.complete_json(
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user_msg},
                ],
                purpose="clarification",
                temperature=0.3,
                max_tokens=512,
                user_id=None,
            )
            raw = json.loads(response.content.strip().strip("`").lstrip("json").strip())
            if not isinstance(raw, list):
                raw = [raw]

            return [
                ClarificationQuestion(
                    field_name=q.get("field_name", missing_fields[i]),
                    question=q.get("question", f"Please provide: {missing_fields[i]}"),
                    options=q.get("options"),
                )
                for i, q in enumerate(raw)
            ]
        except Exception as exc:
            log.warning("clarification_question_gen_failed", error=str(exc))
            # Deterministic fallback
            return [
                ClarificationQuestion(
                    field_name=f,
                    question=f"Please provide: {f}",
                )
                for f in missing_fields
            ]

    def _context_summary(self, intent: StructuredIntent) -> str:
        parts = [f"Task: {intent.task_type}"]
        if intent.entities:
            kv = ", ".join(f"{k}={v}" for k, v in list(intent.entities.items())[:3])
            parts.append(f"Known: {kv}")
        return ". ".join(parts)

    # ── Apply answers (deterministic) ─────────────────────────────────────────

    def apply_answers(
        self,
        classified: ClassifiedIntent,
        answers: list[dict[str, Any]],
    ) -> ClassifiedIntent:
        """
        Merge clarification answers into the intent's entities/constraints.
        Returns updated ClassifiedIntent with clarification_needed=False.
        """
        intent = classified.structured_intent
        updated_entities = dict(intent.entities)
        updated_constraints = dict(intent.constraints)

        for answer in answers:
            field = answer.get("field_name", "")
            value = answer.get("value")
            if field and value is not None:
                # Put in entities (or constraints if it looks like a limit)
                if any(k in field for k in ["max", "min", "limit", "budget", "threshold"]):
                    updated_constraints[field] = value
                else:
                    updated_entities[field] = value

        updated_intent = StructuredIntent(
            task_type=intent.task_type,
            entities=updated_entities,
            constraints=updated_constraints,
            complexity=intent.complexity,
            urgency=intent.urgency,
            deadline=intent.deadline,
            domain=intent.domain,
            raw_input=intent.raw_input,
            language=intent.language,
        )

        return ClassifiedIntent(
            structured_intent=updated_intent,
            clarification_needed=False,
            missing_fields=[],
        )
