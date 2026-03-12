from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from grif.models.enums import (
    ClarificationMode,
    Complexity,
    TaskType,
    Urgency,
)


class StructuredIntent(BaseModel):
    """Normalised, classified intent extracted from user input."""

    task_type: TaskType
    entities: dict[str, Any] = Field(
        default_factory=dict,
        description="Named entities extracted from input: topic, location, budget, etc.",
    )
    constraints: dict[str, Any] = Field(
        default_factory=dict,
        description="Hard constraints: max_price, deadline, language, etc.",
    )
    complexity: Complexity = Complexity.SIMPLE
    urgency: Urgency = Urgency.NORMAL
    deadline: datetime | None = None
    domain: str | None = Field(
        default=None,
        description="High-level domain: travel, research, content, sales, etc.",
    )
    raw_input: str = Field(description="Original user message, verbatim.")
    language: str = Field(default="ru", description="Detected language (BCP-47).")


class ClarificationQuestion(BaseModel):
    field_name: str
    question: str
    options: list[str] | None = None  # If a multiple-choice hint is helpful


class ClarificationRequest(BaseModel):
    """Output of the Clarification Phase: questions to ask the user."""

    mode: ClarificationMode
    questions: list[ClarificationQuestion]
    context_summary: str = Field(
        description="Short summary of what is already understood.",
    )


class ClarificationAnswer(BaseModel):
    """User's answers to clarification questions."""

    field_name: str
    value: Any


class ClassifiedIntent(BaseModel):
    """
    Full output of the pipeline Steps 2 + 2.5.
    If clarification_needed is True, the pipeline pauses and sends
    ClarificationRequest to the user before proceeding.
    """

    structured_intent: StructuredIntent
    clarification_needed: bool = False
    clarification_request: ClarificationRequest | None = None
    missing_fields: list[str] = Field(default_factory=list)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
