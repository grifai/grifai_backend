"""
Step 5: Config Generator.
ONE LLM call (Sonnet) → generates Layer 3 (TaskContext) and full AgentConfig.
This is one of the 4 authorised LLM call points in GRIF.

For FORK: clones parent's Layer 1+2+4, regenerates Layer 3 only.
"""

import json
from typing import Any
from uuid import UUID

import structlog

from grif.blueprints.registry import Blueprint
from grif.llm.gateway import LLMGateway
from grif.models.agent_config import (
    AgentConfig,
    CommunicationConfig,
    ModelConfig,
    Schedule,
    WakeTrigger,
)
from grif.models.enums import TaskType, WakeTriggerType
from grif.models.intent import StructuredIntent
from grif.prompts.assembler import assemble_system_prompt, rebuild_layer_3

log = structlog.get_logger(__name__)


_CONFIG_GEN_SYSTEM = """You are an AI agent configuration engine for GRIF.
Your job: given a user task description and blueprint, generate a Layer 3 TaskContext prompt
and agent configuration parameters as JSON.

Respond ONLY with valid JSON:
{
  "layer_3_task_context": "<concise task-specific instructions for the agent, 100-200 words>",
  "tools": ["web_search", "..."],
  "model_id": "claude-sonnet-4-6 | claude-haiku-4-5-20251001 | gpt-4o | gpt-4o-mini",
  "temperature": 0.1-0.7,
  "max_tokens": 2048-8192,
  "wake_triggers": [],
  "schedule_cron": null,
  "schedule_timezone": "UTC"
}

Tool options available: web_search, telegram_bot, email_client
Wake trigger object format: {"trigger_type": "schedule|condition|event", "condition": "...", "value": null}
For MONITOR tasks: include a condition-based wake_trigger.
For OPERATE/REMIND tasks: include a schedule_cron (cron expression) and schedule_cron wake_trigger.
For simple SEARCH/RESEARCH: no wake_triggers, no schedule.
"""


class ConfigGenerator:
    """
    Step 5: Generates full AgentConfig including the 4-layer system prompt.
    Uses 1 Sonnet call to produce Layer 3 and config parameters.
    """

    def __init__(self, gateway: LLMGateway) -> None:
        self._gateway = gateway

    async def generate(
        self,
        intent: StructuredIntent,
        blueprint: Blueprint,
        user_id: str,
        user_persona: str = "",
        parent_agent_id: UUID | None = None,
        parent_config: AgentConfig | None = None,
    ) -> AgentConfig:
        """
        Generate a fresh AgentConfig (NEW or FORK).
        For FORK: clones parent layers 1+2+4, only regenerates Layer 3.
        """
        # Generate Layer 3 via LLM
        layer_3 = await self._generate_layer_3(intent, blueprint, user_id)

        # Assemble prompt layers
        if parent_config is not None:
            # FORK: inherit layers 1, 2, 4 from parent
            prompt_layers = rebuild_layer_3(parent_config.prompt_layers, layer_3)
            log.info("config_generator_fork", parent_id=str(parent_agent_id))
        else:
            # NEW: build all layers from scratch
            prompt_layers = assemble_system_prompt(
                task_context=layer_3,
                blueprint_id=blueprint.template,
                user_persona=user_persona,
            )

        # Build AgentConfig
        config = AgentConfig(
            user_id=user_id,
            task_type=intent.task_type,
            blueprint_id=blueprint.id,
            prompt_layers=prompt_layers,
            system_prompt=prompt_layers.assemble(),
            tools=blueprint.default_tools,
            model_config=ModelConfig(
                model_id=blueprint.default_model,
                temperature=0.3,
                max_tokens=4096,
            ),
            wake_triggers=self._build_wake_triggers(intent),
            schedule=self._build_schedule(intent),
            communication=CommunicationConfig(language=intent.language),
            parent_agent_id=parent_agent_id,
            metadata={
                "raw_input": intent.raw_input,
                "domain": intent.domain,
                "entities": intent.entities,
                "constraints": intent.constraints,
            },
        )

        log.info(
            "config_generated",
            task_type=intent.task_type,
            blueprint=blueprint.id,
            tools=config.tools,
            has_schedule=config.schedule is not None,
        )
        return config

    async def _generate_layer_3(
        self,
        intent: StructuredIntent,
        blueprint: Blueprint,
        user_id: str,
    ) -> str:
        """
        1 Sonnet LLM call → Layer 3 TaskContext + optional config overrides.
        Returns the layer_3_task_context string.
        """
        user_msg = (
            f"Task type: {intent.task_type}\n"
            f"Blueprint: {blueprint.id} ({blueprint.description})\n"
            f"User request: {intent.raw_input}\n"
            f"Entities: {json.dumps(intent.entities, ensure_ascii=False)}\n"
            f"Constraints: {json.dumps(intent.constraints, ensure_ascii=False)}\n"
            f"Urgency: {intent.urgency}, Complexity: {intent.complexity}\n"
            f"Output language: {intent.language}"
        )

        response = await self._gateway.complete_json(
            messages=[
                {"role": "system", "content": _CONFIG_GEN_SYSTEM},
                {"role": "user", "content": user_msg},
            ],
            purpose="config_generator",
            temperature=0.2,
            max_tokens=1024,
            user_id=user_id,
        )

        try:
            raw = json.loads(response.content.strip().strip("`").lstrip("json").strip())
            return raw.get("layer_3_task_context", self._fallback_layer_3(intent))
        except Exception as exc:
            log.warning("config_gen_json_parse_failed", error=str(exc))
            return self._fallback_layer_3(intent)

    def _fallback_layer_3(self, intent: StructuredIntent) -> str:
        """Deterministic fallback Layer 3 when LLM response can't be parsed."""
        parts = [
            f"Your task: {intent.task_type}.",
            f"User request: {intent.raw_input}",
        ]
        if intent.entities:
            parts.append(f"Key entities: {', '.join(f'{k}={v}' for k, v in intent.entities.items())}")
        if intent.constraints:
            parts.append(f"Constraints: {', '.join(f'{k}={v}' for k, v in intent.constraints.items())}")
        parts.append(f"Urgency: {intent.urgency}. Reply in: {intent.language}.")
        return " ".join(parts)

    def _build_wake_triggers(self, intent: StructuredIntent) -> list[WakeTrigger]:
        triggers: list[WakeTrigger] = []
        if intent.task_type == TaskType.MONITOR:
            threshold = intent.constraints.get("threshold")
            condition = intent.entities.get("condition", "price_below")
            triggers.append(WakeTrigger(
                trigger_type=WakeTriggerType.CONDITION,
                condition=condition,
                value=threshold,
            ))
        elif intent.task_type == TaskType.REMIND and intent.deadline:
            triggers.append(WakeTrigger(
                trigger_type=WakeTriggerType.SCHEDULE,
                condition=intent.deadline.isoformat(),
            ))
        return triggers

    def _build_schedule(self, intent: StructuredIntent) -> Schedule | None:
        if intent.task_type in (TaskType.OPERATE, TaskType.MONITOR):
            freq = intent.entities.get("frequency") or intent.constraints.get("frequency")
            cron = self._frequency_to_cron(freq)
            if cron:
                return Schedule(cron_expression=cron, timezone="UTC")
        return None

    @staticmethod
    def _frequency_to_cron(frequency: Any) -> str | None:
        """Convert natural frequency descriptions to cron expressions."""
        if not frequency:
            return None
        freq_str = str(frequency).lower()
        mapping = {
            "daily": "0 9 * * *",
            "ежедневно": "0 9 * * *",
            "каждый день": "0 9 * * *",
            "weekly": "0 9 * * 1",
            "еженедельно": "0 9 * * 1",
            "каждую неделю": "0 9 * * 1",
            "каждую пятницу": "0 18 * * 5",
            "hourly": "0 * * * *",
            "каждый час": "0 * * * *",
            "3 times a week": "0 9 * * 1,3,5",
            "3 раза в неделю": "0 9 * * 1,3,5",
        }
        for key, cron in mapping.items():
            if key in freq_str:
                return cron
        return None
