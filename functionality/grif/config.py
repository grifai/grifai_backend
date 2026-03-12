from functools import lru_cache
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # App
    app_env: Literal["development", "staging", "production"] = "development"
    debug: bool = False
    secret_key: str = "change-me-in-production"
    app_version: str = "0.1.0"

    # Database
    database_url: str = Field(
        default="postgresql+asyncpg://grif:grif_secret@localhost:5432/grif_db"
    )
    database_sync_url: str = Field(
        default="postgresql+psycopg2://grif:grif_secret@localhost:5432/grif_db"
    )
    db_pool_size: int = 10
    db_max_overflow: int = 20

    # Redis
    redis_url: str = "redis://localhost:6379/0"
    celery_broker_url: str = "redis://localhost:6379/1"
    celery_result_backend: str = "redis://localhost:6379/2"

    # LLM API keys
    anthropic_api_key: str = ""
    openai_api_key: str = ""

    # LLM model aliases
    llm_classifier_model: str = "claude-haiku-4-5-20251001"   # Intent + Self-Eval
    llm_generator_model: str = "claude-sonnet-4-6"            # Config + Conflict
    llm_react_default_model: str = "claude-sonnet-4-6"        # ReAct reasoning
    llm_summarizer_model: str = "gpt-4o-mini"                 # Summarisation

    # Fallback models
    llm_classifier_fallback: str = "gpt-4o-mini"
    llm_generator_fallback: str = "gpt-4o"
    llm_summarizer_fallback: str = "claude-haiku-4-5-20251001"

    # LLM retry settings
    llm_max_retries: int = 3
    llm_retry_base_delay: float = 1.0   # seconds
    llm_timeout: float = 60.0           # seconds

    # Tools
    tavily_api_key: str = ""
    telegram_bot_token: str = ""

    # Rate limiting defaults (configurable per plan)
    max_active_agents: int = 10
    max_tokens_per_day: int = 500_000
    max_recurring_agents: int = 5

    # Progress reporter
    progress_min_interval_seconds: int = 300  # 5 min

    # ReAct loop
    react_max_cycles: int = 20

    # Clarification
    clarification_structured_max_questions: int = 5

    # Router similarity threshold for EXISTING/FORK
    router_jaccard_threshold: float = 0.80

    # Agent sleep / archive TTLs (days)
    agent_log_retention_days: int = 30
    production_memory_retention_days: int = 365
    sleeping_agent_archive_days: int = 90

    # Trust escalation: Write-public goes auto after N approvals
    trust_escalation_approvals: int = 5

    @field_validator("anthropic_api_key", "openai_api_key", mode="before")
    @classmethod
    def _warn_empty_keys(cls, v: str, info: object) -> str:
        # Keys are optional at config load time — the gateway will raise
        # a descriptive error if a missing key is actually needed.
        return v or ""


@lru_cache
def get_settings() -> Settings:
    return Settings()
