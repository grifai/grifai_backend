from enum import StrEnum


class TaskType(StrEnum):
    SEARCH = "search"
    MONITOR = "monitor"
    RESEARCH = "research"
    COACH = "coach"
    COMPARE = "compare"
    EXECUTE = "execute"
    REMIND = "remind"
    GENERATE = "generate"
    OPERATE = "operate"


class AgentState(StrEnum):
    EMBRYO = "embryo"        # Being configured, not yet spawned
    ACTIVE = "active"        # Running (ReAct loop)
    SLEEPING = "sleeping"    # Checkpointed, waiting for wake trigger / event
    RECURRING = "recurring"  # Scheduled to run repeatedly
    ARCHIVED = "archived"    # Permanently stopped, memory retained per TTL


class ToolCategory(StrEnum):
    READ = "read"                        # search, fetch, analyze — no confirmation
    WRITE_SAFE = "write_safe"            # draft, save, create_file — no confirmation
    WRITE_PUBLIC = "write_public"        # post_telegram, send_email — show draft first
    WRITE_IRREVERSIBLE = "write_irreversible"  # buy, book, delete — always confirm


class Complexity(StrEnum):
    SIMPLE = "simple"
    MULTI_STEP = "multi_step"


class Urgency(StrEnum):
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"


class ClarificationMode(StrEnum):
    QUICK_CONFIRM = "quick_confirm"           # Simple tasks, 1 confirmation
    STRUCTURED_INTERVIEW = "structured_interview"  # Complex, 3-5 questions at once
    PROGRESSIVE = "progressive"               # Proactive agents


class RouterDecision(StrEnum):
    NEW = "new"          # Create a fresh agent
    EXISTING = "existing"  # Wake / reuse existing agent (≥80% Jaccard)
    FORK = "fork"        # Clone existing agent, regenerate Layer 3 only
    SKIP = "skip"        # Already handled / deduplicate


class ReactDecision(StrEnum):
    CONTINUE = "continue"    # Keep going — more cycles needed
    REPORT = "report"        # Finished — return result to user
    WAIT = "wait"            # Blocked — need user input / external event
    SLEEP = "sleep"          # Task done for now — checkpoint and sleep
    ESCALATE = "escalate"    # Cannot proceed — escalate to Orchestrator


class PlanPattern(StrEnum):
    PIPELINE = "pipeline"                  # A→B→C
    PARALLEL_MERGE = "parallel_merge"      # A∥B→C
    PIPELINE_REVIEW = "pipeline_review"    # A→B⇄C (with Critic)
    HYPOTHESIS_TESTING = "hypothesis_testing"  # A∥A'→select best


class PhaseMode(StrEnum):
    ONE_SHOT = "one_shot"    # Runs once at setup
    RECURRING = "recurring"  # Runs on schedule indefinitely


class MemoryType(StrEnum):
    FACT = "fact"
    DECISION = "decision"
    PREFERENCE = "preference"
    PRODUCTION = "production"


class WakeTriggerType(StrEnum):
    SCHEDULE = "schedule"          # Cron / time-based
    EVENT = "event"                # External event (webhook, message)
    CONDITION = "condition"        # Watched value crosses threshold
    MANUAL = "manual"              # User explicitly wakes the agent


class SignalType(StrEnum):
    TEXT = "text"
    VOICE = "voice"
    EVENT = "event"
    PATTERN = "pattern"            # Performance signal from external metrics


class PerformanceChannel(StrEnum):
    TELEGRAM = "telegram"
    EMAIL = "email"
    SALES_CRM = "sales_crm"
    CUSTOM_API = "custom_api"
