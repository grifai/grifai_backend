"""
Celery application — configured with Redis broker + RedBeat scheduler.

Workers:
  - default queue: agent task execution
  - beat: RedBeat for cron scheduling of recurring agents

Usage:
  celery -A grif.tasks.celery_app worker --loglevel=info
  celery -A grif.tasks.celery_app beat --scheduler redbeat.RedBeatScheduler
"""

from __future__ import annotations

from celery import Celery
from kombu import Queue

from grif.config import get_settings

settings = get_settings()

# ── App ───────────────────────────────────────────────────────────────────────

celery_app = Celery(
    "grif",
    broker=settings.redis_url,
    backend=settings.redis_url,
    include=["grif.tasks.agent_tasks"],
)

# ── Config ────────────────────────────────────────────────────────────────────

celery_app.conf.update(
    # Serialization
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    # Timezone
    timezone="UTC",
    enable_utc=True,
    # Task routing
    task_default_queue="default",
    task_queues=(
        Queue("default"),
        Queue("wake"),      # Agent wake tasks
        Queue("recurring"), # Recurring agent cycles
        Queue("cleanup"),   # Maintenance tasks
    ),
    task_routes={
        "grif.tasks.agent_tasks.wake_agent_task": {"queue": "wake"},
        "grif.tasks.agent_tasks.run_recurring_cycle": {"queue": "recurring"},
        "grif.tasks.agent_tasks.cleanup_archived_agents": {"queue": "cleanup"},
        "grif.tasks.agent_tasks.process_wake_queue": {"queue": "wake"},
    },
    # Retry policy
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    task_max_retries=3,
    # RedBeat scheduler
    redbeat_redis_url=settings.redis_url,
    redbeat_key_prefix="grif:beat:",
    beat_scheduler="redbeat.RedBeatScheduler",
    # Beat schedule: scan wake queue every 60 seconds
    beat_schedule={
        "process-wake-queue": {
            "task": "grif.tasks.agent_tasks.process_wake_queue",
            "schedule": 60.0,   # every 60 seconds
            "options": {"queue": "wake"},
        },
        "cleanup-archived-agents": {
            "task": "grif.tasks.agent_tasks.cleanup_archived_agents",
            "schedule": 3600.0,  # every hour
            "options": {"queue": "cleanup"},
        },
    },
)
