"""
Prompt layer loader.
Reads YAML templates from prompts/templates/ directory.
"""

from functools import lru_cache
from pathlib import Path

import yaml

_TEMPLATES_DIR = Path(__file__).parent / "templates"


@lru_cache(maxsize=64)
def load_template(name: str) -> str:
    """Load a YAML template file and return its `content` field."""
    path = _TEMPLATES_DIR / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Prompt template not found: {name}")
    with path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data["content"]


@lru_cache(maxsize=1)
def load_core_identity() -> str:
    return load_template("core_identity")


@lru_cache(maxsize=1)
def load_classifier_prompt() -> str:
    return load_template("classifier_prompt")


def load_role_template(blueprint_id: str) -> str:
    """Load role template by blueprint_id. Falls back to generic_worker."""
    try:
        return load_template(blueprint_id)
    except FileNotFoundError:
        return load_template("generic_worker")


def get_template_for_task_type(task_type: str, domain: str | None = None) -> str:
    """
    Return the best matching template name for a given task_type and domain.
    Priority: domain-specific > task_type-specific > generic_worker.
    """
    # Domain-specific mappings
    domain_map: dict[str, str] = {
        "travel": "travel_scout",
        "hotels": "travel_scout",
        "flights": "travel_scout",
        "content": "content_writer",
        "blog": "content_writer",
        "social_media": "content_writer",
        "email_marketing": "content_writer",
        "ecommerce": "product_comparator",
        "consumer_goods": "product_comparator",
        "negotiation": "negotiation_coach",
        "sales": "negotiation_coach",
        "finance": "price_tracker",
        "crypto": "price_tracker",
    }

    # Task-type fallbacks
    task_type_map: dict[str, str] = {
        "research": "research_analyst",
        "compare": "product_comparator",
        "monitor": "price_tracker",
        "generate": "content_writer",
        "operate": "content_writer",
        "coach": "negotiation_coach",
    }

    if domain and domain in domain_map:
        return domain_map[domain]
    if task_type in task_type_map:
        return task_type_map[task_type]
    return "generic_worker"
