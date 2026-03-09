#!/usr/bin/env python3
"""
Migrate jarvis_memory.json -> PostgreSQL (skeleton).

Future: when the contact profile store outgrows a flat JSON file,
run this script to migrate to a proper database.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

MEMORY_FILE = Path("data/jarvis_memory.json")


def migrate():
    raise NotImplementedError(
        "PostgreSQL migration not yet implemented. "
        "Install asyncpg + sqlalchemy and implement the schema first."
    )


if __name__ == "__main__":
    migrate()
