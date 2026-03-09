import os
import getpass
from pathlib import Path

from dotenv import load_dotenv, set_key
from pydantic import Field, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict

_ENV_FILE = Path(__file__).parent.parent / ".env"

# Load .env before anything else so _require() sees existing values
load_dotenv(_ENV_FILE)


def _require(key: str, prompt: str, secret: bool = False) -> str:
    """Return env var value; prompt interactively and persist to .env if missing."""
    val = os.getenv(key, "").strip()
    if val:
        return val
    val = getpass.getpass(f"{prompt}: ") if secret else input(f"{prompt}: ")
    val = val.strip()
    if val:
        _ENV_FILE.touch(exist_ok=True)
        set_key(str(_ENV_FILE), key, val)
        # Expose immediately so Settings() picks it up without re-reading disk
        os.environ[key] = val
        print(f"  Saved {key} to .env")
    return val


# Prompt for required credentials before Settings validates them
_require("TG_API_ID",   "Telegram API ID")
_require("TG_API_HASH", "Telegram API Hash", secret=True)
_require("OPENAI_KEY",  "OpenAI API Key",    secret=True)


class Settings(BaseSettings):
    # Telegram
    tg_api_id: int
    tg_api_hash: str

    # LLM
    openai_key: str
    anthropic_key: str = ""
    model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"

    # Behaviour
    scan_messages: int = 1500
    scan_contacts: int = 50
    batch_wait_sec: int = 5
    context_window: int = 40

    # Databases (placeholders — filled when migrating from flat files)
    database_url: str = "postgresql+asyncpg://jarvis:jarvis@localhost:5432/jarvis"
    redis_url: str = "redis://localhost:6379"
    qdrant_url: str = "http://localhost:6333"

    # TTS
    elevenlabs_key: str = ""

    # Paths
    data_dir: str = "data"
    session_name: str = "jarvis_session"

    # API server
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    model_config = SettingsConfigDict(
        env_file=str(_ENV_FILE),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @computed_field
    @property
    def session_file(self) -> str:
        """Full path to the Telethon session file (no extension)."""
        return f"{self.data_dir}/{self.session_name}"

    @computed_field
    @property
    def memory_file(self) -> Path:
        """Path to the JSON memory store."""
        return Path(self.data_dir) / "jarvis_memory.json"

    def ensure_data_dir(self) -> None:
        Path(self.data_dir).mkdir(exist_ok=True)


settings = Settings()
settings.ensure_data_dir()
