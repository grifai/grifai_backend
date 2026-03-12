import os
import getpass
from pathlib import Path
from dotenv import load_dotenv, set_key

_ENV_FILE = Path(__file__).parent / ".env"
load_dotenv(_ENV_FILE)


def _require(key: str, prompt: str, secret: bool = False) -> str:
    val = os.getenv(key, "").strip()
    if val:
        return val
    val = getpass.getpass(f"{prompt}: ") if secret else input(f"{prompt}: ")
    val = val.strip()
    if val:
        _ENV_FILE.touch(exist_ok=True)
        set_key(str(_ENV_FILE), key, val)
        print(f"  Saved {key} to .env")
    return val


# Telegram
API_ID   = int(_require("TG_API_ID",   "Telegram API ID"))
API_HASH = _require("TG_API_HASH",     "Telegram API Hash", secret=True)

# OpenAI
OPENAI_KEY = _require("OPENAI_KEY", "OpenAI API Key", secret=True)

# Paths
SESSION_FILE = os.getenv("SESSION_FILE", "jarvis_session")
MEMORY_FILE  = Path(os.getenv("MEMORY_FILE", "jarvis_memory.json"))
CACHE_DIR    = Path("jarvis_cache")
CACHE_DIR.mkdir(exist_ok=True)

# Behaviour
BATCH_WAIT_SEC  = int(os.getenv("BATCH_WAIT_SEC",  "5"))
SCAN_CONTACTS   = int(os.getenv("SCAN_CONTACTS",   "50"))
SCAN_MESSAGES   = int(os.getenv("SCAN_MESSAGES",   "150"))
CONTEXT_WINDOW  = int(os.getenv("CONTEXT_WINDOW",  "40"))
MODEL           = os.getenv("MODEL", "claude-haiku-4-5-20251001")
