import asyncio
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

from telethon import TelegramClient

from app.config import settings
from app.llm import get_llm
from app.llm.base import LLMProvider
from app.llm.prompts import DIGEST_PROMPT
from app.memory.rag import VectorMemory
from app.services.tts import TTSService

# ── Result model ───────────────────────────────────────────────────────────────


@dataclass
class DigestResult:
    summary: str
    dialogs_count: int
    hours: int
    generated_at: str
    audio_path: str | None = None


# ── DigestService ─────────────────────────────────────────────────────────────


class DigestService:
    """Generates text digests from RAG history and optionally speaks them."""

    def __init__(
        self,
        llm: LLMProvider,
        vector_memory: VectorMemory,
        tts: TTSService | None = None,
    ):
        self.llm = llm
        self.vm = vector_memory
        self.tts = tts

    # ── Text digest ────────────────────────────────────────────────────────────

    def generate_digest(self, hours: int = 24) -> DigestResult:
        """Build a digest from RAG-indexed messages for the past `hours`."""
        date_from = (datetime.now() - timedelta(hours=hours)).strftime("%Y-%m-%dT%H:%M:%S")
        msgs = self.vm.get_contact_messages(date_from=date_from, max_messages=3000)

        if not msgs:
            return DigestResult(
                summary=f"За последние {hours} часов активных переписок не найдено.",
                dialogs_count=0,
                hours=hours,
                generated_at=datetime.now().isoformat(),
            )

        by_contact: dict[str, list[dict]] = defaultdict(list)
        for m in msgs:
            by_contact[m["contact_name"]].append(m)

        blocks = []
        for contact, contact_msgs in sorted(by_contact.items()):
            sorted_msgs = sorted(contact_msgs, key=lambda x: x.get("date", ""))
            lines = [f"=== {contact} ==="]
            for m in sorted_msgs[-20:]:
                who = "Я" if m["mine"] else contact
                lines.append(f"[{who}]: {m['text'][:200]}")
            blocks.append("\n".join(lines))

        full_text = "\n\n".join(blocks)
        if len(full_text) > 14000:
            full_text = full_text[:14000] + "\n...(обрезано)"

        summary_text = self.llm.generate(
            system_prompt=DIGEST_PROMPT.format(hours=hours),
            user_message=full_text,
            temperature=0.5,
            max_tokens=1200,
        )

        return DigestResult(
            summary=summary_text,
            dialogs_count=len(by_contact),
            hours=hours,
            generated_at=datetime.now().isoformat(),
        )

    # ── Voice digest ───────────────────────────────────────────────────────────

    async def generate_and_speak(
        self,
        hours: int = 24,
        client: TelegramClient | None = None,
    ) -> DigestResult:
        """
        Generate a text digest, convert to speech, optionally send via Telegram.

        If `client` is provided, sends the audio file to Saved Messages ("me").
        """
        result = self.generate_digest(hours)

        if self.tts is None:
            return result

        audio = self.tts.synthesize(result.summary)
        if audio is None:
            return result

        ts = datetime.now().strftime("%Y%m%d_%H%M")
        audio_path = self.tts.save_audio(audio, f"digest_{ts}")
        result.audio_path = audio_path

        if client is not None:
            try:
                caption = f"📋 Дайджест за {hours}ч — {datetime.now().strftime('%d.%m %H:%M')}"
                await client.send_file("me", audio_path, caption=caption)
                print(f"  [digest] Sent audio to Saved Messages ({audio_path})")
            except Exception as exc:
                print(f"  [digest] Failed to send audio via Telegram: {exc}")

        return result


# ── Legacy CLI entry point ────────────────────────────────────────────────────


async def run(hours: int = 24):
    """Standalone CLI runner (python -m app.services.digest or scripts/)."""
    llm = get_llm(api_key=settings.openai_key, model=settings.model)

    client = TelegramClient(settings.session_file, settings.tg_api_id, settings.tg_api_hash)
    await client.start()
    me = await client.get_me()
    my_id = me.id

    cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
    print(f"Сбор сообщений за последние {hours}ч...")

    active_dialogs: list[str] = []
    scanned = 0

    async for dialog in client.iter_dialogs(limit=500):
        if not dialog.is_user:
            continue
        if scanned >= settings.scan_contacts:
            break

        name = dialog.name or str(dialog.id)
        recent: list[tuple[str, str]] = []

        async for msg in client.iter_messages(dialog.id, limit=80):
            if not msg.text:
                continue
            if msg.date < cutoff:
                break
            who = "Я" if msg.sender_id == my_id else name
            recent.append((who, msg.text[:300]))

        if recent:
            recent.reverse()
            block = "\n".join(f"[{who}]: {text}" for who, text in recent)
            active_dialogs.append(f"=== {name} ===\n{block}")

        scanned += 1

    await client.disconnect()

    if not active_dialogs:
        print(f"За последние {hours} часов нет активных переписок.")
        return

    print(f"Активно: {len(active_dialogs)} чатов. Генерирую саммари...")

    full_text = "\n\n".join(active_dialogs)
    if len(full_text) > 14000:
        full_text = full_text[:14000] + "\n...(обрезано)"

    summary = llm.generate(
        system_prompt=DIGEST_PROMPT.format(hours=hours),
        user_message=full_text,
        temperature=0.5,
        max_tokens=1200,
    )

    now = datetime.now().strftime("%d.%m.%Y %H:%M")
    sep = "=" * 60
    print(f"\n{sep}")
    print(f"  JARVIS SUMMARY — {now}  (последние {hours}ч)")
    print(f"  Диалогов с активностью: {len(active_dialogs)}")
    print(f"{sep}\n")
    print(summary)
    print(f"\n{sep}\n")


if __name__ == "__main__":
    hours = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 24
    asyncio.run(run(hours))
