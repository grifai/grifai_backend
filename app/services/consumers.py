"""
Redis Streams consumers for event-driven bot pipeline.

Flow:
  on_message → buffer → publish "incoming_messages"
      ↓
  ghost_writer_consumer → generate draft → publish "reply_drafts"
      ↓
  approval_ui_consumer → terminal UI → send reply → publish "approved_replies"
      ↓
  learner_consumer → ghost_writer.learn_from_approval()

  profile_updates → profiler_consumer → rebuild contact profile
"""
import asyncio
from datetime import datetime

from telethon import TelegramClient

from app.llm import openai_provider as ai
from app.memory.contacts import JarvisMemory
from app.services.event_bus import EventBus, STREAMS
from app.services.ghost_writer import GhostWriter


# ── Helpers (imported lazily to avoid circular imports) ───────────────────────

def _bot_utils():
    from app.bot.client import fetch_messages, format_dialog, ask_approval
    return fetch_messages, format_dialog, ask_approval


# ── Consumer 1: Ghost Writer ───────────────────────────────────────────────────

async def ghost_writer_consumer(
    bus: EventBus,
    ghost_writer: GhostWriter,
    client: TelegramClient,
    memory: JarvisMemory,
    context_window: int = 40,
    scan_messages: int = 1500,
) -> None:
    """incoming_messages → fetch context → generate draft → reply_drafts"""
    fetch_messages, format_dialog, _ = _bot_utils()

    async def handle(event_type: str, data: dict, msg_id: str) -> None:
        contact_id: str = data["contact_id"]
        contact_name: str = data["contact_name"]
        texts: list[str] = data["texts"]
        chat_id = int(data["chat_id"])
        n = len(texts)

        # Fetch recent chat context from Telegram
        msgs, _, _ = await fetch_messages(client, chat_id, context_window + n)
        context_msgs = msgs[:-n] if n < len(msgs) else []
        chat_context = format_dialog(context_msgs[-context_window:])

        # Build relationship profile on first encounter
        if not memory.get_contact(contact_id):
            print(f"  [gw] Building profile for {contact_name}...")
            full_msgs, my_c, their_c = await fetch_messages(client, chat_id, scan_messages)
            if my_c >= 3 and their_c >= 3:
                try:
                    profile = ai.analyze_contact(format_dialog(full_msgs))
                    memory.set_contact(contact_id, contact_name, profile)
                except Exception as exc:
                    print(f"  [gw] Profile error: {exc}")

        draft = ghost_writer.generate_reply(contact_id, texts, chat_context)
        print(f"  [gw] Draft for {contact_name} (conf={draft.confidence:.0%}): {draft.text[:60]!r}")

        await bus.publish(STREAMS["DRAFTS"], "draft_ready", {
            "contact_id": contact_id,
            "contact_name": contact_name,
            "draft_text": draft.text,
            "confidence": draft.confidence,
            "incoming_texts": texts,
            "chat_context": chat_context,
            "chat_id": str(chat_id),
            "timestamp": datetime.now().isoformat(),
        })

    await bus.subscribe(
        STREAMS["INCOMING"], "ghost_writer_group", "gw1", handle, batch_size=5
    )


# ── Consumer 2: Approval UI ────────────────────────────────────────────────────

async def approval_ui_consumer(
    bus: EventBus,
    ghost_writer: GhostWriter,
    client: TelegramClient,
    memory: JarvisMemory,
) -> None:
    """reply_drafts → terminal approval UI → send reply → approved_replies"""
    _, _, ask_approval = _bot_utils()

    async def handle(event_type: str, data: dict, msg_id: str) -> None:
        contact_id: str = data["contact_id"]
        contact_name: str = data["contact_name"]
        chat_id = int(data["chat_id"])
        texts: list[str] = data["incoming_texts"]
        chat_context: str = data.get("chat_context", "")
        initial_draft: str = data["draft_text"]

        print(f"\n{'=' * 55}")
        print(f"From {contact_name} — {len(texts)} message(s)")

        # Run the blocking terminal UI in a thread so the event loop stays live
        loop = asyncio.get_running_loop()

        def run_approval() -> tuple[str, str | None, str]:
            current = initial_draft
            while True:
                action, final = ask_approval(contact_name, contact_id, texts, current, memory)
                if action == "redo":
                    print("Regenerating...")
                    current = ghost_writer.generate_reply(contact_id, texts, chat_context).text
                    continue
                return action, final, current

        action, final, used_draft = await loop.run_in_executor(None, run_approval)

        if action in ("approved", "revised") and final:
            await client.send_message(chat_id, final)
            print(f"Sent: {final!r}")
            memory.add_example(contact_name, " | ".join(texts), used_draft, action, final)
            await bus.publish(STREAMS["APPROVED"], "reply_sent", {
                "contact_id": contact_id,
                "incoming_text": " | ".join(texts),
                "reply_text": final,
                "timestamp": datetime.now().isoformat(),
            })
        else:
            print("Skipped")
            memory.add_example(contact_name, " | ".join(texts), used_draft, "skipped")

    # batch_size=1: process approvals one at a time (terminal is single-user)
    await bus.subscribe(
        STREAMS["DRAFTS"], "approval_ui_group", "ui1", handle, batch_size=1
    )


# ── Consumer 3: Learner ────────────────────────────────────────────────────────

async def learner_consumer(
    bus: EventBus,
    ghost_writer: GhostWriter,
) -> None:
    """approved_replies → ghost_writer.learn_from_approval()"""

    async def handle(event_type: str, data: dict, msg_id: str) -> None:
        ghost_writer.learn_from_approval(
            data["contact_id"],
            data["incoming_text"],
            data["reply_text"],
        )

    await bus.subscribe(STREAMS["APPROVED"], "learner_group", "learner1", handle)


# ── Consumer 4: Profiler ───────────────────────────────────────────────────────

async def profiler_consumer(
    bus: EventBus,
    client: TelegramClient,
    memory: JarvisMemory,
    scan_messages: int = 1500,
) -> None:
    """profile_updates → re-analyze contact dialogue with LLM → update memory"""
    fetch_messages, format_dialog, _ = _bot_utils()

    async def handle(event_type: str, data: dict, msg_id: str) -> None:
        contact_id: str = data["contact_id"]
        trigger: str = data.get("trigger", "manual")
        c = memory.data.get("contacts", {}).get(contact_id, {})
        contact_name = c.get("name", contact_id)

        print(f"  [profiler] Rebuilding profile for {contact_name} (trigger={trigger})")
        try:
            msgs, my_c, their_c = await fetch_messages(
                client, int(contact_id), scan_messages
            )
            if my_c >= 3 and their_c >= 3:
                profile = ai.analyze_contact(format_dialog(msgs))
                memory.set_contact(contact_id, contact_name, profile)
                print(f"  [profiler] Done for {contact_name}")
            else:
                print(f"  [profiler] Not enough messages for {contact_name}")
        except Exception as exc:
            print(f"  [profiler] Error for {contact_name}: {exc}")

    await bus.subscribe(
        STREAMS["PROFILE_UPDATES"], "profiler_group", "profiler1", handle
    )
