import asyncio
import json
from datetime import datetime

from telethon import TelegramClient, events

from app.llm import openai_provider as ai
from app.memory import rag
from app.memory.contacts import JarvisMemory
from app.services.ghost_writer import GhostWriter
from app.services.event_bus import EventBus, STREAMS


def format_dialog(msgs: list[dict]) -> str:
    lines = []
    for m in msgs:
        who = "Me" if m["mine"] else "Them"
        lines.append(f"[{who}]: {m['text'][:300]}")
    return "\n".join(lines)


async def fetch_messages(
    client: TelegramClient, chat_id: int, limit: int
) -> tuple[list[dict], int, int]:
    me = await client.get_me()
    my_id = me.id
    msgs: list[dict] = []
    my_count = 0
    their_count = 0

    async for m in client.iter_messages(chat_id, limit=limit):
        if not m.text:
            continue
        is_mine = m.sender_id == my_id
        msgs.append({"text": m.text, "mine": is_mine, "date": m.date.isoformat()})
        if is_mine:
            my_count += 1
        else:
            their_count += 1

    msgs.reverse()
    return msgs, my_count, their_count


# ── Approval UI ───────────────────────────────────────────────────────────────

def ask_approval(
    sender: str,
    contact_id: str,
    texts: list[str],
    draft: str,
    memory: JarvisMemory,
) -> tuple[str, str | None]:
    print()
    for t in texts:
        print(f"  >> {t}")
    print("-" * 55)

    if draft == "[SKIP]":
        print("Jarvis: no reply needed")
        return "skipped", None

    print(f"Jarvis: {draft!r}")
    print("-" * 55)

    while True:
        mode = memory.get_contact_ai_mode(contact_id)
        ai_label = "AI:ON " if mode == "auto" else "AI:OFF"
        print(f"  1 Send   2 Edit   3 Regen   4 Skip   5 Profile   6 [{ai_label}] Toggle")

        ch = input("-> ").strip()
        if ch == "1":
            return "approved", draft
        elif ch == "2":
            return "revised", input("Your text: ").strip()
        elif ch == "3":
            return "redo", None
        elif ch == "4":
            return "skipped", None
        elif ch == "5":
            c = memory.get_contact(contact_id)
            if c and c.get("profile"):
                print(json.dumps(c["profile"], ensure_ascii=False, indent=2)[:800])
            else:
                print("Profile not built yet")
        elif ch == "6":
            new_mode = "never" if mode == "auto" else "auto"
            memory.set_contact_ai_mode(contact_id, new_mode)
            status = "ВЫКЛЮЧЕН" if new_mode == "never" else "ВКЛЮЧЁН"
            print(f"  AI {status} для {sender}. Следующие сообщения от него {'будут игнорироваться' if new_mode == 'never' else 'снова обрабатываться'}.")
        else:
            print("  Enter 1-6")


# ── Message batching ──────────────────────────────────────────────────────────

class JarvisBot:
    def __init__(
        self,
        client: TelegramClient,
        memory: JarvisMemory,
        model: str,
        batch_wait_sec: int,
        scan_messages: int,
        context_window: int,
        scan_contacts: int,
        ghost_writer: GhostWriter | None = None,
        event_bus: EventBus | None = None,
    ):
        self.client = client
        self.memory = memory
        self.model = model
        self.batch_wait = batch_wait_sec
        self.scan_messages = scan_messages
        self.context_window = context_window
        self.scan_contacts = scan_contacts
        self.ghost_writer = ghost_writer
        self.event_bus = event_bus
        self._buffers: dict[int, dict] = {}

    async def prescan(self):
        print(f"\nScanning personal dialogs (target: {self.scan_contacts})...")

        all_my_msgs: list[str] = []
        scanned = 0
        skipped = 0
        personal_found = 0

        async for dialog in self.client.iter_dialogs(limit=500):
            if personal_found >= self.scan_contacts:
                break

            if not dialog.is_user:
                skipped += 1
                continue

            personal_found += 1
            contact_id = str(dialog.id)
            name = dialog.name or str(dialog.id)

            if self.memory.get_contact(contact_id):
                print(f"  skip {name} — profile up to date")
                scanned += 1
                continue

            msgs, my_count, their_count = await fetch_messages(
                self.client, dialog.id, self.scan_messages
            )

            if my_count < 3 or their_count < 3:
                print(f"  skip {name} — too few messages ({my_count}+{their_count})")
                skipped += 1
                continue

            all_my_msgs.extend(m["text"] for m in msgs if m["mine"])

            dialog_text = format_dialog(msgs)
            print(f"  [{personal_found}/{self.scan_contacts}] analyzing {name}...", end=" ", flush=True)
            try:
                profile = ai.analyze_contact(dialog_text, self.model)
                self.memory.set_contact(contact_id, name, profile)
                rel = profile.get("relationship", "?")[:50] if isinstance(profile, dict) else "?"
                print(f"ok ({rel})")
            except Exception as e:
                print(f"error: {e}")

            scanned += 1
            await asyncio.sleep(0.5)

        if all_my_msgs and not self.memory.get_my_profile():
            print(f"\nBuilding general style profile ({len(all_my_msgs)} messages)...")
            style = ai.analyze_my_style(all_my_msgs, self.model)
            self.memory.set_my_profile(style)
            print("done")

        print(f"\nScan complete: {scanned} analyzed, {skipped} skipped")
        print(f"Total profiles: {len(self.memory.data['contacts'])}")

    async def _process_batch(self, chat_id: int):
        if chat_id not in self._buffers:
            return
        buf = self._buffers.pop(chat_id)
        sender = buf["sender"]
        contact_id = buf["contact_id"]
        texts = buf["texts"]
        last_event = buf["events"][-1]

        n = len(texts)
        print(f"\n{'=' * 55}")
        print(f"From {sender} — {'batch of ' + str(n) if n > 1 else '1 message'}")

        # ── Event-driven path (Redis Streams) ──────────────────────────────────
        if self.event_bus:
            await self.event_bus.publish(STREAMS["INCOMING"], "batch_received", {
                "contact_id": contact_id,
                "contact_name": sender,
                "texts": texts,
                "chat_id": str(chat_id),
                "timestamp": datetime.now().isoformat(),
            })
            print(f"  → queued to incoming_messages")
            return

        # ── Inline fallback (no Redis) ─────────────────────────────────────────
        msgs, _, _ = await fetch_messages(
            self.client, chat_id, self.context_window + n
        )
        context_msgs = msgs[:-n] if n < len(msgs) else []
        chat_context = format_dialog(context_msgs[-self.context_window:])

        if not self.memory.get_contact(contact_id):
            print(f"  Building profile for {sender}...")
            full_msgs, my_c, their_c = await fetch_messages(
                self.client, chat_id, self.scan_messages
            )
            if my_c >= 3 and their_c >= 3:
                try:
                    profile = ai.analyze_contact(format_dialog(full_msgs), self.model)
                    self.memory.set_contact(contact_id, sender, profile)
                    print("  Profile built")
                except Exception as e:
                    print(f"  Profile error: {e}")

        if self.ghost_writer:
            reply_draft = self.ghost_writer.generate_reply(contact_id, texts, chat_context)
            draft = reply_draft.text
            print(f"  GhostWriter confidence: {reply_draft.confidence:.0%}")
        else:
            rag_results = rag.search(" ".join(texts), k=5, min_score=0.4)
            rag_context = rag.format_rag_context(rag_results)
            if rag_context:
                print(f"  RAG: {len(rag_results)} relevant messages found")
            draft = ai.generate_reply(
                sender, contact_id, texts, chat_context, self.memory, self.model,
                rag_context=rag_context,
            )

        while True:
            action, final = ask_approval(sender, contact_id, texts, draft, self.memory)
            if action == "redo":
                print("Regenerating...")
                if self.ghost_writer:
                    draft = self.ghost_writer.generate_reply(contact_id, texts, chat_context).text
                else:
                    draft = ai.generate_reply(
                        sender, contact_id, texts, chat_context, self.memory, self.model
                    )
                continue
            if action in ("approved", "revised") and final:
                await last_event.reply(final)
                print(f"Sent: {final!r}")
                self.memory.add_example(sender, " | ".join(texts), draft, action, final)
                if self.ghost_writer:
                    self.ghost_writer.learn_from_approval(contact_id, " | ".join(texts), final)
            else:
                print("Skipped")
                self.memory.add_example(sender, " | ".join(texts), draft, "skipped")
            break

    async def _buffer_message(
        self, chat_id: int, sender: str, contact_id: str, text: str, event
    ):
        if chat_id not in self._buffers:
            self._buffers[chat_id] = {
                "sender": sender,
                "contact_id": contact_id,
                "texts": [],
                "events": [],
                "timer": None,
            }
        buf = self._buffers[chat_id]
        buf["texts"].append(text)
        buf["events"].append(event)

        if buf["timer"] is not None:
            buf["timer"].cancel()

        loop = asyncio.get_event_loop()
        buf["timer"] = loop.call_later(
            self.batch_wait,
            lambda cid=chat_id: asyncio.ensure_future(self._process_batch(cid)),
        )

    def register_handlers(self):
        @self.client.on(events.NewMessage(incoming=True))
        async def on_message(event):
            if not (event.is_private and event.raw_text):
                return
            if self.memory.get_contact_ai_mode(str(event.chat_id)) == "never":
                return
            sender = await event.get_sender()
            name = (
                f"{sender.first_name or ''} {sender.last_name or ''}".strip()
                or str(sender.id)
            )
            await self._buffer_message(
                event.chat_id, name, str(event.chat_id), event.raw_text, event
            )
