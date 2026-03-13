"""
Max messenger source — читает переписки и отправляет ответы через pymax (userbot).
Установка: pip install maxapi-python
"""
import asyncio

from pymax import MaxClient, Message
from pymax.filters import Filters

MAX_SESSION_DIR = "max_session"


def _format_dialog(msgs: list[dict]) -> str:
    lines = []
    for m in msgs:
        who = "Me" if m["mine"] else "Them"
        lines.append(f"[{who}]: {m['text'][:300]}")
    return "\n".join(lines)


class MaxSource:
    def __init__(
        self,
        phone: str,
        on_message,
        memory,
        model: str,
        scan_contacts: int = 50,
        scan_messages: int = 150,
    ):
        self.phone = phone
        self._on_message = on_message
        self.memory = memory
        self.model = model
        self.scan_contacts = scan_contacts
        self.scan_messages = scan_messages

        self._client = MaxClient(phone=phone, work_dir=MAX_SESSION_DIR)
        self._me_id: int | None = None
        self._register_handlers()

    # ── Prescan ───────────────────────────────────────────────────────────────

    async def prescan(self):
        import ai

        print(f"\nMax: scanning contacts (target: {self.scan_contacts})...")

        try:
            dialogs = await self._client.get_dialogs()
        except Exception as e:
            print(f"  Could not fetch dialogs: {e}")
            return

        if not dialogs:
            print("  No dialogs found")
            return

        scanned = 0
        for dialog in dialogs[: self.scan_contacts]:
            chat_id = dialog.chat_id
            name = getattr(dialog, "name", None) or str(chat_id)
            contact_id = f"max_{chat_id}"

            if self.memory.get_contact(contact_id):
                print(f"  skip {name} — profile up to date")
                continue

            try:
                history = await self._client.fetch_history(chat_id=chat_id)
            except Exception as e:
                print(f"  skip {name} — fetch error: {e}")
                continue

            msgs = self._convert_history(history)
            my_count = sum(1 for m in msgs if m["mine"])
            their_count = sum(1 for m in msgs if not m["mine"])

            if my_count < 3 or their_count < 3:
                print(f"  skip {name} — too few messages ({my_count}+{their_count})")
                continue

            dialog_text = _format_dialog(msgs)
            print(f"  analyzing {name}...", end=" ", flush=True)
            try:
                profile = ai.analyze_contact(dialog_text, self.model)
                self.memory.set_contact(contact_id, name, profile)
                rel = (
                    profile.get("relationship", "?")[:50]
                    if isinstance(profile, dict)
                    else "?"
                )
                print(f"ok ({rel})")
            except Exception as e:
                print(f"error: {e}")

            scanned += 1
            await asyncio.sleep(0.5)

        print(f"Max scan complete: {scanned} analyzed")

    # ── Handlers ──────────────────────────────────────────────────────────────

    def _register_handlers(self):
        @self._client.on_message(Filters.private())
        async def on_message(msg: Message):
            if not msg.text:
                return
            if self._me_id and getattr(msg, "sender_id", None) == self._me_id:
                return

            contact_id = f"max_{msg.chat_id}"
            if self.memory.get_contact_ai_mode(contact_id) == "never":
                return

            sender_name = getattr(msg, "sender_name", None) or str(
                getattr(msg, "sender_id", msg.chat_id)
            )

            reply = await self._on_message(
                platform="max",
                chat_id=str(msg.chat_id),
                sender_name=sender_name,
                sender_id=str(getattr(msg, "sender_id", msg.chat_id)),
                text=msg.text,
            )
            if reply:
                await self.send(msg.chat_id, reply)

    # ── Send ──────────────────────────────────────────────────────────────────

    async def send(self, chat_id: int | str, text: str):
        await self._client.send_message(chat_id=int(chat_id), text=text)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _convert_history(self, history) -> list[dict]:
        msgs = []
        for m in history:
            text = getattr(m, "text", None)
            if not text:
                continue
            sender_id = getattr(m, "sender_id", None)
            is_mine = self._me_id is not None and sender_id == self._me_id
            msgs.append({"text": text, "mine": is_mine})
        return msgs

    # ── Start ─────────────────────────────────────────────────────────────────

    async def start(self):
        @self._client.on_start
        async def on_start():
            self._me_id = getattr(self._client.me, "id", None)
            print(f"Max: connected (id={self._me_id})")

        await self._client.start()
