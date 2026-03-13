"""
WhatsApp source — получает сообщения от Baileys-моста и отправляет ответы.
"""
import asyncio
import httpx
from aiohttp import web

WA_BRIDGE_URL = "http://localhost:3001"
WEBHOOK_HOST = "0.0.0.0"
WEBHOOK_PORT = 8765


def _format_dialog(msgs: list[dict]) -> str:
    lines = []
    for m in msgs:
        who = "Me" if m["mine"] else "Them"
        lines.append(f"[{who}]: {m['text'][:300]}")
    return "\n".join(lines)

class WhatsAppSource:
    def __init__(self, on_message, memory, model, scan_contacts=50, scan_messages=150):
        self._on_message = on_message
        self.memory = memory
        self.model = model
        self.scan_contacts = scan_contacts
        self.scan_messages = scan_messages
        self._app = web.Application()
        self._app.router.add_post("/whatsapp", self._handle_webhook)

    # ── Prescan ───────────────────────────────────────────────────────────────

    async def prescan(self):
        import ai
        print(f"\nWhatsApp: scanning contacts (target: {self.scan_contacts})...")

        async with httpx.AsyncClient(timeout=10) as client:
            try:
                r = await client.get(f"{WA_BRIDGE_URL}/chats")
                chats = r.json().get("chats", [])
            except Exception as e:
                print(f"  Could not fetch chats: {e}")
                return

        if not chats:
            print("  No chats found yet (send a message first to populate store)")
            return

        scanned = 0
        all_my_msgs: list[str] = []

        for chat in chats[:self.scan_contacts]:
            chat_id = chat["id"]
            name = chat["name"]
            contact_id = f"wa_{chat_id}"

            async with httpx.AsyncClient(timeout=15) as client:
                try:
                    r = await client.post(
                        f"{WA_BRIDGE_URL}/messages",
                        json={"chat_id": chat_id, "limit": self.scan_messages},
                    )
                    msgs = r.json().get("messages", [])
                except Exception as e:
                    print(f"  skip {name} — fetch error: {e}")
                    continue

            all_my_msgs.extend(m["text"] for m in msgs if m["mine"])

            if self.memory.get_contact(contact_id):
                print(f"  skip {name} — profile up to date")
                continue

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
                rel = profile.get("relationship", "?")[:50] if isinstance(profile, dict) else "?"
                print(f"ok ({rel})")
            except Exception as e:
                print(f"error: {e}")

            scanned += 1
            await asyncio.sleep(0.5)

        if all_my_msgs and not self.memory.get_my_profile():
            print("WhatsApp: building general style profile...", end=" ", flush=True)
            try:
                style = ai.analyze_my_style(all_my_msgs, self.model)
                self.memory.set_my_profile(style)
                print("ok")
            except Exception as e:
                print(f"error: {e}")

        print(f"WhatsApp scan complete: {scanned} analyzed")

    # ── Webhook ───────────────────────────────────────────────────────────────

    async def _handle_webhook(self, request: web.Request) -> web.Response:
        data = await request.json()
        asyncio.create_task(self._process(
            chat_id=data["chat_id"],
            sender_name=data["sender_name"],
            sender_id=data["sender_id"],
            text=data["text"],
        ))
        return web.json_response({"ok": True})

    async def _process(self, chat_id: str, sender_name: str, sender_id: str, text: str):
        reply = await self._on_message(
            platform="whatsapp",
            chat_id=chat_id,
            sender_name=sender_name,
            sender_id=sender_id,
            text=text,
        )
        if reply:
            await self.send(chat_id, reply)

    # ── Send ──────────────────────────────────────────────────────────────────

    async def send(self, chat_id: str, text: str):
        async with httpx.AsyncClient() as client:
            await client.post(f"{WA_BRIDGE_URL}/send", json={"chat_id": chat_id, "text": text})

    async def is_connected(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=2) as client:
                r = await client.get(f"{WA_BRIDGE_URL}/status")
                return r.json().get("connected", False)
        except Exception:
            return False

    async def start(self):
        runner = web.AppRunner(self._app)
        await runner.setup()
        site = web.TCPSite(runner, WEBHOOK_HOST, WEBHOOK_PORT)
        await site.start()
        connected = await self.is_connected()
        status = "connected" if connected else "bridge not running"
        print(f"WhatsApp: {status} (webhook on :{WEBHOOK_PORT})")
