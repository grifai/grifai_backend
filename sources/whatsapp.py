"""
WhatsApp source — получает сообщения от Baileys-моста и отправляет ответы.
"""
import asyncio
from pathlib import Path

import httpx
from aiohttp import web

_PORT_FILE = Path(__file__).parent.parent / "whatsapp_bridge" / "whatsapp_session" / "port.txt"
_PORT_RANGE = range(3001, 3006)
WEBHOOK_HOST = "0.0.0.0"
WEBHOOK_PORT = 8765


def _find_bridge_url() -> str:
    """Читает порт из файла моста, иначе перебирает диапазон."""
    if _PORT_FILE.exists():
        try:
            port = int(_PORT_FILE.read_text().strip())
            return f"http://localhost:{port}"
        except Exception:
            pass
    # Fallback: пробуем порты по очереди синхронно через os
    for port in _PORT_RANGE:
        try:
            import urllib.request
            urllib.request.urlopen(f"http://localhost:{port}/status", timeout=1)
            return f"http://localhost:{port}"
        except Exception:
            continue
    return f"http://localhost:{_PORT_RANGE.start}"


WA_BRIDGE_URL = _find_bridge_url()


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
        # Обновляем URL при создании объекта (мост мог запуститься после импорта)
        self._bridge_url = _find_bridge_url()
        self.scan_messages = scan_messages
        self.queue: asyncio.Queue = asyncio.Queue()
        self._app = web.Application()
        self._app.router.add_post("/whatsapp", self._handle_webhook)

    # ── Prescan ───────────────────────────────────────────────────────────────

    async def prescan(self):
        import ai
        print(f"\nWhatsApp: scanning contacts (target: {self.scan_contacts})...")

        chats = []
        for attempt in range(6):
            async with httpx.AsyncClient(timeout=10) as client:
                try:
                    r = await client.get(f"{self._bridge_url}/chats-all")
                    chats = r.json().get("chats", [])
                except Exception as e:
                    print(f"  Ошибка: {e}")
                    break
            if chats:
                break
            print(f"  История ещё загружается, ждём 5 сек... ({attempt + 1}/6)")
            await asyncio.sleep(5)

        if not chats:
            print("  Чаты не найдены — история WhatsApp ещё не синхронизирована.")
            print("  Подожди минуту и попробуй снова через меню → Пересканировать.")
            return

        print(f"  Бридж вернул {len(chats)} чатов, берём первые {self.scan_contacts}")
        with_history = sum(1 for c in chats if c.get("has_history"))
        print(f"  Из них с историей сообщений: {with_history}")

        scanned = 0
        for chat in chats[:self.scan_contacts]:
            chat_id = chat["id"]
            name = chat["name"]
            contact_id = f"wa_{chat_id}"

            async with httpx.AsyncClient(timeout=15) as client:
                try:
                    r = await client.post(
                        f"{self._bridge_url}/messages",
                        json={"chat_id": chat_id, "limit": self.scan_messages},
                    )
                    msgs = r.json().get("messages", [])
                except Exception as e:
                    print(f"  skip {name} — fetch error: {e}")
                    continue

            my_count = sum(1 for m in msgs if m["mine"])
            their_count = sum(1 for m in msgs if not m["mine"])

            # Всегда сохраняем контакт (даже без профиля) — чтобы он был в списке compose
            self.memory.set_contact(contact_id, name, {})

            if my_count < 3 or their_count < 3:
                print(f"  skip {name} — too few messages ({my_count}+{their_count}), saved to contacts")
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

        print(f"WhatsApp scan complete: {scanned} analyzed")

    # ── Review unread / recent ────────────────────────────────────────────────

    async def review_unread(self, on_approval, show_all: bool = False):
        """
        Проходит по чатам с непрочитанными (или всем последним) сообщениями.
        show_all=True — показывать все чаты, не только с unread.
        """
        print("\nЗагрузка чатов WhatsApp...")

        async with httpx.AsyncClient(timeout=10) as client:
            try:
                r = await client.get(f"{self._bridge_url}/chats-all")
                all_chats = r.json().get("chats", [])
            except Exception as e:
                print(f"  Не удалось загрузить чаты: {e}")
                return

        if not all_chats:
            print("  Чаты не найдены. Подожди синхронизации истории.")
            return

        # Фильтруем: непрочитанные или все
        if show_all:
            chats = [c for c in all_chats if not c.get("last_mine", True)][:30]
            label = f"последних чатов с входящими"
        else:
            chats = [c for c in all_chats if c.get("unread", 0) > 0]
            label = f"непрочитанных"

        if not chats:
            if not show_all:
                print("  Нет непрочитанных. Показать все последние? (да/нет)")
                # вернём управление — caller решит
                return "no_unread"
            print("  Нет чатов с входящими сообщениями.")
            return

        print(f"Найдено {len(chats)} {label}\n")

        for chat in chats:
            chat_id = chat["id"]
            name    = chat["name"]
            unread  = chat.get("unread", 0)

            # Получаем историю сообщений
            async with httpx.AsyncClient(timeout=15) as client:
                try:
                    r = await client.post(
                        f"{self._bridge_url}/messages",
                        json={"chat_id": chat_id, "limit": 20},
                    )
                    msgs = r.json().get("messages", [])
                except Exception:
                    msgs = []

            # Берём последние входящие (до 5)
            incoming_msgs = [m for m in msgs if not m.get("mine") and m.get("text")]
            if not incoming_msgs and not chat.get("last_message"):
                continue

            unread_label = f" [{unread} непрочит.]" if unread > 0 else ""
            print(f"\n{'=' * 55}")
            print(f"[WhatsApp] {name}{unread_label}")

            if incoming_msgs:
                for m in incoming_msgs[-5:]:
                    print(f"  >> {m['text'][:200]}")
                text = "\n".join(m["text"] for m in incoming_msgs[-5:])
            else:
                # Нет истории, но знаем что есть непрочитанные
                print(f"  >> (история не загружена, {unread} непрочит. сообщений)")
                continue

            reply = await on_approval(
                platform="whatsapp",
                chat_id=chat_id,
                sender_name=name,
                sender_id=chat_id,
                text=text,
            )
            if reply:
                await self.send(chat_id, reply)

    # ── Webhook ───────────────────────────────────────────────────────────────

    async def _handle_webhook(self, request: web.Request) -> web.Response:
        data = await request.json()
        await self.queue.put(data)
        return web.json_response({"ok": True})

    # ── Contacts from bridge ──────────────────────────────────────────────────

    async def get_contacts(self) -> list[tuple[str, str, dict]]:
        """
        Возвращает список (contact_id, name, profile) из моста.
        Используется в compose когда в памяти ещё нет контактов.
        """
        async with httpx.AsyncClient(timeout=10) as client:
            try:
                r = await client.get(f"{self._bridge_url}/chats-all")
                chats = r.json().get("chats", [])
            except Exception:
                return []

        result = []
        for chat in chats:
            contact_id = f"wa_{chat['id']}"
            name = chat.get("name") or chat["id"].replace("@s.whatsapp.net", "")
            # Берём профиль из памяти если есть
            mem = self.memory.get_contact(contact_id)
            profile = mem.get("profile", {}) if mem else {}
            result.append((contact_id, name, profile))

        result.sort(key=lambda x: x[1].lower())
        return result

    # ── Send ──────────────────────────────────────────────────────────────────

    async def send(self, chat_id: str, text: str):
        async with httpx.AsyncClient() as client:
            await client.post(f"{self._bridge_url}/send", json={"chat_id": chat_id, "text": text})

    async def is_bridge_up(self) -> bool:
        """Проверяет, запущен ли HTTP-сервер моста."""
        try:
            async with httpx.AsyncClient(timeout=2) as client:
                await client.get(f"{self._bridge_url}/status")
                return True
        except Exception:
            return False

    async def is_connected(self) -> bool:
        """Проверяет, подключён ли WhatsApp."""
        try:
            async with httpx.AsyncClient(timeout=2) as client:
                r = await client.get(f"{self._bridge_url}/status")
                return r.json().get("connected", False)
        except Exception:
            return False

    async def start(self):
        runner = web.AppRunner(self._app)
        await runner.setup()
        site = web.TCPSite(runner, WEBHOOK_HOST, WEBHOOK_PORT)
        await site.start()
        wa_status = "подключён ✓" if await self.is_connected() else "ожидает подключения"
        print(f"Мост: {self._bridge_url} | Вебхук: :{WEBHOOK_PORT} | WhatsApp: {wa_status}")
