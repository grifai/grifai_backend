"""
Max messenger source — получает сообщения через Bot API и отправляет ответы.
Установка: pip install maxapi
Документация: https://github.com/max-messenger/max-botapi-python

Требует Bot Token от MasterBot в Max (аналог @BotFather в Telegram).
"""
import asyncio

from maxapi import Bot, Dispatcher
from maxapi.types import MessageCreated


class MaxSource:
    def __init__(self, token: str, on_message, memory, model: str,
                 scan_contacts: int = 50, scan_messages: int = 150):
        self.token = token
        self._on_message = on_message
        self.memory = memory
        self.model = model

        self._bot = Bot(token)
        self._dp = Dispatcher()
        self._register_handlers()

    # ── Handlers ──────────────────────────────────────────────────────────────

    def _register_handlers(self):
        @self._dp.message_created()
        async def on_message(event: MessageCreated):
            text = getattr(event.message, "text", None)
            if not text:
                return

            chat_id = str(event.message.recipient.chat_id)
            sender = event.message.sender
            sender_id = str(sender.user_id) if sender else chat_id
            sender_name = getattr(sender, "name", None) or sender_id

            contact_id = f"max_{sender_id}"
            if self.memory.get_contact_ai_mode(contact_id) == "never":
                return

            reply = await self._on_message(
                platform="max",
                chat_id=chat_id,
                sender_name=sender_name,
                sender_id=sender_id,
                text=text,
            )
            if reply:
                await self.send(chat_id, reply)

    # ── Send ──────────────────────────────────────────────────────────────────

    async def send(self, chat_id: str, text: str):
        await self._bot.send_message(chat_id=int(chat_id), text=text)

    # ── Prescan ───────────────────────────────────────────────────────────────

    async def prescan(self):
        # Bot API Max не предоставляет доступ к истории чатов —
        # профили контактов будут строиться по мере поступления сообщений.
        print("Max: prescan недоступен через Bot API (история чатов закрыта)")
        print("Max: профили контактов накапливаются по мере общения")

    # ── Start ─────────────────────────────────────────────────────────────────

    async def start(self):
        me = await self._bot.get_me()
        name = getattr(me, "name", "unknown")
        print(f"Max: connected as {name}")
        asyncio.create_task(self._dp.start_polling(self._bot))
