"""
VK source — получает сообщения через Long Poll и отправляет ответы от имени пользователя.
Требует токен пользователя (не бота) с правами messages.
"""
import asyncio
import random
import aiohttp

VK_API_URL = "https://api.vk.com/method"
VK_API_VERSION = "5.199"


def _format_dialog(msgs: list[dict]) -> str:
    lines = []
    for m in msgs:
        who = "Me" if m["mine"] else "Them"
        lines.append(f"[{who}]: {m['text'][:300]}")
    return "\n".join(lines)


class VKSource:
    def __init__(self, on_message, memory, model, token: str,
                 scan_contacts=50, scan_messages=150):
        self._on_message = on_message
        self.memory = memory
        self.model = model
        self.token = token
        self.scan_contacts = scan_contacts
        self.scan_messages = scan_messages
        self._my_id: int | None = None

    # ── VK API helper ─────────────────────────────────────────────────────────

    async def _api(self, session: aiohttp.ClientSession, method: str, **params) -> dict:
        params.update(access_token=self.token, v=VK_API_VERSION)
        async with session.get(f"{VK_API_URL}/{method}", params=params) as r:
            data = await r.json()
        if "error" in data:
            raise RuntimeError(f"VK API {method}: {data['error']['error_msg']}")
        return data["response"]

    # ── Prescan ───────────────────────────────────────────────────────────────

    async def prescan(self):
        import ai
        print(f"\nVK: scanning contacts (target: {self.scan_contacts})...")

        async with aiohttp.ClientSession() as session:
            try:
                me = await self._api(session, "users.get")
                self._my_id = me[0]["id"]
            except Exception as e:
                print(f"  Could not get VK user info: {e}")
                return

            try:
                resp = await self._api(session, "messages.getConversations",
                                       count=self.scan_contacts, filter="all")
                conversations = resp.get("items", [])
            except Exception as e:
                print(f"  Could not fetch conversations: {e}")
                return

        if not conversations:
            print("  No conversations found")
            return

        scanned = 0
        all_my_msgs: list[str] = []

        async with aiohttp.ClientSession() as session:
            for item in conversations:
                conv = item["conversation"]
                peer = conv["peer"]

                # только личные диалоги (не группы, не боты)
                if peer["type"] != "user":
                    continue

                peer_id = peer["id"]
                contact_id = f"vk_{peer_id}"

                # имя контакта
                try:
                    user_info = await self._api(session, "users.get",
                                                user_ids=peer_id, fields="first_name,last_name")
                    u = user_info[0]
                    name = f"{u['first_name']} {u['last_name']}"
                except Exception:
                    name = str(peer_id)

                # история сообщений
                try:
                    hist = await self._api(session, "messages.getHistory",
                                           peer_id=peer_id, count=self.scan_messages,
                                           rev=0)
                    raw_msgs = hist.get("items", [])
                except Exception as e:
                    print(f"  skip {name} — fetch error: {e}")
                    continue

                msgs = [
                    {"text": m["text"], "mine": m["from_id"] == self._my_id}
                    for m in raw_msgs if m.get("text")
                ]

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
            print("VK: building general style profile...", end=" ", flush=True)
            try:
                style = ai.analyze_my_style(all_my_msgs, self.model)
                self.memory.set_my_profile(style)
                print("ok")
            except Exception as e:
                print(f"error: {e}")

        print(f"VK scan complete: {scanned} analyzed")

    # ── Long Poll ─────────────────────────────────────────────────────────────

    async def _get_longpoll_server(self, session: aiohttp.ClientSession) -> dict:
        return await self._api(session, "messages.getLongPollServer", lp_version=3)

    async def _poll(self, session: aiohttp.ClientSession, server: dict):
        url = f"https://{server['server']}"
        params = {"act": "a_check", "key": server["key"],
                  "ts": server["ts"], "wait": 25, "mode": 2, "version": 3}
        async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=30)) as r:
            return await r.json()

    async def _listen(self):
        async with aiohttp.ClientSession() as session:
            server = await self._get_longpoll_server(session)

            while True:
                try:
                    data = await self._poll(session, server)
                except Exception as e:
                    print(f"VK poll error: {e}, retrying...")
                    await asyncio.sleep(3)
                    server = await self._get_longpoll_server(session)
                    continue

                failed = data.get("failed")
                if failed == 1:
                    server["ts"] = data["ts"]
                elif failed in (2, 3):
                    server = await self._get_longpoll_server(session)
                    continue

                server["ts"] = data.get("ts", server["ts"])

                for update in data.get("updates", []):
                    # тип 4 — новое сообщение
                    if update[0] != 4:
                        continue
                    flags = update[2]
                    # бит 2 = исходящее — пропускаем
                    if flags & 2:
                        continue

                    peer_id = update[3]
                    # только личные (peer_id < 2_000_000_000)
                    if peer_id >= 2_000_000_000:
                        continue

                    msg_id = update[1]
                    text = update[5]
                    if not text:
                        continue

                    asyncio.create_task(
                        self._process(session, peer_id, msg_id, text)
                    )

    async def _process(self, session: aiohttp.ClientSession,
                       peer_id: int, msg_id: int, text: str):
        contact_id = f"vk_{peer_id}"

        try:
            user_info = await self._api(session, "users.get",
                                        user_ids=peer_id, fields="first_name,last_name")
            u = user_info[0]
            sender_name = f"{u['first_name']} {u['last_name']}"
        except Exception:
            sender_name = str(peer_id)

        reply = await self._on_message(
            platform="vk",
            chat_id=str(peer_id),
            sender_name=sender_name,
            sender_id=str(peer_id),
            text=text,
        )
        if reply:
            # случайная задержка 1-4 сек, чтобы не выглядело роботом
            await asyncio.sleep(random.uniform(1, 4))
            await self.send(session, peer_id, reply)

    # ── Send ──────────────────────────────────────────────────────────────────

    async def send(self, session: aiohttp.ClientSession, peer_id: int, text: str):
        random_id = random.randint(1, 2**31)
        await self._api(session, "messages.send",
                        peer_id=peer_id, message=text, random_id=random_id)

    # ── Start ─────────────────────────────────────────────────────────────────

    async def start(self):
        async with aiohttp.ClientSession() as session:
            try:
                me = await self._api(session, "users.get")
                self._my_id = me[0]["id"]
                name = f"{me[0]['first_name']} {me[0]['last_name']}"
                print(f"VK: connected as {name} (id{self._my_id})")
            except Exception as e:
                print(f"VK: failed to connect — {e}")
                return

        asyncio.create_task(self._listen())
