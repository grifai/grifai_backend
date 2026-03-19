/**
 * WhatsApp Bridge — Baileys + Express
 *
 * Endpoints:
 *   GET  /status          — статус подключения
 *   GET  /chats           — список чатов
 *   POST /messages        — история сообщений { chat_id, limit }
 *   POST /send            — отправить сообщение { chat_id, text }
 */

const {
  default: makeWASocket,
  useMultiFileAuthState,
  fetchLatestBaileysVersion,
  DisconnectReason,
  makeCacheableSignalKeyStore,
  isRealMessage,
} = require("@whiskeysockets/baileys");

const express = require("express");
const axios = require("axios");
const pino = require("pino");
const path = require("path");
const fs = require("fs");

// ── Config ────────────────────────────────────────────────────────────────────

const PORT_START  = parseInt(process.env.BRIDGE_PORT || "3001");
const PORT_RANGE  = [PORT_START, PORT_START+1, PORT_START+2, PORT_START+3, PORT_START+4];
const WEBHOOK_URL = process.env.WEBHOOK_URL  || "http://localhost:8765/whatsapp";
const SESSION_DIR = path.join(__dirname, "whatsapp_session");
const STORE_FILE  = path.join(__dirname, "whatsapp_session", "store.json");
const PORT_FILE   = path.join(__dirname, "whatsapp_session", "port.txt");
const MAX_MSGS_PER_CHAT = 200;

// ── State ─────────────────────────────────────────────────────────────────────

let sock              = null;
let isConnected       = false;
let myJid             = null;
let disconnectTimer   = null;

// chat_id → [ { id, text, mine, timestamp } ]
const messageStore = new Map();

// chat_id → name (populated from chats.set / messages)
const chatNames = new Map();

// chat_id → unread count
const unreadCounts = new Map();

// ── Persist store to disk ─────────────────────────────────────────────────────

function loadStore() {
  try {
    if (fs.existsSync(STORE_FILE)) {
      const data = JSON.parse(fs.readFileSync(STORE_FILE, "utf8"));
      for (const [id, msgs] of Object.entries(data.messages || {})) {
        messageStore.set(id, msgs);
      }
      for (const [id, name] of Object.entries(data.names || {})) {
        chatNames.set(id, name);
      }
      console.log(`Загружено ${messageStore.size} чатов из кэша`);
    }
  } catch (e) {
    console.log("Не удалось загрузить кэш:", e.message);
  }
}

function saveStore() {
  try {
    const data = {
      messages: Object.fromEntries(messageStore),
      names:    Object.fromEntries(chatNames),
    };
    fs.writeFileSync(STORE_FILE, JSON.stringify(data));
  } catch (e) { /* ignore */ }
}

// Сохраняем каждые 30 секунд
setInterval(saveStore, 30_000);

// ── Helpers ───────────────────────────────────────────────────────────────────

function extractText(msg) {
  return (
    msg.message?.conversation ||
    msg.message?.extendedTextMessage?.text ||
    msg.message?.imageMessage?.caption ||
    msg.message?.videoMessage?.caption ||
    ""
  );
}

function storeMessage(chatId, entry) {
  if (!messageStore.has(chatId)) messageStore.set(chatId, []);
  const buf = messageStore.get(chatId);
  // Дедупликация по id
  if (entry.id && buf.some(m => m.id === entry.id)) return;
  buf.push(entry);
  if (buf.length > MAX_MSGS_PER_CHAT) buf.shift();
}

// ── WhatsApp connection ───────────────────────────────────────────────────────

async function connectToWhatsApp() {
  loadStore();

  const { state, saveCreds } = await useMultiFileAuthState(SESSION_DIR);
  const { version } = await fetchLatestBaileysVersion();

  sock = makeWASocket({
    version,
    auth: {
      creds: state.creds,
      keys: makeCacheableSignalKeyStore(state.keys, pino({ level: "silent" })),
    },
    logger: pino({ level: "silent" }),
    syncFullHistory: true,   // запрашиваем историю при подключении
    getMessage: async (key) => {
      // Нужно Baileys для повторной доставки
      const msgs = messageStore.get(key.remoteJid) || [];
      const found = msgs.find(m => m.id === key.id);
      return found ? { conversation: found.text } : undefined;
    },
  });

  sock.ev.on("creds.update", saveCreds);

  // ── Соединение ──────────────────────────────────────────────────────────────
  sock.ev.on("connection.update", ({ connection, lastDisconnect, qr }) => {
    if (qr) {
      console.log("\nОтсканируй QR-код: откройте WhatsApp → Настройки → Связанные устройства → Привязать\n");
      // Печатаем QR вручную
      require("qrcode-terminal").generate(qr, { small: true });
    }
    if (connection === "open") {
      if (disconnectTimer) { clearTimeout(disconnectTimer); disconnectTimer = null; }
      isConnected = true;
      myJid = sock.user?.id;
      console.log(`WhatsApp: подключён ✓ (${sock.user?.name || myJid})`);
    }
    if (connection === "close") {
      const code = lastDisconnect?.error?.output?.statusCode;
      if (code === DisconnectReason.loggedOut) {
        isConnected = false;
        console.log("Сессия закрыта. Удали папку whatsapp_session и перезапусти.");
      } else {
        // Кратковременный реконнект при синхронизации истории — не сбрасываем сразу
        disconnectTimer = setTimeout(() => { isConnected = false; }, 5000);
        console.log("WhatsApp: переподключение...");
        setTimeout(connectToWhatsApp, 3000);
      }
    }
  });

  // ── Имена чатов ─────────────────────────────────────────────────────────────
  sock.ev.on("chats.set", ({ chats }) => {
    for (const chat of chats) {
      const name = chat.name || chatNames.get(chat.id) || chat.id.replace("@s.whatsapp.net", "").replace("@g.us", "Group");
      chatNames.set(chat.id, name);
      if (chat.unreadCount > 0) unreadCounts.set(chat.id, chat.unreadCount);
    }
    console.log(`Получено ${chats.length} чатов (${unreadCounts.size} непрочитанных)`);
  });

  sock.ev.on("chats.update", (updates) => {
    for (const update of updates) {
      if (update.name) chatNames.set(update.id, update.name);
      if (update.unreadCount !== undefined) {
        if (update.unreadCount > 0) unreadCounts.set(update.id, update.unreadCount);
        else unreadCounts.delete(update.id);
      }
    }
  });

  sock.ev.on("contacts.set", ({ contacts }) => {
    for (const c of contacts) {
      const name = c.notify || c.name || chatNames.get(c.id) || c.id.replace("@s.whatsapp.net", "");
      chatNames.set(c.id, name);
    }
  });

  sock.ev.on("contacts.upsert", (contacts) => {
    for (const c of contacts) {
      const name = c.notify || c.name || chatNames.get(c.id) || c.id.replace("@s.whatsapp.net", "");
      chatNames.set(c.id, name);
    }
  });

  // ── Сообщения (новые + история) ─────────────────────────────────────────────
  sock.ev.on("messages.upsert", async ({ messages, type }) => {
    for (const msg of messages) {
      if (!msg.message) continue;
      if (!isRealMessage(msg)) continue;

      const chatId = msg.key.remoteJid;
      if (!chatId || chatId.endsWith("@broadcast")) continue;

      const text = extractText(msg);
      if (!text) continue;

      const isMe = msg.key.fromMe;
      const entry = {
        id:        msg.key.id,
        text,
        mine:      isMe,
        timestamp: Number(msg.messageTimestamp),
      };

      storeMessage(chatId, entry);

      // Имя из pushName
      if (!isMe && msg.pushName) {
        chatNames.set(chatId, msg.pushName);
      }

      // Пересылаем на Python только новые входящие
      if (type === "notify" && !isMe) {
        const senderName = chatNames.get(chatId) ||
          chatId.replace("@s.whatsapp.net", "").replace("@g.us", "Group");
        try {
          await axios.post(WEBHOOK_URL, {
            chat_id:     chatId,
            sender_id:   chatId,
            sender_name: senderName,
            text,
          });
        } catch (_) { /* Python может не быть запущен */ }
      }
    }
  });

  // ── История через sync ───────────────────────────────────────────────────────
  sock.ev.on("messaging-history.set", ({ messages, chats, contacts, isLatest }) => {
    // Сохраняем имена
    for (const c of contacts || []) {
      const name = c.notify || c.name || chatNames.get(c.id) || c.id.replace("@s.whatsapp.net", "");
      chatNames.set(c.id, name);
    }
    for (const chat of chats || []) {
      const name = chat.name || chatNames.get(chat.id) || chat.id.replace("@s.whatsapp.net", "").replace("@g.us", "Group");
      chatNames.set(chat.id, name);
    }
    // Сохраняем сообщения
    let count = 0;
    for (const msg of messages || []) {
      if (!msg.message) continue;
      const chatId = msg.key?.remoteJid;
      if (!chatId || chatId.endsWith("@broadcast")) continue;
      const text = extractText(msg);
      if (!text) continue;
      storeMessage(chatId, {
        id:        msg.key.id,
        text,
        mine:      msg.key.fromMe,
        timestamp: Number(msg.messageTimestamp),
      });
      count++;
    }
    if (count > 0) {
      console.log(`История: загружено ${count} сообщений из ${messageStore.size} чатов`);
      saveStore();
    }
  });
}

// ── REST API ──────────────────────────────────────────────────────────────────

const app = express();
app.use(express.json());

app.get("/status", (_req, res) => {
  res.json({ connected: isConnected, chats: messageStore.size });
});

app.get("/chats", (_req, res) => {
  const chats = [];
  for (const [id, msgs] of messageStore.entries()) {
    if (msgs.length === 0) continue;
    const sorted = [...msgs].sort((a, b) => b.timestamp - a.timestamp);
    const last   = sorted[0];
    const name   = chatNames.get(id) ||
      id.replace("@s.whatsapp.net", "").replace("@g.us", "Group");
    chats.push({ id, name, last_message: last.text, timestamp: last.timestamp });
  }
  chats.sort((a, b) => b.timestamp - a.timestamp);
  res.json({ chats });
});

// Все известные чаты (из WhatsApp roster), с флагом unread и наличием истории
app.get("/chats-all", (_req, res) => {
  const result = [];

  // Берём объединение: chatNames (из roster) + messageStore (из истории)
  const allIds = new Set([...chatNames.keys(), ...messageStore.keys()]);

  for (const id of allIds) {
    if (id.endsWith("@broadcast") || id.endsWith("@g.us")) continue;
    const msgs   = messageStore.get(id) || [];
    const sorted = [...msgs].sort((a, b) => b.timestamp - a.timestamp);
    const last   = sorted[0];
    const name   = chatNames.get(id) || id.replace("@s.whatsapp.net", "");
    result.push({
      id,
      name,
      unread:       unreadCounts.get(id) || 0,
      has_history:  msgs.length > 0,
      last_message: last?.text || "",
      last_mine:    last?.mine ?? null,
      timestamp:    last?.timestamp || 0,
    });
  }

  // Сортировка: сначала непрочитанные, потом по времени
  result.sort((a, b) => {
    if (a.unread > 0 && b.unread === 0) return -1;
    if (b.unread > 0 && a.unread === 0) return 1;
    return b.timestamp - a.timestamp;
  });

  res.json({ chats: result });
});

app.post("/messages", (req, res) => {
  const { chat_id, limit = 150 } = req.body;
  const all  = messageStore.get(chat_id) || [];
  const msgs = [...all].sort((a, b) => a.timestamp - b.timestamp).slice(-limit);
  res.json({ messages: msgs });
});

app.post("/send", async (req, res) => {
  const { chat_id, text } = req.body;
  if (!isConnected) return res.status(503).json({ error: "Not connected" });
  try {
    const sent = await sock.sendMessage(chat_id, { text });
    storeMessage(chat_id, {
      id:        sent.key.id,
      text,
      mine:      true,
      timestamp: Math.floor(Date.now() / 1000),
    });
    res.json({ ok: true });
  } catch (e) {
    res.status(500).json({ error: e.message });
  }
});

// ── Start ─────────────────────────────────────────────────────────────────────

function tryListen(ports) {
  if (ports.length === 0) {
    console.error(`Все порты заняты (${PORT_RANGE.join(", ")}). Укажи свободный через BRIDGE_PORT=XXXX`);
    process.exit(1);
  }
  const port = ports[0];
  const server = app.listen(port, () => {
    console.log(`WhatsApp bridge запущен на порту ${port}`);
    // Сохраняем выбранный порт чтобы Python мог его прочитать
    fs.mkdirSync(SESSION_DIR, { recursive: true });
    fs.writeFileSync(PORT_FILE, String(port));
  });
  server.on("error", (err) => {
    if (err.code === "EADDRINUSE") {
      console.log(`Порт ${port} занят, пробую ${ports[1]}...`);
      tryListen(ports.slice(1));
    } else {
      throw err;
    }
  });
}

tryListen(PORT_RANGE);
connectToWhatsApp();
