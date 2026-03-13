const makeWASocket = require('@whiskeysockets/baileys').default
const { useMultiFileAuthState, DisconnectReason, fetchLatestBaileysVersion, makeInMemoryStore } = require('@whiskeysockets/baileys')
const express = require('express')
const qrcode = require('qrcode-terminal')
const { Boom } = require('@hapi/boom')
const pino = require('pino')

const app = express()
app.use(express.json())

const PORT = process.env.WA_PORT || 3001
const PYTHON_WEBHOOK = process.env.PYTHON_WEBHOOK || 'http://localhost:8765/whatsapp'

let sock = null
let isConnected = false
const logger = pino({ level: 'silent' })

// In-memory store для хранения чатов и сообщений
const store = makeInMemoryStore({ logger })

// ── WhatsApp connection ────────────────────────────────────────────────────────

async function connect() {
    const { state, saveCreds } = await useMultiFileAuthState('whatsapp_session')
    const { version } = await fetchLatestBaileysVersion()

    sock = makeWASocket({
        version,
        auth: state,
        printQRInTerminal: false,
        logger,
        getMessage: async (key) => {
            const msg = await store.loadMessage(key.remoteJid, key.id)
            return msg?.message || undefined
        },
    })

    store.bind(sock.ev)
    sock.ev.on('creds.update', saveCreds)

    sock.ev.on('connection.update', ({ connection, lastDisconnect, qr }) => {
        if (qr) {
            console.log('\n📱 Scan QR code in WhatsApp (Linked Devices):')
            qrcode.generate(qr, { small: true })
        }

        if (connection === 'open') {
            isConnected = true
            console.log('✅ WhatsApp connected')
        }

        if (connection === 'close') {
            isConnected = false
            const code = new Boom(lastDisconnect?.error)?.output?.statusCode
            const shouldReconnect = code !== DisconnectReason.loggedOut
            console.log('❌ Disconnected, code:', code)
            if (shouldReconnect) {
                console.log('🔄 Reconnecting...')
                setTimeout(connect, 3000)
            } else {
                console.log('Logged out. Delete whatsapp_session/ and restart.')
            }
        }
    })

    sock.ev.on('messages.upsert', async ({ messages, type }) => {
        if (type !== 'notify') return

        for (const msg of messages) {
            if (!msg.message || msg.key.fromMe) continue

            const text =
                msg.message.conversation ||
                msg.message.extendedTextMessage?.text ||
                null

            if (!text) continue

            const jid = msg.key.remoteJid
            if (jid.endsWith('@g.us')) continue  // только личные

            const sender = jid.replace('@s.whatsapp.net', '')
            const pushName = msg.pushName || sender

            console.log(`[WA] ${pushName}: ${text}`)

            try {
                await fetch(PYTHON_WEBHOOK, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        platform: 'whatsapp',
                        chat_id: jid,
                        sender_id: sender,
                        sender_name: pushName,
                        text,
                        timestamp: Date.now(),
                    }),
                })
            } catch (e) {
                console.error('Failed to forward to Python:', e.message)
            }
        }
    })
}

// ── HTTP API for Python ────────────────────────────────────────────────────────

app.get('/status', (req, res) => {
    res.json({ connected: isConnected })
})

// Список личных чатов
app.get('/chats', (req, res) => {
    if (!isConnected) return res.status(503).json({ error: 'Not connected' })

    const chats = Object.values(store.chats.all())
        .filter(c => c.id.endsWith('@s.whatsapp.net'))
        .map(c => ({
            id: c.id,
            name: c.name || c.id.replace('@s.whatsapp.net', ''),
        }))

    res.json({ chats })
})

// История сообщений конкретного чата
app.post('/messages', async (req, res) => {
    const { chat_id, limit = 150 } = req.body
    if (!sock || !isConnected) return res.status(503).json({ error: 'Not connected' })

    try {
        const msgs = await sock.fetchMessageHistory(limit, { key: { remoteJid: chat_id } }, new Date())
        const result = []

        for (const msg of (msgs || [])) {
            const text =
                msg.message?.conversation ||
                msg.message?.extendedTextMessage?.text ||
                null
            if (!text) continue
            result.push({
                text,
                mine: msg.key.fromMe,
                date: new Date(Number(msg.messageTimestamp) * 1000).toISOString(),
                sender: msg.pushName || msg.key.participant || '',
            })
        }

        res.json({ messages: result })
    } catch (e) {
        // fetchMessageHistory может быть недоступен — возвращаем из store
        const stored = store.messages[chat_id]
        const result = []
        if (stored) {
            for (const msg of stored.array.slice(-limit)) {
                const text =
                    msg.message?.conversation ||
                    msg.message?.extendedTextMessage?.text ||
                    null
                if (!text) continue
                result.push({
                    text,
                    mine: msg.key.fromMe,
                    date: new Date(Number(msg.messageTimestamp) * 1000).toISOString(),
                    sender: msg.pushName || '',
                })
            }
        }
        res.json({ messages: result })
    }
})

// Отправить сообщение
app.post('/send', async (req, res) => {
    const { chat_id, text } = req.body
    if (!sock || !isConnected) return res.status(503).json({ error: 'Not connected' })
    try {
        await sock.sendMessage(chat_id, { text })
        res.json({ ok: true })
    } catch (e) {
        res.status(500).json({ error: e.message })
    }
})

app.listen(PORT, () => {
    console.log(`WhatsApp bridge listening on port ${PORT}`)
    connect()
})
