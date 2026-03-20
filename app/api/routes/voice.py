"""
Voice API — транскрипция голосовых сообщений и запись звонков.

Endpoints:
  POST /api/v1/voice/transcribe   — загрузить аудиофайл → получить текст
  WebSocket /ws/voice             — стриминг звонка чанками → саммари в конце

Поддерживаемые форматы: webm, mp4, m4a, ogg, wav, mp3 (всё что принимает Whisper)

Web-пример (браузер):
  const recorder = new MediaRecorder(stream)
  // ... накопить chunks ...
  const blob = new Blob(chunks, { type: 'audio/webm' })
  const fd = new FormData()
  fd.append('file', blob, 'voice.webm')
  fetch('/api/v1/voice/transcribe', { method: 'POST', body: fd })

Mobile-пример (React Native / Expo):
  const fd = new FormData()
  fd.append('file', { uri, type: 'audio/m4a', name: 'voice.m4a' })
  fetch('/api/v1/voice/transcribe', { method: 'POST', body: fd })
"""
import os
import tempfile

from fastapi import APIRouter, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from openai import AsyncOpenAI

from app.config import settings

router = APIRouter()

_ALLOWED_EXTENSIONS = {".webm", ".mp4", ".m4a", ".ogg", ".wav", ".mp3", ".oga"}
_MAX_SIZE_MB = 25  # Whisper API limit


def _openai_client() -> AsyncOpenAI:
    return AsyncOpenAI(api_key=settings.openai_key)


# ── POST /voice/transcribe ─────────────────────────────────────────────────────


@router.post("/voice/transcribe")
async def transcribe(file: UploadFile):
    """
    Принимает аудиофайл, возвращает транскрипт.

    Body: multipart/form-data  field=file
    Returns: { "text": "..." }
    """
    ext = os.path.splitext(file.filename or "")[1].lower() or ".webm"
    if ext not in _ALLOWED_EXTENSIONS:
        raise HTTPException(400, f"Unsupported format: {ext}")

    data = await file.read()
    if len(data) > _MAX_SIZE_MB * 1024 * 1024:
        raise HTTPException(413, f"File too large (max {_MAX_SIZE_MB} MB)")
    if len(data) == 0:
        raise HTTPException(400, "Empty file")

    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as f:
        f.write(data)
        tmp_path = f.name

    try:
        client = _openai_client()
        with open(tmp_path, "rb") as audio_f:
            result = await client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_f,
            )
        return JSONResponse({"text": result.text})
    except Exception as e:
        raise HTTPException(500, f"Transcription error: {e}")
    finally:
        os.unlink(tmp_path)


# ── WebSocket /ws/voice ────────────────────────────────────────────────────────


@router.websocket("/voice")
async def voice_stream(websocket: WebSocket):
    """
    WebSocket для записи звонка в реальном времени.

    Протокол:
      Клиент → сервер:
        { "type": "chunk", "data": "<base64 audio>" }   — очередной кусок аудио
        { "type": "end" }                                — конец записи

      Сервер → клиент:
        { "type": "transcript", "text": "...", "chunk": N }  — транскрипт чанка
        { "type": "summary", "text": "..." }                  — итоговое саммари
        { "type": "error", "message": "..." }

    Клиент накапливает аудио (напр. каждые 30 сек), кодирует в base64 и шлёт.
    Сервер транскрибирует каждый чанк и в конце отдаёт саммари.
    """
    import base64

    await websocket.accept()

    client = _openai_client()
    transcript_parts: list[str] = []
    chunk_index = 0

    try:
        while True:
            msg = await websocket.receive_json()
            msg_type = msg.get("type")

            if msg_type == "chunk":
                raw = base64.b64decode(msg["data"])
                if not raw:
                    continue

                ext = msg.get("ext", ".webm")
                with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as f:
                    f.write(raw)
                    tmp_path = f.name

                try:
                    with open(tmp_path, "rb") as audio_f:
                        result = await client.audio.transcriptions.create(
                            model="whisper-1", file=audio_f
                        )
                    text = result.text.strip()
                    if text:
                        transcript_parts.append(text)
                        chunk_index += 1
                        await websocket.send_json(
                            {"type": "transcript", "text": text, "chunk": chunk_index}
                        )
                except Exception as e:
                    await websocket.send_json({"type": "error", "message": str(e)})
                finally:
                    os.unlink(tmp_path)

            elif msg_type == "end":
                if not transcript_parts:
                    await websocket.send_json(
                        {"type": "summary", "text": "Речь не обнаружена."}
                    )
                    break

                full_transcript = " ".join(transcript_parts)
                summary = await _summarize(client, full_transcript)
                await websocket.send_json({"type": "summary", "text": summary})
                break

    except WebSocketDisconnect:
        pass


async def _summarize(client: AsyncOpenAI, transcript: str) -> str:
    resp = await client.chat.completions.create(
        model=settings.model,
        max_tokens=600,
        messages=[
            {
                "role": "system",
                "content": (
                    "Создай краткую сводку звонка. Выдели: ключевые темы, "
                    "принятые решения, задачи. Отвечай на том же языке, что и транскрипт."
                ),
            },
            {"role": "user", "content": f"Транскрипт:\n\n{transcript}"},
        ],
    )
    return resp.choices[0].message.content.strip()
