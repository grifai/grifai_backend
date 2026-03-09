"""Text-to-speech service — ElevenLabs primary, OpenAI fallback."""
import re
from pathlib import Path

import httpx


# ── Text chunking helper ───────────────────────────────────────────────────────

def _chunk_text(text: str, max_chars: int) -> list[str]:
    """Split text at sentence/paragraph boundaries to stay under max_chars."""
    if len(text) <= max_chars:
        return [text]

    chunks: list[str] = []
    # Try to split on double newline (paragraphs), then single newline, then ". "
    separators = ["\n\n", "\n", ". ", "? ", "! ", " "]

    remaining = text
    while len(remaining) > max_chars:
        cut = max_chars
        for sep in separators:
            pos = remaining.rfind(sep, 0, max_chars)
            if pos > max_chars // 2:  # only if we're at least halfway through
                cut = pos + len(sep)
                break
        chunks.append(remaining[:cut].strip())
        remaining = remaining[cut:].strip()

    if remaining:
        chunks.append(remaining)
    return chunks


# ── TTSService ────────────────────────────────────────────────────────────────

class TTSService:
    """
    ElevenLabs TTS with OpenAI fallback.

    Priority:
      1. ElevenLabs  (if elevenlabs_key is set)
      2. OpenAI TTS  (if openai_key is set)
      3. None        (graceful no-op; caller skips audio)
    """

    ELEVENLABS_URL = "https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
    OPENAI_TTS_URL = "https://api.openai.com/v1/audio/speech"

    def __init__(
        self,
        elevenlabs_key: str = "",
        openai_key: str = "",
        voice_id: str = "default_russian_male",
    ):
        self.elevenlabs_key = elevenlabs_key
        self.openai_key = openai_key
        self.voice_id = voice_id

    # ── ElevenLabs ────────────────────────────────────────────────────────────

    def _elevenlabs_single(self, text: str) -> bytes:
        url = self.ELEVENLABS_URL.format(voice_id=self.voice_id)
        with httpx.Client(timeout=60) as client:
            resp = client.post(
                url,
                headers={
                    "xi-api-key": self.elevenlabs_key,
                    "Content-Type": "application/json",
                },
                json={
                    "text": text,
                    "model_id": "eleven_multilingual_v2",
                    "voice_settings": {"stability": 0.5, "similarity_boost": 0.8},
                },
            )
        resp.raise_for_status()
        return resp.content

    def text_to_speech(self, text: str) -> bytes:
        """ElevenLabs TTS. Splits text > 5000 chars into chunks and concatenates."""
        if not self.elevenlabs_key:
            raise ValueError("ELEVENLABS_KEY is not set")
        chunks = _chunk_text(text, max_chars=4900)
        return b"".join(self._elevenlabs_single(chunk) for chunk in chunks)

    # ── OpenAI TTS ────────────────────────────────────────────────────────────

    def _openai_single(self, text: str) -> bytes:
        with httpx.Client(timeout=60) as client:
            resp = client.post(
                self.OPENAI_TTS_URL,
                headers={
                    "Authorization": f"Bearer {self.openai_key}",
                    "Content-Type": "application/json",
                },
                json={"model": "tts-1", "input": text, "voice": "onyx"},
            )
        resp.raise_for_status()
        return resp.content

    def text_to_speech_openai(self, text: str) -> bytes:
        """OpenAI TTS. Splits text > 4000 chars into chunks and concatenates."""
        if not self.openai_key:
            raise ValueError("OPENAI_KEY is not set")
        chunks = _chunk_text(text, max_chars=4000)
        return b"".join(self._openai_single(chunk) for chunk in chunks)

    # ── Unified entry point with fallback ─────────────────────────────────────

    def synthesize(self, text: str) -> bytes | None:
        """
        Try ElevenLabs → OpenAI → None (never raises).
        Returns None if both providers are unavailable or fail.
        """
        if self.elevenlabs_key:
            try:
                return self.text_to_speech(text)
            except Exception as exc:
                print(f"[TTS] ElevenLabs error: {exc}; falling back to OpenAI")

        if self.openai_key:
            try:
                return self.text_to_speech_openai(text)
            except Exception as exc:
                print(f"[TTS] OpenAI TTS error: {exc}")

        return None

    # ── Save to disk ──────────────────────────────────────────────────────────

    def save_audio(self, audio_bytes: bytes, filename: str) -> str:
        """Save MP3 bytes to data/audio/{filename}.mp3. Returns absolute path."""
        audio_dir = Path("data") / "audio"
        audio_dir.mkdir(parents=True, exist_ok=True)
        # Sanitize filename
        safe = re.sub(r"[^\w\-]", "_", filename)
        path = audio_dir / f"{safe}.mp3"
        path.write_bytes(audio_bytes)
        return str(path)
