"""
Call recorder — записывает микрофон или системный звук,
транскрибирует через Whisper и суммирует.

Зависимости:
    pip install sounddevice scipy numpy

Для захвата ОБЕИХ сторон звонка (macOS):
    1. brew install blackhole-2ch
    2. Audio MIDI Setup → создай Multi-Output Device (BlackHole + колонки)
    3. Audio MIDI Setup → создай Aggregate Device (BlackHole + микрофон)
    4. Передай имя агрегатного устройства: CallRecorder(device="Aggregate Device")

На Linux аналогично через PulseAudio Monitor (device="pulse_monitor").
"""
import asyncio
import os
import tempfile

try:
    import numpy as np
    import scipy.io.wavfile as wavfile
    import sounddevice as sd
    _DEPS_OK = True
except ImportError:
    _DEPS_OK = False

import ai

SAMPLE_RATE = 16000  # Whisper работает оптимально на 16 kHz


class CallRecorder:
    def __init__(self, model: str, device=None):
        """
        model  — LLM-модель для суммаризации
        device — имя или индекс аудио-устройства (None = микрофон по умолчанию)
        """
        self.model = model
        self.device = device
        self._recording = False
        self._frames: list = []
        self._stream = None

    def _callback(self, indata, frames, time_info, status):
        if self._recording:
            self._frames.append(indata.copy())

    def start(self):
        if not _DEPS_OK:
            raise RuntimeError(
                "Установи зависимости: pip install sounddevice scipy numpy"
            )
        self._frames = []
        self._recording = True
        self._stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="int16",
            device=self.device,
            callback=self._callback,
        )
        self._stream.start()

    def stop(self) -> "np.ndarray":
        self._recording = False
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        if not self._frames:
            return np.array([], dtype=np.int16)
        return np.concatenate(self._frames, axis=0)

    async def record_and_summarize(self) -> str:
        """Запись до нажатия Enter → транскрипция → суммаризация."""
        loop = asyncio.get_event_loop()

        self.start()
        device_label = self.device or "микрофон по умолчанию"
        print(f"Запись с устройства: {device_label}")
        print("Нажми Enter чтобы остановить...", flush=True)
        await loop.run_in_executor(None, input)

        audio = self.stop()
        if audio.size == 0:
            return "Аудио не записано."

        duration = len(audio) / SAMPLE_RATE
        print(f"Записано {duration:.1f} сек. Транскрибирую...")

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            tmp_path = f.name

        try:
            wavfile.write(tmp_path, SAMPLE_RATE, audio)
            transcript = await loop.run_in_executor(
                None, lambda: ai.transcribe_voice(tmp_path)
            )
            if not transcript.strip():
                return "Транскрипт пустой — речь не обнаружена."

            print(f"\nТранскрипт:\n{transcript}\n")
            print("Суммирую...")
            summary = await loop.run_in_executor(
                None, lambda: ai.summarize_call(transcript, self.model)
            )
            return summary
        finally:
            os.unlink(tmp_path)


def list_devices():
    if not _DEPS_OK:
        print("sounddevice не установлен. Запусти: pip install sounddevice scipy numpy")
        return
    print(sd.query_devices())
