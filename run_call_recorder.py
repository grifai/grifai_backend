"""
Call recorder — записывает звонок, транскрибирует и суммирует.

Работает с любым приложением (Telegram, WhatsApp, Zoom и т.д.),
так как захватывает аудио с устройства ввода, а не из приложения.

Запуск:
    python run_call_recorder.py                         # микрофон
    python run_call_recorder.py --device "BlackHole"    # системный звук
    python run_call_recorder.py --list-devices          # список устройств

Для захвата обеих сторон звонка (macOS):
    1. brew install blackhole-2ch
    2. В «Настройки звука» выбери Multi-Output Device как выход
    3. В Audio MIDI Setup создай Aggregate Device (BlackHole + микрофон)
    4. python run_call_recorder.py --device "Aggregate Device"
"""
import argparse
import asyncio
import sys

import ai
import config
from sources.call_recorder import CallRecorder, list_devices


async def run(device=None):
    ai.init_openai(config.OPENAI_KEY)
    recorder = CallRecorder(model=config.MODEL, device=device)

    while True:
        print("\n" + "=" * 55)
        print("  Call Recorder")
        print(f"  Устройство: {device or 'микрофон по умолчанию'}")
        print("  1  Начать запись")
        print("  2  Список аудио-устройств")
        print("  q  Выход")
        print("=" * 55)

        ch = input("-> ").strip()

        if ch == "1":
            try:
                summary = await recorder.record_and_summarize()
                print("\n" + "=" * 55)
                print("СВОДКА ЗВОНКА:")
                print(summary)
                print("=" * 55)
            except RuntimeError as e:
                print(f"Ошибка: {e}")
        elif ch == "2":
            list_devices()
        elif ch in ("q", "Q"):
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Call recorder with AI summary")
    parser.add_argument("--device", default=None, help="Имя или индекс аудио-устройства")
    parser.add_argument("--list-devices", action="store_true", help="Показать устройства")
    args = parser.parse_args()

    if args.list_devices:
        list_devices()
        sys.exit(0)

    asyncio.run(run(device=args.device))
