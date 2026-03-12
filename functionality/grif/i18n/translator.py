"""
i18n — language detection and locale helpers.

No LLM calls — deterministic heuristics.

Responsibilities:
  - Detect language from text (Cyrillic → ru, Latin → en, etc.)
  - Map BCP-47 codes to display names
  - Format agent output for a target locale (date formats, number formats)
  - Provide locale-aware prompt suffixes ("respond in Russian")
"""

from __future__ import annotations

import re

# ── Language detection ────────────────────────────────────────────────────────

# Script ranges for heuristic detection
_CYRILLIC = re.compile(r"[\u0400-\u04FF]")
_CJK = re.compile(r"[\u4E00-\u9FFF\u3040-\u30FF]")  # Chinese + Japanese
_ARABIC = re.compile(r"[\u0600-\u06FF]")
_DEVANAGARI = re.compile(r"[\u0900-\u097F]")

# Minimum char ratio to declare a script dominant
_SCRIPT_THRESHOLD = 0.15


def detect_language(text: str) -> str:
    """
    Heuristic language detection from script frequency.
    Returns BCP-47 language code. Defaults to 'en'.
    """
    if not text or len(text.strip()) < 3:
        return "en"

    clean = re.sub(r"\s+", "", text)
    total = len(clean)
    if total == 0:
        return "en"

    def _ratio(pattern: re.Pattern) -> float:
        return len(pattern.findall(clean)) / total

    if _ratio(_CYRILLIC) >= _SCRIPT_THRESHOLD:
        return "ru"
    if _ratio(_CJK) >= _SCRIPT_THRESHOLD:
        return "zh"
    if _ratio(_ARABIC) >= _SCRIPT_THRESHOLD:
        return "ar"
    if _ratio(_DEVANAGARI) >= _SCRIPT_THRESHOLD:
        return "hi"
    return "en"


# ── Locale metadata ───────────────────────────────────────────────────────────

_LOCALE_NAMES: dict[str, str] = {
    "ru": "Russian",
    "en": "English",
    "zh": "Chinese",
    "ar": "Arabic",
    "hi": "Hindi",
    "de": "German",
    "fr": "French",
    "es": "Spanish",
    "pt": "Portuguese",
    "it": "Italian",
    "ja": "Japanese",
    "ko": "Korean",
}

_PROMPT_SUFFIX: dict[str, str] = {
    "ru": "Отвечай на русском языке.",
    "en": "Respond in English.",
    "zh": "请用中文回答。",
    "ar": "أجب باللغة العربية.",
    "hi": "हिंदी में जवाब दें।",
    "de": "Antworte auf Deutsch.",
    "fr": "Réponds en français.",
    "es": "Responde en español.",
    "pt": "Responda em português.",
    "it": "Rispondi in italiano.",
    "ja": "日本語で答えてください。",
    "ko": "한국어로 답하세요.",
}

_DATE_FORMATS: dict[str, str] = {
    "ru": "%d.%m.%Y",
    "en": "%Y-%m-%d",
    "de": "%d.%m.%Y",
    "fr": "%d/%m/%Y",
    "zh": "%Y年%m月%d日",
    "ja": "%Y年%m月%d日",
}


def language_name(code: str) -> str:
    """Return full language name for a BCP-47 code."""
    return _LOCALE_NAMES.get(code.lower(), code)


def prompt_language_suffix(code: str) -> str:
    """Return instruction to respond in a specific language."""
    return _PROMPT_SUFFIX.get(code.lower(), f"Respond in {language_name(code)}.")


def date_format(code: str) -> str:
    """Return strftime format string for a locale."""
    return _DATE_FORMATS.get(code.lower(), "%Y-%m-%d")


def is_supported(code: str) -> bool:
    return code.lower() in _LOCALE_NAMES


def supported_languages() -> list[str]:
    return list(_LOCALE_NAMES.keys())


# ── Prompt injection ──────────────────────────────────────────────────────────

def inject_language_instruction(
    messages: list[dict],
    language: str,
    position: str = "system",
) -> list[dict]:
    """
    Append language instruction to the last system message,
    or prepend a new system message if none exists.
    """
    suffix = prompt_language_suffix(language)
    result = list(messages)

    if position == "system":
        # Find last system message and append suffix
        for i in reversed(range(len(result))):
            if result[i].get("role") == "system":
                result[i] = {
                    **result[i],
                    "content": result[i]["content"] + f"\n\n{suffix}",
                }
                return result
        # No system message — prepend one
        result.insert(0, {"role": "system", "content": suffix})

    elif position == "user":
        # Append to last user message
        for i in reversed(range(len(result))):
            if result[i].get("role") == "user":
                result[i] = {
                    **result[i],
                    "content": result[i]["content"] + f"\n\n{suffix}",
                }
                return result

    return result
