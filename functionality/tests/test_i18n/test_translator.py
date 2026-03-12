"""Tests for i18n/translator.py — fully deterministic."""
import pytest

from grif.i18n.translator import (
    date_format,
    detect_language,
    inject_language_instruction,
    is_supported,
    language_name,
    prompt_language_suffix,
    supported_languages,
)


# ─── detect_language ──────────────────────────────────────────────────────────

def test_detect_russian() -> None:
    assert detect_language("Найди отель в Париже дешевле 150 евро") == "ru"


def test_detect_english() -> None:
    assert detect_language("Find a hotel in Paris under 150 euros") == "en"


def test_detect_short_text_defaults_to_en() -> None:
    assert detect_language("Hi") == "en"


def test_detect_empty_text_defaults_to_en() -> None:
    assert detect_language("") == "en"


def test_detect_mixed_cyrillic_latin_prefers_russian() -> None:
    # Mostly Cyrillic
    text = "Привет! Hello. Как дела? Fine thanks. Хорошо."
    lang = detect_language(text)
    assert lang == "ru"


def test_detect_whitespace_only() -> None:
    assert detect_language("   ") == "en"


def test_detect_numbers_only() -> None:
    # No script characters
    assert detect_language("12345 67890") == "en"


# ─── language_name ────────────────────────────────────────────────────────────

def test_language_name_ru() -> None:
    assert language_name("ru") == "Russian"


def test_language_name_en() -> None:
    assert language_name("en") == "English"


def test_language_name_unknown_returns_code() -> None:
    assert language_name("xx") == "xx"


def test_language_name_case_insensitive() -> None:
    assert language_name("RU") == "Russian"
    assert language_name("EN") == "English"


# ─── prompt_language_suffix ───────────────────────────────────────────────────

def test_prompt_suffix_ru() -> None:
    suffix = prompt_language_suffix("ru")
    assert "русском" in suffix.lower() or "ru" in suffix.lower()


def test_prompt_suffix_en() -> None:
    suffix = prompt_language_suffix("en")
    assert "english" in suffix.lower()


def test_prompt_suffix_unknown_language() -> None:
    suffix = prompt_language_suffix("xx")
    assert len(suffix) > 0  # Falls back gracefully


# ─── date_format ─────────────────────────────────────────────────────────────

def test_date_format_ru() -> None:
    fmt = date_format("ru")
    assert "%d" in fmt and "%m" in fmt and "%Y" in fmt


def test_date_format_en() -> None:
    assert date_format("en") == "%Y-%m-%d"


def test_date_format_unknown_defaults_to_iso() -> None:
    assert date_format("xx") == "%Y-%m-%d"


# ─── is_supported ─────────────────────────────────────────────────────────────

def test_is_supported_true_for_known() -> None:
    assert is_supported("ru") is True
    assert is_supported("en") is True
    assert is_supported("zh") is True


def test_is_supported_false_for_unknown() -> None:
    assert is_supported("xx") is False
    assert is_supported("zz") is False


# ─── supported_languages ──────────────────────────────────────────────────────

def test_supported_languages_returns_list() -> None:
    langs = supported_languages()
    assert isinstance(langs, list)
    assert len(langs) >= 5
    assert "ru" in langs
    assert "en" in langs


# ─── inject_language_instruction ──────────────────────────────────────────────

def test_inject_appends_to_existing_system_message() -> None:
    messages = [
        {"role": "system", "content": "You are GRIF."},
        {"role": "user", "content": "Hello"},
    ]
    result = inject_language_instruction(messages, "ru")
    system_content = result[0]["content"]
    assert "You are GRIF." in system_content
    # Russian instruction appended
    suffix = prompt_language_suffix("ru")
    assert suffix in system_content


def test_inject_creates_system_message_when_none() -> None:
    messages = [{"role": "user", "content": "Hello"}]
    result = inject_language_instruction(messages, "en")
    # Should have inserted a system message
    assert result[0]["role"] == "system"
    assert "English" in result[0]["content"]


def test_inject_does_not_mutate_original() -> None:
    messages = [{"role": "system", "content": "Original"}]
    original_content = messages[0]["content"]
    inject_language_instruction(messages, "ru")
    # Original list unchanged
    assert messages[0]["content"] == original_content


def test_inject_user_position_appends_to_user() -> None:
    messages = [
        {"role": "system", "content": "System"},
        {"role": "user", "content": "Find hotels"},
    ]
    result = inject_language_instruction(messages, "ru", position="user")
    user_msg = next(m for m in result if m["role"] == "user")
    assert prompt_language_suffix("ru") in user_msg["content"]
