"""Ghost-writing service — generates Telegram replies on behalf of the user."""
import json
from datetime import datetime

from pydantic import BaseModel

from app.llm.base import LLMProvider
from app.llm.prompts import STYLE_PROFILE_PROMPT, REPLY_SYSTEM_PROMPT
from app.memory.rag import VectorMemory, format_rag_context
from app.memory.contacts import JarvisMemory


class StyleProfile(BaseModel):
    contact_name: str = ""
    address: str = ""
    msg_length: str = ""
    emoji_freq: str = ""
    emoji_list: list[str] = []
    stickers: bool = False
    punctuation: str = ""
    capitalization: str = ""
    slang: list[str] = []
    humor: str = ""
    language: str = ""
    greeting: str = ""
    farewell: str = ""
    built_at: str = ""
    examples_count: int = 0


class ReplyDraft(BaseModel):
    text: str
    confidence: float = 0.8
    reasoning: str = ""


class GhostWriter:
    def __init__(
        self,
        llm: LLMProvider,
        vector_memory: VectorMemory,
        contact_store: JarvisMemory,
    ):
        self.llm = llm
        self.vm = vector_memory
        self.store = contact_store

    # ── Style profile ──────────────────────────────────────────────────────────

    def build_style_profile(self, contact_id: str) -> StyleProfile:
        contact = self.store.data.get("contacts", {}).get(contact_id, {})
        contact_name = contact.get("name", contact_id)

        # Collect message examples: approved pairs + recent my RAG messages
        approved = self.store.get_contact_examples(contact_id, n=20)
        examples_lines: list[str] = []

        for ex in approved:
            inc = ex.get("incoming", "")[:120]
            rep = ex.get("reply", "")[:120]
            examples_lines.append(f'Им: "{inc}" → Я: "{rep}"')

        # Also pull last 50 of my own messages for this contact from RAG
        rag_msgs = self.vm.get_contact_messages(
            contact_filter=contact_name, only_mine=True, max_messages=50
        )
        for m in rag_msgs[-30:]:  # keep latest 30 to avoid huge prompts
            examples_lines.append(f"Я: {m.get('text', '')[:100]}")

        if not examples_lines:
            # Not enough data — return a minimal default profile
            return StyleProfile(
                contact_name=contact_name,
                built_at=datetime.now().isoformat(),
            )

        examples_text = "\n".join(examples_lines[:60])
        prompt = STYLE_PROFILE_PROMPT.format(
            contact_name=contact_name,
            examples=examples_text,
        )

        raw = self.llm.generate(
            system_prompt="Ты — аналитик переписок. Возвращай ТОЛЬКО валидный JSON.",
            user_message=prompt,
            temperature=0.2,
            max_tokens=600,
        )

        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            # Try to extract JSON from the response
            start = raw.find("{")
            end = raw.rfind("}") + 1
            try:
                data = json.loads(raw[start:end]) if start != -1 else {}
            except json.JSONDecodeError:
                data = {}

        profile = StyleProfile(
            contact_name=contact_name,
            built_at=datetime.now().isoformat(),
            examples_count=len(approved),
            **{k: v for k, v in data.items() if k in StyleProfile.model_fields},
        )
        self.store.set_contact_style_profile(contact_id, profile.model_dump())
        return profile

    def _get_or_build_style_profile(self, contact_id: str) -> StyleProfile:
        raw = self.store.get_contact_style_profile(contact_id)
        if raw:
            return StyleProfile(**raw)
        return self.build_style_profile(contact_id)

    # ── Reply generation ───────────────────────────────────────────────────────

    def generate_reply(
        self,
        contact_id: str,
        incoming_messages: list[str],
        chat_context: str,
    ) -> ReplyDraft:
        contact = self.store.data.get("contacts", {}).get(contact_id, {})
        contact_name = contact.get("name", contact_id)

        # a) Load style profile
        style = self._get_or_build_style_profile(contact_id)
        style_text = _format_style_profile(style)

        # b) Few-shot examples (up to 5 approved pairs)
        examples = self.store.get_contact_examples(contact_id, n=5)
        examples_block = _format_examples(examples) if examples else "(нет примеров пока)"

        # c) RAG: search relevant history
        query = " ".join(incoming_messages)
        rag_results = self.vm.search(
            query, k=6, min_score=0.35, contact_filter=contact_name
        )
        rag_context = format_rag_context(rag_results, max_chars=600) or "(нет релевантной истории)"

        # d) Build prompt
        system = REPLY_SYSTEM_PROMPT.format(
            contact_name=contact_name,
            style_profile=style_text,
            examples_block=examples_block,
            rag_context=rag_context,
            chat_context=chat_context or "(нет контекста)",
            incoming_messages="\n".join(f"- {t}" for t in incoming_messages),
        )

        text = self.llm.generate(
            system_prompt=system,
            user_message="Напиши ответ.",
            temperature=0.7,
            max_tokens=300,
        ).strip()

        confidence = 0.5 if text == "[SKIP]" else 0.85
        return ReplyDraft(text=text, confidence=confidence)

    # ── Learning from approvals ────────────────────────────────────────────────

    def learn_from_approval(
        self, contact_id: str, incoming: str, approved_reply: str
    ) -> None:
        self.store.add_contact_example(contact_id, incoming, approved_reply)

        examples = self.store.get_contact_examples(contact_id, n=20)
        # Rebuild style profile every 10 new examples
        if len(examples) > 0 and len(examples) % 10 == 0:
            self.build_style_profile(contact_id)


# ── Formatting helpers ─────────────────────────────────────────────────────────

def _format_style_profile(p: StyleProfile) -> str:
    if not p.msg_length and not p.address:
        return "(профиль ещё не построен)"
    parts = []
    if p.address:
        parts.append(f"Обращение: {p.address}")
    if p.msg_length:
        parts.append(f"Длина: {p.msg_length}")
    if p.emoji_freq:
        emoji_str = f"{p.emoji_freq}"
        if p.emoji_list:
            emoji_str += f" ({', '.join(p.emoji_list[:5])})"
        parts.append(f"Emoji: {emoji_str}")
    if p.punctuation:
        parts.append(f"Пунктуация: {p.punctuation}")
    if p.capitalization:
        parts.append(f"Регистр: {p.capitalization}")
    if p.slang:
        parts.append(f"Сленг: {', '.join(p.slang[:8])}")
    if p.humor:
        parts.append(f"Юмор: {p.humor}")
    if p.language:
        parts.append(f"Язык: {p.language}")
    if p.greeting:
        parts.append(f"Начинает: {p.greeting}")
    if p.farewell:
        parts.append(f"Прощается: {p.farewell}")
    return "\n".join(parts)


def _format_examples(examples: list[dict]) -> str:
    if not examples:
        return "(нет примеров)"
    lines = []
    for ex in examples:
        inc = ex.get("incoming", "")[:100]
        rep = ex.get("reply", "")[:100]
        lines.append(f'Им: "{inc}" → Ты: "{rep}"')
    return "\n".join(lines)
