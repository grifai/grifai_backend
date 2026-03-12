import json

from openai import OpenAI

from .base import LLMProvider
from .prompts import ANALYSIS_PROMPT, MY_STYLE_PROMPT, REPLY_SYSTEM


class OpenAIProvider(LLMProvider):

    def __init__(self, api_key: str, model: str = "gpt-4o-mini"):
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def generate(
        self,
        system_prompt: str,
        user_message: str,
        temperature: float = 0.7,
        max_tokens: int = 1000,
        response_format: str = "text",
    ) -> str:
        kwargs = dict(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
        )
        if response_format == "json":
            kwargs["response_format"] = {"type": "json_object"}

        resp = self.client.chat.completions.create(**kwargs)
        return resp.choices[0].message.content.strip()

    def generate_with_history(
        self,
        system_prompt: str,
        messages: list[dict],
        temperature: float = 0.7,
        max_tokens: int = 1000,
    ) -> str:
        full_messages = [{"role": "system", "content": system_prompt}] + messages
        resp = self.client.chat.completions.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=full_messages,
        )
        return resp.choices[0].message.content.strip()


# ── Module-level backwards-compatible wrapper ─────────────────────────────────
# Callers use: from app.llm import openai_provider as ai
#              ai.init_openai(key); ai.analyze_contact(...); ai.generate_reply(...)

_provider: OpenAIProvider | None = None


def init_openai(api_key: str, model: str = "gpt-4o-mini"):
    global _provider
    _provider = OpenAIProvider(api_key=api_key, model=model)


def init(api_key: str = "", model: str = ""):
    """Alias for init_openai. Pulls keys from .env if not provided."""
    from app.llm import get_llm

    global _provider
    instance = get_llm(api_key=api_key, model=model) if api_key else get_llm()
    # get_llm may return any LLMProvider; store as-is
    _provider = instance  # type: ignore[assignment]


def _get() -> OpenAIProvider:
    if _provider is None:
        raise RuntimeError("Call init_openai() before using AI functions")
    return _provider


# ── Public AI helpers ──────────────────────────────────────────────────────────


def analyze_contact(dialog_text: str, model: str = "") -> dict:
    """Deep contact analysis via LLM -> JSON profile."""
    raw = _get().generate(
        system_prompt=ANALYSIS_PROMPT,
        user_message=f"Conversation:\n\n{dialog_text}",
        temperature=0.3,
        max_tokens=800,
        response_format="json",
    )
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0]
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"raw_analysis": raw, "parse_error": True}


def analyze_my_style(all_my_msgs: list[str], model: str = "") -> str:
    """Analyze overall user style from messages across all chats."""
    sample = all_my_msgs[-80:]
    block = "\n".join(f"- {m[:200]}" for m in sample)
    return _get().generate(
        system_prompt=MY_STYLE_PROMPT,
        user_message=f"My messages from various chats:\n{block}",
        temperature=0.3,
        max_tokens=400,
    )


def generate_reply(
    sender: str,
    contact_id: str,
    incoming_batch: list[str],
    chat_context: str,
    memory,
    model: str = "",
    rag_context: str = "",
) -> str:
    system = REPLY_SYSTEM

    my_profile = memory.get_my_profile()
    if my_profile:
        system += f"\n### My general messaging style:\n{my_profile}"

    contact = memory.get_contact(contact_id)
    if contact and contact.get("profile"):
        p = contact["profile"]
        if isinstance(p, dict) and not p.get("parse_error"):
            system += f"\n\n### Relationship profile with {sender}:"
            system += f"\nRelationship: {p.get('relationship', '?')}"
            system += f"\nVibe: {p.get('vibe', '?')}"
            system += f"\nHumor: {p.get('humor', '?')}"
            ms = p.get("my_style", {})
            if ms:
                system += f"\nMy style WITH THEM: tone={ms.get('tone','?')}, length={ms.get('msg_length','?')}"
                if ms.get("phrases"):
                    system += f", phrases: {', '.join(ms['phrases'][:8])}"
                if ms.get("quirks"):
                    system += f", quirks: {ms['quirks']}"
            ts = p.get("their_style", {})
            if ts:
                system += f"\nTheir style: {ts.get('tone', '?')}"
                if ts.get("humor_markers"):
                    system += f"\nHow to tell they're joking: {ts['humor_markers']}"
            if p.get("important_context"):
                system += f"\nImportant context: {p['important_context']}"
        elif isinstance(p, dict) and p.get("raw_analysis"):
            system += f"\n\n### Analysis of {sender}:\n{p['raw_analysis'][:500]}"

    if rag_context:
        system += f"\n\n### Похожие прошлые сообщения (контекст из истории):\n{rag_context}"

    examples = memory.get_decision_examples(sender)
    if examples:
        system += f"\n{examples}"

    if len(incoming_batch) == 1:
        incoming_text = incoming_batch[0]
    else:
        incoming_text = "Batch of messages:\n" + "\n".join(f"  -> {m}" for m in incoming_batch)

    user_msg = f"Contact: {sender}\n"
    if chat_context:
        user_msg += f"\nRecent conversation:\n{chat_context}\n---\n"
    user_msg += f"\nIncoming:\n{incoming_text}\n\nMy reply:"

    return _get().generate_with_history(
        system_prompt=system,
        messages=[{"role": "user", "content": user_msg}],
        temperature=0.7,
        max_tokens=400,
    )
