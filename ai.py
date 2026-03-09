import json
from openai import OpenAI
from memory import JarvisMemory

_client: OpenAI | None = None


def init_openai(api_key: str):
    global _client
    _client = OpenAI(api_key=api_key)


def _oai() -> OpenAI:
    if _client is None:
        raise RuntimeError("Call init_openai() before using AI functions")
    return _client


# ── Prompts ───────────────────────────────────────────────────────────────────

_ANALYSIS_PROMPT = """You are a chat analyzer. You are given a conversation from a messenger.
"Me" is the user on whose behalf the AI will write.
"Them" is the contact.

Analyze the conversation and create a DETAILED profile. This will be used so the AI can
write on the user's behalf and the contact won't notice the difference.

Reply STRICTLY as JSON (no markdown, no ```):
{
  "relationship": "who this person is (friend/family/colleague/...) and the nature of the relationship",
  "vibe": "overall atmosphere (casual/formal/warm/bantering/...)",
  "humor": "type of humor if any (sarcasm/trolling/absurdist/memes/none), examples",
  "my_style": {
    "msg_length": "typical length of my messages",
    "tone": "tone (brief/verbose/rough/gentle/...)",
    "phrases": ["characteristic phrases and words I frequently use"],
    "punctuation": "how I write: with periods? without? caps? emoji?",
    "greeting": "how I start/end conversations with this person",
    "quirks": "specifics: abbreviations, slang, profanity, language switching"
  },
  "their_style": {
    "tone": "how the contact writes",
    "humor_markers": "signals that they are joking/trolling",
    "triggers": "what they usually expect a reply to vs just venting"
  },
  "topics": "what they usually talk about",
  "dynamics": "who initiates, who leads, any patterns",
  "important_context": "any important details: shared projects, plans, inside jokes"
}"""

_MY_STYLE_PROMPT = """Analyze ALL of my messages from different conversations and describe my GENERAL
messaging style.

Reply briefly — this will be the base profile:
- Message length
- Default tone
- Characteristic phrases and filler words
- Punctuation, caps, emoji
- Quirks (slang, profanity, language switching)
- How I open/close conversations"""

_REPLY_SYSTEM = """You are Jarvis, an AI assistant. You are writing a Telegram reply ON BEHALF of the user.

KEY RULES:
1. Imitate the user's style EXACTLY — phrases, length, punctuation, tone
2. Account for the RELATIONSHIP with the contact: uncle = one style, colleague = another
3. Understand CONTEXT: sarcasm, trolling, inside jokes — react appropriately
4. If the contact is joking/trolling — respond in kind, don't take it literally
5. If a batch of messages arrived — one coherent reply to all of them
6. If no reply is needed — return [SKIP]
7. Write in the same language as the contact
"""


# ── Public API ─────────────────────────────────────────────────────────────────

def analyze_contact(dialog_text: str, model: str) -> dict:
    """Deep contact analysis via LLM -> JSON profile."""
    resp = _oai().chat.completions.create(
        model=model,
        max_tokens=800,
        messages=[
            {"role": "system", "content": _ANALYSIS_PROMPT},
            {"role": "user", "content": f"Conversation:\n\n{dialog_text}"},
        ],
    )
    raw = resp.choices[0].message.content.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0]
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"raw_analysis": raw, "parse_error": True}


def analyze_my_style(all_my_msgs: list[str], model: str) -> str:
    """Analyze overall user style from messages across all chats."""
    sample = all_my_msgs[-80:]
    block = "\n".join(f"- {m[:200]}" for m in sample)
    resp = _oai().chat.completions.create(
        model=model,
        max_tokens=400,
        messages=[
            {"role": "system", "content": _MY_STYLE_PROMPT},
            {"role": "user", "content": f"My messages from various chats:\n{block}"},
        ],
    )
    return resp.choices[0].message.content.strip()


def generate_reply(
    sender: str,
    contact_id: str,
    incoming_batch: list[str],
    chat_context: str,
    memory: JarvisMemory,
    model: str,
    rag_context: str = "",
) -> str:
    system = _REPLY_SYSTEM

    # General style
    my_profile = memory.get_my_profile()
    if my_profile:
        system += f"\n### My general messaging style:\n{my_profile}"

    # Per-contact profile
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

    # RAG: relevant past messages
    if rag_context:
        system += f"\n\n### Похожие прошлые сообщения (контекст из истории):\n{rag_context}"

    # Past decisions
    examples = memory.get_decision_examples(sender)
    if examples:
        system += f"\n{examples}"

    # User message
    if len(incoming_batch) == 1:
        incoming_text = incoming_batch[0]
    else:
        incoming_text = "Batch of messages:\n" + "\n".join(
            f"  -> {m}" for m in incoming_batch
        )

    user_msg = f"Contact: {sender}\n"
    if chat_context:
        user_msg += f"\nRecent conversation:\n{chat_context}\n---\n"
    user_msg += f"\nIncoming:\n{incoming_text}\n\nMy reply:"

    resp = _oai().chat.completions.create(
        model=model,
        max_tokens=400,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user_msg},
        ],
    )
    return resp.choices[0].message.content.strip()
