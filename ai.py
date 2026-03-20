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


def _call(system: str, user: str, model: str, max_tokens: int) -> str:
    resp = _oai().chat.completions.create(
        model=model,
        max_tokens=max_tokens,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    return resp.choices[0].message.content.strip()


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

_REPLY_SYSTEM = """You are Grif, an AI assistant. You are writing a Telegram reply ON BEHALF of the user.

KEY RULES:
1. Imitate the user's style EXACTLY — phrases, length, punctuation, tone
2. Account for the RELATIONSHIP with the contact: uncle = one style, colleague = another
3. Understand CONTEXT: sarcasm, trolling, inside jokes — react appropriately
4. If the contact is joking/trolling — respond in kind, don't take it literally
5. If a batch of messages arrived — one coherent reply to all of them
6. If no reply is needed — return [SKIP]
7. Write in the same language as the contact
"""


_COMPOSE_SYSTEM = """You are Grif, an AI assistant. You are COMPOSING a new message ON BEHALF of the user (not replying — initiating).

KEY RULES:
1. Imitate the user's style EXACTLY — phrases, length, punctuation, tone
2. Account for the RELATIONSHIP with the contact
3. Express the given INTENT naturally, as the user would say it
4. Keep it natural — don't be too formal or too wordy
5. Write in the language appropriate for this contact
6. Return ONLY the message text, nothing else
"""

# ── Public API ─────────────────────────────────────────────────────────────────

def transcribe_voice(file_path: str) -> str:
    """Транскрибирует аудиофайл через OpenAI Whisper."""
    with open(file_path, "rb") as f:
        result = _oai().audio.transcriptions.create(model="whisper-1", file=f)
    return result.text


def summarize_call(transcript: str, model: str) -> str:
    """Суммирует транскрипт звонка."""
    system = (
        "Ты помощник, который создаёт краткие сводки звонков. "
        "Выдели: ключевые темы, принятые решения, задачи. "
        "Отвечай на том же языке, что и транскрипт."
    )
    return _call(system, f"Транскрипт:\n\n{transcript}", model, 600)


def analyze_contact(dialog_text: str, model: str) -> dict:
    """Deep contact analysis via LLM -> JSON profile."""
    raw = _call(_ANALYSIS_PROMPT, f"Conversation:\n\n{dialog_text}", model, 800)
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
    return _call(_MY_STYLE_PROMPT, f"My messages from various chats:\n{block}", model, 400)


def generate_reply(
    sender: str,
    contact_id: str,
    incoming_batch: list[str],
    chat_context: str,
    memory: JarvisMemory,
    model: str,
    rag_context: str = "",
    refinement: str = "",
) -> str:
    system = _REPLY_SYSTEM

    personal_prompt = memory.get_personal_prompt()
    if personal_prompt:
        system += f"\n\n### User's personal instructions:\n{personal_prompt}"

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
        incoming_text = "Batch of messages:\n" + "\n".join(
            f"  -> {m}" for m in incoming_batch
        )

    user_msg = f"Contact: {sender}\n"
    if chat_context:
        user_msg += f"\nRecent conversation:\n{chat_context}\n---\n"
    user_msg += f"\nIncoming:\n{incoming_text}\n\nMy reply:"

    if refinement:
        user_msg += f"\n\n[Refinement instruction: {refinement}]"

    return _call(system, user_msg, model, 400)


def compose_message(
    sender: str,
    contact_id: str,
    intent: str,
    memory: JarvisMemory,
    model: str,
    refinement: str = "",
) -> str:
    """Compose a new outgoing message expressing the given intent in user's style."""
    system = _COMPOSE_SYSTEM

    personal_prompt = memory.get_personal_prompt()
    if personal_prompt:
        system += f"\n\n### User's personal instructions:\n{personal_prompt}"

    my_profile = memory.get_my_profile()
    if my_profile:
        system += f"\n\n### My general messaging style:\n{my_profile}"

    contact = memory.get_contact(contact_id)
    if contact and contact.get("profile"):
        p = contact["profile"]
        if isinstance(p, dict) and not p.get("parse_error"):
            system += f"\n\n### Relationship with {sender}:"
            system += f"\nRelationship: {p.get('relationship', '?')}"
            system += f"\nVibe: {p.get('vibe', '?')}"
            ms = p.get("my_style", {})
            if ms:
                system += f"\nMy style with them: tone={ms.get('tone','?')}, length={ms.get('msg_length','?')}"
                if ms.get("phrases"):
                    system += f", phrases: {', '.join(ms['phrases'][:5])}"
                if ms.get("quirks"):
                    system += f", quirks: {ms['quirks']}"
            if p.get("important_context"):
                system += f"\nContext: {p['important_context']}"

    user_msg = f"Contact: {sender}\nIntent (what I want to say): {intent}\n\nWrite the message:"
    if refinement:
        user_msg += f"\n\n[Refinement instruction: {refinement}]"
    return _call(system, user_msg, model, 300)
