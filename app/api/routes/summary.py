import io
from collections import defaultdict
from datetime import datetime, timedelta

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from app.api.schemas import SummaryResponse
from app.llm.prompts import DIGEST_PROMPT
from app.memory.rag import VectorMemory

router = APIRouter()


async def _build_summary(request: Request, hours: int) -> SummaryResponse:
    vm: VectorMemory = request.app.state.vector_memory
    llm = request.app.state.llm

    date_from = (datetime.now() - timedelta(hours=hours)).strftime("%Y-%m-%dT%H:%M:%S")
    msgs = vm.get_contact_messages(date_from=date_from, max_messages=3000)

    if not msgs:
        return SummaryResponse(
            summary=f"За последние {hours} часов активных переписок не найдено.",
            dialogs_count=0,
            hours=hours,
            generated_at=datetime.now().isoformat(),
        )

    by_contact: dict[str, list[dict]] = defaultdict(list)
    for m in msgs:
        by_contact[m["contact_name"]].append(m)

    blocks = []
    for contact, contact_msgs in sorted(by_contact.items()):
        sorted_msgs = sorted(contact_msgs, key=lambda x: x.get("date", ""))
        lines = [f"=== {contact} ==="]
        for m in sorted_msgs[-20:]:
            who = "Я" if m["mine"] else contact
            lines.append(f"[{who}]: {m['text'][:200]}")
        blocks.append("\n".join(lines))

    full_text = "\n\n".join(blocks)
    if len(full_text) > 14000:
        full_text = full_text[:14000] + "\n...(обрезано)"

    summary_text = llm.generate(
        system_prompt=DIGEST_PROMPT.format(hours=hours),
        user_message=full_text,
        temperature=0.5,
        max_tokens=1200,
    )

    return SummaryResponse(
        summary=summary_text,
        dialogs_count=len(by_contact),
        hours=hours,
        generated_at=datetime.now().isoformat(),
    )


@router.get("/summary", response_model=SummaryResponse)
async def get_summary(request: Request, hours: int = 24):
    return await _build_summary(request, hours)


@router.post("/summary/generate", response_model=SummaryResponse)
async def generate_summary(request: Request, hours: int = 24):
    """Force-generate a fresh summary."""
    return await _build_summary(request, hours)


@router.get("/summary/audio")
async def get_summary_audio(request: Request, hours: int = 24):
    """Generate a digest and return it as an MP3 audio stream."""
    tts = getattr(request.app.state, "tts", None)
    if tts is None:
        raise HTTPException(
            status_code=503,
            detail="TTS not configured — set ELEVENLABS_KEY or ensure OPENAI_KEY is present",
        )

    summary = await _build_summary(request, hours)

    audio = tts.synthesize(summary.summary)
    if audio is None:
        raise HTTPException(status_code=503, detail="TTS synthesis failed")

    filename = f"digest_{hours}h_{datetime.now().strftime('%Y%m%d_%H%M')}.mp3"
    return StreamingResponse(
        io.BytesIO(audio),
        media_type="audio/mpeg",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )
