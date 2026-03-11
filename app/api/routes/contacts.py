from fastapi import APIRouter, HTTPException, Request

from app.api.schemas import (
    ContactAIModeRequest,
    ContactMessagesResponse,
    ContactResponse,
    MessageItem,
)
from app.memory.contacts import JarvisMemory
from app.memory.rag import VectorMemory

router = APIRouter()


def _memory(request: Request) -> JarvisMemory:
    return request.app.state.memory


def _vm(request: Request) -> VectorMemory:
    return request.app.state.vector_memory


def _to_response(contact_id: str, c: dict) -> ContactResponse:
    p = c.get("profile", {})
    rel = (
        p.get("relationship")
        if isinstance(p, dict) and not p.get("parse_error")
        else None
    )
    return ContactResponse(
        contact_id=contact_id,
        name=c["name"],
        ai_mode=c.get("ai_mode", "auto"),
        relationship=rel,
        updated=c["updated"],
        profile=p if isinstance(p, dict) and not p.get("parse_error") else None,
    )


@router.get("/contacts", response_model=list[ContactResponse])
async def list_contacts(request: Request):
    memory = _memory(request)
    return [_to_response(cid, c) for cid, c in memory.data.get("contacts", {}).items()]


@router.get("/contacts/{contact_id}", response_model=ContactResponse)
async def get_contact(contact_id: str, request: Request):
    memory = _memory(request)
    contacts = memory.data.get("contacts", {})
    if contact_id not in contacts:
        raise HTTPException(status_code=404, detail="Contact not found")
    return _to_response(contact_id, contacts[contact_id])


@router.patch("/contacts/{contact_id}/ai-mode", response_model=ContactResponse)
async def set_ai_mode(contact_id: str, body: ContactAIModeRequest, request: Request):
    if body.mode not in ("auto", "never", "ask"):
        raise HTTPException(
            status_code=400, detail="mode must be 'auto', 'never', or 'ask'"
        )
    memory = _memory(request)
    contacts = memory.data.get("contacts", {})
    if contact_id not in contacts:
        raise HTTPException(status_code=404, detail="Contact not found")
    memory.set_contact_ai_mode(contact_id, body.mode)
    return _to_response(contact_id, contacts[contact_id])


@router.get("/contacts/{contact_id}/messages", response_model=ContactMessagesResponse)
async def get_contact_messages(
    contact_id: str,
    request: Request,
    limit: int = 50,
    date_from: str | None = None,
):
    memory = _memory(request)
    contacts = memory.data.get("contacts", {})
    if contact_id not in contacts:
        raise HTTPException(status_code=404, detail="Contact not found")
    name = contacts[contact_id]["name"]

    vm = _vm(request)
    msgs = vm.get_contact_messages(
        contact_filter=name,
        date_from=date_from,
        max_messages=limit,
    )
    items = [
        MessageItem(
            text=m["text"],
            mine=m["mine"],
            date=m["date"],
            contact_name=m["contact_name"],
        )
        for m in msgs
    ]
    return ContactMessagesResponse(
        contact_id=contact_id,
        name=name,
        messages=items,
        total=len(items),
    )


@router.post("/contacts/{contact_id}/analyze", response_model=ContactResponse)
async def analyze_contact(contact_id: str, request: Request):
    memory = _memory(request)
    contacts = memory.data.get("contacts", {})
    if contact_id not in contacts:
        raise HTTPException(status_code=404, detail="Contact not found")
    name = contacts[contact_id]["name"]

    vm = _vm(request)
    msgs = vm.get_contact_messages(contact_filter=name, max_messages=500)
    if not msgs:
        raise HTTPException(
            status_code=422, detail=f"No indexed messages found for '{name}'"
        )

    lines = []
    for m in msgs[-200:]:
        who = "Me" if m["mine"] else "Them"
        lines.append(f"[{who}]: {m['text'][:300]}")
    dialog_text = "\n".join(lines)

    from app.llm import openai_provider as ai

    profile = ai.analyze_contact(dialog_text)

    memory.set_contact(contact_id, name, profile)
    return _to_response(contact_id, memory.data["contacts"][contact_id])
