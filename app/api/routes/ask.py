from fastapi import APIRouter, HTTPException, Request

from app.api.schemas import AskRequest, AskResponse, SourceItem
from app.memory.rag import VectorMemory
from app.services.search import parse_query, route, answer_search

router = APIRouter()


@router.post("/ask", response_model=AskResponse)
async def ask(body: AskRequest, request: Request):
    vm: VectorMemory = request.app.state.vector_memory

    if vm.index_size() == 0:
        raise HTTPException(status_code=503, detail="RAG index is empty. Run: python scripts/index.py")

    try:
        params = parse_query(body.query)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Query parsing failed: {e}")

    params["query"] = body.query
    intent = route(params)
    contact = params.get("contact") or ""
    only_mine = params.get("only_mine", False) if intent != "analyze" else False
    date_from = params.get("date_from")
    date_to = params.get("date_to")

    if intent == "analyze" and contact:
        msgs = vm.get_contact_messages(contact, False, date_from, date_to)
        if not msgs:
            return AskResponse(answer="Сообщений не найдено.", intent=intent, contact=contact)
        answer_text = vm.answer(body.query, msgs)
        sources = [
            SourceItem(text=m["text"], contact_name=m["contact_name"],
                       mine=m["mine"], date=m["date"], score=m.get("score", 1.0))
            for m in msgs[-5:]
        ]
        return AskResponse(answer=answer_text, intent=intent, contact=contact, sources=sources)

    k = 12 if contact else 20
    results = vm.search(
        body.query, k=k, only_mine=only_mine,
        contact_filter=contact, min_score=0.3,
        date_from=date_from, date_to=date_to,
    )
    if not results:
        return AskResponse(answer="Ничего не найдено.", intent=intent, contact=contact or None)

    answer_text = answer_search(body.query, results)
    sources = [
        SourceItem(text=r["text"], contact_name=r["contact_name"],
                   mine=r["mine"], date=r["date"], score=r.get("score", 0.0))
        for r in results[:5]
    ]
    return AskResponse(answer=answer_text, intent=intent, contact=contact or None, sources=sources)
