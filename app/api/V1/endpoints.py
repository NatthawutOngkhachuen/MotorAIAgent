from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from app.api.V1.models import SearchRequest, ChatRequest
from app.services.query_service import search_by_keyword, get_full_graph
from app.services.chat_service import stream_answer, clear_graph_cache
from app.db import postgresql as pg

router = APIRouter()


@router.post("/search")
async def search(req: SearchRequest):
    return search_by_keyword(req.keyword)


@router.get("/graph")
async def graph():
    try:
        return get_full_graph()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/graph/refresh")
async def refresh_graph():
    clear_graph_cache()
    return {"status": "cache cleared"}


@router.post("/chat")
async def chat(req: ChatRequest):
    return StreamingResponse(
        stream_answer(
            question=req.question,
            language=req.language,
            session_id=req.session_id,
            user_id=req.user_id,
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control":    "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/history/{session_id}")
async def get_history(session_id: str):
    try:
        messages = pg.load_all_messages(session_id)
        return {"session_id": session_id, "messages": messages}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/session/new")
async def new_session(user_id: str = "00000000-0000-0000-0000-000000000001"):
    try:
        session_id = pg.create_session(user_id)
        return {"session_id": session_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def health():
    try:
        pg.load_recent_messages("00000000-0000-0000-0000-000000000000", limit=1)
        return {"postgresql": "ok"}
    except Exception as e:
        return {"postgresql": f"error: {e}"}