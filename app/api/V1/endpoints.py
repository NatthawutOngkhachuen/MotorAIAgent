from fastapi import APIRouter, HTTPException
from app.api.V1.models import SearchRequest, ChatRequest
from app.services.query_service import search_by_keyword, get_full_graph
from app.services.chat_service import answer_question, clear_graph_cache
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
    """ล้าง graph cache เมื่อข้อมูล Neo4j อัปเดต"""
    clear_graph_cache()
    return {"status": "cache cleared"}


@router.post("/chat")
async def chat(req: ChatRequest):
    try:
        return answer_question(
            question=req.question,
            language=req.language,
            session_id=req.session_id,
            user_id=req.user_id,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


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