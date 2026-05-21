from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from app.api.V1.auth_dependencies import get_current_user_id
from app.schemas.chat_schema import ChatRequest
from app.services.chat_service import stream_answer
from app.repositories import chat_repository as pg
from app.services.query_service import search_by_keyword, get_full_graph

router = APIRouter()


@router.post("/chat")
async def chat(req: ChatRequest, current_user_id: str = Depends(get_current_user_id)):
    return StreamingResponse(
        stream_answer(
            question=req.question,
            language=req.language,
            session_id=req.session_id,
            user_id=current_user_id,
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control":    "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/history/{session_id}")
async def get_history(
    session_id: str,
    current_user_id: str = Depends(get_current_user_id),
):
    try:
        if not pg.session_belongs_to_user(session_id, current_user_id):
            raise HTTPException(status_code=404, detail="Session not found")
        messages = pg.load_all_messages(session_id, current_user_id)
        return {"session_id": session_id, "messages": messages}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/sessions")
async def list_sessions(current_user_id: str = Depends(get_current_user_id)):
    try:
        return {
            "sessions": pg.list_sessions_by_user(current_user_id),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/session/{session_id}")
async def delete_session(
    session_id: str,
    current_user_id: str = Depends(get_current_user_id),
):
    try:
        deleted = pg.delete_session_for_user(session_id, current_user_id)
        if not deleted:
            raise HTTPException(status_code=404, detail="Session not found")
        return {"deleted": True, "session_id": session_id}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/graph")
async def graph():
    try:
        return get_full_graph()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
