from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from app.api.V1.auth_dependencies import get_current_user_id
from app.api.V1.models import ChatRequest
from app.services.recommendation_chat_service import stream_recommendation_answer


router = APIRouter(
    prefix="/recommendation/item-based",
    tags=["recommendation-chat"],
)


@router.post("/chat")
async def recommendation_chat(
    req: ChatRequest,
    current_user_id: str = Depends(get_current_user_id),
):
   
    return StreamingResponse(
        stream_recommendation_answer(
            question=req.question,
            language=req.language,
            session_id=req.session_id,
            user_id=current_user_id,
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
