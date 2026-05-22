from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from app.api.V1.auth_dependencies import get_current_user_id
from app.schemas.chat_schema import ChatRequest
from app.services.recommendation.chat_service import stream_recommendation_answer


router = APIRouter(tags=["recommendation-chat"])


def _stream(
    *,
    question: str,
    language: str,
    session_id: str | None,
    user_id: str,
    recommendation_mode: str,
) -> StreamingResponse:
    return StreamingResponse(
        stream_recommendation_answer(
            question=question,
            language=language,
            session_id=session_id,
            user_id=user_id,
            recommendation_mode=recommendation_mode,
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/recommendation/user-based/start")
async def start_user_based_chat(
    current_user_id: str = Depends(get_current_user_id),
):
    return _stream(
        question="",
        language="th",
        session_id=None,
        user_id=current_user_id,
        recommendation_mode="user_based",
    )


@router.post("/recommendation/cluster-based/start")
async def start_cluster_based_chat(
    current_user_id: str = Depends(get_current_user_id),
):
    return _stream(
        question="",
        language="th",
        session_id=None,
        user_id=current_user_id,
        recommendation_mode="cluster_based",
    )


@router.post("/recommendation/user-based/chat")
async def user_based_chat(
    req: ChatRequest,
    current_user_id: str = Depends(get_current_user_id),
):
    return _stream(
        question=req.question,
        language=req.language,
        session_id=req.session_id,
        user_id=current_user_id,
        recommendation_mode="user_based",
    )


@router.post("/recommendation/cluster-based/chat")
async def cluster_based_chat(
    req: ChatRequest,
    current_user_id: str = Depends(get_current_user_id),
):
    return _stream(
        question=req.question,
        language=req.language,
        session_id=req.session_id,
        user_id=current_user_id,
        recommendation_mode="cluster_based",
    )
