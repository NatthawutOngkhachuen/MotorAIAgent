import json
import time
from typing import AsyncGenerator

from app.db.chat_repository import (
    create_session,
    load_recent_messages,
    save_message,
    session_belongs_to_user,
    update_session_active,
)
from app.services.recommendation.graph_retriever import GraphRetriever
from app.services.recommendation.response_generator import ResponseGenerator
from app.services.recommendation.router import RecommendationRouter


GUEST_USER_ID = "00000000-0000-0000-0000-000000000001"


async def stream_recommendation_answer(
    question: str,
    language: str = "th",
    session_id: str | None = None,
    user_id: str | None = None,
) -> AsyncGenerator[str, None]:

    user_id = user_id or GUEST_USER_ID

    if session_id and not session_belongs_to_user(session_id, user_id):
        session_id = None

    if not session_id:
        session_id = create_session(user_id)

    yield f"data: {json.dumps({'type': 'session', 'session_id': session_id}, ensure_ascii=False)}\n\n"

    start_time = time.time()

    route_result = None
    graph_evidence = []
    full_answer = ""

    try:

        recent_messages = load_recent_messages(session_id, limit=6)

        router = RecommendationRouter()
        graph_retriever = GraphRetriever()
        response_generator = ResponseGenerator()

        route_result = router.route(
            user_message=question,
            top_k=3,
        )

        if route_result.graph_item_ids:
            graph_evidence = graph_retriever.retrieve_by_item_ids(
                item_ids=route_result.graph_item_ids,
            )

        metadata = {
            "type": "metadata",
            "route": route_result.route,
            "response_type": route_result.response_type,
            "graph_item_ids": route_result.graph_item_ids,
            "candidate_count": len(route_result.candidates),
            "evidence_count": len(graph_evidence),
        }
        yield f"data: {json.dumps(metadata, ensure_ascii=False)}\n\n"

        async for token in response_generator.astream(
            user_message=question,
            route_result=route_result,
            graph_evidence=graph_evidence,
        ):
            full_answer += token
            yield f"data: {json.dumps({'type': 'token', 'token': token}, ensure_ascii=False)}\n\n"

        completion_tail = response_generator.complete_if_truncated(
            current_answer=full_answer,
            route_result=route_result,
            graph_evidence=graph_evidence,
        )
        if completion_tail:
            full_answer += completion_tail
            yield f"data: {json.dumps({'type': 'token', 'token': completion_tail}, ensure_ascii=False)}\n\n"

        elapsed = round(time.time() - start_time, 1)

        yield f"data: {json.dumps({'type': 'done', 'elapsed': elapsed}, ensure_ascii=False)}\n\n"

        rag_sources = [
            {
                "source": "motor_ai_agent",
                "route": route_result.route if route_result else None,
                "response_type": route_result.response_type if route_result else None,
                "graph_item_ids": route_result.graph_item_ids if route_result else [],
                "candidates": route_result.candidates if route_result else [],
                "evidence_models": [
                    {
                        "item_id": item.get("item_id"),
                        "brand": item.get("brand"),
                        "model": item.get("model"),
                    }
                    for item in graph_evidence
                ],
            }
        ]

        save_message(session_id, user_id, "user", question)
        save_message(
            session_id,
            user_id,
            "assistant",
            full_answer,
            rag_sources=rag_sources,
        )
        update_session_active(session_id)

    except Exception as e:
        error_message = f"เกิดข้อผิดพลาดในการประมวลผลคำถามครับ: {str(e)}"

        yield f"data: {json.dumps({'type': 'error', 'message': error_message}, ensure_ascii=False)}\n\n"

        try:
            save_message(session_id, user_id, "user", question)
            save_message(
                session_id,
                user_id,
                "assistant",
                error_message,
                rag_sources=[{"source": "motor_ai_agent", "error": str(e)}],
            )
            update_session_active(session_id)
        except Exception:
            pass
