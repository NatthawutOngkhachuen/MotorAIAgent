from __future__ import annotations

import json
import os
import time
from typing import Any, AsyncGenerator

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage

from app.services.ollama_client import get_ollama_base_url, make_chat_ollama
from app.services.recommendation.langchain_slot_extractor import LangChainSlotExtractor
from app.services.recommendation.recommenders.user_based import UserBasedRecommender
from app.services.recommendation.slot_filling import SlotFillingService


load_dotenv()

GUEST_USER_ID = "00000000-0000-0000-0000-000000000001"


class UserPreferenceChatService:
    def __init__(
        self,
        slot_service: SlotFillingService | None = None,
        recommender: UserBasedRecommender | None = None,
        graph_retriever: Any | None = None,
    ):
        self.slot_service = slot_service or SlotFillingService(extractor=LangChainSlotExtractor())
        self.recommender = recommender or UserBasedRecommender()
        self.graph_retriever = graph_retriever

    async def stream(
        self,
        question: str,
        language: str = "th",
        session_id: str | None = None,
        user_id: str | None = None,
        recommendation_mode: str = "user_based",
    ) -> AsyncGenerator[str, None]:
        del language
        user_id = user_id or GUEST_USER_ID
        db = self._chat_repository()
        session_id = self._get_or_create_session(session_id, user_id)
        yield self._event("session", {"session_id": session_id})

        started_at = time.time()
        question = (question or "").strip()

        if not question:
            greeting, state = self.slot_service.start()
            yield self._event("metadata", {"stage": "slot_filling", "state": state.to_dict()})
            yield self._event("token", {"token": greeting})
            db["save_message"](
                session_id,
                user_id,
                "assistant",
                greeting,
                rag_sources=[{"source": "slot_filling", "state": state.to_dict()}],
            )
            db["update_session_active"](session_id)
            yield self._event("done", {"elapsed": round(time.time() - started_at, 1)})
            return

        history = db["load_all_messages"](session_id, user_id)
        state = self.slot_service.rebuild_state_from_messages(history)
        state, next_question = self.slot_service.handle_message(
            question,
            state,
            chat_history=history,
            session_id=session_id,
        )

        db["save_message"](session_id, user_id, "user", question)

        if next_question:
            yield self._event("metadata", {"stage": "slot_filling", "state": state.to_dict()})
            yield self._event("token", {"token": next_question})
            db["save_message"](
                session_id,
                user_id,
                "assistant",
                next_question,
                rag_sources=[{"source": "slot_filling", "state": state.to_dict()}],
            )
            db["update_session_active"](session_id)
            yield self._event("done", {"elapsed": round(time.time() - started_at, 1)})
            return

        nearest: dict[str, Any] | None = None
        cluster: dict[str, Any] | None = None
        if recommendation_mode == "cluster_based":
            cluster = self.recommender.recommend_cluster(state.preferences, top_k=5)
            candidates = cluster["candidates"]
            source = "cluster_based_slot_filling"
        else:
            nearest = self.recommender.recommend_nearest_user(state.preferences, top_k=3)
            candidates = nearest["candidates"]
            source = "user_based_slot_filling"

        item_ids = [candidate["item_id"] for candidate in candidates]

        disable_llm = os.getenv("DISABLE_RECOMMENDATION_LLM", "").lower() in {"1", "true", "yes"}

        try:
            graph_retriever = self.graph_retriever or self._graph_retriever()
            graph_evidence = graph_retriever.retrieve_by_item_ids(item_ids)
        except Exception:
            graph_evidence = []

        metadata = {
            "stage": "recommendation",
            "recommendation_mode": recommendation_mode,
            "state": state.to_dict(),
            "nearest": None if nearest is None else {
                "matched_user_id": nearest.get("matched_user_id"),
                "matched_similarity": nearest.get("matched_similarity"),
            },
            "cluster": None if cluster is None else {
                "cluster": cluster.get("cluster"),
                "cluster_similarity": cluster.get("cluster_similarity"),
                "cluster_size": cluster.get("cluster_size"),
            },
            "graph_item_ids": item_ids,
            "candidate_count": len(candidates),
            "evidence_count": len(graph_evidence),
        }
        yield self._event("metadata", metadata)

        full_answer = ""
        async for token in self._stream_final_answer(
            user_message=question,
            preferences=state.preferences,
            nearest=nearest,
            cluster=cluster,
            candidates=candidates,
            graph_evidence=graph_evidence,
            disable_llm=disable_llm,
        ):
            full_answer += token
            yield self._event("token", {"token": token})

        if not full_answer.strip():
            full_answer = self._fallback_answer(state.preferences, nearest, cluster, candidates)
            yield self._event("token", {"token": full_answer})

        db["save_message"](
            session_id,
            user_id,
            "assistant",
            full_answer,
            rag_sources=[
                {
                    "source": source,
                    "recommendation_mode": recommendation_mode,
                    "nearest": nearest,
                    "cluster": cluster,
                    "graph_item_ids": item_ids,
                    "state": state.to_dict(),
                }
            ],
        )
        db["update_session_active"](session_id)
        yield self._event("done", {"elapsed": round(time.time() - started_at, 1)})

    def _get_or_create_session(self, session_id: str | None, user_id: str) -> str:
        db = self._chat_repository()
        if session_id and db["session_belongs_to_user"](session_id, user_id):
            return session_id
        return db["create_session"](user_id)

    def _chat_repository(self) -> dict[str, Any]:
        from app.db.chat_repository import (
            create_session,
            load_all_messages,
            save_message,
            session_belongs_to_user,
            update_session_active,
        )

        return {
            "create_session": create_session,
            "load_all_messages": load_all_messages,
            "save_message": save_message,
            "session_belongs_to_user": session_belongs_to_user,
            "update_session_active": update_session_active,
        }

    def _graph_retriever(self) -> Any:
        from app.services.recommendation.graph_retriever import GraphRetriever

        return GraphRetriever()

    async def _stream_final_answer(
        self,
        user_message: str,
        preferences: dict[str, Any],
        nearest: dict[str, Any] | None,
        cluster: dict[str, Any] | None,
        candidates: list[dict[str, Any]],
        graph_evidence: list[dict[str, Any]],
        disable_llm: bool = False,
    ) -> AsyncGenerator[str, None]:
        fallback = self._fallback_answer(preferences, nearest, cluster, candidates)
        if disable_llm:
            yield fallback
            return

        llm = make_chat_ollama(
            model=os.getenv("OLLAMA_MODEL", "gpt-oss:120b"),
            base_url=get_ollama_base_url(),
            temperature=0.4,
            num_predict=int(os.getenv("FINAL_RECOMMENDATION_NUM_PREDICT", "260")),
            timeout=int(os.getenv("FINAL_RECOMMENDATION_TIMEOUT_SECONDS", "8")),
        )
        prompt = self._build_final_prompt(user_message, preferences, nearest, cluster, candidates, graph_evidence)

        try:
            async for chunk in llm.astream(
                [
                    SystemMessage(
                        content=(
                            "You are MotorAiAgent, a Thai motorcycle recommendation assistant. "
                            "Answer in Thai, naturally and concisely. Recommend only the provided candidates. "
                            "Use graph evidence when available. Do not invent specs."
                        )
                    ),
                    HumanMessage(content=prompt),
                ]
            ):
                if chunk.content:
                    yield chunk.content
        except Exception:
            yield fallback

    def _build_final_prompt(
        self,
        user_message: str,
        preferences: dict[str, Any],
        nearest: dict[str, Any] | None,
        cluster: dict[str, Any] | None,
        candidates: list[dict[str, Any]],
        graph_evidence: list[dict[str, Any]],
    ) -> str:
        context = {
            "latest_user_message": user_message,
            "preferences": preferences,
            "nearest_user_result": {
                "matched_user_id": nearest.get("matched_user_id") if nearest else None,
                "matched_similarity": nearest.get("matched_similarity") if nearest else None,
            },
            "cluster_result": None if cluster is None else {
                "cluster": cluster.get("cluster"),
                "cluster_similarity": cluster.get("cluster_similarity"),
                "cluster_size": cluster.get("cluster_size"),
            },
            "candidates": candidates,
            "graph_evidence": graph_evidence,
        }
        return (
            "สรุปคำแนะนำรถจาก context ต่อไปนี้ให้ผู้ใช้เข้าใจง่าย "
            "เริ่มด้วยรุ่นที่เหมาะที่สุด 1 รุ่น แล้วเสริมตัวเลือกอื่นได้ไม่เกิน 3 รุ่น "
            "ตอบเป็น 3 bullet เท่านั้น bullet ละไม่เกิน 1 บรรทัด ห้ามใส่รายละเอียดที่ไม่มีใน evidence\n\n"
            + json.dumps(context, ensure_ascii=False)
        )

    def _fallback_answer(
        self,
        preferences: dict[str, Any],
        nearest: dict[str, Any] | None,
        cluster: dict[str, Any] | None,
        candidates: list[dict[str, Any]],
    ) -> str:
        if not candidates:
            return "ตอนนี้ยังหารุ่นที่เหมาะจากข้อมูลผู้ใช้เดิมไม่ได้ครับ"

        first = candidates[0]
        lines = [
            self._fallback_opening(first, nearest, cluster),
            "",
            "ตัวเลือกที่ระบบแนะนำ:",
        ]
        for candidate in candidates[:3]:
            price = candidate.get("price_est_thb")
            price_text = f" ราคาโดยประมาณ {int(price):,} บาท" if str(price).isdigit() else ""
            lines.append(f"- {candidate.get('brand') or ''} {candidate.get('model')}{price_text}".strip())

        cluster_id = cluster.get("cluster") if cluster else None
        if cluster_id is not None:
            lines.append(f"\nผู้ใช้นี้ถูกจัดเข้า cluster {cluster_id} ซึ่งมีผู้ใช้เดิม {cluster.get('cluster_size')} คนครับ")
        return "\n".join(lines)

    def _fallback_opening(
        self,
        first: dict[str, Any],
        nearest: dict[str, Any] | None,
        cluster: dict[str, Any] | None,
    ) -> str:
        display_name = f"{first.get('brand') or ''} {first.get('model')}".strip()
        if nearest:
            return (
                f"จากคำตอบที่ให้มา รุ่นที่น่าเริ่มดูที่สุดคือ {display_name} "
                f"เพราะใกล้กับผู้ใช้เดิม {nearest.get('matched_user_id')} มากที่สุดครับ"
            )
        if cluster:
            return (
                f"จากคำตอบที่ให้มา รุ่นที่น่าเริ่มดูที่สุดคือ {display_name} "
                f"เพราะอยู่ในกลุ่มผู้ใช้ cluster {cluster.get('cluster')} ที่ใกล้กับคุณที่สุดครับ"
            )
        return f"จากคำตอบที่ให้มา รุ่นที่น่าเริ่มดูที่สุดคือ {display_name} ครับ"

    def _event(self, event_type: str, payload: dict[str, Any]) -> str:
        data = {"type": event_type, **payload}
        return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


async def stream_recommendation_answer(
    question: str,
    language: str = "th",
    session_id: str | None = None,
    user_id: str | None = None,
    recommendation_mode: str = "user_based",
) -> AsyncGenerator[str, None]:
    service = UserPreferenceChatService()
    async for event in service.stream(
        question=question,
        language=language,
        session_id=session_id,
        user_id=user_id,
        recommendation_mode=recommendation_mode,
    ):
        yield event
