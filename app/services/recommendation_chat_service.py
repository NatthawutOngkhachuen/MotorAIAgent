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
            full_answer = self._fallback_answer(state.preferences, nearest, cluster, candidates, graph_evidence)
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
        fallback = self._fallback_answer(preferences, nearest, cluster, candidates, graph_evidence)
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
                            "Answer in Thai with a warm, practical, consultative tone. "
                            "Recommend only the provided candidates. Use graph evidence explicitly when available. "
                            "Do not invent specs. Avoid robotic phrasing."
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
            "ให้เหมือนผู้ช่วยที่เข้าใจบริบท ไม่ใช่รายงานจากระบบ "
            "เริ่มด้วยประโยคสั้นๆ ว่ารุ่นไหนน่าเริ่มดูที่สุดและเพราะอะไร "
            "จากนั้นให้เหตุผล 2-3 ข้อโดยอ้างอิง graph_evidence เช่น เหมาะกับการใช้งาน สไตล์ ความสบาย ความปลอดภัย หรือความประหยัด "
            "ปิดท้ายด้วยตัวเลือกสำรองไม่เกิน 2 รุ่นถ้ามีเหตุผลจาก evidence รองรับ "
            "ห้ามพูดว่าระบบ/GraphRAG/evidence และห้ามใส่รายละเอียดที่ไม่มีใน evidence\n\n"
            + json.dumps(context, ensure_ascii=False)
        )

    def _fallback_answer(
        self,
        preferences: dict[str, Any],
        nearest: dict[str, Any] | None,
        cluster: dict[str, Any] | None,
        candidates: list[dict[str, Any]],
        graph_evidence: list[dict[str, Any]] | None = None,
    ) -> str:
        if not candidates:
            return "ตอนนี้ยังหารุ่นที่เหมาะจากข้อมูลผู้ใช้เดิมไม่ได้ครับ"

        first = candidates[0]
        evidence_by_item_id = {
            str(item.get("item_id")): item for item in graph_evidence or []
        }
        first_evidence = evidence_by_item_id.get(str(first.get("item_id")), {})
        reasons = self._fallback_evidence_reasons(first_evidence, preferences)
        lines = [
            self._fallback_opening(first, nearest, cluster),
            "",
            "เหตุผลที่รุ่นนี้เข้าทางคุณ:",
        ]
        if reasons:
            lines.extend(f"- {reason}" for reason in reasons[:3])
        else:
            lines.append("- โปรไฟล์ความต้องการของคุณใกล้กับผู้ใช้เดิมที่เลือกรุ่นนี้ จึงเป็นรุ่นที่ควรเริ่มลองดูครับ")

        alternatives = candidates[1:3]
        if alternatives:
            lines.extend(["", "ตัวเลือกที่น่าดูเพิ่ม:"])
            for candidate in alternatives:
                price = candidate.get("price_est_thb")
                price_text = f" ราคาโดยประมาณ {int(price):,} บาท" if str(price).isdigit() else ""
                item_evidence = evidence_by_item_id.get(str(candidate.get("item_id")), {})
                reason = self._short_candidate_reason(item_evidence)
                suffix = f" - {reason}" if reason else ""
                lines.append(f"- {candidate.get('brand') or ''} {candidate.get('model')}{price_text}{suffix}".strip())

        lines.extend(["", "ผมแนะนำให้ใช้รุ่นแรกเป็นจุดเริ่มต้น แล้วค่อยเทียบฟีลนั่งกับตัวเลือกสำรองครับ"])

        cluster_id = cluster.get("cluster") if cluster else None
        if cluster_id is not None:
            lines.append(f"\nหมายเหตุ: ความต้องการนี้ใกล้กับกลุ่มผู้ใช้ cluster {cluster_id} ที่มีผู้ใช้เดิม {cluster.get('cluster_size')} คนครับ")
        return "\n".join(lines)

    def _fallback_evidence_reasons(
        self,
        item_evidence: dict[str, Any],
        preferences: dict[str, Any],
    ) -> list[str]:
        evidence = item_evidence.get("evidence", {}) if item_evidence else {}
        reasons: list[str] = []

        usage = set(preferences.get("usage_fit") or [])
        if usage:
            use_case = evidence.get("use_case", [])
            if use_case:
                reasons.append(f"การใช้งานที่ระบุมาเข้ากับจุดเด่นเรื่อง {self._format_evidence_values(use_case[:3])}")

        style = evidence.get("style", [])
        if style and preferences.get("style"):
            reasons.append(f"โทนรถไปทาง {self._format_evidence_values(style[:3])} ซึ่งใกล้กับสไตล์ที่คุณบอก")

        performance = evidence.get("performance", [])
        if performance and preferences.get("performance") not in {None, "unknown", ""}:
            reasons.append(f"ด้านการขับขี่มีข้อมูลเด่นเรื่อง {self._format_evidence_values(performance[:3])}")

        comfort = evidence.get("comfort", [])
        if comfort and preferences.get("comfort") not in {None, "unknown", ""}:
            reasons.append(f"เรื่องความสบายมีจุดที่น่าดูคือ {self._format_evidence_values(comfort[:3])}")

        safety = evidence.get("safety", [])
        if safety and preferences.get("safety_level") not in {None, "unknown", ""}:
            reasons.append(f"ฝั่งความปลอดภัยมี {self._format_evidence_values(safety[:3])}")

        efficiency = evidence.get("efficiency", [])
        if efficiency and preferences.get("fuel_saving") is True:
            reasons.append(f"ถ้าอยากประหยัดน้ำมัน รุ่นนี้มีข้อมูลเรื่อง {self._format_evidence_values(efficiency[:2])}")

        storage = evidence.get("storage", [])
        if storage and preferences.get("storage_need") is True:
            reasons.append(f"เรื่องพื้นที่เก็บของมี {self._format_evidence_values(storage[:2])}")

        if not reasons:
            summary = item_evidence.get("summary_text")
            if summary:
                first_line = str(summary).splitlines()[0]
                reasons.append(first_line)

        return reasons

    def _short_candidate_reason(self, item_evidence: dict[str, Any]) -> str:
        evidence = item_evidence.get("evidence", {}) if item_evidence else {}
        for key in ["use_case", "style", "comfort", "performance", "safety", "efficiency", "storage"]:
            values = evidence.get(key, [])
            if values:
                return f"เด่นเรื่อง {self._format_evidence_values(values[:2])}"
        return ""

    def _format_evidence_values(self, values: list[Any]) -> str:
        label_map = {
            "city": "ขี่ในเมือง",
            "daily": "ใช้งานทุกวัน",
            "delivery": "ส่งของ",
            "family": "ใช้กับครอบครัว",
            "long_distance": "เดินทางไกล",
            "rough_road": "ถนนขรุขระ",
            "shopping": "ซื้อของ/จ่ายตลาด",
            "storage_heavy": "บรรทุกของเยอะ",
            "trip": "ออกทริป",
            "work": "ไปทำงาน",
            "sporty": "สปอร์ต",
            "premium": "พรีเมียม",
            "classic": "คลาสสิก",
            "modern": "โมเดิร์น",
            "compact": "กะทัดรัด",
            "adventure": "สายลุย",
            "beauty": "ดีไซน์สวย",
            "cute": "น่ารัก",
            "performance_low": "ขี่เรื่อยๆ ไม่เน้นแรง",
            "performance_medium": "แรงพอประมาณ",
            "performance_high": "อัตราเร่งดี",
            "comfort_low": "ความสบายพื้นฐาน",
            "comfort_medium": "นั่งสบายพอประมาณ",
            "comfort_high": "นั่งสบาย",
            "safety_low": "ความปลอดภัยพื้นฐาน",
            "safety_medium": "ความปลอดภัยระดับกลาง",
            "safety_high": "ความปลอดภัยสูง",
            "fuel_saving": "ประหยัดน้ำมัน",
            "storage_need": "มีพื้นที่เก็บของ",
            "easy_to_ride": "ขี่ง่าย",
            "maintenance_easy": "ดูแลง่าย",
        }
        formatted = []
        for value in values:
            token = str(value).strip()
            if not token:
                continue
            formatted.append(label_map.get(token, token.replace("_", " ")))
        return ", ".join(formatted)

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
