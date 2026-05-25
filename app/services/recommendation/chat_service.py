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
from app.services.recommendation.slot_filling import SlotFillingService, SlotFillingState


load_dotenv()

GUEST_USER_ID = "00000000-0000-0000-0000-000000000001"


class UserPreferenceChatService:
    def __init__(
        self,
        slot_service: SlotFillingService | None = None,
        recommender: UserBasedRecommender | None = None,
        graph_retriever: Any | None = None,
    ):
        self.slot_service = slot_service or SlotFillingService(extractor_factory=LangChainSlotExtractor)
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
        latest_state = db["load_latest_slot_state"](session_id, user_id)
        if latest_state:
            state = SlotFillingState.from_dict(latest_state)
        else:
            state = self.slot_service.rebuild_state_from_messages(history)

        if state.is_complete:
            follow_up_question, new_state = self.slot_service.start_follow_up()
            db["save_message"](session_id, user_id, "user", question)
            yield self._event("metadata", {"stage": "slot_filling", "state": new_state.to_dict()})
            yield self._event("token", {"token": follow_up_question})
            db["save_message"](
                session_id,
                user_id,
                "assistant",
                follow_up_question,
                rag_sources=[{"source": "slot_filling", "state": new_state.to_dict()}],
            )
            db["update_session_active"](session_id)
            yield self._event("done", {"elapsed": round(time.time() - started_at, 1)})
            return

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

        yield self._event(
            "thinking",
            {"message": "ได้ครับ เดี๋ยวผมคัดรุ่นที่เข้าทางให้ อาจจะใช้เวลาสักครู่นะครับ"},
        )

        nearest: dict[str, Any] | None = None
        cluster: dict[str, Any] | None = None
        if recommendation_mode == "cluster_based":
            cluster = self.recommender.recommend_cluster(state.preferences, top_k=None)
            candidates = cluster["candidates"]
            source = "cluster_based_slot_filling"
        else:
            nearest = self.recommender.recommend_nearest_user(state.preferences, top_k=1)
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
            "analysis_label": self._analysis_label(nearest, cluster),
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
        from app.repositories.chat_repository import (
            create_session,
            load_all_messages,
            save_message,
            session_belongs_to_user,
            update_session_active,
        )
        from app.services.recommendation.slot_state_store import load_latest_slot_state

        return {
            "create_session": create_session,
            "load_all_messages": load_all_messages,
            "load_latest_slot_state": load_latest_slot_state,
            "save_message": save_message,
            "session_belongs_to_user": session_belongs_to_user,
            "update_session_active": update_session_active,
        }

    def _analysis_label(
        self,
        nearest: dict[str, Any] | None,
        cluster: dict[str, Any] | None,
    ) -> str | None:
        if nearest and nearest.get("matched_user_id"):
            return f"UID {nearest.get('matched_user_id')}"
        if cluster and cluster.get("cluster") is not None:
            return f"cluster {cluster.get('cluster')}"
        return None

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
            temperature=float(os.getenv("FINAL_RECOMMENDATION_TEMPERATURE", "0.7")),
            num_predict=int(os.getenv("FINAL_RECOMMENDATION_NUM_PREDICT", "1200")),
        )
        prompt = self._build_final_prompt(user_message, preferences, nearest, cluster, candidates, graph_evidence)

        try:
            raw_answer = await self._complete_final_answer(llm, prompt)
            answer = self._clean_model_answer(raw_answer)
            if self._needs_rewrite(raw_answer, answer):
                rewritten = await self._rewrite_final_answer(llm, prompt, raw_answer)
                answer = self._clean_model_answer(rewritten)
            if answer.strip():
                yield answer
            else:
                yield fallback
        except Exception:
            yield fallback

    async def _complete_final_answer(self, llm: Any, prompt: str) -> str:
        parts: list[str] = []
        async for chunk in llm.astream(
            [
                SystemMessage(content=self._final_answer_system_prompt()),
                HumanMessage(content=prompt),
            ]
        ):
            if chunk.content:
                parts.append(str(chunk.content))
        return "".join(parts)

    async def _rewrite_final_answer(self, llm: Any, prompt: str, raw_answer: str) -> str:
        rewrite_prompt = (
            "The previous answer was not suitable for a customer. Rewrite it in natural Thai using only "
            "the source data below. Markdown is allowed when it improves readability, including a compact table. "
            "Do not use HTML tags or raw JSON. Do not mention internal system terms.\n\n"
            "Source data:\n"
            f"{prompt}\n\n"
            "Previous answer to rewrite:\n"
            f"{raw_answer}"
        )
        parts: list[str] = []
        async for chunk in llm.astream(
            [
                SystemMessage(content=self._final_answer_system_prompt()),
                HumanMessage(content=rewrite_prompt),
            ]
        ):
            if chunk.content:
                parts.append(str(chunk.content))
        return "".join(parts)

    def _final_answer_system_prompt(self) -> str:
        return (
            "You are MotorAiAgent, a Thai motorcycle recommendation assistant. "
            "Answer in Thai like a thoughtful showroom consultant talking to one real customer. "
            "Recommend only the provided candidates. Use the provided vehicle facts as grounding, "
            "but turn them into natural advice instead of copying labels. "
            "Do not invent specs. Avoid robotic phrasing. "
            "Markdown is allowed when it makes the answer easier to read, including a compact table. "
            "Do not use HTML tags, <br>, raw JSON, or dump raw data. "
            "Do not mention GraphRAG, evidence, context, cluster, UUID, score, rank, or internal systems."
        )

    def _clean_model_answer(self, text: str) -> str:
        replacements = {
            "<br>": "\n",
            "<br/>": "\n",
            "<br />": "\n",
            "`": "",
        }
        for old, new in replacements.items():
            text = text.replace(old, new)
        lines = []
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped:
                if lines and lines[-1] != "":
                    lines.append("")
                continue
            if set(stripped) <= {"-", " "}:
                continue
            lines.append(stripped)
        return "\n".join(lines).strip()

    def _needs_rewrite(self, raw_answer: str, cleaned_answer: str) -> bool:
        raw = raw_answer.strip()
        if len(cleaned_answer.strip()) < 80:
            return True
        cleaned = cleaned_answer.strip()
        natural_endings = (".", "!", "?", "ครับ", "ค่ะ", "นะครับ", "นะคะ", ")", "]")
        if cleaned and not cleaned.endswith(natural_endings):
            return True
        blocked_fragments = ["<br", "<p", "<ul", "<li", "```"]
        if any(fragment in raw for fragment in blocked_fragments):
            return True
        internal_terms = ["GraphRAG", "evidence", "context", "cluster", "UUID", "score", "rank"]
        return any(term in cleaned_answer for term in internal_terms)

    def _build_final_prompt(
        self,
        user_message: str,
        preferences: dict[str, Any],
        nearest: dict[str, Any] | None,
        cluster: dict[str, Any] | None,
        candidates: list[dict[str, Any]],
        graph_evidence: list[dict[str, Any]],
    ) -> str:
        response_candidates = candidates if cluster else candidates[:1]
        response_item_ids = {str(candidate.get("item_id")) for candidate in response_candidates}
        response_graph_evidence = [
            item for item in self._filter_graph_evidence_for_preferences(graph_evidence, preferences)
            if str(item.get("item_id")) in response_item_ids
        ]
        if cluster:
            mode_rules = (
                "Mode: cluster-based recommendation.\n"
                "- Talk about every candidate in response_candidates. Do not skip any model.\n"
                "- A compact Markdown table is allowed and preferred if there are many models.\n"
                "- Give each model a short practical note, then close with which 1-2 models are the easiest to start comparing.\n"
            )
        else:
            mode_rules = (
                "Mode: user-based recommendation.\n"
                "- Talk only about the first response_candidate, which comes from the most similar previous user.\n"
                "- Do not discuss additional models outside this one, even if they appear elsewhere in the data.\n"
                "- Make a clear recommendation for this one model and explain why it fits the customer.\n"
            )
        context = {
            "latest_user_message": user_message,
            "preferences": preferences,
            "response_candidates": response_candidates,
            "relevant_graph_evidence": response_graph_evidence,
        }
        prompt = (
            "You are MotorAiAgent's showroom consultant. Answer in Thai for a real customer.\n\n"
            f"{mode_rules}\n"
            "Content rules:\n"
            "- Recommend only the models in response_candidates.\n"
            "- Use each candidate's brand and model exactly as provided in response_candidates. Do not infer or change brand/model pairs.\n"
            "- Use relevant_graph_evidence for grounding, but do not say evidence, context, GraphRAG, cluster, UUID, score, rank, cosine, or internal system terms.\n"
            "- Do not invent specs, prices, features, or model names that are not in the data.\n"
            "- When mentioning price ranges, write full baht numbers, for example 30,000-40,000 บาท. Do not write shortened thousand-baht ranges such as 30-40 พันบาท.\n"
            "- If a data point is not relevant to the customer's needs, skip it.\n"
            "- Markdown is allowed. Use it only to improve readability, not as a rigid report.\n"
            "- Do not use HTML tags such as <br>, <p>, <ul>, or <li>. Do not output JSON.\n\n"
            "Tone:\n"
            "- Warm, practical, and consultative like a good motorcycle salesperson.\n"
            "- Natural Thai. Avoid stiff template phrases.\n"
            "- Start with the recommendation directly, then explain briefly.\n\n"
            "Data:\n"
        )
        return prompt + json.dumps(context, ensure_ascii=False)

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
            "เหตุผลที่ผมว่าน่าเริ่มดู:",
        ]
        if reasons:
            lines.extend(f"- {reason}" for reason in reasons[:3])
        else:
            lines.append("- โดยรวมแล้วเป็นตัวเลือกที่เข้ากับโจทย์ที่ให้มา และน่าเริ่มลองเทียบฟีลขับครับ")

        alternatives = candidates[1:3]
        if alternatives:
            lines.extend(["", "ตัวเลือกที่น่าดูเพิ่ม:"])
            for candidate in alternatives:
                price = candidate.get("price_est_thb")
                price_text = f" ราคาโดยประมาณ {int(price):,} บาท" if str(price).isdigit() else ""
                item_evidence = evidence_by_item_id.get(str(candidate.get("item_id")), {})
                reason = self._short_candidate_reason(item_evidence, preferences)
                suffix = f" - {reason}" if reason else ""
                lines.append(f"- {candidate.get('brand') or ''} {candidate.get('model')}{price_text}{suffix}".strip())

        lines.extend(["", "ผมแนะนำให้ใช้รุ่นแรกเป็นตัวตั้ง แล้วลองเทียบฟีลนั่งกับตัวสำรองอีกทีครับ"])
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
            use_case = self._matching_values("use_case", evidence.get("use_case", []), preferences)
            if use_case:
                reasons.append(f"เข้ากับการใช้งานที่บอกมา เช่น {self._format_evidence_values(use_case[:3])}")

        style = self._matching_values("style", evidence.get("style", []), preferences)
        if style and preferences.get("style"):
            reasons.append(f"สไตล์ไปทาง {self._format_evidence_values(style[:3])} ซึ่งตรงกับแนวที่ชอบ")

        performance = self._matching_values("performance", evidence.get("performance", []), preferences)
        if performance and preferences.get("performance") not in {None, "unknown", ""}:
            reasons.append(f"ฟีลขับตอบโจทย์เรื่อง {self._format_evidence_values(performance[:3])}")

        comfort = self._matching_values("comfort", evidence.get("comfort", []), preferences)
        if comfort and preferences.get("comfort") not in {None, "unknown", ""}:
            reasons.append(f"เรื่องความสบายมีจุดที่น่าดู เช่น {self._format_evidence_values(comfort[:3])}")

        safety = self._matching_values("safety", evidence.get("safety", []), preferences)
        if safety and preferences.get("safety_level") not in {None, "unknown", ""}:
            reasons.append(f"ฝั่งความปลอดภัยมี {self._format_evidence_values(safety[:3])}")

        efficiency = self._matching_values("efficiency", evidence.get("efficiency", []), preferences)
        if efficiency and preferences.get("fuel_saving") is True:
            reasons.append(f"ถ้าเน้นประหยัดน้ำมัน รุ่นนี้มีจุดเด่นเรื่อง {self._format_evidence_values(efficiency[:2])}")

        storage = self._matching_values("storage", evidence.get("storage", []), preferences)
        if storage and preferences.get("storage_need") is True:
            reasons.append(f"เรื่องพื้นที่เก็บของมี {self._format_evidence_values(storage[:2])}")

        return reasons

    def _short_candidate_reason(self, item_evidence: dict[str, Any], preferences: dict[str, Any]) -> str:
        evidence = item_evidence.get("evidence", {}) if item_evidence else {}
        for key in ["use_case", "style", "comfort", "performance", "safety", "efficiency", "storage"]:
            values = self._matching_values(key, evidence.get(key, []), preferences)
            if values:
                return f"เด่นเรื่อง {self._format_evidence_values(values[:2])}"
        return ""

    def _filter_graph_evidence_for_preferences(
        self,
        graph_evidence: list[dict[str, Any]],
        preferences: dict[str, Any],
    ) -> list[dict[str, Any]]:
        filtered = []
        for item in graph_evidence:
            evidence = item.get("evidence", {}) if item else {}
            relevant_evidence: dict[str, list[Any]] = {}
            for key in ["use_case", "style", "comfort", "performance", "safety", "efficiency", "storage"]:
                values = self._matching_values(key, evidence.get(key, []), preferences)
                if values:
                    relevant_evidence[key] = values[:4]
            filtered.append(
                {
                    "item_id": item.get("item_id"),
                    "brand": item.get("brand"),
                    "model": item.get("model"),
                    "summary_text": "",
                    "evidence": relevant_evidence,
                }
            )
        return filtered

    def _matching_values(
        self,
        key: str,
        values: list[Any],
        preferences: dict[str, Any],
    ) -> list[Any]:
        if not values:
            return []
        if key == "use_case":
            allowed = set(preferences.get("usage_fit") or [])
            return self._values_matching_tokens(values, allowed)
        if key == "style":
            allowed = set(preferences.get("style") or [])
            return self._values_matching_tokens(values, allowed)
        if key == "performance":
            return self._values_matching_tokens(values, {f"performance_{preferences.get('performance')}"})
        if key == "comfort":
            return self._values_matching_tokens(values, {f"comfort_{preferences.get('comfort')}"})
        if key == "safety":
            return self._values_matching_tokens(values, {f"safety_{preferences.get('safety_level')}"})
        if key == "efficiency" and preferences.get("fuel_saving") is True:
            return list(values)
        if key == "storage" and preferences.get("storage_need") is True:
            return list(values)
        return []

    def _values_matching_tokens(self, values: list[Any], allowed_tokens: set[str]) -> list[Any]:
        allowed_tokens = {token for token in allowed_tokens if token and "unknown" not in token}
        if not allowed_tokens:
            return []
        aliases = {
            "city": ["city", "ในเมือง", "เมือง", "รถติด"],
            "daily": ["daily", "ทุกวัน", "ประจำวัน"],
            "delivery": ["delivery", "ส่งของ", "เดลิเวอรี่", "ไรเดอร์", "grab"],
            "family": ["family", "ครอบครัว", "คนซ้อน"],
            "long_distance": ["long_distance", "เดินทางไกล", "ทางไกล", "ต่างจังหวัด"],
            "rough_road": ["rough_road", "ขรุขระ", "ถนนไม่ดี", "ลุย", "ทางดิน"],
            "shopping": ["shopping", "ซื้อของ", "จ่ายตลาด"],
            "storage_heavy": ["storage_heavy", "บรรทุก", "ของเยอะ"],
            "trip": ["trip", "ทริป", "เที่ยว"],
            "work": ["work", "ทำงาน", "ใช้งาน"],
            "sporty": ["sporty", "สปอร์ต", "เท่", "วัยรุ่น", "แรง"],
            "modern": ["modern", "โมเดิร์น", "ทันสมัย"],
            "premium": ["premium", "พรีเมียม", "หรู", "ดูดี"],
            "classic": ["classic", "คลาสสิก", "วินเทจ", "ย้อนยุค", "ตำนาน"],
            "compact": ["compact", "กะทัดรัด", "คันเล็ก", "เบา", "คล่อง"],
            "adventure": ["adventure", "สายลุย", "แอดเวนเจอร์", "ลุย"],
            "beauty": ["beauty", "สวย", "แฟชั่น", "ดีไซน์"],
            "cute": ["cute", "น่ารัก"],
            "performance_low": ["performance_low", "ไม่เน้นแรง", "ขี่เรื่อย", "ขี่ช้า"],
            "performance_medium": ["performance_medium", "แรงพอประมาณ", "ขี่ทั่วไป"],
            "performance_high": ["performance_high", "อัตราเร่ง", "แรง", "เร็ว", "ออกตัวไว"],
            "comfort_low": ["comfort_low", "พื้นฐาน"],
            "comfort_medium": ["comfort_medium", "พอประมาณ", "กลาง"],
            "comfort_high": ["comfort_high", "นั่งสบาย", "สบาย"],
            "safety_low": ["safety_low", "พื้นฐาน"],
            "safety_medium": ["safety_medium", "ระดับกลาง", "กลาง"],
            "safety_high": ["safety_high", "ปลอดภัยสูง", "abs", "ความปลอดภัยสูง"],
        }
        keywords = []
        for token in allowed_tokens:
            keywords.extend(aliases.get(token, [token]))
        matches = []
        for value in values:
            text = str(value).strip().lower()
            if any(keyword.lower() in text for keyword in keywords):
                matches.append(value)
        return matches

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
        return f"จากโจทย์ที่ให้มา ผมว่า {display_name} น่าเริ่มดูที่สุดครับ"

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
