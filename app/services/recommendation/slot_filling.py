from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from app.services.recommendation.vectorizer import (
    BOOL_FEATURES,
    LEVEL_FEATURES,
    empty_preference_state,
    merge_preferences,
    preference_to_vector,
)


QUESTION_FLOW = [
    (
        ["usage_fit"],
        "สวัสดีครับ ผมช่วยแนะนำรถที่เหมาะกับคุณได้ เดี๋ยวขอถามสั้นๆ ประมาณ 4-5 ข้อก่อนนะครับ ปกติจะใช้รถแบบไหนบ้าง เช่น ขี่ในเมือง ใช้ทุกวัน ไปทำงาน ออกทริป ส่งของ หรือเจอถนนขรุขระ?",
    ),
    (
        ["style"],
        "ชอบสไตล์รถประมาณไหนครับ เช่น สปอร์ต พรีเมียม คลาสสิก โมเดิร์น น่ารัก กะทัดรัด หรือสายลุย?",
    ),
    (
        ["performance", "comfort"],
        "เวลาเร่งแซงหรือออกตัว อยากได้ประมาณไหนครับ: ขอแค่ขี่เรื่อยๆ ก็พอ, ขอแรงพอประมาณ, หรืออยากได้แรงๆ และเวลานั่งอยากได้สบายมากไหมครับ?",
    ),
    (
        ["safety_level", "easy_to_ride"],
        "ให้ความสำคัญกับความปลอดภัยมากแค่ไหนครับ ต่ำ กลาง หรือสูง และอยากได้รถที่ขี่ง่ายเป็นพิเศษไหมครับ?",
    ),
    (
        ["fuel_saving", "storage_need"],
        "อยากได้รถที่ประหยัดน้ำมันเป็นพิเศษไหมครับ และต้องการพื้นที่เก็บของเยอะประมาณไหน?",
    ),
]


CONVERSATION_DEFAULT_ZERO_SLOTS = {
    "budget_level": "unknown",
    "technology_level": "unknown",
    "maintenance_easy": "unknown",
}


YES_WORDS = {"ใช่", "เอา", "ต้องการ", "โอเค", "ok", "yes", "y", "ครับ", "ค่ะ", "ได้"}
NO_WORDS = {"ไม่", "ไม่เอา", "ไม่ต้องการ", "no", "n", "ไม่ครับ", "ไม่ค่ะ"}


@dataclass
class SlotFillingState:
    preferences: dict[str, Any] = field(default_factory=empty_preference_state)
    asked_slots: list[str] = field(default_factory=list)
    last_asked_slots: list[str] = field(default_factory=list)
    is_complete: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "preferences": self.preferences,
            "asked_slots": self.asked_slots,
            "last_asked_slots": self.last_asked_slots,
            "is_complete": self.is_complete,
        }


class SlotFillingService:
    def __init__(self, extractor: Any | None = None):
        self.extractor = extractor

    def start(self) -> tuple[str, SlotFillingState]:
        state = SlotFillingState()
        question = QUESTION_FLOW[0][1]
        state.last_asked_slots = list(QUESTION_FLOW[0][0])
        state.asked_slots = list(QUESTION_FLOW[0][0])
        return question, state

    def handle_message(
        self,
        user_message: str,
        state: SlotFillingState | None = None,
        chat_history: list[dict[str, Any]] | None = None,
        session_id: str = "recommendation-session",
    ) -> tuple[SlotFillingState, str | None]:
        state = state or SlotFillingState()
        extracted = self.extract_preferences(
            user_message,
            state.last_asked_slots,
            chat_history=chat_history or [],
            current_preferences=state.preferences,
            session_id=session_id,
        )
        state.preferences = merge_preferences(state.preferences, extracted)
        self._apply_conversation_defaults(state.preferences)
        state.is_complete = self.is_complete(state.preferences)

        if state.is_complete:
            return state, None

        next_slots, question = self.next_question(state.preferences, state.asked_slots)
        state.last_asked_slots = next_slots
        for slot in next_slots:
            if slot not in state.asked_slots:
                state.asked_slots.append(slot)
        return state, question

    def rebuild_state_from_messages(self, messages: list[dict[str, Any]]) -> SlotFillingState:
        question, state = self.start()
        del question

        for message in messages:
            role = message.get("role")
            content = message.get("content") or ""
            if role == "assistant":
                slots = self._slots_from_assistant_question(content)
                if slots:
                    state.last_asked_slots = slots
                    for slot in slots:
                        if slot not in state.asked_slots:
                            state.asked_slots.append(slot)
            elif role == "user":
                state, _ = self.handle_message(content, state)

        state.is_complete = self.is_complete(state.preferences)
        self._apply_conversation_defaults(state.preferences)
        return state

    def extract_preferences(
        self,
        text: str,
        last_asked_slots: list[str] | None = None,
        chat_history: list[dict[str, Any]] | None = None,
        current_preferences: dict[str, Any] | None = None,
        session_id: str = "recommendation-session",
    ) -> dict[str, Any]:
        text_norm = text.strip().lower()
        result: dict[str, Any] = {
            "usage_fit": [],
            "style": [],
        }

        self._extract_yes_no_answer(text_norm, last_asked_slots or [], result)
        self._extract_budget(text_norm, result)
        self._extract_usage(text_norm, result)
        self._extract_style(text_norm, result)
        self._extract_levels(text_norm, result)
        self._extract_booleans(text_norm, result)

        if self.extractor is None:
            return result

        try:
            langchain_result = self.extractor.extract(
                user_message=text,
                chat_history=chat_history or [],
                current_preferences=current_preferences or empty_preference_state(),
                last_asked_slots=last_asked_slots or [],
                session_id=session_id,
            )
        except Exception:
            return result

        return merge_preferences(langchain_result, result)

    def next_question(
        self,
        preferences: dict[str, Any],
        asked_slots: list[str] | None = None,
    ) -> tuple[list[str], str]:
        asked_slots = asked_slots or []
        for slots, question in QUESTION_FLOW:
            if not self._slots_complete(preferences, slots):
                return list(slots), question

        return [], ""

    def is_complete(self, preferences: dict[str, Any]) -> bool:
        return all(self._slots_complete(preferences, slots) for slots, _ in QUESTION_FLOW)

    def build_vector(self, preferences: dict[str, Any]) -> list[float]:
        cleaned = dict(preferences or {})
        self._apply_conversation_defaults(cleaned)
        return preference_to_vector(cleaned)

    def _apply_conversation_defaults(self, preferences: dict[str, Any]) -> None:
        preferences.update(CONVERSATION_DEFAULT_ZERO_SLOTS)

    def _slots_complete(self, preferences: dict[str, Any], slots: list[str]) -> bool:
        for slot in slots:
            value = preferences.get(slot)
            if slot in {"usage_fit", "style"}:
                if not value:
                    return False
            elif value in [None, "", "unknown", []]:
                return False
        return True

    def _extract_yes_no_answer(
        self,
        text: str,
        last_asked_slots: list[str],
        result: dict[str, Any],
    ) -> None:
        compact = re.sub(r"\s+", "", text)
        is_yes = compact in YES_WORDS or any(word in text for word in ["ใช่", "เอา", "ต้องการ", "yes"])
        is_no = compact in NO_WORDS or any(word in text for word in ["ไม่เอา", "ไม่ต้องการ"])
        if not is_yes and not is_no:
            return

        value = is_yes and not is_no
        for slot in last_asked_slots:
            if slot in BOOL_FEATURES:
                result[slot] = value

    def _extract_budget(self, text: str, result: dict[str, Any]) -> None:
        if any(word in text for word in ["ไม่แพง", "ถูก", "งบน้อย", "งบไม่สูง", "ราคาประหยัด", "low"]):
            result["budget_level"] = "low"
        elif any(word in text for word in ["กลาง", "พอประมาณ", "ไม่ถูกไม่แพง", "medium"]):
            result["budget_level"] = "medium"
        elif any(word in text for word in ["งบสูง", "ตัวท็อป", "แพงได้", "พรีเมียม", "high"]):
            result["budget_level"] = "high"

    def _extract_usage(self, text: str, result: dict[str, Any]) -> None:
        mapping = {
            "city": ["ในเมือง", "ขี่เมือง", "รถติด", "ซอกแซก", "ไปตลาด"],
            "daily": ["ใช้ทุกวัน", "ขี่ทุกวัน", "ไปทำงาน", "ไปเรียน", "ประจำวัน"],
            "delivery": ["ส่งของ", "เดลิเวอรี่", "delivery", "grab", "ไรเดอร์"],
            "family": ["ครอบครัว", "คนซ้อน", "รับส่งลูก"],
            "long_distance": ["เดินทางไกล", "ทางไกล", "ต่างจังหวัด", "ไกล"],
            "rough_road": ["ถนนขรุขระ", "ถนนไม่ดี", "ลุย", "rough", "ทางดิน"],
            "shopping": ["ช้อป", "shopping", "ซื้อของ", "จ่ายตลาด"],
            "storage_heavy": ["บรรทุก", "ของเยอะ", "สัมภาระเยอะ"],
            "trip": ["ออกทริป", "ทริป", "เที่ยว"],
            "work": ["ทำงาน", "วิ่งงาน", "ใช้งาน"],
        }
        self._append_matches(text, result["usage_fit"], mapping)

    def _extract_style(self, text: str, result: dict[str, Any]) -> None:
        mapping = {
            "adventure": ["สายลุย", "แอดเวนเจอร์", "adventure", "ลุย"],
            "beauty": ["สวย", "แฟชั่น", "ดีไซน์สวย", "beauty"],
            "classic": ["คลาสสิก", "วินเทจ", "ย้อนยุค", "classic"],
            "compact": ["กะทัดรัด", "คันเล็ก", "เบา", "compact"],
            "cute": ["น่ารัก", "cute"],
            "modern": ["โมเดิร์น", "ทันสมัย", "modern"],
            "premium": ["หรู", "ดูดี", "พรีเมียม", "premium"],
            "sporty": ["สปอร์ต", "เท่", "วัยรุ่น", "ทรงสปอร์ต", "แรงๆ"],
        }
        self._append_matches(text, result["style"], mapping)

    def _extract_levels(self, text: str, result: dict[str, Any]) -> None:
        if any(word in text for word in ["แรงมาก", "อัตราเร่งดี", "เครื่องแรง", "แรงๆ"]):
            result["performance"] = "high"
        elif any(word in text for word in ["แรงนิด", "พอแรง", "แรงพอประมาณ"]):
            result["performance"] = "medium"
        elif "ไม่เน้นแรง" in text:
            result["performance"] = "low"

        if any(word in text for word in ["นั่งสบาย", "ขับสบาย", "ซ้อนสบาย", "สบายมาก"]):
            result["comfort"] = "high"
        elif "สบายปานกลาง" in text:
            result["comfort"] = "medium"

        if any(word in text for word in ["ปลอดภัยสูง", "เซฟตี้สูง", "safety สูง", "abs"]):
            result["safety_level"] = "high"
        elif any(word in text for word in ["ปลอดภัยกลาง", "เซฟตี้กลาง"]):
            result["safety_level"] = "medium"
        elif "ไม่เน้นความปลอดภัย" in text:
            result["safety_level"] = "low"

        if any(word in text for word in ["เทคโนโลยีสูง", "ฟีเจอร์เยอะ", "จอ", "สมาร์ทคีย์", "smart key"]):
            result["technology_level"] = "high"
        elif any(word in text for word in ["เทคโนโลยีกลาง", "ฟีเจอร์กลาง"]):
            result["technology_level"] = "medium"

    def _extract_booleans(self, text: str, result: dict[str, Any]) -> None:
        if any(word in text for word in ["ขี่ง่าย", "ขับง่าย", "มือใหม่", "ควบคุมง่าย", "เบา"]):
            result["easy_to_ride"] = True
        if any(word in text for word in ["ประหยัดน้ำมัน", "กินน้ำมันน้อย", "ไม่กินน้ำมัน", "ประหยัด"]):
            result["fuel_saving"] = True
        if any(word in text for word in ["เก็บของ", "ใส่ของ", "ช่องเก็บ", "พื้นที่เก็บ", "บรรทุก"]):
            result["storage_need"] = True
        if any(word in text for word in ["ดูแลง่าย", "ซ่อมง่าย", "ไม่จุกจิก", "ไม่ซ่อมบ่อย"]):
            result["maintenance_easy"] = True

        if any(word in text for word in ["ไม่ต้องขี่ง่าย", "ไม่เน้นขี่ง่าย"]):
            result["easy_to_ride"] = False
        if any(word in text for word in ["ไม่เน้นประหยัด", "ไม่สนประหยัดน้ำมัน"]):
            result["fuel_saving"] = False
        if any(word in text for word in ["ไม่ต้องมีที่เก็บ", "ไม่เอาที่เก็บของ"]):
            result["storage_need"] = False
        if any(word in text for word in ["ไม่สนดูแลง่าย"]):
            result["maintenance_easy"] = False

    def _append_matches(self, text: str, target: list[str], mapping: dict[str, list[str]]) -> None:
        for label, words in mapping.items():
            if any(word in text for word in words) and label not in target:
                target.append(label)

    def _slots_from_assistant_question(self, text: str) -> list[str]:
        for slots, question in QUESTION_FLOW:
            if question[:20] in text or all(slot in text for slot in slots):
                return list(slots)
        return []
