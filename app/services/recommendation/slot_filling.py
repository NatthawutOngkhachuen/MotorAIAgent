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
        "สวัสดีครับ ผมช่วยแนะนำรถที่เหมาะกับคุณได้ เดี๋ยวขอถามสั้นๆ ก่อนนะครับ ปกติใช้รถทำอะไรเป็นหลักครับ เช่น ไปทำงาน/ไปเรียน ขี่ในเมือง ออกทริป เดินทางไกล ส่งของ ซื้อของ ใช้กับครอบครัว บรรทุกของ หรือเจอถนนขรุขระ?",
    ),
    (
        ["style"],
        "ชอบสไตล์รถประมาณไหนครับ เช่น สปอร์ต โมเดิร์น พรีเมียม คลาสสิก กะทัดรัด หรือสายลุย?",
    ),
    (
        ["performance"],
        "ปกติชอบฟีลการขี่ประมาณไหนครับ: เน้นความเร็ว, ขี่ทั่วไป, หรือขี่ช้า?",
    ),
    (
        ["comfort"],
        "เรื่องความสบายในการนั่ง ให้สำคัญระดับไหนครับ: น้อย กลาง หรือมาก?",
    ),
    (
        ["safety_level"],
        "เรื่องความปลอดภัย ให้สำคัญระดับไหนครับ: ต่ำ กลาง หรือสูง?",
    ),
    (
        ["easy_to_ride", "fuel_saving", "storage_need", "maintenance_easy"],
        "อยากได้ฟังก์ชันใช้งานแบบไหนบ้างครับ เลือกเป็นตัวเลขได้เลย:\n1. ขับขี่ง่าย\n2. ประหยัดน้ำมัน\n3. ที่เก็บของเยอะ\n4. ดูแลรักษาง่าย / หาอะไหล่ง่าย",
    ),
]


CONVERSATION_DEFAULT_ZERO_SLOTS = {
    "budget_level": "unknown",
    "technology_level": "unknown",
}

SLOT_QUESTIONS = {
    "usage_fit": "ปกติใช้รถทำอะไรเป็นหลักครับ เช่น ไปทำงาน/ไปเรียน ขี่ในเมือง ออกทริป เดินทางไกล ส่งของ ซื้อของ ใช้กับครอบครัว บรรทุกของ หรือเจอถนนขรุขระ?",
    "style": "ชอบสไตล์รถประมาณไหนครับ เช่น สปอร์ต โมเดิร์น พรีเมียม คลาสสิก กะทัดรัด หรือสายลุย?",
    "performance": "ปกติชอบฟีลการขี่ประมาณไหนครับ: เน้นความเร็ว, ขี่ทั่วไป, หรือขี่ช้า?",
    "comfort": "เรื่องความสบายในการนั่ง ให้สำคัญระดับไหนครับ: น้อย กลาง หรือมาก?",
    "safety_level": "เรื่องความปลอดภัย ให้สำคัญระดับไหนครับ: ต่ำ กลาง หรือสูง?",
    "easy_to_ride": "เรื่องขับขี่ง่าย อยากได้เป็นพิเศษไหมครับ?",
    "fuel_saving": "เรื่องประหยัดน้ำมัน อยากเน้นเป็นพิเศษไหมครับ?",
    "storage_need": "ต้องการพื้นที่เก็บของเยอะไหมครับ?",
    "maintenance_easy": "อยากได้รถที่ดูแลรักษาง่าย หรือหาอะไหล่ง่ายไหมครับ?",
}

FUNCTION_CHOICE_SLOTS = {
    "1": "easy_to_ride",
    "2": "fuel_saving",
    "3": "storage_need",
    "4": "maintenance_easy",
}


YES_WORDS = {"ใช่", "เอา", "ต้องการ", "โอเค", "ok", "yes", "y", "ครับ", "ค่ะ", "ได้"}
NO_WORDS = {"ไม่", "ไม่เอา", "ไม่ต้องการ", "no", "n", "ไม่ครับ", "ไม่ค่ะ"}


@dataclass
class SlotFillingState:
    preferences: dict[str, Any] = field(default_factory=empty_preference_state)
    asked_slots: list[str] = field(default_factory=list)
    last_asked_slots: list[str] = field(default_factory=list)
    is_complete: bool = False
    function_choices_answered: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "preferences": self.preferences,
            "asked_slots": self.asked_slots,
            "last_asked_slots": self.last_asked_slots,
            "is_complete": self.is_complete,
            "function_choices_answered": self.function_choices_answered,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "SlotFillingState":
        if not isinstance(data, dict):
            return cls()
        return cls(
            preferences=merge_preferences(empty_preference_state(), data.get("preferences") or {}),
            asked_slots=list(data.get("asked_slots") or []),
            last_asked_slots=list(data.get("last_asked_slots") or []),
            is_complete=bool(data.get("is_complete", False)),
            function_choices_answered=bool(data.get("function_choices_answered", False)),
        )


class SlotFillingService:
    def __init__(self, extractor: Any | None = None):
        self.extractor = extractor

    def start(self) -> tuple[str, SlotFillingState]:
        state = SlotFillingState()
        question = QUESTION_FLOW[0][1]
        state.last_asked_slots = list(QUESTION_FLOW[0][0])
        state.asked_slots = list(QUESTION_FLOW[0][0])
        return question, state

    def start_follow_up(self) -> tuple[str, SlotFillingState]:
        question, state = self.start()
        question = question.replace(
            "สวัสดีครับ ผมช่วยแนะนำรถที่เหมาะกับคุณได้ เดี๋ยวขอถามสั้นๆ ก่อนนะครับ ",
            "สนใจดูรถแนวไหนเพิ่มเติมอีกมั้ยครับ ",
            1,
        )
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
            use_llm=False,
        )
        if not self._can_skip_llm(extracted, state.preferences, state.last_asked_slots, user_message):
            extracted = self.extract_preferences(
                user_message,
                state.last_asked_slots,
                chat_history=chat_history or [],
                current_preferences=state.preferences,
                session_id=session_id,
                use_llm=True,
            )
        state.preferences = merge_preferences(state.preferences, extracted)
        if self._is_clear_function_choice_answer(user_message, state.last_asked_slots, extracted):
            state.function_choices_answered = True
        self._apply_conversation_defaults(state.preferences)
        state.is_complete = self.is_complete(state.preferences, state)

        if state.is_complete:
            return state, None

        next_slots, question = self.next_question(state.preferences, state.asked_slots, state)
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

        state.is_complete = self.is_complete(state.preferences, state)
        self._apply_conversation_defaults(state.preferences)
        return state

    def extract_preferences(
        self,
        text: str,
        last_asked_slots: list[str] | None = None,
        chat_history: list[dict[str, Any]] | None = None,
        current_preferences: dict[str, Any] | None = None,
        session_id: str = "recommendation-session",
        use_llm: bool = True,
    ) -> dict[str, Any]:
        text_norm = text.strip().lower()
        result: dict[str, Any] = {
            "usage_fit": [],
            "style": [],
        }

        self._extract_yes_no_answer(text_norm, last_asked_slots or [], result)
        self._extract_function_choices(text_norm, last_asked_slots or [], result)
        self._extract_budget(text_norm, result)
        self._extract_usage(text_norm, result)
        self._extract_style(text_norm, result)
        self._extract_levels(text_norm, last_asked_slots or [], result)
        self._extract_booleans(text_norm, result)

        if not use_llm or self.extractor is None:
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
        state: SlotFillingState | None = None,
    ) -> tuple[list[str], str]:
        asked_slots = asked_slots or []
        for slots, question in QUESTION_FLOW:
            unasked_missing_slots = [
                slot
                for slot in slots
                if slot not in asked_slots and not self._slot_complete(preferences, slot, state)
            ]
            if unasked_missing_slots:
                return unasked_missing_slots, self._question_for_slots(unasked_missing_slots, question)

        for slots, question in QUESTION_FLOW:
            missing_slots = [slot for slot in slots if not self._slot_complete(preferences, slot, state)]
            if missing_slots:
                return missing_slots, self._question_for_slots(missing_slots, question)

        return [], ""

    def is_complete(self, preferences: dict[str, Any], state: SlotFillingState | None = None) -> bool:
        return all(self._slots_complete(preferences, slots, state) for slots, _ in QUESTION_FLOW)

    def build_vector(self, preferences: dict[str, Any]) -> list[float]:
        cleaned = dict(preferences or {})
        self._apply_conversation_defaults(cleaned)
        return preference_to_vector(cleaned)

    def _apply_conversation_defaults(self, preferences: dict[str, Any]) -> None:
        preferences.update(CONVERSATION_DEFAULT_ZERO_SLOTS)

    def _slots_complete(
        self,
        preferences: dict[str, Any],
        slots: list[str],
        state: SlotFillingState | None = None,
    ) -> bool:
        return all(self._slot_complete(preferences, slot, state) for slot in slots)

    def _slot_complete(
        self,
        preferences: dict[str, Any],
        slot: str,
        state: SlotFillingState | None = None,
    ) -> bool:
        if state and state.function_choices_answered and slot in BOOL_FEATURES:
            return True
        value = preferences.get(slot)
        if slot in {"usage_fit", "style"}:
            return bool(value)
        return value not in [None, "", "unknown", []]

    def _question_for_slots(self, slots: list[str], fallback: str) -> str:
        if len(slots) == 1:
            return SLOT_QUESTIONS.get(slots[0], fallback)
        questions = [SLOT_QUESTIONS.get(slot, "") for slot in slots]
        questions = [question for question in questions if question]
        if all(slot in BOOL_FEATURES for slot in slots):
            labels = {
                "easy_to_ride": "ขับขี่ง่าย",
                "fuel_saving": "ประหยัดน้ำมัน",
                "storage_need": "ที่เก็บของเยอะ",
                "maintenance_easy": "ดูแลรักษาง่าย / หาอะไหล่ง่าย",
            }
            options = "\n".join(
                f"{number}. {labels[slot]}"
                for number, slot in FUNCTION_CHOICE_SLOTS.items()
                if slot in slots
            )
            return "ขอถามเพิ่มเฉพาะฟังก์ชันที่ยังไม่ชัดนะครับ เลือกเป็นตัวเลขได้เลย:\n" + options
        return "ขอถามเพิ่มอีกนิดนะครับ " + " ".join(questions)

    def _has_meaningful_update(self, result: dict[str, Any]) -> bool:
        if result.get("usage_fit") or result.get("style"):
            return True
        for slot in LEVEL_FEATURES + BOOL_FEATURES:
            if result.get(slot) not in [None, "", "unknown", []]:
                return True
        return False

    def _can_skip_llm(
        self,
        extracted: dict[str, Any],
        current_preferences: dict[str, Any],
        last_asked_slots: list[str],
        user_message: str,
    ) -> bool:
        if not self._has_meaningful_update(extracted):
            return False
        compact_text = re.sub(r"\s+", "", user_message.strip().lower())
        is_simple_answer = len(compact_text) <= 24 or bool(re.fullmatch(r"[1-4,./กับและ]+", compact_text))
        if not is_simple_answer:
            return False
        if last_asked_slots and all(slot in BOOL_FEATURES for slot in last_asked_slots):
            return any(extracted.get(slot) not in [None, "", "unknown", []] for slot in last_asked_slots)
        merged = merge_preferences(current_preferences, extracted)
        return bool(last_asked_slots) and self._slots_complete(merged, last_asked_slots)

    def _is_clear_function_choice_answer(
        self,
        user_message: str,
        last_asked_slots: list[str],
        extracted: dict[str, Any],
    ) -> bool:
        if not last_asked_slots or not all(slot in BOOL_FEATURES for slot in last_asked_slots):
            return False
        text = user_message.strip().lower()
        compact = re.sub(r"\s+", "", text)
        if any(word in compact for word in ["เอาหมด", "ทั้งหมด", "ทุกข้อ", "ครบทุกข้อ"]):
            return True
        if re.search(r"[1-4]", text):
            return True
        if any(word in text for word in ["หนึ่ง", "สอง", "สาม", "สี่"]):
            return True
        return any(extracted.get(slot) not in [None, "", "unknown", []] for slot in last_asked_slots)

    def _extract_yes_no_answer(
        self,
        text: str,
        last_asked_slots: list[str],
        result: dict[str, Any],
    ) -> None:
        compact = re.sub(r"\s+", "", text)
        if len(last_asked_slots) > 1 and all(slot in BOOL_FEATURES for slot in last_asked_slots):
            if compact not in {"ไม่", "ไม่เอา", "ไม่ต้องการ", "no", "n", "ไม่ครับ", "ไม่ค่ะ"}:
                return
        is_yes = compact in YES_WORDS or any(word in text for word in ["ใช่", "เอา", "ต้องการ", "yes"])
        is_no = compact in NO_WORDS or any(word in text for word in ["ไม่เอา", "ไม่ต้องการ"])
        if not is_yes and not is_no:
            return

        value = is_yes and not is_no
        for slot in last_asked_slots:
            if slot in BOOL_FEATURES:
                result[slot] = value

    def _extract_function_choices(
        self,
        text: str,
        last_asked_slots: list[str],
        result: dict[str, Any],
    ) -> None:
        if not last_asked_slots or not all(slot in BOOL_FEATURES for slot in last_asked_slots):
            return

        compact = re.sub(r"\s+", "", text)
        if any(word in compact for word in ["เอาหมด", "ทั้งหมด", "ทุกข้อ", "ครบทุกข้อ"]):
            for slot in last_asked_slots:
                result[slot] = True
            return

        thai_number_map = {
            "หนึ่ง": "1",
            "สอง": "2",
            "สาม": "3",
            "สี่": "4",
        }
        selected = set(re.findall(r"[1-4]", text))
        for word, number in thai_number_map.items():
            if word in text:
                selected.add(number)

        for number in selected:
            slot = FUNCTION_CHOICE_SLOTS[number]
            if slot in last_asked_slots:
                result[slot] = True

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

    def _extract_levels(self, text: str, last_asked_slots: list[str], result: dict[str, Any]) -> None:
        if any(word in text for word in ["เน้นความเร็ว", "เร็ว", "แรงมาก", "อัตราเร่งดี", "เครื่องแรง", "แรงๆ", "ออกตัวไว"]):
            result["performance"] = "high"
        elif any(word in text for word in ["ขี่ทั่วไป", "ทั่วไป", "กลางๆ", "ปกติ", "ชิล", "แรงนิด", "พอแรง", "แรงพอประมาณ"]):
            result["performance"] = "medium"
        elif any(word in text for word in ["ขี่ช้า", "ช้า", "เรื่อยๆ", "ไม่เน้นแรง"]):
            result["performance"] = "low"

        if any(word in text for word in ["นั่งสบาย", "ขับสบาย", "ซ้อนสบาย", "สบายมาก"]):
            result["comfort"] = "high"
        elif any(word in text for word in ["สบายปานกลาง"]):
            result["comfort"] = "medium"
        elif any(word in text for word in ["ไม่เน้นสบาย"]):
            result["comfort"] = "low"

        if "comfort" in last_asked_slots:
            if any(word in text for word in ["มาก", "สูง"]):
                result["comfort"] = "high"
            elif any(word in text for word in ["กลาง", "ปานกลาง", "พอประมาณ"]):
                result["comfort"] = "medium"
            elif any(word in text for word in ["น้อย", "ต่ำ"]):
                result["comfort"] = "low"

        if any(word in text for word in ["ปลอดภัยสูง", "เซฟตี้สูง", "safety สูง", "abs"]):
            result["safety_level"] = "high"
        elif any(word in text for word in ["ปลอดภัยกลาง", "เซฟตี้กลาง"]):
            result["safety_level"] = "medium"
        elif any(word in text for word in ["ไม่เน้นความปลอดภัย"]):
            result["safety_level"] = "low"

        if "safety_level" in last_asked_slots:
            if "สูง" in text:
                result["safety_level"] = "high"
            elif any(word in text for word in ["กลาง", "ปานกลาง"]):
                result["safety_level"] = "medium"
            elif any(word in text for word in ["ต่ำ", "น้อย"]):
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
