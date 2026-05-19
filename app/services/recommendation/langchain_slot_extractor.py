from __future__ import annotations

import json
import os
from typing import Any

from dotenv import load_dotenv
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory

from app.services.ollama_client import get_ollama_base_url, make_chat_ollama
from app.services.recommendation.vectorizer import empty_preference_state, merge_preferences


load_dotenv()


class LangChainSlotExtractor:
    """
    LangChain-based slot extractor.

    It uses RunnableWithMessageHistory so short answers such as "ใช่" or
    "ไม่เอา" are interpreted against the previous assistant question.
    Rule-based extraction remains outside this class as a fallback/guard.
    """

    def __init__(
        self,
        model_name: str | None = None,
        base_url: str | None = None,
        temperature: float = 0.0,
        timeout: int | None = None,
    ):
        self.model_name = model_name or os.getenv("EXTRACTOR_MODEL", os.getenv("OLLAMA_MODEL", "gpt-oss:120b"))
        self.base_url = base_url or get_ollama_base_url("EXTRACTOR")
        self.temperature = temperature
        self.timeout = int(timeout or os.getenv("EXTRACTOR_TIMEOUT_SECONDS", "6"))

        self.llm = make_chat_ollama(
            model=self.model_name,
            prefix="EXTRACTOR",
            base_url=self.base_url,
            temperature=self.temperature,
            num_predict=500,
            timeout=self.timeout,
        )

    def extract(
        self,
        *,
        user_message: str,
        chat_history: list[dict[str, Any]],
        current_preferences: dict[str, Any] | None = None,
        last_asked_slots: list[str] | None = None,
        session_id: str = "recommendation-session",
    ) -> dict[str, Any]:
        history_store = {
            session_id: self._history_from_records(chat_history),
        }

        prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessage(content=self._system_prompt()),
                MessagesPlaceholder(variable_name="history"),
                (
                    "human",
                    (
                        "ข้อความล่าสุดของผู้ใช้: {user_message}\n\n"
                        "current_preferences:\n{current_preferences}\n\n"
                        "last_asked_slots:\n{last_asked_slots}\n\n"
                        "ตอบเป็น JSON object เท่านั้น"
                    ),
                ),
            ]
        )
        runnable = prompt | self.llm
        runnable_with_history = RunnableWithMessageHistory(
            runnable,
            lambda sid: history_store.setdefault(sid, InMemoryChatMessageHistory()),
            input_messages_key="user_message",
            history_messages_key="history",
        )

        response = runnable_with_history.invoke(
            {
                "user_message": user_message,
                "current_preferences": json.dumps(
                    current_preferences or empty_preference_state(),
                    ensure_ascii=False,
                ),
                "last_asked_slots": json.dumps(last_asked_slots or [], ensure_ascii=False),
            },
            config={"configurable": {"session_id": session_id}},
        )

        raw_text = str(getattr(response, "content", response)).strip()
        parsed = self._parse_json(raw_text)
        return merge_preferences(empty_preference_state(), parsed)

    def _history_from_records(self, records: list[dict[str, Any]]) -> InMemoryChatMessageHistory:
        history = InMemoryChatMessageHistory()
        for record in records:
            role = record.get("role")
            content = record.get("content") or ""
            if not content:
                continue
            if role == "user":
                history.add_message(HumanMessage(content=content))
            elif role == "assistant":
                history.add_message(AIMessage(content=content))
        return history

    def _parse_json(self, raw_text: str) -> dict[str, Any]:
        if raw_text.startswith("```"):
            raw_text = raw_text.strip("`")
            raw_text = raw_text.replace("json\n", "", 1).strip()
        start = raw_text.find("{")
        end = raw_text.rfind("}")
        if start >= 0 and end >= start:
            raw_text = raw_text[start : end + 1]
        parsed = json.loads(raw_text)
        if not isinstance(parsed, dict):
            raise ValueError("Extractor did not return a JSON object")
        return parsed

    def _system_prompt(self) -> str:
        return """
คุณคือ LangChain slot extractor สำหรับระบบแนะนำรถจักรยานยนต์ MotorAiAgent

หน้าที่:
- อ่านประวัติการสนทนาและข้อความล่าสุด
- ดึง preference ของผู้ใช้เป็น schema กลางสำหรับสร้าง user vector 27 มิติ
- ถ้าผู้ใช้ตอบสั้น เช่น "ใช่", "ไม่", "เอา", "ไม่เอา" ให้ตีความจากคำถามล่าสุดและ last_asked_slots
- ถ้าผู้ใช้ตอบข้ามหลายเรื่อง ให้เติมทุก slot ที่พบ
- อย่าเดา slot ที่ผู้ใช้ไม่ได้บอก
- ถ้าไม่พบข้อมูล slot ใด ให้ไม่ต้องใส่ key นั้น หรือใส่ unknown/[] ได้

JSON schema ที่อนุญาต:
{
  "budget_level": "low|medium|high|unknown",
  "usage_fit": ["city","daily","delivery","family","long_distance","rough_road","shopping","storage_heavy","trip","work"],
  "style": ["adventure","beauty","classic","compact","cute","modern","premium","sporty"],
  "performance": "low|medium|high|unknown",
  "comfort": "low|medium|high|unknown",
  "safety_level": "low|medium|high|unknown",
  "technology_level": "low|medium|high|unknown",
  "easy_to_ride": true|false|"unknown",
  "fuel_saving": true|false|"unknown",
  "storage_need": true|false|"unknown",
  "maintenance_easy": true|false|"unknown"
}

Meaning guide:
- ไม่แพง, ถูก, งบน้อย -> budget_level low
- พอประมาณ, กลางๆ, ไม่ถูกไม่แพง -> budget_level medium
- ตัวท็อป, แพงได้, พรีเมียม -> budget_level high
- ขี่ในเมือง, รถติด, ซอกแซก -> usage_fit city
- ใช้ทุกวัน, ไปเรียน, ไปทำงาน -> usage_fit daily
- ต่างจังหวัด, ทางไกล -> usage_fit long_distance
- ออกทริป, เที่ยว -> usage_fit trip
- ส่งของ, เดลิเวอรี่, ไรเดอร์, grab -> usage_fit delivery
- ถนนขรุขระ, ถนนไม่ดี, ลุย -> usage_fit rough_road
- เท่, สปอร์ต, วัยรุ่น -> style sporty
- หรู, ดูดี, พรีเมียม -> style premium
- คลาสสิก, วินเทจ, ย้อนยุค -> style classic
- น่ารัก -> style cute
- สวย, แฟชั่น -> style beauty
- โมเดิร์น, ทันสมัย -> style modern
- คันเล็ก, เบา, คล่องตัว -> style compact และ easy_to_ride true
- แรงมาก, อัตราเร่งดี -> performance high
- แรงพอประมาณ -> performance medium
- ไม่เน้นแรง -> performance low
- นั่งสบาย, ขับสบาย, ซ้อนสบาย -> comfort high
- ปลอดภัยสูง, ABS -> safety_level high
- ฟีเจอร์เยอะ, smart key, จอ -> technology_level high
- ขี่ง่าย, มือใหม่ -> easy_to_ride true
- ประหยัดน้ำมัน, ไม่กินน้ำมัน -> fuel_saving true
- เก็บของ, ใส่ของ, บรรทุก -> storage_need true
- ดูแลง่าย, ไม่จุกจิก, ไม่ซ่อมบ่อย -> maintenance_easy true

ตอบ JSON object เท่านั้น ห้าม markdown ห้ามคำอธิบาย
        """.strip()
