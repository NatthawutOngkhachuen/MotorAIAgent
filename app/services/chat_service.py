import os
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from app.db.neo4j import run_query
from app.repositories.chat_repository import (
    create_session,
    load_recent_messages,
    save_message,
    session_belongs_to_user,
    update_session_active,
)
from app.services.ollama_client import get_ollama_base_url, make_chat_ollama
from typing import AsyncGenerator
import time
import json
import re

OLLAMA_MODEL  = os.getenv("OLLAMA_MODEL", "typhoon2")
OLLAMA_BASE_URL = get_ollama_base_url()
GUEST_USER_ID = "00000000-0000-0000-0000-000000000001"

_graph_cache: str | None = None
INITIAL_ASSISTANT_MESSAGE = (
    "สวัสดีครับ ถามเรื่องรุ่น ยี่ห้อ หรือคุณสมบัติที่สนใจได้เลยครับ "
    "เช่น งบประมาณเท่านี้ควรเลือกรุ่นไหน หรือรุ่นไหนเหมาะกับการใช้งานของคุณ"
)

EMOJI_PATTERN = re.compile(
    "["
    "\U0001F300-\U0001FAFF"
    "\U00002700-\U000027BF"
    "\U00002600-\U000026FF"
    "]+",
    flags=re.UNICODE,
)

OFF_TOPIC_KEYWORDS = [
    "กินข้าว",
    "หิว",
    "อาหาร",
    "ร้านอาหาร",
    "ไปเที่ยวกัน",
    "เที่ยวกันไหม",
    "ดูหนัง",
    "เล่นเกม",
    "เพลง",
    "หวย",
    "ข่าว",
]

MOTORCYCLE_KEYWORDS = [
    "รถ",
    "มอเตอร์ไซค์",
    "มอไซค์",
    "รุ่น",
    "ขับ",
    "ขี่",
    "เดินทาง",
    "งบ",
    "ราคา",
    "honda",
    "yamaha",
]

GREETING_KEYWORDS = [
    "สวัสดี",
    "หวัดดี",
    "hello",
    "hi",
]


def get_graph_context() -> str:
    global _graph_cache
    if _graph_cache:
        return _graph_cache

    cypher = """
        MATCH (m:VehicleModel)
        OPTIONAL MATCH (m)-[r1]->(entity)
        OPTIONAL MATCH (entity)-[r2]->(semantic)
        RETURN
            m.name AS model,
            type(r1) AS rel1,
            entity.name AS entity_name,
            semantic.name AS semantic_name
        ORDER BY m.name
    """
    rows = run_query(cypher, {})
    if not rows:
        return "ไม่พบข้อมูลรถในฐานข้อมูล"

    models: dict = {}
    for row in rows:
        model = row.get("model") or "?"
        if model not in models:
            models[model] = {"features": [], "use_case": [], "safety": [], "style": []}
        rel1     = (row.get("rel1") or "").upper()
        entity   = row.get("entity_name") or ""
        semantic = row.get("semantic_name") or ""
        label    = f"{entity} ({semantic})" if semantic else entity
        if not label:
            continue
        if "USE_CASE" in rel1: models[model]["use_case"].append(label)
        elif "SAFETY" in rel1: models[model]["safety"].append(label)
        elif "STYLE"  in rel1: models[model]["style"].append(label)
        else:                  models[model]["features"].append(label)

    lines = []
    for name, d in models.items():
        parts = [f"รุ่น: {name}"]
        if d["use_case"]: parts.append(f"  เหมาะกับ: {', '.join(set(d['use_case']))}")
        if d["style"]:    parts.append(f"  สไตล์: {', '.join(set(d['style']))}")
        if d["safety"]:   parts.append(f"  ความปลอดภัย: {', '.join(set(d['safety']))}")
        if d["features"]: parts.append(f"  คุณสมบัติ: {', '.join(set(d['features'][:6]))}")
        lines.append("\n".join(parts))

    _graph_cache = "ข้อมูลรถมอเตอร์ไซค์:\n\n" + "\n\n".join(lines)
    return _graph_cache


def clear_graph_cache():
    global _graph_cache
    _graph_cache = None


def get_all_model_names() -> list[str]:
    rows = run_query("MATCH (m:VehicleModel) RETURN m.name AS name", {})
    return [r["name"] for r in rows if r.get("name")]


def extract_recommended_models(history: list, all_models: list[str]) -> list[str]:
    recommended = set()
    for m in history:
        if m["role"] == "assistant":
            for model_name in all_models:
                if model_name.lower() in m["content"].lower():
                    recommended.add(model_name)
    return list(recommended)


def clean_assistant_answer(text: str, allow_greeting: bool = False) -> str:
    text = EMOJI_PATTERN.sub("", text)
    replacements = {
        "นะคะ": "นะครับ",
        "ค่ะ": "ครับ",
        "คะ": "ครับ",
        "จ้า": "ครับ",
        "จ๊ะ": "ครับ",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    if not allow_greeting:
        text = re.sub(r"^\s*สวัสดี(?:ครับ)?[!,.،\s-]*", "", text)
    return text.strip()


def is_off_topic_question(question: str, all_models: list[str]) -> bool:
    normalized = question.strip().lower()
    if not normalized:
        return False
    if any(model.lower() in normalized for model in all_models):
        return False
    if any(keyword in normalized for keyword in MOTORCYCLE_KEYWORDS):
        return False
    return any(keyword in normalized for keyword in OFF_TOPIC_KEYWORDS)


def is_duplicate_off_topic_reply(raw_context: list, question: str) -> bool:
    if len(raw_context) < 2:
        return False
    return (
        raw_context[-2].get("role") == "user"
        and raw_context[-2].get("content", "").strip() == question.strip()
        and raw_context[-1].get("role") == "assistant"
    )


def is_greeting_question(question: str) -> bool:
    normalized = question.strip().lower()
    return any(keyword in normalized for keyword in GREETING_KEYWORDS)


def build_system_prompt(language: str, graph_context: str,
                        already_recommended: list[str],
                        allow_greeting: bool = False) -> str:
    already_str = ""
    if already_recommended:
        names = ", ".join(already_recommended)
        if language == "th":
            already_str = (
                f"\n\n⚠️ รุ่นที่แนะนำไปแล้วในการสนทนานี้: {names}\n"
                f"ถ้าลูกค้าขอรุ่นอื่น ห้ามแนะนำรุ่นเหล่านี้ซ้ำอีก"
            )
        else:
            already_str = (
                f"\n\n⚠️ Already recommended: {names}\n"
                f"Do NOT repeat these if customer asks for other models."
            )

    greeting_rule = (
        "You may greet the customer once because this is the first assistant reply."
        if allow_greeting
        else "Do not greet again; continue the conversation directly without saying hello or สวัสดี."
    )

    if language == "en":
        return f"""You are an expert multi-brand motorcycle sales consultant.

Guidelines:
- Answer EXACTLY what the customer asks
- If they ask "any other models?" recommend DIFFERENT models not yet mentioned
- If they say "not that model" exclude it completely
- If multiple database models match the customer, recommend several suitable models in the same answer with clear reasons WHY they suit them
- Ask follow-up if needed (budget, usage, rider type)
- Be warm and conversational
- {greeting_rule}
- The assistant persona is male. In Thai, end politely with "ครับ" only. Do not use "ค่ะ", "คะ", "จ้า", or emojis.
- If the customer asks about something unrelated to motorcycles or our vehicle database, politely say you mainly help with motorcycle recommendations and guide the conversation back to choosing a suitable motorcycle.
- Only use models from the database below
{already_str}

--- MOTORCYCLE DATABASE ---
{graph_context}
---------------------------"""

    return f"""คุณเป็นที่ปรึกษารถมอเตอร์ไซค์หลายแบรนด์ที่เชี่ยวชาญ สุภาพ และเป็นกันเอง

แนวทาง:
- ตอบตรงคำถามที่ถามเสมอ
- ถ้าลูกค้าถามว่า "มีรุ่นอื่นไหม" แนะนำรุ่นที่ยังไม่ได้แนะนำเท่านั้น
- ถ้าลูกค้าบอกว่า "ไม่เอารุ่นนั้น" ห้ามแนะนำรุ่นนั้นอีกเลย
- ถ้ามีหลายรุ่นในฐานข้อมูลที่ตรงกับลูกค้า ให้แนะนำหลายรุ่นในคำตอบเดียว พร้อมอธิบายว่าทำไมแต่ละรุ่นถึงเหมาะกับลูกค้า
- ถามเพิ่มถ้าข้อมูลไม่พอ เช่น งบประมาณ การใช้งาน เพศ
- ตอบภาษาไทยแบบสุภาพ เป็นกันเอง และคุยง่าย
- ตอบให้กระชับ อ่านง่าย ไม่ต้องอธิบายยาวถ้าผู้ใช้ไม่ได้ขอรายละเอียดเชิงลึก
- ถ้าแนะนำรุ่นรถ ให้สรุปเหตุผลสำคัญ 2-4 ข้อก็พอ และถามต่อสั้น ๆ เมื่อจำเป็น
- กำหนดให้แชทบอทเป็นผู้ชาย ใช้คำลงท้ายสุภาพว่า "ครับ" เท่านั้น ห้ามใช้ "ค่ะ", "คะ", "จ้า" และห้ามใช้อิโมจิ
- {"ทักทายลูกค้าได้ เพราะลูกค้าเริ่มด้วยคำทักทาย" if allow_greeting else "ไม่ต้องทักสวัสดี ให้ตอบต่อจากข้อมูลหรือคำถามของลูกค้าได้เลย"}
- ถ้าลูกค้าถามเรื่องที่ไม่เกี่ยวกับรถมอเตอร์ไซค์หรือสินค้าที่มีในฐานข้อมูล ให้ตอบสั้น ๆ อย่างสุภาพว่าเราช่วยเรื่องแนะนำรถมอเตอร์ไซค์เป็นหลัก แล้วชวนกลับมาคุยเรื่องรุ่นรถที่เหมาะกับลูกค้า
- แนะนำเฉพาะรุ่นที่มีในฐานข้อมูลเท่านั้น
{already_str}

--- ฐานข้อมูลรถ ---
{graph_context}
-------------------"""


async def stream_answer(question: str,
                        language: str = "th",
                        session_id: str = None,
                        user_id: str = None
                        ) -> AsyncGenerator[str, None]: 
    user_id = user_id or GUEST_USER_ID

    if session_id and not session_belongs_to_user(session_id, user_id):
        session_id = None

    is_new_session = False
    if not session_id:
        session_id = create_session(user_id)
        is_new_session = True
        

    if is_new_session:
        save_message(session_id, user_id, "assistant", INITIAL_ASSISTANT_MESSAGE)

    raw_context   = load_recent_messages(session_id, limit=6)
    graph_context = get_graph_context()
    all_models    = get_all_model_names()
    model_count   = graph_context.count("รุ่น:")

    already_recommended = extract_recommended_models(raw_context, all_models)
    is_first_message = len(raw_context) == 0
    allow_greeting = is_first_message and is_greeting_question(question)
    is_off_topic = is_off_topic_question(question, all_models)

    history = []
    for m in raw_context:
        if m["role"] == "user":
            history.append(HumanMessage(content=m["content"]))
        else:
            history.append(AIMessage(content=m["content"]))

    llm_messages = [
        SystemMessage(content=build_system_prompt(
            language, graph_context, already_recommended, allow_greeting
        )),
        *history,
        HumanMessage(content=question),
    ]

    llm = make_chat_ollama(
        model=OLLAMA_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=0.7,
        num_predict=1200,
    )

    # ส่ง session_id และ model_count ก่อน
    yield f"data: {json.dumps({'type': 'session', 'session_id': session_id, 'model_count': model_count})}\n\n"

    if is_off_topic and is_duplicate_off_topic_reply(raw_context, question):
        yield f"data: {json.dumps({'type': 'done', 'elapsed': 0})}\n\n"
        return

    start_time  = time.time()
    full_answer = ""

    if is_off_topic:
        full_answer = "ขอโทษนะครับ ผมช่วยเรื่องแนะนำรถมอเตอร์ไซค์เป็นหลักครับ ถ้าต้องการ ผมช่วยดูรุ่นที่เหมาะกับการใช้งาน งบประมาณ หรือสไตล์ที่คุณชอบได้ครับ"
    else:
        async for chunk in llm.astream(llm_messages):
            token = chunk.content
            if token:
                full_answer += token

    full_answer = clean_assistant_answer(full_answer, allow_greeting=allow_greeting)
    if full_answer:
        yield f"data: {json.dumps({'type': 'token', 'token': full_answer})}\n\n"

    elapsed = round(time.time() - start_time, 1)

    yield f"data: {json.dumps({'type': 'done', 'elapsed': elapsed})}\n\n"

    # บันทึกลง PostgreSQL
    save_message(session_id, user_id, "user", question)
    save_message(session_id,
                user_id, 
                "assistant",
                full_answer,
                rag_sources=[{"source": "neo4j", "model_count": model_count}])
    update_session_active(session_id)
