import os
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from app.db.neo4j import run_query
from app.db.chat_repository import (
    create_session,
    load_recent_messages,
    save_message,
    update_session_active,
)
from app.services.ollama_client import get_ollama_base_url, make_chat_ollama
from typing import AsyncGenerator
import time
import json

OLLAMA_MODEL  = os.getenv("OLLAMA_MODEL", "typhoon2")
OLLAMA_BASE_URL = get_ollama_base_url()
GUEST_USER_ID = "00000000-0000-0000-0000-000000000001"

_graph_cache: str | None = None


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


def build_system_prompt(language: str, graph_context: str,
                        already_recommended: list[str]) -> str:
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

    if language == "en":
        return f"""You are an expert Honda motorcycle sales consultant.

Guidelines:
- Answer EXACTLY what the customer asks
- If they ask "any other models?" recommend DIFFERENT models not yet mentioned
- If they say "not that model" exclude it completely
- Recommend 1-2 models with clear reasons WHY they suit the customer
- Ask follow-up if needed (budget, usage, rider type)
- Be warm and conversational
- Only use models from the database below
{already_str}

--- MOTORCYCLE DATABASE ---
{graph_context}
---------------------------"""

    return f"""คุณเป็นที่ปรึกษารถมอเตอร์ไซค์ Honda ที่เชี่ยวชาญและเป็นมิตร

แนวทาง:
- ตอบตรงคำถามที่ถามเสมอ
- ถ้าลูกค้าถามว่า "มีรุ่นอื่นไหม" แนะนำรุ่นที่ยังไม่ได้แนะนำเท่านั้น
- ถ้าลูกค้าบอกว่า "ไม่เอารุ่นนั้น" ห้ามแนะนำรุ่นนั้นอีกเลย
- แนะนำ 1-2 รุ่นพร้อมอธิบายว่าทำไมถึงเหมาะกับลูกค้า
- ถามเพิ่มถ้าข้อมูลไม่พอ เช่น งบประมาณ การใช้งาน เพศ
- ตอบภาษาไทย เป็นกันเอง อบอุ่น
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

    if not session_id:
        session_id = create_session(user_id)
        

    raw_context   = load_recent_messages(session_id, limit=6)
    graph_context = get_graph_context()
    all_models    = get_all_model_names()
    model_count   = graph_context.count("รุ่น:")

    already_recommended = extract_recommended_models(raw_context, all_models)

    history = []
    for m in raw_context:
        if m["role"] == "user":
            history.append(HumanMessage(content=m["content"]))
        else:
            history.append(AIMessage(content=m["content"]))

    llm_messages = [
        SystemMessage(content=build_system_prompt(
            language, graph_context, already_recommended
        )),
        *history,
        HumanMessage(content=question),
    ]

    llm = make_chat_ollama(
        model=OLLAMA_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=0.7,
        num_predict=400,
    )

    # ส่ง session_id และ model_count ก่อน
    yield f"data: {json.dumps({'type': 'session', 'session_id': session_id, 'model_count': model_count})}\n\n"

    start_time  = time.time()
    full_answer = ""

    # Stream ทีละ token
    async for chunk in llm.astream(llm_messages):
        token = chunk.content
        if token:
            full_answer += token
            yield f"data: {json.dumps({'type': 'token', 'token': token})}\n\n"

    elapsed = round(time.time() - start_time, 1)

    # บอก frontend ว่าเสร็จแล้ว พร้อมเวลาที่ใช้
    yield f"data: {json.dumps({'type': 'done', 'elapsed': elapsed})}\n\n"

    # บันทึกลง PostgreSQL
    save_message(session_id, user_id, "user", question)
    save_message(session_id,
                user_id, 
                "assistant",
                full_answer,
                rag_sources=[{"source": "neo4j", "model_count": model_count}])
    update_session_active(session_id)
