import os
import json
import time
from collections.abc import AsyncGenerator

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

from app.db import postgresql as pg
from app.db.neo4j import run_query

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "typhoon2")
GUEST_USER_ID = "00000000-0000-0000-0000-000000000001"
RECENT_MESSAGE_LIMIT = 6
LLM_NUM_PREDICT = 400

_graph_cache: str | None = None


def _unique_in_order(values: list[str], limit: int | None = None) -> list[str]:
    seen: set[str] = set()
    unique_values: list[str] = []

    for value in values:
        if value in seen:
            continue
        seen.add(value)
        unique_values.append(value)

        if limit and len(unique_values) >= limit:
            break

    return unique_values


def _format_values(values: list[str], limit: int | None = None) -> str:
    return ", ".join(_unique_in_order(values, limit=limit))


def _sse_event(event_type: str, **payload: object) -> str:
    return f"data: {json.dumps({'type': event_type, **payload}, ensure_ascii=False)}\n\n"


def _build_chat_history(raw_context: list[dict]) -> list[HumanMessage | AIMessage]:
    history: list[HumanMessage | AIMessage] = []

    for message in raw_context:
        content = message["content"]
        if message["role"] == "user":
            history.append(HumanMessage(content=content))
        else:
            history.append(AIMessage(content=content))

    return history


def _build_llm_messages(
    question: str,
    language: str,
    graph_context: str,
    raw_context: list[dict],
    all_models: list[str],
) -> list[SystemMessage | HumanMessage | AIMessage]:
    already_recommended = extract_recommended_models(raw_context, all_models)

    return [
        SystemMessage(
            content=build_system_prompt(
                language,
                graph_context,
                already_recommended,
            )
        ),
        *_build_chat_history(raw_context),
        HumanMessage(content=question),
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

    models: dict[str, dict[str, list[str]]] = {}
    for row in rows:
        model = row.get("model") or "?"
        if model not in models:
            models[model] = {"features": [], "use_case": [], "safety": [], "style": []}
        rel1 = (row.get("rel1") or "").upper()
        entity = row.get("entity_name") or ""
        semantic = row.get("semantic_name") or ""
        label = f"{entity} ({semantic})" if semantic else entity
        if not label:
            continue

        if "USE_CASE" in rel1:
            models[model]["use_case"].append(label)
        elif "SAFETY" in rel1:
            models[model]["safety"].append(label)
        elif "STYLE" in rel1:
            models[model]["style"].append(label)
        else:
            models[model]["features"].append(label)

    lines = []
    for name, details in models.items():
        parts = [f"รุ่น: {name}"]
        if details["use_case"]:
            parts.append(f"  เหมาะกับ: {_format_values(details['use_case'])}")
        if details["style"]:
            parts.append(f"  สไตล์: {_format_values(details['style'])}")
        if details["safety"]:
            parts.append(f"  ความปลอดภัย: {_format_values(details['safety'])}")
        if details["features"]:
            parts.append(f"  คุณสมบัติ: {_format_values(details['features'], limit=6)}")
        lines.append("\n".join(parts))

    _graph_cache = "ข้อมูลรถมอเตอร์ไซค์:\n\n" + "\n\n".join(lines)
    return _graph_cache


def clear_graph_cache():
    global _graph_cache
    _graph_cache = None


def get_all_model_names() -> list[str]:
    rows = run_query("MATCH (m:VehicleModel) RETURN m.name AS name", {})
    return [r["name"] for r in rows if r.get("name")]


def extract_recommended_models(history: list[dict], all_models: list[str]) -> list[str]:
    recommended = []

    for message in history:
        if message["role"] == "assistant":
            for model_name in all_models:
                if (
                    model_name.lower() in message["content"].lower()
                    and model_name not in recommended
                ):
                    recommended.append(model_name)

    return recommended


def build_system_prompt(
    language: str,
    graph_context: str,
    already_recommended: list[str],
) -> str:
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


async def stream_answer(
    question: str,
    language: str = "th",
    session_id: str | None = None,
    user_id: str | None = None,
) -> AsyncGenerator[str, None]:
    user_id = user_id or GUEST_USER_ID

    if not session_id:
        session_id = pg.create_session(user_id)

    raw_context = pg.load_recent_messages(session_id, limit=RECENT_MESSAGE_LIMIT)
    pg.save_message(session_id, user_id, "user", question)

    graph_context = get_graph_context()
    all_models = get_all_model_names()
    model_count = graph_context.count("รุ่น:")

    llm_messages = _build_llm_messages(
        question=question,
        language=language,
        graph_context=graph_context,
        raw_context=raw_context,
        all_models=all_models,
    )

    llm = ChatOllama(model=OLLAMA_MODEL, temperature=0.7, num_predict=LLM_NUM_PREDICT)

    # ส่ง session_id และ model_count ก่อน
    yield _sse_event("session", session_id=session_id, model_count=model_count)

    start_time = time.time()
    full_answer = ""

    try:
        # Stream ทีละ token
        async for chunk in llm.astream(llm_messages):
            token = chunk.content
            if token:
                full_answer += token
                yield _sse_event("token", token=token)

        elapsed = round(time.time() - start_time, 1)

        # บอก frontend ว่าเสร็จแล้ว พร้อมเวลาที่ใช้
        yield _sse_event("done", elapsed=elapsed)
    finally:
        if full_answer:
            pg.save_message(
                session_id,
                user_id,
                "assistant",
                full_answer,
                rag_sources=[{"source": "neo4j", "model_count": model_count}],
            )
        pg.update_session_active(session_id)
