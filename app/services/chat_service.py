from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from app.db.neo4j import run_query
from app.db import postgresql as pg
import re

OLLAMA_MODEL  = "typhoon2"
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
            models[model] = {
                "features": [], "use_case": [],
                "safety": [], "style": [],
            }
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
    """ดึงชื่อรุ่นทั้งหมดจาก Neo4j"""
    rows = run_query("MATCH (m:VehicleModel) RETURN m.name AS name", {})
    return [r["name"] for r in rows if r.get("name")]


def extract_recommended_models(history: list, all_models: list[str]) -> list[str]:
    """หารุ่นที่ AI เคยแนะนำในประวัติแล้ว"""
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
            already_str = f"\n\n⚠️ รุ่นที่แนะนำไปแล้วในการสนทนานี้: {names}\nถ้าลูกค้าขอรุ่นอื่น ห้ามแนะนำรุ่นเหล่านี้ซ้ำอีก ให้เลือกรุ่นที่ยังไม่ได้แนะนำ"
        else:
            already_str = f"\n\n⚠️ Already recommended in this conversation: {names}\nIf customer asks for other models, do NOT repeat these. Suggest different ones."

    if language == "en":
        return f"""You are an expert Honda motorcycle sales consultant.

Guidelines:
- Answer EXACTLY what the customer asks
- If they ask "any other models?" → recommend DIFFERENT models not yet mentioned
- If they say "not that model" → exclude it completely
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
- ถ้าลูกค้าถามว่า "มีรุ่นอื่นไหม" → แนะนำรุ่นที่ยังไม่ได้แนะนำเท่านั้น
- ถ้าลูกค้าบอกว่า "ไม่เอารุ่นนั้น" → ห้ามแนะนำรุ่นนั้นอีกเลย
- แนะนำ 1-2 รุ่นพร้อมอธิบายว่าทำไมถึงเหมาะกับลูกค้า
- ถามเพิ่มถ้าข้อมูลไม่พอ เช่น งบประมาณ การใช้งาน เพศ
- ตอบภาษาไทย เป็นกันเอง อบอุ่น
- แนะนำเฉพาะรุ่นที่มีในฐานข้อมูลเท่านั้น
{already_str}

--- ฐานข้อมูลรถ ---
{graph_context}
-------------------"""


def answer_question(question: str, language: str = "th",
                    session_id: str = None, user_id: str = None) -> dict:
    user_id = user_id or GUEST_USER_ID

    # 1. Session
    if not session_id:
        session_id = pg.create_session(user_id)

    # 2. ประวัติ 6 ข้อความล่าสุด
    raw_context = pg.load_recent_messages(session_id, limit=6)

    # 3. Graph context + ชื่อรุ่นทั้งหมด
    graph_context  = get_graph_context()
    all_models     = get_all_model_names()
    model_count    = graph_context.count("รุ่น:")

    # 4. หารุ่นที่แนะนำไปแล้วในประวัติ
    already_recommended = extract_recommended_models(raw_context, all_models)

    # 5. สร้าง history messages — เก็บแค่ข้อความจริง
    history = []
    for m in raw_context:
        content = m["content"]
        if m["role"] == "user":
            history.append(HumanMessage(content=content))
        else:
            history.append(AIMessage(content=content))

    # 6. รวม messages
    llm_messages = [
        SystemMessage(content=build_system_prompt(
            language, graph_context, already_recommended
        )),
        *history,
        HumanMessage(content=question),
    ]

    # 7. LLM
    llm = ChatOllama(
        model=OLLAMA_MODEL,
        temperature=0.7,
        num_predict=400,
    )
    response = llm.invoke(llm_messages)
    answer   = response.content

    # 8. บันทึก
    pg.save_message(session_id, user_id, "user",      question)
    pg.save_message(session_id, user_id, "assistant", answer,
                    rag_sources=[{"source": "neo4j", "model_count": model_count}])
    pg.update_session_active(session_id)

    return {
        "session_id":    session_id,
        "question":      question,
        "answer":        answer,
        "context_nodes": model_count,
        "context_edges": 0,
    }