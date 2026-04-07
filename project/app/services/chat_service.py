from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage
from app.db.neo4j import run_query

OLLAMA_MODEL = "typhoon2"


def search_vehicles_from_graph() -> str:
    """ดึงข้อมูลรถทั้งหมดจาก Neo4j"""
    cypher = """
        MATCH (m:VehicleModel)
        OPTIONAL MATCH (m)-[r1]->(entity)
        OPTIONAL MATCH (entity)-[r2]->(semantic)
        RETURN
            m.name AS model,
            type(r1) AS rel1,
            entity.name AS entity_name,
            labels(entity) AS entity_labels,
            type(r2) AS rel2,
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
                "style": [], "use_case": [], "performance": [],
                "comfort": [], "safety": [], "storage": [],
                "brand": [], "decision_factor": [], "other": [],
            }

        rel1     = (row.get("rel1") or "").upper()
        entity   = row.get("entity_name") or ""
        semantic = row.get("semantic_name") or ""
        entry    = f"{entity} ({semantic})" if semantic else entity

        if   "STYLE"      in rel1: models[model]["style"].append(entry)
        elif "USE_CASE"   in rel1: models[model]["use_case"].append(entry)
        elif "PERFORMANCE"in rel1: models[model]["performance"].append(entry)
        elif "COMFORT"    in rel1: models[model]["comfort"].append(entry)
        elif "SAFETY"     in rel1: models[model]["safety"].append(entry)
        elif "STORAGE"    in rel1: models[model]["storage"].append(entry)
        elif "BRAND"      in rel1: models[model]["brand"].append(entry)
        elif "DECISION"   in rel1: models[model]["decision_factor"].append(entry)
        elif rel1:                 models[model]["other"].append(f"{rel1}: {entry}")

    lines = ["ข้อมูลรถมอเตอร์ไซค์ในฐานข้อมูล:\n"]
    for name, d in models.items():
        lines.append(f"รุ่น: {name}")
        if d["brand"]:           lines.append(f"  แบรนด์: {', '.join(set(d['brand']))}")
        if d["style"]:           lines.append(f"  สไตล์: {', '.join(set(d['style']))}")
        if d["use_case"]:        lines.append(f"  การใช้งาน: {', '.join(set(d['use_case']))}")
        if d["performance"]:     lines.append(f"  สมรรถนะ: {', '.join(set(d['performance']))}")
        if d["comfort"]:         lines.append(f"  ความสะดวกสบาย: {', '.join(set(d['comfort']))}")
        if d["safety"]:          lines.append(f"  ความปลอดภัย: {', '.join(set(d['safety']))}")
        if d["storage"]:         lines.append(f"  พื้นที่เก็บของ: {', '.join(set(d['storage']))}")
        if d["decision_factor"]: lines.append(f"  ปัจจัยการตัดสินใจ: {', '.join(set(d['decision_factor']))}")
        if d["other"]:           lines.append(f"  อื่นๆ: {', '.join(set(d['other']))}")
        lines.append("")
    return "\n".join(lines)


def build_system_prompt(language: str) -> str:
    if language == "en":
        return """You are a Honda motorcycle recommendation expert.
Rules:
1. Analyze the user's needs (gender, style, usage, budget).
2. Recommend 2-3 models based strictly on the provided database.
3. Explain why each model suits the user.
4. Reply in English only. Never use Thai language.
5. If data is insufficient, ask a follow-up question.
6. Never recommend models not in the database."""
    else:
        return """[INST] ตอบเป็นภาษาไทยเท่านั้น [/INST]
คุณเป็นผู้เชี่ยวชาญแนะนำรถมอเตอร์ไซค์ Honda
ตอบเป็นภาษาไทยเท่านั้น ห้ามตอบเป็นภาษาอังกฤษ
กฎการตอบ:
1. วิเคราะห์ความต้องการ เช่น เพศ สไตล์ การใช้งาน งบประมาณ
2. แนะนำรถ 2-3 รุ่น โดยอิงจากข้อมูลในฐานข้อมูลเท่านั้น
3. บอกเหตุผลว่าทำไมถึงเหมาะกับผู้ใช้
4. ตอบเป็นภาษาไทย กระชับ เป็นมิตร
5. ถ้าข้อมูลไม่เพียงพอ ให้ถามเพิ่มเติม
6. ห้ามแนะนำรถที่ไม่มีในฐานข้อมูล"""


def answer_question(question: str, language: str = "th") -> dict:
    """Pipeline หลัก: ดึง Neo4j → LangChain + Ollama → คำตอบ"""
    context = search_vehicles_from_graph()

    llm = ChatOllama(model=OLLAMA_MODEL, temperature=0.7)

    # บังคับภาษาใน HumanMessage ด้วย
    lang_instruction = "ตอบเป็นภาษาไทยเท่านั้น" if language == "th" else "Reply in English only"

    messages = [
        SystemMessage(content=build_system_prompt(language)),
        HumanMessage(content=(
            f"ข้อมูลรถในฐานข้อมูล:\n{context}\n\n"
            f"คำถามผู้ใช้: {question}\n\n"
            f"[{lang_instruction}]"
        )),
    ]

    response = llm.invoke(messages)
    model_count = context.count("รุ่น:")

    return {
        "question": question,
        "answer": response.content,
        "context_nodes": model_count,
        "context_edges": 0,
    }


def answer_question(question: str, language: str = "th") -> dict:
    context = search_vehicles_from_graph()
    llm = ChatOllama(model=OLLAMA_MODEL, temperature=0.7)

    if language == "th":
        user_content = (
            f"ข้อมูลรถในฐานข้อมูล:\n{context}\n\n"
            f"คำถามผู้ใช้: {question}\n\n"
            f"กรุณาตอบคำถามนี้เป็นภาษาไทยเท่านั้น"
        )
    else:
        user_content = (
            f"Vehicle database:\n{context}\n\n"
            f"User question: {question}\n\n"
            f"Please answer in English only."
        )

    messages = [
        SystemMessage(content=build_system_prompt(language)),
        HumanMessage(content=user_content),
    ]

    response = llm.invoke(messages)
    model_count = context.count("รุ่น:")

    return {
        "question": question,
        "answer": response.content,
        "context_nodes": model_count,
        "context_edges": 0,
    }
