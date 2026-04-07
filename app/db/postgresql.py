import os
import json
import psycopg2
import psycopg2.extras
from dotenv import load_dotenv
from pathlib import Path
load_dotenv(Path(__file__).resolve().parents[2] / ".env")

def get_connection():
    return psycopg2.connect(
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=int(os.getenv("POSTGRES_PORT", 5432)),
        dbname=os.getenv("POSTGRES_DB", "motorcycle_chatbot"),
        user=os.getenv("POSTGRES_USER", "postgres"),
        password=os.getenv("POSTGRES_PASSWORD", "postgres"),
    )


# ─── Sessions ─────────────────────────────────────────────────────────────────

def create_session(user_id: str) -> str:
    """สร้าง session ใหม่ คืน session_id"""
    sql = """
        INSERT INTO sessions (user_id)
        VALUES (%s)
        RETURNING session_id::text
    """
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (user_id,))
            return cur.fetchone()[0]


def update_session_active(session_id: str):
    """อัปเดต last_active ทุกครั้งที่มีข้อความใหม่"""
    sql = "UPDATE sessions SET last_active = NOW() WHERE session_id = %s"
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (session_id,))


# ─── Messages ─────────────────────────────────────────────────────────────────

def save_message(session_id: str, user_id: str, role: str,
                 content: str, rag_sources: list = None):
    """บันทึก 1 ข้อความลง chat_messages"""
    sql = """
        INSERT INTO chat_messages (session_id, user_id, role, content, rag_sources)
        VALUES (%s, %s, %s, %s, %s)
        RETURNING id::text
    """
    sources_json = json.dumps(rag_sources or [], ensure_ascii=False)
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (session_id, user_id, role, content, sources_json))
            return cur.fetchone()[0]


def load_recent_messages(session_id: str, limit: int = 15) -> list:
    """โหลด N ข้อความล่าสุดของ session (เรียงเก่า → ใหม่)"""
    sql = """
        SELECT role, content
        FROM (
            SELECT role, content, created_at
            FROM chat_messages
            WHERE session_id = %s
            ORDER BY created_at DESC
            LIMIT %s
        ) sub
        ORDER BY created_at ASC
    """
    with get_connection() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(sql, (session_id, limit))
            return [dict(r) for r in cur.fetchall()]


def load_all_messages(session_id: str) -> list:
    """โหลดประวัติทั้งหมดของ session (สำหรับแสดง UI)"""
    sql = """
        SELECT role, content, created_at::text AS created_at
        FROM chat_messages
        WHERE session_id = %s
        ORDER BY created_at ASC
    """
    with get_connection() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(sql, (session_id,))
            return [dict(r) for r in cur.fetchall()]