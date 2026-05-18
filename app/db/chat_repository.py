import json
import psycopg2
import psycopg2.extras
from app.db.postgresql import get_connection, release_connection


def create_session(user_id: str) -> str:
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO sessions (user_id) VALUES (%s) RETURNING session_id::text",
                (user_id,)
            )
            conn.commit()
            return cur.fetchone()[0]
    finally:
        release_connection(conn)


def update_session_active(session_id: str):
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE sessions SET last_active = NOW() WHERE session_id = %s",
                (session_id,)
            )
            conn.commit()
    finally:
        release_connection(conn)


def save_message(session_id: str, user_id: str, role: str,
                 content: str, rag_sources: list = None):
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO chat_messages (session_id, user_id, role, content, rag_sources) "
                "VALUES (%s, %s, %s, %s, %s) RETURNING id::text",
                (session_id, user_id, role, content,
                 json.dumps(rag_sources or [], ensure_ascii=False))
            )
            conn.commit()
            return cur.fetchone()[0]
    finally:
        release_connection(conn)


def load_recent_messages(session_id: str, limit: int = 4) -> list:
    sql = """
        SELECT role, content FROM (
            SELECT role, content, created_at
            FROM chat_messages
            WHERE session_id = %s
            ORDER BY created_at DESC
            LIMIT %s
        ) sub ORDER BY created_at ASC
    """
    conn = get_connection()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(sql, (session_id, limit))
            return [dict(r) for r in cur.fetchall()]
    finally:
        release_connection(conn)


def load_all_messages(session_id: str) -> list:
    conn = get_connection()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                "SELECT role, content, created_at::text AS created_at "
                "FROM chat_messages WHERE session_id = %s ORDER BY created_at ASC",
                (session_id,)
            )
            return [dict(r) for r in cur.fetchall()]
    finally:
        release_connection(conn)