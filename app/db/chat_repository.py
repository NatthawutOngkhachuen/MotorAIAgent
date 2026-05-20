import json
import psycopg2
import psycopg2.extras
from app.db.postgresql import get_connection, release_connection


SLOT_STATE_SOURCES = {
    "slot_filling",
    "user_based_slot_filling",
    "cluster_based_slot_filling",
}


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


def session_belongs_to_user(session_id: str, user_id: str) -> bool:
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT 1 FROM sessions WHERE session_id = %s AND user_id = %s",
                (session_id, user_id),
            )
            return cur.fetchone() is not None
    finally:
        release_connection(conn)


def list_sessions_by_user(user_id: str, limit: int = 50) -> list:
    conn = get_connection()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """
                SELECT
                    s.session_id::text AS session_id,
                    COALESCE(NULLIF(first_msg.content, ''), 'New chat') AS title,
                    s.created_at::text AS created_at,
                    s.last_active::text AS last_active
                FROM sessions s
                LEFT JOIN LATERAL (
                    SELECT cm.content
                    FROM chat_messages cm
                    WHERE cm.session_id = s.session_id
                      AND cm.user_id = %s
                      AND cm.role = 'user'
                    ORDER BY cm.created_at ASC
                    LIMIT 1
                ) first_msg ON TRUE
                WHERE s.user_id = %s
                ORDER BY s.last_active DESC NULLS LAST, s.created_at DESC
                LIMIT %s
                """,
                (user_id, user_id, limit),
            )
            return [dict(r) for r in cur.fetchall()]
    finally:
        release_connection(conn)


def delete_session_for_user(session_id: str, user_id: str) -> bool:
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT 1 FROM sessions WHERE session_id = %s AND user_id = %s",
                (session_id, user_id),
            )
            if cur.fetchone() is None:
                conn.rollback()
                return False

            cur.execute(
                "DELETE FROM chat_messages WHERE session_id = %s AND user_id = %s",
                (session_id, user_id),
            )
            cur.execute(
                "DELETE FROM sessions WHERE session_id = %s AND user_id = %s",
                (session_id, user_id),
            )
            conn.commit()
            return True
    except Exception:
        conn.rollback()
        raise
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


def load_all_messages(session_id: str, user_id: str) -> list:
    conn = get_connection()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                "SELECT role, content, created_at::text AS created_at "
                "FROM chat_messages "
                "WHERE session_id = %s AND user_id = %s "
                "ORDER BY created_at ASC",
                (session_id, user_id)
            )
            return [dict(r) for r in cur.fetchall()]
    finally:
        release_connection(conn)


def load_latest_slot_state(session_id: str, user_id: str) -> dict | None:
    conn = get_connection()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """
                SELECT rag_sources
                FROM chat_messages
                WHERE session_id = %s
                  AND user_id = %s
                  AND role = 'assistant'
                  AND rag_sources IS NOT NULL
                ORDER BY created_at DESC
                LIMIT 25
                """,
                (session_id, user_id),
            )
            for row in cur.fetchall():
                sources = row.get("rag_sources")
                if isinstance(sources, str):
                    try:
                        sources = json.loads(sources)
                    except json.JSONDecodeError:
                        continue
                state = extract_slot_state_from_sources(sources)
                if state:
                    return state
            return None
    finally:
        release_connection(conn)


def extract_slot_state_from_sources(sources: list | None) -> dict | None:
    if not isinstance(sources, list):
        return None
    for source in sources:
        if (
            isinstance(source, dict)
            and source.get("source") in SLOT_STATE_SOURCES
            and isinstance(source.get("state"), dict)
        ):
            return source["state"]
    return None
