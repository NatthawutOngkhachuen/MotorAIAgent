from __future__ import annotations

import json

import psycopg2.extras

from app.db.postgresql import get_connection, release_connection


SLOT_STATE_SOURCES = {
    "slot_filling",
    "user_based_slot_filling",
    "cluster_based_slot_filling",
}


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
