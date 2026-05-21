import psycopg2.extras
from app.db.postgresql import get_connection, release_connection


def find_user_by_id(user_id: str):
    conn = get_connection()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """
                SELECT id::text, name, age, gender, created_at::text
                FROM users
                WHERE id = %s
                """,
                (user_id,)
            )
            return cur.fetchone()
    finally:
        release_connection(conn)

