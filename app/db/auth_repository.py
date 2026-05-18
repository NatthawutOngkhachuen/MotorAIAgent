import psycopg2.extras
from app.db.postgresql import get_connection, release_connection


def find_auth_by_username(username: str):
    conn = get_connection()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """
                SELECT
                    a.id AS auth_account_id,
                    a.user_id::text AS user_id,
                    a.username,
                    a.password
                FROM auth_account a
                WHERE a.username = %s
                """,
                (username,)
            )
            return cur.fetchone()
    finally:
        release_connection(conn)


def create_auth_account(user_id: str, username: str, password_hash: str):
    conn = get_connection()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """
                INSERT INTO auth_account (user_id, username, password)
                VALUES (%s, %s, %s)
                RETURNING id AS auth_account_id, user_id::text, username
                """,
                (user_id, username, password_hash)
            )
            result = cur.fetchone()
            conn.commit()
            return result
    except Exception:
        conn.rollback()
        raise
    finally:
        release_connection(conn)