import psycopg2.extras
from app.db.postgresql import get_connection, release_connection


def create_user_with_auth(username: str, password_hash: str, name: str, age: int = None, gender: int = None):
    conn = get_connection()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            # เช็ก username ซ้ำ
            cur.execute(
                "SELECT id FROM auth_account WHERE username = %s",
                (username,)
            )

            if cur.fetchone():
                raise ValueError("Username already exists")

            # สร้าง user profile
            cur.execute(
                """
                INSERT INTO users (name, age, gender)
                VALUES (%s, %s, %s)
                RETURNING id::text, name, age, gender
                """,
                (name, age, gender)
            )
            user = cur.fetchone()

            # สร้าง auth account
            cur.execute(
                """
                INSERT INTO auth_account (user_id, username, password)
                VALUES (%s, %s, %s)
                RETURNING id AS auth_account_id, username
                """,
                (user["id"], username, password_hash)
            )
            auth = cur.fetchone()

            conn.commit()

            return {
                "user_id": user["id"],
                "auth_account_id": auth["auth_account_id"],
                "username": auth["username"],
                "name": user["name"],
                "age": user["age"],
                "gender": user["gender"],
            }

    except Exception:
        conn.rollback()
        raise
    finally:
        release_connection(conn)