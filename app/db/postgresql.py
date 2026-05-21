import os
from pathlib import Path
from dotenv import load_dotenv
from psycopg2 import pool as pg_pool

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

_pool = None


def _create_pool():
    database_url = os.getenv("DATABASE_URL")

    if database_url:
        return pg_pool.SimpleConnectionPool(
            minconn=1,
            maxconn=5,
            dsn=database_url,
            sslmode=os.getenv("POSTGRES_SSLMODE", "require"),
        )

    required_env = [
        "POSTGRES_HOST",
        "POSTGRES_PORT",
        "POSTGRES_DB",
        "POSTGRES_USER",
        "POSTGRES_PASSWORD",
    ]

    missing = [key for key in required_env if not os.getenv(key)]
    if missing:
        raise RuntimeError(f"Missing required environment variables: {', '.join(missing)}")

    return pg_pool.SimpleConnectionPool(
        minconn=1,
        maxconn=5,
        host=os.getenv("POSTGRES_HOST"),
        port=int(os.getenv("POSTGRES_PORT")),
        dbname=os.getenv("POSTGRES_DB"),
        user=os.getenv("POSTGRES_USER"),
        password=os.getenv("POSTGRES_PASSWORD"),
        sslmode=os.getenv("POSTGRES_SSLMODE", "prefer"),
    )


def _get_pool():
    global _pool

    if _pool is None:
        _pool = _create_pool()
        print("PostgreSQL connection pool created successfully")
    return _pool


def get_connection():
    return _get_pool().getconn()

def release_connection(conn):
    _get_pool().putconn(conn)
