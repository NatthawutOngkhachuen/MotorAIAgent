import os
from pathlib import Path
from dotenv import load_dotenv
from psycopg2 import pool as pg_pool

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

database_url = os.getenv("DATABASE_URL")

if database_url:
    _pool = pg_pool.SimpleConnectionPool(
        minconn=1,
        maxconn=5,
        dsn=database_url,
        sslmode=os.getenv("POSTGRES_SSLMODE", "require"),
    )
else:
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

    _pool = pg_pool.SimpleConnectionPool(
        minconn=1,
        maxconn=5,
        host=os.getenv("POSTGRES_HOST"),
        port=int(os.getenv("POSTGRES_PORT")),
        dbname=os.getenv("POSTGRES_DB"),
        user=os.getenv("POSTGRES_USER"),
        password=os.getenv("POSTGRES_PASSWORD"),
        sslmode=os.getenv("POSTGRES_SSLMODE", "prefer"),
    )

print("PostgreSQL connection pool created successfully")

def get_connection():
    return _pool.getconn()

def release_connection(conn):
    _pool.putconn(conn)
