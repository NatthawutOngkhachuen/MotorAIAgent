import os
import json
import redis
from dotenv import load_dotenv

load_dotenv()

REDIS_URL    = os.getenv("REDIS_URL", "redis://localhost:6379")
TTL_SECONDS  = int(os.getenv("REDIS_TTL", 1800))  # 30 นาที
MAX_MESSAGES = 15

_client = redis.from_url(REDIS_URL, decode_responses=True)


def _ctx_key(session_id: str) -> str:
    return f"chat:ctx:{session_id}"


def get_context(session_id: str) -> list | None:
    """ดึง context จาก Redis  คืน None ถ้าไม่มี (Cache Miss)"""
    key = _ctx_key(session_id)
    raw = _client.lrange(key, 0, MAX_MESSAGES - 1)
    if not raw:
        return None
    return [json.loads(m) for m in reversed(raw)]


def push_message(session_id: str, role: str, content: str):
    """เพิ่มข้อความใหม่ + รีเซ็ต TTL 30 นาที"""
    key  = _ctx_key(session_id)
    msg  = json.dumps({"role": role, "content": content}, ensure_ascii=False)
    pipe = _client.pipeline()
    pipe.lpush(key, msg)
    pipe.ltrim(key, 0, MAX_MESSAGES - 1)
    pipe.expire(key, TTL_SECONDS)
    pipe.execute()


def seed_from_db(session_id: str, messages: list):
    """Cache Miss → เอาข้อมูลจาก DB มา warm ใน Redis"""
    if not messages:
        return
    key  = _ctx_key(session_id)
    pipe = _client.pipeline()
    for msg in messages:
        pipe.lpush(key, json.dumps(
            {"role": msg["role"], "content": msg["content"]},
            ensure_ascii=False
        ))
    pipe.ltrim(key, 0, MAX_MESSAGES - 1)
    pipe.expire(key, TTL_SECONDS)
    pipe.execute()


def delete_context(session_id: str):
    """ลบ context ของ session (เช่น เมื่อ user กด New Chat)"""
    _client.delete(_ctx_key(session_id))


def ping() -> bool:
    """ตรวจว่า Redis ยังเชื่อมต่ออยู่"""
    try:
        return _client.ping()
    except Exception:
        return False