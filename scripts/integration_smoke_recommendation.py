from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.services.recommendation_chat_service import stream_recommendation_answer


GUEST_USER_ID = "00000000-0000-0000-0000-000000000001"


async def collect_events(question: str, session_id: str | None = None) -> list[dict]:
    events: list[dict] = []
    async for chunk in stream_recommendation_answer(
        question=question,
        session_id=session_id,
        user_id=GUEST_USER_ID,
    ):
        for part in chunk.strip().split("\n\n"):
            if not part.startswith("data: "):
                continue
            events.append(json.loads(part[6:]))
    return events


async def main() -> None:
    started = time.perf_counter()
    all_events: list[dict] = []

    start_events = await collect_events("")
    all_events.extend(start_events)
    session_id = next(event["session_id"] for event in start_events if event["type"] == "session")
    first_text = "".join(event.get("token", "") for event in start_events if event["type"] == "token")
    print("START_SESSION", session_id)
    print("START_TEXT", first_text)

    messages = [
        "งบไม่แพงมาก ใช้ไปเรียนทุกวันในเมือง ชอบรถสปอร์ตเท่ๆ",
        "ขอแรงพอประมาณ นั่งสบาย ปลอดภัยสูง ฟีเจอร์เยอะ",
        "ใช่ แล้วก็ไม่ต้องมีที่เก็บของ แต่ขอดูแลง่าย",
    ]

    for index, message in enumerate(messages, start=1):
        events = await collect_events(message, session_id=session_id)
        all_events.extend(events)
        text = "".join(event.get("token", "") for event in events if event["type"] == "token")
        metadata = [event for event in events if event["type"] == "metadata"]
        print(f"ROUND_{index}_TEXT", text.replace("\n", " ")[:500])
        if metadata:
            print(f"ROUND_{index}_STAGE", metadata[-1].get("stage"))

    elapsed = time.perf_counter() - started
    final_metadata = [event for event in all_events if event["type"] == "metadata"][-1]
    print("FINAL_STAGE", final_metadata.get("stage"))
    print("FINAL_CANDIDATE_COUNT", final_metadata.get("candidate_count"))
    print("FINAL_GRAPH_ITEM_IDS", final_metadata.get("graph_item_ids"))
    print("ELAPSED_SECONDS", round(elapsed, 2))

    if elapsed > 30:
        raise AssertionError(f"Integration smoke exceeded 30 seconds: {elapsed:.2f}s")
    if final_metadata.get("stage") != "recommendation":
        raise AssertionError("Flow did not reach recommendation stage")


if __name__ == "__main__":
    os.environ.setdefault("DISABLE_RECOMMENDATION_LLM", "true")
    asyncio.run(main())
