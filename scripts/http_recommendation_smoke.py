from __future__ import annotations

import json
import argparse
import sys
import time
from pathlib import Path
from urllib.request import Request, urlopen

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.services.jwt_service import create_access_token


API_ROOT = "http://127.0.0.1:8000/api/v1/recommendation"
GUEST_USER_ID = "00000000-0000-0000-0000-000000000001"


def parse_sse(text: str) -> list[dict]:
    events = []
    for part in text.strip().split("\n\n"):
        lines = [line for line in part.splitlines() if line.startswith("data: ")]
        if not lines:
            continue
        events.append(json.loads("\n".join(line[6:] for line in lines)))
    return events


def post(base_url: str, path: str, payload: dict | None = None) -> list[dict]:
    token, _ = create_access_token(GUEST_USER_ID)
    body = json.dumps(payload or {}, ensure_ascii=False).encode("utf-8")
    request = Request(
        base_url + path,
        data=body,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urlopen(request, timeout=60) as response:
        return parse_sse(response.read().decode("utf-8"))


def run_flow(mode: str) -> None:
    started = time.perf_counter()
    base_url = f"{API_ROOT}/{mode}"
    start_events = post(base_url, "/start")
    session_id = next(event["session_id"] for event in start_events if event["type"] == "session")
    print(f"HTTP_{mode}_START_SESSION", session_id)
    print(f"HTTP_{mode}_START_TEXT", "".join(e.get("token", "") for e in start_events if e["type"] == "token"))

    messages = [
        "งบไม่แพงมาก ใช้ไปเรียนทุกวันในเมือง ชอบรถสปอร์ตเท่ๆ",
        "ขอแรงพอประมาณ นั่งสบาย ปลอดภัยสูง ฟีเจอร์เยอะ",
        "ใช่ แล้วก็ไม่ต้องมีที่เก็บของ แต่ขอดูแลง่าย",
    ]
    final_events = []
    for index, message in enumerate(messages, start=1):
        events = post(base_url, "/chat", {"question": message, "language": "th", "session_id": session_id})
        final_events = events
        text = "".join(e.get("token", "") for e in events if e["type"] == "token")
        metadata = [e for e in events if e["type"] == "metadata"]
        print(f"HTTP_{mode}_ROUND_{index}_TEXT", text.replace("\n", " ")[:500])
        if metadata:
            print(f"HTTP_{mode}_ROUND_{index}_STAGE", metadata[-1].get("stage"))

    elapsed = round(time.perf_counter() - started, 2)
    final_metadata = [e for e in final_events if e["type"] == "metadata"][-1]
    print(f"HTTP_{mode}_FINAL_MODE", final_metadata.get("recommendation_mode"))
    print(f"HTTP_{mode}_FINAL_STAGE", final_metadata.get("stage"))
    print(f"HTTP_{mode}_FINAL_CANDIDATE_COUNT", final_metadata.get("candidate_count"))
    print(f"HTTP_{mode}_FINAL_GRAPH_ITEM_IDS", final_metadata.get("graph_item_ids"))
    print(f"HTTP_{mode}_ELAPSED_SECONDS", elapsed)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["user-based", "cluster-based", "both"], default="both")
    args = parser.parse_args()

    modes = ["user-based", "cluster-based"] if args.mode == "both" else [args.mode]
    for mode in modes:
        run_flow(mode)


if __name__ == "__main__":
    main()
