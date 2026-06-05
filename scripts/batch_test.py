"""Concurrent batch test — fires all students in test.csv to the /tutor API in parallel."""

import asyncio
import csv
import time
from collections import defaultdict
from pathlib import Path

import httpx

API_URL  = "http://3.38.221.216:8000/tutor"
CSV_PATH = Path(__file__).parent.parent / "test.csv"
TIMEOUT  = 600  # seconds per request


def build_payloads(csv_path: Path) -> list[dict]:
    rows: dict[str, list] = defaultdict(list)
    names: dict[str, dict[str, str]] = defaultdict(dict)

    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            uid = row["user_id"]
            rows[uid].append((row["Timestamp"], int(row["skill_id"]), int(row["correct"])))
            names[uid][str(row["skill_id"])] = row["name"]

    payloads = []
    for uid, events in rows.items():
        events.sort(key=lambda x: x[0])
        sequence = [[sid, correct] for _, sid, correct in events]
        payloads.append({
            "student_id":       uid,
            "sequence":         sequence,
            "skill_id_to_name": names[uid],
        })
    return payloads


async def call_one(client: httpx.AsyncClient, payload: dict) -> tuple:
    start = time.monotonic()
    try:
        r = await client.post(API_URL, json=payload, timeout=TIMEOUT)
        status = r.status_code
        n_feedback = len(r.json().get("feedback", [])) if status == 200 else 0
    except Exception as e:
        status = f"ERR: {e}"
        n_feedback = 0
    elapsed = time.monotonic() - start
    return payload["student_id"], status, elapsed, n_feedback


async def main():
    payloads = build_payloads(CSV_PATH)
    print(f"Firing {len(payloads)} students concurrently at {API_URL}")

    limits = httpx.Limits(max_connections=120, max_keepalive_connections=50)
    async with httpx.AsyncClient(limits=limits) as client:
        t0 = time.monotonic()
        results = await asyncio.gather(*[call_one(client, p) for p in payloads])
        wall = time.monotonic() - t0

    ok      = [r for r in results if r[1] == 200]
    errors  = [r for r in results if r[1] != 200]
    latencies = sorted([r[2] for r in ok])
    p50 = latencies[len(latencies) // 2]     if latencies else 0
    p95 = latencies[int(len(latencies) * 0.95)] if latencies else 0

    print(f"\n=== Results ===")
    print(f"Total wall time : {wall:.1f}s")
    print(f"Success         : {len(ok)} / {len(payloads)}")
    print(f"p50 latency     : {p50:.1f}s")
    print(f"p95 latency     : {p95:.1f}s")
    if errors:
        print(f"\nErrors ({len(errors)}):")
        for sid, status, elapsed, _ in errors:
            print(f"  {sid}: {status} ({elapsed:.1f}s)")

    print("\nPer-student latency (sorted):")
    for sid, status, elapsed, n_fb in sorted(results, key=lambda x: x[2]):
        mark = "OK" if status == 200 else "!!"
        print(f"  [{mark}] {sid}: {elapsed:.1f}s  feedback={n_fb}")


if __name__ == "__main__":
    asyncio.run(main())
