#!/usr/bin/env python
"""
Simple concurrent load test for prediction endpoint.

Example:
python scripts/load_test_predict.py --url http://127.0.0.1:8000/api/predict/ --requests 100 --concurrency 10 --cookie "sessionid=..."
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed


def _one_request(url: str, payload: dict, cookie: str = ""):
    started = time.perf_counter()
    req = urllib.request.Request(
        url=url,
        data=json.dumps(payload).encode("utf-8"),
        method="POST",
        headers={
            "Content-Type": "application/json",
            **({"Cookie": cookie} if cookie else {}),
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            body = resp.read().decode("utf-8", errors="ignore")
            status = int(resp.status)
    except Exception as exc:
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        return {"ok": False, "status": 0, "ms": elapsed_ms, "error": str(exc)}

    elapsed_ms = (time.perf_counter() - started) * 1000.0
    ok = 200 <= status < 300
    return {"ok": ok, "status": status, "ms": elapsed_ms, "body": body[:200]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--requests", type=int, default=50)
    parser.add_argument("--concurrency", type=int, default=10)
    parser.add_argument("--cookie", default="")
    parser.add_argument(
        "--payload",
        default='{"semester":1,"subject":"Data Structures","assignment_marks":4,"quiz_marks":8,"attendance_percentage":90,"midterm_marks":22,"previous_cgpa":3.2}',
    )
    args = parser.parse_args()

    payload = json.loads(args.payload)

    results = []
    with ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as ex:
        futures = [
            ex.submit(_one_request, args.url, payload, args.cookie)
            for _ in range(max(1, args.requests))
        ]
        for fut in as_completed(futures):
            results.append(fut.result())

    latencies = [r["ms"] for r in results]
    ok_count = sum(1 for r in results if r["ok"])
    err_count = len(results) - ok_count
    p50 = statistics.median(latencies) if latencies else 0.0
    p95_idx = max(0, min(len(latencies) - 1, int(len(latencies) * 0.95) - 1))
    p95 = sorted(latencies)[p95_idx] if latencies else 0.0

    report = {
        "total_requests": len(results),
        "success": ok_count,
        "errors": err_count,
        "success_rate": round(ok_count / max(1, len(results)), 4),
        "latency_ms": {
            "mean": round(sum(latencies) / max(1, len(latencies)), 2),
            "p50": round(p50, 2),
            "p95": round(p95, 2),
            "max": round(max(latencies) if latencies else 0.0, 2),
        },
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
