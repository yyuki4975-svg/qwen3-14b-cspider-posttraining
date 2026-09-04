#!/usr/bin/env python3
"""Evaluate a Base model on the untouched CSpider gate via OpenAI API.

The gate uses deterministic decoding and executable SQL result comparison.  It
is a fast go/no-go screen; official Spider/CSpider evaluation remains the final
metric after training.
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import os
import re
import sqlite3
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


def args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--questions", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--base-url", default="http://127.0.0.1:8121/v1")
    p.add_argument("--model", required=True)
    p.add_argument("--api-key-env", default="CSPIDER_API_KEY")
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--timeout", type=int, default=300)
    p.add_argument("--retries", type=int, default=2)
    return p.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def clean_sql(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.I | re.S).strip()
    fenced = re.search(r"```(?:sql)?\s*(.*?)```", text, flags=re.I | re.S)
    if fenced:
        text = fenced.group(1).strip()
    text = re.sub(r"^(?:SQL|答案)\s*[:：]\s*", "", text, flags=re.I).strip()
    return text.rstrip(";").strip()


def request_chat(a: argparse.Namespace, messages: list[dict[str, str]]) -> dict[str, Any]:
    key = os.environ.get(a.api_key_env, "")
    if not key:
        raise RuntimeError(f"Environment variable {a.api_key_env} is empty")
    payload = {
        "model": a.model,
        "messages": messages[:2],
        "temperature": 0,
        "top_p": 1,
        "max_tokens": a.max_tokens,
        "stream": False,
    }
    req = urllib.request.Request(
        a.base_url.rstrip("/") + "/chat/completions",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=a.timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def norm_cell(x: Any) -> Any:
    if x is None:
        return None
    if isinstance(x, float):
        if math.isnan(x):
            return "nan"
        return round(x, 8)
    if isinstance(x, bytes):
        return x.hex()
    return str(x).strip() if isinstance(x, str) else x


def run_sql(path: Path, query: str) -> tuple[bool, str, list[tuple[Any, ...]]]:
    q = query.strip()
    if not q.lower().startswith(("select", "with")) or ";" in q.rstrip(";"):
        return False, "not_single_read_only_sql", []
    if re.search(r"\b(insert|update|delete|drop|alter|create|attach|detach|pragma|vacuum)\b", q, re.I):
        return False, "unsafe_sql", []
    con = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)
    ticks = 0

    def progress() -> int:
        nonlocal ticks
        ticks += 1000
        return int(ticks > 2_000_000)

    con.set_progress_handler(progress, 1000)
    try:
        rows = con.execute(q).fetchmany(10_001)
        if len(rows) > 10_000:
            return False, "result_over_10000_rows", []
        return True, "", [tuple(norm_cell(x) for x in row) for row in rows]
    except Exception as e:
        return False, f"{type(e).__name__}:{e}", []
    finally:
        con.close()


def equivalent(gold: str, gold_rows: list[tuple[Any, ...]], pred_rows: list[tuple[Any, ...]]) -> bool:
    if re.search(r"\border\s+by\b", gold, re.I):
        return gold_rows == pred_rows
    return collections.Counter(gold_rows) == collections.Counter(pred_rows)


def main() -> None:
    a = args()
    source = load_jsonl(a.questions)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    prior = {}
    if a.output.exists():
        for row in load_jsonl(a.output):
            prior[row["id"]] = row
    results = dict(prior)
    for n, item in enumerate(source, 1):
        if item["id"] in results:
            continue
        started = time.perf_counter()
        error = ""
        response: dict[str, Any] = {}
        for attempt in range(a.retries + 1):
            try:
                response = request_chat(a, item["messages"])
                break
            except Exception as e:
                error = f"api:{type(e).__name__}:{e}"
                if attempt < a.retries:
                    time.sleep(2**attempt)
        content = ""
        finish_reason = ""
        usage = {}
        if response:
            choice = response["choices"][0]
            content = choice.get("message", {}).get("content", "") or ""
            finish_reason = choice.get("finish_reason", "") or ""
            usage = response.get("usage", {})
        pred = clean_sql(content)
        meta = item["metadata"]
        db = Path(meta["database"])
        gold_ok, gold_error, gold_rows = run_sql(db, meta["gold_sql"])
        pred_ok, pred_error, pred_rows = run_sql(db, pred) if pred else (False, "empty", [])
        is_correct = bool(gold_ok and pred_ok and equivalent(meta["gold_sql"], gold_rows, pred_rows))
        result = {
            "id": item["id"],
            "db_id": meta["db_id"],
            "difficulty": meta["difficulty"],
            "question": meta["question"],
            "gold_sql": meta["gold_sql"],
            "prediction": pred,
            "api_error": error,
            "gold_error": gold_error,
            "prediction_error": pred_error,
            "sql_executable": pred_ok,
            "execution_correct": is_correct,
            "finish_reason": finish_reason,
            "usage": usage,
            "latency_s": round(time.perf_counter() - started, 3),
        }
        results[item["id"]] = result
        with a.output.open("w", encoding="utf-8") as f:
            for src in source:
                if src["id"] in results:
                    f.write(json.dumps(results[src["id"]], ensure_ascii=False) + "\n")
        print(f"[{n:03d}/{len(source):03d}] {item['id']} exec={pred_ok} correct={is_correct}")

    rows = [results[x["id"]] for x in source if x["id"] in results]
    valid = [r for r in rows if not r["api_error"]]
    summary = {
        "rows": len(rows),
        "api_success": len(valid),
        "sql_executable_rate": round(sum(r["sql_executable"] for r in valid) / max(1, len(valid)), 4),
        "execution_accuracy": round(sum(r["execution_correct"] for r in valid) / max(1, len(valid)), 4),
        "by_difficulty": {},
    }
    for diff in ("easy", "medium", "hard", "extra"):
        group = [r for r in valid if r["difficulty"] == diff]
        summary["by_difficulty"][diff] = {
            "n": len(group),
            "executable_rate": round(sum(r["sql_executable"] for r in group) / max(1, len(group)), 4),
            "execution_accuracy": round(sum(r["execution_correct"] for r in group) / max(1, len(group)), 4),
        }
    summary_path = a.output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("FINAL_SUMMARY")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
