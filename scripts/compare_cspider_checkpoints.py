#!/usr/bin/env python3
"""Compare CSpider Base and multiple LoRA checkpoints on the same fixed gate."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any


LEVELS = ("easy", "medium", "hard", "extra")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument(
        "--candidate",
        action="append",
        required=True,
        metavar="LABEL=JSONL",
        help="Repeat for each checkpoint, e.g. checkpoint-100=/path/result.jsonl",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except Exception as exc:
                raise RuntimeError(f"{path}:{line_no}: {exc}") from exc
            if not isinstance(value, dict):
                raise RuntimeError(f"{path}:{line_no}: row is not an object")
            rows.append(value)
    return rows


def index_rows(path: Path) -> dict[str, dict[str, Any]]:
    rows = read_jsonl(path)
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        qid = str(row.get("id", "")).strip()
        if not qid:
            raise RuntimeError(f"{path}: row without id")
        if qid in result:
            raise RuntimeError(f"{path}: duplicate id {qid}")
        if "execution_correct" not in row or "sql_executable" not in row:
            raise RuntimeError(
                f"{path}: {qid} lacks execution_correct/sql_executable"
            )
        result[qid] = row
    return result


def exact_mcnemar_p(fixed: int, regressed: int) -> float:
    discordant = fixed + regressed
    if discordant == 0:
        return 1.0
    tail = min(fixed, regressed)
    probability = sum(
        math.comb(discordant, i) for i in range(tail + 1)
    ) / (2**discordant)
    return min(1.0, 2.0 * probability)


def metrics(rows: dict[str, dict[str, Any]]) -> dict[str, Any]:
    values = list(rows.values())
    total = len(values)
    api_success = sum(not str(row.get("api_error", "")).strip() for row in values)
    executable = sum(bool(row["sql_executable"]) for row in values)
    correct = sum(bool(row["execution_correct"]) for row in values)
    by_difficulty: dict[str, dict[str, Any]] = {}
    for level in LEVELS:
        group = [row for row in values if str(row.get("difficulty", "")).lower() == level]
        by_difficulty[level] = {
            "n": len(group),
            "api_success": sum(not str(row.get("api_error", "")).strip() for row in group),
            "executable": sum(bool(row["sql_executable"]) for row in group),
            "correct": sum(bool(row["execution_correct"]) for row in group),
        }
    return {
        "rows": total,
        "api_success": api_success,
        "executable": executable,
        "invalid_sql": total - executable,
        "sql_executable_rate": executable / total if total else 0.0,
        "correct": correct,
        "execution_accuracy": correct / total if total else 0.0,
        "by_difficulty": by_difficulty,
    }


def failure_reasons(rows: dict[str, dict[str, Any]]) -> list[tuple[str, int]]:
    reasons: Counter[str] = Counter()
    for row in rows.values():
        if bool(row["execution_correct"]):
            continue
        if str(row.get("api_error", "")).strip():
            reasons["API_ERROR"] += 1
        elif not bool(row["sql_executable"]):
            error = str(row.get("prediction_error", "")).strip()
            if "no such column" in error:
                reasons["INVALID_NO_SUCH_COLUMN"] += 1
            elif "syntax error" in error or "incomplete input" in error:
                reasons["INVALID_SYNTAX"] += 1
            elif "interrupted" in error:
                reasons["INVALID_INTERRUPTED"] += 1
            elif "bindings" in error:
                reasons["INVALID_PARAMETER"] += 1
            else:
                reasons["INVALID_OTHER"] += 1
        else:
            reasons["EXECUTABLE_WRONG"] += 1
    return reasons.most_common()


def main() -> None:
    args = parse_args()
    candidates: dict[str, Path] = {}
    for specification in args.candidate:
        if "=" not in specification:
            raise RuntimeError(f"Invalid --candidate: {specification}")
        label, raw_path = specification.split("=", 1)
        label = label.strip()
        if not label or label in candidates:
            raise RuntimeError(f"Invalid/duplicate candidate label: {label}")
        candidates[label] = Path(raw_path)

    base_rows = index_rows(args.base)
    base_ids = set(base_rows)
    base_metrics = metrics(base_rows)
    if len(base_rows) != 200:
        raise RuntimeError(f"Base must contain 200 rows, got {len(base_rows)}")

    output: dict[str, Any] = {
        "base_path": str(args.base),
        "base": base_metrics,
        "selection_criteria": {
            "rows": 200,
            "api_success": 200,
            "primary_metric": "hard_execution_accuracy",
            "hard_correct_gain_min": 5,
            "overall_correct_min": base_metrics["correct"],
        },
        "candidates": {},
    }

    csv_rows: list[dict[str, Any]] = []
    selected: list[tuple[str, dict[str, Any]]] = []
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for label, path in candidates.items():
        rows = index_rows(path)
        if set(rows) != base_ids:
            missing = sorted(base_ids - set(rows))[:10]
            extra = sorted(set(rows) - base_ids)[:10]
            raise RuntimeError(
                f"{label}: IDs differ from Base; missing={missing}, extra={extra}"
            )
        for qid in base_ids:
            if str(rows[qid].get("difficulty", "")).lower() != str(
                base_rows[qid].get("difficulty", "")
            ).lower():
                raise RuntimeError(f"{label}: difficulty mismatch for {qid}")

        current = metrics(rows)
        fixed_ids = sorted(
            qid
            for qid in base_ids
            if not bool(base_rows[qid]["execution_correct"])
            and bool(rows[qid]["execution_correct"])
        )
        regressed_ids = sorted(
            qid
            for qid in base_ids
            if bool(base_rows[qid]["execution_correct"])
            and not bool(rows[qid]["execution_correct"])
        )
        fixed_by_level = Counter(
            str(rows[qid].get("difficulty", "")).lower() for qid in fixed_ids
        )
        regressed_by_level = Counter(
            str(rows[qid].get("difficulty", "")).lower() for qid in regressed_ids
        )
        net_fixed = len(fixed_ids) - len(regressed_ids)
        hard_correct_gain = (
            current["by_difficulty"]["hard"]["correct"]
            - base_metrics["by_difficulty"]["hard"]["correct"]
        )
        selection_checks = {
            "rows_200": current["rows"] == 200,
            "api_success_200": current["api_success"] == 200,
            "hard_correct_gain_at_least_5": hard_correct_gain >= 5,
            "overall_correct_not_below_base": current["correct"]
            >= base_metrics["correct"],
        }
        is_selected = all(selection_checks.values())
        detail = {
            "path": str(path),
            "metrics": current,
            "delta_correct": current["correct"] - base_metrics["correct"],
            "delta_accuracy_pp": round(
                100 * (current["execution_accuracy"] - base_metrics["execution_accuracy"]),
                3,
            ),
            "hard_correct_gain": hard_correct_gain,
            "hard_accuracy_gain_pp": round(100 * hard_correct_gain / 60, 3),
            "fixed": len(fixed_ids),
            "regressed": len(regressed_ids),
            "net_fixed": net_fixed,
            "fixed_by_difficulty": dict(fixed_by_level),
            "regressed_by_difficulty": dict(regressed_by_level),
            "mcnemar_exact_p": exact_mcnemar_p(len(fixed_ids), len(regressed_ids)),
            "failure_reasons": failure_reasons(rows),
            "selection_checks": selection_checks,
            "outcome": "SUCCESS" if is_selected else "NOT_SELECTED",
        }
        output["candidates"][label] = detail
        if is_selected:
            selected.append((label, detail))

        changed_path = args.output_dir / f"{label}-changed-vs-base.jsonl"
        with changed_path.open("w", encoding="utf-8", newline="\n") as handle:
            for qid in fixed_ids + regressed_ids:
                handle.write(
                    json.dumps(
                        {
                            "id": qid,
                            "difficulty": rows[qid].get("difficulty"),
                            "change": "fixed" if qid in fixed_ids else "regressed",
                            "question": rows[qid].get("question"),
                            "gold_sql": rows[qid].get("gold_sql"),
                            "base_prediction": base_rows[qid].get("prediction"),
                            "candidate_prediction": rows[qid].get("prediction"),
                            "candidate_executable": rows[qid].get("sql_executable"),
                            "candidate_error": rows[qid].get("prediction_error"),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

        csv_rows.append(
            {
                "model": label,
                "correct": current["correct"],
                "accuracy_pct": round(100 * current["execution_accuracy"], 2),
                "delta_correct": detail["delta_correct"],
                "executable_pct": round(100 * current["sql_executable_rate"], 2),
                "invalid_sql": current["invalid_sql"],
                "easy_correct": current["by_difficulty"]["easy"]["correct"],
                "medium_correct": current["by_difficulty"]["medium"]["correct"],
                "hard_correct": current["by_difficulty"]["hard"]["correct"],
                "extra_correct": current["by_difficulty"]["extra"]["correct"],
                "fixed": len(fixed_ids),
                "regressed": len(regressed_ids),
                "net_fixed": net_fixed,
                "mcnemar_p": round(detail["mcnemar_exact_p"], 6),
                "outcome": detail["outcome"],
            }
        )

    if selected:
        selected.sort(
            key=lambda item: (
                item[1]["metrics"]["by_difficulty"]["hard"]["correct"],
                item[1]["metrics"]["correct"],
                item[1]["metrics"]["sql_executable_rate"],
            ),
            reverse=True,
        )
        output["winner"] = selected[0][0]
        output["overall_result"] = "SUCCESS"
    else:
        output["winner"] = None
        output["overall_result"] = "NO_SELECTION"

    json_path = args.output_dir / "unified-comparison.json"
    csv_path = args.output_dir / "unified-comparison.csv"
    json_path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)

    print("BASE")
    print(
        f"correct={base_metrics['correct']}/200 "
        f"exec={base_metrics['sql_executable_rate']:.2%} "
        f"invalid={base_metrics['invalid_sql']}"
    )
    for row in csv_rows:
        print("\n" + row["model"])
        print(
            f"correct={row['correct']}/200 ({row['accuracy_pct']:.2f}%), "
            f"delta={row['delta_correct']:+d}, exec={row['executable_pct']:.2f}%, "
            f"invalid={row['invalid_sql']}"
        )
        print(
            f"easy={row['easy_correct']}, medium={row['medium_correct']}, "
            f"hard={row['hard_correct']}, extra={row['extra_correct']}"
        )
        print(
            f"fixed={row['fixed']}, regressed={row['regressed']}, "
            f"net={row['net_fixed']:+d}, McNemar p={row['mcnemar_p']}, "
            f"outcome={row['outcome']}"
        )
        checks = output["candidates"][row["model"]]["selection_checks"]
        for name, satisfied in checks.items():
            print(f"  [{'YES' if satisfied else 'NO'}] {name}")
    print("\nOVERALL_RESULT:", output["overall_result"])
    print("WINNER:", output["winner"])
    print("JSON:", json_path)
    print("CSV:", csv_path)


if __name__ == "__main__":
    main()
