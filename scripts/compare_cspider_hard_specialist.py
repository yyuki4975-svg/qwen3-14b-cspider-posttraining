#!/usr/bin/env python3
"""Pre-registered comparison for the CSpider Hard-specialist LoRA experiment."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import compare_cspider_checkpoints as common


def main() -> None:
    args = common.parse_args()
    candidates: dict[str, Path] = {}
    for specification in args.candidate:
        if "=" not in specification:
            raise RuntimeError(f"Invalid --candidate: {specification}")
        label, raw_path = specification.split("=", 1)
        label = label.strip()
        if not label or label in candidates:
            raise RuntimeError(f"Invalid/duplicate candidate label: {label}")
        candidates[label] = Path(raw_path)

    base_rows = common.index_rows(args.base)
    if len(base_rows) != 200:
        raise RuntimeError(f"Base must contain 200 rows, got {len(base_rows)}")
    base_ids = set(base_rows)
    base_metrics = common.metrics(base_rows)
    base_hard = base_metrics["by_difficulty"]["hard"]
    base_nonhard_correct = sum(
        base_metrics["by_difficulty"][level]["correct"]
        for level in ("easy", "medium", "extra")
    )

    gate = {
        "rows": 200,
        "api_success": 200,
        "hard_correct_min": base_hard["correct"] + 5,
        "hard_net_fixed_min": 5,
        "hard_executable_min": base_hard["executable"],
        "overall_correct_min": base_metrics["correct"],
        "overall_sql_executable_rate_min": 0.95,
        "invalid_sql_max": base_metrics["invalid_sql"] + 2,
        "nonhard_correct_min": base_nonhard_correct - 3,
    }
    output: dict[str, Any] = {
        "experiment": "Qwen3-14B CSpider Hard-specialist LoRA",
        "base_path": str(args.base),
        "base": base_metrics,
        "pre_registered_gate": gate,
        "candidates": {},
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_rows: list[dict[str, Any]] = []
    passing: list[tuple[str, dict[str, Any]]] = []

    for label, path in candidates.items():
        rows = common.index_rows(path)
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

        current = common.metrics(rows)
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
        hard_fixed = [
            qid
            for qid in fixed_ids
            if str(rows[qid].get("difficulty", "")).lower() == "hard"
        ]
        hard_regressed = [
            qid
            for qid in regressed_ids
            if str(rows[qid].get("difficulty", "")).lower() == "hard"
        ]
        hard_net = len(hard_fixed) - len(hard_regressed)
        nonhard_correct = sum(
            current["by_difficulty"][level]["correct"]
            for level in ("easy", "medium", "extra")
        )
        hard = current["by_difficulty"]["hard"]
        checks = {
            "rows_200": current["rows"] == gate["rows"],
            "api_success_200": current["api_success"] == gate["api_success"],
            "hard_correct_gain_at_least_5": hard["correct"]
            >= gate["hard_correct_min"],
            "hard_net_fixed_at_least_5": hard_net
            >= gate["hard_net_fixed_min"],
            "hard_executable_not_below_base": hard["executable"]
            >= gate["hard_executable_min"],
            "overall_correct_not_below_base": current["correct"]
            >= gate["overall_correct_min"],
            "overall_executable_at_least_95pct": current["sql_executable_rate"]
            >= gate["overall_sql_executable_rate_min"],
            "invalid_sql_at_most_base_plus_2": current["invalid_sql"]
            <= gate["invalid_sql_max"],
            "nonhard_regression_at_most_3": nonhard_correct
            >= gate["nonhard_correct_min"],
        }
        passed = all(checks.values())
        detail = {
            "path": str(path),
            "metrics": current,
            "hard_correct_delta": hard["correct"] - base_hard["correct"],
            "hard_fixed": len(hard_fixed),
            "hard_regressed": len(hard_regressed),
            "hard_net_fixed": hard_net,
            "hard_mcnemar_exact_p": common.exact_mcnemar_p(
                len(hard_fixed), len(hard_regressed)
            ),
            "overall_correct_delta": current["correct"] - base_metrics["correct"],
            "nonhard_correct": nonhard_correct,
            "nonhard_correct_delta": nonhard_correct - base_nonhard_correct,
            "all_fixed": len(fixed_ids),
            "all_regressed": len(regressed_ids),
            "failure_reasons": common.failure_reasons(rows),
            "gate_checks": checks,
            "final_gate": "PASS" if passed else "FAIL",
        }
        output["candidates"][label] = detail
        if passed:
            passing.append((label, detail))

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
                "overall_correct": current["correct"],
                "overall_delta": detail["overall_correct_delta"],
                "hard_correct": hard["correct"],
                "hard_delta": detail["hard_correct_delta"],
                "hard_fixed": len(hard_fixed),
                "hard_regressed": len(hard_regressed),
                "hard_net": hard_net,
                "hard_mcnemar_p": round(detail["hard_mcnemar_exact_p"], 6),
                "nonhard_delta": detail["nonhard_correct_delta"],
                "executable_pct": round(100 * current["sql_executable_rate"], 2),
                "invalid_sql": current["invalid_sql"],
                "gate": detail["final_gate"],
            }
        )

    if passing:
        passing.sort(
            key=lambda item: (
                item[1]["metrics"]["by_difficulty"]["hard"]["correct"],
                item[1]["hard_net_fixed"],
                item[1]["metrics"]["correct"],
                item[1]["metrics"]["sql_executable_rate"],
                -item[1]["all_regressed"],
            ),
            reverse=True,
        )
        output["overall_result"] = "PASS"
        output["winner"] = passing[0][0]
    else:
        output["overall_result"] = "FAIL"
        output["winner"] = None

    json_path = args.output_dir / "hard-specialist-comparison.json"
    csv_path = args.output_dir / "hard-specialist-comparison.csv"
    json_path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)

    print(
        f"BASE overall={base_metrics['correct']}/200, "
        f"hard={base_hard['correct']}/60, hard_exec={base_hard['executable']}/60, "
        f"nonhard={base_nonhard_correct}/140"
    )
    for row in csv_rows:
        print("\n" + row["model"])
        print(
            f"overall={row['overall_correct']}/200 ({row['overall_delta']:+d}), "
            f"hard={row['hard_correct']}/60 ({row['hard_delta']:+d}), "
            f"hard fixed/regressed/net={row['hard_fixed']}/"
            f"{row['hard_regressed']}/{row['hard_net']:+d}"
        )
        print(
            f"nonhard delta={row['nonhard_delta']:+d}, "
            f"exec={row['executable_pct']:.2f}%, invalid={row['invalid_sql']}, "
            f"hard McNemar p={row['hard_mcnemar_p']}, gate={row['gate']}"
        )
        for name, passed in output["candidates"][row["model"]]["gate_checks"].items():
            print(f"  [{'PASS' if passed else 'FAIL'}] {name}")
    print("\nOVERALL_RESULT:", output["overall_result"])
    print("WINNER:", output["winner"])
    print("JSON:", json_path)
    print("CSV:", csv_path)


if __name__ == "__main__":
    main()
