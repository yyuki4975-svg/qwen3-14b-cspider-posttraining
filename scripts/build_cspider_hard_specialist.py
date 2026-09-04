#!/usr/bin/env python3
"""Build the pre-registered Qwen3-14B CSpider Hard-specialist SFT set.

This script deliberately imports the reviewed generic v3 builder from the same
directory and replaces only the capability labelling and deterministic selector.
Upload this file together with build_cspider_pilot_v3.py.
"""

from __future__ import annotations

import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any

import build_cspider_pilot_v3 as core


BUILD_VERSION = "cspider-hard-specialist1000-v1-20260819"

DIFFICULTY_QUOTAS = {
    "hard": 700,
    "extra": 100,
    "medium": 150,
    "easy": 50,
}

# Every minimum below must be met by Hard rows, not by protection rows.
# Labels overlap, so the sum is intentionally greater than 700.
HARD_CAPABILITY_MINIMUMS = {
    "schema_linking_join": 250,
    "nested_query": 150,
    "set_operation": 120,
    "argmax_argmin": 150,
    "aggregation_group_having": 150,
    "literal_copy": 100,
    "projection_distinct_return": 150,
}

# The 300 non-Hard rows are explicit capability-retention anchors.
ANCHOR_CAPABILITY_MINIMUMS = {
    "schema_linking_join": 80,
    "nested_query": 35,
    "set_operation": 20,
    "argmax_argmin": 35,
    "aggregation_group_having": 80,
    "literal_copy": 35,
    "projection_distinct_return": 180,
}

MAX_PER_DATABASE = 40
MAX_PER_EXACT_STRUCTURE = 10
CAPABILITIES = tuple(HARD_CAPABILITY_MINIMUMS)


def specialist_capabilities(
    sql: str, prompt: str, literal_copy: bool
) -> set[str]:
    caps = set(core.sql_capabilities_original(sql, prompt, literal_copy))
    upper = sql.upper()
    select_count = len(re.findall(r"\bSELECT\b", upper))
    if select_count >= 2:
        caps.add("nested_query")
    if re.search(r"\b(?:UNION|INTERSECT|EXCEPT)\b", upper):
        caps.add("set_operation")
    # The old combined label is intentionally not used as a quota.
    caps.discard("nested_set_operation")
    return caps


def _cap_counts(
    selected: list[dict[str, Any]], *, hard: bool
) -> Counter[str]:
    result: Counter[str] = Counter()
    for item in selected:
        if (item["difficulty"] == "hard") != hard:
            continue
        result.update(item["capabilities"])
    return result


def _deficits(
    counts: Counter[str], targets: dict[str, int]
) -> dict[str, int]:
    return {
        capability: target - counts[capability]
        for capability, target in targets.items()
        if counts[capability] < target
    }


def _validate_selection(selected: list[dict[str, Any]]) -> None:
    difficulty = Counter(item["difficulty"] for item in selected)
    hard_caps = _cap_counts(selected, hard=True)
    anchor_caps = _cap_counts(selected, hard=False)
    db_counts = Counter(item["db_id"] for item in selected)
    structure_counts = Counter(item["structure_signature"] for item in selected)
    failures: dict[str, Any] = {}
    if difficulty != Counter(DIFFICULTY_QUOTAS):
        failures["difficulty"] = {
            "actual": dict(difficulty),
            "required": DIFFICULTY_QUOTAS,
        }
    hard_missing = _deficits(hard_caps, HARD_CAPABILITY_MINIMUMS)
    if hard_missing:
        failures["hard_capabilities"] = hard_missing
    anchor_missing = _deficits(anchor_caps, ANCHOR_CAPABILITY_MINIMUMS)
    if anchor_missing:
        failures["anchor_capabilities"] = anchor_missing
    if db_counts and max(db_counts.values()) > MAX_PER_DATABASE:
        failures["database_concentration"] = db_counts.most_common(10)
    if structure_counts and max(structure_counts.values()) > MAX_PER_EXACT_STRUCTURE:
        failures["structure_concentration"] = structure_counts.most_common(10)
    if failures:
        raise RuntimeError(
            "Specialist selection failed closed:\n"
            + json.dumps(failures, ensure_ascii=False, indent=2)
        )


def choose_specialist_selection(
    candidates: list[dict[str, Any]], seed: int, restarts: int
) -> list[dict[str, Any]]:
    available_difficulty = Counter(item["difficulty"] for item in candidates)
    hard_available = Counter(
        cap
        for item in candidates
        if item["difficulty"] == "hard"
        for cap in item["capabilities"]
    )
    anchor_available = Counter(
        cap
        for item in candidates
        if item["difficulty"] != "hard"
        for cap in item["capabilities"]
    )
    preflight: dict[str, Any] = {}
    for level, target in DIFFICULTY_QUOTAS.items():
        if available_difficulty[level] < target:
            preflight[f"difficulty:{level}"] = {
                "available": available_difficulty[level],
                "required": target,
            }
    for cap, target in HARD_CAPABILITY_MINIMUMS.items():
        if hard_available[cap] < target:
            preflight[f"hard:{cap}"] = {
                "available": hard_available[cap],
                "required": target,
            }
    for cap, target in ANCHOR_CAPABILITY_MINIMUMS.items():
        if anchor_available[cap] < target:
            preflight[f"anchor:{cap}"] = {
                "available": anchor_available[cap],
                "required": target,
            }
    if preflight:
        raise RuntimeError(
            "Hard-specialist preflight is infeasible. Use the full original "
            "CSpider training pool; do not duplicate or relabel rows:\n"
            + json.dumps(preflight, ensure_ascii=False, indent=2)
        )

    errors: list[str] = []
    for restart in range(restarts):
        rng = random.Random(seed + restart * 104729)
        selected: list[dict[str, Any]] = []
        selected_ids: set[str] = set()
        level_counts: Counter[str] = Counter()
        cap_counts: Counter[str] = Counter()
        db_counts: Counter[str] = Counter()
        family_counts: Counter[str] = Counter()
        structure_counts: Counter[str] = Counter()

        def eligible(item: dict[str, Any]) -> bool:
            return (
                item["candidate_key"] not in selected_ids
                and level_counts[item["difficulty"]]
                < DIFFICULTY_QUOTAS[item["difficulty"]]
                and db_counts[item["db_id"]] < MAX_PER_DATABASE
                and structure_counts[item["structure_signature"]]
                < MAX_PER_EXACT_STRUCTURE
            )

        def add(item: dict[str, Any]) -> None:
            core.add_selected(
                item,
                selected,
                selected_ids,
                level_counts,
                cap_counts,
                db_counts,
                family_counts,
                structure_counts,
            )

        def fill_capability_phase(*, hard: bool, targets: dict[str, int]) -> bool:
            while True:
                counts = _cap_counts(selected, hard=hard)
                deficits = _deficits(counts, targets)
                if not deficits:
                    return True
                choices: list[tuple[float, float, dict[str, Any]]] = []
                for item in candidates:
                    if not eligible(item):
                        continue
                    if (item["difficulty"] == "hard") != hard:
                        continue
                    gains = set(item["capabilities"]) & set(deficits)
                    if not gains:
                        continue
                    score = 0.0
                    for cap in gains:
                        score += 20.0 + 20.0 * deficits[cap] / targets[cap]
                    score += 8.0 / (1 + db_counts[item["db_id"]])
                    score += 5.0 / (1 + family_counts[item["family"]])
                    score += 5.0 / (1 + structure_counts[item["structure_signature"]])
                    choices.append((score, rng.random(), item))
                if not choices:
                    errors.append(
                        f"restart={restart}, hard={hard}, unresolved={deficits}"
                    )
                    return False
                add(max(choices, key=lambda value: (value[0], value[1]))[2])

        if not fill_capability_phase(hard=True, targets=HARD_CAPABILITY_MINIMUMS):
            continue
        if not fill_capability_phase(hard=False, targets=ANCHOR_CAPABILITY_MINIMUMS):
            continue

        failed = False
        for level in DIFFICULTY_QUOTAS:
            while level_counts[level] < DIFFICULTY_QUOTAS[level]:
                choices: list[tuple[float, float, dict[str, Any]]] = []
                for item in candidates:
                    if not eligible(item) or item["difficulty"] != level:
                        continue
                    score = 0.0
                    score += 12.0 / (1 + db_counts[item["db_id"]])
                    score += 8.0 / (1 + family_counts[item["family"]])
                    score += 8.0 / (1 + structure_counts[item["structure_signature"]])
                    score += 0.5 * len(item["capabilities"])
                    choices.append((score, rng.random(), item))
                if not choices:
                    errors.append(f"restart={restart}: cannot fill {level}")
                    failed = True
                    break
                add(max(choices, key=lambda value: (value[0], value[1]))[2])
            if failed:
                break
        if failed:
            continue

        try:
            _validate_selection(selected)
        except RuntimeError as exc:
            errors.append(f"restart={restart}: {exc}")
            continue
        rng.shuffle(selected)
        return selected

    raise RuntimeError(
        "No feasible Hard-specialist selection after deterministic restarts. "
        + " | ".join(errors[-5:])
    )


def enrich_manifest(output_dir: Path) -> None:
    audit_path = output_dir / "selection_audit.jsonl"
    manifest_path = output_dir / "manifest.json"
    rows = core.read_jsonl(audit_path)
    _validate_selection(rows)
    hard_counts = _cap_counts(rows, hard=True)
    anchor_counts = _cap_counts(rows, hard=False)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update(
        {
            "specialist_build_version": BUILD_VERSION,
            "specialist_target": "CSpider Hard execution accuracy",
            "difficulty_quotas": DIFFICULTY_QUOTAS,
            "hard_capability_minimums": HARD_CAPABILITY_MINIMUMS,
            "selected_hard_capabilities": dict(hard_counts),
            "anchor_capability_minimums": ANCHOR_CAPABILITY_MINIMUMS,
            "selected_anchor_capabilities": dict(anchor_counts),
            "max_per_database": MAX_PER_DATABASE,
            "max_per_exact_structure": MAX_PER_EXACT_STRUCTURE,
            "specialist_quality_gate": "PASS",
        }
    )
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    checksum_path = output_dir / "sha256sums.txt"
    checked = [
        output_dir / "pilot1000.jsonl",
        audit_path,
        manifest_path,
    ]
    checksum_path.write_text(
        "\n".join(f"{core.sha256(path)}  {path.name}" for path in checked) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    # Preserve the reviewed implementation before replacing the public hook.
    core.sql_capabilities_original = core.sql_capabilities
    core.BUILD_VERSION = BUILD_VERSION
    core.DIFFICULTY_QUOTAS = DIFFICULTY_QUOTAS
    core.DIFFICULTIES = tuple(DIFFICULTY_QUOTAS)
    # The specialist selector enforces split-specific minima itself.
    core.CAPABILITY_MINIMUMS = {}
    core.CAPABILITIES = CAPABILITIES
    core.sql_capabilities = specialist_capabilities
    core.choose_selection = choose_specialist_selection

    args = core.parse_args()
    core.main()
    enrich_manifest(args.output_dir.resolve())
    print("HARD_SPECIALIST_FINAL_GATE: PASS")
    print("ENRICHED_MANIFEST:", args.output_dir.resolve() / "manifest.json")


if __name__ == "__main__":
    main()
