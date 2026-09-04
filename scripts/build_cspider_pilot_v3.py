#!/usr/bin/env python3
"""Build a deterministic, executable, capability-balanced CSpider Pilot v3 set.

This builder is intentionally fail-closed: it writes the final training files only
after difficulty quotas, capability coverage, gate isolation, executability, token
length, and uniqueness checks all pass.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import shutil
import sqlite3
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from transformers import AutoTokenizer


BUILD_VERSION = "cspider-pilot1000-v3.2-20260819"

DIFFICULTY_QUOTAS = {
    "easy": 100,
    "medium": 434,
    "hard": 326,
    "extra": 140,
}

# These are overlapping multi-label targets; their sum is intentionally > 1000.
CAPABILITY_MINIMUMS = {
    "schema_linking_join": 250,
    "projection_distinct_return": 200,
    "aggregation_group_having": 200,
    "nested_set_operation": 160,
    "argmax_argmin": 100,
    "literal_copy": 120,
}

CAPABILITIES = tuple(CAPABILITY_MINIMUMS)
DIFFICULTIES = tuple(DIFFICULTY_QUOTAS)
COARSE_FAMILIES = (
    "basic",
    "order_limit",
    "aggregation",
    "join",
    "group_having",
    "nested",
    "set_operation",
)

SQL_STRING_RE = re.compile(r"'((?:''|[^'])*)'", re.DOTALL)
CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-train", type=Path, required=True)
    parser.add_argument("--gate-dev", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--db-search-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260819)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--selection-restarts", type=int, default=64)
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


def metadata(row: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key in ("metadata", "meta"):
        if isinstance(row.get(key), dict):
            result.update(row[key])
    return result


def get_value(row: dict[str, Any], *names: str) -> Any:
    meta = metadata(row)
    for name in names:
        if row.get(name) not in (None, ""):
            return row[name]
        if meta.get(name) not in (None, ""):
            return meta[name]
    return None


def get_id(row: dict[str, Any]) -> str:
    value = get_value(row, "id", "question_id", "qid")
    if value is not None:
        return str(value)
    raw = json.dumps(row, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


def get_level(row: dict[str, Any]) -> str:
    value = get_value(row, "difficulty", "level", "hardness")
    if value is None:
        raise ValueError("missing_difficulty")
    return str(value).strip().lower()


def get_db_id(row: dict[str, Any]) -> str:
    value = get_value(row, "db_id", "database_id", "database", "db_name")
    if value is None:
        raise ValueError("missing_db_id")
    return str(value).strip()


def get_messages(row: dict[str, Any]) -> list[dict[str, Any]]:
    messages = row.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError("missing_messages")
    if not all(isinstance(message, dict) for message in messages):
        raise ValueError("invalid_messages")
    return messages


def get_user_text(row: dict[str, Any]) -> str:
    return "\n".join(
        str(message.get("content", ""))
        for message in get_messages(row)
        if message.get("role") == "user"
    ).strip()


def get_assistant_text(row: dict[str, Any]) -> str:
    answers = [
        str(message.get("content", ""))
        for message in get_messages(row)
        if message.get("role") == "assistant"
    ]
    return answers[-1].strip() if answers else ""


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def normalize_sql(sql: str) -> str:
    return re.sub(r"\s+", " ", sql.strip().rstrip(";")).strip().lower()


def sql_structure_signature(sql: str) -> str:
    value = SQL_STRING_RE.sub("'<STR>'", sql)
    value = re.sub(r"\b\d+(?:\.\d+)?\b", "<NUM>", value)
    return normalize_sql(value)


def clean_sql_only(sql: str) -> str:
    value = sql.strip()
    lower = value.lower()
    if not value:
        raise ValueError("empty_sql")
    if "```" in value or re.search(r"(?im)^\s*(sql|answer|答案|解释)\s*[:：]", value):
        raise ValueError("markdown_or_explanation")
    if "<think>" in lower or "</think>" in lower:
        raise ValueError("thinking_tag")
    if "?" in value:
        raise ValueError("parameter_placeholder")
    if not re.match(r"(?is)^\s*(SELECT|WITH)\b", value):
        raise ValueError("not_select")
    if re.match(r"(?is)^\s*SELECT\s+(?:DISTINCT\s+)?FROM\b", value):
        raise ValueError("empty_projection")
    return value.rstrip(";").strip()


def sql_string_literals(sql: str) -> list[str]:
    return [match.group(1).replace("''", "'") for match in SQL_STRING_RE.finditer(sql)]


def validate_and_label_literals(sql: str, prompt: str) -> bool:
    copied = False
    for literal in sql_string_literals(sql):
        if literal and literal in prompt:
            copied = True
        if CJK_RE.search(literal) and literal not in prompt:
            raise ValueError("chinese_literal_not_copied_from_prompt")
    return copied


def first_projection(sql: str) -> str:
    match = re.search(r"(?is)\bSELECT\s+(?:DISTINCT\s+)?(.+?)\s+FROM\b", sql)
    return match.group(1).strip() if match else ""


def table_references(sql: str) -> set[str]:
    refs = {
        match.group(1).strip('`"[]').lower()
        for match in re.finditer(
            r"(?is)\b(?:FROM|JOIN)\s+([`\"\[]?[A-Za-z_][\w$]*[`\"\]]?)",
            sql,
        )
    }
    return refs


def has_implicit_multi_table(sql: str) -> bool:
    # Detect the common SQLite form: FROM table_a AS a, table_b AS b.
    for match in re.finditer(
        r"(?is)\bFROM\s+(.+?)(?=\bWHERE\b|\bGROUP\s+BY\b|\bORDER\s+BY\b|\bLIMIT\b|$)",
        sql,
    ):
        fragment = re.sub(r"\([^()]*\)", "", match.group(1))
        if "," in fragment:
            return True
    return False


def sql_capabilities(sql: str, prompt: str, literal_copy: bool) -> set[str]:
    upper = sql.upper()
    capabilities: set[str] = set()
    select_count = len(re.findall(r"\bSELECT\b", upper))
    refs = table_references(sql)

    if re.search(r"\bJOIN\b", upper) or has_implicit_multi_table(sql) or len(refs) >= 2:
        capabilities.add("schema_linking_join")

    projection = first_projection(sql)
    if (
        re.search(r"\bSELECT\s+DISTINCT\b", upper)
        or (projection and projection != "*")
        or "," in projection
    ):
        capabilities.add("projection_distinct_return")

    if re.search(r"\b(?:COUNT|AVG|SUM|MIN|MAX)\s*\(", upper) or re.search(
        r"\b(?:GROUP\s+BY|HAVING)\b", upper
    ):
        capabilities.add("aggregation_group_having")

    if select_count >= 2 or re.search(r"\b(?:UNION|INTERSECT|EXCEPT)\b", upper):
        capabilities.add("nested_set_operation")

    order_limit = re.search(
        r"(?is)\bORDER\s+BY\b.+?\b(?:ASC|DESC)\b.+?\bLIMIT\s+1\b",
        sql,
    )
    aggregate_extreme_subquery = re.search(
        r"(?is)(?:=|IN)\s*\(\s*SELECT\s+(?:MIN|MAX)\s*\(",
        sql,
    )
    if order_limit or aggregate_extreme_subquery:
        capabilities.add("argmax_argmin")

    if literal_copy:
        capabilities.add("literal_copy")

    return capabilities


def coarse_family(sql: str) -> str:
    upper = sql.upper()
    select_count = len(re.findall(r"\bSELECT\b", upper))
    if re.search(r"\b(?:UNION|INTERSECT|EXCEPT)\b", upper):
        return "set_operation"
    if select_count >= 2:
        return "nested"
    if re.search(r"\b(?:GROUP\s+BY|HAVING)\b", upper):
        return "group_having"
    if re.search(r"\bJOIN\b", upper) or has_implicit_multi_table(sql):
        return "join"
    if re.search(r"\b(?:COUNT|AVG|SUM|MIN|MAX)\s*\(", upper):
        return "aggregation"
    if re.search(r"\b(?:ORDER\s+BY|LIMIT)\b", upper):
        return "order_limit"
    return "basic"


def token_count(tokenizer: Any, messages: list[dict[str, Any]]) -> int:
    # tokenize=False avoids BatchEncoding/list ambiguity seen in the old builder.
    rendered = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    encoded = tokenizer(
        rendered,
        add_special_tokens=False,
        truncation=False,
        return_attention_mask=False,
        return_token_type_ids=False,
    )
    input_ids = encoded["input_ids"]
    if input_ids and isinstance(input_ids[0], list):
        if len(input_ids) != 1:
            raise ValueError("unexpected_token_batch")
        input_ids = input_ids[0]
    count = len(input_ids)
    if count <= 2:
        raise ValueError("invalid_token_count")
    return count


def build_db_index(root: Path) -> dict[str, list[Path]]:
    index: dict[str, list[Path]] = defaultdict(list)
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in (".sqlite", ".sqlite3", ".db"):
            continue
        index[path.stem.lower()].append(path)
        index[path.parent.name.lower()].append(path)
    return {
        key: sorted(set(paths), key=lambda item: (len(str(item)), str(item)))
        for key, paths in index.items()
    }


def resolve_db(
    row: dict[str, Any], db_id: str, db_root: Path, db_index: dict[str, list[Path]]
) -> list[Path]:
    candidates: list[Path] = []
    direct = get_value(row, "db_path", "database_path", "sqlite_path")
    if direct:
        path = Path(str(direct)).expanduser()
        if not path.is_absolute():
            path = db_root / path
        if path.is_file():
            candidates.append(path)
    candidates.extend(db_index.get(db_id.lower(), []))
    seen: set[Path] = set()
    result: list[Path] = []
    for path in candidates:
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            result.append(resolved)
    return result


def execute_sql(paths: Iterable[Path], sql: str) -> tuple[Path | None, str]:
    errors: list[str] = []
    for path in paths:
        connection: sqlite3.Connection | None = None
        try:
            connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=10)
            connection.execute("PRAGMA query_only=ON")
            progress = {"calls": 0}

            def stop_long_query() -> int:
                progress["calls"] += 1
                return 1 if progress["calls"] > 200_000 else 0

            connection.set_progress_handler(stop_long_query, 1000)
            cursor = connection.execute(sql)
            cursor.fetchall()
            return path, ""
        except Exception as exc:
            errors.append(f"{path}:{type(exc).__name__}:{exc}")
        finally:
            if connection is not None:
                connection.close()
    return None, " | ".join(errors[-3:])


def add_selected(
    item: dict[str, Any],
    selected: list[dict[str, Any]],
    selected_ids: set[str],
    level_counts: Counter[str],
    cap_counts: Counter[str],
    db_counts: Counter[str],
    family_counts: Counter[str],
    structure_counts: Counter[str],
) -> None:
    selected.append(item)
    selected_ids.add(item["candidate_key"])
    level_counts[item["difficulty"]] += 1
    cap_counts.update(item["capabilities"])
    db_counts[item["db_id"]] += 1
    family_counts[item["family"]] += 1
    structure_counts[item["structure_signature"]] += 1


def choose_selection(
    candidates: list[dict[str, Any]], seed: int, restarts: int
) -> list[dict[str, Any]]:
    availability = Counter(
        capability
        for item in candidates
        for capability in item["capabilities"]
    )

    errors: list[str] = []
    for restart in range(restarts):
        rng = random.Random(seed + restart * 104_729)
        selected: list[dict[str, Any]] = []
        selected_ids: set[str] = set()
        level_counts: Counter[str] = Counter()
        cap_counts: Counter[str] = Counter()
        db_counts: Counter[str] = Counter()
        family_counts: Counter[str] = Counter()
        structure_counts: Counter[str] = Counter()

        while True:
            deficits = {
                capability: target - cap_counts[capability]
                for capability, target in CAPABILITY_MINIMUMS.items()
                if cap_counts[capability] < target
            }
            if not deficits:
                break

            choices: list[tuple[float, float, dict[str, Any]]] = []
            for item in candidates:
                if item["candidate_key"] in selected_ids:
                    continue
                level = item["difficulty"]
                if level_counts[level] >= DIFFICULTY_QUOTAS[level]:
                    continue
                gains = set(item["capabilities"]) & set(deficits)
                if not gains:
                    continue
                score = 0.0
                for capability in gains:
                    target = CAPABILITY_MINIMUMS[capability]
                    scarcity = target / max(availability[capability], 1)
                    remaining_fraction = deficits[capability] / target
                    score += 10.0 + 20.0 * scarcity + 10.0 * remaining_fraction
                score += 2.0 / (1 + db_counts[item["db_id"]])
                score += 1.0 / (1 + family_counts[item["family"]])
                score += 0.5 / (1 + structure_counts[item["structure_signature"]])
                choices.append((score, rng.random(), item))

            if not choices:
                errors.append(
                    f"restart={restart}: capability deficits cannot be filled: {deficits}"
                )
                break
            chosen = max(choices, key=lambda value: (value[0], value[1]))[2]
            add_selected(
                chosen,
                selected,
                selected_ids,
                level_counts,
                cap_counts,
                db_counts,
                family_counts,
                structure_counts,
            )
        else:
            continue

        if any(
            cap_counts[capability] < target
            for capability, target in CAPABILITY_MINIMUMS.items()
        ):
            continue

        fill_failed = False
        for level in DIFFICULTIES:
            while level_counts[level] < DIFFICULTY_QUOTAS[level]:
                choices = []
                for item in candidates:
                    if item["candidate_key"] in selected_ids or item["difficulty"] != level:
                        continue
                    score = 0.0
                    score += 5.0 / (1 + db_counts[item["db_id"]])
                    score += 3.0 / (1 + family_counts[item["family"]])
                    score += 2.0 / (1 + structure_counts[item["structure_signature"]])
                    score += 0.1 * len(item["capabilities"])
                    choices.append((score, rng.random(), item))
                if not choices:
                    errors.append(f"restart={restart}: cannot fill difficulty {level}")
                    fill_failed = True
                    break
                chosen = max(choices, key=lambda value: (value[0], value[1]))[2]
                add_selected(
                    chosen,
                    selected,
                    selected_ids,
                    level_counts,
                    cap_counts,
                    db_counts,
                    family_counts,
                    structure_counts,
                )
            if fill_failed:
                break
        if fill_failed:
            continue

        if level_counts != Counter(DIFFICULTY_QUOTAS):
            errors.append(f"restart={restart}: wrong difficulty counts {dict(level_counts)}")
            continue
        if any(
            cap_counts[capability] < target
            for capability, target in CAPABILITY_MINIMUMS.items()
        ):
            errors.append(f"restart={restart}: wrong capability counts {dict(cap_counts)}")
            continue

        rng.shuffle(selected)
        return selected

    raise RuntimeError(
        "No feasible deterministic selection was found. Last errors: "
        + " | ".join(errors[-5:])
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    for path, label in (
        (args.source_train, "source train"),
        (args.gate_dev, "gate dev"),
        (args.model_dir, "model directory"),
        (args.db_search_root, "database search root"),
    ):
        if not path.exists():
            raise RuntimeError(f"Missing {label}: {path}")

    print("Build version:", BUILD_VERSION, flush=True)
    print("Indexing SQLite databases...", flush=True)
    db_index = build_db_index(args.db_search_root)
    print("Indexed database keys:", len(db_index), flush=True)

    source_rows = read_jsonl(args.source_train)
    gate_rows = read_jsonl(args.gate_dev)
    gate_ids = {get_id(row) for row in gate_rows}
    gate_prompts = {
        normalize_text(get_user_text(row))
        for row in gate_rows
        if get_user_text(row)
    }

    print("Loading local tokenizer...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_dir,
        local_files_only=True,
        trust_remote_code=False,
    )

    rejects: Counter[str] = Counter()
    seen_ids: set[str] = set()
    seen_prompts: set[str] = set()
    # Corrected: exact SQL is deduplicated only inside the same database. The old
    # global SQL dedup incorrectly removed valid cross-database instances.
    seen_db_sql: set[tuple[str, str]] = set()
    candidates: list[dict[str, Any]] = []

    for index, row in enumerate(source_rows, 1):
        try:
            qid = get_id(row)
            level = get_level(row)
            db_id = get_db_id(row)
            messages = get_messages(row)
            prompt = get_user_text(row)
            assistant = get_assistant_text(row)
            if level not in DIFFICULTY_QUOTAS:
                raise ValueError("unknown_difficulty")
            prompt_key = normalize_text(prompt)
            if qid in gate_ids or prompt_key in gate_prompts:
                raise ValueError("gate_overlap")
            if qid in seen_ids:
                raise ValueError("duplicate_id")
            if not prompt_key:
                raise ValueError("empty_prompt")
            if prompt_key in seen_prompts:
                raise ValueError("duplicate_prompt")

            sql = clean_sql_only(assistant)
            sql_key = normalize_sql(sql)
            db_sql_key = (db_id.lower(), sql_key)
            if db_sql_key in seen_db_sql:
                raise ValueError("duplicate_db_sql")

            declared_gold = get_value(row, "gold_sql", "query", "sql")
            if declared_gold and normalize_sql(str(declared_gold)) != sql_key:
                raise ValueError("assistant_gold_mismatch")

            copied_literal = validate_and_label_literals(sql, prompt)
            count = token_count(tokenizer, messages)
            if count > args.max_tokens:
                raise ValueError("over_max_tokens")

            db_paths = resolve_db(row, db_id, args.db_search_root, db_index)
            if not db_paths:
                raise ValueError("database_not_found")
            valid_db, sql_error = execute_sql(db_paths, sql)
            if valid_db is None:
                rejects["gold_not_executable"] += 1
                rejects[f"gold_error:{sql_error[:160]}"] += 1
                continue

            capabilities = sql_capabilities(sql, prompt, copied_literal)
            family = coarse_family(sql)
            structure = sql_structure_signature(sql)
            candidate = {
                "row": row,
                "id": qid,
                "candidate_key": hashlib.sha256(
                    f"{qid}\0{db_id}\0{sql_key}".encode("utf-8")
                ).hexdigest(),
                "difficulty": level,
                "family": family,
                "capabilities": sorted(capabilities),
                "db_id": db_id,
                "db_path": str(valid_db),
                "tokens": count,
                "sql_hash": hashlib.sha256(sql_key.encode("utf-8")).hexdigest(),
                "db_sql_hash": hashlib.sha256(
                    f"{db_id.lower()}\0{sql_key}".encode("utf-8")
                ).hexdigest(),
                "structure_signature": structure,
            }
            candidates.append(candidate)
            seen_ids.add(qid)
            seen_prompts.add(prompt_key)
            seen_db_sql.add(db_sql_key)
        except ValueError as exc:
            rejects[str(exc)] += 1
        except Exception as exc:
            rejects[f"unexpected:{type(exc).__name__}:{str(exc)[:120]}"] += 1

        if index % 250 == 0:
            print(f"[PROGRESS] scanned={index}, accepted={len(candidates)}", flush=True)

    available_by_difficulty = Counter(item["difficulty"] for item in candidates)
    available_by_capability = Counter(
        capability for item in candidates for capability in item["capabilities"]
    )
    available_capability_by_difficulty = {
        level: Counter(
            capability
            for item in candidates
            if item["difficulty"] == level
            for capability in item["capabilities"]
        )
        for level in DIFFICULTIES
    }

    print("\nPREFLIGHT_CANDIDATES")
    print("Difficulty:", json.dumps(dict(available_by_difficulty), ensure_ascii=False))
    print("Capabilities:", json.dumps(dict(available_by_capability), ensure_ascii=False))

    feasibility_errors: dict[str, Any] = {}
    for level, target in DIFFICULTY_QUOTAS.items():
        if available_by_difficulty[level] < target:
            feasibility_errors[f"difficulty:{level}"] = {
                "available": available_by_difficulty[level],
                "required": target,
            }
    for capability, target in CAPABILITY_MINIMUMS.items():
        quota_limited_max = sum(
            min(
                available_capability_by_difficulty[level][capability],
                DIFFICULTY_QUOTAS[level],
            )
            for level in DIFFICULTIES
        )
        if quota_limited_max < target:
            feasibility_errors[f"capability:{capability}"] = {
                "available": available_by_capability[capability],
                "quota_limited_max": quota_limited_max,
                "required": target,
            }
    if feasibility_errors:
        raise RuntimeError(
            "Preflight infeasible; no output was written:\n"
            + json.dumps(feasibility_errors, ensure_ascii=False, indent=2)
        )

    selected = choose_selection(candidates, args.seed, args.selection_restarts)
    selected_by_difficulty = Counter(item["difficulty"] for item in selected)
    selected_by_capability = Counter(
        capability for item in selected for capability in item["capabilities"]
    )
    selected_by_family = Counter(item["family"] for item in selected)
    selected_gate_overlap = sum(
        1
        for item in selected
        if item["id"] in gate_ids
        or normalize_text(get_user_text(item["row"])) in gate_prompts
    )
    token_values = [item["tokens"] for item in selected]

    quality_gates = {
        "rows_exactly_1000": len(selected) == 1000,
        "difficulty_quotas_exact": selected_by_difficulty == Counter(DIFFICULTY_QUOTAS),
        "capability_minimums_met": all(
            selected_by_capability[capability] >= target
            for capability, target in CAPABILITY_MINIMUMS.items()
        ),
        "all_gold_sql_executable": len(selected) == 1000,
        "gate_overlap_zero": selected_gate_overlap == 0,
        "all_tokens_within_limit": min(token_values) > 2
        and max(token_values) <= args.max_tokens,
        "unique_ids": len({item["id"] for item in selected}) == 1000,
        "unique_prompts": len(
            {normalize_text(get_user_text(item["row"])) for item in selected}
        )
        == 1000,
        "unique_database_sql_pairs": len({item["db_sql_hash"] for item in selected})
        == 1000,
    }
    if not all(quality_gates.values()):
        raise RuntimeError(
            "Final quality gate failed; no output was written:\n"
            + json.dumps(quality_gates, ensure_ascii=False, indent=2)
        )

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    staging_dir = output_dir / f".staging-{os.getpid()}"
    if staging_dir.exists():
        shutil.rmtree(staging_dir)
    staging_dir.mkdir()
    try:
        data_path = staging_dir / "pilot1000.jsonl"
        audit_path = staging_dir / "selection_audit.jsonl"
        manifest_path = staging_dir / "manifest.json"
        checksum_path = staging_dir / "sha256sums.txt"

        write_jsonl(data_path, (item["row"] for item in selected))
        write_jsonl(
            audit_path,
            (
                {
                    key: value
                    for key, value in item.items()
                    if key not in ("row", "candidate_key")
                }
                for item in selected
            ),
        )

        manifest = {
            "build_version": BUILD_VERSION,
            "seed": args.seed,
            "source_train": str(args.source_train.resolve()),
            "gate_excluded": str(args.gate_dev.resolve()),
            "database_search_root": str(args.db_search_root.resolve()),
            "max_tokens": args.max_tokens,
            "rows": len(selected),
            "difficulty_quotas": DIFFICULTY_QUOTAS,
            "selected_by_difficulty": dict(selected_by_difficulty),
            "capability_minimums": CAPABILITY_MINIMUMS,
            "candidate_capability_availability": dict(available_by_capability),
            "selected_by_capability": dict(selected_by_capability),
            "selected_by_family": dict(selected_by_family),
            "gold_sql_executable": len(selected),
            "gate_overlap": selected_gate_overlap,
            "unique_ids": len({item["id"] for item in selected}),
            "unique_prompts": len(
                {normalize_text(get_user_text(item["row"])) for item in selected}
            ),
            "unique_database_sql_pairs": len(
                {item["db_sql_hash"] for item in selected}
            ),
            "unique_exact_sql_texts": len({item["sql_hash"] for item in selected}),
            "unique_sql_structures": len(
                {item["structure_signature"] for item in selected}
            ),
            "unique_databases": len({item["db_id"] for item in selected}),
            "token_statistics": {
                "min": min(token_values),
                "mean": round(statistics.fmean(token_values), 2),
                "p50": statistics.median(token_values),
                "p95": sorted(token_values)[int(0.95 * (len(token_values) - 1))],
                "max": max(token_values),
            },
            "candidate_rejects": dict(rejects),
            "quality_gates": quality_gates,
            "final_gate": "PASS",
        }
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        checksum_lines = [
            f"{sha256(path)}  {path.name}"
            for path in (data_path, audit_path, manifest_path)
        ]
        checksum_path.write_text("\n".join(checksum_lines) + "\n", encoding="utf-8")

        for path in (data_path, audit_path, manifest_path, checksum_path):
            os.replace(path, output_dir / path.name)
    finally:
        if staging_dir.exists():
            shutil.rmtree(staging_dir)

    print("\nFINAL_SUMMARY")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    print("\nFINAL_GATE: PASS")
    print("OUTPUT_DIR:", output_dir)


if __name__ == "__main__":
    main()
