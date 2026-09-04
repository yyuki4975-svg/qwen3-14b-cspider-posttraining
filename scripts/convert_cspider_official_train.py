#!/usr/bin/env python3
"""Convert official CSpider train.json into the project's chat JSONL contract.

The converter deliberately delegates difficulty labelling to the official
Spider/CSpider Evaluator and renders database schemas from official tables.json.
It writes output atomically and produces a reproducibility manifest plus SHA256.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


DEFAULT_SYSTEM = "你是中文 Text-to-SQL 助手。根据给定数据库结构生成一条只读 SQLite SQL；只输出 SQL。"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-train", type=Path, required=True)
    parser.add_argument("--tables", type=Path, required=True)
    parser.add_argument("--database-root", type=Path, required=True)
    parser.add_argument("--evaluation-script", type=Path, required=True)
    parser.add_argument("--reference-jsonl", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--source-name", default="CSpider-1.0")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def load_official_evaluator(path: Path) -> Any:
    if not path.is_file():
        raise FileNotFoundError(f"official evaluator not found: {path}")
    sys.path.insert(0, str(path.parent.resolve()))
    spec = importlib.util.spec_from_file_location("cspider_official_evaluation", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import evaluator: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    evaluator_type = getattr(module, "Evaluator", None)
    if evaluator_type is None:
        raise RuntimeError("official evaluation.py does not expose Evaluator")
    evaluator = evaluator_type()
    if not hasattr(evaluator, "eval_hardness"):
        raise RuntimeError("official Evaluator does not expose eval_hardness")
    return evaluator


def load_system_prompt(reference: Path | None) -> str:
    if reference is None:
        return DEFAULT_SYSTEM
    if not reference.is_file():
        raise FileNotFoundError(f"reference JSONL not found: {reference}")
    with reference.open(encoding="utf-8") as handle:
        first = json.loads(next(line for line in handle if line.strip()))
    for message in first.get("messages", []):
        if message.get("role") == "system" and str(message.get("content", "")).strip():
            return str(message["content"]).strip()
    raise RuntimeError("reference JSONL has no system message")


def normalize_type(value: Any) -> str:
    value = str(value or "text").strip().lower()
    return value if value else "text"


def build_schema_map(tables_rows: list[dict[str, Any]]) -> dict[str, str]:
    schemas: dict[str, str] = {}
    for row in tables_rows:
        db_id = str(row.get("db_id", "")).strip()
        if not db_id:
            raise ValueError("tables.json row missing db_id")

        table_names = row.get("table_names_original") or row.get("table_names")
        columns = row.get("column_names_original") or row.get("column_names")
        column_types = row.get("column_types") or []
        if not isinstance(table_names, list) or not isinstance(columns, list):
            raise ValueError(f"invalid table schema for {db_id}")

        grouped: dict[int, list[str]] = defaultdict(list)
        for column_index, column in enumerate(columns):
            if not isinstance(column, list) or len(column) != 2:
                raise ValueError(f"invalid column entry for {db_id}: {column!r}")
            table_index, column_name = column
            if int(table_index) < 0:
                continue
            column_type = (
                normalize_type(column_types[column_index])
                if column_index < len(column_types)
                else "text"
            )
            grouped[int(table_index)].append(f"{column_name} {column_type}")

        lines = []
        for table_index, table_name in enumerate(table_names):
            lines.append(f"{table_name}({', '.join(grouped.get(table_index, []))})")

        def qualified(column_index: int) -> str:
            table_index, column_name = columns[int(column_index)]
            if int(table_index) < 0:
                raise ValueError(f"invalid qualified column for {db_id}: {column_index}")
            return f"{table_names[int(table_index)]}.{column_name}"

        primary_keys = row.get("primary_keys") or []
        if primary_keys:
            lines.append("PRIMARY KEY: " + ", ".join(qualified(i) for i in primary_keys))

        foreign_keys = row.get("foreign_keys") or []
        if foreign_keys:
            lines.append(
                "FOREIGN KEY: "
                + ", ".join(f"{qualified(left)} -> {qualified(right)}" for left, right in foreign_keys)
            )

        schemas[db_id] = "\n".join(lines)
    return schemas


def find_database(database_root: Path, db_id: str) -> Path:
    candidates = [
        database_root / db_id / f"{db_id}.sqlite",
        database_root / db_id / f"{db_id}.db",
        database_root / f"{db_id}.sqlite",
        database_root / f"{db_id}.db",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    recursive = sorted(
        path.resolve()
        for path in database_root.rglob("*")
        if path.is_file()
        and path.stem.lower() == db_id.lower()
        and path.suffix.lower() in {".sqlite", ".sqlite3", ".db"}
    )
    if len(recursive) != 1:
        raise FileNotFoundError(
            f"expected exactly one database for {db_id}, found {len(recursive)} under {database_root}"
        )
    return recursive[0]


def atomic_write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    count = 0
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
                count += 1
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise
    return count


def main() -> None:
    args = parse_args()
    for path in (args.source_train, args.tables, args.evaluation_script):
        if not path.is_file():
            raise FileNotFoundError(path)
    if not args.database_root.is_dir():
        raise NotADirectoryError(args.database_root)

    raw_rows = load_json(args.source_train)
    table_rows = load_json(args.tables)
    if not isinstance(raw_rows, list) or not raw_rows:
        raise ValueError("source train must be a non-empty JSON array")
    if not isinstance(table_rows, list) or not table_rows:
        raise ValueError("tables must be a non-empty JSON array")

    evaluator = load_official_evaluator(args.evaluation_script)
    schema_map = build_schema_map(table_rows)
    system_prompt = load_system_prompt(args.reference_jsonl)
    database_paths: dict[str, Path] = {}
    converted: list[dict[str, Any]] = []
    difficulty_counts: Counter[str] = Counter()
    seen_ids: set[str] = set()

    for index, raw in enumerate(raw_rows):
        if not isinstance(raw, dict):
            raise ValueError(f"row {index} is not an object")
        db_id = str(raw.get("db_id", "")).strip()
        question = str(raw.get("question", "")).strip()
        query = str(raw.get("query", "")).strip().rstrip(";").strip()
        parsed_sql = raw.get("sql")
        if not db_id or not question or not query or not isinstance(parsed_sql, dict):
            raise ValueError(f"row {index} lacks db_id/question/query/parsed sql")
        if db_id not in schema_map:
            raise KeyError(f"tables.json lacks db_id={db_id}")
        if db_id not in database_paths:
            database_paths[db_id] = find_database(args.database_root, db_id)

        difficulty = str(evaluator.eval_hardness(parsed_sql)).strip().lower()
        if difficulty not in {"easy", "medium", "hard", "extra"}:
            raise ValueError(f"unknown hardness at row {index}: {difficulty!r}")
        difficulty_counts[difficulty] += 1

        row_hash = hashlib.sha256(
            f"{index}\0{db_id}\0{question}\0{query}".encode("utf-8")
        ).hexdigest()[:16]
        qid = f"cspider-train-{index:05d}-{row_hash}"
        if qid in seen_ids:
            raise RuntimeError(f"duplicate generated id: {qid}")
        seen_ids.add(qid)

        user_prompt = (
            "请把问题转换为一条可在 SQLite 执行的只读 SQL。\n"
            "只输出 SQL，不要解释，不要 Markdown，不要思考过程。\n\n"
            "数据库结构：\n"
            f"{schema_map[db_id]}\n\n"
            f"问题：{question}"
        )
        converted.append(
            {
                "id": qid,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                    {"role": "assistant", "content": query},
                ],
                "metadata": {
                    "source": args.source_name,
                    "split": "train",
                    "source_index": index,
                    "db_id": db_id,
                    "question": question,
                    "difficulty": difficulty,
                    "gold_sql": query,
                    "database": str(database_paths[db_id]),
                },
            }
        )

    written = atomic_write_jsonl(args.output, converted)
    if written != len(raw_rows):
        raise RuntimeError(f"row count mismatch: wrote {written}, expected {len(raw_rows)}")

    manifest = {
        "format": "cspider-chat-jsonl-v1",
        "source_train": str(args.source_train.resolve()),
        "source_train_sha256": sha256_file(args.source_train),
        "tables": str(args.tables.resolve()),
        "tables_sha256": sha256_file(args.tables),
        "evaluation_script": str(args.evaluation_script.resolve()),
        "evaluation_script_sha256": sha256_file(args.evaluation_script),
        "database_root": str(args.database_root.resolve()),
        "output": str(args.output.resolve()),
        "output_sha256": sha256_file(args.output),
        "rows": written,
        "databases": len(database_paths),
        "difficulty": dict(sorted(difficulty_counts.items())),
        "system_prompt": system_prompt,
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (args.output.parent / "sha256sums.txt").write_text(
        f"{manifest['output_sha256']}  {args.output.name}\n"
        f"{sha256_file(args.manifest)}  {args.manifest.name}\n",
        encoding="utf-8",
    )

    print("CONVERTED_ROWS:", written)
    print("DATABASES:", len(database_paths))
    print("DIFFICULTY:", json.dumps(dict(difficulty_counts), ensure_ascii=False, sort_keys=True))
    print("OUTPUT:", args.output)
    print("MANIFEST:", args.manifest)


if __name__ == "__main__":
    main()
