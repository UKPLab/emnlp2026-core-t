#!/usr/bin/env python3
"""
SQL executor.

This script takes the outputs of `sql_generator.py` and executes each SQL against the referenced SQLite tables. It does NOT perform
any evaluation/scoring; it only runs the query and records execution outcomes.

Usage:
    python sql_executor.py --dataset bird --llm-model "huggingface:Qwen/Qwen2.5-7B-Instruct" --embedding-model "fireworks:WhereIsAI/UAE-Large-V1"
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv


TABLE_ID_SEPARATOR = "#sep#"


def _project_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "sql_database").exists():
            return parent
    return current.parent


PROJECT_ROOT = _project_root()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Best-effort env loading; the script should still run if .env is absent.
try:
    load_dotenv(PROJECT_ROOT / ".env")
except Exception:
    pass

from utils import Configuration, DatabaseType, get_results_run_dir  # noqa: E402
from utils.cache_scripts.sql_generation_cache import get_sql_execution_cache  # noqa: E402
from utils.db_connector import execute_sql_on_attached_tables  # noqa: E402


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, payload: Any, *, indent: int = 2) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=indent, ensure_ascii=False)


def _sanitize_sql_for_execution(raw_sql: Optional[str]) -> Optional[str]:
    """
    Best-effort cleanup for cached/LLM SQL strings before handing them to SQLite.

    This strips common Markdown code fences and <think> blocks that frequently
    appear in cached entries from older runs.
    """
    if raw_sql is None:
        return None
    text = str(raw_sql).strip()
    if not text:
        return None
    # Remove <think> blocks (some models emit them).
    try:
        text = re.sub(r"<think>[\s\S]*?</think>", "", text, flags=re.IGNORECASE).strip()
    except Exception:
        pass
    # Strip Markdown fences.
    if text.startswith("```sqlite"):
        text = text[len("```sqlite") :].strip()
    if text.startswith("```sql"):
        text = text[len("```sql") :].strip()
    if text.startswith("```"):
        text = text[len("```") :].strip()
    if text.endswith("```"):
        text = text[: -len("```")].strip()
    return text or None


def _extract_tables_for_execution(details: Dict[str, Any]) -> Tuple[List[str], str]:
    """
    Prefer the most faithful full identifiers available.
    `sql_generator.py` writes both `selected_tables_used` and `resolved_tables`.
    """
    # Follows the preference order:
    #  1) sql_tables_full_ids (if present)
    #  2) fallback to original table list
    #  3) sql_tables (last resort)
    for key in ("sql_tables_full_ids", "resolved_tables", "selected_tables_used", "tables", "selected_tables_input", "sql_tables"):
        val = details.get(key)
        if isinstance(val, list) and val:
            return [str(x) for x in val if isinstance(x, str) and x], str(key)
    return [], ""


def _extract_alias_mapping(details: Dict[str, Any]) -> Optional[Dict[str, str]]:
    mapping = details.get("table_alias_mapping")
    if isinstance(mapping, dict) and mapping:
        out: Dict[str, str] = {}
        for k, v in mapping.items():
            if isinstance(k, str) and isinstance(v, str):
                out[k] = v
        return out or None
    return None


def _unique_db_ids(table_full_ids: List[str], alias_to_full_id: Optional[Dict[str, str]]) -> List[str]:
    dbs: List[str] = []
    candidates: List[str] = []
    if alias_to_full_id:
        candidates.extend(list(alias_to_full_id.values()))
    candidates.extend(table_full_ids or [])
    for identifier in candidates:
        if TABLE_ID_SEPARATOR in identifier:
            db_id = identifier.split(TABLE_ID_SEPARATOR, 1)[0]
            if db_id and db_id not in dbs:
                dbs.append(db_id)
    return dbs


def _execute_one(
    *,
    sql: str,
    table_full_ids: List[str],
    alias_to_full_id: Optional[Dict[str, str]],
    config: Configuration,
    row_limit: Optional[int],
    timeout_seconds: int,
) -> Tuple[List[Dict[str, Any]], bool]:
    # ATTACH-based execution (no copying into memory).
    return execute_sql_on_attached_tables(
        sql,
        table_full_ids,
        config,
        row_limit=row_limit,
        alias_to_full_id=alias_to_full_id,
        timeout_seconds=int(timeout_seconds),
    )


def run_execute_with_execution_cache(
    sql: str,
    table_full_ids: List[str],
    config: Configuration,
    *,
    alias_to_full_id: Optional[Dict[str, str]] = None,
    row_limit: Optional[int] = None,
    timeout_seconds: int = 60,
    use_cache: bool = True,
) -> Tuple[List[Dict[str, Any]], bool, Optional[str], bool]:
    """Run ``execute_sql_on_attached_tables``, optionally via dataset-global cache (``SQLExecutionCache``).

    Returns:
        (rows, duplicates_skipped, error, cache_hit)
        ``error`` is ``None`` on success. ``cache_hit`` is True only when ``use_cache`` and a cache entry was used.
    """
    if not use_cache:
        rows: List[Dict[str, Any]] = []
        dup = False
        err: Optional[str] = None
        try:
            rows, dup = execute_sql_on_attached_tables(
                sql,
                table_full_ids,
                config,
                row_limit=row_limit,
                alias_to_full_id=alias_to_full_id,
                timeout_seconds=int(timeout_seconds),
            )
        except Exception as e:
            err = str(e)
        return rows, dup, err, False

    exec_cache = get_sql_execution_cache(config.get_dataset_cache_dir("sql_generation"))
    ds = config.database_type.value
    cached = exec_cache.get_execution(
        sql=sql,
        table_full_ids=table_full_ids,
        alias_to_full_id=alias_to_full_id,
        row_limit=row_limit,
        timeout_seconds=int(timeout_seconds),
        dataset=ds,
    )
    if cached is not None:
        if cached.get("success"):
            return (
                list(cached.get("rows") or []),
                bool(cached.get("duplicates_skipped", False)),
                None,
                True,
            )
        err_val = cached.get("error")
        return (
            [],
            bool(cached.get("duplicates_skipped", False)),
            str(err_val) if err_val is not None else "SQL execution failed (cached)",
            True,
        )

    rows_miss: List[Dict[str, Any]] = []
    dup_miss = False
    err_miss: Optional[str] = None
    try:
        rows_miss, dup_miss = execute_sql_on_attached_tables(
            sql,
            table_full_ids,
            config,
            row_limit=row_limit,
            alias_to_full_id=alias_to_full_id,
            timeout_seconds=int(timeout_seconds),
        )
    except Exception as e:
        err_miss = str(e)

    try:
        exec_cache.set_execution(
            sql=sql,
            table_full_ids=table_full_ids,
            alias_to_full_id=alias_to_full_id,
            row_limit=row_limit,
            timeout_seconds=int(timeout_seconds),
            dataset=ds,
            success=err_miss is None,
            rows=rows_miss,
            duplicates_skipped=dup_miss,
            error=err_miss,
        )
    except Exception:
        pass

    return rows_miss, dup_miss, err_miss, False


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Execute generated SQLs (no evaluation).")
    p.add_argument(
        "--sql-generation-dir",
        type=str,
        default=None,
        help=(
            "Path to a results_sql_generation* folder containing sqls.json/sqls_details.json. "
            "If provided, this overrides default results path resolution."
        ),
    )
    p.add_argument(
        "--sqls",
        type=str,
        default=None,
        help="Path to sqls.json from sql_generator.py (defaults to results/<dataset>/{llm}_{embedding_model}/results_sql_generation/sqls.json).",
    )
    p.add_argument(
        "--sqls-details",
        type=str,
        default=None,
        help="Path to sqls_details.json from sql_generator.py (defaults to results/<dataset>/{llm}_{embedding_model}/results_sql_generation/sqls_details.json).",
    )
    p.add_argument(
        "--dataset",
        dest="dataset",
        choices=[dt.value for dt in DatabaseType],
        default=DatabaseType.BIRD.value,
        help="Dataset used for DB path resolution and results path resolution.",
    )
    # Backward-compatible alias.
    p.add_argument(
        "--database-type",
        dest="dataset",
        choices=[dt.value for dt in DatabaseType],
        default=DatabaseType.BIRD.value,
        help="(deprecated) Same as --dataset.",
    )
    p.add_argument("--llm-model", type=str, default="openai:gpt-4o-mini", help="LLM tag used for results run folder naming.")
    p.add_argument(
        "--embedding-model",
        type=str,
        default="fireworks:WhereIsAI/UAE-Large-V1",
        help="Embedding model tag used for results run folder naming.",
    )
    p.add_argument(
        "--optimize-sql",
        action="store_true",
        default=False,
        help="Best-effort SQL normalization via sqlglot (if installed).",
    )
    p.add_argument(
        "--row-limit",
        type=int,
        default=300000,
        help="Max rows to fetch per query (default: 300000). Use -1 for no limit.",
    )
    p.add_argument(
        "--timeout-seconds",
        type=int,
        default=60,
        help="SQLite execution timeout in seconds (default: 60).",
    )
    p.add_argument("--start", type=int, default=0, help="Start index (default: 0).")
    p.add_argument("--limit", type=int, default=None, help="Optional number of items to process.")
    p.add_argument("--no-rows", action="store_true", default=False, help="Do not store rows in outputs (summary only).")
    p.add_argument(
        "--cache-execution",
        action="store_true",
        default=False,
        help=(
            "Read/write sql_execution_cache.json under cache/<dataset>/sql_generation/. "
            "When omitted (default), every query is executed against SQLite with no execution cache."
        ),
    )
    p.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output JSON path (defaults to results/<dataset>/{llm}_{embedding_model}/results_sql_generation/sqls_exec.json).",
    )
    p.add_argument(
        "--out-details",
        type=str,
        default=None,
        help="Output JSON path (defaults to results/<dataset>/{llm}_{embedding_model}/results_sql_generation/sqls_exec_details.json).",
    )
    p.add_argument("--indent", type=int, default=2, help="JSON indentation level.")
    return p.parse_args(argv)


def main(args: argparse.Namespace) -> int:
    if args.sql_generation_dir:
        gen_dir = Path(args.sql_generation_dir).expanduser().resolve()
    else:
        gen_dir = get_results_run_dir(
            dataset=str(args.dataset),
            step_dirname="results_sql_generation",
            llm_model=str(args.llm_model),
            embedding_model=str(args.embedding_model),
            project_root=PROJECT_ROOT,
        )
    sqls_path = Path(args.sqls).expanduser().resolve() if args.sqls else (gen_dir / "sqls.json").resolve()
    details_path = Path(args.sqls_details).expanduser().resolve() if args.sqls_details else (gen_dir / "sqls_details.json").resolve()
    if not sqls_path.exists():
        print(f"❌ sqls file not found: {sqls_path}")
        return 1
    if not details_path.exists():
        print(f"❌ sqls_details file not found: {details_path}")
        return 1

    sqls_payload = _read_json(sqls_path)
    details_payload = _read_json(details_path)
    if not isinstance(sqls_payload, list):
        print("❌ --sqls must be a JSON list[str|null].")
        return 1
    if not isinstance(details_payload, list):
        print("❌ --sqls-details must be a JSON list[dict].")
        return 1

    sqls: List[Optional[str]] = [str(x) if isinstance(x, str) else None for x in sqls_payload]
    details: List[Dict[str, Any]] = [x for x in details_payload if isinstance(x, dict)]
    n = min(len(sqls), len(details))
    if n == 0:
        print("Nothing to do.")
        return 0

    start = max(0, int(args.start or 0))
    end = n if args.limit is None else min(n, start + int(args.limit))
    if start >= end:
        print(f"Nothing to do: start={start} end={end} (available={n})")
        return 0

    dataset = DatabaseType(args.dataset)
    # llm_model/embedding_model are used for run folder naming + any downstream path helpers.
    config = Configuration(
        database_type=dataset,
        llm_model=str(args.llm_model),
        embedding_model=str(args.embedding_model),
        temperature=0.0,
    )

    row_limit: Optional[int]
    if args.row_limit is None:
        row_limit = 50
    else:
        row_limit = None if int(args.row_limit) < 0 else int(args.row_limit)

    out_summary: List[Dict[str, Any]] = []
    out_details: List[Dict[str, Any]] = []

    print(f"Loaded SQLs: {len(sqls)} | details: {len(details)} | processing: {end-start}")
    print(
        f"row_limit: {row_limit if row_limit is not None else 'none'} | "
        f"timeout_seconds: {int(args.timeout_seconds)} | "
        f"cache_execution: {bool(args.cache_execution)}"
    )

    for i in range(start, end):
        d = details[i]
        sql = sqls[i] if i < len(sqls) else None
        if not sql and isinstance(d.get("sql"), str):
            sql = str(d.get("sql"))
        sql = _sanitize_sql_for_execution(sql)

        if sql and bool(args.optimize_sql):
            try:
                import sqlglot

                # Follows the optimize routine:
                # - parse as sqlite
                # - if sqlglot.optimizer.optimize exists, run it
                # - otherwise fall back to dialect-normalized SQL
                expr = sqlglot.parse_one(str(sql), read="sqlite")
                optimizer = getattr(sqlglot, "optimizer", None)
                if optimizer and hasattr(optimizer, "optimize"):
                    sql = optimizer.optimize(expr).sql(dialect="sqlite")
                else:
                    sql = expr.sql(dialect="sqlite")
            except Exception:
                pass

        table_full_ids, table_full_ids_source = _extract_tables_for_execution(d)
        alias_to_full_id = _extract_alias_mapping(d)
        db_ids = _unique_db_ids(table_full_ids, alias_to_full_id)

        qid = d.get("question_id", i)
        db_id = d.get("db_id")

        t0 = time.time()
        rows: List[Dict[str, Any]] = []
        error: Optional[str] = None
        duplicates_skipped = False
        executed = False
        execution_cache_hit = False

        # - If SQL is missing, treat as "not executed" (no exception).
        # - If tables are missing, treat as "not executed" (no exception).
        if sql and (table_full_ids or alias_to_full_id):
            try:
                rows, duplicates_skipped, error, execution_cache_hit = run_execute_with_execution_cache(
                    str(sql),
                    table_full_ids,
                    config,
                    alias_to_full_id=alias_to_full_id,
                    row_limit=row_limit,
                    timeout_seconds=int(args.timeout_seconds),
                    use_cache=bool(args.cache_execution),
                )
                executed = error is None
            except Exception:  # pragma: no cover - defensive fallback
                execution_cache_hit = False
                try:
                    rows, duplicates_skipped = _execute_one(
                        sql=str(sql),
                        table_full_ids=table_full_ids,
                        alias_to_full_id=alias_to_full_id,
                        config=config,
                        row_limit=row_limit,
                        timeout_seconds=int(args.timeout_seconds),
                    )
                    executed = True
                    error = None
                except Exception as e:
                    error = str(e)
                    rows = []
                    duplicates_skipped = False
                    executed = False

        elapsed_s = float(time.time() - t0)
        row_count = int(len(rows or []))

        summary = {
            "question_id": qid,
            "index": i,
            "db_id": db_id,
            "executed": bool(executed),
            "error": error,
            "row_count": row_count,
            "execution_time_s": elapsed_s,
            "duplicates_skipped": bool(duplicates_skipped),
            "db_ids_involved": db_ids,
            "tables_used_source": table_full_ids_source,
            "execution_cache_hit": bool(execution_cache_hit),
        }
        out_summary.append(summary)

        detail = dict(summary)
        detail.update(
            {
                "sql": sql,
                "tables_used": table_full_ids,
                "table_alias_mapping": alias_to_full_id or {},
                "tables_used_source": table_full_ids_source,
            }
        )
        if not bool(args.no_rows):
            detail["rows"] = rows
        out_details.append(detail)

        ok = executed and not error
        print(
            f"  - question_id={qid} idx={i} db={db_id} "
            f"tables={len(table_full_ids)} "
            f"rows={row_count} -> {'OK' if ok else 'FAIL'}"
        )

    # Write executor outputs alongside sql generation outputs (same run folder).
    out_path = Path(args.out).expanduser().resolve() if args.out else (gen_dir / "sqls_exec.json").resolve()
    out_details_path = (
        Path(args.out_details).expanduser().resolve() if args.out_details else (gen_dir / "sqls_exec_details.json").resolve()
    )
    _write_json(out_path, out_summary, indent=int(args.indent))
    _write_json(out_details_path, out_details, indent=int(args.indent))
    print(f"Wrote execution summary: {out_path}")
    print(f"Wrote execution details: {out_details_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(_parse_args()))

