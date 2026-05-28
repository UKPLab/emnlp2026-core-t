#!/usr/bin/env python3
"""
SQL execution evaluation.

This script evaluates generated SQL outputs by comparing executed predicted rows
against gold execution results stored in:

    data/{dataset}/dev_query_results.json

Primary metric:
    SQL execution accuracy = number of questions whose predicted SQL executes and
    returns the same result as gold / total number of questions.

Evaluation behavior:
    - Failed SQL execution counts as incorrect.
    - Values-only comparison ignores column names but preserves column/value order.
    - If the gold SQL has an outer ORDER BY, row order is respected.
    - Otherwise, rows are compared as an unordered multiset.
    - Numeric tolerance is applied only to non-bool int/float values.
    - Empty gold results [] are valid and are evaluated by default.
    - Reports HasAns / NoAns execution-accuracy slices (non-empty vs empty gold results).
    - Reports execution accuracy on table-perfect-recall slices (all gold ⊆ pred tables):

Usage:
    python evaluate_sqls.py \
        --dataset bird \
        --llm-model "huggingface:Qwen/Qwen2.5-7B-Instruct" \
        --embedding-model "fireworks:WhereIsAI/UAE-Large-V1"
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from dotenv import load_dotenv


def _project_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "data").exists() and (parent / "sql_database").exists():
            return parent
    return current.parent


PROJECT_ROOT = _project_root()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    load_dotenv(PROJECT_ROOT / ".env")
except Exception:
    pass

from utils import Configuration, DatabaseType, get_results_run_dir  # noqa: E402

TABLE_KEY_SEP = "#sep#"


def _normalize_table_id(table_id: Any, *, db_id: Optional[str] = None) -> str:
    """
    Normalize table identifiers so gold and predicted tables are comparable.

    Gold (dev_gold_tables.json) often stores bare names: ``BUILDINGS``.
    SQL executor stores full ids: ``dw#sep#BUILDINGS``.
    """
    s = str(table_id or "").strip()
    if not s:
        return ""

    if TABLE_KEY_SEP in s:
        dbp, tp = s.split(TABLE_KEY_SEP, 1)
        return f"{dbp.strip().lower()}{TABLE_KEY_SEP}{tp.strip().lower()}"

    db = str(db_id or "").strip().lower()
    tname = s.lower()
    if db:
        return f"{db}{TABLE_KEY_SEP}{tname}"
    return tname


def _table_set_from_ids(table_ids: Sequence[str], *, db_id: Optional[str] = None) -> set:
    return {_normalize_table_id(t, db_id=db_id) for t in table_ids if _normalize_table_id(t, db_id=db_id)}


from sql_executor import (  # noqa: E402
    _extract_alias_mapping,
    _extract_tables_for_execution,
    _sanitize_sql_for_execution,
    run_execute_with_execution_cache,
)  # noqa: E402


# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, payload: Any, *, indent: int = 2) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=indent, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Value/result comparison helpers
# ---------------------------------------------------------------------------


def _is_number_not_bool(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _normalize_scalar(value: Any) -> Any:
    """
    Normalize scalar values for comparison.

    This is intentionally conservative:
      - Does not coerce strings to numbers.
      - Does not lowercase strings.
      - Does not strip strings by default, because SQL string values may contain
        meaningful leading/trailing spaces.
    """
    if isinstance(value, float):
        if math.isnan(value):
            return "__NaN__"
        if math.isinf(value):
            return "__INF__" if value > 0 else "__-INF__"
    return value


def _values_equal_with_tolerance(val1: Any, val2: Any, float_tolerance: float) -> bool:
    """
    Compare two scalar values.

    Numeric tolerance is applied only when both values are int/float and neither is bool.
    Strings are not coerced to numbers.
    """
    val1 = _normalize_scalar(val1)
    val2 = _normalize_scalar(val2)

    if val1 is None and val2 is None:
        return True
    if val1 is None or val2 is None:
        return False

    if isinstance(val1, bool) or isinstance(val2, bool):
        return val1 == val2

    if _is_number_not_bool(val1) and _is_number_not_bool(val2):
        try:
            return abs(float(val1) - float(val2)) <= float(float_tolerance)
        except (ValueError, TypeError, OverflowError):
            return False

    return val1 == val2


def _compare_value_tuples_with_tolerance(
    t1: Sequence[Any],
    t2: Sequence[Any],
    *,
    float_tolerance: float,
) -> bool:
    if len(t1) != len(t2):
        return False
    return all(
        _values_equal_with_tolerance(a, b, float_tolerance=float_tolerance)
        for a, b in zip(t1, t2)
    )


def _row_to_values_tuple(row: Any) -> Tuple[Any, ...]:
    """
    Convert a row into a tuple of values while preserving column/value order.

    For dict rows, Python preserves insertion order. This assumes the upstream
    executor serialized rows in SELECT-column order.
    """
    if isinstance(row, dict):
        return tuple(_normalize_scalar(v) for v in row.values())
    if isinstance(row, (list, tuple)):
        return tuple(_normalize_scalar(v) for v in row)
    return (_normalize_scalar(row),)


def _row_to_key_value_tuple(row: Any) -> Tuple[Tuple[str, Any], ...]:
    """
    Convert a row into a tuple of (column_name, value) pairs.

    We sort by column name only to avoid false negatives caused by dict insertion
    differences when column names are being evaluated.
    """
    if isinstance(row, dict):
        return tuple((str(k), _normalize_scalar(v)) for k, v in sorted(row.items(), key=lambda kv: str(kv[0])))
    if isinstance(row, (list, tuple)):
        return tuple((str(i), _normalize_scalar(v)) for i, v in enumerate(row))
    return (("0", _normalize_scalar(row)),)


def _key_value_rows_equal_with_tolerance(
    r1: Tuple[Tuple[str, Any], ...],
    r2: Tuple[Tuple[str, Any], ...],
    *,
    float_tolerance: float,
) -> bool:
    if len(r1) != len(r2):
        return False

    for (k1, v1), (k2, v2) in zip(r1, r2):
        if k1 != k2:
            return False
        if not _values_equal_with_tolerance(v1, v2, float_tolerance=float_tolerance):
            return False

    return True


def _rows_equal(
    row1: Any,
    row2: Any,
    *,
    values_only: bool,
    float_tolerance: float,
) -> bool:
    if values_only:
        return _compare_value_tuples_with_tolerance(
            _row_to_values_tuple(row1),
            _row_to_values_tuple(row2),
            float_tolerance=float_tolerance,
        )

    return _key_value_rows_equal_with_tolerance(
        _row_to_key_value_tuple(row1),
        _row_to_key_value_tuple(row2),
        float_tolerance=float_tolerance,
    )


def _canonical_row_key(row: Any, *, values_only: bool) -> str:
    """
    Deterministic key used only for sorting exact-ish row representations.

    We still perform tolerance-aware equality after sorting. For small result sets
    with tolerance, we use greedy matching instead.
    """
    if values_only:
        return json.dumps(_row_to_values_tuple(row), ensure_ascii=False, sort_keys=False, default=str)
    return json.dumps(_row_to_key_value_tuple(row), ensure_ascii=False, sort_keys=False, default=str)


def _has_float_like_values(rows: Sequence[Any]) -> bool:
    for row in rows:
        values: Iterable[Any]
        if isinstance(row, dict):
            values = row.values()
        elif isinstance(row, (list, tuple)):
            values = row
        else:
            values = (row,)

        for value in values:
            if isinstance(value, float):
                return True
    return False


def _unordered_rows_equal_with_tolerance(
    pred_rows: List[Any],
    gold_rows: List[Any],
    *,
    values_only: bool,
    float_tolerance: float,
    greedy_match_threshold: int = 10000,
) -> bool:
    """
    Compare result rows as an unordered multiset.

    For exact comparisons or very large results, we sort by canonical key and compare
    pairwise. For smaller float-tolerance cases, use greedy matching because tolerance
    equality is not always compatible with sort-and-zip pairing.
    """
    if len(pred_rows) != len(gold_rows):
        return False

    if not pred_rows and not gold_rows:
        return True

    needs_tolerance_matching = float_tolerance > 0 and (
        _has_float_like_values(pred_rows) or _has_float_like_values(gold_rows)
    )

    if needs_tolerance_matching and len(pred_rows) <= greedy_match_threshold:
        unmatched_gold = list(gold_rows)
        for pred in pred_rows:
            found_at: Optional[int] = None
            for j, gold in enumerate(unmatched_gold):
                if _rows_equal(
                    pred,
                    gold,
                    values_only=values_only,
                    float_tolerance=float_tolerance,
                ):
                    found_at = j
                    break

            if found_at is None:
                return False

            unmatched_gold.pop(found_at)

        return not unmatched_gold

    pred_sorted = sorted(pred_rows, key=lambda r: _canonical_row_key(r, values_only=values_only))
    gold_sorted = sorted(gold_rows, key=lambda r: _canonical_row_key(r, values_only=values_only))

    return all(
        _rows_equal(
            pred,
            gold,
            values_only=values_only,
            float_tolerance=float_tolerance,
        )
        for pred, gold in zip(pred_sorted, gold_sorted)
    )


def compare_ordered_rows(
    result_pred: Any,
    result_gold: Any,
    *,
    values_only: bool,
    float_tolerance: float,
) -> bool:
    """
    Row-by-row comparison with row order preserved.
    """
    if not isinstance(result_pred, list) or not isinstance(result_gold, list):
        return False

    if len(result_pred) != len(result_gold):
        return False

    return all(
        _rows_equal(
            row_p,
            row_g,
            values_only=values_only,
            float_tolerance=float_tolerance,
        )
        for row_p, row_g in zip(result_pred, result_gold)
    )


def compare_query_results(
    result_pred: Any,
    result_gold: Any,
    *,
    values_only: bool,
    float_tolerance: float,
    order_sensitive: bool,
) -> bool:
    """
    Compare two query results.

    If order_sensitive=True:
        compare row-by-row in the given order.

    If order_sensitive=False:
        compare rows as an unordered multiset.

    If values_only=True:
        compare row values only, preserving value order within each row.

    If values_only=False:
        compare column names and values.
    """
    if not isinstance(result_pred, list) or not isinstance(result_gold, list):
        return False

    if len(result_pred) != len(result_gold):
        return False

    if order_sensitive:
        return compare_ordered_rows(
            result_pred,
            result_gold,
            values_only=values_only,
            float_tolerance=float_tolerance,
        )

    return _unordered_rows_equal_with_tolerance(
        result_pred,
        result_gold,
        values_only=values_only,
        float_tolerance=float_tolerance,
    )


# ---------------------------------------------------------------------------
# Metadata and SQL helpers
# ---------------------------------------------------------------------------


def _truncate_rows(rows: Any, max_rows: int) -> Any:
    if not isinstance(rows, list):
        return rows
    if max_rows < 0:
        return rows
    return rows[: int(max_rows)]


def _extract_outer_order_sql(gold_sql: Optional[str]) -> Optional[str]:
    """
    Return the outermost ORDER BY fragment from gold SQL, or None.

    Example return:
        ORDER BY name ASC
    """
    if not gold_sql or not str(gold_sql).strip():
        return None

    try:
        import sqlglot
        from sqlglot import exp

        text = str(gold_sql).strip().rstrip(";")
        parsed = sqlglot.parse_one(text, read="sqlite")

        order = None
        if isinstance(parsed, exp.Union):
            order = parsed.args.get("order")
        elif isinstance(parsed, exp.Select):
            order = parsed.args.get("order")

        if not order:
            return None

        frag = order.sql(dialect="sqlite")
        return frag if frag else None

    except Exception:
        return None


def _gold_sql_is_order_sensitive(gold_sql: Optional[str], gold_rows: Any) -> bool:
    """
    Row order should matter when the gold SQL has an outer ORDER BY.

    For 0/1-row outputs, order is irrelevant even if ORDER BY exists.
    """
    if not isinstance(gold_rows, list) or len(gold_rows) <= 1:
        return False
    return bool(_extract_outer_order_sql(gold_sql))


def _config_for_dataset(dataset: str) -> Configuration:
    key = str(dataset or "bird").strip().lower()
    try:
        dt = DatabaseType(key)
    except ValueError:
        dt = DatabaseType.BIRD
    return Configuration(database_type=dt)


def _load_dev_metadata(path: Path) -> Optional[List[Dict[str, Any]]]:
    """
    Load data/<dataset>/dev.json style list[dict].

    Used for per-index difficulty and gold SQL.
    """
    if not path.exists():
        return None

    try:
        payload = _read_json(path)
        if not isinstance(payload, list):
            return None
        return [x if isinstance(x, dict) else {} for x in payload]
    except Exception:
        return None


def _difficulty_label_for_index(dev_rows: Optional[List[Dict[str, Any]]], i: int) -> str:
    if not dev_rows or i < 0 or i >= len(dev_rows):
        return "unknown"

    d = str(dev_rows[i].get("difficulty", "") or "").strip().lower()
    if d in ("simple", "moderate", "challenging"):
        return d

    return "unknown"


def _prepare_rows_for_compare(
    pred_rows: Any,
    gold_rows: Any,
    *,
    ex: Dict[str, Any],
    gold_sql: Optional[str],
    max_gold_result_rows: int,
    truncated_gold_result_rows: int,
    config: Configuration,
    reorder_exec_timeout: int,
    truncate_legacy_if_no_order_by: bool,
) -> Tuple[Any, Any, Dict[str, Any]]:
    """
    Align pred/gold rows for comparison.

    For huge gold results:
      - If gold SQL has an outer ORDER BY, try re-executing predicted SQL wrapped
        with the same ORDER BY and LIMIT k.
      - If no ORDER BY exists, the safest default is to compare full multisets.
      - Legacy prefix truncation is available only if explicitly enabled.

    Note:
      Prefix comparison can overestimate correctness, so metadata records when it
      was used.
    """
    meta: Dict[str, Any] = {
        "truncated": False,
        "truncation_compare_mode": "full_multiset",
        "order_aligned_reexec": False,
        "reexec_error": None,
        "compare_ordered": False,
        "prefix_length": None,
        "note": None,
    }

    if not isinstance(pred_rows, list) or not isinstance(gold_rows, list):
        return pred_rows, gold_rows, meta

    try:
        max_rows = int(max_gold_result_rows)
        trunc_rows = int(truncated_gold_result_rows)
    except Exception:
        return pred_rows, gold_rows, meta

    if max_rows < 0 or trunc_rows < 0:
        return pred_rows, gold_rows, meta

    if len(gold_rows) <= max_rows:
        return pred_rows, gold_rows, meta

    k = max(0, min(int(trunc_rows), len(gold_rows)))
    meta["truncated"] = True
    meta["prefix_length"] = k

    gold_prefix = gold_rows[:k]
    order_sql = _extract_outer_order_sql(gold_sql)

    raw_pred = ex.get("sql")
    pred_sql = _sanitize_sql_for_execution(
        raw_pred if isinstance(raw_pred, str) else str(raw_pred) if raw_pred else None
    )

    if order_sql and pred_sql and k > 0:
        table_full_ids, _ = _extract_tables_for_execution(ex)
        alias_map = _extract_alias_mapping(ex)
        inner = pred_sql.rstrip().rstrip(";")
        wrapped = f"SELECT * FROM ({inner}) AS _core_t_eval_sub {order_sql} LIMIT {k}"

        try:
            ordered_rows, _dup, reexec_err, _hit = run_execute_with_execution_cache(
                wrapped,
                table_full_ids,
                config,
                row_limit=int(k),
                alias_to_full_id=alias_map,
                timeout_seconds=int(reorder_exec_timeout),
            )
            if reexec_err is None:
                meta["truncation_compare_mode"] = "order_aligned_reexec"
                meta["order_aligned_reexec"] = True
                meta["compare_ordered"] = True
                return ordered_rows, gold_prefix, meta
            meta["reexec_error"] = str(reexec_err)
        except Exception as e:
            meta["reexec_error"] = str(e)

    if not order_sql and not truncate_legacy_if_no_order_by:
        meta["truncation_compare_mode"] = "full_multiset_skipped_truncation"
        meta["note"] = "Gold SQL has no outer ORDER BY; comparing full result multisets without prefix truncation."
        return pred_rows, gold_rows, meta

    meta["truncation_compare_mode"] = "legacy_prefix"
    meta["compare_ordered"] = bool(order_sql)
    return pred_rows[:k], gold_prefix, meta


# Deprecated helper kept for tests/backward compatibility.
def _maybe_truncate_for_compare(
    pred_rows: Any,
    gold_rows: Any,
    *,
    max_gold_result_rows: int,
    truncated_gold_result_rows: int,
) -> Tuple[Any, Any, bool]:
    if not isinstance(pred_rows, list) or not isinstance(gold_rows, list):
        return pred_rows, gold_rows, False

    try:
        max_rows = int(max_gold_result_rows)
        trunc_rows = int(truncated_gold_result_rows)
    except Exception:
        return pred_rows, gold_rows, False

    if max_rows < 0 or trunc_rows < 0:
        return pred_rows, gold_rows, False

    if len(gold_rows) <= max_rows:
        return pred_rows, gold_rows, False

    k = max(0, trunc_rows)
    return pred_rows[:k], gold_rows[:k], True


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate generated SQLs using execution accuracy."
    )

    p.add_argument(
        "--dataset",
        dest="dataset",
        type=str,
        default="bird",
        help="Dataset used for default gold path and default exec-details path.",
    )

    p.add_argument(
        "--database-type",
        dest="dataset",
        type=str,
        default="bird",
        help="Deprecated alias for --dataset.",
    )

    p.add_argument(
        "--llm-model",
        type=str,
        default="openai:gpt-4o-mini",
        help=(
            "LLM tag used for default results run folder naming. "
            "Not used when --custom-data-path, --sql-generation-dir, or --exec-details is set."
        ),
    )

    p.add_argument(
        "--embedding-model",
        type=str,
        default="fireworks:WhereIsAI/UAE-Large-V1",
        help=(
            "Embedding model tag used for default results run folder naming. "
            "Not used when --custom-data-path, --sql-generation-dir, or --exec-details is set."
        ),
    )

    p.add_argument(
        "--custom-data-path",
        type=str,
        default=None,
        help=(
            "Folder with SQL generation/execution artifacts (sqls_exec_details.json, "
            "sqls_details.json, etc.). eval_summary.json is written here. When set, "
            "--llm-model and --embedding-model are not used."
        ),
    )

    p.add_argument(
        "--sql-generation-dir",
        type=str,
        default=None,
        help=(
            "Path to a results_sql_generation* folder containing sqls_exec_details.json. "
            "If provided, this overrides default results path resolution."
        ),
    )

    p.add_argument(
        "--exec-details",
        type=str,
        default=None,
        help=(
            "Path to sqls_exec_details.json from sql_executor.py. "
            "Defaults to sqls_exec_details.json inside --custom-data-path, "
            "--sql-generation-dir, or results/<dataset>/{llm}_{embedding}/results_sql_generation/."
        ),
    )

    p.add_argument(
        "--sql-details",
        type=str,
        default=None,
        help=(
            "Path to sqls_details.json from sql_generator.py (for at_generation table "
            "metrics). Defaults to sqls_details.json beside --exec-details, or in the "
            "resolved results_sql_generation folder."
        ),
    )

    p.add_argument(
        "--gold",
        type=str,
        default=None,
        help=(
            "Path to gold execution results JSON. "
            "Defaults to data/<dataset>/dev_query_results.json."
        ),
    )

    p.add_argument(
        "--values-only",
        action="store_true",
        default=True,
        help=(
            "Compare values only and ignore column names. "
            "Column/value order within each row is preserved. Default: true."
        ),
    )

    p.add_argument(
        "--include-column-names",
        action="store_true",
        default=False,
        help="Compare including column names/keys. Overrides --values-only.",
    )

    p.add_argument(
        "--float-tolerance",
        type=float,
        default=1e-9,
        help="Tolerance for non-bool numeric comparisons. Default: 1e-9.",
    )

    p.add_argument(
        "--start",
        type=int,
        default=0,
        help="Start index. Default: 0.",
    )

    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional number of items to process.",
    )

    p.add_argument(
        "--skip-empty-gold",
        action="store_true",
        default=False,
        help=(
            "If set, skip examples whose gold result is []. "
            "Default is false because empty gold results are valid and should be evaluated."
        ),
    )

    p.add_argument(
        "--max-rows-in-details",
        type=int,
        default=50,
        help="Max rows to store for pred/gold in --out-details. Default: 50. Use -1 for all.",
    )

    p.add_argument(
        "--max-gold-result-rows",
        type=int,
        default=300000,
        help=(
            "If a gold result has more than this many rows, compare on a smaller prefix "
            "only when safe/aligned. Default: 300000. Use -1 to disable."
        ),
    )

    p.add_argument(
        "--truncated-gold-result-rows",
        type=int,
        default=300000,
        help="Prefix length k when truncating huge gold results. Default: 300000. Use -1 to disable.",
    )

    p.add_argument(
        "--dev-metadata",
        type=str,
        default=None,
        help="Path to dev.json with difficulty and sql. Defaults to data/<dataset>/dev.json when present.",
    )

    p.add_argument(
        "--truncate-legacy-if-no-order-by",
        action="store_true",
        default=False,
        help=(
            "If huge gold results have no parseable outer ORDER BY, use legacy pred[:k]/gold[:k] "
            "truncation. This is unsafe when row order is undefined. Default: false."
        ),
    )

    p.add_argument(
        "--reorder-exec-timeout",
        type=int,
        default=60,
        help="SQLite timeout in seconds when re-executing pred SQL with gold ORDER BY. Default: 60.",
    )

    p.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output summary JSON path. Defaults to the SQL generation output folder.",
    )

    p.add_argument(
        "--out-details",
        type=str,
        default=None,
        help="Optional output JSON path for per-question details.",
    )

    p.add_argument(
        "--write-details",
        action="store_true",
        default=False,
        help="If set, write details to the default output folder even when --out-details is not provided.",
    )

    p.add_argument(
        "--indent",
        type=int,
        default=2,
        help="JSON indentation level.",
    )

    return p.parse_args(argv)


def _resolve_sql_eval_run_dir(args: argparse.Namespace, dataset: str) -> Path:
    """
    Resolve the folder used for default exec/sql paths and eval_summary output.

    Priority: --custom-data-path > --sql-generation-dir > parent of --exec-details >
    default results run dir from --llm-model / --embedding-model.
    """
    if args.custom_data_path:
        run_dir = Path(args.custom_data_path).expanduser().resolve()
        if not run_dir.is_dir():
            raise FileNotFoundError(f"--custom-data-path is not a directory: {run_dir}")
        return run_dir

    if args.sql_generation_dir:
        return Path(args.sql_generation_dir).expanduser().resolve()

    if args.exec_details:
        return Path(args.exec_details).expanduser().resolve().parent

    return get_results_run_dir(
        dataset=str(dataset),
        step_dirname="results_sql_generation",
        llm_model=str(args.llm_model),
        embedding_model=str(args.embedding_model),
        project_root=PROJECT_ROOT,
    )


# ---------------------------------------------------------------------------
# Counting and metrics
# ---------------------------------------------------------------------------


def _coerce_index(item: Dict[str, Any], fallback: int) -> int:
    idx = item.get("index", item.get("question_id", fallback))
    try:
        return int(idx)
    except Exception:
        return int(fallback)


def _load_gold_table_counts(path: Path) -> Optional[List[int]]:
    """
    Load dev_gold_tables.json and return per-question gold table counts by index.

    Expected format:
        list[dict] with a `tables` field.
    """
    if not path.exists():
        return None

    try:
        payload = _read_json(path)
        if not isinstance(payload, list):
            return None

        counts: List[int] = []
        for entry in payload:
            if not isinstance(entry, dict):
                counts.append(0)
                continue

            tables = entry.get("tables", []) or []
            counts.append(int(len(tables)) if isinstance(tables, list) else 0)

        return counts

    except Exception:
        return None


def _load_gold_tables(path: Path) -> Optional[List[List[str]]]:
    """
    Load dev_gold_tables.json and return per-question normalized gold table lists.

    Expected format:
        list[dict] with `db_id` and `tables` fields.
    """
    if not path.exists():
        return None

    try:
        payload = _read_json(path)
        if not isinstance(payload, list):
            return None

        out: List[List[str]] = []
        for entry in payload:
            if not isinstance(entry, dict):
                out.append([])
                continue

            tables = entry.get("tables", []) or []
            db_id = str(entry.get("db_id", "") or "").strip().lower()
            if isinstance(tables, list):
                normed = [_normalize_table_id(t, db_id=db_id) for t in tables]
                out.append([t for t in normed if t])
            else:
                out.append([])

        return out

    except Exception:
        return None


def _table_metrics_for_one(
    pred_tables: Sequence[str],
    gold_tables: Sequence[str],
    *,
    db_id: Optional[str],
) -> Dict[str, Any]:
    pred_set = _table_set_from_ids(pred_tables, db_id=db_id)
    gold_set = _table_set_from_ids(gold_tables, db_id=db_id)
    tp = len(pred_set & gold_set)
    precision = float(tp) / float(len(pred_set)) if pred_set else 0.0
    recall = float(tp) / float(len(gold_set)) if gold_set else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    perfect_recall = bool(gold_set.issubset(pred_set)) if gold_set else False

    return {
        "table_tp": tp,
        "table_precision": precision,
        "table_recall": recall,
        "table_f1": f1,
        "table_perfect_recall": perfect_recall,
        "table_exact_set_match": bool(gold_set and pred_set == gold_set),
        "pred_tables_normalized": sorted(pred_set),
        "gold_tables_normalized": sorted(gold_set),
        "matched_tables_normalized": sorted(pred_set & gold_set),
    }


def _init_counts() -> Dict[str, int]:
    return {
        "total": 0,
        "evaluated": 0,
        "executed_ok": 0,
        "executed_fail": 0,
        "has_rows_field": 0,
        "gold_result_list": 0,
        "gold_result_empty": 0,
        "execution_match": 0,
        "exact_match": 0,  # backward-compatible alias for execution_match
        "skipped_no_rows": 0,
        "skipped_invalid_gold": 0,
        "skipped_empty_gold": 0,
        "order_sensitive": 0,
        "truncated_for_compare": 0,
    }


def _init_table_counts() -> Dict[str, int]:
    return {
        "examples_with_gold_tables": 0,
        "examples_with_pred_tables": 0,
        "table_true_positive": 0,
        "table_pred_total": 0,
        "table_gold_total": 0,
        "table_exact_set_match": 0,
        "table_perfect_recall_examples": 0,
    }


def _extract_generator_tables(detail: Optional[Dict[str, Any]]) -> List[str]:
    """Tables provided to the SQL generator (prompt context), not post-hoc SQL tables."""
    if not isinstance(detail, dict):
        return []
    for key in ("selected_tables_used", "selected_tables_input", "resolved_tables"):
        raw = detail.get(key)
        if isinstance(raw, list) and raw:
            return [str(t) for t in raw if str(t)]
    return []


def _load_sql_details_by_index(path: Path) -> Optional[Dict[int, Dict[str, Any]]]:
    if not path.exists():
        return None
    try:
        payload = _read_json(path)
        if not isinstance(payload, list):
            return None
        out: Dict[int, Dict[str, Any]] = {}
        for fallback_i, entry in enumerate(payload):
            if not isinstance(entry, dict):
                continue
            idx = _coerce_index(entry, fallback_i)
            out[idx] = entry
        return out
    except Exception:
        return None


def _count_bucket_targets(
    *,
    counts_by_case: Dict[str, Dict[str, int]],
    counts_by_difficulty: Dict[str, Dict[str, int]],
    gold_table_count: Optional[int],
    diff_label: str,
    gold_has_answer: bool,
    table_perfect_recall_at_execution: bool,
    table_perfect_recall_at_generation: bool,
) -> List[Dict[str, int]]:
    buckets: List[Dict[str, int]] = [
        counts_by_case["all"],
        counts_by_difficulty[diff_label],
    ]

    if gold_table_count == 1:
        buckets.append(counts_by_case["gold_tables_eq_1"])
    elif isinstance(gold_table_count, int) and gold_table_count >= 2:
        buckets.append(counts_by_case["gold_tables_ge_2"])

    if gold_has_answer:
        buckets.append(counts_by_case["hasans"])
    else:
        buckets.append(counts_by_case["noans"])

    if table_perfect_recall_at_execution:
        buckets.append(counts_by_case["table_perfect_recall_at_execution"])

    if table_perfect_recall_at_generation:
        buckets.append(counts_by_case["table_perfect_recall_at_generation"])

    return buckets


def _counts_to_metrics(counts: Dict[str, int]) -> Dict[str, float]:
    total = int(counts.get("total", 0))
    executed_ok = int(counts.get("executed_ok", 0))
    evaluated = int(counts.get("evaluated", 0))
    correct = int(counts.get("execution_match", counts.get("exact_match", 0)))

    return {
        # Main benchmark-style metric.
        "sql_execution_accuracy": float(correct) / float(total) if total else 0.0,

        # Diagnostic metrics.
        "execution_success_rate": float(executed_ok) / float(total) if total else 0.0,
        "result_match_rate_executed_ok": float(correct) / float(executed_ok) if executed_ok else 0.0,
        "result_match_rate_evaluated": float(correct) / float(evaluated) if evaluated else 0.0,

        # Backward-compatible names.
        "exact_match_rate_all": float(correct) / float(total) if total else 0.0,
        "exact_match_rate_executed_ok": float(correct) / float(executed_ok) if executed_ok else 0.0,
        "exact_match_rate_evaluated": float(correct) / float(evaluated) if evaluated else 0.0,
    }


def _table_metrics(table_counts: Dict[str, int]) -> Dict[str, float]:
    tp = int(table_counts.get("table_true_positive", 0))
    pred_total = int(table_counts.get("table_pred_total", 0))
    gold_total = int(table_counts.get("table_gold_total", 0))
    examples_with_gold = int(table_counts.get("examples_with_gold_tables", 0))
    exact = int(table_counts.get("table_exact_set_match", 0))
    pr_examples = int(table_counts.get("table_perfect_recall_examples", 0))

    precision = float(tp) / float(pred_total) if pred_total else 0.0
    recall = float(tp) / float(gold_total) if gold_total else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    return {
        "table_precision_micro": precision,
        "table_recall_micro": recall,
        "table_f1_micro": f1,
        "table_exact_set_match_rate": float(exact) / float(examples_with_gold) if examples_with_gold else 0.0,
        "table_perfect_recall_rate": float(pr_examples) / float(examples_with_gold) if examples_with_gold else 0.0,
        "table_id_normalization": f"gold bare names aligned to pred via db_id + {TABLE_KEY_SEP}",
    }


def _update_table_counts(
    table_counts: Dict[str, int],
    *,
    pred_tables: Sequence[str],
    gold_tables: Sequence[str],
    db_id: Optional[str] = None,
) -> None:
    pred_set = _table_set_from_ids(pred_tables, db_id=db_id)
    gold_set = _table_set_from_ids(gold_tables, db_id=db_id)

    if gold_set:
        table_counts["examples_with_gold_tables"] += 1
    if pred_set:
        table_counts["examples_with_pred_tables"] += 1

    table_counts["table_true_positive"] += len(pred_set & gold_set)
    table_counts["table_pred_total"] += len(pred_set)
    table_counts["table_gold_total"] += len(gold_set)

    if gold_set and pred_set == gold_set:
        table_counts["table_exact_set_match"] += 1

    if gold_set and gold_set.issubset(pred_set):
        table_counts["table_perfect_recall_examples"] += 1  # micro count of examples with PR=1


def _slice_accuracy_block(
    counts: Dict[str, int],
    *,
    label: str,
    note: str,
) -> Dict[str, Any]:
    metrics = _counts_to_metrics(counts)
    return {
        "label": label,
        "counts": counts,
        "metrics": metrics,
        "sql_execution_accuracy": metrics["sql_execution_accuracy"],
        "numerator_execution_matches": int(counts.get("execution_match", 0)),
        "denominator_total_questions": int(counts.get("total", 0)),
        "note": note,
    }


def _pr_slice_ex_block(
    slice_counts: Dict[str, int],
    all_counts: Dict[str, int],
    *,
    primary_metric: str,
    definition: str,
    table_perfect_recall_rate: float,
    pred_table_source: str,
    slice_metrics: Dict[str, float],
    note: str,
) -> Dict[str, Any]:
    """
    Report EX on the perfect-recall (PR=1) subset two ways:
      - conditional: matches / questions in the PR=1 slice
      - over_all_questions: matches / all evaluated questions (non-PR count as incorrect)
    """
    matches = int(slice_counts.get("execution_match", 0))
    slice_total = int(slice_counts.get("total", 0))
    all_total = int(all_counts.get("total", 0))
    acc_conditional = float(matches) / float(slice_total) if slice_total else 0.0
    acc_over_all = float(matches) / float(all_total) if all_total else 0.0

    return {
        "primary_metric": primary_metric,
        "sql_execution_accuracy_conditional_on_perfect_recall": acc_conditional,
        "sql_execution_accuracy_over_all_questions": acc_over_all,
        "value": acc_conditional,
        "value_over_all_questions": acc_over_all,
        "numerator_execution_matches": matches,
        "denominator_perfect_recall_slice": slice_total,
        "denominator_all_questions": all_total,
        "definition": definition,
        "table_perfect_recall_rate": table_perfect_recall_rate,
        "pred_table_source": pred_table_source,
        "counts": slice_counts,
        "metrics": slice_metrics,
        "metrics_over_all_questions": {
            "sql_execution_accuracy": acc_over_all,
            "numerator_execution_matches": matches,
            "denominator_total_questions": all_total,
        },
        "note": note,
    }


def _most_common_error_types(errors: List[str], limit: int = 20) -> List[Dict[str, Any]]:
    def classify(err: str) -> str:
        e = str(err or "").lower()

        if not e:
            return "none"
        if "syntax" in e:
            return "syntax_error"
        if "no such table" in e or "missing table" in e:
            return "missing_table"
        if "no such column" in e or "missing column" in e:
            return "missing_column"
        if "ambiguous column" in e:
            return "ambiguous_column"
        if "timeout" in e or "timed out" in e:
            return "timeout"
        if "permission" in e or "access" in e:
            return "permission_or_access"
        if "database is locked" in e:
            return "database_locked"
        if "incorrect number" in e or "misuse" in e:
            return "sql_runtime_error"

        return "other_error"

    c = Counter(classify(e) for e in errors if e)
    return [{"error_type": k, "count": v} for k, v in c.most_common(limit)]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(args: argparse.Namespace) -> int:
    dataset = str(args.dataset or "bird").strip()

    try:
        gen_dir = _resolve_sql_eval_run_dir(args, dataset)
    except FileNotFoundError as exc:
        print(f"❌ {exc}")
        return 1

    exec_details_path = (
        Path(args.exec_details).expanduser().resolve()
        if args.exec_details
        else (gen_dir / "sqls_exec_details.json").resolve()
    )

    if args.sql_details:
        sql_details_path = Path(args.sql_details).expanduser().resolve()
    else:
        sql_details_path = (exec_details_path.parent / "sqls_details.json").resolve()
        if not sql_details_path.exists():
            sql_details_path = (gen_dir / "sqls_details.json").resolve()

    gold_path = (
        Path(args.gold).expanduser().resolve()
        if args.gold
        else (PROJECT_ROOT / "data" / dataset / "dev_query_results.json").resolve()
    )

    gold_tables_path = (PROJECT_ROOT / "data" / dataset / "dev_gold_tables.json").resolve()

    dev_meta_path = (
        Path(args.dev_metadata).expanduser().resolve()
        if args.dev_metadata
        else (PROJECT_ROOT / "data" / dataset / "dev.json").resolve()
    )

    dev_rows = _load_dev_metadata(dev_meta_path)
    db_config = _config_for_dataset(dataset)

    if not exec_details_path.exists():
        print(f"❌ exec-details file not found: {exec_details_path}")
        return 1

    if not gold_path.exists():
        print(f"❌ gold file not found: {gold_path}")
        return 1

    exec_payload = _read_json(exec_details_path)
    gold_payload = _read_json(gold_path)

    if not isinstance(exec_payload, list):
        print("❌ --exec-details must be a JSON list[dict].")
        return 1

    if not isinstance(gold_payload, list):
        print("❌ --gold must be a JSON list[list[dict]].")
        return 1

    exec_items: List[Dict[str, Any]] = [x for x in exec_payload if isinstance(x, dict)]
    gold_results: List[Any] = list(gold_payload)

    gold_table_counts = _load_gold_table_counts(gold_tables_path)
    gold_tables_by_index = _load_gold_tables(gold_tables_path)

    n = min(len(exec_items), len(gold_results))
    if n == 0:
        print("Nothing to do.")
        return 0

    start = max(0, int(args.start or 0))
    end = n if args.limit is None else min(n, start + int(args.limit))

    if start >= end:
        print(f"Nothing to do: start={start} end={end} available={n}")
        return 0

    values_only = bool(args.values_only)
    if bool(args.include_column_names):
        values_only = False

    float_tol = float(args.float_tolerance)
    skip_empty_gold = bool(args.skip_empty_gold)

    per_question_summary: List[Dict[str, Any]] = []
    per_question_details: List[Dict[str, Any]] = []

    counts_by_case: Dict[str, Dict[str, int]] = {
        "all": _init_counts(),
        "gold_tables_eq_1": _init_counts(),
        "gold_tables_ge_2": _init_counts(),
        "hasans": _init_counts(),
        "noans": _init_counts(),
        "table_perfect_recall_at_execution": _init_counts(),
        "table_perfect_recall_at_generation": _init_counts(),
    }

    counts_by_difficulty: Dict[str, Dict[str, int]] = {
        "simple": _init_counts(),
        "moderate": _init_counts(),
        "challenging": _init_counts(),
        "unknown": _init_counts(),
    }

    table_counts_at_execution = _init_table_counts()
    table_counts_at_generation = _init_table_counts()
    sql_details_by_index = _load_sql_details_by_index(sql_details_path)
    execution_errors: List[str] = []

    print(f"Loaded exec details: {len(exec_items)} | gold: {len(gold_results)} | processing: {end - start}")
    if sql_details_by_index is not None:
        print(f"sql details: {sql_details_path} ({len(sql_details_by_index)} indexed rows)")
    else:
        print(
            f"sql details: not loaded ({sql_details_path} missing or invalid) — "
            "at_generation perfect-recall slice disabled"
        )
    print(f"values_only: {values_only} | float_tolerance: {float_tol}")
    print("primary metric: sql_execution_accuracy = execution_match / total")
    print("failed executions count as incorrect")
    print(f"empty gold results evaluated: {not skip_empty_gold}")

    if dev_rows is not None:
        print(f"dev metadata: {dev_meta_path} ({len(dev_rows)} rows)")
    else:
        print(f"dev metadata: not loaded ({dev_meta_path} missing or invalid) — difficulty → unknown")

    for local_i, i in enumerate(range(start, end), start=1):
        ex = exec_items[i]
        gold = gold_results[i] if i < len(gold_results) else None

        gold_table_count: Optional[int] = None
        try:
            if gold_table_counts is not None and 0 <= i < len(gold_table_counts):
                gold_table_count = int(gold_table_counts[i])
        except Exception:
            gold_table_count = None

        gold_tables: List[str] = []
        if gold_tables_by_index is not None and 0 <= i < len(gold_tables_by_index):
            gold_tables = gold_tables_by_index[i]

        exec_tables_raw = ex.get("tables_used", []) or []
        exec_tables = [str(t) for t in exec_tables_raw] if isinstance(exec_tables_raw, list) else []

        idx = _coerce_index(ex, i)
        qid = ex.get("question_id", idx)
        db_id = ex.get("db_id")
        db_id_str = str(db_id) if db_id is not None else None

        sql_detail = sql_details_by_index.get(idx) if sql_details_by_index is not None else None
        generator_tables = _extract_generator_tables(sql_detail)

        table_metric_at_execution: Dict[str, Any] = {}
        table_metric_at_generation: Dict[str, Any] = {}
        if gold_tables_by_index is not None:
            _update_table_counts(
                table_counts_at_execution,
                pred_tables=exec_tables,
                gold_tables=gold_tables,
                db_id=db_id_str,
            )
            table_metric_at_execution = _table_metrics_for_one(
                exec_tables,
                gold_tables,
                db_id=db_id_str,
            )
            if generator_tables:
                _update_table_counts(
                    table_counts_at_generation,
                    pred_tables=generator_tables,
                    gold_tables=gold_tables,
                    db_id=db_id_str,
                )
                table_metric_at_generation = _table_metrics_for_one(
                    generator_tables,
                    gold_tables,
                    db_id=db_id_str,
                )

        gold_ok_pre = isinstance(gold, list)
        gold_empty_pre = bool(gold_ok_pre and len(gold) == 0)
        gold_has_answer = bool(gold_ok_pre and not gold_empty_pre)
        table_perfect_recall_at_execution = bool(
            table_metric_at_execution.get("table_perfect_recall", False)
        )
        table_perfect_recall_at_generation = bool(
            table_metric_at_generation.get("table_perfect_recall", False)
        )

        diff_label = _difficulty_label_for_index(dev_rows, i)
        bucket_targets = _count_bucket_targets(
            counts_by_case=counts_by_case,
            counts_by_difficulty=counts_by_difficulty,
            gold_table_count=gold_table_count,
            diff_label=diff_label,
            gold_has_answer=gold_has_answer,
            table_perfect_recall_at_execution=table_perfect_recall_at_execution,
            table_perfect_recall_at_generation=table_perfect_recall_at_generation,
        )

        gold_sql: Optional[str] = None
        if dev_rows is not None and 0 <= i < len(dev_rows):
            gs = dev_rows[i].get("sql")
            if isinstance(gs, str):
                gold_sql = gs

        executed_ok = bool(ex.get("executed", False)) and not ex.get("error")
        has_rows = "rows" in ex and isinstance(ex.get("rows"), list)
        pred_rows = ex.get("rows") if has_rows else None

        if ex.get("error"):
            execution_errors.append(str(ex.get("error")))

        gold_ok = isinstance(gold, list)
        gold_empty = bool(gold_ok and len(gold) == 0)
        gold_is_evaluable = bool(gold_ok and not (skip_empty_gold and gold_empty))

        order_sensitive = _gold_sql_is_order_sensitive(gold_sql, gold)

        for cc in bucket_targets:
            cc["total"] += 1

            if has_rows:
                cc["has_rows_field"] += 1
            else:
                cc["skipped_no_rows"] += 1

            if gold_ok:
                cc["gold_result_list"] += 1
                if gold_empty:
                    cc["gold_result_empty"] += 1
            else:
                cc["skipped_invalid_gold"] += 1

            if skip_empty_gold and gold_empty:
                cc["skipped_empty_gold"] += 1

            if executed_ok:
                cc["executed_ok"] += 1
            else:
                cc["executed_fail"] += 1

            if order_sensitive:
                cc["order_sensitive"] += 1

        execution_match = False
        trunc_meta: Dict[str, Any] = {}

        if executed_ok and has_rows and gold_is_evaluable:
            for cc in bucket_targets:
                cc["evaluated"] += 1

            pred_cmp, gold_cmp, trunc_meta = _prepare_rows_for_compare(
                pred_rows,
                gold,
                ex=ex,
                gold_sql=gold_sql,
                max_gold_result_rows=int(args.max_gold_result_rows),
                truncated_gold_result_rows=int(args.truncated_gold_result_rows),
                config=db_config,
                reorder_exec_timeout=int(args.reorder_exec_timeout),
                truncate_legacy_if_no_order_by=bool(args.truncate_legacy_if_no_order_by),
            )

            if trunc_meta.get("truncated"):
                for cc in bucket_targets:
                    cc["truncated_for_compare"] += 1

            effective_order_sensitive = bool(order_sensitive or trunc_meta.get("compare_ordered"))

            execution_match = compare_query_results(
                pred_cmp,
                gold_cmp,
                values_only=values_only,
                float_tolerance=float_tol,
                order_sensitive=effective_order_sensitive,
            )

            if execution_match:
                for cc in bucket_targets:
                    cc["execution_match"] += 1
                    cc["exact_match"] += 1

        gold_truncated = bool(trunc_meta.get("truncated")) if trunc_meta else False

        summary = {
            "question_id": qid,
            "index": idx,
            "db_id": db_id,
            "difficulty": diff_label,
            "gold_table_count": gold_table_count,
            "executed": bool(ex.get("executed", False)),
            "executed_ok": bool(executed_ok),
            "error": ex.get("error"),
            "row_count": int(ex.get("row_count", len(pred_rows or [])) or 0),
            "gold_row_count": int(len(gold) if isinstance(gold, list) else -1),
            "gold_result_valid_list": bool(gold_ok),
            "gold_result_empty": bool(gold_empty),
            "gold_result_evaluated": bool(gold_is_evaluable),
            "evaluated": bool(executed_ok and has_rows and gold_is_evaluable),
            "values_only": bool(values_only),
            "float_tolerance": float_tol,
            "order_sensitive": bool(order_sensitive),
            "execution_match": bool(execution_match),
            "exact_match": bool(execution_match),  # backward-compatible alias
            "gold_truncated_for_compare": gold_truncated if (executed_ok and has_rows and gold_is_evaluable) else False,
            "truncation_compare_mode": trunc_meta.get("truncation_compare_mode") if trunc_meta else None,
            "order_aligned_reexec": bool(trunc_meta.get("order_aligned_reexec")) if trunc_meta else False,
            "truncation_reexec_error": trunc_meta.get("reexec_error") if trunc_meta else None,
            "truncation_note": trunc_meta.get("note") if trunc_meta else None,
            "gold_has_answer": bool(gold_has_answer),
            "table_perfect_recall_at_execution": table_perfect_recall_at_execution,
            "table_perfect_recall_at_generation": table_perfect_recall_at_generation,
            "table_recall_at_execution": table_metric_at_execution.get("table_recall"),
            "table_recall_at_generation": table_metric_at_generation.get("table_recall"),
            "exec_tables_used": exec_tables,
            "generator_tables": generator_tables,
            "gold_tables": gold_tables,
            "exec_tables_normalized": table_metric_at_execution.get("pred_tables_normalized", []),
            "generator_tables_normalized": table_metric_at_generation.get(
                "pred_tables_normalized", []
            ),
            "gold_tables_normalized": table_metric_at_execution.get("gold_tables_normalized", []),
        }

        per_question_summary.append(summary)

        if args.out_details or bool(args.write_details):
            per_question_details.append(
                {
                    **summary,
                    "sql": ex.get("sql"),
                    "gold_sql": gold_sql,
                    "tables_used": ex.get("tables_used", []),
                    "pred_rows": _truncate_rows(pred_rows, int(args.max_rows_in_details)) if has_rows else None,
                    "gold_rows": _truncate_rows(gold, int(args.max_rows_in_details)) if gold_ok else None,
                }
            )

        if local_i == 1 or local_i % 200 == 0:
            all_counts = counts_by_case["all"]
            total_so_far = int(all_counts["total"])
            correct_so_far = int(all_counts["execution_match"])
            acc = float(correct_so_far) / float(total_so_far) if total_so_far else 0.0
            print(
                f"  - processed {local_i}/{end - start} | "
                f"execution_match={correct_so_far} total={total_so_far} "
                f"sql_execution_accuracy={acc:.4f}"
            )

    m_all = _counts_to_metrics(counts_by_case["all"])
    m_hasans = _counts_to_metrics(counts_by_case["hasans"])
    m_noans = _counts_to_metrics(counts_by_case["noans"])
    m_pr_execution = _counts_to_metrics(counts_by_case["table_perfect_recall_at_execution"])
    m_pr_generation = _counts_to_metrics(counts_by_case["table_perfect_recall_at_generation"])
    table_metrics_at_execution = _table_metrics(table_counts_at_execution)
    table_metrics_at_generation = _table_metrics(table_counts_at_generation)

    m_all["sql_execution_accuracy_hasans"] = m_hasans["sql_execution_accuracy"]
    m_all["sql_execution_accuracy_noans"] = m_noans["sql_execution_accuracy"]
    pr_exec_matches = int(counts_by_case["table_perfect_recall_at_execution"]["execution_match"])
    pr_gen_matches = int(counts_by_case["table_perfect_recall_at_generation"]["execution_match"])
    all_total = int(counts_by_case["all"]["total"])

    m_all["sql_execution_accuracy_table_perfect_recall_at_execution"] = m_pr_execution[
        "sql_execution_accuracy"
    ]
    m_all["sql_execution_accuracy_table_perfect_recall_at_generation"] = m_pr_generation[
        "sql_execution_accuracy"
    ]
    m_all["sql_execution_accuracy_table_perfect_recall_at_execution_over_all"] = (
        float(pr_exec_matches) / float(all_total) if all_total else 0.0
    )
    m_all["sql_execution_accuracy_table_perfect_recall_at_generation_over_all"] = (
        float(pr_gen_matches) / float(all_total) if all_total else 0.0
    )

    overall = {
        "inputs": {
            "custom_data_path": (
                str(Path(args.custom_data_path).expanduser().resolve())
                if args.custom_data_path
                else None
            ),
            "exec_details": str(exec_details_path),
            "sql_details": str(sql_details_path) if sql_details_by_index is not None else None,
            "gold": str(gold_path),
            "gold_tables": str(gold_tables_path) if gold_tables_by_index is not None else None,
            "dev_metadata": str(dev_meta_path) if dev_rows is not None else None,
        },
        "settings": {
            "values_only": bool(values_only),
            "include_column_names": bool(args.include_column_names),
            "float_tolerance": float_tol,
            "start": start,
            "end": end,
            "skip_empty_gold": bool(skip_empty_gold),
            "empty_gold_results_evaluated": bool(not skip_empty_gold),
            "max_gold_result_rows": int(args.max_gold_result_rows),
            "truncated_gold_result_rows": int(args.truncated_gold_result_rows),
            "truncate_legacy_if_no_order_by": bool(args.truncate_legacy_if_no_order_by),
            "reorder_exec_timeout": int(args.reorder_exec_timeout),
        },
        "counts": counts_by_case["all"],
        "metrics": m_all,
        "sql_execution_accuracy": {
            "primary_metric": "sql_execution_accuracy",
            "value": m_all["sql_execution_accuracy"],
            "numerator_execution_matches": int(counts_by_case["all"]["execution_match"]),
            "denominator_total_questions": int(counts_by_case["all"]["total"]),
            "note": (
                "Benchmark-style SQL execution accuracy. Failed executions, invalid results, "
                "and non-matching outputs count as incorrect. Empty gold results are evaluated "
                "unless --skip-empty-gold is set."
            ),
        },
        "sql_execution_accuracy_by_answer_type": {
            "hasans": _slice_accuracy_block(
                counts_by_case["hasans"],
                label="HasAns",
                note=(
                    "Gold execution result is non-empty (answerable). Measures SQL quality when "
                    "the question expects rows."
                ),
            ),
            "noans": _slice_accuracy_block(
                counts_by_case["noans"],
                label="NoAns",
                note=(
                    "Gold execution result is empty []. Empty predictions often match by luck; "
                    "compare against HasAns and table_perfect_recall slices."
                ),
            ),
        },
        "sql_execution_accuracy_table_perfect_recall": {
            "at_execution": _pr_slice_ex_block(
                counts_by_case["table_perfect_recall_at_execution"],
                counts_by_case["all"],
                primary_metric="sql_execution_accuracy_table_perfect_recall_at_execution",
                definition=(
                    "All gold tables appear in sqls_exec_details tables_used (tables attached "
                    f"when the predicted SQL was executed; gold ⊆ pred, db{TABLE_KEY_SEP} form). "
                    "May differ from prompt tables if the model omitted tables in the SQL."
                ),
                table_perfect_recall_rate=table_metrics_at_execution.get(
                    "table_perfect_recall_rate", 0.0
                ),
                pred_table_source="sqls_exec_details.tables_used",
                slice_metrics=m_pr_execution,
                note=(
                    "sql_execution_accuracy_conditional_on_perfect_recall: EX among questions "
                    "with PR=1 only. sql_execution_accuracy_over_all_questions: same numerator "
                    "but divided by all questions (non-PR questions count as incorrect)."
                ),
            ),
            "at_generation": _pr_slice_ex_block(
                counts_by_case["table_perfect_recall_at_generation"],
                counts_by_case["all"],
                primary_metric="sql_execution_accuracy_table_perfect_recall_at_generation",
                definition=(
                    "All gold tables appear in sqls_details selected_tables_used (tables passed "
                    f"to the SQL generator prompt; gold ⊆ pred, db{TABLE_KEY_SEP} form). "
                    "Isolates SQL quality when retrieval supplied every gold table."
                ),
                table_perfect_recall_rate=table_metrics_at_generation.get(
                    "table_perfect_recall_rate", 0.0
                ),
                pred_table_source="sqls_details.selected_tables_used",
                slice_metrics=m_pr_generation,
                note=(
                    "Prefer at_generation for isolating SQL performance from retrieval. "
                    "Report both conditional (PR=1 slice) and over_all_questions denominators."
                ),
            ),
            "note": (
                "Prefer at_generation for isolating SQL performance from retrieval; "
                "at_execution reflects tables actually used when running the generated SQL. "
                "Each slice reports EX conditional on PR=1 and EX over all questions."
            ),
        },
        "diagnostics": {
            "execution_success_rate": m_all["execution_success_rate"],
            "result_match_rate_executed_ok": m_all["result_match_rate_executed_ok"],
            "result_match_rate_evaluated": m_all["result_match_rate_evaluated"],
            "error_type_counts": _most_common_error_types(execution_errors),
        },
        "table_retrieval": {
            "at_execution": {
                "counts": table_counts_at_execution,
                "metrics": table_metrics_at_execution,
                "pred_table_source": "sqls_exec_details.tables_used",
            },
            "at_generation": {
                "counts": table_counts_at_generation,
                "metrics": table_metrics_at_generation,
                "pred_table_source": "sqls_details.selected_tables_used",
            },
            "note": (
                "Gold bare names normalized with each example's db_id to "
                f"db{TABLE_KEY_SEP}TABLE. at_generation uses tables given to the SQL "
                "generator; at_execution uses tables attached when SQL ran."
            ),
        },
        "by_gold_table_count": {
            "all": {
                "counts": counts_by_case["all"],
                "metrics": _counts_to_metrics(counts_by_case["all"]),
            },
            "gold_tables_eq_1": {
                "counts": counts_by_case["gold_tables_eq_1"],
                "metrics": _counts_to_metrics(counts_by_case["gold_tables_eq_1"]),
            },
            "gold_tables_ge_2": {
                "counts": counts_by_case["gold_tables_ge_2"],
                "metrics": _counts_to_metrics(counts_by_case["gold_tables_ge_2"]),
            },
        },
        "by_difficulty": {
            "simple": {
                "counts": counts_by_difficulty["simple"],
                "metrics": _counts_to_metrics(counts_by_difficulty["simple"]),
            },
            "moderate": {
                "counts": counts_by_difficulty["moderate"],
                "metrics": _counts_to_metrics(counts_by_difficulty["moderate"]),
            },
            "challenging": {
                "counts": counts_by_difficulty["challenging"],
                "metrics": _counts_to_metrics(counts_by_difficulty["challenging"]),
            },
            "unknown": {
                "counts": counts_by_difficulty["unknown"],
                "metrics": _counts_to_metrics(counts_by_difficulty["unknown"]),
            },
        },
        "by_answer_type": {
            "hasans": {
                "counts": counts_by_case["hasans"],
                "metrics": m_hasans,
            },
            "noans": {
                "counts": counts_by_case["noans"],
                "metrics": m_noans,
            },
        },
        "by_table_perfect_recall": {
            "at_execution": {
                "counts": counts_by_case["table_perfect_recall_at_execution"],
                "metrics": m_pr_execution,
                "metrics_over_all_questions": {
                    "sql_execution_accuracy": m_all[
                        "sql_execution_accuracy_table_perfect_recall_at_execution_over_all"
                    ],
                    "numerator_execution_matches": pr_exec_matches,
                    "denominator_total_questions": all_total,
                },
            },
            "at_generation": {
                "counts": counts_by_case["table_perfect_recall_at_generation"],
                "metrics": m_pr_generation,
                "metrics_over_all_questions": {
                    "sql_execution_accuracy": m_all[
                        "sql_execution_accuracy_table_perfect_recall_at_generation_over_all"
                    ],
                    "numerator_execution_matches": pr_gen_matches,
                    "denominator_total_questions": all_total,
                },
            },
        },
        "per_question": per_question_summary,
    }

    gen_dir.mkdir(parents=True, exist_ok=True)

    out_path = (
        Path(args.out).expanduser().resolve()
        if args.out
        else (gen_dir / "eval_summary.json").resolve()
    )

    _write_json(out_path, overall, indent=int(args.indent))
    print(f"Wrote evaluation summary: {out_path}")

    total_all = int(counts_by_case["all"]["total"])
    correct_all = int(counts_by_case["all"]["execution_match"])

    print(
        f"SQL execution accuracy: {m_all['sql_execution_accuracy']:.4f} "
        f"({correct_all}/{total_all} total)"
    )

    print(
        f"Execution success rate: {m_all['execution_success_rate']:.4f} "
        f"({counts_by_case['all']['executed_ok']}/{total_all} total)"
    )

    evaluated_all = int(counts_by_case["all"]["evaluated"])
    if evaluated_all:
        print(
            f"Result match rate among evaluated: {m_all['result_match_rate_evaluated']:.4f} "
            f"({correct_all}/{evaluated_all} evaluated)"
        )

    hasans_total = int(counts_by_case["hasans"]["total"])
    noans_total = int(counts_by_case["noans"]["total"])
    pr_exec_total = int(counts_by_case["table_perfect_recall_at_execution"]["total"])
    pr_gen_total = int(counts_by_case["table_perfect_recall_at_generation"]["total"])
    if hasans_total:
        print(
            f"EX (HasAns): {m_hasans['sql_execution_accuracy']:.4f} "
            f"({counts_by_case['hasans']['execution_match']}/{hasans_total})"
        )
    if noans_total:
        print(
            f"EX (NoAns): {m_noans['sql_execution_accuracy']:.4f} "
            f"({counts_by_case['noans']['execution_match']}/{noans_total})"
        )
    if pr_gen_total:
        print(
            f"EX (table PR at generation, conditional): {m_pr_generation['sql_execution_accuracy']:.4f} "
            f"({pr_gen_matches}/{pr_gen_total}) "
            f"| over all: {m_all['sql_execution_accuracy_table_perfect_recall_at_generation_over_all']:.4f} "
            f"({pr_gen_matches}/{all_total}) "
            f"| PR rate: {table_metrics_at_generation.get('table_perfect_recall_rate', 0):.4f}"
        )
    if pr_exec_total:
        print(
            f"EX (table PR at execution, conditional): {m_pr_execution['sql_execution_accuracy']:.4f} "
            f"({pr_exec_matches}/{pr_exec_total}) "
            f"| over all: {m_all['sql_execution_accuracy_table_perfect_recall_at_execution_over_all']:.4f} "
            f"({pr_exec_matches}/{all_total}) "
            f"| PR rate: {table_metrics_at_execution.get('table_perfect_recall_rate', 0):.4f}"
        )

    if args.out_details or bool(args.write_details):
        out_details_path = (
            Path(args.out_details).expanduser().resolve()
            if args.out_details
            else (gen_dir / "eval_summary_details.json").resolve()
        )
        _write_json(out_details_path, per_question_details, indent=int(args.indent))
        print(f"Wrote evaluation details: {out_details_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main(_parse_args()))