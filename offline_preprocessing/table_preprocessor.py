#!/usr/bin/env python3
"""
Table preprocessor.

Inputs:
  - `data/{dataset}/dev_tables.json`
  - CLI args for dataset/model knobs (see below).

Outputs:
  - Cached preprocessed tables (markdown) in the cache directory

Usage:
  python offline_preprocessing/table_preprocessor.py --dataset bird --embedding-model "fireworks:WhereIsAI/UAE-Large-V1" --llm-model "huggingface:Qwen/Qwen2.5-7B-Instruct"
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from dotenv import load_dotenv
from tqdm import tqdm


def _project_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "data").exists():
            return parent
    return current.parent


PROJECT_ROOT = _project_root()
# Put the project root on sys.path so `utils.*` resolves as a package.
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv(PROJECT_ROOT / ".env")

from utils import Configuration, DatabaseType
from utils.db_connector import execute_raw_query, get_table_sample, get_database_schema, get_top_column_values
from utils.storage_manager import UnifiedStorageManager, get_unified_storage_manager

TABLE_ID_SEPARATOR = "#sep#"
SAMPLE_ROWS = 5


class TablePreprocessor:
    """Offline preprocessing of database tables with simple caching."""

    def __init__(
        self,
        database_type: DatabaseType,
        *,
        config: Optional[Configuration] = None,
        storage_manager: Optional[UnifiedStorageManager] = None,
    ):
        self.database_type = database_type
        self.config = config or Configuration(database_type=database_type)
        # New cache layout is handled by UnifiedStorageManager when cache_dir="cache".
        self.preprocessed_cache = storage_manager or UnifiedStorageManager(config=self.config, cache_dir="cache")

    def preprocess_single_table(self, table_identifier: str) -> Dict[str, Any]:
        parsed = _parse_table_identifier(table_identifier)
        if parsed is None:
            return _error_result(table_identifier, "Invalid table identifier format")

        db_id, requested_table = parsed

        # Fast-path: try cache under the requested name first (we store canonical lowercased keys).
        cached = self.preprocessed_cache.get_cached_preprocessed_table(db_id, requested_table)
        if cached and cached.get("markdown_content"):
            return {
                "table_identifier": table_identifier,
                "db_id": db_id,
                "table_name": requested_table,
                "status": "cached",
                "preprocessed_data": cached,
            }

        preprocessed = _preprocess_table(db_id, requested_table, self.database_type)
        if preprocessed.get("error"):
            return _error_result(table_identifier, preprocessed["error"], db_id=db_id, table_name=requested_table)

        markdown = preprocessed.get("markdown_content", "")
        if not markdown:
            return _error_result(
                table_identifier,
                "Preprocessing returned empty markdown_content",
                db_id=db_id,
                table_name=requested_table,
            )

        # Cache under the resolved table name (already lowercased by _preprocess_table),
        # to avoid duplicates caused by case differences.
        resolved_table = preprocessed.get("table_name") or requested_table
        self.preprocessed_cache.cache_preprocessed_table(db_id, resolved_table, markdown)

        return {
            "table_identifier": table_identifier,
            "db_id": db_id,
            "table_name": resolved_table,
            "status": "success",
            "preprocessed_data": preprocessed,
        }

    def process_tables_batch(self, table_identifiers: List[str]) -> Dict[str, Any]:
        preprocessed: List[Dict[str, Any]] = []
        errors: List[Dict[str, Any]] = []
        cached = 0
        success = 0

        for table_id in tqdm(table_identifiers, desc="Preprocessing tables", total=len(table_identifiers)):
            result = self.preprocess_single_table(table_id)
            status = result.get("status", "error")
            if status == "success":
                preprocessed.append(result["preprocessed_data"])
                success += 1
            elif status == "cached":
                preprocessed.append(result["preprocessed_data"])
                cached += 1
            else:
                errors.append(result)

        return {"preprocessed_tables": preprocessed, "errors": errors, "success": success, "cached": cached}

    def process_all_tables(self, table_identifiers: List[str], batch_size: int) -> Dict[str, Any]:
        start_time = time.time()
        total = len(table_identifiers)
        success = 0
        cached = 0
        all_errors: List[Dict[str, Any]] = []

        total_batches = (len(table_identifiers) + batch_size - 1) // batch_size
        for batch_idx in range(total_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, len(table_identifiers))
            batch_identifiers = table_identifiers[start_idx:end_idx]

            print(f"\nProcessing batch {batch_idx + 1}/{total_batches} (tables {start_idx + 1}-{end_idx})")
            batch_results = self.process_tables_batch(batch_identifiers)

            success += batch_results["success"]
            cached += batch_results["cached"]
            all_errors.extend(batch_results["errors"])

        total_processing_time = time.time() - start_time
        return {
            "total_tables": total,
            "successful_preprocessed": success,
            "cached_preprocessed": cached,
            "error_count": len(all_errors),
            "errors_sample": all_errors[:10],
            "processing_time_seconds": total_processing_time,
            "cache_directory": str(self.preprocessed_cache.cache_dir),
        }

    def cache_column_value_hints_for_tables(self, table_identifiers: List[str]) -> Dict[str, Any]:
        start_time = time.time()
        try:
            hints_map = _ensure_column_value_hints_cached(table_identifiers, self.database_type, self.preprocessed_cache)
            total_tables = len(hints_map)
            total_columns = sum(len(cols) for cols in hints_map.values())
            elapsed = time.time() - start_time
            print(f"Cached column value hints for {total_tables} tables, {total_columns} columns in {elapsed:.2f}s")
            return {"tables": total_tables, "columns": total_columns, "elapsed_seconds": elapsed}
        except Exception as e:
            return {"tables": 0, "columns": 0, "elapsed_seconds": time.time() - start_time, "error": str(e)}

    @staticmethod
    def print_preprocessing_summary(results: Dict[str, Any]) -> None:
        print("\n" + "=" * 60)
        print("OFFLINE TABLE PREPROCESSING SUMMARY")
        print("=" * 60)
        print(f"Total Tables: {results['total_tables']}")
        print(f"Successfully Preprocessed: {results['successful_preprocessed']}")
        print(f"Cached (Already Preprocessed): {results['cached_preprocessed']}")
        print(f"Errors: {results.get('error_count', 0)}")
        print(f"Total Processing Time: {results['processing_time_seconds']:.2f} seconds")
        if results["total_tables"] > 0:
            success_rate = (results["successful_preprocessed"] + results["cached_preprocessed"]) / results["total_tables"] * 100
            print(f"Success Rate: {success_rate:.1f}%")
        errors_sample = results.get("errors_sample") or []
        if errors_sample:
            print(f"\nFirst {len(errors_sample)} Errors:")
            for i, error in enumerate(errors_sample):
                print(f"  {i + 1}. {error.get('table_identifier', 'unknown')}: {error.get('error', 'unknown error')}")
        print("\nCache Directory:", results["cache_directory"])


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess tables and cache markdown representations.")
    parser.add_argument(
        "--dataset",
        dest="dataset",
        choices=[dt.value for dt in DatabaseType],
        default=DatabaseType.BIRD.value,
        help="Dataset to operate on.",
    )
    # Backward-compatible alias.
    parser.add_argument(
        "--database-type",
        dest="dataset",
        choices=[dt.value for dt in DatabaseType],
        default=DatabaseType.BIRD.value,
        help="(deprecated) Same as --dataset.",
    )
    parser.add_argument("--llm-model", type=str, default="openai:gpt-4o-mini", help="LLM tag used for cache run folder naming.")
    parser.add_argument(
        "--embedding-model",
        type=str,
        default="fireworks:WhereIsAI/UAE-Large-V1",
        help="Embedding model tag used for cache run folder naming.",
    )
    parser.add_argument("--batch-size", type=int, default=50, help="Number of tables per batch.")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of tables to process (for quick tests).")
    parser.add_argument(
        "--cache-column-hints",
        action="store_true",
        default=True,
        help="Cache top-5 frequent values per column after preprocessing.",
    )
    parser.add_argument(
        "--tables-file",
        type=str,
        default=None,
        help="Optional path to a file listing table identifiers (one per line). If omitted, uses dev_tables.json.",
    )
    return parser.parse_args(argv)


def _normalize_table_name(name: Any) -> str:
    """Best-effort normalization used for identifier matching and caching."""
    s = str(name).strip()
    if (s.startswith("`") and s.endswith("`")) or (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
        s = s[1:-1].strip()
    return s.lower()


def _read_table_identifiers_file(path: Path) -> List[str]:
    with path.open("r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def _dev_tables_requested_by_db(payload: Any) -> Dict[str, Set[str]]:
    """Return mapping: db_id -> {requested_table_name_lower, ...}."""
    requested_by_db: Dict[str, Set[str]] = {}

    # BIRD-style payload (dict keyed by "db#sep#table").
    if isinstance(payload, dict):
        for key, item in payload.items():
            if isinstance(key, str) and TABLE_ID_SEPARATOR in key:
                db_id, table_name = key.split(TABLE_ID_SEPARATOR, 1)
                requested_by_db.setdefault(db_id, set()).add(_normalize_table_name(table_name))
                continue

            # Fallback: attempt to read fields if the key format is different.
            if isinstance(item, dict):
                db_id = item.get("db_id") or item.get("dbId")
                table_name = item.get("table_name_original") or item.get("table_name") or item.get("table")
                if db_id and table_name:
                    requested_by_db.setdefault(str(db_id), set()).add(_normalize_table_name(table_name))
        return requested_by_db

    # Some datasets/tools may store a list[dict] instead.
    if isinstance(payload, list):
        for item in payload:
            if not isinstance(item, dict):
                continue
            db_id = item.get("db_id") or item.get("dbId")
            table_name = item.get("table_name_original") or item.get("table_name") or item.get("table")
            if db_id and table_name:
                requested_by_db.setdefault(str(db_id), set()).add(_normalize_table_name(table_name))

    return requested_by_db


def _filter_to_existing_tables(
    requested_by_db: Dict[str, Set[str]], *, database_type: DatabaseType
) -> Tuple[List[str], int]:
    """Drop tables that don't exist on disk; return (identifiers, skipped_count)."""
    config = Configuration(database_type=database_type)
    identifiers: List[str] = []
    skipped = 0

    for db_id, requested_tables in requested_by_db.items():
        try:
            schema = get_database_schema(db_id, config)
        except Exception:
            skipped += len(requested_tables)
            continue

        available_lower = {str(t).lower() for t in schema.keys()}
        for table_lower in requested_tables:
            if table_lower in available_lower:
                identifiers.append(f"{db_id}{TABLE_ID_SEPARATOR}{table_lower}")
            else:
                skipped += 1

    return identifiers, skipped


def _load_table_identifiers(database_type: DatabaseType, tables_file: Optional[str]) -> List[str]:
    if tables_file:
        return _read_table_identifiers_file(Path(tables_file).expanduser().resolve())

    dev_tables_path = PROJECT_ROOT / "data" / database_type.value / "dev_tables.json"
    if not dev_tables_path.exists():
        print(f"Tables file not found: {dev_tables_path}")
        return []

    try:
        import json

        with dev_tables_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception as e:
        print(f"Failed to read {dev_tables_path}: {e}")
        return []

    requested_by_db = _dev_tables_requested_by_db(payload)
    identifiers, skipped = _filter_to_existing_tables(requested_by_db, database_type=database_type)
    if skipped:
        print(f"Note: skipped {skipped} tables not found on disk schema (dev_tables.json mismatch).")
    return identifiers


def main(args: argparse.Namespace) -> int:
    database_type = DatabaseType(args.dataset)

    table_ids = _load_table_identifiers(database_type, args.tables_file)
    if args.limit is not None:
        table_ids = table_ids[: args.limit]

    if not table_ids:
        print("No table identifiers found. Exiting.")
        return 1

    config = Configuration(
        database_type=database_type,
        llm_model=str(args.llm_model),
        embedding_model=str(args.embedding_model),
    )
    preprocessor = TablePreprocessor(database_type=database_type, config=config)
    results = preprocessor.process_all_tables(table_ids, batch_size=args.batch_size)

    if args.cache_column_hints:
        preprocessor.cache_column_value_hints_for_tables(table_ids)

    preprocessor.print_preprocessing_summary(results)
    return 0


def _parse_table_identifier(table_entry: str) -> Optional[Tuple[str, str]]:
    if TABLE_ID_SEPARATOR in table_entry:
        db_id, table_name = table_entry.split(TABLE_ID_SEPARATOR, 1)
    else:
        parts = table_entry.split("#")
        if len(parts) < 2:
            return None
        db_id, table_name = parts[0], "#".join(parts[1:])
    return db_id, table_name.lower()


def _build_markdown(table_name: str, schema: Dict[str, Any], sample_data: List[Dict[str, Any]]) -> str:
    if not schema:
        return f"Table name: {table_name}\nExample table content:"

    column_names = list(schema.keys())
    header = "| " + " | ".join(column_names) + " |"
    separator = "|" + "|".join(["-------:" for _ in column_names]) + "|"

    lines = [f"Table name: {table_name}", "Example table content:", header, separator]
    if not sample_data:
        lines.append("| " + " | ".join(["" for _ in column_names]) + " |")
        return "\n".join(lines)

    for row in sample_data:
        row_values = ["" if row.get(col) is None else str(row.get(col, "")) for col in column_names]
        lines.append("| " + " | ".join(row_values) + " |")
    return "\n".join(lines)

def _get_actual_table_name(schema: Dict[str, Any], table_name: str) -> Optional[str]:
    """Case-insensitive lookup for the schema's actual table name."""
    target = str(table_name).lower()
    for actual_name in schema.keys():
        if str(actual_name).lower() == target:
            return actual_name
    return None

def resolve_table_name(db_name: str, table_name: str, database_type: DatabaseType) -> str:
    """Resolve `table_name` to the database's actual table name (case-insensitive)."""
    config = Configuration(database_type=database_type)
    schema = get_database_schema(db_name, config)
    actual = _get_actual_table_name(schema, table_name)
    if actual is not None:
        return actual
    available_tables = list(schema.keys())
    raise ValueError(
        f"Table '{table_name}' not found in database '{db_name}'. Available tables: {available_tables}"
    )

def _error_result(
    table_identifier: str, error: str, *, db_id: Optional[str] = None, table_name: Optional[str] = None
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {"table_identifier": table_identifier, "status": "error", "error": error}
    if db_id is not None:
        payload["db_id"] = db_id
    if table_name is not None:
        payload["table_name"] = table_name
    return payload

def _preprocess_table(db_id: str, table_name: str, database_type: DatabaseType) -> Dict[str, Any]:
    config = Configuration(database_type=database_type)
    try:
        schema = get_database_schema(db_id, config)
        actual_table_name = _get_actual_table_name(schema, table_name)
        if actual_table_name is None:
            available = list(schema.keys())
            raise ValueError(f"Table '{table_name}' not found in database '{db_id}'. Available tables: {available}")
        table_schema = schema[actual_table_name]
    except Exception as e:
        return {"markdown_content": "", "db_id": db_id, "table_name": table_name, "error": str(e)}

    try:
        count_query = f"SELECT COUNT(*) AS cnt FROM `{actual_table_name}`"
        count_result = execute_raw_query(db_id, count_query, config)
        total_rows = count_result[0].get("cnt", 0) if count_result else 0
    except Exception:
        total_rows = 0

    sample_size = SAMPLE_ROWS if total_rows >= SAMPLE_ROWS else total_rows
    try:
        sample_data = (
            get_table_sample(db_id, actual_table_name, limit=sample_size, config=config) if sample_size > 0 else []
        )
    except Exception:
        sample_data = []

    markdown_content = _build_markdown(actual_table_name, table_schema, sample_data)
    return {
        "markdown_content": markdown_content,
        "db_id": db_id,
        # Canonical cache keying uses lowercased table_name.
        "table_name": _normalize_table_name(actual_table_name),
        "requested_table_name": _normalize_table_name(table_name),
    }


def _ensure_column_value_hints_cached(
    table_identifiers: List[str], database_type: DatabaseType, storage_manager: UnifiedStorageManager
) -> Dict[str, Dict[str, List[Any]]]:
    hints: Dict[str, Dict[str, List[Any]]] = {}
    config = Configuration(database_type=database_type)
    for table_entry in table_identifiers:
        parsed = _parse_table_identifier(table_entry)
        if parsed is None:
            continue
        db_id, requested_table_name = parsed
        try:
            schema = get_database_schema(db_id, config)
            actual_table_name = _get_actual_table_name(schema, requested_table_name)
            if actual_table_name is None:
                continue
            table_schema = schema[actual_table_name]
            cached = storage_manager.get_cached_preprocessed_table(db_id, actual_table_name) or {}
            markdown_content = cached.get("markdown_content", "")
            metadata = cached.get("metadata") if isinstance(cached.get("metadata"), dict) else {}
            column_hints = metadata.get("column_value_hints") if isinstance(metadata.get("column_value_hints"), dict) else {}

            if not column_hints:
                column_hints = {}
                for column_name in table_schema.keys():
                    try:
                        values = get_top_column_values(db_id, actual_table_name, column_name, limit=5, config=config)
                    except Exception:
                        continue
                    if values:
                        column_hints[column_name] = values

                metadata["column_value_hints"] = column_hints
                storage_manager.cache_preprocessed_table(db_id, actual_table_name, markdown_content, metadata)
            table_key = f"{db_id}{TABLE_ID_SEPARATOR}{actual_table_name}"
            hints[table_key] = column_hints
        except Exception:
            continue
    return hints


if __name__ == "__main__":
    args = _parse_args()
    sys.exit(main(args))