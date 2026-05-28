#!/usr/bin/env python3
"""
Metadata generator.

Inputs:
  - `data/{dataset}/dev_tables.json` (or `--tables-file`)
  - Cached preprocessed tables under `cache/{dataset}/preprocessed_tables/`
  - CLI args for dataset/model knobs (see below).

Outputs:
  - Cached metadata under `cache/{dataset}/metadata/`
  - Per-run LLM usage JSON: `cache/<dataset>/{llm}_{embedding}/metadata/metadata_llm_usage.json`

Usage:
  python offline_preprocessing/metadata_generator.py --dataset bird --embedding-model "fireworks:WhereIsAI/UAE-Large-V1" --llm-model "huggingface:Qwen/Qwen2.5-7B-Instruct"
"""

import argparse
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from dotenv import load_dotenv


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

from utils import Configuration, DatabaseType, llm_usage_from_prompt_and_output
from utils.db_connector import get_database_schema
from utils.cache_scripts.metadata_cache import get_metadata_cache
from utils.prompts import purpose_prompt, qa_prompt, summary_prompt
from utils.storage_manager import UnifiedStorageManager, get_unified_storage_manager


TABLE_ID_SEPARATOR = "#sep#"

METADATA_LLM_USAGE_FILENAME = "metadata_llm_usage.json"


def _aggregate_llm_usage_records(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    total_in = 0.0
    total_out = 0.0
    total_all = 0.0
    total_cost = 0.0
    for rec in records or []:
        u = rec.get("llm_usage") if isinstance(rec.get("llm_usage"), dict) else {}
        try:
            total_in += float(u.get("input_tokens") or 0)
        except Exception:
            pass
        try:
            total_out += float(u.get("output_tokens") or 0)
        except Exception:
            pass
        try:
            total_all += float(u.get("total_tokens") or 0)
        except Exception:
            pass
        try:
            total_cost += float(u.get("cost_usd") or 0)
        except Exception:
            pass
    n = len(records or [])
    return {
        "n_llm_calls": n,
        "input_tokens": total_in,
        "output_tokens": total_out,
        "total_tokens": total_all,
        "cost_usd": total_cost,
        "averages": {
            "input_tokens": total_in / n if n else 0.0,
            "output_tokens": total_out / n if n else 0.0,
            "total_tokens": total_all / n if n else 0.0,
            "cost_usd": total_cost / n if n else 0.0,
        },
    }


def _content_to_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    return str(getattr(content, "content", content))


def _strip_thinking(text: str) -> str:
    return re.sub(r"<think>[\s\S]*?</think>", "", str(text), flags=re.IGNORECASE).strip()


def _parse_json_list(text: Any) -> list:
    cleaned = _strip_thinking(_content_to_text(text)).strip()
    if not cleaned:
        return []
    try:
        val = json.loads(cleaned)
        return val if isinstance(val, list) else []
    except Exception:
        pass
    start = cleaned.find("[")
    end = cleaned.rfind("]")
    if start == -1 or end == -1 or end <= start:
        return []
    try:
        val = json.loads(cleaned[start : end + 1])
        return val if isinstance(val, list) else []
    except Exception:
        return []


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


class MetadataGenerator:
    """Generate (and cache) metadata for preprocessed tables."""

    def __init__(
        self,
        *,
        database_type: DatabaseType,
        llm_model: Optional[str] = None,
        embedding_model: Optional[str] = None,
        storage_manager: Optional[UnifiedStorageManager] = None,
        generate_purpose: bool = True,
        generate_summary: bool = True,
        generate_qa: bool = True,
    ) -> None:
        self.database_type = database_type
        self.config = Configuration(
            database_type=database_type,
            llm_model=str(llm_model) if llm_model else "openai:gpt-4o-mini",
            embedding_model=str(embedding_model) if embedding_model else "fireworks:WhereIsAI/UAE-Large-V1",
        )
        # New cache layout is handled by UnifiedStorageManager when cache_dir="cache".
        self.storage = storage_manager or UnifiedStorageManager(config=self.config, cache_dir="cache")
        self.cache = get_metadata_cache(self.config.get_database_cache_dir("metadata"))
        self.generate_purpose = bool(generate_purpose)
        self.generate_summary = bool(generate_summary)
        self.generate_qa = bool(generate_qa)
        self._model = None
        self._llm_usage_records: List[Dict[str, Any]] = []

    def _get_model(self):
        if self._model is None:
            self._model = self.config.create_llm()
        return self._model

    def _load_preprocessed_tables(self, table_identifiers: List[str]) -> Tuple[List[Dict[str, Any]], int]:
        loaded: List[Dict[str, Any]] = []
        missing = 0

        for identifier in table_identifiers:
            if TABLE_ID_SEPARATOR not in identifier:
                missing += 1
                continue

            db_id, table_name = identifier.split(TABLE_ID_SEPARATOR, 1)
            cached = self.storage.get_cached_preprocessed_table(db_id, table_name)
            if not cached or not cached.get("markdown_content"):
                missing += 1
                continue

            loaded.append(
                {
                    "markdown_content": cached.get("markdown_content", ""),
                    "db_id": cached.get("db_id", db_id),
                    "table_name": cached.get("table_name", table_name),
                }
            )

        return loaded, missing

    def _enrich_preprocessed_tables(
        self, preprocessed_tables: List[Dict[str, Any]]
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        model = None
        enriched_tables: List[Dict[str, Any]] = []
        usage_records: List[Dict[str, Any]] = []

        for pre in preprocessed_tables:
            table_markdown = pre.get("markdown_content", "")
            db_id = pre.get("db_id", "")
            table_name = str(pre.get("table_name", "")).lower()
            if not table_markdown or not db_id or not table_name:
                continue

            cached_enriched = self.cache.get_enriched_table(table_markdown, db_id=db_id, table_name=table_name)
            if isinstance(cached_enriched, dict):
                cached_enriched["db_id"] = db_id
                cached_enriched["table_name"] = table_name
                enriched_tables.append(cached_enriched)
                continue

            purpose = self.cache.get_metadata(table_markdown, "purpose", db_id=db_id, table_name=table_name)
            if self.generate_purpose and purpose is None:
                if model is None:
                    model = self._get_model()
                prompt_f = self.config.format_prompt(purpose_prompt.format(table=table_markdown))
                resp = model.invoke(prompt_f)
                raw_out = _content_to_text(resp)
                usage_records.append(
                    {
                        "db_id": db_id,
                        "table_name": table_name,
                        "field": "purpose",
                        "llm_model": self.config.llm_model,
                        "llm_usage": llm_usage_from_prompt_and_output(
                            model, prompt_f, raw_out, ai_message=resp, cost_usd=0.0
                        ),
                    }
                )
                purpose_text = _strip_thinking(raw_out).strip()
                if purpose_text.lower() == "none":
                    purpose_text = None
                self.cache.set_metadata(table_markdown, "purpose", purpose_text, db_id=db_id, table_name=table_name)
                purpose = purpose_text

            summary = self.cache.get_metadata(table_markdown, "summary", db_id=db_id, table_name=table_name)
            if self.generate_summary and summary is None:
                if model is None:
                    model = self._get_model()
                prompt_f = self.config.format_prompt(summary_prompt.format(table=table_markdown))
                resp = model.invoke(prompt_f)
                raw_out = _content_to_text(resp)
                usage_records.append(
                    {
                        "db_id": db_id,
                        "table_name": table_name,
                        "field": "summary",
                        "llm_model": self.config.llm_model,
                        "llm_usage": llm_usage_from_prompt_and_output(
                            model, prompt_f, raw_out, ai_message=resp, cost_usd=0.0
                        ),
                    }
                )
                summary_text = _strip_thinking(raw_out).strip()
                if summary_text.lower() == "none":
                    summary_text = None
                self.cache.set_metadata(table_markdown, "summary", summary_text, db_id=db_id, table_name=table_name)
                summary = summary_text

            qa_pairs = self.cache.get_metadata(table_markdown, "qa_pairs", db_id=db_id, table_name=table_name)
            if self.generate_qa and qa_pairs is None:
                if model is None:
                    model = self._get_model()
                prompt_f = self.config.format_prompt(qa_prompt.format(table=table_markdown))
                resp = model.invoke(prompt_f)
                raw_out = _content_to_text(resp)
                usage_records.append(
                    {
                        "db_id": db_id,
                        "table_name": table_name,
                        "field": "qa_pairs",
                        "llm_model": self.config.llm_model,
                        "llm_usage": llm_usage_from_prompt_and_output(
                            model, prompt_f, raw_out, ai_message=resp, cost_usd=0.0
                        ),
                    }
                )
                qa_pairs = _parse_json_list(resp)
                self.cache.set_metadata(table_markdown, "qa_pairs", qa_pairs, db_id=db_id, table_name=table_name)
            if not self.generate_qa and qa_pairs is None:
                qa_pairs = []

            enriched = {
                "original_table": table_markdown,
                "db_id": db_id,
                "table_name": table_name,
                "purpose": purpose,
                "summary": summary,
                "qa_pairs": qa_pairs,
            }
            self.cache.set_enriched_table(table_markdown, enriched, db_id=db_id, table_name=table_name)
            enriched_tables.append(enriched)

        return enriched_tables, usage_records

    def _persist_metadata_llm_usage(self) -> Optional[Path]:
        """Write per-call LLM usage for this run to the metadata cache directory."""
        meta_dir = Path(self.config.get_database_cache_dir("metadata"))
        meta_dir.mkdir(parents=True, exist_ok=True)
        out_path = meta_dir / METADATA_LLM_USAGE_FILENAME
        payload: Dict[str, Any] = {
            "llm_model": self.config.llm_model,
            "embedding_model": self.config.embedding_model,
            "database_type": self.database_type.value,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "calls": list(self._llm_usage_records),
            "totals": _aggregate_llm_usage_records(self._llm_usage_records),
        }
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        return out_path.resolve()

    def enrich_batch(self, table_identifiers: List[str], *, dry_run: bool = False) -> Dict[str, Any]:
        start = time.time()
        preprocessed, missing = self._load_preprocessed_tables(table_identifiers)

        if dry_run:
            return {
                "total": len(table_identifiers),
                "loaded_preprocessed": len(preprocessed),
                "missing_preprocessed": missing,
                "enriched": 0,
                "errors": [],
                "elapsed_seconds": time.time() - start,
            }

        if not preprocessed:
            return {
                "total": len(table_identifiers),
                "loaded_preprocessed": 0,
                "missing_preprocessed": missing,
                "enriched": 0,
                "errors": [{"error": "No preprocessed tables were found in cache for this batch."}],
                "elapsed_seconds": time.time() - start,
            }

        try:
            enriched, batch_usage = self._enrich_preprocessed_tables(preprocessed)
            self._llm_usage_records.extend(batch_usage)
            return {
                "total": len(table_identifiers),
                "loaded_preprocessed": len(preprocessed),
                "missing_preprocessed": missing,
                "enriched": len(enriched),
                "errors": [],
                "elapsed_seconds": time.time() - start,
            }
        except Exception as e:
            return {
                "total": len(table_identifiers),
                "loaded_preprocessed": len(preprocessed),
                "missing_preprocessed": missing,
                "enriched": 0,
                "errors": [{"error": str(e)}],
                "elapsed_seconds": time.time() - start,
            }

    def enrich_all(self, table_identifiers: List[str], *, batch_size: int, dry_run: bool = False) -> Dict[str, Any]:
        started = time.time()
        self._llm_usage_records = []

        total = len(table_identifiers)
        enriched_total = 0
        loaded_total = 0
        missing_total = 0
        errors: List[Dict[str, Any]] = []

        total_batches = (total + batch_size - 1) // batch_size
        for batch_idx in range(total_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, total)
            batch_ids = table_identifiers[start_idx:end_idx]

            print(f"\nProcessing batch {batch_idx + 1}/{total_batches} (tables {start_idx + 1}-{end_idx})")
            batch = self.enrich_batch(batch_ids, dry_run=dry_run)

            enriched_total += batch["enriched"]
            loaded_total += batch["loaded_preprocessed"]
            missing_total += batch["missing_preprocessed"]
            errors.extend(batch["errors"])

        metadata_llm_usage_path: Optional[str] = None
        if not dry_run:
            try:
                metadata_llm_usage_path = str(self._persist_metadata_llm_usage())
            except Exception as e:
                errors.append({"error": f"Failed to write {METADATA_LLM_USAGE_FILENAME}: {e}"})

        return {
            "total_tables": total,
            "loaded_preprocessed": loaded_total,
            "missing_preprocessed": missing_total,
            "enriched": enriched_total,
            "errors": errors,
            "elapsed_seconds": time.time() - started,
            "metadata_cache_directory": str(self.config.get_database_cache_dir("metadata")),
            "metadata_llm_usage_path": metadata_llm_usage_path,
            "metadata_llm_usage_totals": _aggregate_llm_usage_records(self._llm_usage_records),
        }


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate and cache metadata for preprocessed tables.")
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
    parser.add_argument("--batch-size", type=int, default=20, help="Number of tables per batch.")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of tables to process (for quick tests).")
    parser.add_argument(
        "--tables-file",
        type=str,
        default=None,
        help="Optional path to a file listing table identifiers (one per line). If omitted, uses dev_tables.json.",
    )
    parser.add_argument("--llm-model", type=str, default=None, help="Override the LLM model used for metadata generation.")
    parser.add_argument(
        "--embedding-model",
        type=str,
        default="fireworks:WhereIsAI/UAE-Large-V1",
        help="Embedding model tag used for cache run folder naming.",
    )
    parser.add_argument("--dry-run", action="store_true", default=False, help="Load inputs only; do not call the LLM.")
    parser.add_argument(
        "--purpose",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable/disable PURPOSE generation (default: enabled).",
    )
    parser.add_argument(
        "--summary",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable/disable SUMMARY generation (default: enabled).",
    )
    parser.add_argument(
        "--qa",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable/disable QA generation (default: enabled).",
    )
    return parser.parse_args(argv)


def _print_summary(results: Dict[str, Any]) -> None:
    print("\n" + "=" * 60)
    print("OFFLINE METADATA GENERATION SUMMARY")
    print("=" * 60)
    print(f"Total Tables: {results['total_tables']}")
    print(f"Loaded Preprocessed: {results['loaded_preprocessed']}")
    print(f"Missing Preprocessed: {results['missing_preprocessed']}")
    print(f"Enriched: {results['enriched']}")
    print(f"Errors: {len(results['errors'])}")
    print(f"Total Processing Time: {results['elapsed_seconds']:.2f} seconds")
    print(f"Metadata Cache Directory: {results['metadata_cache_directory']}")
    if results.get("metadata_llm_usage_path"):
        print(f"LLM usage log: {results['metadata_llm_usage_path']}")
    totals = results.get("metadata_llm_usage_totals")
    if isinstance(totals, dict) and totals.get("n_llm_calls", 0):
        print(
            f"LLM calls (metadata): n={totals.get('n_llm_calls')} "
            f"tokens in/out/total={totals.get('input_tokens', 0):.0f}/"
            f"{totals.get('output_tokens', 0):.0f}/{totals.get('total_tokens', 0):.0f}"
        )

    cache_stats = results.get("metadata_cache_stats")
    if isinstance(cache_stats, dict):
        print("\nMetadata Cache Stats:")
        for k in ("purpose", "summary", "qa_pairs", "enriched_tables", "total"):
            if k in cache_stats:
                print(f"  - {k}: {cache_stats[k]}")

    if results["errors"]:
        sample = results["errors"][:10]
        print(f"\nFirst {len(sample)} Errors:")
        for i, err in enumerate(sample):
            print(f"  {i + 1}. {err.get('error', 'unknown error')}")


def main(args: argparse.Namespace) -> int:
    database_type = DatabaseType(args.dataset)
    table_ids = _load_table_identifiers(database_type, args.tables_file)
    if args.limit is not None:
        table_ids = table_ids[: args.limit]

    if not table_ids:
        print("No table identifiers found. Exiting.")
        return 1

    generator = MetadataGenerator(
        database_type=database_type,
        llm_model=args.llm_model,
        embedding_model=args.embedding_model,
        generate_purpose=args.purpose,
        generate_summary=args.summary,
        generate_qa=args.qa,
    )
    results = generator.enrich_all(table_ids, batch_size=args.batch_size, dry_run=args.dry_run)
    try:
        results["metadata_cache_stats"] = generator.cache.get_cache_stats()
    except Exception:
        pass
    _print_summary(results)
    return 0


if __name__ == "__main__":
    sys.exit(main(_parse_args()))
