#!/usr/bin/env python3
"""
SQL generator.

This script reads a dataset file with questions (e.g., data/bird/dev.json)
and a per-question list of tables (e.g., results/<dataset>/<run_tag>/results_table_augmentation/augmented_tables.json),
then generates one SQLite SQL query per question using an LLM.

Usage:
    python sql_generator.py --dataset bird --sql-model "openai:gpt-4o-mini" --embedding-model "fireworks:WhereIsAI/UAE-Large-V1" --llm-model "huggingface:Qwen/Qwen2.5-7B-Instruct"
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv


TABLE_ID_SEPARATOR = "#sep#"


def _project_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "data").exists():
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

from utils import (  # noqa: E402
    DatabaseType,
    Configuration,
    create_llm,
    get_results_run_dir,
    summarize_llm_invoke_error,
)
from utils.db_connector import get_database_schema, get_table_relationships  # noqa: E402
from utils.prompts import get_sql_generation_prompt_lrm  # noqa: E402
from utils.storage_manager import UnifiedStorageManager  # noqa: E402
from utils.cache_scripts.sql_generation_cache import get_sql_generation_cache  # noqa: E402


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, payload: Any, *, indent: int = 2) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=indent, ensure_ascii=False)


def _content_to_text(content: Any) -> str:
    """Normalize LangChain message content (string or structured parts) to plain text."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    try:
        if isinstance(content, list):
            parts: List[str] = []
            for p in content:
                if isinstance(p, str):
                    parts.append(p)
                elif isinstance(p, dict) and "text" in p:
                    parts.append(str(p.get("text", "")))
                else:
                    parts.append(str(getattr(p, "text", getattr(p, "content", p))))
            return "\n".join(parts)
    except Exception:
        pass
    return str(getattr(content, "content", content))


def _strip_thinking(text: str) -> str:
    return re.sub(r"<think>[\s\S]*?</think>", "", str(text), flags=re.IGNORECASE).strip()


def _sanitize_generated_sql(raw_sql: str) -> str:
    if not raw_sql:
        return ""
    text = _strip_thinking(str(raw_sql)).strip()

    # Strip common Markdown fences.
    if text.startswith("```sqlite"):
        text = text[len("```sqlite") :]
    if text.startswith("```sql"):
        text = text[len("```sql") :]
    if text.startswith("```"):
        text = text[len("```") :]
    if text.endswith("```"):
        text = text[: -len("```")]
    return text.strip()


def _extract_sql_tables(
    sql: str,
    *,
    known_aliases: Optional[Dict[str, str]] = None,
    fallback_full_ids: Optional[List[str]] = None,
) -> Tuple[List[str], List[str]]:
    """
    Extract table names referenced by SQL and map them to full ids (db_id#sep#table).

    This mirrors the execution path preference:
      - If we can compute `sql_tables_full_ids`, use that for execution.
      - Otherwise, callers can fall back to the selected/resolved tables list.

    Returns:
      (sql_tables, sql_tables_full_ids)
    """
    sql = str(sql or "").strip()
    if not sql:
        return [], []

    aliases = known_aliases or {}
    fallback_full_ids = list(fallback_full_ids or [])

    # Best-effort parse using sqlglot; if unavailable or parsing fails, fall back to a
    # conservative regex for common SELECT ... FROM/JOIN patterns.
    try:
        import sqlglot
        from sqlglot import exp

        expr = sqlglot.parse_one(sql, read="sqlite")

        # Collect CTE names to avoid treating them as physical tables.
        cte_names = set()
        for cte in expr.find_all(exp.CTE):
            try:
                name = cte.alias_or_name
            except Exception:
                name = None
            if isinstance(name, str) and name:
                cte_names.add(name.lower())

        table_names: List[str] = []
        seen = set()
        for t in expr.find_all(exp.Table):
            name = getattr(t, "name", None)
            if not isinstance(name, str) or not name:
                continue
            if name.lower() in cte_names:
                continue
            if name.lower() in seen:
                continue
            seen.add(name.lower())
            table_names.append(name)
    except Exception:
        # Regex fallback.
        # 1) Collect CTE names: WITH cte AS (...), other_cte AS (...)
        cte_names = set()
        try:
            cte_pat = re.compile(r"(?is)\bWITH\s+([A-Za-z_][\w]*)\s+AS\s*\(|,\s*([A-Za-z_][\w]*)\s+AS\s*\(")
            for m in cte_pat.finditer(sql):
                nm = m.group(1) or m.group(2)
                if nm:
                    cte_names.add(str(nm).lower())
        except Exception:
            cte_names = set()

        # 2) Capture identifiers after FROM/JOIN (ignore subqueries since they start with '(')
        table_names = []
        seen = set()
        try:
            tbl_pat = re.compile(
                r'(?is)\b(?:FROM|JOIN)\s+((?:`[^`]+`|"[^"]+"|\[[^\]]+\]|[A-Za-z_][\w\.]*))'
            )
            for m in tbl_pat.finditer(sql):
                raw = str(m.group(1) or "").strip()
                if not raw:
                    continue
                # Strip quoting
                if raw.startswith("`") and raw.endswith("`"):
                    raw = raw[1:-1]
                elif raw.startswith('"') and raw.endswith('"'):
                    raw = raw[1:-1]
                elif raw.startswith("[") and raw.endswith("]"):
                    raw = raw[1:-1]
                # If qualified (schema.table), keep the last part
                name = raw.split(".")[-1].strip()
                if not name:
                    continue
                if name.lower() in cte_names:
                    continue
                if name.lower() in seen:
                    continue
                seen.add(name.lower())
                table_names.append(name)
        except Exception:
            return [], []

    # Map extracted table names to full ids. Prefer alias_to_full_id from prompt context.
    full_ids: List[str] = []
    seen_full = set()

    # Case-insensitive alias lookup
    alias_lut = {str(k).lower(): str(v) for k, v in aliases.items() if isinstance(k, str) and isinstance(v, str)}

    for name in table_names:
        full_id = alias_lut.get(str(name).lower())
        if not full_id and fallback_full_ids:
            # Last resort: match on table name within full ids.
            target = str(name).strip().lower()
            for fid in fallback_full_ids:
                if TABLE_ID_SEPARATOR not in str(fid):
                    continue
                _db, tname = str(fid).split(TABLE_ID_SEPARATOR, 1)
                if tname.strip().lower() == target:
                    full_id = str(fid)
                    break
        if not full_id:
            continue
        if full_id.lower() in seen_full:
            continue
        seen_full.add(full_id.lower())
        full_ids.append(full_id)

    return table_names, full_ids


def _normalize_table_id(table_id: Any) -> str:
    s = str(table_id or "").strip()
    if not s:
        return ""
    if TABLE_ID_SEPARATOR in s:
        dbp, tp = s.split(TABLE_ID_SEPARATOR, 1)
        return f"{dbp.strip().lower()}{TABLE_ID_SEPARATOR}{tp.strip().lower()}"
    return s.lower()


def _parse_table_id(table_id: str) -> Optional[Tuple[str, str]]:
    s = _normalize_table_id(table_id)
    if not s:
        return None
    if TABLE_ID_SEPARATOR not in s:
        return None
    db_id, table_name = s.split(TABLE_ID_SEPARATOR, 1)
    return db_id.strip(), table_name.strip()


def _get_actual_table_name(schema: Dict[str, Any], table_name: str) -> Optional[str]:
    target = str(table_name or "").lower()
    for actual in schema.keys():
        if str(actual).lower() == target:
            return str(actual)
    return None


def _invoke_llm(model, prompt: str):
    """Invoke a LangChain chat model with best-effort input format."""
    try:
        return model.invoke(prompt)
    except Exception:
        try:
            from langchain_core.messages import HumanMessage

            return model.invoke([HumanMessage(content=prompt)])
        except Exception as e:
            raise RuntimeError(f"Failed to invoke LLM: {summarize_llm_invoke_error(e)}") from e


def _llm_usage_from_message(ai_message: Any) -> Dict[str, Any]:
    usage = {"input_tokens": None, "output_tokens": None, "total_tokens": None}
    try:
        meta = getattr(ai_message, "response_metadata", {}) or {}
        usage_meta = meta.get("token_usage") or getattr(ai_message, "usage_metadata", {}) or {}
        usage["input_tokens"] = usage_meta.get("input_tokens") or usage_meta.get("prompt_tokens")
        usage["output_tokens"] = usage_meta.get("output_tokens") or usage_meta.get("completion_tokens")
        usage["total_tokens"] = usage_meta.get("total_tokens")
    except Exception:
        pass
    return usage


def _build_sql_context(
    *,
    selected_tables: List[str],
    default_db_id: str,
    storage: UnifiedStorageManager,
    config: Configuration,
) -> Tuple[Dict[str, Dict[str, str]], List[Dict[str, Any]], Dict[str, Dict[str, List[Any]]], Dict[str, str], List[str]]:
    """
    Build:
      - table_schemas: alias -> {column: type}
      - relationships: list[{from_table, from_column, to_table, to_column}] with aliases
      - column_value_hints: alias -> {column: [values]}
      - alias_to_full_id: alias -> db_id#sep#table
      - resolved_full_ids: normalized full identifiers actually used (after schema matching)
    """
    all_schemas: Dict[str, Dict[str, str]] = {}
    alias_to_full_id: Dict[str, str] = {}
    used_names_lower: Dict[str, int] = {}
    resolved_full_ids: List[str] = []

    # Group tables by database id.
    db_tables: Dict[str, List[str]] = {}
    for t_full in selected_tables or []:
        parsed = _parse_table_id(t_full)
        if parsed is None:
            # If table id is missing db_id, assume the question's db_id.
            t_name = str(t_full or "").strip()
            if t_name:
                db_tables.setdefault(default_db_id, []).append(t_name)
            continue
        t_db_id, t_name = parsed
        if t_db_id and t_name:
            db_tables.setdefault(t_db_id, []).append(t_name)

    # Resolve tables against schema for each db, and alias duplicates across all dbs deterministically.
    all_relationships: List[Dict[str, Any]] = []
    for db_id, table_names in db_tables.items():
        try:
            db_schema = get_database_schema(db_id, config)
        except Exception:
            continue

        # Map original actual table name -> alias used in prompt (for relationships)
        original_to_alias: Dict[str, str] = {}

        for name in table_names:
            actual = _get_actual_table_name(db_schema, name)
            if actual is None:
                continue

            base = actual
            key_lower = base.lower()
            if key_lower not in used_names_lower:
                alias = base
                used_names_lower[key_lower] = 1
            else:
                suffix = used_names_lower[key_lower]
                candidate = f"{base}_{suffix}"
                while candidate.lower() in used_names_lower:
                    suffix += 1
                    candidate = f"{base}_{suffix}"
                alias = candidate
                used_names_lower[candidate.lower()] = 1
                used_names_lower[key_lower] = suffix + 1

            all_schemas[alias] = dict(db_schema.get(actual, {}) or {})
            alias_to_full_id[alias] = f"{db_id}{TABLE_ID_SEPARATOR}{actual}"
            original_to_alias[actual] = alias
            resolved_full_ids.append(alias_to_full_id[alias])

        # Relationships for this db: include only those touching selected tables from this db.
        try:
            db_relationships = get_table_relationships(db_id, config) or []
        except Exception:
            db_relationships = []

        name_set = {(_get_actual_table_name(db_schema, t) or t).lower() for t in table_names}
        for rel in db_relationships:
            try:
                f_tbl = str(rel.get("from_table", "") or "")
                t_tbl = str(rel.get("to_table", "") or "")
                if f_tbl.lower() not in name_set and t_tbl.lower() not in name_set:
                    continue
                all_relationships.append(
                    {
                        "from_table": original_to_alias.get(f_tbl, f_tbl),
                        "from_column": rel.get("from_column", ""),
                        "to_table": original_to_alias.get(t_tbl, t_tbl),
                        "to_column": rel.get("to_column", ""),
                    }
                )
            except Exception:
                continue

    if not all_schemas:
        return {}, [], {}, {}, []

    # Column value hints: prefer cached hints from preprocessed tables.
    column_value_hints: Dict[str, Dict[str, List[Any]]] = {}
    for alias, full_id in alias_to_full_id.items():
        parsed = _parse_table_id(full_id)
        if parsed is None:
            continue
        t_db_id, t_name = parsed
        cached = storage.get_cached_preprocessed_table(t_db_id, t_name)
        meta = (cached or {}).get("metadata", {}) or {}
        hints = meta.get("column_value_hints")
        if isinstance(hints, dict) and hints:
            # Ensure values are lists (json should already be)
            cleaned: Dict[str, List[Any]] = {}
            for col, vals in hints.items():
                if isinstance(vals, list) and vals:
                    cleaned[str(col)] = vals
            if cleaned:
                column_value_hints[alias] = cleaned

    return all_schemas, all_relationships, column_value_hints, alias_to_full_id, resolved_full_ids


def generate_sql_for_question(
    *,
    db_id: str,
    question: str,
    evidence: str,
    selected_tables: List[str],
    sql_model_name: str,
    temperature: float,
    storage: UnifiedStorageManager,
    config: Configuration,
) -> Dict[str, Any]:
    if not question:
        return {"sql": None, "error": "Empty question", "tables": selected_tables}
    if not selected_tables:
        return {"sql": None, "error": "No selected tables provided", "tables": selected_tables}

    table_schemas, relationships, value_hints, alias_to_full_id, resolved_full_ids = _build_sql_context(
        selected_tables=selected_tables,
        default_db_id=db_id,
        storage=storage,
        config=config,
    )
    if not table_schemas:
        return {
            "sql": None,
            "error": "Could not resolve any selected tables against the database schema",
            "tables": selected_tables,
            "resolved_tables": resolved_full_ids,
        }

    # Cache lookup (keyed by question + selected tables + sql model).
    # Cache is stored under: cache/<dataset>/sql_generation/sql_generation_cache.json
    try:
        sql_cache = get_sql_generation_cache(config.get_dataset_cache_dir("sql_generation"))
        cached = sql_cache.get_sql(question, selected_tables, config, llm_model_override=str(sql_model_name))
    except Exception:
        cached = None

    if cached and isinstance(cached, dict) and cached.get("sql"):
        cached_sql = _sanitize_generated_sql(str(cached.get("sql") or ""))
        sql_tables, sql_tables_full_ids = _extract_sql_tables(
            cached_sql,
            known_aliases=alias_to_full_id,
            fallback_full_ids=resolved_full_ids,
        )
        return {
            "sql": cached_sql,
            "error": cached.get("error"),
            "tables": list(cached.get("tables") or selected_tables),
            "resolved_tables": resolved_full_ids,
            "table_alias_mapping": alias_to_full_id,
            "sql_tables": sql_tables,
            "sql_tables_full_ids": sql_tables_full_ids,
            "llm_usage": cached.get("llm_usage", {"input_tokens": None, "output_tokens": None, "total_tokens": None}),
            "schema_info": cached.get("schema_info", {}),
            "cache_hit": True,
        }

    prompt = get_sql_generation_prompt_lrm(
        question,
        table_schemas,
        relationships,
        value_hints,
        evidence=evidence or "",
    )

    model = create_llm(str(sql_model_name), temperature=float(temperature))

    llm_usage: Dict[str, Any] = {"input_tokens": None, "output_tokens": None, "total_tokens": None}
    try:
        ai_message = _invoke_llm(model, prompt)
        llm_usage = _llm_usage_from_message(ai_message)
        raw_sql = _content_to_text(getattr(ai_message, "content", ""))
        sql = _sanitize_generated_sql(raw_sql)
        if not sql:
            result = {
                "sql": None,
                "error": "Model returned empty SQL",
                "tables": selected_tables,
                "resolved_tables": resolved_full_ids,
                "table_alias_mapping": alias_to_full_id,
                "sql_tables": [],
                "sql_tables_full_ids": [],
                "llm_usage": llm_usage,
                "schema_info": {
                    "table_count": len(table_schemas),
                    "tables_used": list(table_schemas.keys()),
                    "relationships_count": len(relationships),
                },
                "cache_hit": False,
            }
            try:
                sql_cache = get_sql_generation_cache(config.get_dataset_cache_dir("sql_generation"))
                sql_cache.set_sql(question, selected_tables, result, config, llm_model_override=str(sql_model_name))
            except Exception:
                pass
            return result

        sql_tables, sql_tables_full_ids = _extract_sql_tables(
            sql,
            known_aliases=alias_to_full_id,
            fallback_full_ids=resolved_full_ids,
        )
        result = {
            "sql": sql,
            "error": None,
            "tables": selected_tables,
            "resolved_tables": resolved_full_ids,
            "table_alias_mapping": alias_to_full_id,
            "sql_tables": sql_tables,
            "sql_tables_full_ids": sql_tables_full_ids,
            "llm_usage": llm_usage,
            "schema_info": {
                "table_count": len(table_schemas),
                "tables_used": list(table_schemas.keys()),
                "relationships_count": len(relationships),
            },
            "cache_hit": False,
        }
        try:
            sql_cache = get_sql_generation_cache(config.get_dataset_cache_dir("sql_generation"))
            sql_cache.set_sql(question, selected_tables, result, config, llm_model_override=str(sql_model_name))
        except Exception:
            pass
        return result
    except Exception as e:
        result = {
            "sql": None,
            "error": str(e),
            "tables": selected_tables,
            "resolved_tables": resolved_full_ids,
            "table_alias_mapping": alias_to_full_id,
            "sql_tables": [],
            "sql_tables_full_ids": [],
            "llm_usage": llm_usage,
            "schema_info": {
                "table_count": len(table_schemas),
                "tables_used": list(table_schemas.keys()),
                "relationships_count": len(relationships),
            },
            "cache_hit": False,
        }
        try:
            sql_cache = get_sql_generation_cache(config.get_dataset_cache_dir("sql_generation"))
            sql_cache.set_sql(question, selected_tables, result, config, llm_model_override=str(sql_model_name))
        except Exception:
            pass
        return result


def _coerce_selected_tables(payload: Any) -> List[List[str]]:
    if not isinstance(payload, list):
        return []
    out: List[List[str]] = []
    for entry in payload:
        if isinstance(entry, list):
            out.append([_normalize_table_id(x) for x in entry if _normalize_table_id(x)])
            continue
        if isinstance(entry, dict):
            # tolerate details-like structures
            for key in ("selected_tables", "augmented_tables", "tables"):
                if isinstance(entry.get(key), list):
                    out.append([_normalize_table_id(x) for x in entry.get(key, []) if _normalize_table_id(x)])
                    break
            else:
                out.append([])
            continue
        out.append([])
    return out


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Generate SQLite SQL queries from (question, table list) pairs. "
            "The default tables input is augmented_tables.json from table_augmentor.py "
            "(override with --selected-tables)."
        )
    )
    p.add_argument(
        "--data",
        type=str,
        default=None,
        help="Dataset JSON path (defaults to data/<dataset>/dev.json).",
    )
    p.add_argument(
        "--selected-tables",
        type=str,
        default=None,
        help="Path to per-question table lists JSON (defaults to results/<dataset>/{llm}_{embedding_model}/results_table_augmentation/augmented_tables.json).",
    )
    p.add_argument(
        "--enriched-tables",
        type=str,
        default=None,
        help="Path to enriched tables cache directory (defaults to cache/<dataset>/{llm}_{embedding_model}/metadata/enriched_tables).",
    )
    p.add_argument(
        "--dataset",
        dest="dataset",
        choices=[dt.value for dt in DatabaseType],
        default=DatabaseType.BIRD.value,
        help="Dataset used for cache/results path resolution.",
    )
    # Backward-compatible alias.
    p.add_argument(
        "--database-type",
        dest="dataset",
        choices=[dt.value for dt in DatabaseType],
        default=DatabaseType.BIRD.value,
        help="(deprecated) Same as --dataset.",
    )
    p.add_argument("--sql-model", type=str, required=True, help="SQL LLM id for LangChain init_chat_model.")
    p.add_argument(
        "--llm-model",
        type=str,
        default="openai:gpt-4o-mini",
        help="Run tag used for cache/results path resolution (should match table_selector/dense_retriever llm-model).",
    )
    p.add_argument(
        "--embedding-model",
        type=str,
        default="fireworks:WhereIsAI/UAE-Large-V1",
        help="Embedding model tag used for cache/results run folder naming.",
    )
    p.add_argument("--temperature", type=float, default=0.0, help="LLM temperature.")
    p.add_argument("--num-samples", type=int, default=None, help="Process only the first N questions (deterministic).")
    p.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output JSON path (defaults to results/<dataset>/{llm}_{embedding_model}/results_sql_generation/sqls.json).",
    )
    p.add_argument(
        "--out-details",
        type=str,
        default=None,
        help="Output JSON path with verbose per-question details (defaults to results/<dataset>/{llm}_{embedding_model}/results_sql_generation/sqls_details.json).",
    )
    p.add_argument("--indent", type=int, default=2, help="JSON indentation level.")
    return p.parse_args(argv)


def main(args: argparse.Namespace) -> int:
    dataset = DatabaseType(args.dataset)
    data_path = (
        Path(args.data).expanduser().resolve()
        if args.data
        else (PROJECT_ROOT / "data" / dataset.value / "dev.json").resolve()
    )
    config = Configuration(
        database_type=dataset,
        # IMPORTANT: this is used to resolve cache/results paths (run tag), not the SQL model.
        llm_model=str(args.llm_model),
        embedding_model=str(args.embedding_model),
        temperature=float(args.temperature),
    )

    if args.selected_tables:
        selected_tables_path = Path(args.selected_tables).expanduser().resolve()
    else:
        run_dir = get_results_run_dir(
            dataset=str(config.database_type.value),
            step_dirname="results_table_augmentation",
            llm_model=str(config.llm_model),
            embedding_model=str(config.embedding_model),
            project_root=PROJECT_ROOT,
        )
        selected_tables_path = (run_dir / "augmented_tables.json").resolve()

    if args.enriched_tables:
        enriched_dir = Path(args.enriched_tables).expanduser().resolve()
    else:
        enriched_dir = (Path(config.get_database_cache_dir("metadata")) / "enriched_tables").resolve()

    if not data_path.exists():
        print(f"❌ Data file not found: {data_path}")
        return 1
    if not selected_tables_path.exists():
        print(f"❌ Tables input file not found: {selected_tables_path}")
        return 1
    if not enriched_dir.exists():
        print(f"❌ Enriched-tables directory not found: {enriched_dir}")
        return 1

    storage = UnifiedStorageManager(config=config, cache_dir="cache")

    payload = _read_json(data_path)
    if not isinstance(payload, list):
        print("❌ --data must be a JSON list[dict].")
        return 1
    questions: List[Dict[str, Any]] = [x for x in payload if isinstance(x, dict)]
    if not questions:
        print("❌ No questions found in --data.")
        return 1

    selected_payload = _read_json(selected_tables_path)
    selected_tables_per_q = _coerce_selected_tables(selected_payload)
    if not selected_tables_per_q:
        print("❌ No table lists found in the tables input JSON.")
        return 1

    n = min(len(questions), len(selected_tables_per_q))
    limit = int(args.num_samples) if args.num_samples is not None else n
    limit = max(0, min(limit, n))
    if limit == 0:
        print("Nothing to do.")
        return 0

    sqls_out: List[Optional[str]] = []
    details_out: List[Dict[str, Any]] = []

    print(f"Loaded questions: {len(questions)} | selected tables: {len(selected_tables_per_q)}")
    print(f"Processing: {limit} question(s)")

    for i in range(limit):
        item = questions[i]
        db_id = str(item.get("db_id", "") or "").strip().lower()
        question = str(item.get("question", "") or "")
        evidence = str(item.get("evidence", "") or "")
        qid = item.get("question_id", i)

        selected_tables_raw = selected_tables_per_q[i] if i < len(selected_tables_per_q) else []
        selected_tables_norm: List[str] = []
        for t in selected_tables_raw or []:
            nt = _normalize_table_id(t)
            if not nt:
                continue
            parsed = _parse_table_id(nt)
            if parsed is None:
                # If the table id is missing db_id, assume it's for this db.
                nt = f"{db_id}{TABLE_ID_SEPARATOR}{nt}"
                parsed = _parse_table_id(nt)
            if parsed is None:
                continue
            selected_tables_norm.append(nt)

        # Validate (best-effort) against enriched tables cache files.
        missing_enriched: List[str] = []
        for t in selected_tables_norm:
            enriched_path = (enriched_dir / f"{t}.json").resolve()
            if not enriched_path.exists():
                missing_enriched.append(t)

        selected_db_ids = sorted({t.split(TABLE_ID_SEPARATOR, 1)[0] for t in selected_tables_norm if TABLE_ID_SEPARATOR in t})

        res = generate_sql_for_question(
            db_id=db_id,
            question=question,
            evidence=evidence,
            selected_tables=selected_tables_norm,
            sql_model_name=str(args.sql_model),
            temperature=float(args.temperature),
            storage=storage,
            config=config,
        )

        sqls_out.append(res.get("sql"))
        cache_hit = bool(res.get("cache_hit", False))
        details_out.append(
            {
                "question_id": qid,
                "index": i,
                "db_id": db_id,
                "question": question,
                "evidence": evidence,
                "selected_tables_input": selected_tables_raw,
                "selected_tables_used": selected_tables_norm,
                "selected_db_ids": selected_db_ids,
                "missing_enriched_cache": missing_enriched,
                "sql": res.get("sql"),
                "error": res.get("error"),
                "resolved_tables": res.get("resolved_tables", []),
                "table_alias_mapping": res.get("table_alias_mapping", {}),
                "sql_tables": res.get("sql_tables", []),
                "sql_tables_full_ids": res.get("sql_tables_full_ids", []),
                "llm_usage": res.get("llm_usage", {}),
                "schema_info": res.get("schema_info", {}),
                "cache_hit": bool(res.get("cache_hit", False)),
            }
        )

        ok = bool(res.get("sql")) and not res.get("error")
        print(
            f"  - question_id={qid} question_db={db_id} "
            f"selected_tables={len(selected_tables_norm)} selected_dbs={len(selected_db_ids)} "
            f"cache_hit={'yes' if cache_hit else 'no'} -> {'OK' if ok else 'FAIL'}"
        )

    if args.out:
        out_path = Path(args.out).expanduser().resolve()
    else:
        out_dir = get_results_run_dir(
            dataset=str(config.database_type.value),
            step_dirname="results_sql_generation",
            llm_model=str(config.llm_model),
            embedding_model=str(config.embedding_model),
            project_root=PROJECT_ROOT,
        )
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = (out_dir / "sqls.json").resolve()

    if args.out_details:
        out_details_path = Path(args.out_details).expanduser().resolve()
    else:
        out_dir = get_results_run_dir(
            dataset=str(config.database_type.value),
            step_dirname="results_sql_generation",
            llm_model=str(config.llm_model),
            embedding_model=str(config.embedding_model),
            project_root=PROJECT_ROOT,
        )
        out_dir.mkdir(parents=True, exist_ok=True)
        out_details_path = (out_dir / "sqls_details.json").resolve()
    _write_json(out_path, sqls_out, indent=int(args.indent))
    _write_json(out_details_path, details_out, indent=int(args.indent))
    print(f"Wrote SQLs: {out_path}")
    print(f"Wrote details: {out_details_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main(_parse_args()))

