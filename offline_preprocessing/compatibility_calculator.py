#!/usr/bin/env python3
"""
Offline compatibility calculator.

This script computes (and caches) pairwise compatibility scores between tables.

Inputs:
  - `data/{dataset}/dev_tables.json` (or `--tables-file`)
  - CLI args for dataset/model knobs (see below).

Outputs:
  - Pairwise compatibility cache files under `cache/{dataset}/compatibility/`
  - (Optional) table-parameter caches under `cache/{dataset}/tables_parameters/`

Usage:
  python offline_preprocessing/compatibility_calculator.py --dataset bird --embedding-model "fireworks:WhereIsAI/UAE-Large-V1" --llm-model "huggingface:Qwen/Qwen2.5-7B-Instruct"
"""

import argparse
import csv
import json
import sys
import time
from collections import OrderedDict
from dataclasses import dataclass
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
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv(PROJECT_ROOT / ".env")

from utils import Configuration, DatabaseType
from utils.db_connector import get_connection, get_database_path, get_database_schema
from utils.cache_scripts.faiss_vector_cache import get_faiss_vector_cache
from utils.storage_manager import UnifiedStorageManager, get_unified_storage_manager


TABLE_ID_SEPARATOR = "#sep#"


# -----------------------------------------------------------------------------
# Table identifier loading
# -----------------------------------------------------------------------------

def _normalize_table_name(name: Any) -> str:
    s = str(name).strip()
    if (s.startswith("`") and s.endswith("`")) or (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
        s = s[1:-1].strip()
    return s.lower()


def _read_table_identifiers_file(path: Path) -> List[str]:
    with path.open("r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def _dev_tables_requested_by_db(payload: Any) -> Dict[str, Set[str]]:
    requested_by_db: Dict[str, Set[str]] = {}

    if isinstance(payload, dict):
        for key, item in payload.items():
            if isinstance(key, str) and TABLE_ID_SEPARATOR in key:
                db_id, table_name = key.split(TABLE_ID_SEPARATOR, 1)
                requested_by_db.setdefault(db_id, set()).add(_normalize_table_name(table_name))
                continue
            if isinstance(item, dict):
                db_id = item.get("db_id") or item.get("dbId")
                table_name = item.get("table_name_original") or item.get("table_name") or item.get("table")
                if db_id and table_name:
                    requested_by_db.setdefault(str(db_id), set()).add(_normalize_table_name(table_name))
        return requested_by_db

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


# -----------------------------------------------------------------------------
# Compatibility logic
# -----------------------------------------------------------------------------

@dataclass
class TableCompatibilityScore:
    table1_id: str
    table2_id: str
    jaccard_similarity: float
    semantic_similarity: float
    exact_similarity: float
    uniqueness_score: float
    subset_relationship: float
    overall_compatibility: float
    best_join_columns: Tuple[str, str]
    join_confidence: float


def _safe_float(x: Any) -> float:
    try:
        v = float(x)
    except Exception:
        return 0.0
    if v != v:  # NaN
        return 0.0
    if v == float("inf") or v == float("-inf"):
        return 0.0
    return v


def _overlap_coefficient(a: str, b: str) -> float:
    words1 = set(str(a).replace("_", " ").replace(".", " ").lower().split())
    words2 = set(str(b).replace("_", " ").replace(".", " ").lower().split())
    if not words1 or not words2:
        return 0.0
    inter = len(words1 & words2)
    denom = min(len(words1), len(words2))
    return float(inter / denom) if denom > 0 else 0.0


def _load_json_dict(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _save_json_dict(path: Path, data: Dict[str, Any]) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    except Exception:
        return


class CompatibilityCalculator:
    def __init__(
        self,
        *,
        config: Configuration,
        storage: UnifiedStorageManager,
        embedding_model: str,
        use_csv_fastpath: bool = False,
        use_jaccard: bool = True,
        use_semantic: bool = True,
        use_exact: bool = True,
        use_subset: bool = True,
        use_uniqueness: bool = True,
        max_distinct: Optional[int] = None,
        max_cached_table_profiles: int = 8,
    ) -> None:
        self.config = config
        self.storage = storage

        self.use_jaccard = bool(use_jaccard)
        self.use_semantic = bool(use_semantic)
        self.use_exact = bool(use_exact)
        self.use_subset = bool(use_subset)
        self.use_uniqueness = bool(use_uniqueness)
        self.use_csv_fastpath = bool(use_csv_fastpath)
        self.max_distinct = None if max_distinct is None else int(max_distinct)

        cache_root = Path(self.storage.cache_dir)
        self.tables_params_dir = cache_root / "tables_parameters"
        # Parameter caches are stored directly under `cache/<dataset>/tables_parameters/`.
        self.tables_params_dir.mkdir(parents=True, exist_ok=True)

        self.uniqueness_cache: Dict[str, Any] = _load_json_dict(self.tables_params_dir / "column_uniqueness.json")
        self.jaccard_cache: Dict[str, Any] = _load_json_dict(self.tables_params_dir / "jaccard_similarity.json")
        self.exact_cache: Dict[str, Any] = _load_json_dict(self.tables_params_dir / "exact_similarity.json")
        self.semantic_cache: Dict[str, Any] = _load_json_dict(self.tables_params_dir / "semantic_similarity.json")
        self.subset_cache: Dict[str, Any] = _load_json_dict(self.tables_params_dir / "subset_relationships.json")

        # Embedding cache for semantic similarities (only used on cache miss)
        self.vector_cache = get_faiss_vector_cache(
            cache_dir=str(cache_root / "faiss_vectors"),
            model=embedding_model,
            config=self.config,
        )

        self._params_dirty = False
        # Optional CSV-backed table profile cache:
        # table_id -> {"column_stats": {col: {"null_count", "non_null_count", "distinct_values"}}}
        self._table_profile_cache: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
        # Keep memory bounded; values are large sets for wide/high-cardinality tables.
        self._max_cached_table_profiles = max(1, int(max_cached_table_profiles))

    def _is_null_csv_value(self, v: Any) -> bool:
        """Best-effort NULL normalization for CSV exports."""
        if v is None:
            return True
        s = str(v).strip()
        if s == "":
            return True
        return s.lower() in {"null", "none", "nan", "nat"}

    def _resolve_csv_file_path(self, db_id: str, table_name: str) -> Optional[Path]:
        """
        Resolve CSV path for a table under:
          sql_database/<dataset>/<db_id>/data_csv/<table>.csv
        """
        try:
            db_path = Path(get_database_path(db_id, self.config))
            csv_dir = db_path.parent / "data_csv"
            if not csv_dir.exists():
                return None
            direct = csv_dir / f"{table_name}.csv"
            if direct.exists():
                return direct
            target = str(table_name).lower()
            for p in csv_dir.glob("*.csv"):
                if p.stem.lower() == target:
                    return p
        except Exception:
            return None
        return None

    def _load_table_profile_from_csv(self, db_id: str, table_id: str, table_name: str, schema: Dict[str, str]) -> Optional[Dict[str, Any]]:
        if not self.use_csv_fastpath:
            return None

        profile_key = str(table_id).lower()
        if profile_key in self._table_profile_cache:
            self._table_profile_cache.move_to_end(profile_key)
            return self._table_profile_cache[profile_key]

        csv_path = self._resolve_csv_file_path(db_id, table_name)
        if csv_path is None:
            return None

        schema_cols = list(schema.keys())
        if not schema_cols:
            return None
        schema_cols_set = set(schema_cols)
        stats: Dict[str, Dict[str, Any]] = {
            col: {"null_count": 0, "non_null_count": 0, "distinct_values": set()} for col in schema_cols
        }

        try:
            with csv_path.open("r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                csv_fields = list(reader.fieldnames or [])
                field_by_schema_col: Dict[str, str] = {}
                csv_fields_lower = {str(c).lower(): str(c) for c in csv_fields}
                for col in schema_cols:
                    if col in csv_fields:
                        field_by_schema_col[col] = col
                    else:
                        mapped = csv_fields_lower.get(str(col).lower())
                        if mapped is not None:
                            field_by_schema_col[col] = mapped

                # If there is no overlap, this CSV likely doesn't match the schema.
                if not field_by_schema_col:
                    return None

                for row in reader:
                    row_map = row if isinstance(row, dict) else {}
                    for col, field_name in field_by_schema_col.items():
                        val = row_map.get(field_name)
                        col_stats = stats[col]
                        if self._is_null_csv_value(val):
                            col_stats["null_count"] += 1
                        else:
                            col_stats["non_null_count"] += 1
                            col_stats["distinct_values"].add(str(val))
        except Exception:
            return None

        profile = {"column_stats": stats, "schema_cols": schema_cols_set}
        self._table_profile_cache[profile_key] = profile
        self._table_profile_cache.move_to_end(profile_key)
        while len(self._table_profile_cache) > int(self._max_cached_table_profiles):
            self._table_profile_cache.popitem(last=False)
        return profile

    def _maybe_flush_param_caches(self) -> None:
        if not self._params_dirty:
            return
        _save_json_dict(self.tables_params_dir / "column_uniqueness.json", self.uniqueness_cache)
        _save_json_dict(self.tables_params_dir / "jaccard_similarity.json", self.jaccard_cache)
        _save_json_dict(self.tables_params_dir / "exact_similarity.json", self.exact_cache)
        _save_json_dict(self.tables_params_dir / "semantic_similarity.json", self.semantic_cache)
        _save_json_dict(self.tables_params_dir / "subset_relationships.json", self.subset_cache)
        self._params_dirty = False

    def _resolve_table_schema(self, db_id: str, table_name: str) -> Tuple[str, Dict[str, str]]:
        schema = get_database_schema(db_id, self.config)
        target = str(table_name).lower()
        actual = None
        for t in schema.keys():
            if str(t).lower() == target:
                actual = t
                break
        if actual is None:
            raise ValueError(f"Table '{table_name}' not found in database '{db_id}'")
        table_schema = schema.get(actual) or {}
        if not isinstance(table_schema, dict):
            table_schema = {}
        return str(actual), {str(k): str(v) for k, v in table_schema.items()}

    def _get_pair_cache(self, cache: Dict[str, Any], t1: str, t2: str) -> Optional[Dict[str, float]]:
        key = f"{t1}-{t2}"
        rev = f"{t2}-{t1}"
        val = cache.get(key)
        if isinstance(val, dict):
            return {str(k): _safe_float(v) for k, v in val.items()}
        val = cache.get(rev)
        if isinstance(val, dict):
            return {str(k): _safe_float(v) for k, v in val.items()}
        return None

    def _set_pair_cache(self, cache: Dict[str, Any], t1: str, t2: str, payload: Dict[str, float]) -> None:
        cache[f"{t1}-{t2}"] = payload
        self._params_dirty = True

    def _get_uniqueness(self, db_id: str, table_id: str, conn, schema: Dict[str, str]) -> Dict[str, float]:
        if not self.use_uniqueness:
            return {col: 0.0 for col in schema.keys()}

        prefix = f"{table_id}{TABLE_ID_SEPARATOR}".lower()
        cached = {k: v for k, v in self.uniqueness_cache.items() if isinstance(k, str) and k.lower().startswith(prefix)}
        if cached:
            out: Dict[str, float] = {}
            for k, v in cached.items():
                out[str(k).split(TABLE_ID_SEPARATOR)[-1]] = _safe_float(v)
            return out

        table_name = table_id.split(TABLE_ID_SEPARATOR, 1)[1]
        csv_profile = self._load_table_profile_from_csv(db_id, table_id, table_name, schema)

        # Preferred fast path: use pre-extracted CSV table content.
        if csv_profile is not None:
            out: Dict[str, float] = {}
            col_stats_map = csv_profile.get("column_stats", {}) or {}
            for col in schema.keys():
                try:
                    col_stats = col_stats_map.get(col, {}) or {}
                    null_count = int(col_stats.get("null_count", 0) or 0)
                    non_null_count = int(col_stats.get("non_null_count", 0) or 0)
                    if null_count > 0:
                        score = 0.0
                    elif non_null_count <= 0:
                        score = 0.0
                    else:
                        uniq = len(col_stats.get("distinct_values", set()) or set())
                        score = float(uniq / non_null_count) if non_null_count > 0 else 0.0
                    out[col] = _safe_float(score)
                    self.uniqueness_cache[f"{table_id}{TABLE_ID_SEPARATOR}{col}"] = out[col]
                except Exception:
                    out[col] = 0.0
                    self.uniqueness_cache[f"{table_id}{TABLE_ID_SEPARATOR}{col}"] = 0.0
            self._params_dirty = True
            return out

        # Fallback: original SQL-based behavior.
        cur = conn.cursor()
        out: Dict[str, float] = {}
        for col in schema.keys():
            try:
                cur.execute(f'SELECT COUNT(*) FROM "{table_name}" WHERE "{col}" IS NULL')
                null_count = int(cur.fetchone()[0] or 0)
                if null_count > 0:
                    score = 0.0
                else:
                    cur.execute(f'SELECT COUNT(*) FROM "{table_name}" WHERE "{col}" IS NOT NULL')
                    total = int(cur.fetchone()[0] or 0)
                    if total <= 0:
                        score = 0.0
                    else:
                        cur.execute(
                            f'SELECT COUNT(DISTINCT "{col}") FROM "{table_name}" WHERE "{col}" IS NOT NULL'
                        )
                        uniq = int(cur.fetchone()[0] or 0)
                        score = float(uniq / total) if total > 0 else 0.0
                out[col] = _safe_float(score)
                self.uniqueness_cache[f"{table_id}{TABLE_ID_SEPARATOR}{col}"] = out[col]
            except Exception:
                out[col] = 0.0
                self.uniqueness_cache[f"{table_id}{TABLE_ID_SEPARATOR}{col}"] = 0.0

        self._params_dirty = True
        return out

    def _distinct_values(
        self,
        conn,
        db_id: str,
        table_id: str,
        table_name: str,
        column: str,
        schema: Dict[str, str],
    ) -> Set[str]:
        csv_profile = self._load_table_profile_from_csv(db_id, table_id, table_name, schema)
        if csv_profile is not None:
            col_stats = (csv_profile.get("column_stats", {}) or {}).get(column, {}) or {}
            vals = list(col_stats.get("distinct_values", set()) or set())
            if self.max_distinct is None or self.max_distinct <= 0:
                return set(vals)
            return set(vals[: int(self.max_distinct)])

        # Fallback: original SQL-based behavior.
        cur = conn.cursor()
        # If max_distinct is None/<=0, do not apply a LIMIT (unbounded).
        if self.max_distinct is None or self.max_distinct <= 0:
            cur.execute(f'SELECT DISTINCT "{column}" FROM "{table_name}" WHERE "{column}" IS NOT NULL')
        else:
            cur.execute(
                f'SELECT DISTINCT "{column}" FROM "{table_name}" WHERE "{column}" IS NOT NULL LIMIT {int(self.max_distinct)}'
            )
        return {str(r[0]) for r in cur.fetchall()}

    def _calc_jaccard_for_pair(
        self,
        conn1,
        conn2,
        table1_name: str,
        table2_name: str,
        table1_id: str,
        table2_id: str,
        schema1: Dict[str, str],
        schema2: Dict[str, str],
    ) -> Dict[str, float]:
        cached = self._get_pair_cache(self.jaccard_cache, table1_id, table2_id)
        if cached is not None:
            return cached

        if not self.use_jaccard:
            payload: Dict[str, float] = {}
            self._set_pair_cache(self.jaccard_cache, table1_id, table2_id, payload)
            return payload

        payload: Dict[str, float] = {}
        cols1 = list(schema1.keys())
        cols2 = list(schema2.keys())
        for c1 in cols1:
            try:
                v1 = self._distinct_values(conn1, table1_id.split(TABLE_ID_SEPARATOR, 1)[0], table1_id, table1_name, c1, schema1)
            except Exception:
                continue
            for c2 in cols2:
                try:
                    v2 = self._distinct_values(conn2, table2_id.split(TABLE_ID_SEPARATOR, 1)[0], table2_id, table2_name, c2, schema2)
                except Exception:
                    continue
                if not v1 and not v2:
                    score = 0.0
                else:
                    inter = len(v1 & v2)
                    union = len(v1 | v2)
                    score = float(inter / union) if union > 0 else 0.0
                key = f"{table1_id}{TABLE_ID_SEPARATOR}{c1}-{table2_id}{TABLE_ID_SEPARATOR}{c2}"
                payload[key] = _safe_float(score)

        self._set_pair_cache(self.jaccard_cache, table1_id, table2_id, payload)
        return payload

    def _calc_subset_for_pair(
        self,
        conn1,
        conn2,
        table1_name: str,
        table2_name: str,
        table1_id: str,
        table2_id: str,
        schema1: Dict[str, str],
        schema2: Dict[str, str],
    ) -> Dict[str, float]:
        cached = self._get_pair_cache(self.subset_cache, table1_id, table2_id)
        if cached is not None:
            return cached

        if not self.use_subset:
            payload: Dict[str, float] = {}
            self._set_pair_cache(self.subset_cache, table1_id, table2_id, payload)
            return payload

        payload: Dict[str, float] = {}
        cols1 = list(schema1.keys())
        cols2 = list(schema2.keys())
        for c1 in cols1:
            try:
                v1 = self._distinct_values(conn1, table1_id.split(TABLE_ID_SEPARATOR, 1)[0], table1_id, table1_name, c1, schema1)
            except Exception:
                continue
            for c2 in cols2:
                try:
                    v2 = self._distinct_values(conn2, table2_id.split(TABLE_ID_SEPARATOR, 1)[0], table2_id, table2_name, c2, schema2)
                except Exception:
                    continue
                subset_score = 1.0 if (v1 and v2 and (v1.issubset(v2) or v2.issubset(v1))) else 0.0
                key = f"{table1_id}{TABLE_ID_SEPARATOR}{c1}-{table2_id}{TABLE_ID_SEPARATOR}{c2}"
                payload[key] = float(subset_score)

        self._set_pair_cache(self.subset_cache, table1_id, table2_id, payload)
        return payload

    def _calc_name_sims_for_pair(
        self,
        table1_id: str,
        table2_id: str,
        table1_name: str,
        table2_name: str,
        schema1: Dict[str, str],
        schema2: Dict[str, str],
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        exact_cached = self._get_pair_cache(self.exact_cache, table1_id, table2_id)
        semantic_cached = self._get_pair_cache(self.semantic_cache, table1_id, table2_id)
        if exact_cached is not None and semantic_cached is not None:
            return exact_cached, semantic_cached

        cols1 = list(schema1.keys())
        cols2 = list(schema2.keys())

        exact_payload = exact_cached or {}
        semantic_payload = semantic_cached or {}

        # Exact similarities are cheap, compute if missing and enabled.
        if self.use_exact and exact_cached is None:
            for c1 in cols1:
                for c2 in cols2:
                    key = f"{table1_id}{TABLE_ID_SEPARATOR}{c1}-{table2_id}{TABLE_ID_SEPARATOR}{c2}"
                    exact_payload[key] = _overlap_coefficient(f"{table1_name} {c1}", f"{table2_name} {c2}")
            self._set_pair_cache(self.exact_cache, table1_id, table2_id, exact_payload)

        # Semantic similarities use embeddings, compute only on cache miss and if enabled.
        if self.use_semantic and semantic_cached is None:
            emb1: Dict[str, Optional[List[float]]] = {}
            emb2: Dict[str, Optional[List[float]]] = {}
            for c1 in cols1:
                desc = f"{table1_name} {c1}".replace("_", " ").lower().strip()
                emb1[c1] = self.vector_cache.get_or_create_embedding(desc, "column", {"table_id": table1_id, "column": c1})
            for c2 in cols2:
                desc = f"{table2_name} {c2}".replace("_", " ").lower().strip()
                emb2[c2] = self.vector_cache.get_or_create_embedding(desc, "column", {"table_id": table2_id, "column": c2})
            for c1 in cols1:
                for c2 in cols2:
                    key = f"{table1_id}{TABLE_ID_SEPARATOR}{c1}-{table2_id}{TABLE_ID_SEPARATOR}{c2}"
                    v1 = emb1.get(c1)
                    v2 = emb2.get(c2)
                    if v1 and v2:
                        semantic_payload[key] = self.vector_cache.cosine_similarity(v1, v2)
                    else:
                        semantic_payload[key] = 0.0
            self._set_pair_cache(self.semantic_cache, table1_id, table2_id, semantic_payload)

        # If disabled, ensure dicts exist for downstream logic.
        if not self.use_exact and exact_cached is None:
            self._set_pair_cache(self.exact_cache, table1_id, table2_id, exact_payload)
        if not self.use_semantic and semantic_cached is None:
            self._set_pair_cache(self.semantic_cache, table1_id, table2_id, semantic_payload)

        return exact_payload, semantic_payload

    def calculate_table_compatibility(self, table1_info: Dict[str, Any], table2_info: Dict[str, Any]) -> TableCompatibilityScore:
        t1 = f"{table1_info['db_id']}{TABLE_ID_SEPARATOR}{table1_info['table_name']}".lower()
        t2 = f"{table2_info['db_id']}{TABLE_ID_SEPARATOR}{table2_info['table_name']}".lower()

        cached = self.storage.get_cached_compatibility_score(t1, t2)
        if isinstance(cached, dict) and cached:
            return TableCompatibilityScore(
                table1_id=t1,
                table2_id=t2,
                jaccard_similarity=_safe_float(cached.get("jaccard_similarity", 0.0)),
                semantic_similarity=_safe_float(cached.get("semantic_similarity", 0.0)),
                exact_similarity=_safe_float(cached.get("exact_similarity", 0.0)),
                uniqueness_score=_safe_float(cached.get("uniqueness_score", 0.0)),
                subset_relationship=_safe_float(cached.get("subset_relationship", 0.0)),
                overall_compatibility=_safe_float(cached.get("overall_compatibility", 0.0)),
                best_join_columns=tuple(cached.get("best_join_columns", ("", ""))),  # type: ignore[arg-type]
                join_confidence=_safe_float(cached.get("join_confidence", 0.0)),
            )

        # Resolve schemas + open DB connections (SQLite)
        table1_name_actual, schema1 = self._resolve_table_schema(table1_info["db_id"], table1_info["table_name"])
        table2_name_actual, schema2 = self._resolve_table_schema(table2_info["db_id"], table2_info["table_name"])

        conn1 = get_connection(table1_info["db_id"], self.config)
        conn2 = get_connection(table2_info["db_id"], self.config)
        try:
            uniq1 = self._get_uniqueness(table1_info["db_id"], t1, conn1, schema1)
            uniq2 = self._get_uniqueness(table2_info["db_id"], t2, conn2, schema2)

            exact_sims, semantic_sims = self._calc_name_sims_for_pair(
                t1, t2, table1_name_actual, table2_name_actual, schema1, schema2
            )
            jaccard_sims = self._calc_jaccard_for_pair(conn1, conn2, table1_name_actual, table2_name_actual, t1, t2, schema1, schema2)
            subset_sims = self._calc_subset_for_pair(conn1, conn2, table1_name_actual, table2_name_actual, t1, t2, schema1, schema2)

            # Compute best join over same-type columns with strict gating (uniqueness + subset)
            best_score = 0.0
            best_cols: Tuple[str, str] = ("", "")

            cols1 = list(schema1.keys())
            cols2 = list(schema2.keys())
            for c1 in cols1:
                for c2 in cols2:
                    if str(schema1.get(c1)) != str(schema2.get(c2)):
                        continue
                    max_u = max(_safe_float(uniq1.get(c1, 0.0)), _safe_float(uniq2.get(c2, 0.0))) if self.use_uniqueness else 0.0
                    subset_key = f"{t1}{TABLE_ID_SEPARATOR}{c1}-{t2}{TABLE_ID_SEPARATOR}{c2}"
                    subset_val = _safe_float(subset_sims.get(subset_key, 0.0)) if self.use_subset else 0.0
                    if max_u < 1.0 or subset_val < 1.0:
                        continue

                    jac = _safe_float(jaccard_sims.get(subset_key, 0.0)) if self.use_jaccard else 0.0
                    sem = _safe_float(semantic_sims.get(subset_key, 0.0)) if self.use_semantic else 0.0
                    ex = _safe_float(exact_sims.get(subset_key, 0.0)) if self.use_exact else 0.0
                    name_sim = 0.5 * sem + 0.5 * ex
                    score = (0.5 * jac + 0.5 * name_sim) * max_u

                    if score > best_score:
                        best_score = score
                        best_cols = (c1, c2)

            # Aggregate metrics (use averages as a simple summary)
            avg_jaccard = _safe_float(sum(jaccard_sims.values()) / len(jaccard_sims)) if jaccard_sims else 0.0
            avg_semantic = _safe_float(sum(semantic_sims.values()) / len(semantic_sims)) if semantic_sims else 0.0
            avg_exact = _safe_float(sum(exact_sims.values()) / len(exact_sims)) if exact_sims else 0.0

            all_u = list(uniq1.values()) + list(uniq2.values())
            avg_u = _safe_float(sum(all_u) / len(all_u)) if all_u else 0.0
            avg_subset = _safe_float(sum(subset_sims.values()) / len(subset_sims)) if subset_sims else 0.0

            overall = _safe_float(best_score)
            join_confidence = _safe_float(min(1.0, best_score))

            score_obj = TableCompatibilityScore(
                table1_id=t1,
                table2_id=t2,
                jaccard_similarity=avg_jaccard,
                semantic_similarity=avg_semantic,
                exact_similarity=avg_exact,
                uniqueness_score=avg_u,
                subset_relationship=avg_subset,
                overall_compatibility=overall,
                best_join_columns=best_cols,
                join_confidence=join_confidence,
            )

            # Persist compatibility result (this is the main artifact)
            self.storage.cache_compatibility_score(
                t1,
                t2,
                {
                    "jaccard_similarity": score_obj.jaccard_similarity,
                    "semantic_similarity": score_obj.semantic_similarity,
                    "exact_similarity": score_obj.exact_similarity,
                    "uniqueness_score": score_obj.uniqueness_score,
                    "subset_relationship": score_obj.subset_relationship,
                    "overall_compatibility": score_obj.overall_compatibility,
                    "best_join_columns": score_obj.best_join_columns,
                    "join_confidence": score_obj.join_confidence,
                    "calculation_timestamp": time.time(),
                },
            )

            # Persist parameter caches best-effort (optional artifact)
            self._maybe_flush_param_caches()
            return score_obj
        finally:
            try:
                conn1.close()
            except Exception:
                pass
            try:
                conn2.close()
            except Exception:
                pass


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute and cache pairwise table compatibility scores.")
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
    parser.add_argument("--batch-size", type=int, default=50, help="Number of pairs per batch.")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of tables to process (for quick tests).")
    parser.add_argument(
        "--tables-file",
        type=str,
        default=None,
        help="Optional path to a file listing table identifiers (one per line). If omitted, uses dev_tables.json.",
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default="openai:text-embedding-3-large",
        help="Embedding model for semantic column similarity (provider-prefixed supported).",
    )
    parser.add_argument(
        "--llm-model",
        type=str,
        default="openai:gpt-4o-mini",
        help="LLM tag used for cache run folder naming.",
    )
    parser.add_argument("--dry-run", action="store_true", default=False, help="Load identifiers and print counts only.")

    parser.add_argument("--jaccard", action=argparse.BooleanOptionalAction, default=True, help="Enable/disable jaccard value overlap.")
    parser.add_argument("--semantic", action=argparse.BooleanOptionalAction, default=True, help="Enable/disable semantic similarity (embeddings).")
    parser.add_argument("--exact", action=argparse.BooleanOptionalAction, default=True, help="Enable/disable exact name overlap.")
    parser.add_argument("--subset", action=argparse.BooleanOptionalAction, default=True, help="Enable/disable subset relationship checks.")
    parser.add_argument("--uniqueness", action=argparse.BooleanOptionalAction, default=True, help="Enable/disable uniqueness gating.")
    parser.add_argument(
        "--max-distinct",
        type=int,
        default=0,
        help="Max distinct values per column for jaccard/subset. Use 0 to disable the limit (default).",
    )
    parser.add_argument(
        "--use-csv-fastpath",
        action="store_true",
        default=True,
        help=(
            "If set, prefer loading table content from sql_database/<dataset>/<db_id>/data_csv/*.csv "
            "for uniqueness/distinct computations, with SQL fallback on miss/error. "
            "Default is SQL-only behavior."
        ),
    )
    parser.add_argument(
        "--max-cached-table-profiles",
        type=int,
        default=463,
        help="Maximum number of parsed CSV table profiles to keep in the in-memory LRU cache.",
    )

    return parser.parse_args(argv)


def _print_summary(results: Dict[str, Any]) -> None:
    print("\n" + "=" * 60)
    print("OFFLINE COMPATIBILITY CALCULATION SUMMARY")
    print("=" * 60)
    print(f"Total Pairs: {results['total_pairs']}")
    print(f"Successful: {results['successful']}")
    print(f"Errors: {results['errors']}")
    print(f"Elapsed: {results['elapsed_seconds']:.2f}s")
    print(f"Compatibility Cache Directory: {results['compatibility_cache_directory']}")
    if results.get("errors_sample"):
        sample = results["errors_sample"]
        print(f"\nFirst {len(sample)} Errors:")
        for i, err in enumerate(sample):
            print(f"  {i + 1}. {err}")


def main(args: argparse.Namespace) -> int:
    database_type = DatabaseType(args.dataset)
    table_ids = _load_table_identifiers(database_type, args.tables_file)
    if args.limit is not None:
        table_ids = table_ids[: int(args.limit)]

    if not table_ids:
        print("No table identifiers found. Exiting.")
        return 1

    if args.dry_run:
        total_pairs = (len(table_ids) * (len(table_ids) - 1)) // 2
        print(f"Tables: {len(table_ids)}")
        print(f"Pairs:  {total_pairs}")
        return 0

    config = Configuration(
        database_type=database_type,
        llm_model=str(args.llm_model),
        embedding_model=str(args.embedding_model),
    )
    # New cache layout is handled by UnifiedStorageManager when cache_dir="cache".
    storage = UnifiedStorageManager(config=config, cache_dir="cache")
    calc = CompatibilityCalculator(
        config=config,
        storage=storage,
        embedding_model=str(args.embedding_model),
        use_csv_fastpath=bool(args.use_csv_fastpath),
        use_jaccard=args.jaccard,
        use_semantic=args.semantic,
        use_exact=args.exact,
        use_subset=args.subset,
        use_uniqueness=args.uniqueness,
        max_distinct=None if int(args.max_distinct) <= 0 else int(args.max_distinct),
        max_cached_table_profiles=max(1, int(args.max_cached_table_profiles)),
    )

    started = time.time()
    errors: List[str] = []
    successful = 0

    total_pairs = (len(table_ids) * (len(table_ids) - 1)) // 2
    batch_size = max(1, int(args.batch_size))
    pairs: List[Tuple[int, int]] = []
    for i in range(len(table_ids)):
        for j in range(i + 1, len(table_ids)):
            pairs.append((i, j))

    total_batches = (len(pairs) + batch_size - 1) // batch_size
    for batch_idx in range(total_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, len(pairs))
        batch_pairs = pairs[start_idx:end_idx]
        print(f"\nProcessing batch {batch_idx + 1}/{total_batches} (pairs {start_idx + 1}-{end_idx})")

        for i, j in batch_pairs:
            t1 = table_ids[i]
            t2 = table_ids[j]
            if TABLE_ID_SEPARATOR not in t1 or TABLE_ID_SEPARATOR not in t2:
                continue
            db1, name1 = t1.split(TABLE_ID_SEPARATOR, 1)
            db2, name2 = t2.split(TABLE_ID_SEPARATOR, 1)
            try:
                _ = calc.calculate_table_compatibility(
                    {"db_id": db1, "table_name": name1},
                    {"db_id": db2, "table_name": name2},
                )
                successful += 1
            except Exception as e:
                errors.append(str(e))

    results = {
        "total_pairs": total_pairs,
        "successful": successful,
        "errors": len(errors),
        "errors_sample": errors[:10],
        "elapsed_seconds": time.time() - started,
        "compatibility_cache_directory": str(storage.compatibility_cache_dir),
    }
    _print_summary(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(_parse_args()))

