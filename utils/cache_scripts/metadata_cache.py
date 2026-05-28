"""
Metadata cache.

This is a lightweight, file-backed cache for table-level metadata produced by
offline preprocessing (e.g., table purpose/enriched tables).

Storage:
  cache/<dataset>/metadata/<metadata_type>/<db_id>#sep#<table_name>.json

Key format:
  <db_id>#sep#<table_name>   (table_name is lowercased for stable keys)
"""

import json
from pathlib import Path
from typing import Any, Dict, Optional, Literal


class MetadataCache:
    """File-backed cache for table-level metadata.

    Stores per-table metadata as JSON files under a database-specific cache
    directory. Lookups are keyed by the pair (db_id, table_name) for
    unambiguous identification.
    """

    MetaType = Literal["purpose", "summary", "qa_pairs", "enriched_tables"]
    METADATA_TYPES: tuple[MetaType, ...] = (
        "purpose",
        "summary",
        "qa_pairs",
        "enriched_tables",
    )

    def __init__(self, cache_dir: Optional[str] = None) -> None:
        """Initialize the cache.

        Args:
            cache_dir: Root directory for cache files. If not provided, the
                database-specific directory from `utils.Configuration` is used.
        """
        if cache_dir is None:
            from utils import Configuration

            config = Configuration()
            cache_dir = config.get_database_cache_dir("metadata")

        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        for metadata_type in self.METADATA_TYPES:
            (self.cache_dir / metadata_type).mkdir(exist_ok=True)

    def _get_table_key(self, db_id: str, table_name: str) -> str:
        """Return a filesystem-safe key using the canonical '#sep#' separator.

        This matches the naming convention used by the offline preprocessed table
        cache files: `{db_id}#sep#{table_name}`.
        """
        raw_key = f"{db_id}#sep#{table_name}"
        return raw_key.replace("/", "_slash_").replace("\\", "_backslash_").replace(":", "_colon_")

    def _legacy_table_key(self, db_id: str, table_name: str) -> str:
        """Backward compatible key for older metadata caches."""
        raw_key = f"{db_id}___{table_name}"
        return raw_key.replace("/", "_slash_").replace("\\", "_backslash_").replace(":", "_colon_")

    def _get_cache_path(self, metadata_type: MetaType, cache_key: str) -> Path:
        return self.cache_dir / metadata_type / f"{cache_key}.json"

    def get_metadata(
        self,
        _table_identifier: str,
        metadata_type: MetaType,
        db_id: Optional[str] = None,
        table_name: Optional[str] = None,
    ) -> Optional[Any]:
        # Normalize identifiers (ensure table_name is lowercase for consistent keys)
        if table_name is not None:
            table_name = table_name.lower()
        if not db_id or not table_name:
            return None

        # Prefer the new '#sep#' key, but also support older caches that used
        # 'db___table' (and legacy variants without '#sep#').
        candidate_keys = [
            self._get_table_key(db_id, table_name),
            self._legacy_table_key(db_id, table_name),
            self._get_table_key(db_id, table_name).replace("#sep#", "#"),
        ]

        for table_key in candidate_keys:
            cache_path = self._get_cache_path(metadata_type, table_key)
            if not cache_path.exists():
                continue
            try:
                with cache_path.open("r", encoding="utf-8") as f:
                    data = json.load(f)
                return data.get("metadata")
            except Exception:
                continue

        return None

    def set_metadata(
        self,
        _table_identifier: str,
        metadata_type: MetaType,
        metadata: Any,
        db_id: Optional[str] = None,
        table_name: Optional[str] = None,
    ) -> None:
        # Normalize identifiers (ensure table_name is lowercase for consistent keys)
        if table_name is not None:
            table_name = table_name.lower()
        if not db_id or not table_name:
            return

        table_key = self._get_table_key(db_id, table_name)
        cache_data = {
            "db_id": db_id,
            "table_name": table_name,
            "table_key": table_key,
            "metadata_type": metadata_type,
            "metadata": metadata,
            "cache_version": "2.0",
        }

        cache_path = self._get_cache_path(metadata_type, table_key)
        try:
            with cache_path.open("w", encoding="utf-8") as f:
                json.dump(cache_data, f, indent=2, ensure_ascii=False)
        except Exception:
            return

    def get_enriched_table(
        self,
        table_identifier: str,
        db_id: Optional[str] = None,
        table_name: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        return self.get_metadata(table_identifier, "enriched_tables", db_id, table_name)

    def set_enriched_table(
        self,
        table_identifier: str,
        enriched_table: Dict[str, Any],
        db_id: Optional[str] = None,
        table_name: Optional[str] = None,
    ) -> None:
        self.set_metadata(table_identifier, "enriched_tables", enriched_table, db_id, table_name)

    def get_cache_stats(self) -> Dict[str, int]:
        stats: Dict[str, int] = {}
        for metadata_type in self.METADATA_TYPES:
            cache_subdir = self.cache_dir / metadata_type
            stats[metadata_type] = len(list(cache_subdir.glob("*.json"))) if cache_subdir.exists() else 0
        stats["total"] = sum(stats.values())
        return stats


_metadata_cache_instances: Dict[str, MetadataCache] = {}


def get_metadata_cache(cache_dir: Optional[str] = None) -> MetadataCache:
    """Return a shared `MetadataCache` instance for `cache_dir`."""
    if cache_dir is None:
        from utils import Configuration

        config = Configuration()
        cache_dir = config.get_database_cache_dir("metadata")

    cache_dir = str(Path(cache_dir).resolve())
    cache = _metadata_cache_instances.get(cache_dir)
    if cache is None:
        cache = MetadataCache(cache_dir)
        _metadata_cache_instances[cache_dir] = cache
    return cache

