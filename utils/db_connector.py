"""SQLite database access utilities for CORE-T.

Centralizes all direct interactions with the benchmark SQLite databases
(Spider | BIRD | MMQA | BEAVER):
- Resolve database file paths from configured dataset type
- Open connections and run raw SQL
- Inspect table schemas and foreign-key relationships
- Fetch table samples for previews

Keeping this I/O isolated simplifies testing and decouples higher-level modules
from SQLite specifics.
"""

import os
import sqlite3
import time
from typing import Dict, List, Any, Optional, Tuple
import glob
from contextlib import closing
from pathlib import Path

from utils import Configuration, DatabaseType

def _detect_project_root() -> Path:
    """Best-effort project root detection for offline scripts.

    We walk up parent directories looking for `sql_database/` since that's the
    anchor needed for DB path resolution.
    """
    here = Path(__file__).resolve()
    for parent in here.parents:
        # Prefer an anchor that guarantees DB paths resolve.
        if (parent / "sql_database").exists():
            return parent
            
    # Fallback: assume `utils/` lives directly under project root.
    return here.parent.parent

def get_database_base_path(config: Optional['Configuration'] = None) -> str:
    """Return the filesystem base path for benchmark databases.

    If `config` is not provided, a default `Configuration` is created.
    """
    if config is None:
        config = Configuration()
    root = _detect_project_root()
    return str(root / "sql_database" / config.database_type.value)

def get_available_databases(config: Optional['Configuration'] = None) -> List[str]:
    """List available databases under the configured dataset base path.

    Returns a list of database names (without file extensions).
    """
    database_base_path = get_database_base_path(config)
    
    if not os.path.exists(database_base_path):
        print(f"Warning: Database directory not found: {database_base_path}")
        return []
    
    db_names = []
    
    # Look for .sqlite files directly in the database_base_path (Spider structure)
    sqlite_files = glob.glob(os.path.join(database_base_path, "*.sqlite"))
    for sqlite_file in sqlite_files:
        db_name = os.path.splitext(os.path.basename(sqlite_file))[0]
        db_names.append(db_name)
    
    # Also look for .sqlite files in subdirectories (BIRD structure)
    for item in os.listdir(database_base_path):
        item_path = os.path.join(database_base_path, item)
        if os.path.isdir(item_path):
            # Look for .sqlite files in this subdirectory
            subdir_sqlite_files = glob.glob(os.path.join(item_path, "*.sqlite"))
            for sqlite_file in subdir_sqlite_files:
                db_name = os.path.splitext(os.path.basename(sqlite_file))[0]
                if db_name not in db_names:  # Avoid duplicates
                    db_names.append(db_name)
    
    return db_names

def get_database_path(db_name: str, config: Optional['Configuration'] = None) -> str:
    """Return the absolute path to a specific database file by name."""
    base_path = get_database_base_path(config)
    
    # First try direct path (Spider structure)
    direct_path = os.path.join(base_path, f"{db_name}.sqlite")
    if os.path.exists(direct_path):
        return direct_path
    
    # Then try subdirectory path (BIRD structure)
    subdir_path = os.path.join(base_path, db_name, f"{db_name}.sqlite")
    if os.path.exists(subdir_path):
        return subdir_path
    
    # If neither exists, raise an error
    raise ValueError(f"Database '{db_name}' not found in {base_path}")

def get_connection(db_name: str, config: Optional['Configuration'] = None) -> sqlite3.Connection:
    """Open and return a connection to the named SQLite database."""
    db_path = get_database_path(db_name, config)
    
    if not os.path.exists(db_path):
        raise ValueError(f"Database file not found: {db_path}")
    
    return sqlite3.connect(db_path)


def _get_free_memory_mb() -> Optional[int]:
    """Best-effort free memory check. Returns MB or None if unavailable."""
    try:
        import psutil
        return int(psutil.virtual_memory().available / (1024 * 1024))
    except Exception:
        return None


def _count_table_rows(db_id: str, table_name: str, config: Optional['Configuration'] = None) -> int:
    """Return row count for a table. Uses a simple COUNT(*)."""
    try:
        with closing(get_connection(db_id, config)) as conn:
            cur = conn.cursor()
            cur.execute(f"SELECT COUNT(*) FROM `{table_name}`;")
            val = cur.fetchone()
            return int(val[0]) if val and val[0] is not None else 0
    except Exception:
        return 0


def _estimate_table_bytes_dbstat(conn: sqlite3.Connection, table_name: str) -> Optional[int]:
    """Estimate table size in bytes using SQLite dbstat virtual table if available."""
    try:
        cur = conn.cursor()
        # Ensure dbstat is available; many builds have it by default
        cur.execute("SELECT SUM(pgsize) FROM dbstat WHERE name = ?;", (table_name,))
        val = cur.fetchone()
        if val and val[0] is not None:
            return int(val[0])
        return None
    except Exception:
        return None


def _estimate_table_bytes_fallback(db_id: str, table_name: str, config: Optional['Configuration'] = None, sample_rows: int = 200) -> Optional[int]:
    """Fallback estimator: sample some rows to approximate avg row size, multiply by row count.

    This avoids loading all rows into Python; it fetches only a limited sample.
    """
    try:
        with closing(get_connection(db_id, config)) as conn:
            cur = conn.cursor()
            # Get column names
            cur.execute(f"PRAGMA table_info(`{table_name}`);")
            info = cur.fetchall()
            if not info:
                return None
            col_names = [c[1] for c in info]

            # Sample rows
            cur.execute(f"SELECT * FROM `{table_name}` LIMIT {sample_rows};")
            rows = cur.fetchall()
            if not rows:
                # Empty table: negligible
                return 0
            # Approximate byte size of each value
            def _val_size(v: Any) -> int:
                if v is None:
                    return 1
                if isinstance(v, (int, float)):
                    return 8
                try:
                    s = str(v)
                    return len(s.encode('utf-8'))
                except Exception:
                    return 8
            total_bytes = 0
            for row in rows:
                row_bytes = 0
                for v in row:
                    row_bytes += _val_size(v)
                total_bytes += row_bytes
            avg_row_bytes = max(1, total_bytes // len(rows))
            # Count total rows (fast scalar)
            cur.execute(f"SELECT COUNT(*) FROM `{table_name}`;")
            total_rows = int(cur.fetchone()[0])
            return int(avg_row_bytes * total_rows)
    except Exception:
        return None


def estimate_tables_memory_usage(table_full_ids: List[str], config: Optional['Configuration'] = None) -> Tuple[int, Dict[str, int], str]:
    """Estimate total bytes required to load the given tables into an in-memory DB.

    Tries dbstat first per table; falls back to sampling-based estimate.
    Returns (total_bytes, per_table_map, method_used).
    """
    per_table: Dict[str, int] = {}
    method = "dbstat"
    for identifier in table_full_ids or []:
        if "#sep#" not in identifier:
            continue
        db_id, table_name = identifier.split("#sep#", 1)
        # Try dbstat
        size_bytes: Optional[int] = None
        try:
            with closing(get_connection(db_id, config)) as conn:
                size_bytes = _estimate_table_bytes_dbstat(conn, table_name)
        except Exception:
            size_bytes = None
        if size_bytes is None:
            # Fallback to sampling estimator
            est = _estimate_table_bytes_fallback(db_id, table_name, config)
            if est is not None:
                size_bytes = est
                method = "fallback"
        if size_bytes is not None:
            per_table[identifier] = size_bytes
    total = sum(per_table.values())
    return total, per_table, method


def can_build_in_memory(table_full_ids: List[str], config: Optional['Configuration'] = None) -> Tuple[bool, str]:
    """Decide if in-memory DB build is safe using free memory, a hard disable flag,
    and an estimation of required memory vs. available.

    Checks:
    - Hard disable flag INMEM_DISABLE = '1'
    - Estimate total bytes for tables and compare to free memory with overhead factor
    - Minimum free memory (INMEM_MIN_FREE_MB, default 1024 MB), when detectable
    """
    # Honor hard disable
    if os.environ.get('INMEM_DISABLE', '0') == '1':
        return False, "In-memory execution disabled by INMEM_DISABLE"

    # Free memory threshold and overhead factor for safety
    try:
        min_free_mb = int(os.environ.get('INMEM_MIN_FREE_MB', '1024'))
    except Exception:
        min_free_mb = 1024
    try:
        overhead_factor = float(os.environ.get('INMEM_OVERHEAD_FACTOR', '1.3'))  # 30% overhead by default
    except Exception:
        overhead_factor = 1.3

    free_mb = _get_free_memory_mb()
    
    # Conservative: if we can't detect free memory, default to unsafe to prevent OOM
    if free_mb is None:
        return False, "Cannot detect free memory - defaulting to safe fallback"
    
    if free_mb < min_free_mb:
        return False, f"Low free memory ({free_mb} MB < {min_free_mb} MB)"

    # Estimate total bytes needed for selected tables
    total_bytes, per_table, method = estimate_tables_memory_usage(table_full_ids, config)
    if total_bytes <= 0:
        # Unknown/zero estimate: allow since we have adequate free memory
        return True, "OK (estimate unavailable but adequate free memory)"
    
    required_mb = int((total_bytes * overhead_factor) / (1024 * 1024))
    if required_mb > free_mb:
        return False, f"Estimated requirement {required_mb} MB (method={method}) exceeds free {free_mb} MB"

    return True, f"OK (need ~{required_mb} MB, have {free_mb} MB free)"

def get_database_schema(db_name: str, config: Optional['Configuration'] = None) -> Dict[str, Dict[str, str]]:
    """Return a mapping of table name -> {column_name: column_type} for the database."""
    if config is None:
        config = Configuration()
    conn = get_connection(db_name, config)
    cursor = conn.cursor()
    
    # Get all table names (exclude SQLite internal tables for BIRD and BEAVER)
    if config.database_type in [DatabaseType.BIRD, DatabaseType.BEAVER]:
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';")
    else:
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    
    schema = {}
    for table in tables:
        table_name = table[0]
        
        # Get column information for each table
        cursor.execute(f"PRAGMA table_info(`{table_name}`);")
        columns = cursor.fetchall()
        
        table_schema = {}
        for column in columns:
            column_name = column[1]  # Column name is at index 1
            column_type = column[2]  # Column type is at index 2
            table_schema[column_name] = column_type
        
        schema[table_name] = table_schema
    
    conn.close()
    return schema

def get_table_relationships(db_name: str, config: Optional['Configuration'] = None) -> List[Dict[str, str]]:
    """Return foreign key relationships between tables in the database.

    Each relationship dict contains: from_table, from_column, to_table, to_column.
    """
    if config is None:
        config = Configuration()
    conn = get_connection(db_name, config)
    cursor = conn.cursor()
    
    # Get all table names (exclude SQLite internal tables for BIRD)
    if config.database_type in [DatabaseType.BIRD]:
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';")
    else:
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    
    relationships = []
    for table in tables:
        table_name = table[0]
        
        # Get foreign key information for each table
        cursor.execute(f"PRAGMA foreign_key_list(`{table_name}`);")
        foreign_keys = cursor.fetchall()
        
        for fk in foreign_keys:
            relationships.append({
                "from_table": table_name,
                "from_column": fk[3],  # From column is at index 3
                "to_table": fk[2],     # To table is at index 2
                "to_column": fk[4]     # To column is at index 4
            })
    
    conn.close()
    return relationships

def get_table_sample(db_name: str, table_name: str, limit: int = 5, config: Optional['Configuration'] = None) -> List[Dict[str, Any]]:
    """Return up to `limit` rows from `table_name` as a list of dicts."""
    conn = get_connection(db_name, config)
    cursor = conn.cursor()
    
    # Execute query to get sample data
    cursor.execute(f"SELECT * FROM `{table_name}` LIMIT {limit};")
    rows = cursor.fetchall()
    
    # Get column names
    column_names = [description[0] for description in cursor.description]
    
    # Convert rows to dictionaries
    result = []
    for row in rows:
        row_dict = {}
        for i, value in enumerate(row):
            row_dict[column_names[i]] = value
        result.append(row_dict)
    
    conn.close()
    return result

def _get_sql_result_row_limit(config: Optional['Configuration'] = None) -> Optional[int]:
    """Resolve SQL result row cap from env only (state carries preferred value elsewhere)."""
    try:
        if os.environ.get('SQL_RESULT_ROW_LIMIT') is not None:
            return int(os.environ.get('SQL_RESULT_ROW_LIMIT'))
    except Exception:
        return None
    return None


def execute_raw_query(db_name: str, query: str, config: Optional['Configuration'] = None, row_limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """Execute `query` on the database and return result rows as list[dict].

    If row_limit is provided, fetch up to that many rows; otherwise fetch all.
    """
    conn = get_connection(db_name, config)
    try:
        cursor = conn.cursor()
        cursor.execute(query)
        # Column names may be None for statements without a result set
        column_names = [description[0] for description in cursor.description] if cursor.description else []
        if row_limit is not None:
            rows = cursor.fetchmany(max(0, int(row_limit)))
        else:
            rows = cursor.fetchall()
        result: List[Dict[str, Any]] = []
        for row in rows:
            row_dict: Dict[str, Any] = {}
            for i, value in enumerate(row):
                key = column_names[i] if i < len(column_names) else f"col_{i}"
                row_dict[key] = value
            result.append(row_dict)
        return result
    finally:
        conn.close()


def fetch_table_schema_and_rows(db_id: str, table_name: str, config: Optional['Configuration'] = None) -> Tuple[List[Tuple[str, str]], List[Tuple[Any, ...]], List[str]]:
    """Fetch column definitions and all rows for a table from a source SQLite database.

    Returns (columns_with_types, rows, column_names).
    """
    if config is None:
        config = Configuration()
    conn = get_connection(db_id, config)
    try:
        cursor = conn.cursor()
        cursor.execute(f"PRAGMA table_info(`{table_name}`);")
        info = cursor.fetchall()
        if not info:
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            names = [r[0] for r in cursor.fetchall()]
            matched = None
            t_lower = table_name.lower()
            for n in names:
                if n.lower() == t_lower:
                    matched = n
                    break
            if matched:
                table_name = matched
                cursor.execute(f"PRAGMA table_info(`{table_name}`);")
                info = cursor.fetchall()
        columns_with_types: List[Tuple[str, str]] = []
        column_names: List[str] = []
        for col in info:
            col_name = col[1]
            col_type = col[2] or 'TEXT'
            columns_with_types.append((col_name, col_type))
            column_names.append(col_name)
        cursor.execute(f"SELECT * FROM `{table_name}`;")
        rows = cursor.fetchall()
        return columns_with_types, rows, column_names
    finally:
        conn.close()


def create_tables_in_memory(
    mem_conn: sqlite3.Connection,
    tables_spec: List[Tuple[str, str, str, Optional['Configuration']]],
) -> None:
    """Create tables and insert data into the provided in-memory SQLite connection.

    tables_spec: list of (db_id, source_table_name, mem_table_name, config)

    Memory-efficient implementation: stream rows from source DBs in batches
    instead of materializing entire tables in Python memory.
    """
    mem_cursor = mem_conn.cursor()
    BATCH_SIZE = 1000
    for db_id, source_table, mem_table, cfg in tables_spec:
        src_conn = get_connection(db_id, cfg)
        try:
            src_cursor = src_conn.cursor()
            # Fetch schema (column names and types)
            src_cursor.execute(f"PRAGMA table_info(`{source_table}`);")
            info = src_cursor.fetchall()
            if not info:
                # Try case-insensitive match for table name
                src_cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
                names = [r[0] for r in src_cursor.fetchall()]
                matched = None
                t_lower = source_table.lower()
                for n in names:
                    if n.lower() == t_lower:
                        matched = n
                        break
                if not matched:
                    continue
                source_table = matched
                src_cursor.execute(f"PRAGMA table_info(`{source_table}`);")
                info = src_cursor.fetchall()

            cols_with_types: List[Tuple[str, str]] = []
            col_names: List[str] = []
            for col in info:
                col_name = col[1]
                col_type = col[2] or 'TEXT'
                cols_with_types.append((col_name, col_type))
                col_names.append(col_name)

            if not cols_with_types:
                continue

            # Create table in memory
            cols_sql = ", ".join([f"`{name}` {ctype}" for name, ctype in cols_with_types])
            mem_cursor.execute(f"CREATE TABLE IF NOT EXISTS `{mem_table}` ({cols_sql});")

            # Stream rows from source and insert in batches
            src_cursor.execute(f"SELECT * FROM `{source_table}`;")
            placeholders = ",".join(["?"] * len(col_names))
            insert_sql = f"INSERT INTO `{mem_table}` ({', '.join([f'`{c}`' for c in col_names])}) VALUES ({placeholders});"
            inserted_rows = 0
            while True:
                batch = src_cursor.fetchmany(BATCH_SIZE)
                if not batch:
                    break
                mem_cursor.executemany(insert_sql, batch)
                inserted_rows += len(batch)
        finally:
            src_conn.close()
    mem_conn.commit()


def execute_sql_on_combined_tables(
    sql_query: str,
    table_full_ids: List[str],
    config: Optional["Configuration"] = None,
    row_limit: Optional[int] = None,
    alias_to_full_id: Optional[Dict[str, str]] = None,
    *,
    timeout_seconds: int = 60,
) -> Tuple[List[Dict[str, Any]], bool]:
    """Execute a SQL query on an in-memory SQLite DB populated with referenced tables and their data.

    Returns a tuple: (rows, duplicates_skipped), where duplicates_skipped indicates
    whether any tables were skipped due to duplicate in-memory table names.

    table_full_ids must be of the form 'db_id#sep#table_name'.
    """
    if not sql_query:
        return [], False
    # Build spec and handle potential table name collisions
    tables_spec: List[Tuple[str, str, str, Optional['Configuration']]] = []
    seen_mem_names = set()
    duplicates_skipped = False

    if alias_to_full_id:
        # Use provided aliases as in-memory table names to match SQL references
        for alias, identifier in alias_to_full_id.items():
            if "#sep#" not in identifier:
                continue
            db_id, table_name = identifier.split("#sep#", 1)
            mem_table_name = alias
            key_lower = mem_table_name.lower()
            if key_lower in seen_mem_names:
                # Aliases should be unique by construction, but guard anyway
                print(f"⚠️  Duplicate alias for in-memory load: {mem_table_name} → skipping")
                duplicates_skipped = True
                continue
            seen_mem_names.add(key_lower)
            tables_spec.append((db_id, table_name, mem_table_name, config))
    else:
        for identifier in table_full_ids or []:
            if "#sep#" not in identifier:
                # Skip ambiguous identifiers without db context
                continue
            db_id, table_name = identifier.split("#sep#", 1)
            mem_table_name = table_name
            key_lower = mem_table_name.lower()
            if key_lower in seen_mem_names:
                print(f"⚠️  Skipping duplicate table name in in-memory load: {mem_table_name} from db {db_id}")
                duplicates_skipped = True
                continue
            seen_mem_names.add(key_lower)
            tables_spec.append((db_id, table_name, mem_table_name, config))

    mem_conn = sqlite3.connect(":memory:")
    try:
        create_tables_in_memory(mem_conn, tables_spec)
        # Create ephemeral indexes on all columns for each in-memory alias table.
        # This is non-destructive (scoped to the in-memory DB) and can significantly
        # speed up joins, filters, and ORDER BY operations for generated SQL.
        try:
            idx_cur = mem_conn.cursor()
            alias_tables = [spec[2] for spec in tables_spec]
            for mem_table_name in alias_tables:
                try:
                    idx_cur.execute(f"PRAGMA table_info(`{mem_table_name}`);")
                    cols_info = idx_cur.fetchall()
                    for col in cols_info:
                        try:
                            col_name = col[1]
                            idx_name = f"idx_{mem_table_name}_{col_name}"
                            idx_cur.execute(
                                f"CREATE INDEX IF NOT EXISTS `{idx_name}` ON `{mem_table_name}`(`{col_name}`);"
                            )
                        except Exception:
                            # Ignore failures for unusual column names or other constraints.
                            pass
                except Exception:
                    # If table info cannot be read, skip indexing for that table.
                    continue
        except Exception:
            # If ephemeral indexing fails for any reason, proceed with execution.
            pass
        # Install a timeout via the progress handler
        start_ts = time.time()
        def _progress_cb() -> int:
            if (time.time() - start_ts) > timeout_seconds:
                return 1  # non-zero aborts the current operation
            return 0
        mem_conn.set_progress_handler(_progress_cb, 10000)

        cursor = mem_conn.cursor()
        try:
            cursor.execute(sql_query)
        except Exception as e:
            # Map SQLite interruption to a friendly timeout error message
            mem_conn.set_progress_handler(None, 0)
            msg = str(e)
            if "interrupted" in msg.lower():
                raise RuntimeError(f"SQL execution timeout after {int(timeout_seconds)} seconds")
            raise
        col_names = [d[0] for d in cursor.description] if cursor.description else []
        # Revert to single fetch behavior with optional row limit
        if row_limit is not None:
            rows = cursor.fetchmany(max(0, int(row_limit)))
        else:
            rows = cursor.fetchall()
        result: List[Dict[str, Any]] = []
        for row in rows:
            row_dict: Dict[str, Any] = {}
            for i, value in enumerate(row):
                key = col_names[i] if i < len(col_names) else f"col_{i}"
                row_dict[key] = value
            result.append(row_dict)
        # Clear progress handler
        mem_conn.set_progress_handler(None, 0)
        return result, duplicates_skipped
    finally:
        mem_conn.close()

def execute_sql_on_attached_tables(
    sql_query: str,
    table_full_ids: List[str],
    config: Optional["Configuration"] = None,
    row_limit: Optional[int] = None,
    alias_to_full_id: Optional[Dict[str, str]] = None,
    *,
    timeout_seconds: int = 60,
) -> Tuple[List[Dict[str, Any]], bool]:
    """Execute a SQL query by ATTACH-ing source databases and creating TEMP views for aliases.

    This avoids copying rows into memory. For each involved db_id we ATTACH its SQLite file
    into a single connection. For each alias mapping (alias -> db_id#sep#table_name), we
    create a TEMP VIEW named `alias` that selects from the attached schema's table. The
    provided SQL can then reference aliases directly without change.

    Returns (rows, duplicates_skipped). The duplicates_skipped flag is False for ATTACH-based
    execution as aliases should be unique by construction.
    """
    if not sql_query:
        return [], False

    # Helper: collect unique db_ids and a mapping of db_id -> set of table_names referenced
    referenced_db_ids: List[str] = []
    if alias_to_full_id and isinstance(alias_to_full_id, dict):
        for identifier in alias_to_full_id.values():
            if "#sep#" in identifier:
                db_id, _ = identifier.split("#sep#", 1)
                if db_id not in referenced_db_ids:
                    referenced_db_ids.append(db_id)
    else:
        # Fallback to table_full_ids
        for identifier in table_full_ids or []:
            if "#sep#" in identifier:
                db_id, _ = identifier.split("#sep#", 1)
                if db_id not in referenced_db_ids:
                    referenced_db_ids.append(db_id)

    def _sanitize_schema_name(db_id: str) -> str:
        # SQLite schema (attached db) names must be valid identifiers; keep it simple
        import re
        base = re.sub(r"[^A-Za-z0-9_]", "_", f"db_{db_id}")
        if not base or base[0].isdigit():
            base = f"db_{base}"
        return base

    conn = sqlite3.connect(":memory:")
    try:
        cur = conn.cursor()

        # Attach each referenced database under a unique schema name
        dbid_to_schema: Dict[str, str] = {}
        for db_id in referenced_db_ids:
            try:
                db_path = get_database_path(db_id, config)
                schema = _sanitize_schema_name(db_id)
                # Ensure unique schema name if collisions occur
                suffix = 1
                unique_schema = schema
                while unique_schema in dbid_to_schema.values():
                    suffix += 1
                    unique_schema = f"{schema}_{suffix}"
                cur.execute(f"ATTACH DATABASE ? AS `{unique_schema}`;", (db_path,))
                dbid_to_schema[db_id] = unique_schema
            except Exception as e:
                raise ValueError(f"Failed to ATTACH database {db_id}: {e}")

        # Create TEMP views matching aliases so the generated SQL can run unchanged
        duplicates_skipped = False
        created_views_lower = set()

        if alias_to_full_id and isinstance(alias_to_full_id, dict):
            for alias, identifier in alias_to_full_id.items():
                if "#sep#" not in identifier:
                    continue
                db_id, table_name = identifier.split("#sep#", 1)
                schema = dbid_to_schema.get(db_id)
                if not schema:
                    continue
                alias_lower = str(alias).lower()
                if alias_lower in created_views_lower:
                    # Guard against duplicate alias names (should not happen)
                    duplicates_skipped = True
                    continue
                created_views_lower.add(alias_lower)
                cur.execute(f"CREATE TEMP VIEW `{alias}` AS SELECT * FROM `{schema}`.`{table_name}`;")
        else:
            # Fallback: create views with source table names. This may fail if names collide.
            for identifier in table_full_ids or []:
                if "#sep#" not in identifier:
                    continue
                db_id, table_name = identifier.split("#sep#", 1)
                schema = dbid_to_schema.get(db_id)
                if not schema:
                    continue
                view_name = table_name
                view_lower = view_name.lower()
                if view_lower in created_views_lower:
                    duplicates_skipped = True
                    continue
                created_views_lower.add(view_lower)
                cur.execute(f"CREATE TEMP VIEW `{view_name}` AS SELECT * FROM `{schema}`.`{table_name}`;")

        # Install a timeout via the progress handler
        start_ts = time.time()
        def _progress_cb2() -> int:
            if (time.time() - start_ts) > timeout_seconds:
                return 1
            return 0
        conn.set_progress_handler(_progress_cb2, 10000)

        # Execute the query and collect results with optional row limit
        try:
            cur.execute(sql_query)
        except Exception as e:
            conn.set_progress_handler(None, 0)
            msg = str(e)
            if "interrupted" in msg.lower():
                raise RuntimeError(f"SQL execution timeout after {int(timeout_seconds)} seconds")
            raise
        col_names = [d[0] for d in cur.description] if cur.description else []
        if row_limit is not None:
            rows_raw = cur.fetchmany(max(0, int(row_limit)))
        else:
            rows_raw = cur.fetchall()
        result: List[Dict[str, Any]] = []
        for row in rows_raw:
            row_dict: Dict[str, Any] = {}
            for i, value in enumerate(row):
                key = col_names[i] if i < len(col_names) else f"col_{i}"
                row_dict[key] = value
            result.append(row_dict)
        conn.set_progress_handler(None, 0)
        return result, duplicates_skipped
    finally:
        conn.close()

def get_top_column_values(db_name: str, table_name: str, column_name: str, limit: int = 5, config: Optional['Configuration'] = None) -> List[Any]:
    """Return up to `limit` most frequent non-null values for a column.

    Values are ordered by frequency descending, ties resolved by the database default ordering.
    """
    conn = get_connection(db_name, config)
    cursor = conn.cursor()
    try:
        # Use identifier quoting with backticks for safety
        cursor.execute(
            f"SELECT `{column_name}` AS v, COUNT(*) AS c FROM `{table_name}` WHERE `{column_name}` IS NOT NULL GROUP BY `{column_name}` ORDER BY c DESC LIMIT {limit};"
        )
        rows = cursor.fetchall()
        values = [row[0] for row in rows]
        return values
    finally:
        conn.close()