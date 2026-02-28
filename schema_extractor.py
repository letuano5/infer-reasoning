"""
Extract schema information from SQLite databases in the format
defined by m_schema_example.txt:

## [DB_ID] <schema_id>
### Schema
# Table: <table_name>
(col:TYPE, Primary Key, Examples: [...]),
...
Sample rows from: `<table_name>`
row1
row2
...
[Foreign keys]
fk1
fk2
"""

import sqlite3
import os
import logging

logger = logging.getLogger(__name__)

SQLITE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sqlite")
SAMPLE_ROWS = 3
EXAMPLE_VALUES = 3


def _get_tables(cursor):
    """Get all table names in the database."""
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;")
    return [row[0] for row in cursor.fetchall()]


def _get_columns(cursor, table_name):
    """Get column info: name, type, and whether it's a primary key."""
    cursor.execute(f"PRAGMA table_info(`{table_name}`);")
    # Returns: cid, name, type, notnull, dflt_value, pk
    return cursor.fetchall()


def _get_foreign_keys(cursor, table_name):
    """Get foreign key relationships for a table."""
    cursor.execute(f"PRAGMA foreign_key_list(`{table_name}`);")
    # Returns: id, seq, table, from, to, on_update, on_delete, match
    return cursor.fetchall()


def _get_sample_values(cursor, table_name, column_name, limit=EXAMPLE_VALUES):
    """Get a few distinct non-null example values for a column."""
    try:
        cursor.execute(
            f"SELECT DISTINCT `{column_name}` FROM `{table_name}` "
            f"WHERE `{column_name}` IS NOT NULL LIMIT ?;",
            (limit,),
        )
        return [row[0] for row in cursor.fetchall()]
    except Exception:
        return []


def _get_sample_rows(cursor, table_name, limit=SAMPLE_ROWS):
    """Get a few sample rows from a table."""
    try:
        cursor.execute(f"SELECT * FROM `{table_name}` LIMIT ?;", (limit,))
        return cursor.fetchall()
    except Exception:
        return []


def extract_schema(schema_id: str) -> str:
    """
    Extract schema from sqlite/<schema_id>.sqlite and return formatted string
    following m_schema_example.txt format.
    """
    db_path = os.path.join(SQLITE_DIR, f"{schema_id}.sqlite")
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Database not found: {db_path}")

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    lines = []
    lines.append(f"## [DB_ID] {schema_id}")
    lines.append("### Schema")

    tables = _get_tables(cursor)
    all_foreign_keys = []

    for table_name in tables:
        lines.append(f"# Table: {table_name}")
        lines.append("")

        columns = _get_columns(cursor, table_name)
        col_lines = []
        for col in columns:
            cid, col_name, col_type, notnull, dflt_value, is_pk = col
            col_type = col_type if col_type else "TEXT"
            parts = [f"{col_name}:{col_type}"]
            if is_pk:
                parts.append("Primary Key")
            examples = _get_sample_values(cursor, table_name, col_name)
            if examples:
                formatted = [str(v) for v in examples]
                parts.append(f"Examples: [{', '.join(formatted)}]")
            col_lines.append(f"({', '.join(parts)})")

        lines.append(",\n".join(col_lines))
        lines.append("")

        # Sample rows
        sample_rows = _get_sample_rows(cursor, table_name)
        if sample_rows:
            lines.append(f"Sample rows from: `{table_name}`")
            for row in sample_rows:
                row_str = ", ".join(str(v) for v in row)
                lines.append(row_str)
            lines.append("")

        # Collect foreign keys
        fks = _get_foreign_keys(cursor, table_name)
        for fk in fks:
            # fk: id, seq, ref_table, from_col, to_col, ...
            ref_table = fk[2]
            from_col = fk[3]
            to_col = fk[4]
            all_foreign_keys.append(f"{table_name}.{from_col}={ref_table}.{to_col}")

    if all_foreign_keys:
        lines.append("[Foreign keys]")
        for fk in all_foreign_keys:
            lines.append(fk)
        lines.append("")

    conn.close()
    return "\n".join(lines)


# Cache to avoid re-reading the same database multiple times
_schema_cache = {}


def get_schema(schema_id: str) -> str:
    """Get schema string for a schema_id, with caching."""
    if schema_id not in _schema_cache:
        logger.info(f"Extracting schema for: {schema_id}")
        _schema_cache[schema_id] = extract_schema(schema_id)
    return _schema_cache[schema_id]
