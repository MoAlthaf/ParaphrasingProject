import sqlite3
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Any, Dict
from contextlib import closing


def validate_db_path(db_path: Path) -> None:
    """
    Raise error if the SQLite database path is invalid.
    """
    if not db_path.exists():
        raise FileNotFoundError(f"Database file '{db_path}' not found.")


def connect_to_db(db_path: Path) -> sqlite3.Connection:
    """
    Create a SQLite connection with automatic closing support.

    Returns:
        sqlite3.Connection: An open database connection.
    """
    validate_db_path(db_path)
    return sqlite3.connect(db_path)


def get_table_names(conn: sqlite3.Connection) -> List[str]:
    """
    Return all table names in the connected database.
    """
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    return [row[0] for row in cursor.fetchall()]


def table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
    """
    Check if a table exists in the database.
    """
    cursor = conn.cursor()
    cursor.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?;", (table_name,)
    )
    return cursor.fetchone() is not None


def extract_schema(db_path: Path) -> str:
    """
    Extract schema information from the database (tables and columns).
    """
    schema_lines = [f"Database name: {db_path.name}"]

    with closing(connect_to_db(db_path)) as conn:
        for table_name in get_table_names(conn):
            cursor = conn.cursor()
            cursor.execute(f"PRAGMA table_info({table_name});")
            columns = cursor.fetchall()
            col_names = [col[1] for col in columns]
            schema_lines.append(f"Table: {table_name} ({', '.join(col_names)})")

    return "\n".join(schema_lines)


def get_sample_rows(
    db_path: Path, limit: int = 3
) -> Dict[str, Tuple[List[str], List[Tuple[Any, ...]]]]:
    """
    Retrieve sample rows (and column names) from all tables.
    """
    sample_data = {}

    with closing(connect_to_db(db_path)) as conn:
        for table in get_table_names(conn):
            try:
                cursor = conn.cursor()
                cursor.execute(f"SELECT * FROM {table} LIMIT {limit};")
                rows = cursor.fetchall()
                col_names = [desc[0] for desc in cursor.description]
                sample_data[table] = (col_names, rows)
            except sqlite3.Error as e:
                print(f"Skipping table '{table}' due to error: {e}")

    return sample_data


def run_query(db_path: Path, query: str) -> pd.DataFrame:
    """
    Run a SQL query on the SQLite DB and return results as a DataFrame.

    Raises:
        ValueError if the query fails.
    """
    with closing(connect_to_db(db_path)) as conn:
        try:
            return pd.read_sql_query(query, conn)
        except Exception as e:
            raise ValueError(f"Query failed: {query}. Error: {e}")
