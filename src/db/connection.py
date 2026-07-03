"""SQLite connection helpers: WAL mode, foreign keys, row factory, idempotent init."""
from __future__ import annotations

import sqlite3
from pathlib import Path

SCHEMA_PATH = Path(__file__).parent / "schema.sql"


def get_conn(db_path: str | Path, *, check_same_thread: bool = True) -> sqlite3.Connection:
    """Open a connection with WAL mode, foreign keys, and dict-like row access.

    `check_same_thread=False` is for the Streamlit UI, whose script reruns hop
    threads; access there is still serialized (one rerun at a time per session).
    """
    conn = sqlite3.connect(db_path, check_same_thread=check_same_thread)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    """Execute schema.sql. Idempotent: all statements use CREATE TABLE/INDEX IF NOT EXISTS."""
    schema_sql = SCHEMA_PATH.read_text()
    conn.executescript(schema_sql)
    conn.commit()
