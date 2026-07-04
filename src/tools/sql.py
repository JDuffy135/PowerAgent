"""Gated read-only text-to-SQL escape hatch (ARCHITECTURE.md §5.2).

`run_readonly_sql` is the last resort for the ANALYZE ReAct loop when none of
the typed query tools fit. It only ever runs a single, validated `SELECT`
statement against a read-only connection, with a row cap and a best-effort
wall-clock timeout.
"""
from __future__ import annotations

import signal
import sqlite3
import threading

import sqlglot
from pydantic import BaseModel
from sqlglot import exp

DEFAULT_MAX_ROWS = 200
DEFAULT_TIMEOUT_S = 5.0


class ReadonlySQLError(Exception):
    """Raised when a query fails validation (not a single read-only SELECT)."""


class ReadonlySQLTimeout(Exception):
    """Raised when a query exceeds the wall-clock timeout."""


class ReadonlySQLResult(BaseModel):
    columns: list[str]
    rows: list[dict]
    truncated: bool


def _validate_select(query: str) -> None:
    try:
        statements = sqlglot.parse(query, dialect="sqlite")
    except sqlglot.errors.ParseError as exc:
        raise ReadonlySQLError(f"Could not parse SQL: {exc}") from exc

    statements = [s for s in statements if s is not None]
    if len(statements) != 1:
        raise ReadonlySQLError("Only a single SQL statement is allowed")

    statement = statements[0]
    if not isinstance(statement, exp.Select):
        raise ReadonlySQLError("Only SELECT statements are allowed")


class _timeout_guard:
    """Best-effort wall-clock timeout via SIGALRM. No-op on platforms without
    it, and outside the main thread (signal handlers can only be installed
    there -- Streamlit reruns execute in a worker thread)."""

    def __init__(self, seconds: float):
        self.seconds = seconds
        self._supported = (
            hasattr(signal, "SIGALRM")
            and threading.current_thread() is threading.main_thread()
        )

    def __enter__(self):
        if self._supported and self.seconds > 0:
            def _handler(signum, frame):
                raise ReadonlySQLTimeout(f"Query exceeded {self.seconds}s timeout")

            self._previous = signal.signal(signal.SIGALRM, _handler)
            signal.setitimer(signal.ITIMER_REAL, self.seconds)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._supported and self.seconds > 0:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, self._previous)
        return False


def run_readonly_sql(
    conn: sqlite3.Connection,
    query: str,
    max_rows: int = DEFAULT_MAX_ROWS,
    timeout_s: float = DEFAULT_TIMEOUT_S,
) -> ReadonlySQLResult:
    """Run a validated, read-only SELECT against `conn`.

    Validation: `sqlglot`-parsed as exactly one `SELECT` statement (rejects
    INSERT/UPDATE/DELETE/PRAGMA/ATTACH/multi-statement input). Execution:
    `PRAGMA query_only=ON` for the duration of the call (restored after, even
    on error), row-capped via an outer `LIMIT`, and a best-effort timeout.
    """
    _validate_select(query)

    wrapped = f"SELECT * FROM ({query.rstrip(';')}) AS _readonly_sql LIMIT ?"

    conn.execute("PRAGMA query_only = ON")
    try:
        with _timeout_guard(timeout_s):
            cursor = conn.execute(wrapped, (max_rows + 1,))
            columns = [d[0] for d in cursor.description] if cursor.description else []
            fetched = cursor.fetchall()
    finally:
        conn.execute("PRAGMA query_only = OFF")

    truncated = len(fetched) > max_rows
    rows = [dict(r) for r in fetched[:max_rows]]
    return ReadonlySQLResult(columns=columns, rows=rows, truncated=truncated)
