"""Dev/ops quality-of-life helpers (Stage 9): direct table CRUD, DB backup,
and the ingest-batch audit browser.

This is the **developer maintenance surface**, deliberately outside the agent's
HITL ingest flow: every call maps to an explicit user action in the Dev Tools
tab (edit a cell, click delete), so the action itself is the approval. It is
NOT exposed to the LLM as a tool -- the agent's only write paths remain the
interrupt-gated ones.

Safety rails instead of ceremony:
- table + column names are validated against an allowlist / PRAGMA before being
  interpolated (values always go through `?` placeholders);
- `ingest_batch` is browseable but not editable (sealed audit trail);
- FK violations surface as `AdminError` with the SQLite message, not a stack trace.
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel

# table -> primary-key column. These are the editable domain tables; the
# ingest_batch audit trail is intentionally absent (read-only, see list_batches).
EDITABLE_TABLES: dict[str, str] = {
    "program": "program_id",
    "block": "block_id",
    "exercise": "exercise_id",
    "exercise_alias": "alias",
    "session": "session_id",
    "lift_set": "set_id",
    "programmed_slot": "slot_id",
    "cardio": "cardio_id",
    "bodyweight": "bw_id",
    "pr": "pr_id",
    "injury": "injury_id",
    "measurement": "m_id",
}


class AdminError(Exception):
    """Bad table/column, unknown row, or a constraint violation."""


def _check_table(table: str) -> str:
    if table not in EDITABLE_TABLES:
        raise AdminError(f"Table {table!r} is not editable (allowed: {sorted(EDITABLE_TABLES)})")
    return EDITABLE_TABLES[table]


def table_columns(conn: sqlite3.Connection, table: str) -> list[str]:
    """Column names for an editable table, in schema order."""
    _check_table(table)
    return [r["name"] for r in conn.execute(f"PRAGMA table_info({table})")]


def fetch_table(
    conn: sqlite3.Connection, table: str, *, limit: int = 500, offset: int = 0
) -> list[dict]:
    """Rows as dicts, newest-first by primary key (rowid order ~ insert order)."""
    pk = _check_table(table)
    rows = conn.execute(
        f"SELECT * FROM {table} ORDER BY {pk} DESC LIMIT ? OFFSET ?",
        (limit, offset),
    ).fetchall()
    return [dict(r) for r in rows]


def count_rows(conn: sqlite3.Connection, table: str) -> int:
    _check_table(table)
    return conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]


def _check_columns(conn: sqlite3.Connection, table: str, values: dict) -> None:
    known = set(table_columns(conn, table))
    bad = set(values) - known
    if bad:
        raise AdminError(f"Unknown column(s) for {table}: {sorted(bad)}")


def insert_row(conn: sqlite3.Connection, table: str, values: dict) -> int | str:
    """Insert one row; returns the new primary key. The pk column may be omitted
    (or None) for INTEGER PRIMARY KEY tables to autoassign."""
    pk = _check_table(table)
    values = {k: v for k, v in values.items() if not (k == pk and v is None)}
    if not values:
        raise AdminError("No values to insert")
    _check_columns(conn, table, values)
    cols = list(values)
    placeholders = ", ".join("?" for _ in cols)
    try:
        cur = conn.execute(
            f"INSERT INTO {table} ({', '.join(cols)}) VALUES ({placeholders})",
            tuple(values[c] for c in cols),
        )
        conn.commit()
    except sqlite3.Error as exc:
        conn.rollback()
        raise AdminError(f"Insert into {table} failed: {exc}") from exc
    return values.get(pk, cur.lastrowid)


def update_row(conn: sqlite3.Connection, table: str, pk_value, values: dict) -> None:
    """Update one row by primary key. Refuses unknown rows/columns."""
    pk = _check_table(table)
    values = {k: v for k, v in values.items() if k != pk}
    if not values:
        raise AdminError("No values to update")
    _check_columns(conn, table, values)
    assignments = ", ".join(f"{c} = ?" for c in values)
    try:
        cur = conn.execute(
            f"UPDATE {table} SET {assignments} WHERE {pk} = ?",
            (*values.values(), pk_value),
        )
        conn.commit()
    except sqlite3.Error as exc:
        conn.rollback()
        raise AdminError(f"Update of {table} failed: {exc}") from exc
    if cur.rowcount == 0:
        raise AdminError(f"No {table} row with {pk}={pk_value!r}")


def delete_row(conn: sqlite3.Connection, table: str, pk_value) -> None:
    """Delete one row by primary key. FK violations (e.g. deleting a session
    that still has sets) surface as `AdminError` -- delete children first."""
    pk = _check_table(table)
    try:
        cur = conn.execute(f"DELETE FROM {table} WHERE {pk} = ?", (pk_value,))
        conn.commit()
    except sqlite3.Error as exc:
        conn.rollback()
        raise AdminError(f"Delete from {table} failed: {exc}") from exc
    if cur.rowcount == 0:
        raise AdminError(f"No {table} row with {pk}={pk_value!r}")


# ---------------------------------------------------------------------------
# DB backup
# ---------------------------------------------------------------------------

def backup_db(conn: sqlite3.Connection, dest_dir: str | Path, *, prefix: str = "training") -> Path:
    """Copy the live DB into `dest_dir/<prefix>-backup-<UTC timestamp>.db` using
    SQLite's online backup API (consistent snapshot, WAL-safe)."""
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    dest = dest_dir / f"{prefix}-backup-{stamp}.db"
    target = sqlite3.connect(dest)
    try:
        with target:
            conn.backup(target)
    finally:
        target.close()
    return dest


# ---------------------------------------------------------------------------
# Ingest-batch audit browser (read-only)
# ---------------------------------------------------------------------------

class BatchSummary(BaseModel):
    batch_id: int
    created_at: str
    source_file: str | None
    status: str
    n_sessions: int
    n_sets: int


def list_batches(
    conn: sqlite3.Connection, *, status: str | None = None, limit: int = 100
) -> list[BatchSummary]:
    """Ingest audit trail, newest first. `status` filters to
    pending_review/committed/rejected."""
    where = "WHERE status = ?" if status else ""
    params: tuple = (status, limit) if status else (limit,)
    rows = conn.execute(
        f"""
        SELECT batch_id, created_at, source_file, status, parsed_json
        FROM ingest_batch {where}
        ORDER BY batch_id DESC LIMIT ?
        """,
        params,
    ).fetchall()

    summaries = []
    for r in rows:
        n_sessions = n_sets = 0
        if r["parsed_json"]:
            try:
                payload = json.loads(r["parsed_json"])
                sessions = payload.get("sessions", [])
                n_sessions = len(sessions)
                n_sets = sum(len(s.get("sets", [])) for s in sessions)
            except (json.JSONDecodeError, AttributeError):
                pass  # malformed audit JSON still lists, with zero counts
        summaries.append(
            BatchSummary(
                batch_id=r["batch_id"],
                created_at=r["created_at"],
                source_file=r["source_file"],
                status=r["status"],
                n_sessions=n_sessions,
                n_sets=n_sets,
            )
        )
    return summaries


def get_batch_json(conn: sqlite3.Connection, batch_id: int) -> str:
    """The stored parsed_json for one batch (any status), pretty-printed for the
    audit viewer. Raises AdminError if the id is unknown."""
    row = conn.execute(
        "SELECT parsed_json FROM ingest_batch WHERE batch_id = ?", (batch_id,)
    ).fetchone()
    if row is None:
        raise AdminError(f"No ingest_batch row with batch_id={batch_id}")
    if not row["parsed_json"]:
        return "{}"
    try:
        return json.dumps(json.loads(row["parsed_json"]), indent=2)
    except json.JSONDecodeError:
        return row["parsed_json"]
