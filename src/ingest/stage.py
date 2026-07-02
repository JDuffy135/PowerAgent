"""HITL staging: persist a `ParsedBatch` as a `pending_review` audit-trail row.

This is the bridge from "we parsed a `ParsedBatch`" to "there's a durable record
awaiting the user's approval". `stage_batch` writes exactly one `ingest_batch`
row (status `pending_review`) holding the serialized batch JSON -- it does NOT
touch `lift_set`/`session`/Chroma. Those writes happen only after approval, in
`commit.py` (ARCHITECTURE.md §4.4, §5.3).

`get_pending_batch` rehydrates the stored JSON back into a `ParsedBatch` so the
review renderer and the commit path can work with typed objects, not raw text.
"""
from __future__ import annotations

import sqlite3
from datetime import datetime, timezone

from src.ingest.models import ParsedBatch


class BatchNotFound(Exception):
    """Raised when no `ingest_batch` row exists for a given batch_id."""

    def __init__(self, batch_id: int):
        self.batch_id = batch_id
        super().__init__(f"No ingest_batch row with batch_id={batch_id}")


class BatchNotEditable(Exception):
    """Raised when `update_batch` targets a batch that is no longer pending review."""

    def __init__(self, batch_id: int, status: str):
        self.batch_id = batch_id
        self.status = status
        super().__init__(f"Cannot edit batch {batch_id}: status is {status!r}")


def stage_batch(
    conn: sqlite3.Connection,
    parsed: ParsedBatch,
    source_file: str | None = None,
) -> int:
    """Serialize `parsed` and write one `pending_review` `ingest_batch` row.

    Returns the new `batch_id`. This is the audit-trail row only -- no training
    data is written until `commit_batch` is called after HITL approval.
    """
    created_at = datetime.now(timezone.utc).isoformat()
    parsed_json = parsed.model_dump_json()
    cur = conn.execute(
        """
        INSERT INTO ingest_batch (created_at, source_file, status, parsed_json)
        VALUES (?, ?, 'pending_review', ?)
        """,
        (created_at, source_file, parsed_json),
    )
    conn.commit()
    return cur.lastrowid


def get_pending_batch(conn: sqlite3.Connection, batch_id: int) -> ParsedBatch:
    """Rehydrate the stored batch JSON back into a `ParsedBatch`.

    Works regardless of the row's current status (the commit path reads a
    `pending_review` row; callers may also re-inspect a `committed`/`rejected`
    one for the audit trail). Raises `BatchNotFound` if the id is unknown.
    """
    row = conn.execute(
        "SELECT parsed_json FROM ingest_batch WHERE batch_id = ?",
        (batch_id,),
    ).fetchone()
    if row is None:
        raise BatchNotFound(batch_id)
    return ParsedBatch.model_validate_json(row["parsed_json"])


def update_batch(conn: sqlite3.Connection, batch_id: int, parsed: ParsedBatch) -> None:
    """Overwrite a *pending* batch's `parsed_json` with a corrected `ParsedBatch`.

    This is the HITL correction pass's write path: the user's free-text edits are
    applied to the staged JSON, never to training tables. Only `pending_review`
    rows may be edited -- a committed/rejected batch is a sealed audit record
    (`BatchNotEditable`).
    """
    row = conn.execute(
        "SELECT status FROM ingest_batch WHERE batch_id = ?", (batch_id,)
    ).fetchone()
    if row is None:
        raise BatchNotFound(batch_id)
    if row["status"] != "pending_review":
        raise BatchNotEditable(batch_id, row["status"])

    conn.execute(
        "UPDATE ingest_batch SET parsed_json = ? WHERE batch_id = ?",
        (parsed.model_dump_json(), batch_id),
    )
    conn.commit()
