"""Program/block organizer ops (Stage 9, committed to during Stage 5's
block-assignment decision).

Block assignment at ingest time never has to be perfect: everything here exists
so a wrong or missing assignment is a quick fix later, not a permanent mistake.
Pure SQL over the live connection -- no LLM, no Chroma. The Streamlit organizer
tab is a thin veneer over these functions, which keeps them testable.

These are *user-invoked* administrative edits (the user clicks a button per
change), so they sit outside the ingest HITL interrupt flow by design -- the
click IS the approval. Nothing here touches `lift_set` contents; sessions are
only re-pointed, never rewritten.
"""
from __future__ import annotations

import sqlite3

from pydantic import BaseModel


class OrganizeError(Exception):
    """A precondition failed (unknown id, self-merge, wrong status...)."""


class ProgramRow(BaseModel):
    program_id: int
    name: str
    status: str
    start_date: str | None
    end_date: str | None
    block_count: int
    session_count: int


class BlockRow(BaseModel):
    block_id: int
    program_id: int
    program_name: str
    name: str
    focus: str | None
    week_count: int | None
    start_date: str | None
    end_date: str | None
    session_count: int


class SessionRow(BaseModel):
    session_id: int
    date: str
    block_id: int | None
    block_name: str | None
    day_label: str | None
    session_type: str
    set_count: int


# ---------------------------------------------------------------------------
# Listings (organizer tab data source)
# ---------------------------------------------------------------------------

def list_programs(conn: sqlite3.Connection) -> list[ProgramRow]:
    """All programs (drafts included -- this is the organizer, not analysis)."""
    rows = conn.execute(
        """
        SELECT p.program_id, p.name, p.status, p.start_date, p.end_date,
               COUNT(DISTINCT b.block_id) AS block_count,
               COUNT(DISTINCT s.session_id) AS session_count
        FROM program p
        LEFT JOIN block b ON b.program_id = p.program_id
        LEFT JOIN session s ON s.block_id = b.block_id
        GROUP BY p.program_id
        ORDER BY p.start_date IS NULL, p.start_date, p.program_id
        """
    ).fetchall()
    return [ProgramRow(**dict(r)) for r in rows]


def list_blocks(conn: sqlite3.Connection, program_id: int | None = None) -> list[BlockRow]:
    """Blocks (optionally scoped to one program), with session counts."""
    where = "WHERE b.program_id = ?" if program_id is not None else ""
    params = (program_id,) if program_id is not None else ()
    rows = conn.execute(
        f"""
        SELECT b.block_id, b.program_id, p.name AS program_name, b.name, b.focus,
               b.week_count, b.start_date, b.end_date,
               COUNT(s.session_id) AS session_count
        FROM block b
        JOIN program p ON p.program_id = b.program_id
        LEFT JOIN session s ON s.block_id = b.block_id
        {where}
        GROUP BY b.block_id
        ORDER BY b.start_date IS NULL, b.start_date, b.block_id
        """,
        params,
    ).fetchall()
    return [BlockRow(**dict(r)) for r in rows]


def list_sessions(
    conn: sqlite3.Connection,
    block_id: int | None = None,
    *,
    unattached_only: bool = False,
    limit: int = 200,
) -> list[SessionRow]:
    """Sessions for the reattach picker: one block's, unattached-only, or all
    (most recent first)."""
    clauses, params = [], []
    if unattached_only:
        clauses.append("s.block_id IS NULL")
    elif block_id is not None:
        clauses.append("s.block_id = ?")
        params.append(block_id)
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    rows = conn.execute(
        f"""
        SELECT s.session_id, s.date, s.block_id, b.name AS block_name,
               s.day_label, s.session_type,
               COUNT(ls.set_id) AS set_count
        FROM session s
        LEFT JOIN block b ON b.block_id = s.block_id
        LEFT JOIN lift_set ls ON ls.session_id = s.session_id
        {where}
        GROUP BY s.session_id
        ORDER BY s.date DESC, s.session_id DESC
        LIMIT ?
        """,
        (*params, limit),
    ).fetchall()
    return [SessionRow(**dict(r)) for r in rows]


# ---------------------------------------------------------------------------
# Guards
# ---------------------------------------------------------------------------

def _require(conn: sqlite3.Connection, table: str, pk: str, id_: int) -> sqlite3.Row:
    row = conn.execute(f"SELECT * FROM {table} WHERE {pk} = ?", (id_,)).fetchone()
    if row is None:
        raise OrganizeError(f"No {table} with {pk}={id_}")
    return row


# ---------------------------------------------------------------------------
# Mutations
# ---------------------------------------------------------------------------

def rename_program(conn: sqlite3.Connection, program_id: int, name: str) -> None:
    name = name.strip()
    if not name:
        raise OrganizeError("Program name cannot be empty")
    _require(conn, "program", "program_id", program_id)
    conn.execute("UPDATE program SET name = ? WHERE program_id = ?", (name, program_id))
    conn.commit()


def rename_block(conn: sqlite3.Connection, block_id: int, name: str) -> None:
    name = name.strip()
    if not name:
        raise OrganizeError("Block name cannot be empty")
    _require(conn, "block", "block_id", block_id)
    conn.execute("UPDATE block SET name = ? WHERE block_id = ?", (name, block_id))
    conn.commit()


def reattach_session(
    conn: sqlite3.Connection, session_id: int, block_id: int | None
) -> None:
    """Point a session at a different block (or detach it with `None`). The
    session's sets/cardio/notes ride along untouched -- only the FK moves."""
    _require(conn, "session", "session_id", session_id)
    if block_id is not None:
        _require(conn, "block", "block_id", block_id)
    conn.execute(
        "UPDATE session SET block_id = ? WHERE session_id = ?", (block_id, session_id)
    )
    conn.commit()


def move_block(conn: sqlite3.Connection, block_id: int, program_id: int) -> None:
    """Reattach a block (and, implicitly, all its sessions) to another program."""
    _require(conn, "block", "block_id", block_id)
    _require(conn, "program", "program_id", program_id)
    conn.execute(
        "UPDATE block SET program_id = ? WHERE block_id = ?", (program_id, block_id)
    )
    conn.commit()


def merge_blocks(conn: sqlite3.Connection, src_block_id: int, dst_block_id: int) -> int:
    """Fold `src` into `dst`: move sessions + programmed slots, then delete the
    src block row. Returns the number of sessions moved. The dst block keeps its
    own name/focus/dates; extend them by hand afterwards if needed."""
    if src_block_id == dst_block_id:
        raise OrganizeError("Cannot merge a block into itself")
    _require(conn, "block", "block_id", src_block_id)
    _require(conn, "block", "block_id", dst_block_id)
    try:
        moved = conn.execute(
            "UPDATE session SET block_id = ? WHERE block_id = ?",
            (dst_block_id, src_block_id),
        ).rowcount
        conn.execute(
            "UPDATE programmed_slot SET block_id = ? WHERE block_id = ?",
            (dst_block_id, src_block_id),
        )
        conn.execute("DELETE FROM block WHERE block_id = ?", (src_block_id,))
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    return moved


def merge_programs(
    conn: sqlite3.Connection, src_program_id: int, dst_program_id: int
) -> int:
    """Fold `src` into `dst`: move every block over, then delete the src program
    row. Returns the number of blocks moved. Prose fields (goals/review/notes)
    on src are dropped with the row -- merge those by hand first if they matter."""
    if src_program_id == dst_program_id:
        raise OrganizeError("Cannot merge a program into itself")
    _require(conn, "program", "program_id", src_program_id)
    _require(conn, "program", "program_id", dst_program_id)
    try:
        moved = conn.execute(
            "UPDATE block SET program_id = ? WHERE program_id = ?",
            (dst_program_id, src_program_id),
        ).rowcount
        conn.execute("DELETE FROM program WHERE program_id = ?", (src_program_id,))
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    return moved


def start_draft(
    conn: sqlite3.Connection, program_id: int, start_date: str
) -> None:
    """The "start this draft" flow: flip a `draft` program to `incomplete` and
    stamp its start date, so newly ingested sessions can attach to its blocks
    and it stops being excluded from analysis."""
    row = _require(conn, "program", "program_id", program_id)
    if row["status"] != "draft":
        raise OrganizeError(
            f"Program {program_id} is {row['status']!r}, not a draft"
        )
    conn.execute(
        "UPDATE program SET status = 'incomplete', start_date = ? WHERE program_id = ?",
        (start_date, program_id),
    )
    conn.commit()
