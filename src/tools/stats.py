"""Single-row stat writes for UPDATE_STATS (ARCHITECTURE.md §4.2).

The agent parses a message like "bodyweight was 146 this morning" or "hit a 405
deadlift PR" into one of these inserts. **[DECISION]** Stage 6 scope is
`bodyweight` + `pr` only; injury/measurement phrasing is more varied and is left
for later. Weights arrive already normalized to lb (the parser converts kg, like
the ingest extractor does).

These are the durable writes behind UPDATE_STATS's confirm-before-write
interrupt: the graph never calls them until the user approves.
"""
from __future__ import annotations

import sqlite3


def insert_bodyweight(
    conn: sqlite3.Connection, date: str, weight_lb: float, note: str | None = None
) -> int:
    """Insert one bodyweight row (lb). Returns the new `bw_id`."""
    bw_id = conn.execute(
        "INSERT INTO bodyweight (date, weight_lb, note) VALUES (?, ?, ?)",
        (date, weight_lb, note),
    ).lastrowid
    conn.commit()
    return bw_id


def insert_pr(
    conn: sqlite3.Connection,
    date: str,
    exercise_id: int,
    weight_lb: float,
    reps: int,
    context: str | None = None,
    session_id: int | None = None,
) -> int:
    """Insert one manually-reported PR row (lb). Returns the new `pr_id`.

    `exercise_id` must already be resolved by the caller (UPDATE_STATS resolves
    it at parse time and refuses to confirm an unknown exercise).
    """
    pr_id = conn.execute(
        """
        INSERT INTO pr (date, session_id, exercise_id, weight_lb, reps, context)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (date, session_id, exercise_id, weight_lb, reps, context),
    ).lastrowid
    conn.commit()
    return pr_id
