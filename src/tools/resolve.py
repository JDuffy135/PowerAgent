"""Exercise-name resolution: canonical names, aliases, and fuzzy fallback."""
from __future__ import annotations

import sqlite3
from difflib import get_close_matches
from typing import Literal

from pydantic import BaseModel

FUZZY_CUTOFF = 0.85


class ResolvedExercise(BaseModel):
    exercise_id: int
    name: str
    tier: Literal["competition", "variation", "accessory"]
    matched_via: Literal["exact", "fuzzy"]


def _normalize(raw: str) -> str:
    return " ".join(raw.strip().lower().split())


def _row_to_resolved(row: sqlite3.Row, matched_via: Literal["exact", "fuzzy"]) -> ResolvedExercise:
    return ResolvedExercise(
        exercise_id=row["exercise_id"],
        name=row["name"],
        tier=row["tier"],
        matched_via=matched_via,
    )


def resolve_exercise(conn: sqlite3.Connection, raw_name: str) -> ResolvedExercise | None:
    """Resolve a raw log string to a canonical exercise.

    Order: exact alias match -> fuzzy match against aliases + canonical names -> None.
    """
    normalized = _normalize(raw_name)
    if not normalized:
        return None

    exact = conn.execute(
        """
        SELECT e.exercise_id, e.name, e.tier
        FROM exercise_alias a
        JOIN exercise e ON e.exercise_id = a.exercise_id
        WHERE a.alias = ?
        """,
        (normalized,),
    ).fetchone()
    if exact is not None:
        return _row_to_resolved(exact, "exact")

    # Also allow the canonical name itself to be an "exact" hit.
    exact_canonical = conn.execute(
        "SELECT exercise_id, name, tier FROM exercise WHERE lower(name) = ?",
        (normalized,),
    ).fetchone()
    if exact_canonical is not None:
        return _row_to_resolved(exact_canonical, "exact")

    # Fuzzy fallback: candidate strings are all aliases + all canonical names.
    candidates: dict[str, int] = {}
    for row in conn.execute("SELECT alias, exercise_id FROM exercise_alias"):
        candidates[row["alias"]] = row["exercise_id"]
    for row in conn.execute("SELECT exercise_id, name FROM exercise"):
        candidates[_normalize(row["name"])] = row["exercise_id"]

    matches = get_close_matches(normalized, candidates.keys(), n=5, cutoff=FUZZY_CUTOFF)
    if not matches:
        return None

    matched_exercise_ids = {candidates[m] for m in matches}
    if len(matched_exercise_ids) != 1:
        return None  # ambiguous fuzzy hit across multiple exercises

    exercise_id = matched_exercise_ids.pop()
    row = conn.execute(
        "SELECT exercise_id, name, tier FROM exercise WHERE exercise_id = ?",
        (exercise_id,),
    ).fetchone()
    return _row_to_resolved(row, "fuzzy")


def add_exercise(
    conn: sqlite3.Connection,
    name: str,
    tier: Literal["competition", "variation", "accessory"],
    muscle_group: str | None,
    aliases: list[str],
    commit: bool = True,
) -> int:
    """Insert a new exercise plus its aliases. Returns the new exercise_id.

    Pass ``commit=False`` to enlist these inserts in a caller-managed
    transaction (e.g. the transactional ingest commit path, which must be able
    to roll every insert back together on a mid-commit failure).
    """
    cur = conn.execute(
        "INSERT INTO exercise (name, tier, muscle_group) VALUES (?, ?, ?)",
        (name, tier, muscle_group),
    )
    exercise_id = cur.lastrowid
    for alias in aliases + [name]:
        conn.execute(
            "INSERT OR IGNORE INTO exercise_alias (alias, exercise_id) VALUES (?, ?)",
            (_normalize(alias), exercise_id),
        )
    if commit:
        conn.commit()
    return exercise_id
