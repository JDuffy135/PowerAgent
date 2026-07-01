"""Typed query tools over the training DB. All weights in lb; no unit conversion here.

Usage examples: run `python -m src.tools.queries` (see __main__ block at the bottom).
"""
from __future__ import annotations

import sqlite3
from datetime import date as date_cls
from typing import Literal

from pydantic import BaseModel

from src.tools.resolve import resolve_exercise

# Draft programs are excluded from every tool below. Sessions with a NULL
# block_id are real, unattached logs and are always included.
_DRAFT_EXCLUSION_JOIN = """
    LEFT JOIN block b ON b.block_id = s.block_id
    LEFT JOIN program p ON p.program_id = b.program_id
"""
_DRAFT_EXCLUSION_WHERE = "(s.block_id IS NULL OR p.status != 'draft')"


class ExerciseNotFound(Exception):
    """Raised when an exercise name can't be resolved to a canonical exercise."""

    def __init__(self, raw_name: str):
        self.raw_name = raw_name
        super().__init__(f"Could not resolve exercise: {raw_name!r}")


def _resolve_or_raise(conn: sqlite3.Connection, exercise: str):
    resolved = resolve_exercise(conn, exercise)
    if resolved is None:
        raise ExerciseNotFound(exercise)
    return resolved


# --------------------------------------------------------------------------
# get_best_set
# --------------------------------------------------------------------------

class BestSetResult(BaseModel):
    exercise: str
    weight_lb: float
    reps: int
    date: str
    session_id: int
    set_id: int
    rpe: float | None = None


def get_best_set(
    conn: sqlite3.Connection,
    exercise: str,
    date_from: str,
    date_to: str,
    min_reps: int = 1,
) -> BestSetResult | None:
    """Heaviest weight_lb with reps >= min_reps in [date_from, date_to] (inclusive).

    Tie-break: more reps wins, then later date.
    """
    resolved = _resolve_or_raise(conn, exercise)

    row = conn.execute(
        f"""
        SELECT ls.set_id, ls.weight_lb, ls.reps, ls.rpe, s.session_id, s.date
        FROM lift_set ls
        JOIN session s ON s.session_id = ls.session_id
        {_DRAFT_EXCLUSION_JOIN}
        WHERE ls.exercise_id = ?
          AND s.date BETWEEN ? AND ?
          AND ls.reps >= ?
          AND ls.weight_lb IS NOT NULL
          AND {_DRAFT_EXCLUSION_WHERE}
        ORDER BY ls.weight_lb DESC, ls.reps DESC, s.date DESC
        LIMIT 1
        """,
        (resolved.exercise_id, date_from, date_to, min_reps),
    ).fetchone()

    if row is None:
        return None

    return BestSetResult(
        exercise=resolved.name,
        weight_lb=row["weight_lb"],
        reps=row["reps"],
        date=row["date"],
        session_id=row["session_id"],
        set_id=row["set_id"],
        rpe=row["rpe"],
    )


# --------------------------------------------------------------------------
# get_lifts
# --------------------------------------------------------------------------

class SetDetail(BaseModel):
    set_id: int
    set_index: int
    weight_lb: float | None
    reps: int | None
    rpe: float | None
    is_top_set: bool
    is_failed: bool
    raw_text: str | None


class SessionLifts(BaseModel):
    session_id: int
    date: str
    sets: list[SetDetail]


def get_lifts(
    conn: sqlite3.Connection,
    exercise: str,
    date_from: str,
    date_to: str,
    top_sets_only: bool = False,
) -> list[SessionLifts]:
    """All sets for an exercise in the window, grouped by session, chronological."""
    resolved = _resolve_or_raise(conn, exercise)

    top_set_filter = "AND ls.is_top_set = 1" if top_sets_only else ""

    rows = conn.execute(
        f"""
        SELECT s.session_id, s.date, ls.set_id, ls.set_index, ls.weight_lb, ls.reps,
               ls.rpe, ls.is_top_set, ls.is_failed, ls.raw_text
        FROM lift_set ls
        JOIN session s ON s.session_id = ls.session_id
        {_DRAFT_EXCLUSION_JOIN}
        WHERE ls.exercise_id = ?
          AND s.date BETWEEN ? AND ?
          AND {_DRAFT_EXCLUSION_WHERE}
          {top_set_filter}
        ORDER BY s.date ASC, s.session_id ASC, ls.set_index ASC
        """,
        (resolved.exercise_id, date_from, date_to),
    ).fetchall()

    sessions: dict[int, SessionLifts] = {}
    for row in rows:
        session_id = row["session_id"]
        if session_id not in sessions:
            sessions[session_id] = SessionLifts(session_id=session_id, date=row["date"], sets=[])
        sessions[session_id].sets.append(
            SetDetail(
                set_id=row["set_id"],
                set_index=row["set_index"],
                weight_lb=row["weight_lb"],
                reps=row["reps"],
                rpe=row["rpe"],
                is_top_set=bool(row["is_top_set"]),
                is_failed=bool(row["is_failed"]),
                raw_text=row["raw_text"],
            )
        )

    return sorted(sessions.values(), key=lambda sl: (sl.date, sl.session_id))


# --------------------------------------------------------------------------
# get_e1rm_trend
# --------------------------------------------------------------------------

class E1RMPoint(BaseModel):
    bucket: str
    e1rm: float
    source_weight_lb: float
    source_reps: int
    source_date: str


def _epley(weight: float, reps: int) -> float:
    return weight * (1 + reps / 30)


def _iso_week_bucket(iso_date: str) -> str:
    y, m, d = (int(part) for part in iso_date.split("-"))
    iso_year, iso_week, _ = date_cls(y, m, d).isocalendar()
    return f"{iso_year}-W{iso_week:02d}"


def get_e1rm_trend(
    conn: sqlite3.Connection,
    exercise: str,
    date_from: str,
    date_to: str,
    by: Literal["week", "block"] = "week",
) -> list[E1RMPoint]:
    """Epley e1RM per qualifying set (reps <= 10, is_failed = 0), max per bucket."""
    resolved = _resolve_or_raise(conn, exercise)

    rows = conn.execute(
        f"""
        SELECT ls.weight_lb, ls.reps, s.date, s.block_id, b.name AS block_name
        FROM lift_set ls
        JOIN session s ON s.session_id = ls.session_id
        {_DRAFT_EXCLUSION_JOIN}
        WHERE ls.exercise_id = ?
          AND s.date BETWEEN ? AND ?
          AND ls.reps <= 10
          AND ls.is_failed = 0
          AND ls.weight_lb IS NOT NULL
          AND {_DRAFT_EXCLUSION_WHERE}
        """,
        (resolved.exercise_id, date_from, date_to),
    ).fetchall()

    best_per_bucket: dict[str, tuple[float, float, int, str]] = {}
    for row in rows:
        if by == "week":
            bucket = _iso_week_bucket(row["date"])
        else:
            bucket = row["block_name"] if row["block_name"] else "unattached"

        e1rm = _epley(row["weight_lb"], row["reps"])
        current = best_per_bucket.get(bucket)
        if current is None or e1rm > current[0]:
            best_per_bucket[bucket] = (e1rm, row["weight_lb"], row["reps"], row["date"])

    points = [
        E1RMPoint(
            bucket=bucket,
            e1rm=round(vals[0], 1),
            source_weight_lb=vals[1],
            source_reps=vals[2],
            source_date=vals[3],
        )
        for bucket, vals in best_per_bucket.items()
    ]
    return sorted(points, key=lambda p: p.bucket)


# --------------------------------------------------------------------------
# get_bodyweight_trend
# --------------------------------------------------------------------------

class BodyweightRow(BaseModel):
    date: str
    weight_lb: float


class BodyweightTrend(BaseModel):
    rows: list[BodyweightRow]
    first: float | None
    last: float | None
    delta: float | None
    min: float | None
    max: float | None


def get_bodyweight_trend(
    conn: sqlite3.Connection, date_from: str, date_to: str
) -> BodyweightTrend:
    rows = conn.execute(
        """
        SELECT date, weight_lb FROM bodyweight
        WHERE date BETWEEN ? AND ?
        ORDER BY date ASC
        """,
        (date_from, date_to),
    ).fetchall()

    bw_rows = [BodyweightRow(date=r["date"], weight_lb=r["weight_lb"]) for r in rows]

    if not bw_rows:
        return BodyweightTrend(rows=[], first=None, last=None, delta=None, min=None, max=None)

    weights = [r.weight_lb for r in bw_rows]
    first, last = weights[0], weights[-1]
    return BodyweightTrend(
        rows=bw_rows,
        first=first,
        last=last,
        delta=round(last - first, 2),
        min=min(weights),
        max=max(weights),
    )


if __name__ == "__main__":
    from src.db.connection import get_conn

    conn = get_conn("data/training.db")

    print("get_best_set:", get_best_set(conn, "bench press", "2026-03-01", "2026-03-31"))
    print("get_lifts (top sets):", get_lifts(conn, "deadlift", "2026-01-01", "2026-06-30", top_sets_only=True))
    print("get_e1rm_trend:", get_e1rm_trend(conn, "deadlift", "2026-01-01", "2026-06-30", by="week"))
    print("get_bodyweight_trend:", get_bodyweight_trend(conn, "2026-01-01", "2026-06-30"))
