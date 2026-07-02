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
    equipment_note: str | None = None
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
               ls.rpe, ls.is_top_set, ls.is_failed, ls.equipment_note, ls.raw_text
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
                equipment_note=row["equipment_note"],
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


# --------------------------------------------------------------------------
# get_sessions
# --------------------------------------------------------------------------

class SessionSummary(BaseModel):
    session_id: int
    date: str
    block_id: int | None
    week_number: int | None
    day_number: int | None
    day_label: str | None
    duration_min: int | None
    session_type: Literal["lifting", "cardio", "other"]
    raw_note: str | None


def get_sessions(
    conn: sqlite3.Connection,
    date_from: str,
    date_to: str,
    block_id: int | None = None,
    session_type: Literal["lifting", "cardio", "other"] | None = None,
) -> list[SessionSummary]:
    """Sessions in [date_from, date_to], chronological. Draft programs excluded."""
    filters = ["s.date BETWEEN ? AND ?", _DRAFT_EXCLUSION_WHERE]
    params: list = [date_from, date_to]
    if block_id is not None:
        filters.append("s.block_id = ?")
        params.append(block_id)
    if session_type is not None:
        filters.append("s.session_type = ?")
        params.append(session_type)

    rows = conn.execute(
        f"""
        SELECT s.session_id, s.date, s.block_id, s.week_number, s.day_number,
               s.day_label, s.duration_min, s.session_type, s.raw_note
        FROM session s
        {_DRAFT_EXCLUSION_JOIN}
        WHERE {' AND '.join(filters)}
        ORDER BY s.date ASC, s.session_id ASC
        """,
        params,
    ).fetchall()

    return [
        SessionSummary(
            session_id=r["session_id"],
            date=r["date"],
            block_id=r["block_id"],
            week_number=r["week_number"],
            day_number=r["day_number"],
            day_label=r["day_label"],
            duration_min=r["duration_min"],
            session_type=r["session_type"],
            raw_note=r["raw_note"],
        )
        for r in rows
    ]


# --------------------------------------------------------------------------
# get_frequency
# --------------------------------------------------------------------------

class FrequencyPoint(BaseModel):
    bucket: str
    session_count: int


def get_frequency(
    conn: sqlite3.Connection,
    exercise: str,
    date_from: str,
    date_to: str,
    by: Literal["week", "block"] = "week",
) -> list[FrequencyPoint]:
    """Number of distinct sessions an exercise was performed in, per bucket."""
    resolved = _resolve_or_raise(conn, exercise)

    rows = conn.execute(
        f"""
        SELECT DISTINCT s.session_id, s.date, s.block_id, b.name AS block_name
        FROM lift_set ls
        JOIN session s ON s.session_id = ls.session_id
        {_DRAFT_EXCLUSION_JOIN}
        WHERE ls.exercise_id = ?
          AND s.date BETWEEN ? AND ?
          AND {_DRAFT_EXCLUSION_WHERE}
        """,
        (resolved.exercise_id, date_from, date_to),
    ).fetchall()

    counts: dict[str, int] = {}
    for row in rows:
        bucket = _iso_week_bucket(row["date"]) if by == "week" else (row["block_name"] or "unattached")
        counts[bucket] = counts.get(bucket, 0) + 1

    return sorted(
        (FrequencyPoint(bucket=bucket, session_count=count) for bucket, count in counts.items()),
        key=lambda p: p.bucket,
    )


# --------------------------------------------------------------------------
# get_volume_trend
# --------------------------------------------------------------------------

class VolumePoint(BaseModel):
    bucket: str
    hard_sets: int
    tonnage_lb: float


_MUSCLE_GROUPS = {
    "chest", "triceps", "upper back", "lower back", "biceps", "core",
    "front deltoids", "side deltoids", "rear deltoids", "glutes",
    "adductors", "abductors", "quads", "hamstrings", "calves",
    "posterior chain",
}


def _latest_bodyweight(conn: sqlite3.Connection) -> float:
    """Most recent recorded bodyweight, 0.0 if none has ever been logged."""
    row = conn.execute(
        "SELECT weight_lb FROM bodyweight ORDER BY date DESC LIMIT 1"
    ).fetchone()
    return row["weight_lb"] if row is not None else 0.0


def get_volume_trend(
    conn: sqlite3.Connection,
    exercise_or_muscle_group: str,
    date_from: str,
    date_to: str,
    by: Literal["week", "block"] = "week",
) -> list[VolumePoint]:
    """Hard-set count + tonnage (lb) per bucket for one exercise or a whole muscle group.

    Tonnage for bodyweight-only sets (weight_lb IS NULL, e.g. bodyweight pullups)
    is estimated using the user's most recent logged bodyweight (0 lb if none
    has ever been recorded), added to the set's reps for that contribution.
    """
    normalized_group = exercise_or_muscle_group.strip().lower()
    exercise_ids: list[int] | None = None
    if normalized_group not in _MUSCLE_GROUPS:
        resolved = _resolve_or_raise(conn, exercise_or_muscle_group)
        exercise_ids = [resolved.exercise_id]

    if exercise_ids is not None:
        exercise_filter = "ls.exercise_id IN ({})".format(",".join("?" * len(exercise_ids)))
        exercise_params: list = list(exercise_ids)
    else:
        exercise_filter = "e.muscle_group = ?"
        exercise_params = [normalized_group]

    rows = conn.execute(
        f"""
        SELECT ls.weight_lb, ls.reps, s.date, s.block_id, b.name AS block_name
        FROM lift_set ls
        JOIN session s ON s.session_id = ls.session_id
        JOIN exercise e ON e.exercise_id = ls.exercise_id
        {_DRAFT_EXCLUSION_JOIN}
        WHERE {exercise_filter}
          AND s.date BETWEEN ? AND ?
          AND ls.reps IS NOT NULL
          AND {_DRAFT_EXCLUSION_WHERE}
        """,
        (*exercise_params, date_from, date_to),
    ).fetchall()

    bodyweight_cache: float | None = None
    hard_sets: dict[str, int] = {}
    tonnage: dict[str, float] = {}
    for row in rows:
        bucket = _iso_week_bucket(row["date"]) if by == "week" else (row["block_name"] or "unattached")
        weight = row["weight_lb"]
        if weight is None:
            if bodyweight_cache is None:
                bodyweight_cache = _latest_bodyweight(conn)
            weight = bodyweight_cache
        hard_sets[bucket] = hard_sets.get(bucket, 0) + 1
        tonnage[bucket] = tonnage.get(bucket, 0.0) + weight * row["reps"]

    buckets = sorted(set(hard_sets) | set(tonnage))
    return [
        VolumePoint(
            bucket=bucket,
            hard_sets=hard_sets.get(bucket, 0),
            tonnage_lb=round(tonnage.get(bucket, 0.0), 1),
        )
        for bucket in buckets
    ]


# --------------------------------------------------------------------------
# get_prs
# --------------------------------------------------------------------------

class PRResult(BaseModel):
    pr_id: int
    date: str
    exercise: str
    weight_lb: float
    reps: int
    context: str | None
    session_id: int | None


def get_prs(
    conn: sqlite3.Connection,
    exercise: str | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
) -> list[PRResult]:
    """Manually-recorded PRs from the `pr` table (not derived; see `find_recent_prs`)."""
    filters = ["1=1"]
    params: list = []
    if exercise is not None:
        resolved = _resolve_or_raise(conn, exercise)
        filters.append("pr.exercise_id = ?")
        params.append(resolved.exercise_id)
    if date_from is not None:
        filters.append("pr.date >= ?")
        params.append(date_from)
    if date_to is not None:
        filters.append("pr.date <= ?")
        params.append(date_to)

    rows = conn.execute(
        f"""
        SELECT pr.pr_id, pr.date, e.name AS exercise, pr.weight_lb, pr.reps,
               pr.context, pr.session_id
        FROM pr
        JOIN exercise e ON e.exercise_id = pr.exercise_id
        WHERE {' AND '.join(filters)}
        ORDER BY pr.date ASC
        """,
        params,
    ).fetchall()

    return [
        PRResult(
            pr_id=r["pr_id"],
            date=r["date"],
            exercise=r["exercise"],
            weight_lb=r["weight_lb"],
            reps=r["reps"],
            context=r["context"],
            session_id=r["session_id"],
        )
        for r in rows
    ]


# --------------------------------------------------------------------------
# find_recent_prs (auto-derive PR candidates)
# --------------------------------------------------------------------------

class PRCandidate(BaseModel):
    exercise: str
    exercise_id: int
    weight_lb: float
    reps: int
    e1rm: float
    date: str
    session_id: int
    set_id: int
    previous_best_e1rm: float | None
    previous_best_date: str | None


def find_recent_prs(
    conn: sqlite3.Connection,
    date_from: str,
    date_to: str,
    exercise: str | None = None,
) -> list[PRCandidate]:
    """Auto-derive PR candidates: sets in the window whose e1RM beats every prior set
    for that exercise (all-time, not just within the window).

    Read-only — nothing is written to the `pr` table. Pass the returned
    candidates the user accepts to `commit_prs` to insert them.
    """
    exercise_filter = ""
    params: list = []
    if exercise is not None:
        resolved = _resolve_or_raise(conn, exercise)
        exercise_filter = "AND ls.exercise_id = ?"
        params.append(resolved.exercise_id)

    rows = conn.execute(
        f"""
        SELECT ls.set_id, ls.exercise_id, e.name AS exercise_name, ls.weight_lb,
               ls.reps, s.date, s.session_id
        FROM lift_set ls
        JOIN session s ON s.session_id = ls.session_id
        JOIN exercise e ON e.exercise_id = ls.exercise_id
        {_DRAFT_EXCLUSION_JOIN}
        WHERE ls.weight_lb IS NOT NULL
          AND ls.reps IS NOT NULL
          AND ls.is_failed = 0
          AND {_DRAFT_EXCLUSION_WHERE}
          {exercise_filter}
        ORDER BY ls.exercise_id, s.date ASC, ls.set_id ASC
        """,
        params,
    ).fetchall()

    # Running best e1RM per exercise across all-time (not just the window), so
    # a set inside the window is only a "PR" if it beats everything before it.
    running_best: dict[int, tuple[float, str]] = {}
    best_in_window: dict[int, PRCandidate] = {}

    for row in rows:
        exercise_id = row["exercise_id"]
        e1rm = _epley(row["weight_lb"], row["reps"])
        prior = running_best.get(exercise_id)

        in_window = date_from <= row["date"] <= date_to
        beats_prior = prior is None or e1rm > prior[0]

        if in_window and beats_prior:
            candidate = PRCandidate(
                exercise=row["exercise_name"],
                exercise_id=exercise_id,
                weight_lb=row["weight_lb"],
                reps=row["reps"],
                e1rm=round(e1rm, 1),
                date=row["date"],
                session_id=row["session_id"],
                set_id=row["set_id"],
                previous_best_e1rm=round(prior[0], 1) if prior else None,
                previous_best_date=prior[1] if prior else None,
            )
            existing = best_in_window.get(exercise_id)
            if existing is None or e1rm > existing.e1rm:
                best_in_window[exercise_id] = candidate

        if beats_prior:
            running_best[exercise_id] = (e1rm, row["date"])

    return sorted(best_in_window.values(), key=lambda c: c.date)


def commit_prs(conn: sqlite3.Connection, candidates: list[PRCandidate]) -> int:
    """Insert user-accepted PR candidates into the `pr` table.

    Skips a candidate if an identical (exercise_id, date, weight_lb, reps) row
    already exists, so re-accepting the same candidate twice is a no-op.
    """
    inserted = 0
    for candidate in candidates:
        exists = conn.execute(
            """
            SELECT 1 FROM pr
            WHERE exercise_id = ? AND date = ? AND weight_lb = ? AND reps = ?
            """,
            (candidate.exercise_id, candidate.date, candidate.weight_lb, candidate.reps),
        ).fetchone()
        if exists:
            continue
        conn.execute(
            """
            INSERT INTO pr (date, session_id, exercise_id, weight_lb, reps, context)
            VALUES (?, ?, ?, ?, ?, 'auto-derived')
            """,
            (candidate.date, candidate.session_id, candidate.exercise_id,
             candidate.weight_lb, candidate.reps),
        )
        inserted += 1
    conn.commit()
    return inserted


# --------------------------------------------------------------------------
# get_injuries
# --------------------------------------------------------------------------

class InjuryResult(BaseModel):
    injury_id: int
    start_date: str
    end_date: str | None
    area: str
    severity: str | None
    note: str | None


def get_injuries(
    conn: sqlite3.Connection,
    active_only: bool = False,
    area: str | None = None,
) -> list[InjuryResult]:
    filters = ["1=1"]
    params: list = []
    if active_only:
        filters.append("end_date IS NULL")
    if area is not None:
        filters.append("area = ?")
        params.append(area)

    rows = conn.execute(
        f"""
        SELECT injury_id, start_date, end_date, area, severity, note
        FROM injury
        WHERE {' AND '.join(filters)}
        ORDER BY start_date ASC
        """,
        params,
    ).fetchall()

    return [
        InjuryResult(
            injury_id=r["injury_id"],
            start_date=r["start_date"],
            end_date=r["end_date"],
            area=r["area"],
            severity=r["severity"],
            note=r["note"],
        )
        for r in rows
    ]


# --------------------------------------------------------------------------
# get_measurements
# --------------------------------------------------------------------------

class MeasurementResult(BaseModel):
    m_id: int
    date: str
    site: str
    value_in: float
    note: str | None


def get_measurements(
    conn: sqlite3.Connection,
    site: str | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
) -> list[MeasurementResult]:
    filters = ["1=1"]
    params: list = []
    if site is not None:
        filters.append("site = ?")
        params.append(site)
    if date_from is not None:
        filters.append("date >= ?")
        params.append(date_from)
    if date_to is not None:
        filters.append("date <= ?")
        params.append(date_to)

    rows = conn.execute(
        f"""
        SELECT m_id, date, site, value_in, note
        FROM measurement
        WHERE {' AND '.join(filters)}
        ORDER BY date ASC
        """,
        params,
    ).fetchall()

    return [
        MeasurementResult(
            m_id=r["m_id"], date=r["date"], site=r["site"],
            value_in=r["value_in"], note=r["note"],
        )
        for r in rows
    ]


# --------------------------------------------------------------------------
# get_programs
# --------------------------------------------------------------------------

class ProgramResult(BaseModel):
    program_id: int
    name: str
    status: Literal["complete", "incomplete", "draft"]
    start_date: str | None
    end_date: str | None
    goals_text: str | None
    review_text: str | None
    notes: str | None


def get_programs(
    conn: sqlite3.Connection,
    status: Literal["complete", "incomplete", "draft"] | None = None,
) -> list[ProgramResult]:
    """List programs. The only tool that can surface `draft` programs explicitly."""
    filters = ["1=1"]
    params: list = []
    if status is not None:
        filters.append("status = ?")
        params.append(status)

    rows = conn.execute(
        f"""
        SELECT program_id, name, status, start_date, end_date, goals_text, review_text, notes
        FROM program
        WHERE {' AND '.join(filters)}
        ORDER BY start_date ASC
        """,
        params,
    ).fetchall()

    return [
        ProgramResult(
            program_id=r["program_id"], name=r["name"], status=r["status"],
            start_date=r["start_date"], end_date=r["end_date"],
            goals_text=r["goals_text"], review_text=r["review_text"], notes=r["notes"],
        )
        for r in rows
    ]


# --------------------------------------------------------------------------
# get_block_outline
# --------------------------------------------------------------------------

class ProgrammedSlotResult(BaseModel):
    slot_id: int
    week_number: int | None
    day_number: int | None
    day_label: str | None
    exercise: str | None
    prescription: str
    target_weight_lb: float | None
    notes: str | None


def get_block_outline(conn: sqlite3.Connection, block_id: int) -> list[ProgrammedSlotResult]:
    """All `programmed_slot` rows for a block, in program order."""
    rows = conn.execute(
        """
        SELECT ps.slot_id, ps.week_number, ps.day_number, ps.day_label,
               e.name AS exercise, ps.prescription, ps.target_weight_lb, ps.notes
        FROM programmed_slot ps
        LEFT JOIN exercise e ON e.exercise_id = ps.exercise_id
        WHERE ps.block_id = ?
        ORDER BY ps.week_number ASC, ps.day_number ASC, ps.slot_id ASC
        """,
        (block_id,),
    ).fetchall()

    return [
        ProgrammedSlotResult(
            slot_id=r["slot_id"], week_number=r["week_number"], day_number=r["day_number"],
            day_label=r["day_label"], exercise=r["exercise"], prescription=r["prescription"],
            target_weight_lb=r["target_weight_lb"], notes=r["notes"],
        )
        for r in rows
    ]


# --------------------------------------------------------------------------
# compare_programmed_vs_actual
# --------------------------------------------------------------------------

class ProgrammedVsActualRow(BaseModel):
    week_number: int | None
    day_number: int | None
    exercise: str
    prescription: str
    target_weight_lb: float | None
    actual_top_weight_lb: float | None
    actual_top_reps: int | None
    actual_session_id: int | None
    actual_date: str | None
    actual_raw_note: str | None


class UnmatchedActual(BaseModel):
    """Performed work with no corresponding programmed_slot in this block."""
    session_id: int
    date: str
    exercise: str
    top_weight_lb: float | None
    top_reps: int | None
    raw_note: str | None


class CompareProgrammedVsActualResult(BaseModel):
    rows: list[ProgrammedVsActualRow]
    unmatched_actual: list[UnmatchedActual]
    note: str | None = None


def compare_programmed_vs_actual(
    conn: sqlite3.Connection,
    block_id: int,
    exercise: str | None = None,
) -> CompareProgrammedVsActualResult:
    """Join `programmed_slot` against performed `lift_set` rows for a block.

    Matching key: (week_number, day_number, exercise_id) within the block.
    When one side is missing data, the mismatch is reported explicitly rather
    than silently dropped:
    - A programmed slot with no matching performed session is still returned
      in `rows` with the `actual_*` fields set to None.
    - Performed work with no matching programmed slot is returned in
      `unmatched_actual`, including its raw session note.
    - If the block has no `programmed_slot` rows at all, `rows` is empty and
      `note` explains that nothing was programmed, with performed work still
      surfaced via `unmatched_actual`.
    """
    exercise_filter = ""
    exercise_id: int | None = None
    if exercise is not None:
        exercise_id = _resolve_or_raise(conn, exercise).exercise_id
        exercise_filter = "AND ps.exercise_id = ?"

    slot_params: list = [block_id]
    if exercise_id is not None:
        slot_params.append(exercise_id)

    slots = conn.execute(
        f"""
        SELECT ps.slot_id, ps.week_number, ps.day_number, ps.exercise_id,
               e.name AS exercise, ps.prescription, ps.target_weight_lb
        FROM programmed_slot ps
        LEFT JOIN exercise e ON e.exercise_id = ps.exercise_id
        WHERE ps.block_id = ?
          {exercise_filter}
        ORDER BY ps.week_number ASC, ps.day_number ASC
        """,
        slot_params,
    ).fetchall()

    actual_exercise_filter = ""
    actual_params: list = [block_id]
    if exercise_id is not None:
        actual_params.append(exercise_id)
        actual_exercise_filter = "AND ls.exercise_id = ?"

    actual_rows = conn.execute(
        f"""
        SELECT s.session_id, s.date, s.week_number, s.day_number, ls.exercise_id,
               e.name AS exercise, ls.weight_lb, ls.reps, ls.is_top_set, s.raw_note
        FROM session s
        JOIN lift_set ls ON ls.session_id = s.session_id
        JOIN exercise e ON e.exercise_id = ls.exercise_id
        WHERE s.block_id = ?
          {actual_exercise_filter}
        ORDER BY s.date ASC, ls.is_top_set DESC
        """,
        actual_params,
    ).fetchall()

    # Best (top) actual set per (week_number, day_number, exercise_id).
    actual_by_key: dict[tuple, dict] = {}
    for row in actual_rows:
        key = (row["week_number"], row["day_number"], row["exercise_id"])
        current = actual_by_key.get(key)
        if current is None or (row["is_top_set"] and not current["is_top_set"]) or (
            not current["is_top_set"] and not row["is_top_set"]
            and (row["weight_lb"] or 0) > (current["weight_lb"] or 0)
        ):
            actual_by_key[key] = dict(row)

    matched_keys: set[tuple] = set()
    rows: list[ProgrammedVsActualRow] = []
    for slot in slots:
        key = (slot["week_number"], slot["day_number"], slot["exercise_id"])
        actual = actual_by_key.get(key)
        matched_keys.add(key)
        rows.append(
            ProgrammedVsActualRow(
                week_number=slot["week_number"],
                day_number=slot["day_number"],
                exercise=slot["exercise"],
                prescription=slot["prescription"],
                target_weight_lb=slot["target_weight_lb"],
                actual_top_weight_lb=actual["weight_lb"] if actual else None,
                actual_top_reps=actual["reps"] if actual else None,
                actual_session_id=actual["session_id"] if actual else None,
                actual_date=actual["date"] if actual else None,
                actual_raw_note=actual["raw_note"] if actual else None,
            )
        )

    unmatched_actual = [
        UnmatchedActual(
            session_id=data["session_id"],
            date=data["date"],
            exercise=data["exercise"],
            top_weight_lb=data["weight_lb"],
            top_reps=data["reps"],
            raw_note=data["raw_note"],
        )
        for key, data in actual_by_key.items()
        if key not in matched_keys
    ]
    unmatched_actual.sort(key=lambda u: u.date)

    note = None
    if not slots and actual_rows:
        note = (
            "No programmed data found for this block — nothing to compare against. "
            "Showing the performed work that does exist instead."
        )
    elif not slots and not actual_rows:
        note = "No programmed data and no performed data found for this block."

    return CompareProgrammedVsActualResult(rows=rows, unmatched_actual=unmatched_actual, note=note)


if __name__ == "__main__":
    from src.db.connection import get_conn

    conn = get_conn("data/training.db")

    print("get_best_set:", get_best_set(conn, "bench press", "2026-03-01", "2026-03-31"))
    print("get_lifts (top sets):", get_lifts(conn, "deadlift", "2026-01-01", "2026-06-30", top_sets_only=True))
    print("get_e1rm_trend:", get_e1rm_trend(conn, "deadlift", "2026-01-01", "2026-06-30", by="week"))
    print("get_bodyweight_trend:", get_bodyweight_trend(conn, "2026-01-01", "2026-06-30"))
