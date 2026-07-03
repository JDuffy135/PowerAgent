"""Streamlit-free chart-prep logic for the Trends tab (Stage 10).

Turns typed query-tool outputs (`src/tools/queries.py`) into chart-ready
pandas DataFrames. Everything decision-shaped for the Trends tab lives here
(unit-tested in `tests/test_trends.py`); `trends_tab.py` is the rendering
veneer.

Empty inputs yield empty DataFrames that still carry the expected columns, so
the tab can branch on `.empty` without special-casing.
"""
from __future__ import annotations

import calendar
import datetime as dt
import sqlite3

import pandas as pd

from src.tools.queries import (
    BodyweightTrend,
    E1RMPoint,
    MeasurementResult,
    PRResult,
    VolumePoint,
)

DEFAULT_RANGE_MONTHS = 6


def default_date_range(
    today: dt.date | None = None, months: int = DEFAULT_RANGE_MONTHS
) -> tuple[str, str]:
    """ISO `(date_from, date_to)` pair ending today, starting `months` back.

    Day-of-month is clamped when the start month is shorter (Aug 31 - 6mo ->
    Feb 28/29).
    """
    today = today or dt.date.today()
    month_index = today.year * 12 + (today.month - 1) - months
    year, month = divmod(month_index, 12)
    month += 1
    day = min(today.day, calendar.monthrange(year, month)[1])
    return dt.date(year, month, day).isoformat(), today.isoformat()


def list_exercises(conn: sqlite3.Connection, main_lifts_only: bool = True) -> list[str]:
    """Exercise names for selectors, sorted; default trims to the SBD tiers."""
    if main_lifts_only:
        rows = conn.execute(
            "SELECT name FROM exercise WHERE tier IN ('competition', 'variation') ORDER BY name"
        ).fetchall()
    else:
        rows = conn.execute("SELECT name FROM exercise ORDER BY name").fetchall()
    return [r["name"] for r in rows]


def list_measurement_sites(conn: sqlite3.Connection) -> list[str]:
    rows = conn.execute("SELECT DISTINCT site FROM measurement ORDER BY site").fetchall()
    return [r["site"] for r in rows]


def bodyweight_frame(trend: BodyweightTrend) -> pd.DataFrame:
    return pd.DataFrame(
        [{"date": r.date, "weight_lb": r.weight_lb} for r in trend.rows],
        columns=["date", "weight_lb"],
    )


def e1rm_frame(points: list[E1RMPoint], by: str = "week") -> pd.DataFrame:
    """`x` is the source-set date in week mode (temporal axis, so recorded-PR
    points can be layered at their own dates) and the block name in block mode
    (ordinal axis; no PR overlay there)."""
    rows = [
        {
            "x": p.source_date if by == "week" else p.bucket,
            "e1rm": p.e1rm,
            "bucket": p.bucket,
            "source": f"{p.source_weight_lb:g} lb x {p.source_reps} on {p.source_date}",
        }
        for p in points
    ]
    return pd.DataFrame(rows, columns=["x", "e1rm", "bucket", "source"])


def pr_frame(prs: list[PRResult], singles_only: bool = True) -> pd.DataFrame:
    """Recorded PRs as scatter points; `singles_only` keeps true 1RMs (reps = 1)."""
    rows = [
        {"date": p.date, "weight_lb": p.weight_lb, "reps": p.reps, "context": p.context or ""}
        for p in prs
        if not singles_only or p.reps == 1
    ]
    return pd.DataFrame(rows, columns=["date", "weight_lb", "reps", "context"])


def measurement_frame(measurements: list[MeasurementResult]) -> pd.DataFrame:
    rows = [{"date": m.date, "site": m.site, "value_in": m.value_in} for m in measurements]
    return pd.DataFrame(rows, columns=["date", "site", "value_in"])


def volume_frame(points: list[VolumePoint]) -> pd.DataFrame:
    rows = [
        {"bucket": p.bucket, "hard_sets": p.hard_sets, "tonnage_lb": p.tonnage_lb}
        for p in points
    ]
    return pd.DataFrame(rows, columns=["bucket", "hard_sets", "tonnage_lb"])
