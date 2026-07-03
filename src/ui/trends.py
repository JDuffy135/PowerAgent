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

from src.agent.units import to_display_weight
from src.tools.queries import (
    BodyweightTrend,
    E1RMPoint,
    MeasurementResult,
    PRResult,
    VolumePoint,
)

DEFAULT_RANGE_MONTHS = 6


def weight_label(unit: str) -> str:
    """Axis-title suffix for a weight series in the display unit, e.g. `(kg)`."""
    return f"({unit})"


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


# Weight columns hold values already converted to the display unit (`unit`),
# so charts read them straight; storage upstream stays canonical lb (§2). The
# column names keep their historical spelling for stable Altair encodings —
# they are the display series, not the stored one. `measurement_frame` is never
# converted (measurements are inches, not weights).


def bodyweight_frame(trend: BodyweightTrend, unit: str = "lb") -> pd.DataFrame:
    return pd.DataFrame(
        [{"date": r.date, "weight_lb": to_display_weight(r.weight_lb, unit)} for r in trend.rows],
        columns=["date", "weight_lb"],
    )


def e1rm_frame(points: list[E1RMPoint], by: str = "week", unit: str = "lb") -> pd.DataFrame:
    """`x` is the source-set date in week mode (temporal axis, so recorded-PR
    points can be layered at their own dates) and the block name in block mode
    (ordinal axis; no PR overlay there)."""
    rows = [
        {
            "x": p.source_date if by == "week" else p.bucket,
            "e1rm": to_display_weight(p.e1rm, unit),
            "bucket": p.bucket,
            "source": f"{format_source(p.source_weight_lb, unit)} x {p.source_reps} on {p.source_date}",
        }
        for p in points
    ]
    return pd.DataFrame(rows, columns=["x", "e1rm", "bucket", "source"])


def format_source(weight_lb: float, unit: str) -> str:
    """Compact `<value> <unit>` for a source-set tooltip (no trailing `.0`)."""
    value = to_display_weight(weight_lb, unit)
    return f"{value:g} {unit}"


def pr_frame(prs: list[PRResult], singles_only: bool = True, unit: str = "lb") -> pd.DataFrame:
    """Recorded PRs as scatter points; `singles_only` keeps true 1RMs (reps = 1)."""
    rows = [
        {
            "date": p.date,
            "weight_lb": to_display_weight(p.weight_lb, unit),
            "reps": p.reps,
            "context": p.context or "",
        }
        for p in prs
        if not singles_only or p.reps == 1
    ]
    return pd.DataFrame(rows, columns=["date", "weight_lb", "reps", "context"])


def measurement_frame(measurements: list[MeasurementResult]) -> pd.DataFrame:
    rows = [{"date": m.date, "site": m.site, "value_in": m.value_in} for m in measurements]
    return pd.DataFrame(rows, columns=["date", "site", "value_in"])


def volume_frame(points: list[VolumePoint], unit: str = "lb") -> pd.DataFrame:
    rows = [
        {
            "bucket": p.bucket,
            "hard_sets": p.hard_sets,
            "tonnage_lb": to_display_weight(p.tonnage_lb, unit),
        }
        for p in points
    ]
    return pd.DataFrame(rows, columns=["bucket", "hard_sets", "tonnage_lb"])
