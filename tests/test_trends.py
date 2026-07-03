"""Chart-prep helpers for the Trends tab (`src/ui/trends.py`, streamlit-free).

The `trends_tab.py` veneer is only import-smoke-tested (see `test_ui.py`);
everything decision-shaped is exercised here against the seeded DB.
"""
from __future__ import annotations

import datetime as dt

from src.tools.queries import (
    MUSCLE_GROUPS,
    get_bodyweight_trend,
    get_e1rm_trend,
    get_measurements,
    get_prs,
    get_volume_trend,
)
from src.ui import trends

RANGE = ("2026-01-01", "2026-12-31")


# ---------------------------------------------------------------------------
# default_date_range
# ---------------------------------------------------------------------------

def test_default_date_range_spans_six_months():
    date_from, date_to = trends.default_date_range(today=dt.date(2026, 7, 3))
    assert (date_from, date_to) == ("2026-01-03", "2026-07-03")


def test_default_date_range_clamps_short_months():
    date_from, _ = trends.default_date_range(today=dt.date(2026, 8, 31))
    assert date_from == "2026-02-28"


def test_default_date_range_crosses_year_boundary():
    date_from, _ = trends.default_date_range(today=dt.date(2026, 2, 1), months=6)
    assert date_from == "2025-08-01"


# ---------------------------------------------------------------------------
# selector helpers
# ---------------------------------------------------------------------------

def test_list_exercises_main_lifts_subset_of_all(conn):
    main = trends.list_exercises(conn, main_lifts_only=True)
    everything = trends.list_exercises(conn, main_lifts_only=False)
    assert set(main) < set(everything)  # seeded accessories exist
    assert "Deadlift" in main
    assert everything == sorted(everything)


def test_list_measurement_sites(conn):
    assert trends.list_measurement_sites(conn) == ["arm", "waist"]


def test_muscle_groups_public_and_sorted():
    assert MUSCLE_GROUPS == sorted(MUSCLE_GROUPS)
    assert "posterior chain" in MUSCLE_GROUPS


# ---------------------------------------------------------------------------
# frame builders
# ---------------------------------------------------------------------------

def test_bodyweight_frame_matches_trend_rows(conn):
    trend = get_bodyweight_trend(conn, *RANGE)
    frame = trends.bodyweight_frame(trend)
    assert list(frame.columns) == ["date", "weight_lb"]
    assert len(frame) == len(trend.rows) > 0
    assert frame["weight_lb"].iloc[-1] == trend.last


def test_bodyweight_frame_empty_keeps_columns(conn):
    frame = trends.bodyweight_frame(get_bodyweight_trend(conn, "1990-01-01", "1990-12-31"))
    assert frame.empty and list(frame.columns) == ["date", "weight_lb"]


def test_e1rm_frame_week_mode_uses_source_dates(conn):
    points = get_e1rm_trend(conn, "Deadlift", *RANGE, by="week")
    frame = trends.e1rm_frame(points, by="week")
    assert list(frame.columns) == ["x", "e1rm", "bucket", "source"]
    assert len(frame) == len(points) > 0
    # Week mode charts on the temporal source date, not the week label.
    assert frame["x"].tolist() == [p.source_date for p in points]
    assert frame["source"].iloc[0].endswith(points[0].source_date)


def test_e1rm_frame_block_mode_uses_bucket_labels(conn):
    points = get_e1rm_trend(conn, "Deadlift", *RANGE, by="block")
    frame = trends.e1rm_frame(points, by="block")
    assert frame["x"].tolist() == [p.bucket for p in points]


def test_pr_frame_singles_filter(conn):
    prs = get_prs(conn, "Deadlift", *RANGE)
    singles = trends.pr_frame(prs, singles_only=True)
    everything = trends.pr_frame(prs, singles_only=False)
    assert set(singles["reps"]) <= {1}
    assert len(everything) == len(prs)
    assert list(singles.columns) == ["date", "weight_lb", "reps", "context"]


def test_measurement_frame_one_row_per_measurement(conn):
    rows = get_measurements(conn, site="arm", date_from=RANGE[0], date_to=RANGE[1])
    frame = trends.measurement_frame(rows)
    assert list(frame.columns) == ["date", "site", "value_in"]
    assert set(frame["site"]) == {"arm"}
    assert len(frame) == len(rows) > 0


def test_volume_frame_muscle_group_and_exercise(conn):
    group_frame = trends.volume_frame(get_volume_trend(conn, "posterior chain", *RANGE))
    exercise_frame = trends.volume_frame(get_volume_trend(conn, "Deadlift", *RANGE))
    assert list(group_frame.columns) == ["bucket", "hard_sets", "tonnage_lb"]
    assert not group_frame.empty and not exercise_frame.empty
    # A muscle group aggregates at least as much as any single exercise in it.
    assert group_frame["hard_sets"].sum() >= exercise_frame["hard_sets"].sum()
