"""Trends tab: time-series charts over the typed query tools.

Rendering veneer only — chart-prep logic lives in the streamlit-free
`src/ui/trends.py`. All data goes through `src/tools/queries.py`, so draft
programs are already excluded and weights stay canonical lb.
"""
from __future__ import annotations

import datetime as dt

import altair as alt
import streamlit as st

from src.tools.queries import (
    MUSCLE_GROUPS,
    ExerciseNotFound,
    get_bodyweight_trend,
    get_e1rm_trend,
    get_measurements,
    get_prs,
    get_volume_trend,
)
from src.ui import trends

_PR_COLOR = "#d62728"


def render(conn) -> None:
    default_from, default_to = trends.default_date_range()
    col1, col2 = st.columns(2)
    with col1:
        date_from = st.date_input(
            "From", value=dt.date.fromisoformat(default_from), key="trends_from"
        )
    with col2:
        date_to = st.date_input(
            "To", value=dt.date.fromisoformat(default_to), key="trends_to"
        )
    if date_from > date_to:
        st.error("From-date is after to-date.")
        return

    date_from, date_to = date_from.isoformat(), date_to.isoformat()
    _render_bodyweight(conn, date_from, date_to)
    _render_one_rm(conn, date_from, date_to)
    _render_measurements(conn, date_from, date_to)
    _render_volume(conn, date_from, date_to)


def _render_bodyweight(conn, date_from: str, date_to: str) -> None:
    st.subheader("Bodyweight")
    trend = get_bodyweight_trend(conn, date_from, date_to)
    frame = trends.bodyweight_frame(trend)
    if frame.empty:
        st.info("No bodyweight entries in this range.")
        return

    cols = st.columns(5)
    for col, label, value in zip(
        cols,
        ("First", "Last", "Change", "Min", "Max"),
        (trend.first, trend.last, trend.delta, trend.min, trend.max),
    ):
        col.metric(label, f"{value:g} lb")

    chart = (
        alt.Chart(frame)
        .mark_line(point=True)
        .encode(
            x=alt.X("date:T", title="Date"),
            y=alt.Y("weight_lb:Q", title="Bodyweight (lb)", scale=alt.Scale(zero=False)),
            tooltip=[alt.Tooltip("date:T"), alt.Tooltip("weight_lb:Q", title="lb")],
        )
    )
    st.altair_chart(chart, width="stretch")


def _render_one_rm(conn, date_from: str, date_to: str) -> None:
    st.subheader("1RM per lift")
    col1, col2, col3 = st.columns(3)
    with col1:
        show_all = st.checkbox("Show all exercises", value=False, key="trends_1rm_all")
        names = trends.list_exercises(conn, main_lifts_only=not show_all)
        if not names:
            st.info("No exercises recorded yet.")
            return
        exercise = st.selectbox("Exercise", names, key="trends_1rm_exercise")
    with col2:
        by = st.radio("Bucket", ["week", "block"], horizontal=True, key="trends_1rm_by")
    with col3:
        singles_only = st.checkbox("1RM PRs only (reps = 1)", value=True, key="trends_1rm_singles")

    try:
        e1rm = trends.e1rm_frame(get_e1rm_trend(conn, exercise, date_from, date_to, by=by), by=by)
        prs = trends.pr_frame(get_prs(conn, exercise, date_from, date_to), singles_only=singles_only)
    except ExerciseNotFound:
        st.info(f"No data for {exercise}.")
        return
    if e1rm.empty and prs.empty:
        st.info("No qualifying sets or PRs for this lift in the range.")
        return

    if by == "week":
        layers = []
        if not e1rm.empty:
            layers.append(
                alt.Chart(e1rm)
                .mark_line(point=True)
                .encode(
                    x=alt.X("x:T", title="Date"),
                    y=alt.Y("e1rm:Q", title="Weight (lb)", scale=alt.Scale(zero=False)),
                    tooltip=[
                        alt.Tooltip("bucket:N", title="Week"),
                        alt.Tooltip("e1rm:Q", title="e1RM (lb)"),
                        alt.Tooltip("source:N", title="Source set"),
                    ],
                )
            )
        if not prs.empty:
            layers.append(
                alt.Chart(prs)
                .mark_point(size=120, filled=True, color=_PR_COLOR)
                .encode(
                    x=alt.X("date:T", title="Date"),
                    y=alt.Y("weight_lb:Q", scale=alt.Scale(zero=False)),
                    tooltip=[
                        alt.Tooltip("date:T"),
                        alt.Tooltip("weight_lb:Q", title="PR (lb)"),
                        alt.Tooltip("reps:Q", title="Reps"),
                        alt.Tooltip("context:N", title="Context"),
                    ],
                )
            )
        st.altair_chart(alt.layer(*layers), width="stretch")
        st.caption("Line: weekly best Epley e1RM. Red points: recorded PRs.")
    else:
        # Block buckets aren't temporal, so PRs aren't overlaid here.
        chart = (
            alt.Chart(e1rm)
            .mark_line(point=True)
            .encode(
                x=alt.X("x:N", sort=None, title="Block"),
                y=alt.Y("e1rm:Q", title="e1RM (lb)", scale=alt.Scale(zero=False)),
                tooltip=[
                    alt.Tooltip("bucket:N", title="Block"),
                    alt.Tooltip("e1rm:Q", title="e1RM (lb)"),
                    alt.Tooltip("source:N", title="Source set"),
                ],
            )
        )
        st.altair_chart(chart, width="stretch")
        st.caption("Best Epley e1RM per block. Switch to week bucketing for the PR overlay.")


def _render_measurements(conn, date_from: str, date_to: str) -> None:
    st.subheader("Measurements")
    sites = trends.list_measurement_sites(conn)
    if not sites:
        st.info("No measurements recorded yet.")
        return

    picked = st.multiselect("Sites", sites, default=sites, key="trends_meas_sites")
    if not picked:
        st.info("Pick at least one site.")
        return

    rows: list = []
    for site in picked:
        rows.extend(get_measurements(conn, site=site, date_from=date_from, date_to=date_to))
    frame = trends.measurement_frame(rows)
    if frame.empty:
        st.info("No measurements in this range.")
        return

    chart = (
        alt.Chart(frame)
        .mark_line(point=True)
        .encode(
            x=alt.X("date:T", title="Date"),
            y=alt.Y("value_in:Q", title="Size (in)", scale=alt.Scale(zero=False)),
            color=alt.Color("site:N", title="Site"),
            tooltip=[
                alt.Tooltip("date:T"),
                alt.Tooltip("site:N"),
                alt.Tooltip("value_in:Q", title="in"),
            ],
        )
    )
    st.altair_chart(chart, width="stretch")


def _render_volume(conn, date_from: str, date_to: str) -> None:
    st.subheader("Volume")
    col1, col2 = st.columns(2)
    with col1:
        kind = st.radio("Scope", ["Muscle group", "Exercise"], horizontal=True, key="trends_vol_kind")
        if kind == "Muscle group":
            target = st.selectbox("Muscle group", MUSCLE_GROUPS, key="trends_vol_mg")
        else:
            names = trends.list_exercises(conn, main_lifts_only=False)
            if not names:
                st.info("No exercises recorded yet.")
                return
            target = st.selectbox("Exercise", names, key="trends_vol_ex")
    with col2:
        by = st.radio("Bucket", ["week", "block"], horizontal=True, key="trends_vol_by")

    try:
        frame = trends.volume_frame(get_volume_trend(conn, target, date_from, date_to, by=by))
    except ExerciseNotFound:
        st.info(f"No data for {target}.")
        return
    if frame.empty:
        st.info("No sets in this range.")
        return

    bucket_title = "Week" if by == "week" else "Block"
    sets_col, tonnage_col = st.columns(2)
    with sets_col:
        st.altair_chart(
            alt.Chart(frame)
            .mark_bar()
            .encode(
                x=alt.X("bucket:N", sort=None, title=bucket_title),
                y=alt.Y("hard_sets:Q", title="Hard sets"),
                tooltip=[alt.Tooltip("bucket:N"), alt.Tooltip("hard_sets:Q", title="Sets")],
            ),
            width="stretch",
        )
    with tonnage_col:
        st.altair_chart(
            alt.Chart(frame)
            .mark_bar()
            .encode(
                x=alt.X("bucket:N", sort=None, title=bucket_title),
                y=alt.Y("tonnage_lb:Q", title="Tonnage (lb)"),
                tooltip=[alt.Tooltip("bucket:N"), alt.Tooltip("tonnage_lb:Q", title="lb")],
            ),
            width="stretch",
        )
    st.caption(
        "Bodyweight-only sets count toward tonnage at the latest logged bodyweight."
    )
