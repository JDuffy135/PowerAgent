import pytest

from src.tools.queries import (
    ExerciseNotFound,
    PRCandidate,
    _epley,
    commit_prs,
    compare_programmed_vs_actual,
    find_recent_prs,
    get_best_set,
    get_block_outline,
    get_bodyweight_trend,
    get_e1rm_trend,
    get_frequency,
    get_injuries,
    get_lifts,
    get_measurements,
    get_prs,
    get_programs,
    get_sessions,
    get_volume_trend,
)


def _block_id(conn, name):
    return conn.execute("SELECT block_id FROM block WHERE name = ?", (name,)).fetchone()["block_id"]


def test_get_best_set_march_bench_anchor(conn):
    result = get_best_set(conn, "bench press", "2026-03-01", "2026-03-31")
    assert result is not None
    assert result.weight_lb == 230
    assert result.reps == 1
    assert result.date == "2026-03-19"


def test_get_best_set_respects_min_reps(conn):
    # Full range covers both deadlift sessions: 385x1 (top) and 315x3 backoffs.
    result = get_best_set(conn, "deadlift", "2026-01-01", "2026-06-30", min_reps=3)
    assert result is not None
    assert result.weight_lb == 315
    assert result.reps == 3


def test_get_best_set_unresolvable_exercise_raises(conn):
    with pytest.raises(ExerciseNotFound):
        get_best_set(conn, "underwater basket weaving", "2026-01-01", "2026-06-30")


def test_get_lifts_top_sets_only(conn):
    sessions = get_lifts(conn, "deadlift", "2026-01-01", "2026-06-30", top_sets_only=True)
    assert len(sessions) == 2  # 2026-04-02 and 2026-06-01
    for session in sessions:
        assert len(session.sets) == 1
        assert session.sets[0].is_top_set is True

    weights = sorted(s.sets[0].weight_lb for s in sessions)
    assert weights == [335, 385]


def test_get_lifts_returns_all_sets_chronologically(conn):
    sessions = get_lifts(conn, "deadlift", "2026-01-01", "2026-06-30")
    dates = [s.date for s in sessions]
    assert dates == sorted(dates)
    total_sets = sum(len(s.sets) for s in sessions)
    assert total_sets == 4 + 5  # S5: 1 top + 3 backoff; S9: 1 top + 4 backoff


def test_epley_formula_exact():
    assert round(_epley(385, 1), 1) == 397.8
    assert round(_epley(315, 3), 1) == 346.5


def test_get_e1rm_trend_weekly_buckets_and_math(conn):
    points = get_e1rm_trend(conn, "deadlift", "2026-01-01", "2026-06-30", by="week")
    # Two deadlift sessions in different ISO weeks -> two buckets.
    assert len(points) == 2
    assert [p.bucket for p in points] == sorted(p.bucket for p in points)

    june_point = next(p for p in points if p.source_date == "2026-06-01")
    assert june_point.e1rm == 397.8
    assert june_point.source_weight_lb == 385
    assert june_point.source_reps == 1


def test_get_bodyweight_trend_delta(conn):
    trend = get_bodyweight_trend(conn, "2026-01-01", "2026-06-30")
    assert trend.first == 138.0
    assert trend.last == 146.0
    assert trend.delta == pytest.approx(8.0)


def test_draft_program_excluded_from_all_tools(conn):
    draft_program_id = conn.execute(
        "INSERT INTO program (name, status) VALUES ('Abandoned Outline', 'draft')"
    ).lastrowid
    draft_block_id = conn.execute(
        """
        INSERT INTO block (program_id, name, focus, week_count)
        VALUES (?, 'Draft Block', 'strength', 4)
        """,
        (draft_program_id,),
    ).lastrowid
    draft_session_id = conn.execute(
        """
        INSERT INTO session (date, block_id, session_type)
        VALUES ('2026-06-15', ?, 'lifting')
        """,
        (draft_block_id,),
    ).lastrowid
    deadlift_id = conn.execute(
        "SELECT exercise_id FROM exercise WHERE name = 'Deadlift'"
    ).fetchone()["exercise_id"]
    conn.execute(
        """
        INSERT INTO lift_set (session_id, exercise_id, set_index, weight_lb, reps, is_top_set)
        VALUES (?, ?, 1, 999, 1, 1)
        """,
        (draft_session_id, deadlift_id),
    )
    conn.commit()

    best = get_best_set(conn, "deadlift", "2026-01-01", "2026-12-31")
    assert best.weight_lb == 385  # not the 999 draft set

    lifts = get_lifts(conn, "deadlift", "2026-01-01", "2026-12-31")
    all_weights = [st.weight_lb for sess in lifts for st in sess.sets]
    assert 999 not in all_weights

    trend = get_e1rm_trend(conn, "deadlift", "2026-01-01", "2026-12-31", by="week")
    assert all(p.source_weight_lb != 999 for p in trend)


# --------------------------------------------------------------------------
# get_sessions
# --------------------------------------------------------------------------

def test_get_sessions_date_range(conn):
    sessions = get_sessions(conn, "2026-01-01", "2026-06-30")
    assert len(sessions) == 10
    dates = [s.date for s in sessions]
    assert dates == sorted(dates)


def test_get_sessions_filters_by_block_and_type(conn):
    peaking_block_id = _block_id(conn, "Peaking Block")
    lifting_sessions = get_sessions(
        conn, "2026-01-01", "2026-06-30", block_id=peaking_block_id, session_type="lifting"
    )
    assert all(s.block_id == peaking_block_id for s in lifting_sessions)
    assert all(s.session_type == "lifting" for s in lifting_sessions)

    cardio_sessions = get_sessions(conn, "2026-01-01", "2026-06-30", session_type="cardio")
    assert len(cardio_sessions) == 1
    assert cardio_sessions[0].duration_min == 26


# --------------------------------------------------------------------------
# get_frequency
# --------------------------------------------------------------------------

def test_get_frequency_counts_distinct_sessions(conn):
    points = get_frequency(conn, "bench press", "2026-01-01", "2026-06-30", by="week")
    assert all(p.session_count == 1 for p in points)
    assert len(points) == 3  # bench appears in 3 distinct sessions (S3, S4, S10)


def test_get_frequency_unresolvable_raises(conn):
    with pytest.raises(ExerciseNotFound):
        get_frequency(conn, "nonexistent exercise", "2026-01-01", "2026-06-30")


# --------------------------------------------------------------------------
# get_volume_trend
# --------------------------------------------------------------------------

def test_get_volume_trend_by_exercise(conn):
    points = get_volume_trend(conn, "deadlift", "2026-01-01", "2026-06-30", by="block")
    assert len(points) == 2  # Strength Block 1, Peaking Block
    strength_point = next(p for p in points if p.bucket == "Strength Block 1")
    # S5: 335x2 (top) + 3x285x4 = 335*2 + 3*285*4
    assert strength_point.hard_sets == 4
    assert strength_point.tonnage_lb == pytest.approx(335 * 2 + 3 * 285 * 4)


def test_get_volume_trend_by_muscle_group_aggregates_exercises(conn):
    points = get_volume_trend(conn, "posterior chain", "2026-01-01", "2026-06-30", by="block")
    # squat + deadlift both tag as posterior chain; Strength Block 1 has both.
    strength_point = next(p for p in points if p.bucket == "Strength Block 1")
    assert strength_point.hard_sets == 4 + 5  # deadlift (S5) + squat (S6)


def test_get_volume_trend_bodyweight_exercise_uses_latest_bodyweight(conn):
    # Weighted Pullups store only the ADDED weight (45); tonnage should use
    # added weight, not bodyweight, since weight_lb is NOT NULL for this exercise.
    points = get_volume_trend(conn, "weighted pullups", "2026-01-01", "2026-06-30")
    assert points[0].tonnage_lb == pytest.approx(45 * 8 * 3)


def test_get_volume_trend_bodyweight_only_set_defaults_to_zero_when_no_bw_logged(conn):
    exercise_id = conn.execute(
        "SELECT exercise_id FROM exercise WHERE name = 'Weighted Pullups'"
    ).fetchone()["exercise_id"]
    session_id = conn.execute(
        "INSERT INTO session (date, session_type) VALUES ('2026-07-01', 'lifting')"
    ).lastrowid
    conn.execute(
        """
        INSERT INTO lift_set (session_id, exercise_id, set_index, weight_lb, reps)
        VALUES (?, ?, 1, NULL, 10)
        """,
        (session_id, exercise_id),
    )
    conn.commit()
    conn.execute("DELETE FROM bodyweight")
    conn.commit()

    points = get_volume_trend(conn, "weighted pullups", "2026-07-01", "2026-07-01")
    assert points[0].tonnage_lb == 0.0  # 0 lb bodyweight fallback * 10 reps


def test_get_volume_trend_bodyweight_only_set_uses_latest_bodyweight(conn):
    exercise_id = conn.execute(
        "SELECT exercise_id FROM exercise WHERE name = 'Weighted Pullups'"
    ).fetchone()["exercise_id"]
    session_id = conn.execute(
        "INSERT INTO session (date, session_type) VALUES ('2026-07-01', 'lifting')"
    ).lastrowid
    conn.execute(
        """
        INSERT INTO lift_set (session_id, exercise_id, set_index, weight_lb, reps)
        VALUES (?, ?, 1, NULL, 10)
        """,
        (session_id, exercise_id),
    )
    conn.commit()
    # Seeded bodyweight's latest entry is 146.0 on 2026-06-01.

    points = get_volume_trend(conn, "weighted pullups", "2026-07-01", "2026-07-01")
    assert points[0].tonnage_lb == pytest.approx(146.0 * 10)


# --------------------------------------------------------------------------
# get_prs / find_recent_prs / commit_prs
# --------------------------------------------------------------------------

def test_get_prs_returns_seeded_pr(conn):
    prs = get_prs(conn)
    assert len(prs) == 1
    assert prs[0].exercise == "Deadlift"
    assert prs[0].weight_lb == 385


def test_get_prs_filters_by_exercise(conn):
    assert get_prs(conn, exercise="bench press") == []


def test_find_recent_prs_detects_beat_and_skips_non_beat(conn):
    # Squat: 315x1 (2026-05-27) beats prior best 295x3 (2026-04-16) on e1RM.
    candidates = find_recent_prs(conn, "2026-05-01", "2026-05-31", exercise="squat")
    assert len(candidates) == 1
    assert candidates[0].weight_lb == 315
    assert candidates[0].previous_best_e1rm is not None


def test_find_recent_prs_narrow_window_excludes_it(conn):
    # Same squat PR, but querying a window that doesn't include 2026-05-27.
    candidates = find_recent_prs(conn, "2026-01-01", "2026-01-31", exercise="squat")
    assert candidates == []


def test_commit_prs_inserts_and_is_idempotent(conn):
    candidates = find_recent_prs(conn, "2026-05-01", "2026-05-31", exercise="squat")
    before = len(get_prs(conn))
    inserted = commit_prs(conn, candidates)
    assert inserted == 1
    assert len(get_prs(conn)) == before + 1

    # Re-accepting the same candidate is a no-op, not a duplicate insert.
    inserted_again = commit_prs(conn, candidates)
    assert inserted_again == 0
    assert len(get_prs(conn)) == before + 1


# --------------------------------------------------------------------------
# get_injuries / get_measurements / get_programs / get_block_outline
# --------------------------------------------------------------------------

def test_get_injuries_active_only(conn):
    assert len(get_injuries(conn)) == 1
    assert get_injuries(conn, active_only=True) == []  # seeded injury has an end_date


def test_get_measurements_filters_by_site(conn):
    arm = get_measurements(conn, site="arm")
    assert len(arm) == 3
    assert [m.value_in for m in arm] == [14.5, 14.75, 15.0]


def test_get_programs_status_filter(conn):
    assert len(get_programs(conn)) == 1
    assert get_programs(conn, status="draft") == []
    assert len(get_programs(conn, status="incomplete")) == 1


def test_get_block_outline_returns_ordered_slots(conn):
    strength_block_id = _block_id(conn, "Strength Block 1")
    outline = get_block_outline(conn, strength_block_id)
    assert [s.week_number for s in outline] == [1, 2, 3]
    assert outline[0].exercise == "Deadlift"


# --------------------------------------------------------------------------
# compare_programmed_vs_actual
# --------------------------------------------------------------------------

def test_compare_programmed_vs_actual_exact_match(conn):
    strength_block_id = _block_id(conn, "Strength Block 1")
    result = compare_programmed_vs_actual(conn, strength_block_id, exercise="deadlift")
    assert len(result.rows) == 1
    row = result.rows[0]
    assert row.target_weight_lb == 335
    assert row.actual_top_weight_lb == 335
    assert row.actual_top_reps == 2


def test_compare_programmed_vs_actual_missing_actual(conn):
    strength_block_id = _block_id(conn, "Strength Block 1")
    result = compare_programmed_vs_actual(conn, strength_block_id, exercise="bench press")
    assert len(result.rows) == 1
    assert result.rows[0].actual_top_weight_lb is None
    assert result.rows[0].actual_session_id is None


def test_compare_programmed_vs_actual_unmatched_actual_surfaced(conn):
    strength_block_id = _block_id(conn, "Strength Block 1")
    result = compare_programmed_vs_actual(conn, strength_block_id)
    unmatched_exercises = {u.exercise for u in result.unmatched_actual}
    assert "Weighted Pullups" in unmatched_exercises  # no programmed_slot for pullups


def test_compare_programmed_vs_actual_no_programmed_data_fallback(conn):
    hypertrophy_block_id = _block_id(conn, "Hypertrophy Phase 1")
    result = compare_programmed_vs_actual(conn, hypertrophy_block_id)
    assert result.rows == []
    assert result.unmatched_actual  # actual work still surfaced
    assert "No programmed data" in result.note


def test_compare_programmed_vs_actual_no_data_at_all(conn):
    empty_program_id = conn.execute(
        "INSERT INTO program (name, status) VALUES ('Empty', 'incomplete')"
    ).lastrowid
    empty_block_id = conn.execute(
        "INSERT INTO block (program_id, name) VALUES (?, 'Empty Block')",
        (empty_program_id,),
    ).lastrowid
    conn.commit()

    result = compare_programmed_vs_actual(conn, empty_block_id)
    assert result.rows == []
    assert result.unmatched_actual == []
    assert "No programmed data and no performed data" in result.note
