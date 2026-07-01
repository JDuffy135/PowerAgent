import pytest

from src.tools.queries import (
    ExerciseNotFound,
    _epley,
    get_best_set,
    get_bodyweight_trend,
    get_e1rm_trend,
    get_lifts,
)


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
