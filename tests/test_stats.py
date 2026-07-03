"""UPDATE_STATS write-tool tests against the seeded in-memory DB."""
from __future__ import annotations

from src.tools.resolve import resolve_exercise
from src.tools.stats import insert_bodyweight, insert_pr


def test_insert_bodyweight(conn):
    before = conn.execute("SELECT COUNT(*) AS c FROM bodyweight").fetchone()["c"]
    bw_id = insert_bodyweight(conn, "2026-07-02", 147.5, note="morning")
    after = conn.execute("SELECT COUNT(*) AS c FROM bodyweight").fetchone()["c"]
    assert after == before + 1
    row = conn.execute("SELECT * FROM bodyweight WHERE bw_id = ?", (bw_id,)).fetchone()
    assert row["weight_lb"] == 147.5
    assert row["note"] == "morning"


def test_insert_pr(conn):
    ex = resolve_exercise(conn, "deadlift")
    pr_id = insert_pr(conn, "2026-07-02", ex.exercise_id, 405.0, 1, context="gym")
    row = conn.execute("SELECT * FROM pr WHERE pr_id = ?", (pr_id,)).fetchone()
    assert row["weight_lb"] == 405.0
    assert row["reps"] == 1
    assert row["exercise_id"] == ex.exercise_id
    assert row["context"] == "gym"
