import sqlite3

import pytest


def test_schema_loads_and_seed_present(conn):
    row = conn.execute("SELECT COUNT(*) AS n FROM exercise").fetchone()
    assert row["n"] == 10

    row = conn.execute("SELECT COUNT(*) AS n FROM session").fetchone()
    assert row["n"] >= 10


def test_foreign_key_violation_raises(conn):
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            """
            INSERT INTO lift_set (session_id, exercise_id, set_index, weight_lb, reps)
            VALUES (999999, 1, 1, 100, 5)
            """
        )
