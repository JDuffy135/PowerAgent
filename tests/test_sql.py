import pytest

from src.tools.sql import ReadonlySQLError, run_readonly_sql


def test_run_readonly_sql_executes_select(conn):
    result = run_readonly_sql(conn, "SELECT date, weight_lb FROM bodyweight ORDER BY date")
    assert result.columns == ["date", "weight_lb"]
    assert result.rows[0]["date"] == "2026-01-03"
    assert result.truncated is False


@pytest.mark.parametrize(
    "bad_query",
    [
        "DELETE FROM bodyweight",
        "UPDATE bodyweight SET weight_lb = 0",
        "INSERT INTO bodyweight (date, weight_lb) VALUES ('2026-01-01', 1)",
        "SELECT 1; DROP TABLE bodyweight",
        "PRAGMA table_info(bodyweight)",
        "ATTACH DATABASE 'x' AS y",
        "not even sql",
    ],
)
def test_run_readonly_sql_rejects_non_select(conn, bad_query):
    with pytest.raises(ReadonlySQLError):
        run_readonly_sql(conn, bad_query)


def test_run_readonly_sql_does_not_mutate_data(conn):
    with pytest.raises(ReadonlySQLError):
        run_readonly_sql(conn, "DELETE FROM bodyweight")
    count = conn.execute("SELECT COUNT(*) AS n FROM bodyweight").fetchone()["n"]
    assert count == 9  # seeded rows untouched


def test_run_readonly_sql_query_only_does_not_leak(conn):
    run_readonly_sql(conn, "SELECT * FROM bodyweight")
    # PRAGMA query_only must be restored to OFF after the call.
    conn.execute("INSERT INTO bodyweight (date, weight_lb) VALUES ('2026-07-01', 150.0)")
    conn.commit()
    count = conn.execute("SELECT COUNT(*) AS n FROM bodyweight").fetchone()["n"]
    assert count == 10


def test_run_readonly_sql_row_cap_truncates(conn):
    result = run_readonly_sql(conn, "SELECT * FROM lift_set", max_rows=3)
    assert len(result.rows) == 3
    assert result.truncated is True


def test_run_readonly_sql_row_cap_not_truncated_when_under_limit(conn):
    result = run_readonly_sql(conn, "SELECT * FROM bodyweight", max_rows=100)
    assert result.truncated is False
