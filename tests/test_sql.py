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


def test_run_readonly_sql_works_outside_main_thread():
    """Streamlit executes reruns in a worker thread, where signal-based
    timeouts are unavailable (signal.signal is main-thread-only). The query
    must still run -- the timeout guard degrades to a no-op."""
    import threading

    from src.db.connection import get_conn, init_db
    from src.seed import seed

    # Same connection mode the Streamlit app uses (reruns hop threads).
    threaded_conn = get_conn(":memory:", check_same_thread=False)
    init_db(threaded_conn)
    seed(threaded_conn)

    results: dict = {}

    def worker():
        try:
            results["result"] = run_readonly_sql(
                threaded_conn, "SELECT COUNT(*) AS n FROM bodyweight"
            )
        except Exception as exc:  # pragma: no cover - the failure being tested
            results["error"] = exc

    thread = threading.Thread(target=worker)
    thread.start()
    thread.join()
    threaded_conn.close()

    assert "error" not in results, f"raised in worker thread: {results.get('error')}"
    assert results["result"].rows[0]["n"] == 9
