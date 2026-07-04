"""Dev Tools tab widget flows via `streamlit.testing.v1.AppTest`.

These run the real Streamlit script lifecycle (session_state, reruns, widget
keys), which import-smoke tests can't cover -- the danger-zone reset bug
(writing a widget's session_state key after the widget was instantiated) only
reproduces here.
"""
from __future__ import annotations

import sqlite3

import pytest
from streamlit.testing.v1 import AppTest

from src.db.connection import get_conn, init_db
from src.seed import seed


@pytest.fixture()
def db_path(tmp_path):
    """Seeded on-disk DB: AppTest runs the script in its own thread, so the
    connection must be created there, not shared from the test thread."""
    path = tmp_path / "apptest.db"
    connection = get_conn(str(path))
    init_db(connection)
    seed(connection)
    connection.close()
    return str(path)


def _app(db_path: str) -> None:
    import sqlite3

    from src.ui import devtools_tab

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    devtools_tab.render(conn)


def _run(db_path: str) -> AppTest:
    at = AppTest.from_function(_app, args=(db_path,), default_timeout=30)
    at.run()
    assert not at.exception
    return at


def _clear_button(at: AppTest):
    return next(b for b in at.button if "Clear all training data" in (b.label or ""))


def test_danger_zone_button_gated_on_phrase(db_path):
    at = _run(db_path)
    assert _clear_button(at).disabled

    at.text_input(key="clear_all_phrase").set_value("wrong phrase").run()
    assert not at.exception
    assert _clear_button(at).disabled

    at.text_input(key="clear_all_phrase").set_value("delete all data").run()
    assert not at.exception
    assert not _clear_button(at).disabled


def test_danger_zone_wipe_resets_input_and_shows_flash(db_path):
    at = _run(db_path)
    at.text_input(key="clear_all_phrase").set_value("delete all data").run()
    _clear_button(at).click().run()
    assert not at.exception  # the widget-key reset must not raise

    check = sqlite3.connect(db_path)
    try:
        for table in ("program", "block", "session", "lift_set", "pr", "bodyweight"):
            assert check.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] == 0
    finally:
        check.close()

    # Success message survives the rerun; phrase input is cleared again.
    assert any("Cleared" in s.value for s in at.success)
    assert not at.text_input(key="clear_all_phrase").value
    assert _clear_button(at).disabled
