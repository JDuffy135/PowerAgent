"""Dev-tools CRUD / backup / batch-browser ops (Stage 9)."""
from __future__ import annotations

import sqlite3

import pytest

from src.ingest.models import ParsedBatch, ParsedSession
from src.ingest.stage import stage_batch
from src.tools.admin import (
    AdminError,
    backup_db,
    count_rows,
    delete_row,
    fetch_table,
    get_batch_json,
    insert_row,
    list_batches,
    table_columns,
    update_row,
)


# ---------------------------------------------------------------------------
# Table CRUD
# ---------------------------------------------------------------------------

def test_fetch_table_and_columns(conn):
    cols = table_columns(conn, "bodyweight")
    assert cols == ["bw_id", "date", "weight_lb", "note"]
    rows = fetch_table(conn, "bodyweight", limit=5)
    assert rows and set(rows[0]) == set(cols)


def test_non_allowlisted_table_rejected(conn):
    for op in (lambda: fetch_table(conn, "ingest_batch"),
               lambda: fetch_table(conn, "sqlite_master"),
               lambda: insert_row(conn, "nope", {"x": 1})):
        with pytest.raises(AdminError):
            op()


def test_insert_update_delete_round_trip(conn):
    before = count_rows(conn, "bodyweight")
    new_id = insert_row(conn, "bodyweight", {"date": "2026-07-02", "weight_lb": 147.2})
    assert count_rows(conn, "bodyweight") == before + 1

    update_row(conn, "bodyweight", new_id, {"weight_lb": 147.6, "note": "am"})
    row = conn.execute("SELECT * FROM bodyweight WHERE bw_id = ?", (new_id,)).fetchone()
    assert row["weight_lb"] == 147.6
    assert row["note"] == "am"

    delete_row(conn, "bodyweight", new_id)
    assert count_rows(conn, "bodyweight") == before


def test_unknown_column_and_row_rejected(conn):
    with pytest.raises(AdminError):
        insert_row(conn, "bodyweight", {"date": "2026-07-02", "weight_lb": 1, "bogus": 5})
    with pytest.raises(AdminError):
        update_row(conn, "bodyweight", 999999, {"weight_lb": 100})
    with pytest.raises(AdminError):
        delete_row(conn, "bodyweight", 999999)


def test_fk_violation_surfaces_as_admin_error(conn):
    # A session with lift_set children can't be deleted before its sets.
    session_id = conn.execute(
        "SELECT session_id FROM lift_set LIMIT 1"
    ).fetchone()[0]
    with pytest.raises(AdminError):
        delete_row(conn, "session", session_id)


def test_text_pk_table(conn):
    exercise_id = conn.execute("SELECT exercise_id FROM exercise LIMIT 1").fetchone()[0]
    insert_row(conn, "exercise_alias", {"alias": "test alias xyz", "exercise_id": exercise_id})
    delete_row(conn, "exercise_alias", "test alias xyz")
    assert conn.execute(
        "SELECT COUNT(*) FROM exercise_alias WHERE alias = 'test alias xyz'"
    ).fetchone()[0] == 0


# ---------------------------------------------------------------------------
# Backup
# ---------------------------------------------------------------------------

def test_backup_db_snapshot(conn, tmp_path):
    dest = backup_db(conn, tmp_path)
    assert dest.exists() and dest.name.startswith("training-backup-")
    copy = sqlite3.connect(dest)
    try:
        n = copy.execute("SELECT COUNT(*) FROM session").fetchone()[0]
    finally:
        copy.close()
    assert n == count_rows(conn, "session")


# ---------------------------------------------------------------------------
# Ingest-batch audit browser
# ---------------------------------------------------------------------------

def test_list_batches_and_json(conn):
    parsed = ParsedBatch(sessions=[ParsedSession(date="2026-06-01", raw_note="squats")])
    batch_id = stage_batch(conn, parsed, source_file="w1d1.txt")

    batches = list_batches(conn)
    top = batches[0]
    assert top.batch_id == batch_id
    assert top.status == "pending_review"
    assert top.source_file == "w1d1.txt"
    assert top.n_sessions == 1

    assert list_batches(conn, status="committed") == [
        b for b in batches if b.status == "committed"
    ]

    pretty = get_batch_json(conn, batch_id)
    assert '"raw_note": "squats"' in pretty
    with pytest.raises(AdminError):
        get_batch_json(conn, 999999)
