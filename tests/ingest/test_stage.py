"""Tests for HITL staging: the pending-review audit-trail row (Step 3)."""
from __future__ import annotations

import pytest

from src.ingest.models import ParsedBatch, ParsedSession, ParsedSet
from src.ingest.stage import BatchNotFound, get_pending_batch, stage_batch


def _sample_batch() -> ParsedBatch:
    return ParsedBatch(
        sessions=[
            ParsedSession(
                date="2026-07-01",
                day_label="w1d1",
                session_type="lifting",
                raw_note="Squat: 315x1",
                sets=[
                    ParsedSet(
                        exercise_raw="Squat",
                        set_index=1,
                        weight_lb=315.0,
                        reps=1,
                        is_top_set=True,
                        raw_text="315x1",
                    )
                ],
            )
        ]
    )


def test_stage_batch_writes_pending_review_row(conn):
    batch_id = stage_batch(conn, _sample_batch(), source_file="log.txt")

    row = conn.execute(
        "SELECT status, source_file, parsed_json FROM ingest_batch WHERE batch_id = ?",
        (batch_id,),
    ).fetchone()
    assert row["status"] == "pending_review"
    assert row["source_file"] == "log.txt"
    assert row["parsed_json"]  # non-empty serialized JSON


def test_stage_batch_writes_no_training_rows(conn):
    """Staging is the audit row only -- no session/lift_set writes."""
    before = conn.execute("SELECT COUNT(*) AS n FROM session").fetchone()["n"]
    stage_batch(conn, _sample_batch())
    after = conn.execute("SELECT COUNT(*) AS n FROM session").fetchone()["n"]
    assert before == after


def test_get_pending_batch_round_trips(conn):
    original = _sample_batch()
    batch_id = stage_batch(conn, original)

    rehydrated = get_pending_batch(conn, batch_id)
    assert rehydrated == original


def test_get_pending_batch_unknown_id_raises(conn):
    with pytest.raises(BatchNotFound):
        get_pending_batch(conn, 999)
