"""Tests for the transactional commit path (Step 3).

Covers the round-trip into SQLite (queryable via the Step 1 tools), new-exercise
creation without duplication, per-set equipment_note preservation, transactional
rollback on a broken batch, double-commit no-op, and the reject path.
"""
from __future__ import annotations

import pytest

from src.ingest.commit import (
    BatchNotCommittable,
    UnresolvedExercise,
    commit_batch,
    reject_batch,
)
from src.ingest.models import (
    NewExerciseCandidate,
    ParsedBatch,
    ParsedCardio,
    ParsedSession,
    ParsedSet,
)
from src.ingest.stage import stage_batch
from src.tools.queries import get_best_set, get_lifts
from src.tools.resolve import resolve_exercise


def _known_batch() -> ParsedBatch:
    """A squat session using a known (seeded) exercise alias."""
    return ParsedBatch(
        sessions=[
            ParsedSession(
                date="2026-07-01",
                day_label="w1d1",
                session_type="lifting",
                raw_note="Squat: 315x1 then backoffs 🔥",
                sets=[
                    ParsedSet(
                        exercise_raw="Squat",
                        set_index=1,
                        weight_lb=315.0,
                        reps=1,
                        is_top_set=True,
                        raw_text="315x1",
                    ),
                    ParsedSet(
                        exercise_raw="Squat",
                        set_index=2,
                        weight_lb=275.0,
                        reps=5,
                        equipment_note="safety pins at hole 5",
                        raw_text="275x5",
                        confidence=0.8,
                    ),
                ],
            )
        ]
    )


def test_commit_round_trips_into_sqlite(conn):
    batch_id = stage_batch(conn, _known_batch())
    result = commit_batch(conn, batch_id, embed_prose=False)

    assert result.committed is True
    assert result.sessions_created == 1
    assert result.sets_created == 2
    assert result.exercises_created == 0  # "Squat" already resolves

    # Queryable via the Step 1 tools.
    best = get_best_set(conn, "squat", "2026-07-01", "2026-07-01")
    assert best.weight_lb == 315.0
    assert best.reps == 1

    lifts = get_lifts(conn, "squat", "2026-07-01", "2026-07-01")
    assert len(lifts) == 1
    assert len(lifts[0].sets) == 2

    status = conn.execute(
        "SELECT status FROM ingest_batch WHERE batch_id = ?", (batch_id,)
    ).fetchone()["status"]
    assert status == "committed"


def test_equipment_note_survives_round_trip(conn):
    batch_id = stage_batch(conn, _known_batch())
    commit_batch(conn, batch_id, embed_prose=False)

    lifts = get_lifts(conn, "squat", "2026-07-01", "2026-07-01")
    notes = [s.equipment_note for s in lifts[0].sets]
    assert "safety pins at hole 5" in notes


def test_new_exercise_candidate_is_created_not_duplicated(conn):
    batch = ParsedBatch(
        sessions=[
            ParsedSession(
                date="2026-07-02",
                raw_note="BSS: 3x10 @ 40",
                sets=[
                    ParsedSet(
                        exercise_raw="Bulgarian Split Squats",
                        set_index=i + 1,
                        weight_lb=40.0,
                        reps=10,
                        raw_text="3x10 @ 40",
                    )
                    for i in range(3)
                ],
            )
        ],
        new_exercise_candidates=[
            NewExerciseCandidate(
                raw_name="Bulgarian Split Squats",
                suggested_name="Bulgarian Split Squat",
                suggested_tier="accessory",
                suggested_muscle_group="quads",
            )
        ],
    )
    before = conn.execute("SELECT COUNT(*) AS n FROM exercise").fetchone()["n"]

    batch_id = stage_batch(conn, batch)
    result = commit_batch(conn, batch_id, embed_prose=False)

    after = conn.execute("SELECT COUNT(*) AS n FROM exercise").fetchone()["n"]
    assert result.exercises_created == 1
    assert after == before + 1
    assert resolve_exercise(conn, "Bulgarian Split Squats") is not None

    # Re-committing the same exercise via a second batch must not duplicate it.
    batch2_id = stage_batch(conn, batch)
    result2 = commit_batch(conn, batch2_id, embed_prose=False)
    assert result2.exercises_created == 0
    final = conn.execute("SELECT COUNT(*) AS n FROM exercise").fetchone()["n"]
    assert final == before + 1


def test_broken_batch_rolls_back_transactionally(conn):
    """An unresolved exercise with no candidate must abort the whole commit and
    leave the DB + batch status untouched."""
    batch = ParsedBatch(
        sessions=[
            ParsedSession(
                date="2026-07-03",
                raw_note="mystery lift",
                sets=[
                    ParsedSet(
                        exercise_raw="Totally Unknown Lift",
                        set_index=1,
                        weight_lb=100.0,
                        reps=5,
                        raw_text="100x5",
                    )
                ],
            )
        ]
        # NOTE: no new_exercise_candidates -> unresolvable.
    )
    sessions_before = conn.execute("SELECT COUNT(*) AS n FROM session").fetchone()["n"]

    batch_id = stage_batch(conn, batch)
    with pytest.raises(UnresolvedExercise):
        commit_batch(conn, batch_id, embed_prose=False)

    sessions_after = conn.execute("SELECT COUNT(*) AS n FROM session").fetchone()["n"]
    assert sessions_after == sessions_before  # rolled back, nothing inserted

    status = conn.execute(
        "SELECT status FROM ingest_batch WHERE batch_id = ?", (batch_id,)
    ).fetchone()["status"]
    assert status == "pending_review"  # status untouched


def test_double_commit_is_a_noop(conn):
    batch_id = stage_batch(conn, _known_batch())
    commit_batch(conn, batch_id, embed_prose=False)
    sessions_after_first = conn.execute("SELECT COUNT(*) AS n FROM session").fetchone()["n"]

    result2 = commit_batch(conn, batch_id, embed_prose=False)
    assert result2.committed is False
    assert result2.sets_created == 0

    sessions_after_second = conn.execute("SELECT COUNT(*) AS n FROM session").fetchone()["n"]
    assert sessions_after_second == sessions_after_first  # no double-insert


def test_reject_writes_nothing_and_is_idempotent(conn):
    batch_id = stage_batch(conn, _known_batch())
    sessions_before = conn.execute("SELECT COUNT(*) AS n FROM session").fetchone()["n"]

    assert reject_batch(conn, batch_id) is True
    assert reject_batch(conn, batch_id) is False  # idempotent no-op

    sessions_after = conn.execute("SELECT COUNT(*) AS n FROM session").fetchone()["n"]
    assert sessions_after == sessions_before

    status = conn.execute(
        "SELECT status FROM ingest_batch WHERE batch_id = ?", (batch_id,)
    ).fetchone()["status"]
    assert status == "rejected"


def test_commit_after_reject_raises(conn):
    batch_id = stage_batch(conn, _known_batch())
    reject_batch(conn, batch_id)
    with pytest.raises(BatchNotCommittable):
        commit_batch(conn, batch_id, embed_prose=False)


def test_programmed_slots_skipped_but_preserved_in_audit(conn):
    from src.ingest.models import ParsedProgrammedSlot
    from src.ingest.stage import get_pending_batch

    batch = ParsedBatch(
        sessions=[
            ParsedSession(
                date="2026-07-04",
                raw_note="bench w/ plan",
                sets=[
                    ParsedSet(
                        exercise_raw="Bench Press",
                        set_index=1,
                        weight_lb=225.0,
                        reps=3,
                        raw_text="225x3",
                    )
                ],
                programmed_slots=[
                    ParsedProgrammedSlot(
                        exercise_raw="Bench Press",
                        prescription="1x3 @ RPE 8",
                        target_weight_lb=225.0,
                    )
                ],
            )
        ]
    )
    before = conn.execute("SELECT COUNT(*) AS n FROM programmed_slot").fetchone()["n"]
    batch_id = stage_batch(conn, batch)
    result = commit_batch(conn, batch_id, embed_prose=False)

    assert result.programmed_slots_skipped == 1
    after = conn.execute("SELECT COUNT(*) AS n FROM programmed_slot").fetchone()["n"]
    assert after == before  # commit must not insert any programmed_slot rows
    # Still preserved verbatim in the audit trail.
    assert get_pending_batch(conn, batch_id).sessions[0].programmed_slots[0].prescription == "1x3 @ RPE 8"


def test_cardio_committed(conn):
    batch = ParsedBatch(
        sessions=[
            ParsedSession(
                date="2026-07-05",
                session_type="cardio",
                duration_min=26,
                raw_note="easy bike",
                cardio=[
                    ParsedCardio(
                        modality="bike", duration_min=26, intensity="light", raw_text="bike 26 min"
                    )
                ],
            )
        ]
    )
    batch_id = stage_batch(conn, batch)
    result = commit_batch(conn, batch_id, embed_prose=False)
    assert result.cardio_created == 1
    assert conn.execute("SELECT COUNT(*) AS n FROM cardio").fetchone()["n"] >= 1


def test_commit_embeds_prose_with_fake_backend(conn, fake_embedder, chroma_client):
    batch_id = stage_batch(conn, _known_batch())
    result = commit_batch(
        conn, batch_id, embedder=fake_embedder, chroma_client=chroma_client
    )
    assert result.notes_embedded == 1

    collection = chroma_client.get_collection("personal_notes")
    assert collection.count() == 1
    meta = collection.get(include=["metadatas"])["metadatas"][0]
    assert meta["doc_type"] == "session_note"
    assert "Low Bar Squat" in meta["exercises"]  # "Squat" resolved to canonical name
