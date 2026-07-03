"""Tests for the HITL review renderer (Step 3)."""
from __future__ import annotations

from src.ingest.models import (
    NewExerciseCandidate,
    ParsedBatch,
    ParsedCardio,
    ParsedProgrammedSlot,
    ParsedSession,
    ParsedSet,
)
from src.ingest.review import CONFIDENCE_FLAG, render_batch


def test_render_shows_sessions_exercises_and_sets():
    batch = ParsedBatch(
        sessions=[
            ParsedSession(
                date="2026-07-01",
                day_label="w1d1",
                raw_note="squat day",
                sets=[
                    ParsedSet(
                        exercise_raw="Squat",
                        exercise_id=2,
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
    out = render_batch(batch)
    assert "2026-07-01" in out
    assert "Squat" in out
    assert "top-set" in out
    assert "1 session(s), 1 set(s)" in out


def test_low_confidence_fields_are_flagged():
    batch = ParsedBatch(
        sessions=[
            ParsedSession(
                raw_note="mystery",
                sets=[
                    ParsedSet(
                        exercise_raw="Bench",
                        set_index=1,
                        weight_lb=225.0,
                        reps=3,
                        raw_text="225x3",
                        confidence=0.6,
                    )
                ],
            )
        ]
    )
    out = render_batch(batch)
    assert CONFIDENCE_FLAG in out
    assert "0.60" in out


def test_full_confidence_has_no_flag():
    batch = ParsedBatch(
        sessions=[
            ParsedSession(
                raw_note="clean",
                sets=[
                    ParsedSet(
                        exercise_raw="Bench",
                        set_index=1,
                        weight_lb=225.0,
                        reps=3,
                        raw_text="225x3",
                        confidence=1.0,
                    )
                ],
            )
        ]
    )
    assert CONFIDENCE_FLAG not in render_batch(batch)


def test_cardio_and_programmed_slots_rendered():
    batch = ParsedBatch(
        sessions=[
            ParsedSession(
                session_type="cardio",
                raw_note="bike",
                cardio=[ParsedCardio(modality="bike", duration_min=26, raw_text="bike 26 min")],
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
    out = render_batch(batch)
    assert "bike" in out
    assert "1x3 @ RPE 8" in out


def test_new_exercise_candidates_rendered_and_flagged():
    batch = ParsedBatch(
        new_exercise_candidates=[
            NewExerciseCandidate(
                raw_name="Bulgarian Split Squats",
                suggested_name="Bulgarian Split Squat",
                suggested_tier="accessory",
                confidence=0.5,
            )
        ]
    )
    out = render_batch(batch)
    assert "Bulgarian Split Squats" in out
    assert "New exercise candidates" in out
    assert CONFIDENCE_FLAG in out


# ---------------------------------------------------------------------------
# display_unit (Stage 11a) -- kg is presentation-only; lb stays canonical
# ---------------------------------------------------------------------------

def _one_set_batch() -> ParsedBatch:
    return ParsedBatch(
        sessions=[
            ParsedSession(
                date="2026-07-01",
                raw_note="squat",
                sets=[
                    ParsedSet(
                        exercise_raw="Squat",
                        exercise_id=2,
                        set_index=1,
                        weight_lb=225.0,
                        reps=5,
                        raw_text="225x5",
                    )
                ],
                programmed_slots=[
                    ParsedProgrammedSlot(
                        exercise_raw="Squat",
                        prescription="1x1 @ RPE 8",
                        target_weight_lb=315.0,
                    )
                ],
            )
        ]
    )


def test_render_default_unit_is_lb():
    out = render_batch(_one_set_batch())
    assert "225 lb" in out
    assert "315 lb" in out
    assert "kg" not in out


def test_render_kg_converts_sets_and_slots():
    out = render_batch(_one_set_batch(), unit="kg")
    # 225 lb -> 102.1 kg, 315 lb -> 142.9 kg (presentation only).
    assert "102.1 kg" in out
    assert "142.9 kg" in out
    assert "lb" not in out
