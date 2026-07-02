"""Correction-pass tests: full re-emit, re-resolution, and update_batch."""
from __future__ import annotations

import pytest

from src.ingest.correct import apply_correction
from src.ingest.models import NewExerciseCandidate, ParsedBatch, ParsedSession, ParsedSet
from src.ingest.stage import (
    BatchNotEditable,
    get_pending_batch,
    stage_batch,
    update_batch,
)
from src.ingest.commit import commit_batch


def _batch(weight=315.0, exercise="Squat") -> ParsedBatch:
    return ParsedBatch(
        sessions=[
            ParsedSession(
                date="2026-07-01",
                raw_note="squat day",
                sets=[
                    ParsedSet(
                        exercise_raw=exercise,
                        set_index=1,
                        weight_lb=weight,
                        reps=5,
                        raw_text=f"{weight}x5",
                    )
                ],
            )
        ]
    )


def _llm_returning(batch: ParsedBatch):
    prompts = []

    def _call(prompt: str) -> str:
        prompts.append(prompt)
        return batch.model_dump_json()

    _call.prompts = prompts
    return _call


def test_apply_correction_full_reemit_with_context_in_prompt(conn):
    original = _batch(315.0)
    llm = _llm_returning(_batch(320.0))

    corrected = apply_correction(original, "weight was 320", llm=llm, conn=conn)

    assert corrected.sessions[0].sets[0].weight_lb == 320.0
    prompt = llm.prompts[0]
    assert "315" in prompt            # original JSON included
    assert "weight was 320" in prompt  # user text included


def test_apply_correction_reresolves_exercise_ids(conn):
    llm = _llm_returning(_batch(315.0, exercise="Squat"))
    corrected = apply_correction(_batch(), "no-op", llm=llm, conn=conn)
    assert corrected.sessions[0].sets[0].exercise_id is not None


def test_apply_correction_drops_candidate_that_now_resolves(conn):
    # The LLM output renamed the exercise to a known alias but kept the stale candidate.
    reemitted = _batch(315.0, exercise="Squat")
    reemitted.new_exercise_candidates = [
        NewExerciseCandidate(
            raw_name="Squat", suggested_name="Squat", suggested_tier="competition"
        )
    ]
    corrected = apply_correction(_batch(), "fix the name", llm=_llm_returning(reemitted), conn=conn)
    assert corrected.new_exercise_candidates == []


def test_apply_correction_adds_candidate_for_new_unresolved_name(conn):
    corrected = apply_correction(
        _batch(),
        "that was actually a machine hack squat",
        llm=_llm_returning(_batch(exercise="Machine Hack Squat")),
        conn=conn,
    )
    assert corrected.sessions[0].sets[0].exercise_id is None
    assert [c.raw_name for c in corrected.new_exercise_candidates] == ["Machine Hack Squat"]


def test_apply_correction_invalid_output_raises(conn):
    with pytest.raises(ValueError):
        apply_correction(_batch(), "x", llm=lambda p: "not json", conn=conn)
    with pytest.raises(ValueError):
        apply_correction(_batch(), "x", llm=lambda p: '{"sessions": "wrong-type"}', conn=conn)


def test_update_batch_edits_pending_row(conn):
    batch_id = stage_batch(conn, _batch(315.0))
    update_batch(conn, batch_id, _batch(320.0))
    assert get_pending_batch(conn, batch_id).sessions[0].sets[0].weight_lb == 320.0


def test_update_batch_refuses_committed_row(conn):
    batch_id = stage_batch(conn, _batch())
    commit_batch(conn, batch_id, embed_prose=False)
    with pytest.raises(BatchNotEditable):
        update_batch(conn, batch_id, _batch(320.0))
