"""Golden-file tests for the extraction pipeline (HANDOFF_STEP_2.md).

There's no live Ollama server in CI/dev sandboxes, so these tests inject a
stub LLM that returns each fixture's golden JSON verbatim -- exercising the
real pipeline (JSON parsing, Pydantic validation, exercise resolution,
new-exercise-candidate surfacing) without depending on a running model.
`extract_training_data`'s `llm` parameter exists for exactly this reason.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.ingest.extract import extract_training_data
from src.ingest.models import ParsedBatch

FIXTURES_DIR = Path(__file__).parent / "fixtures"

GOLDEN_CASES = [
    "kg_lb_mixed",
    "projected_vs_actual",
    "pin_plate_config",
    "top_single_backoffs",
    "skipped_exercise",
    "slang_emoji",
    "unresolved_exercise",
]


def _load_fixture(name: str) -> tuple[str, dict]:
    raw_text = (FIXTURES_DIR / f"{name}.txt").read_text(encoding="utf-8").rstrip("\n")
    golden = json.loads((FIXTURES_DIR / f"{name}.json").read_text(encoding="utf-8"))
    return raw_text, golden


@pytest.mark.parametrize("name", GOLDEN_CASES)
def test_golden_fixture_parses_to_expected_batch(name):
    raw_text, golden = _load_fixture(name)
    stub_llm = lambda _prompt: json.dumps(golden)

    batch = extract_training_data(raw_text, llm=stub_llm)

    assert batch == ParsedBatch.model_validate(golden)


def test_kg_normalizes_to_lb():
    raw_text, golden = _load_fixture("kg_lb_mixed")
    batch = extract_training_data(raw_text, llm=lambda _p: json.dumps(golden))

    set_row = batch.sessions[0].sets[0]
    assert set_row.weight_lb == pytest.approx(374.8, abs=0.05)
    assert "170KG" in set_row.raw_text


def test_projected_vs_actual_split_into_slot_and_set():
    raw_text, golden = _load_fixture("projected_vs_actual")
    batch = extract_training_data(raw_text, llm=lambda _p: json.dumps(golden))

    session = batch.sessions[0]
    assert session.programmed_slots[0].target_weight_lb == pytest.approx(374.8, abs=0.05)
    assert session.sets[0].weight_lb == 375.0


def test_pin_plate_config_keeps_equipment_note_and_weight():
    raw_text, golden = _load_fixture("pin_plate_config")
    batch = extract_training_data(raw_text, llm=lambda _p: json.dumps(golden))

    sets = batch.sessions[0].sets
    assert [s.weight_lb for s in sets] == [143.0, 121.0, 121.0]
    assert all(s.equipment_note for s in sets)


def test_top_single_flagged_and_backoffs_are_not():
    raw_text, golden = _load_fixture("top_single_backoffs")
    batch = extract_training_data(raw_text, llm=lambda _p: json.dumps(golden))

    sets = batch.sessions[0].sets
    assert sets[0].is_top_set is True
    assert all(not s.is_top_set for s in sets[1:])


def test_skipped_exercise_produces_no_set_rows_but_keeps_note():
    raw_text, golden = _load_fixture("skipped_exercise")
    batch = extract_training_data(raw_text, llm=lambda _p: json.dumps(golden))

    session = batch.sessions[0]
    assert all(s.exercise_raw != "Leg Press" for s in session.sets)
    assert "N/A" in session.raw_note
    assert len(session.sets) == 3  # squat sets still present


def test_slang_and_emoji_preserved_but_numbers_unaffected():
    raw_text, golden = _load_fixture("slang_emoji")
    batch = extract_training_data(raw_text, llm=lambda _p: json.dumps(golden))

    session = batch.sessions[0]
    assert "🔥" in session.raw_note
    assert "sintiendo fuerte" in session.raw_note
    assert session.sets[0].weight_lb == 225.0
    assert session.sets[0].reps == 3


def test_resolve_exercise_fills_exercise_id_for_known_names(conn):
    raw_text, golden = _load_fixture("top_single_backoffs")  # "Squat" -> alias of Low Bar Squat
    batch = extract_training_data(raw_text, conn=conn, llm=lambda _p: json.dumps(golden))

    for lift_set in batch.sessions[0].sets:
        assert lift_set.exercise_id is not None
    assert batch.new_exercise_candidates == []


def test_unresolved_exercise_becomes_new_exercise_candidate(conn):
    raw_text, golden = _load_fixture("unresolved_exercise")  # "Bulgarian Split Squats" isn't seeded
    batch = extract_training_data(raw_text, conn=conn, llm=lambda _p: json.dumps(golden))

    for lift_set in batch.sessions[0].sets:
        assert lift_set.exercise_id is None

    assert len(batch.new_exercise_candidates) == 1
    candidate = batch.new_exercise_candidates[0]
    assert candidate.raw_name == "Bulgarian Split Squats"


def test_no_llm_call_writes_to_db(conn):
    """extract_training_data must never mutate the DB (HANDOFF_STEP_2.md: 'keep pure')."""
    raw_text, golden = _load_fixture("unresolved_exercise")
    before = conn.execute("SELECT COUNT(*) AS n FROM exercise").fetchone()["n"]

    extract_training_data(raw_text, conn=conn, llm=lambda _p: json.dumps(golden))

    after = conn.execute("SELECT COUNT(*) AS n FROM exercise").fetchone()["n"]
    assert before == after


def test_invalid_json_from_llm_raises_value_error():
    with pytest.raises(ValueError):
        extract_training_data("some log text", llm=lambda _p: "not json")


def test_schema_invalid_payload_raises_value_error():
    with pytest.raises(ValueError):
        extract_training_data("some log text", llm=lambda _p: json.dumps({"sessions": "not-a-list"}))
