"""HITL correction pass: user free-text edits -> corrected `ParsedBatch` (ARCHITECTURE.md §4.4).

**[DECISION]** The correction LLM re-emits the *full* batch (original JSON + the
user's correction text in the prompt) rather than a patch/diff: simpler contract,
and the whole result re-validates against `ParsedBatch` in one shot. The
correct→re-render loop is capped by the ingest node (`MAX_CORRECTION_ROUNDS`),
not here.

Like extraction, this is pure with respect to the DB: `conn` is only used
read-only via `resolve_exercise` to re-tag `exercise_id`s on the re-emitted
batch (the LLM cannot be trusted to preserve them). Persisting the corrected
batch is `stage.update_batch`'s job.
"""
from __future__ import annotations

import json
import sqlite3

from pydantic import ValidationError

from src.ingest.extract import LLMCallable, get_llm
from src.ingest.models import NewExerciseCandidate, ParsedBatch
from src.tools.resolve import _normalize, resolve_exercise

CORRECTION_SYSTEM_PROMPT = """You are correcting a previously parsed powerlifting training-log batch. \
You are given the current parsed batch as JSON and the user's correction request. \
Re-emit the FULL corrected batch as a JSON object matching the exact same schema. \
Output ONLY the JSON object -- no prose, no markdown code fences.

Rules:
- Apply ONLY the changes the user asked for; keep every other field identical to the input.
- Weights stay in pounds (convert kg with 1 kg = 2.20462 lb, 1 decimal place).
- Never invent or drop sessions/sets the user didn't mention.
- Preserve `raw_text`/`raw_note` fields verbatim unless the user explicitly corrects them.
- If the user renames an exercise, update `exercise_raw` (and the matching
  new-exercise candidate if one exists); a separate step re-resolves canonical IDs.
"""


def apply_correction(
    batch: ParsedBatch,
    user_text: str,
    llm: LLMCallable | None = None,
    conn: sqlite3.Connection | None = None,
) -> ParsedBatch:
    """Apply the user's free-text correction to `batch` via a full LLM re-emit.

    Raises `ValueError` if the LLM output is not valid JSON or fails schema
    validation (the caller keeps the previous staged batch in that case).
    """
    call = llm if llm is not None else get_llm(
        "ingest_correct", system_prompt=CORRECTION_SYSTEM_PROMPT
    )

    prompt = (
        f"Current parsed batch JSON:\n{batch.model_dump_json(indent=2)}\n\n"
        f"User correction request:\n{user_text}"
    )
    raw_response = call(prompt)
    try:
        payload = json.loads(raw_response)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Correction pass did not return valid JSON: {exc}") from exc

    try:
        corrected = ParsedBatch.model_validate(payload)
    except ValidationError as exc:
        raise ValueError(f"Correction pass output failed schema validation: {exc}") from exc

    if conn is not None:
        _reresolve_exercises(conn, corrected)

    return corrected


def _reresolve_exercises(conn: sqlite3.Connection, batch: ParsedBatch) -> None:
    """Re-tag `exercise_id` on every set/slot of the re-emitted batch.

    Unlike extraction's first pass, the LLM-emitted batch may already carry
    new-exercise candidates (possibly user-edited via the correction), so we
    keep those and only append candidates for unresolved names not yet covered.
    Candidates whose raw name now resolves (e.g. the user corrected a typo to a
    known exercise) are dropped.
    """
    known_keys = {_normalize(c.raw_name) for c in batch.new_exercise_candidates}
    appended: dict[str, NewExerciseCandidate] = {}

    def resolve(raw_name: str) -> int | None:
        resolved = resolve_exercise(conn, raw_name)
        if resolved is not None:
            return resolved.exercise_id
        key = _normalize(raw_name)
        if key and key not in known_keys and key not in appended:
            appended[key] = NewExerciseCandidate(
                raw_name=raw_name,
                suggested_name=raw_name.strip(),
                suggested_tier="accessory",
                suggested_muscle_group=None,
                confidence=0.5,
            )
        return None

    for session in batch.sessions:
        for parsed_set in session.sets:
            parsed_set.exercise_id = resolve(parsed_set.exercise_raw)
        for slot in session.programmed_slots:
            slot.exercise_id = resolve(slot.exercise_raw)

    batch.new_exercise_candidates = [
        c for c in batch.new_exercise_candidates
        if resolve_exercise(conn, c.raw_name) is None
    ]
    batch.new_exercise_candidates.extend(appended.values())
