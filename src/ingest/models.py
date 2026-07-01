"""Pydantic models for the LLM extraction pipeline (ARCHITECTURE.md §5.3).

`ParsedBatch` mirrors the SQLite schema (session -> sets/cardio/programmed_slot)
but stays in "raw" form: exercise names are unresolved strings until matched
against the exercise dictionary, and every field the parser had to guess at
carries a `confidence`. Nothing here is written to the DB -- that's Step 3's
HITL commit path (`stage_batch` / `commit_batch`).
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class NewExerciseCandidate(BaseModel):
    """An exercise name that didn't resolve to a known canonical exercise.

    Step 3's HITL review confirms/edits these before `add_exercise()` is ever
    called; this step only proposes them.
    """

    raw_name: str
    suggested_name: str
    suggested_tier: Literal["competition", "variation", "accessory"]
    suggested_muscle_group: str | None = None
    confidence: float = 1.0


class ParsedSet(BaseModel):
    """One performed set. Mirrors `lift_set` plus resolution/confidence metadata."""

    exercise_raw: str
    exercise_id: int | None = None  # filled in by resolve_exercise() when a conn is supplied
    set_index: int
    weight_lb: float | None = None  # canonical lb; see raw_text for the original unit/notation
    reps: int | None = None
    rpe: float | None = None
    is_paused: bool = False
    is_amrap: bool = False
    is_top_set: bool = False
    is_failed: bool = False
    equipment_note: str | None = None  # pin/plate configs, seat heights, etc.
    raw_text: str
    confidence: float = 1.0


class ParsedProgrammedSlot(BaseModel):
    """Planned/projected prescription, kept separate from performed `ParsedSet` rows
    so "projected vs actual" stays a join rather than a parsing problem."""

    exercise_raw: str
    exercise_id: int | None = None
    prescription: str
    target_weight_lb: float | None = None
    notes: str | None = None
    confidence: float = 1.0


class ParsedCardio(BaseModel):
    modality: str | None = None
    distance_mi: float | None = None
    duration_min: float | None = None
    intensity: str | None = None
    raw_text: str
    confidence: float = 1.0


class ParsedSession(BaseModel):
    date: str | None = None  # ISO-8601; None if the parser couldn't determine it
    day_label: str | None = None  # raw text as logged: 'w2d1', 'CARDIO', etc.
    week_number: int | None = None
    day_number: int | None = None
    session_type: Literal["lifting", "cardio", "other"] = "lifting"
    duration_min: int | None = None
    raw_note: str  # full original log text for this session (prose preserved as-is)
    sets: list[ParsedSet] = Field(default_factory=list)
    programmed_slots: list[ParsedProgrammedSlot] = Field(default_factory=list)
    cardio: list[ParsedCardio] = Field(default_factory=list)
    confidence: float = 1.0


class ParsedBatch(BaseModel):
    sessions: list[ParsedSession] = Field(default_factory=list)
    new_exercise_candidates: list[NewExerciseCandidate] = Field(default_factory=list)
