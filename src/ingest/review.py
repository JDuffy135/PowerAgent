"""Human-readable rendering of a `ParsedBatch` for HITL review (ARCHITECTURE.md §4.4).

`render_batch` is a pure function: `ParsedBatch` in, formatted text out. Step 4's
LangGraph `interrupt()` will show this string to the user before anything is
committed. Building it now, decoupled from the graph, keeps it testable.

Anything the parser was unsure about (`confidence < 1.0`) is visibly flagged with
a `CONFIDENCE_FLAG` marker so the reviewer's eye goes straight to it.
"""
from __future__ import annotations

from src.ingest.models import (
    ParsedBatch,
    ParsedCardio,
    ParsedProgrammedSlot,
    ParsedSession,
    ParsedSet,
)

CONFIDENCE_FLAG = "⚠"  # warning sign; precedes anything with confidence < 1.0


def _flag(confidence: float) -> str:
    """Return a visible marker for sub-1.0 confidence, else empty string."""
    if confidence < 1.0:
        return f"  {CONFIDENCE_FLAG} confidence {confidence:.2f}"
    return ""


def _fmt_weight(weight_lb: float | None) -> str:
    if weight_lb is None:
        return "BW"  # bodyweight-only / no load recorded
    if weight_lb == int(weight_lb):
        return f"{int(weight_lb)} lb"
    return f"{weight_lb} lb"


def _render_set(parsed_set: ParsedSet) -> str:
    parts = [f"set {parsed_set.set_index}: {_fmt_weight(parsed_set.weight_lb)}"]
    if parsed_set.reps is not None:
        parts.append(f"x{parsed_set.reps}")
    if parsed_set.rpe is not None:
        parts.append(f"@ RPE {parsed_set.rpe}")

    tags = [
        name
        for name, on in (
            ("top-set", parsed_set.is_top_set),
            ("paused", parsed_set.is_paused),
            ("amrap", parsed_set.is_amrap),
            ("failed", parsed_set.is_failed),
        )
        if on
    ]
    line = " ".join(parts)
    if tags:
        line += f"  [{', '.join(tags)}]"
    if parsed_set.equipment_note:
        line += f"  (equipment: {parsed_set.equipment_note})"
    line += f'  raw="{parsed_set.raw_text}"'
    return line + _flag(parsed_set.confidence)


def _render_slot(slot: ParsedProgrammedSlot) -> str:
    line = f"{slot.exercise_raw}: {slot.prescription}"
    if slot.target_weight_lb is not None:
        line += f" (target {_fmt_weight(slot.target_weight_lb)})"
    if slot.notes:
        line += f" -- {slot.notes}"
    return line + _flag(slot.confidence)


def _render_cardio(cardio: ParsedCardio) -> str:
    bits = [cardio.modality or "cardio"]
    if cardio.distance_mi is not None:
        bits.append(f"{cardio.distance_mi} mi")
    if cardio.duration_min is not None:
        bits.append(f"{cardio.duration_min} min")
    if cardio.intensity:
        bits.append(f"({cardio.intensity})")
    line = ", ".join(bits) + f'  raw="{cardio.raw_text}"'
    return line + _flag(cardio.confidence)


def _render_session(index: int, session: ParsedSession) -> list[str]:
    header = f"Session {index}: {session.date or '(no date)'}"
    if session.day_label:
        header += f"  [{session.day_label}]"
    header += f"  type={session.session_type}"
    lines = [header + _flag(session.confidence)]

    if session.sets:
        # Group sets under their raw exercise name, preserving first-seen order.
        by_exercise: dict[str, list[ParsedSet]] = {}
        for parsed_set in session.sets:
            by_exercise.setdefault(parsed_set.exercise_raw, []).append(parsed_set)
        lines.append("  Sets:")
        for exercise_raw, sets in by_exercise.items():
            resolved = "resolved" if sets[0].exercise_id is not None else "NEW EXERCISE"
            lines.append(f"    {exercise_raw}  [{resolved}]")
            for parsed_set in sets:
                lines.append(f"      {_render_set(parsed_set)}")

    if session.programmed_slots:
        lines.append("  Programmed (planned):")
        for slot in session.programmed_slots:
            lines.append(f"    {_render_slot(slot)}")

    if session.cardio:
        lines.append("  Cardio:")
        for cardio in session.cardio:
            lines.append(f"    {_render_cardio(cardio)}")

    return lines


def render_batch(parsed: ParsedBatch) -> str:
    """Render a `ParsedBatch` as a readable review summary.

    Sessions -> exercises -> sets, plus cardio, programmed slots, and any
    new-exercise candidates. Every field with `confidence < 1.0` is flagged so
    the reviewer can focus corrections there.
    """
    lines: list[str] = ["=== Ingest review ==="]

    n_sessions = len(parsed.sessions)
    n_sets = sum(len(s.sets) for s in parsed.sessions)
    lines.append(f"{n_sessions} session(s), {n_sets} set(s) parsed.")
    lines.append("")

    if not parsed.sessions:
        lines.append("(no sessions parsed)")
    for i, session in enumerate(parsed.sessions, start=1):
        lines.extend(_render_session(i, session))
        lines.append("")

    if parsed.new_exercise_candidates:
        lines.append("--- New exercise candidates (need confirmation) ---")
        for candidate in parsed.new_exercise_candidates:
            line = (
                f"  {candidate.raw_name!r} -> {candidate.suggested_name!r} "
                f"(tier={candidate.suggested_tier}, "
                f"muscle_group={candidate.suggested_muscle_group})"
            )
            lines.append(line + _flag(candidate.confidence))
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"
