"""Persist an approved GENERATE draft as a `draft` program (ARCHITECTURE.md §4.2).

`persist_draft` is the single durable write of the GENERATE flow — called only
from the confirm node's approved branch, per the HITL invariant. It writes one
`program(status='draft')`, one `block`, and the `programmed_slot` rows in a
single transaction (all-or-nothing).

Draft exclusion (ARCHITECTURE.md §8.2) already keeps these rows out of every
analysis tool; `get_programs('draft')` / `get_block_outline` surface them.

Slot exercises resolve best-effort against the exercise dictionary (same rule
as `commit_batch`'s programmed-slot path): an unresolvable planned exercise
inserts with `exercise_id=NULL` rather than failing the save, and the raw name
is reported back so the user knows which ones didn't match.
"""
from __future__ import annotations

import sqlite3
from typing import TYPE_CHECKING

from pydantic import BaseModel

from src.tools.resolve import resolve_exercise

if TYPE_CHECKING:  # avoid a runtime import cycle (generate.py imports this module)
    from src.agent.nodes.generate import DraftProgram


class DraftSaveResult(BaseModel):
    program_id: int
    block_id: int
    slots_created: int = 0
    unresolved_exercises: list[str] = []


def persist_draft(conn: sqlite3.Connection, draft: "DraftProgram") -> DraftSaveResult:
    """Insert the draft program/block/slots transactionally; commit on success."""
    try:
        program_id = conn.execute(
            "INSERT INTO program (name, status, goals_text, notes) VALUES (?, 'draft', ?, ?)",
            (draft.program_name, draft.goals_text, draft.notes),
        ).lastrowid

        block_id = conn.execute(
            "INSERT INTO block (program_id, name, focus, week_count) VALUES (?, ?, ?, ?)",
            (program_id, draft.block_name, draft.focus, draft.week_count),
        ).lastrowid

        slots_created = 0
        unresolved: list[str] = []
        for slot in draft.slots:
            resolved = resolve_exercise(conn, slot.exercise)
            if resolved is None and slot.exercise not in unresolved:
                unresolved.append(slot.exercise)
            conn.execute(
                """
                INSERT INTO programmed_slot (block_id, week_number, day_number, day_label,
                                             exercise_id, prescription, target_weight_lb, notes)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    block_id,
                    slot.week_number,
                    slot.day_number,
                    slot.day_label,
                    resolved.exercise_id if resolved is not None else None,
                    slot.prescription,
                    slot.target_weight_lb,
                    slot.notes,
                ),
            )
            slots_created += 1

        conn.commit()
    except Exception:
        conn.rollback()
        raise

    return DraftSaveResult(
        program_id=program_id,
        block_id=block_id,
        slots_created=slots_created,
        unresolved_exercises=unresolved,
    )
