"""Transactional commit of an approved ingest batch (ARCHITECTURE.md §5.3).

`commit_batch` turns a `pending_review` `ingest_batch` row into durable training
data: it resolves/creates exercises, inserts `session` + `lift_set` + `cardio`
rows, flips the batch status to `committed`, and (post-commit) embeds session
prose into Chroma. Everything that touches SQLite happens in **one transaction**
-- a mid-commit failure (bad FK, unresolved exercise) rolls the whole thing back
and leaves `ingest_batch.status` untouched.

State machine (idempotent / no double-insert):
- `pending_review` -> perform the commit -> `committed`
- `committed`      -> no-op (returns `committed=False`, zero counts)
- `rejected`       -> `BatchNotCommittable`

`reject_batch` is the other terminal transition: `pending_review` -> `rejected`,
writing no training data. Rejecting an already-rejected batch is a no-op;
rejecting a committed one raises.

**Programmed slots this step:** `programmed_slot.block_id` is `NOT NULL`, but
block assignment isn't built until a later step, so parsed programmed slots are
*not* inserted here -- they're counted as `programmed_slots_skipped` and remain
preserved verbatim in the batch's `parsed_json` audit trail for later.
"""
from __future__ import annotations

import sqlite3

from pydantic import BaseModel

from src.ingest.embed import SessionNote, embed_session_notes
from src.ingest.models import ParsedBatch, ParsedSession
from src.ingest.stage import BatchNotFound, get_pending_batch
from src.tools.resolve import _normalize, add_exercise, resolve_exercise


class CommitResult(BaseModel):
    batch_id: int
    committed: bool  # True if this call performed the commit; False if already committed
    sessions_created: int = 0
    sets_created: int = 0
    cardio_created: int = 0
    programmed_slots_skipped: int = 0
    exercises_created: int = 0
    notes_embedded: int = 0


class BatchNotCommittable(Exception):
    """Raised when a batch can't make the requested transition (e.g. commit a
    rejected batch, or reject a committed one)."""

    def __init__(self, batch_id: int, status: str, action: str):
        self.batch_id = batch_id
        self.status = status
        super().__init__(f"Cannot {action} batch {batch_id}: status is {status!r}")


class UnresolvedExercise(Exception):
    """A set references an exercise that neither resolves nor has a
    new-exercise candidate. Triggers a full rollback of the commit."""

    def __init__(self, raw_name: str):
        self.raw_name = raw_name
        super().__init__(
            f"Cannot resolve exercise {raw_name!r} and no new-exercise candidate was provided"
        )


def _status(conn: sqlite3.Connection, batch_id: int) -> str:
    row = conn.execute(
        "SELECT status FROM ingest_batch WHERE batch_id = ?", (batch_id,)
    ).fetchone()
    if row is None:
        raise BatchNotFound(batch_id)
    return row["status"]


def commit_batch(
    conn: sqlite3.Connection,
    batch_id: int,
    *,
    embedder=None,
    chroma_client=None,
    embed_prose: bool = True,
) -> CommitResult:
    """Commit an approved batch's parsed data into SQLite (transactionally),
    then embed its session prose into Chroma.

    `embedder`/`chroma_client` are the embed seams from `embed.py`; tests inject
    fakes. Set `embed_prose=False` to skip Chroma entirely (SQLite commit only).
    """
    status = _status(conn, batch_id)
    if status == "committed":
        return CommitResult(batch_id=batch_id, committed=False)
    if status != "pending_review":
        raise BatchNotCommittable(batch_id, status, "commit")

    parsed = get_pending_batch(conn, batch_id)

    try:
        counts, notes = _insert_batch(conn, parsed)
        conn.execute(
            "UPDATE ingest_batch SET status = 'committed' WHERE batch_id = ?",
            (batch_id,),
        )
        conn.commit()
    except Exception:
        conn.rollback()
        raise

    # Chroma is not part of the SQLite transaction (different store, not
    # transactional). Embedding is best-effort *after* the durable SQLite commit.
    notes_embedded = 0
    if embed_prose and notes:
        notes_embedded = embed_session_notes(notes, embedder=embedder, client=chroma_client)

    return CommitResult(
        batch_id=batch_id,
        committed=True,
        notes_embedded=notes_embedded,
        **counts,
    )


def reject_batch(conn: sqlite3.Connection, batch_id: int) -> bool:
    """Mark a pending batch `rejected`, writing no training data.

    Returns True if this call changed the status, False if it was already
    rejected (idempotent). Raises `BatchNotCommittable` for a committed batch.
    """
    status = _status(conn, batch_id)
    if status == "rejected":
        return False
    if status != "pending_review":
        raise BatchNotCommittable(batch_id, status, "reject")

    conn.execute(
        "UPDATE ingest_batch SET status = 'rejected' WHERE batch_id = ?",
        (batch_id,),
    )
    conn.commit()
    return True


# --------------------------------------------------------------------------
# Internal: exercise resolution/creation + row inserts (all within one txn)
# --------------------------------------------------------------------------

def _resolve_and_create_exercises(
    conn: sqlite3.Connection, parsed: ParsedBatch
) -> tuple[dict[str, tuple[int, str]], int]:
    """Return (normalized-raw-name -> (exercise_id, canonical_name), created_count).

    Known names reuse their existing `exercise_id`; confirmed new-exercise
    candidates are created via `add_exercise(commit=False)` (auto-confirmed this
    step -- Step 4's HITL supplies user edits). Existing exercises are never
    duplicated. A set whose raw name neither resolves nor has a candidate raises
    `UnresolvedExercise`, which rolls the whole commit back.
    """
    candidate_by_key = {
        _normalize(c.raw_name): c for c in parsed.new_exercise_candidates
    }

    mapping: dict[str, tuple[int, str]] = {}
    created = 0

    for session in parsed.sessions:
        for parsed_set in session.sets:  # slots are skipped this step; don't create for them
            raw = parsed_set.exercise_raw
            key = _normalize(raw)
            if key in mapping:
                continue

            resolved = resolve_exercise(conn, raw)
            if resolved is not None:
                mapping[key] = (resolved.exercise_id, resolved.name)
                continue

            candidate = candidate_by_key.get(key)
            if candidate is None:
                raise UnresolvedExercise(raw)

            new_id = add_exercise(
                conn,
                candidate.suggested_name,
                candidate.suggested_tier,
                candidate.suggested_muscle_group,
                [candidate.raw_name],
                commit=False,
            )
            created += 1
            mapping[key] = (new_id, candidate.suggested_name)

    return mapping, created


def _insert_session(conn: sqlite3.Connection, session: ParsedSession) -> int:
    return conn.execute(
        """
        INSERT INTO session (date, block_id, week_number, day_number, day_label,
                             duration_min, session_type, raw_note)
        VALUES (?, NULL, ?, ?, ?, ?, ?, ?)
        """,
        (
            session.date,
            session.week_number,
            session.day_number,
            session.day_label,
            session.duration_min,
            session.session_type,
            session.raw_note,
        ),
    ).lastrowid


def _insert_batch(
    conn: sqlite3.Connection, parsed: ParsedBatch
) -> tuple[dict[str, int], list[SessionNote]]:
    """Insert every session and its child rows. Assumes the caller owns the
    transaction (no commit here)."""
    exercise_map, exercises_created = _resolve_and_create_exercises(conn, parsed)

    sessions_created = sets_created = cardio_created = slots_skipped = 0
    notes: list[SessionNote] = []

    for session in parsed.sessions:
        session_id = _insert_session(conn, session)
        sessions_created += 1

        canonical_names: list[str] = []
        for parsed_set in session.sets:
            exercise_id, canonical_name = exercise_map[_normalize(parsed_set.exercise_raw)]
            if canonical_name not in canonical_names:
                canonical_names.append(canonical_name)
            conn.execute(
                """
                INSERT INTO lift_set (session_id, exercise_id, set_index, weight_lb,
                                      reps, rpe, is_paused, is_amrap, is_top_set,
                                      is_failed, equipment_note, raw_text)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    exercise_id,
                    parsed_set.set_index,
                    parsed_set.weight_lb,
                    parsed_set.reps,
                    parsed_set.rpe,
                    int(parsed_set.is_paused),
                    int(parsed_set.is_amrap),
                    int(parsed_set.is_top_set),
                    int(parsed_set.is_failed),
                    parsed_set.equipment_note,
                    parsed_set.raw_text,
                ),
            )
            sets_created += 1

        for cardio in session.cardio:
            conn.execute(
                """
                INSERT INTO cardio (session_id, modality, distance_mi, duration_min,
                                    intensity, raw_text)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    cardio.modality,
                    cardio.distance_mi,
                    cardio.duration_min,
                    cardio.intensity,
                    cardio.raw_text,
                ),
            )
            cardio_created += 1

        slots_skipped += len(session.programmed_slots)

        notes.append(
            SessionNote(
                session_id=session_id,
                date=session.date,
                raw_note=session.raw_note,
                exercises=canonical_names,
            )
        )

    counts = {
        "sessions_created": sessions_created,
        "sets_created": sets_created,
        "cardio_created": cardio_created,
        "programmed_slots_skipped": slots_skipped,
        "exercises_created": exercises_created,
    }
    return counts, notes
