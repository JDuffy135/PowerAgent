"""INGEST pipeline nodes: parse -> HITL review loop -> block assignment -> commit.

Wires the Step 2/3 pipeline into LangGraph (ARCHITECTURE.md §4.2, §4.4):

    parse_upload -> extract_training_data -> stage_batch      (ingest_parse)
    interrupt(render_batch) -> approve/reject/corrections     (ingest_review, loops)
    interrupt(block question) -> commit_batch                 (ingest_commit, loops on bad reply)

Split into three nodes because `interrupt()` replays its node from the top on
resume: parsing/staging must live *before* the first interrupt-bearing node so
the LLM extraction runs exactly once, and each review round is a fresh node
execution (loop-back edge), not a loop around `interrupt()`.

HITL contract decisions locked in this stage:
- Corrections are a **full re-emit** by the correction LLM (`src.ingest.correct`),
  persisted to the staged batch only; the loop is capped at
  `MAX_CORRECTION_ROUNDS`, after which only approve/reject are accepted.
- On approval the user is asked which program/block the batch belongs to
  (**[DECISION]** option a) -- pick an existing block, create a program+block on
  the fly, or `none` to leave it unattached and organize later.

Nothing durable happens outside `commit_batch`/`reject_batch` -- the Step 3
invariant is reused, not reimplemented.
"""
from __future__ import annotations

import sqlite3
from typing import Callable

from langchain_core.messages import AIMessage
from langgraph.types import interrupt

from src.agent.state import AgentState
from src.ingest.correct import apply_correction
from src.ingest.extract import LLMCallable, extract_training_data
from src.ingest.loaders import UnsupportedFileType, parse_upload
from src.ingest.commit import commit_batch, reject_batch
from src.ingest.review import render_batch
from src.ingest.stage import get_pending_batch, stage_batch, update_batch

MAX_CORRECTION_ROUNDS = 5  # [DECISION] cap the correct->re-render loop

REVIEW_INSTRUCTIONS = (
    "Reply `approve` to commit, `reject` to discard, "
    "or describe corrections in plain text."
)
REVIEW_INSTRUCTIONS_AT_CAP = (
    f"Correction limit reached ({MAX_CORRECTION_ROUNDS} rounds). "
    "Reply `approve` to commit as shown, or `reject` to discard."
)

_APPROVE_WORDS = {"approve", "approved", "yes", "y", "commit", "ok", "lgtm"}
_REJECT_WORDS = {"reject", "rejected", "no", "n", "discard", "cancel"}


# ---------------------------------------------------------------------------
# ingest_parse: file -> ParsedBatch -> staged pending_review row
# ---------------------------------------------------------------------------

def make_ingest_parse_node(
    conn: sqlite3.Connection,
    extract_llm_factory: Callable[[], LLMCallable | None] = lambda: None,
):
    def ingest_parse(state: AgentState) -> dict:
        path = state.get("file_path")
        if not path:
            return {
                "review_decision": "error",
                "messages": [AIMessage(content=(
                    "I need a file to ingest. Use `/ingest <path>` "
                    "(e.g. `/ingest logs/week3.txt`)."
                ))],
            }

        try:
            text = parse_upload(path)
        except (FileNotFoundError, UnsupportedFileType, NotImplementedError) as exc:
            return {
                "review_decision": "error",
                "messages": [AIMessage(content=f"Could not load {path}: {exc}")],
            }

        try:
            batch = extract_training_data(text, conn=conn, llm=extract_llm_factory())
        except ValueError as exc:
            return {
                "review_decision": "error",
                "messages": [AIMessage(content=f"Extraction failed for {path}: {exc}")],
            }

        batch_id = stage_batch(conn, batch, source_file=str(path))
        return {
            "pending_batch_id": batch_id,
            "correction_rounds": 0,
            "review_decision": None,
            "review_note": None,
        }

    return ingest_parse


# ---------------------------------------------------------------------------
# ingest_review: interrupt with the rendered batch; approve/reject/correct
# ---------------------------------------------------------------------------

def _parse_review_reply(reply: object) -> str:
    text = str(reply).strip()
    lowered = text.lower()
    if lowered in _APPROVE_WORDS:
        return "approve"
    if lowered in _REJECT_WORDS:
        return "reject"
    return "correct"


def make_ingest_review_node(
    conn: sqlite3.Connection,
    correction_llm_factory: Callable[[], LLMCallable | None] = lambda: None,
):
    def ingest_review(state: AgentState) -> dict:
        batch_id = state["pending_batch_id"]
        batch = get_pending_batch(conn, batch_id)
        rounds = state.get("correction_rounds", 0)
        at_cap = rounds >= MAX_CORRECTION_ROUNDS

        prompt_parts = []
        if state.get("review_note"):
            prompt_parts.append(f"[note] {state['review_note']}\n")
        prompt_parts.append(render_batch(batch, state.get("display_unit", "lb")))
        prompt_parts.append(REVIEW_INSTRUCTIONS_AT_CAP if at_cap else REVIEW_INSTRUCTIONS)

        reply = interrupt({
            "kind": "ingest_review",
            "batch_id": batch_id,
            "prompt": "\n".join(prompt_parts),
        })

        decision = _parse_review_reply(reply)
        if decision == "approve":
            return {"review_decision": "approve", "review_note": None}
        if decision == "reject":
            reject_batch(conn, batch_id)
            return {
                "review_decision": "reject",
                "pending_batch_id": None,
                "review_note": None,
                "messages": [AIMessage(content=(
                    f"Batch {batch_id} rejected -- nothing was written to the "
                    "training data. The parse is kept in the audit trail."
                ))],
            }

        # Free-text correction.
        if at_cap:
            return {
                "review_decision": "correct",
                "review_note": "Corrections are no longer accepted for this batch.",
            }

        try:
            corrected = apply_correction(
                batch, str(reply), llm=correction_llm_factory(), conn=conn
            )
            update_batch(conn, batch_id, corrected)
            note = None
        except ValueError as exc:
            note = f"Correction could not be applied ({exc}); showing the unchanged parse."
        return {
            "review_decision": "correct",
            "correction_rounds": rounds + 1,
            "review_note": note,
        }

    return ingest_review


# ---------------------------------------------------------------------------
# ingest_commit: interrupt with the block-assignment question, then commit
# ---------------------------------------------------------------------------

BLOCK_INSTRUCTIONS = (
    "Which program/block does this batch belong to? (You can reorganize later.)\n"
    "  - reply `none` to leave it unattached\n"
    "  - reply a block id from the list below\n"
    "  - reply `new <program name> / <block name>` to create both"
)


def _list_blocks(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    return conn.execute(
        """
        SELECT b.block_id, b.name AS block_name, b.focus,
               p.program_id, p.name AS program_name, p.status
        FROM block b JOIN program p ON p.program_id = b.program_id
        ORDER BY p.program_id, b.block_id
        """
    ).fetchall()


def _block_prompt(conn: sqlite3.Connection, note: str | None) -> str:
    lines = []
    if note:
        lines.append(f"[note] {note}\n")
    lines.append(BLOCK_INSTRUCTIONS)
    blocks = _list_blocks(conn)
    if blocks:
        lines.append("Existing blocks:")
        for row in blocks:
            focus = f", {row['focus']}" if row["focus"] else ""
            lines.append(
                f"  [{row['block_id']}] {row['program_name']} :: "
                f"{row['block_name']} ({row['status']}{focus})"
            )
    else:
        lines.append("(no programs/blocks exist yet)")
    return "\n".join(lines)


def _batch_start_date(conn: sqlite3.Connection, batch_id: int) -> str | None:
    batch = get_pending_batch(conn, batch_id)
    dates = sorted(s.date for s in batch.sessions if s.date)
    return dates[0] if dates else None


def _create_program_and_block(
    conn: sqlite3.Connection, program_name: str, block_name: str, start_date: str | None
) -> int:
    """Create (or reuse, matched case-insensitively) the program, create the block.

    New programs get status 'incomplete': logs are being committed against them,
    so by the three-state lifecycle they are started-but-not-finished.
    """
    row = conn.execute(
        "SELECT program_id FROM program WHERE lower(name) = lower(?)", (program_name,)
    ).fetchone()
    if row is not None:
        program_id = row["program_id"]
    else:
        program_id = conn.execute(
            "INSERT INTO program (name, status, start_date) VALUES (?, 'incomplete', ?)",
            (program_name, start_date),
        ).lastrowid

    block_id = conn.execute(
        "INSERT INTO block (program_id, name, start_date) VALUES (?, ?, ?)",
        (program_id, block_name, start_date),
    ).lastrowid
    conn.commit()
    return block_id


def _parse_block_reply(
    conn: sqlite3.Connection, reply: object
) -> tuple[str, object]:
    """Return one of: ('none', None) | ('existing', block_id) |
    ('new', (program_name, block_name)) | ('invalid', message)."""
    text = str(reply).strip()
    lowered = text.lower()

    if lowered in {"", "none", "skip", "later", "unattached"}:
        return ("none", None)

    if text.isdigit():
        block_id = int(text)
        row = conn.execute(
            "SELECT 1 FROM block WHERE block_id = ?", (block_id,)
        ).fetchone()
        if row is None:
            return ("invalid", f"No block with id {block_id}.")
        return ("existing", block_id)

    if lowered.startswith("new "):
        rest = text[4:]
        if "/" in rest:
            program_name, _, block_name = rest.partition("/")
            program_name, block_name = program_name.strip(), block_name.strip()
            if program_name and block_name:
                return ("new", (program_name, block_name))
        return ("invalid", "Use the form: new <program name> / <block name>")

    return ("invalid", f"Didn't understand {text!r}.")


def make_ingest_commit_node(
    conn: sqlite3.Connection,
    *,
    embedder=None,
    chroma_client=None,
    embed_prose: bool = True,
):
    def ingest_commit(state: AgentState) -> dict:
        batch_id = state["pending_batch_id"]

        reply = interrupt({
            "kind": "block_assign",
            "batch_id": batch_id,
            "prompt": _block_prompt(conn, state.get("review_note")),
        })

        kind, value = _parse_block_reply(conn, reply)
        if kind == "invalid":
            return {"review_decision": "ask_block", "review_note": value}

        if kind == "new":
            program_name, block_name = value
            block_id = _create_program_and_block(
                conn, program_name, block_name, _batch_start_date(conn, batch_id)
            )
            attached = f"new block {block_name!r} (id {block_id}) in program {program_name!r}"
        elif kind == "existing":
            block_id = value
            attached = f"block id {block_id}"
        else:
            block_id = None
            attached = "no block (unattached; organize later)"

        result = commit_batch(
            conn,
            batch_id,
            block_id=block_id,
            embedder=embedder,
            chroma_client=chroma_client,
            embed_prose=embed_prose,
        )

        summary = (
            f"Committed batch {batch_id}: {result.sessions_created} session(s), "
            f"{result.sets_created} set(s), {result.cardio_created} cardio row(s), "
            f"{result.programmed_slots_created} programmed slot(s), "
            f"{result.exercises_created} new exercise(s), "
            f"{result.notes_embedded} note(s) embedded. Attached to {attached}."
        )
        if result.programmed_slots_skipped:
            summary += (
                f" ({result.programmed_slots_skipped} programmed slot(s) not inserted "
                "-- they need a block; they're preserved in the audit trail.)"
            )

        return {
            "review_decision": "done",
            "pending_batch_id": None,
            "review_note": None,
            "messages": [AIMessage(content=summary)],
        }

    return ingest_commit
