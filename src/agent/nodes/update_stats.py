"""UPDATE_STATS nodes: parse one reported stat, then confirm-before-write (ARCHITECTURE.md §4.2).

**[DECISION] Stage 6 scope is bodyweight + PRs only.** The parse LLM detects
which of the two the user reported ("bodyweight was 146 this morning" ->
bodyweight; "hit a 405 deadlift PR" -> pr) and normalizes the weight to lb
(converting kg, like the ingest extractor). Injury/measurement phrasing is more
varied and is deferred.

Split into two nodes for the same reason INGEST is split: `interrupt()` replays
its node from the top on resume, so the LLM parse (which must run exactly once)
lives in `update_stats_parse`, and the confirm interrupt sits at the top of
`update_stats_confirm`. Nothing is written until the user confirms -- the durable
inserts (`src.tools.stats`) run only on the "yes" branch.
"""
from __future__ import annotations

import json
import sqlite3
import time
from datetime import date as date_cls
from typing import Callable, Literal

from langchain_core.messages import AIMessage
from langgraph.types import interrupt
from pydantic import BaseModel, ValidationError

from src.agent.state import AgentState
from src.agent.units import format_weight
from src.ingest.embed import FORM_CUE_DOC_TYPE, embed_review
from src.tools.organize import set_block_review
from src.tools.resolve import resolve_exercise
from src.tools.stats import insert_bodyweight, insert_pr

UPDATE_STATS_SYSTEM_PROMPT = """You extract ONE reported training item from the user's message \
for a powerlifting log. Output ONLY a JSON object matching the schema -- no prose.

Supported kinds:
- kind="bodyweight": a scale weight; put the value in weight_lb.
- kind="pr": a new best on a lift; put the exercise name in `exercise` exactly as written, the \
load in weight_lb, and the reps (use 1 for a 1-rep max).
- kind="block_review": the user is recording a review/reflection on a training block or mesocycle \
("here's my review of the last block: ..."); put the review prose in `text`.
- kind="form_cue": the user is recording a technique cue for a lift ("form cue for squat: spread \
the floor", "cue for bench: tuck elbows"); put the lift name in `exercise` and the cue in `text`.
- kind="none": none of the above (an injury, a measurement, or just a question).

Rules:
- Normalize every weight to POUNDS. Convert kg with 1 kg = 2.20462 lb, rounded to 1 decimal.
- date is ISO YYYY-MM-DD if the user gives one (e.g. "yesterday" -> resolve it); otherwise null \
(the system fills in today)."""


class StatUpdate(BaseModel):
    kind: Literal["bodyweight", "pr", "block_review", "form_cue", "none"]
    weight_lb: float | None = None
    exercise: str | None = None
    reps: int | None = None
    date: str | None = None
    context: str | None = None
    note: str | None = None
    text: str | None = None       # review / form-cue prose (block_review, form_cue)


_YES_WORDS = {"yes", "y", "confirm", "ok", "commit", "save", "correct", "yep"}
_NO_WORDS = {"no", "n", "cancel", "discard", "nope", "stop"}


# ---------------------------------------------------------------------------
# update_stats_parse: message -> StatUpdate -> staged pending_stat
# ---------------------------------------------------------------------------

def make_update_stats_parse_node(
    conn: sqlite3.Connection,
    llm_factory: Callable[[], object | None] = lambda: None,
):
    def update_stats_parse(state: AgentState) -> dict:
        text = state["messages"][-1].content if state.get("messages") else ""
        text = text if isinstance(text, str) else str(text)

        llm = llm_factory()
        if llm is None:
            from src.ingest.extract import get_llm

            llm = get_llm(
                "update_stats",
                system_prompt=UPDATE_STATS_SYSTEM_PROMPT,
                schema=StatUpdate.model_json_schema(),
            )

        try:
            parsed = StatUpdate.model_validate(json.loads(llm(text)))
        except (json.JSONDecodeError, ValidationError):
            return _decline("I couldn't read that as a stat update. Try e.g. "
                            "\"bodyweight 146 today\" or \"hit a 405x1 deadlift PR\".")

        today = date_cls.today().isoformat()

        if parsed.kind == "bodyweight":
            if parsed.weight_lb is None:
                return _decline("I saw a bodyweight update but couldn't read the weight.")
            pending = {
                "kind": "bodyweight",
                "weight_lb": parsed.weight_lb,
                "date": parsed.date or today,
                "note": parsed.note,
            }
            return {"pending_stat": pending, "review_decision": None, "review_note": None}

        if parsed.kind == "pr":
            if not parsed.exercise or parsed.weight_lb is None or parsed.reps is None:
                return _decline("I saw a PR but need the exercise, weight, and reps to record it.")
            resolved = resolve_exercise(conn, parsed.exercise)
            if resolved is None:
                return _decline(
                    f"I don't have an exercise matching {parsed.exercise!r} yet. "
                    "Ingest a log that uses it (so it's in the exercise dictionary) first."
                )
            pending = {
                "kind": "pr",
                "exercise_id": resolved.exercise_id,
                "exercise_name": resolved.name,
                "weight_lb": parsed.weight_lb,
                "reps": parsed.reps,
                "date": parsed.date or today,
                "context": parsed.context,
            }
            return {"pending_stat": pending, "review_decision": None, "review_note": None}

        if parsed.kind == "form_cue":
            if not parsed.exercise or not (parsed.text and parsed.text.strip()):
                return _decline("I saw a form cue but need both the lift and the cue text.")
            resolved = resolve_exercise(conn, parsed.exercise)
            if resolved is None:
                return _decline(
                    f"I don't have an exercise matching {parsed.exercise!r} yet. "
                    "Ingest a log that uses it first, then add the cue."
                )
            pending = {
                "kind": "form_cue",
                "exercise_id": resolved.exercise_id,
                "exercise_name": resolved.name,
                "text": parsed.text.strip(),
                "date": parsed.date or today,
            }
            return {"pending_stat": pending, "review_decision": None, "review_note": None}

        if parsed.kind == "block_review":
            if not (parsed.text and parsed.text.strip()):
                return _decline("I saw a block review but couldn't read the review text.")
            block = _latest_block(conn)
            if block is None:
                return _decline(
                    "There's no block to attach a review to yet. Ingest some training "
                    "into a block first (or add the review from the Organizer tab)."
                )
            block_id, block_name = block
            pending = {
                "kind": "block_review",
                "block_id": block_id,
                "block_name": block_name,
                "text": parsed.text.strip(),
                "date": parsed.date or today,
            }
            return {"pending_stat": pending, "review_decision": None, "review_note": None}

        return _decline(
            "I can record bodyweight, PRs, block reviews, and form cues right now. "
            "For anything else (injuries, measurements), ingest a log or hang tight."
        )

    return update_stats_parse


def _latest_block(conn: sqlite3.Connection) -> tuple[int, str] | None:
    """The most recently dated block, as `(block_id, name)`. Block reviews from
    chat attach here (the Organizer tab is the path for reviewing an older block)."""
    row = conn.execute(
        """
        SELECT block_id, name FROM block
        ORDER BY start_date IS NULL, start_date DESC, block_id DESC
        LIMIT 1
        """
    ).fetchone()
    return (row["block_id"], row["name"]) if row else None


def _decline(message: str) -> dict:
    return {"pending_stat": None, "review_decision": "none",
            "messages": [AIMessage(content=message)]}


# ---------------------------------------------------------------------------
# update_stats_confirm: interrupt with the parsed stat, then write on "yes"
# ---------------------------------------------------------------------------

def _snippet(text: str, limit: int = 100) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[:limit].rstrip() + "…"


def _render_pending(pending: dict, unit: str) -> str:
    kind = pending["kind"]
    if kind == "bodyweight":
        summary = f"bodyweight {format_weight(pending['weight_lb'], unit)} on {pending['date']}"
    elif kind == "pr":
        ctx = f" ({pending['context']})" if pending.get("context") else ""
        summary = (
            f"PR: {pending['exercise_name']} "
            f"{format_weight(pending['weight_lb'], unit)} x {pending['reps']} "
            f"on {pending['date']}{ctx}"
        )
    elif kind == "form_cue":
        summary = f"form cue for {pending['exercise_name']}: \"{_snippet(pending['text'])}\""
    else:  # block_review
        summary = f"review of block {pending['block_name']!r}: \"{_snippet(pending['text'])}\""
    return f"Record this? {summary}. Reply `yes` to save or `no` to discard."


def make_update_stats_confirm_node(
    conn: sqlite3.Connection,
    *,
    embedder=None,
    chroma_client=None,
    embed_reviews: bool = True,
):
    def update_stats_confirm(state: AgentState) -> dict:
        pending = state["pending_stat"]
        unit = state.get("display_unit", "lb")

        note = state.get("review_note")
        prompt = (f"[note] {note}\n" if note else "") + _render_pending(pending, unit)
        reply = interrupt({"kind": "stat_confirm", "prompt": prompt})

        lowered = str(reply).strip().lower()
        if lowered in _NO_WORDS:
            return {
                "review_decision": "done",
                "pending_stat": None,
                "review_note": None,
                "messages": [AIMessage(content="Okay — nothing recorded.")],
            }
        if lowered not in _YES_WORDS:
            return {"review_decision": "reask",
                    "review_note": f"Didn't understand {str(reply)!r}. Reply `yes` or `no`."}

        kind = pending["kind"]
        if kind == "bodyweight":
            insert_bodyweight(conn, pending["date"], pending["weight_lb"], pending.get("note"))
            confirm = f"Recorded bodyweight {format_weight(pending['weight_lb'], unit)} on {pending['date']}."
        elif kind == "pr":
            insert_pr(
                conn,
                pending["date"],
                pending["exercise_id"],
                pending["weight_lb"],
                pending["reps"],
                context=pending.get("context"),
            )
            confirm = (
                f"Recorded PR: {pending['exercise_name']} "
                f"{format_weight(pending['weight_lb'], unit)} x {pending['reps']} on {pending['date']}."
            )
        elif kind == "form_cue":
            if embed_reviews:
                embed_review(
                    pending["text"],
                    f"form_cue_{pending['exercise_id']}_{int(time.time() * 1000)}",
                    FORM_CUE_DOC_TYPE,
                    date=pending["date"],
                    exercises=[pending["exercise_name"]],
                    embedder=embedder,
                    client=chroma_client,
                )
            confirm = f"Saved a form cue for {pending['exercise_name']}."
        else:  # block_review
            set_block_review(
                conn,
                pending["block_id"],
                pending["text"],
                embedder=embedder,
                chroma_client=chroma_client,
                embed=embed_reviews,
            )
            confirm = f"Saved your review of block {pending['block_name']!r}."

        return {
            "review_decision": "done",
            "pending_stat": None,
            "review_note": None,
            "messages": [AIMessage(content=confirm)],
        }

    return update_stats_confirm
