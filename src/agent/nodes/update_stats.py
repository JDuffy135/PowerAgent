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
from datetime import date as date_cls
from typing import Callable, Literal

from langchain_core.messages import AIMessage
from langgraph.types import interrupt
from pydantic import BaseModel, ValidationError

from src.agent.state import AgentState
from src.agent.units import format_weight
from src.tools.resolve import resolve_exercise
from src.tools.stats import insert_bodyweight, insert_pr

UPDATE_STATS_SYSTEM_PROMPT = """You extract ONE reported training stat from the user's message \
for a powerlifting log. Only two kinds are supported: a bodyweight reading, or a personal record \
(PR) on a lift. Output ONLY a JSON object matching the schema -- no prose.

Rules:
- kind="bodyweight" when they report a scale weight; put the value in weight_lb.
- kind="pr" when they report hitting a new best on a lift; put the exercise name in `exercise` \
exactly as written, the load in weight_lb, and the reps (use 1 for a 1-rep max).
- Normalize every weight to POUNDS. Convert kg with 1 kg = 2.20462 lb, rounded to 1 decimal.
- date is ISO YYYY-MM-DD if the user gives one (e.g. "yesterday" -> resolve it); otherwise null \
(the system fills in today).
- kind="none" if the message reports neither a bodyweight nor a PR (e.g. an injury, a measurement, \
or just a question)."""


class StatUpdate(BaseModel):
    kind: Literal["bodyweight", "pr", "none"]
    weight_lb: float | None = None
    exercise: str | None = None
    reps: int | None = None
    date: str | None = None
    context: str | None = None
    note: str | None = None


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

        return _decline(
            "I can only record bodyweight and PRs right now. For anything else "
            "(injuries, measurements), ingest a log or hang tight for a later update."
        )

    return update_stats_parse


def _decline(message: str) -> dict:
    return {"pending_stat": None, "review_decision": "none",
            "messages": [AIMessage(content=message)]}


# ---------------------------------------------------------------------------
# update_stats_confirm: interrupt with the parsed stat, then write on "yes"
# ---------------------------------------------------------------------------

def _render_pending(pending: dict, unit: str) -> str:
    if pending["kind"] == "bodyweight":
        summary = f"bodyweight {format_weight(pending['weight_lb'], unit)} on {pending['date']}"
    else:
        ctx = f" ({pending['context']})" if pending.get("context") else ""
        summary = (
            f"PR: {pending['exercise_name']} "
            f"{format_weight(pending['weight_lb'], unit)} x {pending['reps']} "
            f"on {pending['date']}{ctx}"
        )
    return f"Record this? {summary}. Reply `yes` to save or `no` to discard."


def make_update_stats_confirm_node(conn: sqlite3.Connection):
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

        if pending["kind"] == "bodyweight":
            insert_bodyweight(conn, pending["date"], pending["weight_lb"], pending.get("note"))
            confirm = f"Recorded bodyweight {format_weight(pending['weight_lb'], unit)} on {pending['date']}."
        else:
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

        return {
            "review_decision": "done",
            "pending_stat": None,
            "review_note": None,
            "messages": [AIMessage(content=confirm)],
        }

    return update_stats_confirm
