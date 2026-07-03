"""GENERATE nodes: evidence-gathering + structured program draft + HITL confirm
(ARCHITECTURE.md §4.2 — the program writer, heaviest reasoning in the system).

Two nodes, split for the interrupt-replay contract (same shape as INGEST /
UPDATE_STATS — no LLM work may live in the node that interrupts):

- `generate`: (1) a bounded ReAct loop over the same tools ANALYZE uses (recent
  e1RMs, volume trends, active injuries, block reviews via note search,
  programmed-vs-actual) to gather grounding evidence, then (2) a structured-
  output draft call producing a `DraftProgram` (Pydantic, mirroring the
  `ParsedProgrammedSlot` shape so slots are machine-insertable — **[DECISION]**).
  The rendered draft is emitted as an AIMessage and stashed in
  `state["pending_draft"]`.
- `generate_confirm`: `interrupt()` at the very top ("save this draft?");
  on yes, `persist_draft` writes program(status='draft') + block +
  programmed_slot rows — the only durable write in the flow. Draft exclusion
  keeps the saved rows out of analysis automatically.

**[DECISION]** Generation guardrails: the user's training philosophy is encoded
in `TRAINING_PHILOSOPHY` and injected into both prompts, alongside hard rules
(respect active injuries, 4-week RPE waves, etc.).

Model routing: the ReAct loop uses `get_chat_model("generate")` and the draft
call uses `get_llm("generate", ...)` (the raw prompt->JSON seam) — both
cloud-flippable per config (**[DECISION]** default: cloud, claude-sonnet-5).
"""
from __future__ import annotations

import json
import sqlite3
from datetime import date as date_cls
from typing import Callable

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.types import interrupt
from pydantic import BaseModel, Field, ValidationError

from src.agent.state import AgentState
from src.agent.tools import make_analyze_tools
from src.agent.units import format_weight
from src.tools.draft import persist_draft

MAX_TOOL_CALLS = 8       # evidence-gathering cap (mirrors ANALYZE)
MAX_EVIDENCE_CHARS = 2000  # per-item serialization cap in the draft prompt


# ---------------------------------------------------------------------------
# Draft models — ParsedProgrammedSlot-style, machine-insertable [DECISION]
# ---------------------------------------------------------------------------

class DraftSlot(BaseModel):
    """One planned prescription; mirrors a `programmed_slot` row."""

    exercise: str                      # raw name; resolved best-effort at save time
    week_number: int
    day_number: int
    day_label: str | None = None       # e.g. 'w1d1'
    prescription: str                  # e.g. '1x1 @ RPE 6, 4x4 @ RPE 7'
    target_weight_lb: float | None = None
    notes: str | None = None


class DraftProgram(BaseModel):
    """A drafted training block, ready to persist as program+block+slots."""

    program_name: str
    block_name: str
    focus: str | None = None           # 'hypertrophy' | 'strength' | 'peaking' | ...
    week_count: int | None = None
    goals_text: str | None = None
    notes: str | None = None
    slots: list[DraftSlot] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Prompts — [DECISION] the user's training philosophy, encoded as guardrails
# ---------------------------------------------------------------------------

TRAINING_PHILOSOPHY = """\
Training philosophy (hard guardrails — every draft must respect these):
1. Easy-to-moderate squat/bench/deadlift (SBD) work; accessory volume pushed to \
failure — as much volume as the lifter can recover from between sessions.
2. SBD programming runs in 4-week waves with ramping top-set RPE: week 1 easy, \
week 4 near-maximal (e.g. a deadlift top single might go RPE 5 -> 7 -> 8 -> 9 \
across the block).
3. Program targeted SBD variation lifts for weak points at least once a week in \
every block EXCEPT peaking blocks, where most (if not all) SBD work should be \
competition-specific.
4. 4-5 training days per week unless the lifter specifies otherwise.
5. Work around active injuries: substitute a tolerable movement pattern for the \
aggravating lift for the whole block (e.g. shoulder injury blocking bench -> \
neutral-grip machine pressing), reducing volume/intensity if needed.
6. If accessory volume needs are unknown for a muscle group, start around 10 \
hard sets per week for that muscle group, counting SBD sets only when they have \
7+ reps (e.g. 15 weekly bench sets of which 3 are 8-rep sets -> those 3 count, \
so prescribe ~7 chest accessory sets on top).
7. If SBD volume needs are unknown, start with ~7-9 weekly deadlift sets, 8-10 \
squat sets, and 10-15 bench sets (variations included).
8. Default frequency: squat and deadlift 2x/week (one primary + one secondary \
day); bench 3x/week (heavy, light, moderate days)."""

GENERATE_GATHER_PROMPT = """You are the program-writing engine of a powerlifting-coach \
assistant. Before drafting a training block you gather evidence about the lifter by \
calling the provided tools — never invent history.

Rules:
- Today's date is {today}. All dates are ISO YYYY-MM-DD. Weights in the database are pounds.
- Gather what a coach would want before programming: recent e1RM trends and best sets on \
the competition lifts, recent weekly volume by lift or muscle group, ACTIVE INJURIES \
(always call get_injuries with active_only=true), what past blocks looked like \
(get_programs / get_block_outline / compare_programmed_vs_actual), and relevant notes \
(search_training_notes for weak points, pain, block reviews).
- Call tools one or a few at a time. When you have enough evidence, STOP calling tools and \
reply with a brief note that you're done — a separate step writes the draft.

{philosophy}
"""

GENERATE_DRAFT_PROMPT = """You are the program-writing engine of a powerlifting-coach \
assistant. Using ONLY the user's request and the evidence provided, draft ONE training \
block as a JSON object matching the schema. Output ONLY the JSON object — no prose.

Rules:
- Ground every load prescription in the evidence (recent e1RMs / best sets). Prescribe by \
RPE where possible; fill target_weight_lb only when the evidence supports a concrete number \
(always in POUNDS).
- Emit one slot per exercise per training day, with week_number and day_number filled for \
every slot, in order. A 4-week block with 4 days and ~6 exercises per day means ~96 slots — \
emit them all; do not abbreviate with "repeat week 1".
- prescription strings look like '1x1 @ RPE 7, 4x4 @ RPE 7.5' or '3x10-12 to failure'.
- Use exercise names the lifter's log already uses when the evidence shows them.
- If the evidence shows an ACTIVE injury, the draft MUST program around it (rule 5).
- week_count must equal the number of distinct week_numbers in slots.

{philosophy}
"""


def _format_evidence(evidence: list[dict]) -> str:
    lines = ["Evidence gathered (weights in lb):"]
    for i, item in enumerate(evidence, 1):
        blob = json.dumps(item.get("result"), default=str)
        if len(blob) > MAX_EVIDENCE_CHARS:
            blob = blob[:MAX_EVIDENCE_CHARS] + "…(truncated)"
        args = json.dumps(item.get("args", {}), default=str)
        lines.append(f"{i}. {item.get('tool')}({args}) => {blob}")
    if len(lines) == 1:
        lines.append("(none — draft conservatively from the philosophy's starting defaults)")
    return "\n".join(lines)


def render_draft(draft: DraftProgram, unit: str = "lb") -> str:
    """Readable week -> day -> slot rendering for the HITL review."""
    header = [f"# Draft: {draft.program_name} :: {draft.block_name}"]
    meta = []
    if draft.focus:
        meta.append(f"focus: {draft.focus}")
    if draft.week_count:
        meta.append(f"{draft.week_count} week(s)")
    if meta:
        header.append("(" + ", ".join(meta) + ")")
    lines = [" ".join(header)]
    if draft.goals_text:
        lines.append(f"Goals: {draft.goals_text}")
    if draft.notes:
        lines.append(f"Notes: {draft.notes}")

    current: tuple[int | None, int | None] | None = None
    for slot in draft.slots:
        key = (slot.week_number, slot.day_number)
        if key != current:
            label = f" ({slot.day_label})" if slot.day_label else ""
            lines.append(f"\n## Week {slot.week_number}, Day {slot.day_number}{label}")
            current = key
        target = (
            f" @ ~{format_weight(slot.target_weight_lb, unit)}"
            if slot.target_weight_lb is not None
            else ""
        )
        note = f"  [{slot.notes}]" if slot.notes else ""
        lines.append(f"- {slot.exercise}: {slot.prescription}{target}{note}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# generate: ReAct evidence loop -> structured draft -> stash pending_draft
# ---------------------------------------------------------------------------

def make_generate_node(
    conn: sqlite3.Connection,
    model_factory: Callable[[], object],
    draft_llm_factory: Callable[[], object | None] = lambda: None,
    *,
    embedder=None,
    chroma_client=None,
    today: str | None = None,
):
    tools = make_analyze_tools(conn, embedder=embedder, chroma_client=chroma_client)
    tools_by_name = {t.name: t for t in tools}

    def generate(state: AgentState) -> dict:
        today_str = today or date_cls.today().isoformat()
        llm = model_factory().bind_tools(tools)

        gather_prompt = GENERATE_GATHER_PROMPT.format(
            today=today_str, philosophy=TRAINING_PHILOSOPHY
        )
        scratch = [SystemMessage(content=gather_prompt), *state["messages"]]

        evidence: list[dict] = []
        for _ in range(MAX_TOOL_CALLS):
            ai = llm.invoke(scratch)
            scratch.append(ai)
            tool_calls = getattr(ai, "tool_calls", None) or []
            if not tool_calls:
                break
            for call in tool_calls:
                name, args, call_id = call["name"], call.get("args", {}), call.get("id")
                tool = tools_by_name.get(name)
                if tool is None:
                    result = {"error": f"unknown tool {name!r}"}
                else:
                    try:
                        result = tool.invoke(args)
                    except Exception as exc:
                        result = {"error": f"{type(exc).__name__}: {exc}"}
                evidence.append({"tool": name, "args": args, "result": result})
                scratch.append(
                    ToolMessage(content=json.dumps(result, default=str), tool_call_id=call_id)
                )

        # Draft pass: raw structured-output seam, runs once (no interrupt here).
        draft_llm = draft_llm_factory()
        if draft_llm is None:
            from src.agent.llm_provider import get_llm

            draft_llm = get_llm(
                "generate",
                system_prompt=GENERATE_DRAFT_PROMPT.format(philosophy=TRAINING_PHILOSOPHY),
                schema=DraftProgram.model_json_schema(),
            )

        request = state["messages"][-1].content if state.get("messages") else ""
        prompt = f"User request: {request}\n\n{_format_evidence(evidence)}"
        try:
            draft = DraftProgram.model_validate(json.loads(draft_llm(prompt)))
        except (json.JSONDecodeError, ValidationError) as exc:
            return {
                "pending_draft": None,
                "evidence": evidence,
                "review_decision": "none",
                "messages": [AIMessage(content=(
                    "I couldn't produce a valid program draft "
                    f"({type(exc).__name__}). Try again, or narrow the request "
                    "(e.g. one 4-week block for a specific focus)."
                ))],
            }

        if not draft.slots:
            return {
                "pending_draft": None,
                "evidence": evidence,
                "review_decision": "none",
                "messages": [AIMessage(content=(
                    "The draft came back with no programmed slots, so there's nothing to "
                    "save. Try rephrasing the request."
                ))],
            }

        unit = state.get("display_unit", "lb")
        return {
            "pending_draft": draft.model_dump(),
            "evidence": evidence,
            "review_decision": None,
            "review_note": None,
            "messages": [AIMessage(content=render_draft(draft, unit))],
        }

    return generate


# ---------------------------------------------------------------------------
# generate_confirm: interrupt, then persist_draft on "yes"
# ---------------------------------------------------------------------------

CONFIRM_PROMPT = (
    "Save this as a draft program? It stays out of all analysis until started. "
    "Reply `yes` to save or `no` to discard."
)

_YES_WORDS = {"yes", "y", "save", "approve", "ok", "commit", "yep"}
_NO_WORDS = {"no", "n", "discard", "reject", "cancel", "nope"}


def make_generate_confirm_node(conn: sqlite3.Connection):
    def generate_confirm(state: AgentState) -> dict:
        note = state.get("review_note")
        prompt = (f"[note] {note}\n" if note else "") + CONFIRM_PROMPT
        reply = interrupt({"kind": "draft_confirm", "prompt": prompt})

        lowered = str(reply).strip().lower()
        if lowered in _NO_WORDS:
            return {
                "review_decision": "done",
                "pending_draft": None,
                "review_note": None,
                "messages": [AIMessage(content="Okay — draft discarded, nothing was saved.")],
            }
        if lowered not in _YES_WORDS:
            return {"review_decision": "reask",
                    "review_note": f"Didn't understand {str(reply)!r}. Reply `yes` or `no`."}

        draft = DraftProgram.model_validate(state["pending_draft"])
        result = persist_draft(conn, draft)

        summary = (
            f"Saved draft program {draft.program_name!r} (program id {result.program_id}, "
            f"block id {result.block_id}, {result.slots_created} programmed slot(s), "
            "status 'draft' — excluded from analysis until started)."
        )
        if result.unresolved_exercises:
            names = ", ".join(repr(n) for n in result.unresolved_exercises)
            summary += (
                f" Heads up: {names} didn't match the exercise dictionary; those slots "
                "saved without an exercise link."
            )

        return {
            "review_decision": "done",
            "pending_draft": None,
            "review_note": None,
            "messages": [AIMessage(content=summary)],
        }

    return generate_confirm
