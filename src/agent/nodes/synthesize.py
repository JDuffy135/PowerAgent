"""SYNTHESIZE + store-offer nodes (ARCHITECTURE.md §4.2).

SYNTHESIZE composes the final user-facing answer from the `evidence` ANALYZE
gathered. It is the **single presentation point**: weights are converted from the
canonical lb to the user's `display_unit` here (`src.agent.units`), and never
before. Evidence carries the source sets/dates behind every e1RM/PR figure, so
the composed answer can stay auditable.

**[DECISION] Evidence overflow:** when ANALYZE hit its cap
(`evidence_truncated`), a fixed disclaimer is appended in code (not left to the
LLM) telling the user the answer is partial and to narrow the question scope.

**[DECISION] "Store this analysis?":** implemented now. SYNTHESIZE stashes the
answer and flags `offer_store`; the separate `store_offer` node then asks (via
`interrupt()`) whether to embed the analysis into Chroma `personal_notes` under
`doc_type='analysis'`. The interrupt sits at the very top of its own node (no LLM
call before it) so the resume-replay contract from Stage 5 holds.
"""
from __future__ import annotations

import json
from datetime import date as date_cls
from typing import Callable

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.types import interrupt

from src.agent.state import AgentState
from src.agent.units import convert_weights

MAX_EVIDENCE_CHARS = 2000  # per-item cap so a huge tool dump can't blow the prompt

OVERFLOW_DISCLAIMER = (
    "\n\n_Heads up: I reached my evidence-gathering limit for this question, so "
    "this answer is based on partial data. For a more precise answer, try narrowing "
    "the scope — a specific exercise, a shorter date range, or a single metric._"
)

NO_EVIDENCE_MESSAGE = (
    "I couldn't find any training data to answer that. If you haven't ingested the "
    "relevant logs yet, use `/ingest <path>` first."
)

SYNTHESIZE_SYSTEM_PROMPT = """You are the answer-writer for a powerlifting-coach assistant. \
Write a concise, direct answer to the user's question using ONLY the evidence provided. \
Do not invent numbers the evidence doesn't contain.

Rules:
- The evidence weights are already in the user's preferred unit ({unit}) -- report them as-is, \
do not convert.
- When you state an e1RM or a PR, cite the source (weight x reps on its date) so the claim is \
checkable; the evidence includes those source fields.
- If the evidence is thin or a lookup failed, say so plainly rather than guessing.
- Keep it tight: a few sentences or a short list, no preamble."""


def _format_evidence(evidence: list[dict], unit: str) -> str:
    lines = [f"Weights below are in {unit}."]
    for i, item in enumerate(evidence, 1):
        converted = convert_weights(item.get("result"), unit)
        blob = json.dumps(converted, default=str)
        if len(blob) > MAX_EVIDENCE_CHARS:
            blob = blob[:MAX_EVIDENCE_CHARS] + "…(truncated)"
        args = json.dumps(item.get("args", {}), default=str)
        lines.append(f"{i}. {item.get('tool')}({args}) => {blob}")
    return "\n".join(lines)


def make_synthesize_node(model_factory: Callable[[], object]):
    def synthesize(state: AgentState) -> dict:
        evidence = state.get("evidence") or []
        truncated = state.get("evidence_truncated", False)
        unit = state.get("display_unit", "lb")

        if not evidence:
            # Nothing to synthesize; don't burn a model call.
            message = NO_EVIDENCE_MESSAGE
            if truncated:
                message += OVERFLOW_DISCLAIMER
            return {
                "messages": [AIMessage(content=message)],
                "analysis_text": message,
                "offer_store": False,
            }

        question = state["messages"][-1].content if state.get("messages") else ""
        prompt = (
            f"User question: {question}\n\n"
            f"Evidence gathered:\n{_format_evidence(evidence, unit)}"
        )
        response = model_factory().invoke([
            SystemMessage(content=SYNTHESIZE_SYSTEM_PROMPT.format(unit=unit)),
            HumanMessage(content=prompt),
        ])
        answer = response.content if isinstance(response.content, str) else str(response.content)
        if truncated:
            answer += OVERFLOW_DISCLAIMER

        return {
            "messages": [AIMessage(content=answer)],
            "analysis_text": answer,
            "offer_store": True,
        }

    return synthesize


# ---------------------------------------------------------------------------
# store_offer: interrupt asking whether to save the analysis to personal_notes
# ---------------------------------------------------------------------------

STORE_PROMPT = "Store this analysis to your notes for future reference? (yes/no)"

_YES_WORDS = {"yes", "y", "save", "store", "ok", "sure", "yep"}


def _wants_store(reply: object) -> bool:
    return str(reply).strip().lower() in _YES_WORDS


def make_store_offer_node(*, embedder=None, chroma_client=None, embed_analyses: bool = True):
    def store_offer(state: AgentState) -> dict:
        reply = interrupt({"kind": "store_analysis", "prompt": STORE_PROMPT})

        if not _wants_store(reply):
            return {"offer_store": False}

        if not embed_analyses:
            return {"offer_store": False, "messages": [AIMessage(content="Okay — saved.")]}

        from src.ingest.embed import embed_analysis

        embed_analysis(
            state.get("analysis_text") or "",
            date=date_cls.today().isoformat(),
            embedder=embedder,
            client=chroma_client,
        )
        return {
            "offer_store": False,
            "messages": [AIMessage(content="Saved this analysis to your notes.")],
        }

    return store_offer
