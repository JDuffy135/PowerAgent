"""LangGraph agent state (ARCHITECTURE.md §4.3)."""
from __future__ import annotations

from typing import Annotated, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]
    intent: str | None                # router output; presettable (e.g. CLI /ingest)
    evidence: list[dict]              # tool results accumulated by ANALYZE (Stage 6)
    pending_batch_id: int | None      # ingest batch awaiting HITL review
    display_unit: str                 # 'lb' | 'kg'

    # Ingest-flow plumbing (beyond the §4.3 minimum):
    file_path: str | None             # upload path for the INGEST pipeline
    correction_rounds: int            # HITL correction passes used on the pending batch
    review_decision: str | None       # last ingest-node outcome, drives conditional edges
    review_note: str | None           # one-shot note prepended to the next interrupt prompt
