"""ROUTER node: classify user intent (ARCHITECTURE.md §4.2).

One job, tiny prompt: map the latest user message to one of five intents via
structured JSON output. If the intent is already set on the state (the CLI's
`/ingest <path>` command presets `intent='ingest'`), the LLM is skipped
entirely -- a dedicated command is more reliable than classification with
small local models **[DECISION]**.

Any unparseable/invalid model output falls back to `chat` rather than raising:
a misroute costs one clarifying exchange; a crash costs the session.
"""
from __future__ import annotations

import json
from typing import Callable, Literal

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, ValidationError

from src.agent.state import AgentState

INTENTS = ("ingest", "analyze", "generate", "update_stats", "chat")

ROUTER_SYSTEM_PROMPT = """You classify one user message for a powerlifting-coach assistant. \
Pick exactly one intent:
- ingest: wants to upload/import a training log or program file
- analyze: asks about training history, trends, PRs, volume, injuries, or past notes
- generate: asks you to write/draft a training program or block
- update_stats: reports one new fact to record (bodyweight, a PR, an injury, a measurement)
- chat: greetings, questions about the assistant, anything else

Respond with ONLY this JSON object, no other text:
{"intent": "<ingest|analyze|generate|update_stats|chat>"}"""


class RouterOutput(BaseModel):
    intent: Literal["ingest", "analyze", "generate", "update_stats", "chat"]


def classify_intent(text: str, model) -> str:
    """Classify one message; falls back to 'chat' on any malformed model output."""
    response = model.invoke(
        [SystemMessage(content=ROUTER_SYSTEM_PROMPT), HumanMessage(content=text)]
    )
    content = response.content if isinstance(response.content, str) else str(response.content)

    # Tolerate prose/fences around the JSON object: take the outermost braces.
    start, end = content.find("{"), content.rfind("}")
    if start == -1 or end <= start:
        return "chat"
    try:
        return RouterOutput.model_validate(json.loads(content[start : end + 1])).intent
    except (json.JSONDecodeError, ValidationError):
        return "chat"


def make_router_node(model_factory: Callable[[], object]):
    """Build the router node. `model_factory` defers model construction so no
    Ollama connection happens at graph-build time (tests inject stubs)."""

    def router_node(state: AgentState) -> dict:
        if state.get("intent"):
            return {}  # preset (e.g. CLI /ingest) -- skip classification
        last = state["messages"][-1]
        text = last.content if isinstance(last.content, str) else str(last.content)
        return {"intent": classify_intent(text, model_factory())}

    return router_node
