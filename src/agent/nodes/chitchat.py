"""CHITCHAT fallback node + placeholders for the not-yet-built branches."""
from __future__ import annotations

from typing import Callable

from langchain_core.messages import AIMessage, SystemMessage

from src.agent.state import AgentState

CHITCHAT_SYSTEM_PROMPT = (
    "You are a friendly, concise powerlifting-coach assistant. You can ingest "
    "training logs (`/ingest <path>`), and soon you'll analyze training history "
    "and draft programs. Keep replies short."
)


def make_chitchat_node(model_factory: Callable[[], object]):
    def chitchat(state: AgentState) -> dict:
        response = model_factory().invoke(
            [SystemMessage(content=CHITCHAT_SYSTEM_PROMPT), *state["messages"]]
        )
        return {"messages": [response]}

    return chitchat


def make_placeholder_node(name: str, stage: str):
    """A branch that exists in the topology but isn't implemented yet."""

    def placeholder(state: AgentState) -> dict:
        return {"messages": [AIMessage(content=(
            f"The {name} capability isn't implemented yet (it lands in {stage})."
        ))]}

    return placeholder
