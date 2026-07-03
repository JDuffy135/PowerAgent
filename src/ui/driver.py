"""Streamlit-free chat-turn driver over the agent graph.

The CLI's `run_turn` loops interrupts inline (blocking `input()`); a browser UI
can't block, so each `graph.invoke` becomes one `drive_turn` call and the
interrupt round-trip spans Streamlit reruns: the caller stores `TurnResult`
state (`printed`, pending prompt), renders the prompt with approve/reject
buttons, and feeds the user's reply back via `resume_payload`.
"""
from __future__ import annotations

from langchain_core.messages import AIMessage
from langgraph.types import Command

from pydantic import BaseModel, ConfigDict


class TurnResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    replies: tuple[str, ...]          # new assistant messages, in order
    interrupt_prompt: str | None      # non-None -> the graph is paused on HITL
    printed: int                      # messages consumed so far (pass back on resume)


def drive_turn(graph, config: dict, payload, *, printed: int = 0) -> TurnResult:
    """One `graph.invoke`: collect assistant messages that appeared this call
    and surface the interrupt prompt if the graph paused.

    `payload` is either fresh graph input (`make_input(...)`) or a
    `Command(resume=...)` built by `resume_payload`. `printed` is the message
    count from the previous `TurnResult` so replies are never re-shown.
    """
    result = graph.invoke(payload, config)
    messages = result.get("messages", [])
    replies = tuple(
        m.content
        for m in messages[printed:]
        if isinstance(m, AIMessage) and m.content
    )

    interrupt_prompt = None
    if "__interrupt__" in result:
        value = result["__interrupt__"][0].value
        interrupt_prompt = (
            value.get("prompt", str(value)) if isinstance(value, dict) else str(value)
        )

    return TurnResult(
        replies=replies,
        interrupt_prompt=interrupt_prompt,
        printed=len(messages),
    )


def resume_payload(reply: str) -> Command:
    """Wrap the user's answer to an interrupt (button click or free text)."""
    return Command(resume=reply)
