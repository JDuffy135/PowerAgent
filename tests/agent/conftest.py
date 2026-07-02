"""Shared stubs for the agent-graph tests.

House rule: no live models in tests. Chat models are stubbed with a minimal
`.invoke` object (all our nodes need), raw-callable LLM seams with a
scripted `prompt -> str` function, and the checkpointer is in-memory.
"""
from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage

from src.ingest.models import (
    ParsedBatch,
    ParsedProgrammedSlot,
    ParsedSession,
    ParsedSet,
)


class StubChatModel:
    """Scripted chat model: returns canned responses in order, records calls."""

    def __init__(self, *responses: str):
        self.responses = list(responses)
        self.calls: list = []

    def invoke(self, messages):
        self.calls.append(messages)
        return AIMessage(content=self.responses.pop(0))


class RaisingChatModel:
    """Fails the test if any node actually invokes it."""

    def invoke(self, messages):
        raise AssertionError("chat model should not have been called")


def scripted_llm(*responses: str):
    """A raw `prompt -> str` LLM callable that replays canned responses and
    records the prompts it saw (on `.prompts`)."""
    remaining = list(responses)
    prompts: list[str] = []

    def _call(prompt: str) -> str:
        prompts.append(prompt)
        return remaining.pop(0)

    _call.prompts = prompts
    return _call


def golden_batch() -> ParsedBatch:
    """What the stub extraction LLM 'parses' from the fixture log: one session
    with a resolvable squat, an unknown accessory, and one programmed slot."""
    return ParsedBatch(
        sessions=[
            ParsedSession(
                date="2026-06-24",
                day_label="w1d1",
                week_number=1,
                day_number=1,
                raw_note="Squats moved well. Tried seal rows for the first time.",
                sets=[
                    ParsedSet(
                        exercise_raw="low bar squat",
                        set_index=1,
                        weight_lb=315.0,
                        reps=5,
                        rpe=8.0,
                        is_top_set=True,
                        raw_text="315x5 @8",
                    ),
                    ParsedSet(
                        exercise_raw="Seal Rows",
                        set_index=1,
                        weight_lb=135.0,
                        reps=10,
                        raw_text="135x10",
                        confidence=0.9,
                    ),
                ],
                programmed_slots=[
                    ParsedProgrammedSlot(
                        exercise_raw="low bar squat",
                        prescription="1x5 @ RPE 8",
                        target_weight_lb=315.0,
                    )
                ],
            )
        ]
    )


@pytest.fixture()
def checkpointer():
    from langgraph.checkpoint.memory import InMemorySaver

    return InMemorySaver()


@pytest.fixture()
def log_file(tmp_path):
    path = tmp_path / "w1d1.txt"
    path.write_text("W1D1\nSQUAT 315x5 @8\nSEAL ROWS 135x10\n", encoding="utf-8")
    return path
