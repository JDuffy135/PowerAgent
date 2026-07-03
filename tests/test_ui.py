"""Streamlit-free UI logic: the chat-turn driver and the data-editor diff.

The `*_tab.py` modules are rendering veneers (not tested here beyond import);
everything decision-shaped lives in `src/ui/driver.py` / `src/ui/editing.py`.
"""
from __future__ import annotations

import math
from types import SimpleNamespace

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.types import Command

from src.ui.driver import drive_turn, resume_payload
from src.ui.editing import diff_rows


# ---------------------------------------------------------------------------
# drive_turn
# ---------------------------------------------------------------------------

class FakeGraph:
    """Stands in for the compiled graph: returns queued results, records payloads."""

    def __init__(self, results):
        self.results = list(results)
        self.calls = []

    def invoke(self, payload, config):
        self.calls.append(payload)
        return self.results.pop(0)


def test_drive_turn_collects_new_ai_messages_only():
    messages = [
        HumanMessage(content="old question"),
        AIMessage(content="old answer"),
        HumanMessage(content="new question"),
        AIMessage(content=""),  # tool-call-only message: no content, skipped
        ToolMessage(content="tool output", tool_call_id="x"),
        AIMessage(content="new answer"),
    ]
    graph = FakeGraph([{"messages": messages}])
    result = drive_turn(graph, {}, {"messages": []}, printed=2)
    assert result.replies == ("new answer",)
    assert result.interrupt_prompt is None
    assert result.printed == len(messages)


def test_drive_turn_surfaces_interrupt_prompt_dict_and_str():
    interrupt = SimpleNamespace(value={"prompt": "=== Ingest review ===\napprove?"})
    graph = FakeGraph([
        {"messages": [AIMessage(content="parsed it")], "__interrupt__": [interrupt]},
        {"messages": [AIMessage(content="parsed it"), AIMessage(content="committed")]},
    ])
    first = drive_turn(graph, {}, {"messages": []})
    assert first.replies == ("parsed it",)
    assert first.interrupt_prompt.startswith("=== Ingest review ===")

    second = drive_turn(graph, {}, resume_payload("approve"), printed=first.printed)
    assert second.replies == ("committed",)
    assert second.interrupt_prompt is None
    # The resume actually went through as a Command.
    assert isinstance(graph.calls[1], Command)
    assert graph.calls[1].resume == "approve"


def test_drive_turn_string_interrupt_payload():
    interrupt = SimpleNamespace(value="pick a block")
    graph = FakeGraph([{"messages": [], "__interrupt__": [interrupt]}])
    result = drive_turn(graph, {}, {"messages": []})
    assert result.interrupt_prompt == "pick a block"
    assert result.replies == ()


# ---------------------------------------------------------------------------
# diff_rows
# ---------------------------------------------------------------------------

ORIGINAL = [
    {"bw_id": 2, "date": "2026-06-02", "weight_lb": 147.0, "note": None},
    {"bw_id": 1, "date": "2026-06-01", "weight_lb": 146.5, "note": "am"},
]


def test_diff_no_changes_is_empty():
    plan = diff_rows(ORIGINAL, [dict(r) for r in ORIGINAL], "bw_id")
    assert plan.empty


def test_diff_update_carries_only_changed_columns():
    edited = [dict(r) for r in ORIGINAL]
    edited[0]["weight_lb"] = 147.4
    plan = diff_rows(ORIGINAL, edited, "bw_id")
    assert plan.updates == [(2, {"weight_lb": 147.4})]
    assert not plan.inserts and not plan.deletes


def test_diff_insert_from_blank_pk_and_nan_cleanup():
    edited = [dict(r) for r in ORIGINAL] + [
        {"bw_id": math.nan, "date": "2026-06-03", "weight_lb": 147.8, "note": math.nan},
        {"bw_id": None, "date": "", "weight_lb": None, "note": None},  # blank row: noise
    ]
    plan = diff_rows(ORIGINAL, edited, "bw_id")
    assert plan.inserts == [{"date": "2026-06-03", "weight_lb": 147.8}]


def test_diff_delete_missing_rows():
    plan = diff_rows(ORIGINAL, [dict(ORIGINAL[0])], "bw_id")
    assert plan.deletes == [1]


def test_diff_nan_equals_none_no_spurious_update():
    edited = [dict(r) for r in ORIGINAL]
    edited[0]["note"] = math.nan  # pandas' rendering of the stored NULL
    plan = diff_rows(ORIGINAL, edited, "bw_id")
    assert plan.empty


def test_diff_hand_typed_pk_is_insert():
    edited = [dict(r) for r in ORIGINAL] + [
        {"bw_id": 50, "date": "2026-06-04", "weight_lb": 148.0, "note": None}
    ]
    plan = diff_rows(ORIGINAL, edited, "bw_id")
    assert plan.inserts == [{"bw_id": 50, "date": "2026-06-04", "weight_lb": 148.0, "note": None}]


# ---------------------------------------------------------------------------
# Streamlit tab modules import cleanly (no side effects at import time)
# ---------------------------------------------------------------------------

def test_tab_modules_import():
    import src.ui.backfill_tab  # noqa: F401
    import src.ui.chat_tab  # noqa: F401
    import src.ui.devtools_tab  # noqa: F401
    import src.ui.organizer_tab  # noqa: F401
    import src.ui.trends_tab  # noqa: F401
