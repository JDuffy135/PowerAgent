"""GENERATE -> draft-confirm graph flow, driven by scripted stubs.

The tool-calling stub gathers evidence via real tools against the seeded DB;
the scripted raw-LLM stub emits the structured DraftProgram JSON. Draft rows
land in program/block/programmed_slot only on the approved branch, with
status='draft' (excluded from analysis by default). No live models.
"""
from __future__ import annotations

import json

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.types import Command

from src.agent.graph import build_graph
from src.tools.queries import get_block_outline, get_programs
from tests.agent.conftest import (
    RaisingChatModel,
    StubChatModel,
    ToolCallingStubModel,
    scripted_llm,
    tool_call_message,
)

THREAD = {"configurable": {"thread_id": "generate-thread"}}


def _draft_json(**overrides) -> str:
    draft = {
        "program_name": "2026 Meet 2 Prep",
        "block_name": "Strength Block 1",
        "focus": "strength",
        "week_count": 1,
        "goals_text": "Build the top-end for meet 2.",
        "notes": None,
        "slots": [
            {
                "exercise": "bench press",  # alias -> resolves to Bench Press
                "week_number": 1,
                "day_number": 1,
                "day_label": "w1d1",
                "prescription": "1x1 @ RPE 6, 4x4 @ RPE 7",
                "target_weight_lb": 225.0,
                "notes": None,
            },
            {
                "exercise": "Safety Bar Squat",  # NOT in the dictionary
                "week_number": 1,
                "day_number": 2,
                "day_label": "w1d2",
                "prescription": "4x5 @ RPE 7",
                "target_weight_lb": None,
                "notes": "weak-point variation",
            },
        ],
    }
    draft.update(overrides)
    return json.dumps(draft)


def _generate_graph(conn, checkpointer, gather_model, draft_llm, **overrides):
    kwargs = dict(
        checkpointer=checkpointer,
        router_model_factory=lambda: StubChatModel('{"intent": "generate"}'),
        chat_model_factory=lambda: RaisingChatModel(),
        analyze_model_factory=lambda: RaisingChatModel(),
        synthesize_model_factory=lambda: RaisingChatModel(),
        generate_model_factory=lambda: gather_model,
        generate_llm_factory=lambda: draft_llm,
    )
    kwargs.update(overrides)
    return build_graph(conn, **kwargs)


def _q(text):
    return {"messages": [HumanMessage(content=text)], "intent": None, "file_path": None}


def _last_ai(result):
    for message in reversed(result["messages"]):
        if isinstance(message, AIMessage) and message.content:
            return message.content
    raise AssertionError("no AIMessage")


def _gather_model():
    """Stub that pulls injuries (real tool) then stops gathering."""
    return ToolCallingStubModel(
        tool_call_message("get_injuries", {"active_only": True}),
        AIMessage(content="done gathering"),
    )


def test_generate_gathers_evidence_and_interrupts_with_draft(conn, checkpointer):
    draft_llm = scripted_llm(_draft_json())
    graph = _generate_graph(conn, checkpointer, _gather_model(), draft_llm)

    result = graph.invoke(_q("write me a strength block"), THREAD)

    # Evidence came from the real tool against the seeded DB.
    assert result["evidence"][0]["tool"] == "get_injuries"
    # The draft prompt carried the user request and the evidence.
    assert "write me a strength block" in draft_llm.prompts[0]
    assert "get_injuries" in draft_llm.prompts[0]

    # Rendered draft printed before the confirm interrupt.
    rendered = _last_ai(result)
    assert "2026 Meet 2 Prep" in rendered
    assert "Safety Bar Squat" in rendered
    assert "Week 1, Day 1" in rendered
    assert result["__interrupt__"][0].value["kind"] == "draft_confirm"

    # Nothing durable before approval.
    assert get_programs(conn, "draft") == []


def test_generate_approve_persists_draft_program(conn, checkpointer):
    graph = _generate_graph(conn, checkpointer, _gather_model(), scripted_llm(_draft_json()))

    graph.invoke(_q("write me a strength block"), THREAD)
    result = graph.invoke(Command(resume="yes"), THREAD)

    assert "__interrupt__" not in result
    summary = _last_ai(result)
    assert "Saved draft program" in summary
    assert "Safety Bar Squat" in summary  # unresolved exercise flagged

    drafts = get_programs(conn, "draft")
    assert len(drafts) == 1
    assert drafts[0].name == "2026 Meet 2 Prep"
    assert drafts[0].status == "draft"

    block = conn.execute(
        "SELECT * FROM block WHERE program_id = ?", (drafts[0].program_id,)
    ).fetchone()
    assert block["name"] == "Strength Block 1"
    assert block["focus"] == "strength"
    assert block["week_count"] == 1

    outline = get_block_outline(conn, block["block_id"])
    assert len(outline) == 2
    assert outline[0].exercise == "Bench Press"       # alias resolved
    assert outline[0].target_weight_lb == 225.0
    assert outline[1].exercise is None                # unresolved -> NULL link
    assert outline[1].prescription == "4x5 @ RPE 7"


def test_generate_reject_writes_nothing(conn, checkpointer):
    graph = _generate_graph(conn, checkpointer, _gather_model(), scripted_llm(_draft_json()))

    graph.invoke(_q("write me a strength block"), THREAD)
    result = graph.invoke(Command(resume="no"), THREAD)

    assert "__interrupt__" not in result
    assert "discarded" in _last_ai(result)
    assert get_programs(conn, "draft") == []
    assert conn.execute("SELECT COUNT(*) c FROM program WHERE status='draft'").fetchone()["c"] == 0


def test_generate_confirm_reasks_on_gibberish(conn, checkpointer):
    graph = _generate_graph(conn, checkpointer, _gather_model(), scripted_llm(_draft_json()))

    graph.invoke(_q("write me a strength block"), THREAD)
    result = graph.invoke(Command(resume="maybe??"), THREAD)

    # Re-interrupts with a one-shot note; still nothing written.
    assert result["__interrupt__"][0].value["kind"] == "draft_confirm"
    assert "Didn't understand" in result["__interrupt__"][0].value["prompt"]
    assert get_programs(conn, "draft") == []

    result = graph.invoke(Command(resume="yes"), THREAD)
    assert len(get_programs(conn, "draft")) == 1


def test_generate_invalid_draft_json_ends_without_interrupt(conn, checkpointer):
    graph = _generate_graph(
        conn, checkpointer, _gather_model(), scripted_llm("not json at all")
    )

    result = graph.invoke(_q("write me a strength block"), THREAD)

    assert "__interrupt__" not in result
    assert "couldn't produce a valid program draft" in _last_ai(result)
    assert get_programs(conn, "draft") == []


def test_generate_empty_slots_ends_without_interrupt(conn, checkpointer):
    graph = _generate_graph(
        conn, checkpointer, _gather_model(), scripted_llm(_draft_json(slots=[]))
    )

    result = graph.invoke(_q("write me a strength block"), THREAD)

    assert "__interrupt__" not in result
    assert "no programmed slots" in _last_ai(result)
    assert get_programs(conn, "draft") == []


def test_saved_draft_stays_out_of_analysis(conn, checkpointer):
    """Draft exclusion: e1RM/lift queries ignore sessions, and drafts have no
    sessions at all — but get_programs(None) must still show the draft while
    get_programs('incomplete'/'complete') must not."""
    graph = _generate_graph(conn, checkpointer, _gather_model(), scripted_llm(_draft_json()))
    graph.invoke(_q("write me a strength block"), THREAD)
    graph.invoke(Command(resume="yes"), THREAD)

    names = {p.name for p in get_programs(conn)}
    assert "2026 Meet 2 Prep" in names
    assert all(p.status != "draft" for p in get_programs(conn, "incomplete"))
    assert all(p.status != "draft" for p in get_programs(conn, "complete"))
