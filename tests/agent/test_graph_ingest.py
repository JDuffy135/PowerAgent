"""Graph-level tests: routing fan-out and the full ingest HITL flows.

Everything runs with stub LLMs against the seeded in-memory DB and an in-memory
checkpointer -- no Ollama, no Chroma writes (`embed_prose=False`).
"""
from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.types import Command

from src.agent.graph import build_graph
from tests.agent.conftest import RaisingChatModel, StubChatModel, golden_batch, scripted_llm

THREAD = {"configurable": {"thread_id": "test-thread"}}


def _make_graph(conn, checkpointer, **overrides):
    kwargs = dict(
        checkpointer=checkpointer,
        router_model_factory=lambda: RaisingChatModel(),
        chat_model_factory=lambda: RaisingChatModel(),
        extract_llm_factory=lambda: scripted_llm(golden_batch().model_dump_json()),
        correction_llm_factory=lambda: RaisingChatModel(),
        embed_prose=False,
    )
    kwargs.update(overrides)
    return build_graph(conn, **kwargs)


def _ingest_input(path):
    return {
        "messages": [HumanMessage(content=f"/ingest {path}")],
        "intent": "ingest",
        "file_path": str(path),
    }


def _chat_input(text):
    return {"messages": [HumanMessage(content=text)], "intent": None, "file_path": None}


def _last_ai(result) -> str:
    for message in reversed(result["messages"]):
        if isinstance(message, AIMessage):
            return message.content
    raise AssertionError("no AIMessage in result")


def _counts(conn):
    return {
        table: conn.execute(f"SELECT COUNT(*) AS c FROM {table}").fetchone()["c"]
        for table in ("session", "lift_set", "programmed_slot", "program", "block")
    }


# ---------------------------------------------------------------------------
# Routing fan-out
# ---------------------------------------------------------------------------
# ANALYZE/UPDATE_STATS/GENERATE routing + flows are covered in
# test_graph_analyze.py / test_graph_update_stats.py / test_graph_generate.py.

def test_chat_intent_routes_to_chitchat(conn, checkpointer):
    graph = _make_graph(
        conn,
        checkpointer,
        router_model_factory=lambda: StubChatModel('{"intent": "chat"}'),
        chat_model_factory=lambda: StubChatModel("Hey! Ready to lift?"),
    )
    result = graph.invoke(_chat_input("hello"), THREAD)
    assert _last_ai(result) == "Hey! Ready to lift?"


# ---------------------------------------------------------------------------
# Ingest error paths (no interrupt, nothing staged)
# ---------------------------------------------------------------------------

def test_ingest_without_path_asks_for_command(conn, checkpointer):
    graph = _make_graph(conn, checkpointer)
    result = graph.invoke(
        {"messages": [HumanMessage(content="/ingest")], "intent": "ingest", "file_path": None},
        THREAD,
    )
    assert "__interrupt__" not in result
    assert "/ingest <path>" in _last_ai(result)


def test_ingest_missing_file_reports_error(conn, checkpointer, tmp_path):
    graph = _make_graph(conn, checkpointer)
    result = graph.invoke(_ingest_input(tmp_path / "nope.txt"), THREAD)
    assert "__interrupt__" not in result
    assert "Could not load" in _last_ai(result)


# ---------------------------------------------------------------------------
# Approve / reject flows
# ---------------------------------------------------------------------------

def test_ingest_approve_unattached_commits(conn, checkpointer, log_file):
    graph = _make_graph(conn, checkpointer)
    before = _counts(conn)

    result = graph.invoke(_ingest_input(log_file), THREAD)
    assert "__interrupt__" in result
    review = result["__interrupt__"][0].value
    assert review["kind"] == "ingest_review"
    assert "Ingest review" in review["prompt"]
    assert "Seal Rows" in review["prompt"]
    assert before == _counts(conn)  # nothing durable before approval

    result = graph.invoke(Command(resume="approve"), THREAD)
    assert result["__interrupt__"][0].value["kind"] == "block_assign"
    assert before == _counts(conn)  # still nothing until block question answered

    result = graph.invoke(Command(resume="none"), THREAD)
    assert "__interrupt__" not in result
    after = _counts(conn)
    assert after["session"] == before["session"] + 1
    assert after["lift_set"] == before["lift_set"] + 2
    assert after["programmed_slot"] == before["programmed_slot"]  # unattached: slot skipped
    assert after["program"] == before["program"]

    summary = _last_ai(result)
    assert "Committed batch" in summary
    assert "unattached" in summary
    assert "preserved in the audit trail" in summary

    status = conn.execute("SELECT status FROM ingest_batch ORDER BY batch_id DESC").fetchone()
    assert status["status"] == "committed"

    # The committed session is unattached and Seal Rows was created as new.
    row = conn.execute("SELECT block_id FROM session WHERE date='2026-06-24'").fetchone()
    assert row["block_id"] is None
    assert conn.execute(
        "SELECT 1 FROM exercise WHERE lower(name) = 'seal rows'"
    ).fetchone() is not None


def test_ingest_reject_writes_nothing(conn, checkpointer, log_file):
    graph = _make_graph(conn, checkpointer)
    before = _counts(conn)

    graph.invoke(_ingest_input(log_file), THREAD)
    result = graph.invoke(Command(resume="reject"), THREAD)

    assert "__interrupt__" not in result
    assert "rejected" in _last_ai(result)
    assert before == _counts(conn)
    status = conn.execute("SELECT status FROM ingest_batch ORDER BY batch_id DESC").fetchone()
    assert status["status"] == "rejected"


# ---------------------------------------------------------------------------
# Correction loop
# ---------------------------------------------------------------------------

def test_correction_full_reemit_then_approve(conn, checkpointer, log_file):
    corrected = golden_batch()
    corrected.sessions[0].sets[0].weight_lb = 320.0
    correction_llm = scripted_llm(corrected.model_dump_json())

    graph = _make_graph(conn, checkpointer, correction_llm_factory=lambda: correction_llm)

    graph.invoke(_ingest_input(log_file), THREAD)
    result = graph.invoke(Command(resume="the top squat set was actually 320"), THREAD)

    # Correction prompt carried the original JSON + the user's text.
    assert "315" in correction_llm.prompts[0]
    assert "actually 320" in correction_llm.prompts[0]

    # Re-rendered review shows the corrected weight.
    review = result["__interrupt__"][0].value
    assert review["kind"] == "ingest_review"
    assert "320 lb" in review["prompt"]

    graph.invoke(Command(resume="approve"), THREAD)
    result = graph.invoke(Command(resume="none"), THREAD)
    assert "Committed batch" in _last_ai(result)

    weight = conn.execute(
        "SELECT weight_lb FROM lift_set WHERE is_top_set=1 ORDER BY set_id DESC"
    ).fetchone()["weight_lb"]
    assert weight == 320.0


def test_correction_failure_keeps_previous_parse(conn, checkpointer, log_file):
    graph = _make_graph(
        conn, checkpointer, correction_llm_factory=lambda: scripted_llm("not json")
    )

    graph.invoke(_ingest_input(log_file), THREAD)
    result = graph.invoke(Command(resume="change something"), THREAD)

    review = result["__interrupt__"][0].value
    assert "Correction could not be applied" in review["prompt"]
    assert "315 lb" in review["prompt"]  # unchanged parse re-rendered


def test_correction_loop_is_capped(conn, checkpointer, log_file, monkeypatch):
    monkeypatch.setattr("src.agent.nodes.ingest.MAX_CORRECTION_ROUNDS", 0)
    graph = _make_graph(
        conn, checkpointer, correction_llm_factory=lambda: RaisingChatModel()
    )

    result = graph.invoke(_ingest_input(log_file), THREAD)
    assert "Correction limit reached" in result["__interrupt__"][0].value["prompt"]

    # Free text at the cap is not applied (RaisingChatModel would fail); loop re-asks.
    result = graph.invoke(Command(resume="please change the squat"), THREAD)
    assert "no longer accepted" in result["__interrupt__"][0].value["prompt"]

    # approve still works.
    result = graph.invoke(Command(resume="approve"), THREAD)
    assert result["__interrupt__"][0].value["kind"] == "block_assign"


# ---------------------------------------------------------------------------
# Block assignment
# ---------------------------------------------------------------------------

def test_block_assignment_existing_block(conn, checkpointer, log_file):
    block_id = conn.execute("SELECT block_id FROM block LIMIT 1").fetchone()["block_id"]
    graph = _make_graph(conn, checkpointer)

    graph.invoke(_ingest_input(log_file), THREAD)
    result = graph.invoke(Command(resume="approve"), THREAD)
    prompt = result["__interrupt__"][0].value["prompt"]
    assert f"[{block_id}]" in prompt  # existing blocks are listed

    slots_before = conn.execute(
        "SELECT COUNT(*) AS c FROM programmed_slot WHERE block_id = ?", (block_id,)
    ).fetchone()["c"]
    result = graph.invoke(Command(resume=str(block_id)), THREAD)
    assert "__interrupt__" not in result

    row = conn.execute("SELECT block_id FROM session WHERE date='2026-06-24'").fetchone()
    assert row["block_id"] == block_id

    # The programmed slot landed on the block, resolved to the squat.
    slot = conn.execute(
        """SELECT ps.*, e.name FROM programmed_slot ps
           LEFT JOIN exercise e ON e.exercise_id = ps.exercise_id
           WHERE ps.block_id = ? ORDER BY ps.slot_id DESC""",
        (block_id,),
    ).fetchone()
    assert conn.execute(
        "SELECT COUNT(*) AS c FROM programmed_slot WHERE block_id = ?", (block_id,)
    ).fetchone()["c"] == slots_before + 1
    assert slot["prescription"] == "1x5 @ RPE 8"
    assert slot["name"] == "Low Bar Squat"


def test_block_assignment_new_program_and_block(conn, checkpointer, log_file):
    graph = _make_graph(conn, checkpointer)

    graph.invoke(_ingest_input(log_file), THREAD)
    graph.invoke(Command(resume="approve"), THREAD)
    result = graph.invoke(Command(resume="new 2026 Meet 2 Prep / Intro Block"), THREAD)
    assert "__interrupt__" not in result
    assert "Intro Block" in _last_ai(result)

    program = conn.execute(
        "SELECT * FROM program WHERE name = '2026 Meet 2 Prep'"
    ).fetchone()
    assert program is not None
    assert program["status"] == "incomplete"
    assert program["start_date"] == "2026-06-24"  # earliest session date in the batch

    block = conn.execute(
        "SELECT * FROM block WHERE name = 'Intro Block' AND program_id = ?",
        (program["program_id"],),
    ).fetchone()
    assert block is not None

    row = conn.execute("SELECT block_id FROM session WHERE date='2026-06-24'").fetchone()
    assert row["block_id"] == block["block_id"]


def test_block_assignment_reuses_existing_program_by_name(conn, checkpointer, log_file):
    programs_before = conn.execute("SELECT COUNT(*) AS c FROM program").fetchone()["c"]
    graph = _make_graph(conn, checkpointer)

    graph.invoke(_ingest_input(log_file), THREAD)
    graph.invoke(Command(resume="approve"), THREAD)
    graph.invoke(Command(resume="new 2026 meet 1 prep / Extra Block"), THREAD)

    assert conn.execute("SELECT COUNT(*) AS c FROM program").fetchone()["c"] == programs_before
    block = conn.execute("SELECT * FROM block WHERE name = 'Extra Block'").fetchone()
    seeded_program = conn.execute(
        "SELECT program_id FROM program WHERE name = '2026 Meet 1 Prep'"
    ).fetchone()
    assert block["program_id"] == seeded_program["program_id"]


def test_block_assignment_invalid_replies_reask(conn, checkpointer, log_file):
    graph = _make_graph(conn, checkpointer)

    graph.invoke(_ingest_input(log_file), THREAD)
    graph.invoke(Command(resume="approve"), THREAD)

    result = graph.invoke(Command(resume="whatever nonsense"), THREAD)
    prompt = result["__interrupt__"][0].value["prompt"]
    assert result["__interrupt__"][0].value["kind"] == "block_assign"
    assert "Didn't understand" in prompt

    result = graph.invoke(Command(resume="99999"), THREAD)
    assert "No block with id 99999" in result["__interrupt__"][0].value["prompt"]

    result = graph.invoke(Command(resume="none"), THREAD)
    assert "__interrupt__" not in result
    assert "Committed batch" in _last_ai(result)
