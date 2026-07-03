"""UPDATE_STATS parse -> confirm-before-write graph flow (scripted stubs, no models)."""
from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.types import Command

from src.agent.graph import build_graph
from src.agent.nodes.update_stats import StatUpdate
from tests.agent.conftest import RaisingChatModel, StubChatModel, scripted_llm

THREAD = {"configurable": {"thread_id": "stats-thread"}}


def _stats_graph(conn, checkpointer, stat_json):
    return build_graph(
        conn,
        checkpointer=checkpointer,
        router_model_factory=lambda: StubChatModel('{"intent": "update_stats"}'),
        chat_model_factory=lambda: RaisingChatModel(),
        update_stats_llm_factory=lambda: scripted_llm(stat_json),
    )


def _msg(text):
    return {"messages": [HumanMessage(content=text)], "intent": None, "file_path": None}


def _last_ai(result):
    for message in reversed(result["messages"]):
        if isinstance(message, AIMessage) and message.content:
            return message.content
    raise AssertionError("no AIMessage")


def test_bodyweight_confirm_and_insert(conn, checkpointer):
    stat = StatUpdate(kind="bodyweight", weight_lb=147.5, date="2026-07-02").model_dump_json()
    graph = _stats_graph(conn, checkpointer, stat)
    before = conn.execute("SELECT COUNT(*) AS c FROM bodyweight").fetchone()["c"]

    result = graph.invoke(_msg("bodyweight was 147.5 today"), THREAD)
    interrupt = result["__interrupt__"][0].value
    assert interrupt["kind"] == "stat_confirm"
    assert "147.5 lb" in interrupt["prompt"]
    assert before == conn.execute("SELECT COUNT(*) AS c FROM bodyweight").fetchone()["c"]

    result = graph.invoke(Command(resume="yes"), THREAD)
    assert "__interrupt__" not in result
    assert "Recorded bodyweight" in _last_ai(result)
    row = conn.execute(
        "SELECT * FROM bodyweight ORDER BY bw_id DESC LIMIT 1"
    ).fetchone()
    assert row["weight_lb"] == 147.5 and row["date"] == "2026-07-02"


def test_pr_confirm_and_insert(conn, checkpointer):
    stat = StatUpdate(
        kind="pr", exercise="deadlift", weight_lb=405.0, reps=1, date="2026-07-02", context="gym"
    ).model_dump_json()
    graph = _stats_graph(conn, checkpointer, stat)

    result = graph.invoke(_msg("hit a 405 deadlift PR"), THREAD)
    assert "Deadlift" in result["__interrupt__"][0].value["prompt"]

    result = graph.invoke(Command(resume="yes"), THREAD)
    assert "Recorded PR" in _last_ai(result)
    row = conn.execute("SELECT * FROM pr ORDER BY pr_id DESC LIMIT 1").fetchone()
    assert row["weight_lb"] == 405.0 and row["reps"] == 1


def test_confirm_no_writes_nothing(conn, checkpointer):
    stat = StatUpdate(kind="bodyweight", weight_lb=147.5, date="2026-07-02").model_dump_json()
    graph = _stats_graph(conn, checkpointer, stat)
    before = conn.execute("SELECT COUNT(*) AS c FROM bodyweight").fetchone()["c"]

    graph.invoke(_msg("bodyweight 147.5"), THREAD)
    result = graph.invoke(Command(resume="no"), THREAD)

    assert "nothing recorded" in _last_ai(result)
    assert before == conn.execute("SELECT COUNT(*) AS c FROM bodyweight").fetchone()["c"]


def test_confirm_reask_on_garbled_reply(conn, checkpointer):
    stat = StatUpdate(kind="bodyweight", weight_lb=147.5, date="2026-07-02").model_dump_json()
    graph = _stats_graph(conn, checkpointer, stat)

    graph.invoke(_msg("bodyweight 147.5"), THREAD)
    result = graph.invoke(Command(resume="maybe?"), THREAD)
    assert result["__interrupt__"][0].value["kind"] == "stat_confirm"
    assert "Didn't understand" in result["__interrupt__"][0].value["prompt"]

    result = graph.invoke(Command(resume="yes"), THREAD)
    assert "Recorded bodyweight" in _last_ai(result)


def test_out_of_scope_declines_without_interrupt(conn, checkpointer):
    stat = StatUpdate(kind="none").model_dump_json()
    graph = _stats_graph(conn, checkpointer, stat)

    result = graph.invoke(_msg("my knee hurts"), THREAD)
    assert "__interrupt__" not in result
    assert "only record bodyweight and PRs" in _last_ai(result)


def test_pr_with_unknown_exercise_declines(conn, checkpointer):
    stat = StatUpdate(
        kind="pr", exercise="zercher zombie squat", weight_lb=200.0, reps=1
    ).model_dump_json()
    graph = _stats_graph(conn, checkpointer, stat)

    result = graph.invoke(_msg("hit a zercher zombie squat pr"), THREAD)
    assert "__interrupt__" not in result
    assert "don't have an exercise matching" in _last_ai(result)


def test_display_unit_kg_in_confirm_prompt(conn, checkpointer):
    stat = StatUpdate(kind="bodyweight", weight_lb=100.0, date="2026-07-02").model_dump_json()
    graph = _stats_graph(conn, checkpointer, stat)

    result = graph.invoke(
        {"messages": [HumanMessage(content="bw 100")], "intent": None,
         "file_path": None, "display_unit": "kg"},
        THREAD,
    )
    # 100 lb -> 45.4 kg at presentation; the stored value stays lb.
    assert "45.4 kg" in result["__interrupt__"][0].value["prompt"]
