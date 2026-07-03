"""ANALYZE -> SYNTHESIZE -> store-offer graph flow, driven by scripted stubs.

The tool-calling stub emits tool calls that the loop executes against the seeded
in-memory DB (real tools, real evidence); the synthesize stub writes the final
answer. No live models, no Chroma unless a test injects the fakes.
"""
from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.types import Command

from src.agent.graph import build_graph
from src.ingest.embed import PERSONAL_NOTES_COLLECTION
from tests.agent.conftest import (
    RaisingChatModel,
    StubChatModel,
    ToolCallingStubModel,
    tool_call_message,
)

THREAD = {"configurable": {"thread_id": "analyze-thread"}}


def _analyze_graph(conn, checkpointer, analyze_model, synth_model, **overrides):
    kwargs = dict(
        checkpointer=checkpointer,
        router_model_factory=lambda: StubChatModel('{"intent": "analyze"}'),
        chat_model_factory=lambda: RaisingChatModel(),
        analyze_model_factory=lambda: analyze_model,
        synthesize_model_factory=lambda: synth_model,
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


def test_analyze_gathers_evidence_then_synthesizes(conn, checkpointer):
    analyze_model = ToolCallingStubModel(
        tool_call_message("get_best_set", {
            "exercise": "bench press", "date_from": "2026-03-01", "date_to": "2026-03-31",
        }),
        AIMessage(content="done gathering"),
    )
    synth_model = StubChatModel("Your best March bench was 230 lb x1 on 2026-03-19.")
    graph = _analyze_graph(conn, checkpointer, analyze_model, synth_model)

    result = graph.invoke(_q("what was my best bench in March?"), THREAD)

    # Evidence came from the real tool against the seeded DB.
    assert result["evidence"][0]["tool"] == "get_best_set"
    assert result["evidence"][0]["result"]["weight_lb"] == 230.0

    # SYNTHESIZE's answer is the assistant message; then the store offer interrupts.
    assert "230 lb" in _last_ai(result)
    assert result["__interrupt__"][0].value["kind"] == "store_analysis"

    # The synthesize stub received the (unit-labelled) evidence prompt.
    human = synth_model.calls[0][1].content
    assert "230" in human and "lb" in human


def test_analyze_no_evidence_short_circuits_without_store_offer(conn, checkpointer):
    # Model ends immediately without calling any tool.
    analyze_model = ToolCallingStubModel(AIMessage(content="nothing to do"))
    synth_model = RaisingChatModel()  # must NOT be called when evidence is empty
    graph = _analyze_graph(conn, checkpointer, analyze_model, synth_model)

    result = graph.invoke(_q("hmm"), THREAD)
    assert "__interrupt__" not in result
    assert "couldn't find any training data" in _last_ai(result)


def test_analyze_overflow_appends_disclaimer(conn, checkpointer, monkeypatch):
    monkeypatch.setattr("src.agent.nodes.analyze.MAX_TOOL_CALLS", 2)
    # Two turns, both requesting tools -> loop exhausts -> truncated.
    analyze_model = ToolCallingStubModel(
        tool_call_message("get_bodyweight_trend",
                          {"date_from": "2026-01-01", "date_to": "2026-06-30"}, "c1"),
        tool_call_message("get_bodyweight_trend",
                          {"date_from": "2026-01-01", "date_to": "2026-06-30"}, "c2"),
    )
    synth_model = StubChatModel("Bodyweight rose over the prep.")
    graph = _analyze_graph(conn, checkpointer, analyze_model, synth_model)

    result = graph.invoke(_q("summarize everything about my training"), THREAD)
    assert result["evidence_truncated"] is True
    answer = _last_ai(result)
    assert "narrowing the scope" in answer
    assert result["__interrupt__"][0].value["kind"] == "store_analysis"


def test_store_offer_yes_embeds_analysis(conn, checkpointer, fake_embedder, chroma_client):
    analyze_model = ToolCallingStubModel(
        tool_call_message("get_best_set", {
            "exercise": "deadlift", "date_from": "2026-01-01", "date_to": "2026-12-31",
        }),
        AIMessage(content="done"),
    )
    synth_model = StubChatModel("Best deadlift: 385 lb x1 on 2026-06-01.")
    graph = _analyze_graph(
        conn, checkpointer, analyze_model, synth_model,
        embedder=fake_embedder, chroma_client=chroma_client,
    )

    graph.invoke(_q("best deadlift ever?"), THREAD)
    result = graph.invoke(Command(resume="yes"), THREAD)

    assert "__interrupt__" not in result
    assert "Saved this analysis" in _last_ai(result)
    collection = chroma_client.get_or_create_collection(PERSONAL_NOTES_COLLECTION)
    stored = collection.get(where={"doc_type": "analysis"})
    assert len(stored["ids"]) == 1
    assert "385 lb" in stored["documents"][0]


def test_store_offer_no_writes_nothing(conn, checkpointer, fake_embedder, chroma_client):
    analyze_model = ToolCallingStubModel(
        tool_call_message("get_best_set", {
            "exercise": "deadlift", "date_from": "2026-01-01", "date_to": "2026-12-31",
        }),
        AIMessage(content="done"),
    )
    synth_model = StubChatModel("Best deadlift: 385 lb x1.")
    graph = _analyze_graph(
        conn, checkpointer, analyze_model, synth_model,
        embedder=fake_embedder, chroma_client=chroma_client,
    )

    graph.invoke(_q("best deadlift ever?"), THREAD)
    result = graph.invoke(Command(resume="no"), THREAD)

    assert "__interrupt__" not in result
    collection = chroma_client.get_or_create_collection(PERSONAL_NOTES_COLLECTION)
    assert collection.get(where={"doc_type": "analysis"})["ids"] == []
