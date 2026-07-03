"""Agent graph assembly (ARCHITECTURE.md §4.1) + checkpointer.

Topology (Stage 6): ROUTER fans out to INGEST, ANALYZE, UPDATE_STATS, CHITCHAT,
and a placeholder GENERATE (Stage 7). Every `interrupt()` sits at the top of its
own node so the resume-replay contract holds (see `nodes/ingest.py`):

    router -> ingest_parse -> ingest_review <-> (correction loop)
                                   |-> ingest_commit <-> (bad-reply loop) -> END
                                   '-> END (reject)
    router -> analyze (ReAct loop) -> synthesize -> store_offer(interrupt) -> END
                                                 '-> END (no evidence to store)
    router -> update_stats_parse -> update_stats_confirm(interrupt) <-> (reask) -> END
                                '-> END (declined / out of scope)

**[DECISION]** Checkpoints live in a separate `data/checkpoints.db`
(`SqliteSaver`), keeping `training.db`'s schema purely domain data.

All model/embedder dependencies are injectable so graph tests run with stubs
and no live Ollama; when a factory is omitted the real provider is resolved
lazily at node-call time, never at build time. `embed_prose`/`embed_analyses`
gate the two Chroma write paths (session notes, stored analyses) for offline tests.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Callable

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph

from src.agent import llm_provider
from src.agent.nodes.analyze import make_analyze_node
from src.agent.nodes.chitchat import make_chitchat_node, make_placeholder_node
from src.agent.nodes.ingest import (
    make_ingest_commit_node,
    make_ingest_parse_node,
    make_ingest_review_node,
)
from src.agent.nodes.router import make_router_node
from src.agent.nodes.synthesize import make_store_offer_node, make_synthesize_node
from src.agent.nodes.update_stats import (
    make_update_stats_confirm_node,
    make_update_stats_parse_node,
)
from src.agent.state import AgentState

DEFAULT_CHECKPOINTS_PATH = Path(__file__).parent.parent.parent / "data" / "checkpoints.db"


def get_checkpointer(path: str | Path | None = None) -> SqliteSaver:
    """Open the SqliteSaver at `checkpoints_db` from config (default
    `data/checkpoints.db`). Separate file from `training.db` by decision."""
    if path is None:
        raw = llm_provider.load_config().get("checkpoints_db")
        if raw:
            path = Path(raw)
            if not path.is_absolute():
                path = llm_provider.CONFIG_PATH.parent / path
        else:
            path = DEFAULT_CHECKPOINTS_PATH
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return SqliteSaver(sqlite3.connect(path, check_same_thread=False))


def build_graph(
    conn: sqlite3.Connection,
    *,
    checkpointer,
    router_model_factory: Callable[[], object] | None = None,
    chat_model_factory: Callable[[], object] | None = None,
    extract_llm_factory: Callable[[], object] | None = None,
    correction_llm_factory: Callable[[], object] | None = None,
    analyze_model_factory: Callable[[], object] | None = None,
    synthesize_model_factory: Callable[[], object] | None = None,
    update_stats_llm_factory: Callable[[], object] | None = None,
    embedder=None,
    chroma_client=None,
    embed_prose: bool = True,
    embed_analyses: bool = True,
):
    """Compile the agent graph over `conn` (the live training DB connection).

    Factories default to the real config-driven providers; tests inject stubs.
    `extract_llm_factory`/`correction_llm_factory` returning None means "let the
    ingest functions use their own `get_llm` default" -- so the default lambda
    here returns None rather than eagerly building an Ollama callable.
    """
    router_models = router_model_factory or (lambda: llm_provider.get_chat_model("router"))
    chat_models = chat_model_factory or (lambda: llm_provider.get_chat_model("chitchat"))
    extract_llms = extract_llm_factory or (lambda: None)
    correction_llms = correction_llm_factory or (lambda: None)
    analyze_models = analyze_model_factory or (lambda: llm_provider.get_chat_model("analyze"))
    synthesize_models = synthesize_model_factory or (lambda: llm_provider.get_chat_model("synthesize"))
    update_stats_llms = update_stats_llm_factory or (lambda: None)

    graph = StateGraph(AgentState)
    graph.add_node("router", make_router_node(router_models))
    graph.add_node("ingest_parse", make_ingest_parse_node(conn, extract_llms))
    graph.add_node("ingest_review", make_ingest_review_node(conn, correction_llms))
    graph.add_node(
        "ingest_commit",
        make_ingest_commit_node(
            conn, embedder=embedder, chroma_client=chroma_client, embed_prose=embed_prose
        ),
    )
    graph.add_node("chitchat", make_chitchat_node(chat_models))
    graph.add_node(
        "analyze",
        make_analyze_node(conn, analyze_models, embedder=embedder, chroma_client=chroma_client),
    )
    graph.add_node("synthesize", make_synthesize_node(synthesize_models))
    graph.add_node(
        "store_offer",
        make_store_offer_node(
            embedder=embedder, chroma_client=chroma_client, embed_analyses=embed_analyses
        ),
    )
    graph.add_node("update_stats_parse", make_update_stats_parse_node(conn, update_stats_llms))
    graph.add_node("update_stats_confirm", make_update_stats_confirm_node(conn))
    graph.add_node("generate", make_placeholder_node("program-writing", "Stage 7"))

    graph.add_edge(START, "router")
    graph.add_conditional_edges(
        "router",
        lambda state: state.get("intent") or "chat",
        {
            "ingest": "ingest_parse",
            "analyze": "analyze",
            "generate": "generate",
            "update_stats": "update_stats_parse",
            "chat": "chitchat",
        },
    )
    graph.add_conditional_edges(
        "ingest_parse",
        lambda state: "error" if state.get("review_decision") == "error" else "review",
        {"error": END, "review": "ingest_review"},
    )
    graph.add_conditional_edges(
        "ingest_review",
        lambda state: state["review_decision"],
        {"approve": "ingest_commit", "reject": END, "correct": "ingest_review"},
    )
    graph.add_conditional_edges(
        "ingest_commit",
        lambda state: state["review_decision"],
        {"ask_block": "ingest_commit", "done": END},
    )

    # ANALYZE -> SYNTHESIZE -> (optional) store-this-analysis interrupt -> END
    graph.add_edge("analyze", "synthesize")
    graph.add_conditional_edges(
        "synthesize",
        lambda state: "store" if state.get("offer_store") else "end",
        {"store": "store_offer", "end": END},
    )
    graph.add_edge("store_offer", END)

    # UPDATE_STATS: parse -> (confirm loop | decline) -> END
    graph.add_conditional_edges(
        "update_stats_parse",
        lambda state: "confirm" if state.get("pending_stat") else "end",
        {"confirm": "update_stats_confirm", "end": END},
    )
    graph.add_conditional_edges(
        "update_stats_confirm",
        lambda state: state["review_decision"],
        {"reask": "update_stats_confirm", "done": END},
    )

    for node in ("chitchat", "generate"):
        graph.add_edge(node, END)

    return graph.compile(checkpointer=checkpointer)
