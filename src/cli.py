"""Minimal chat REPL over the agent graph (ARCHITECTURE.md §7).

Run with `python -m src.cli`. Requires a live Ollama server for routing,
extraction, and chitchat; graph logic itself is covered by stub-LLM tests.

**[DECISION]** File ingestion is a dedicated command, `/ingest <path>`, which
presets `intent='ingest'` and skips the router -- more reliable than intent
classification with small local models. A future UI replaces this with
drag-and-drop / a file picker; the graph input shape stays the same.

The interrupt/resume round-trip: whenever `graph.invoke` returns with
`__interrupt__`, the interrupt's `prompt` is printed and the next input line is
fed back via `Command(resume=...)` -- covering both the batch-review loop
(approve / reject / free-text corrections) and the block-assignment question.
"""
from __future__ import annotations

import uuid
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.types import Command

from src.agent.graph import build_graph, get_checkpointer
from src.agent.llm_provider import CONFIG_PATH, load_config
from src.db.connection import get_conn, init_db

BANNER = """Powerlifting Coach -- CLI REPL
Commands:
  /ingest <path>   ingest a training log file (HITL review before commit)
  exit | quit      leave
Anything else is routed by intent (analyze/generate/update_stats land in Stages 6-7).
"""


def make_input(line: str) -> dict:
    """Turn one REPL line into graph input. `/ingest <path>` presets the intent
    so the router is skipped; everything else goes through classification."""
    if line.startswith("/ingest"):
        path = line[len("/ingest"):].strip() or None
        return {
            "messages": [HumanMessage(content=line)],
            "intent": "ingest",
            "file_path": path,
            "review_decision": None,
            "review_note": None,
        }
    return {
        "messages": [HumanMessage(content=line)],
        "intent": None,
        "file_path": None,
        "review_decision": None,
        "review_note": None,
    }


def run_turn(graph, config: dict, graph_input, read_input=input, write=print) -> None:
    """One user turn: invoke, then service any interrupt round-trips until the
    graph reaches END, and print the final assistant message."""
    result = graph.invoke(graph_input, config)
    while "__interrupt__" in result:
        payload = result["__interrupt__"][0].value
        write(payload.get("prompt", str(payload)) if isinstance(payload, dict) else str(payload))
        reply = read_input("review> ")
        result = graph.invoke(Command(resume=reply), config)

    for message in reversed(result.get("messages", [])):
        if isinstance(message, AIMessage):
            write(f"coach> {message.content}")
            return


def main() -> None:
    cfg = load_config()
    db_path = Path(cfg.get("db_path", "data/training.db"))
    if not db_path.is_absolute():
        db_path = CONFIG_PATH.parent / db_path
    db_path.parent.mkdir(parents=True, exist_ok=True)

    conn = get_conn(db_path)
    init_db(conn)
    checkpointer = get_checkpointer()
    graph = build_graph(conn, checkpointer=checkpointer)
    config = {
        "configurable": {"thread_id": uuid.uuid4().hex},
        "recursion_limit": 100,  # headroom for long correction loops
    }

    print(BANNER)
    while True:
        try:
            line = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue
        if line.lower() in {"exit", "quit", "/exit", "/quit"}:
            break
        run_turn(graph, config, make_input(line))

    conn.close()


if __name__ == "__main__":
    main()
