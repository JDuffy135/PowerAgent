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

import shlex
import uuid
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.types import Command

from src.agent.graph import build_graph, get_checkpointer
from src.agent.llm_provider import CONFIG_PATH, load_config
from src.db.connection import get_conn, init_db
from src.ingest.knowledge import KnowledgeDoc, get_metadata_llm, ingest_knowledge_file

BANNER = """Powerlifting Coach -- CLI REPL
Commands:
  /ingest <path>   ingest a training log file (HITL review before commit)
  /learn <path> [--topic T] [--title ...] [--author ...] [--year Y] [--source ...]
                   add reference material (study/article/PDF) to the knowledge
                   base -- no review; the LLM guesses any metadata you omit
  exit | quit      leave
Anything else is routed by intent: ask about your history ("best bench in
March?"), report a stat ("bodyweight 146 today", "hit a 405x1 deadlift PR"),
ask for a program ("write me a 4-week strength block"), or just chat.
"""


def parse_learn(line: str) -> tuple[str | None, KnowledgeDoc]:
    """Parse a `/learn <path> [--flag value ...]` line into (path, KnowledgeDoc).

    Recognized flags mirror the knowledge metadata fields: `--source --title
    --topic --author --year`. Anything the user omits stays `None` on the doc, so
    the LLM guess pass fills it in (or it defaults to NULL). Returns `(None, ...)`
    if no path was given.
    """
    tokens = shlex.split(line[len("/learn"):].strip())
    path: str | None = None
    fields: dict[str, str] = {}
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if tok.startswith("--"):
            key = tok[2:]
            value = tokens[i + 1] if i + 1 < len(tokens) else ""
            if key in KnowledgeDoc.model_fields:
                fields[key] = value
            i += 2
            continue
        if path is None:
            path = tok
        i += 1

    if "year" in fields:
        try:
            fields["year"] = int(fields["year"])  # type: ignore[assignment]
        except ValueError:
            fields.pop("year")

    return path, KnowledgeDoc(**fields)


_UNSET = object()


def run_learn(line: str, *, llm=_UNSET, embedder=None, chroma_client=None, write=print) -> None:
    """Handle a `/learn` line: load the file, guess missing metadata, embed it.

    `llm` defaults to the config-routed metadata guesser (`get_metadata_llm`);
    tests pass a stub (or `None` to skip guessing) so no live model is needed.
    """
    path, doc = parse_learn(line)
    if not path:
        write("usage: /learn <path> [--topic T] [--title ...] [--author ...] [--year Y]")
        return
    if llm is _UNSET:
        llm = get_metadata_llm()
    try:
        n = ingest_knowledge_file(
            path,
            doc=doc,
            llm=llm,
            embedder=embedder,
            client=chroma_client,
        )
    except FileNotFoundError:
        write(f"coach> file not found: {path}")
        return
    except Exception as exc:  # loader/embed failures shouldn't kill the REPL
        write(f"coach> could not learn {path}: {exc}")
        return
    write(f"coach> learned {path} ({n} chunk{'s' if n != 1 else ''} embedded into the knowledge base).")


def make_input(line: str) -> dict:
    """Turn one REPL line into graph input. `/ingest <path>` presets the intent
    so the router is skipped; everything else goes through classification."""
    # Reset per-turn scratch so nothing leaks across turns on the persistent thread.
    fresh = {
        "review_decision": None,
        "review_note": None,
        "evidence": [],
        "evidence_truncated": False,
        "analysis_text": None,
        "offer_store": False,
        "pending_stat": None,
        "pending_draft": None,
    }
    if line.startswith("/ingest"):
        path = line[len("/ingest"):].strip() or None
        return {
            "messages": [HumanMessage(content=line)],
            "intent": "ingest",
            "file_path": path,
            **fresh,
        }
    return {
        "messages": [HumanMessage(content=line)],
        "intent": None,
        "file_path": None,
        **fresh,
    }


def run_turn(graph, config: dict, graph_input, read_input=input, write=print) -> None:
    """One user turn: invoke, print any new assistant messages, then service each
    interrupt round-trip until the graph reaches END.

    New AIMessages are printed as they appear (tracked by index into the running
    `messages` list) *before* prompting on an interrupt -- so, e.g., SYNTHESIZE's
    analysis shows up right before the "store this?" question, not swallowed by it.
    """
    printed = 0
    result = graph.invoke(graph_input, config)
    while True:
        messages = result.get("messages", [])
        for message in messages[printed:]:
            if isinstance(message, AIMessage) and message.content:
                write(f"coach> {message.content}")
        printed = len(messages)

        if "__interrupt__" not in result:
            return
        payload = result["__interrupt__"][0].value
        write(payload.get("prompt", str(payload)) if isinstance(payload, dict) else str(payload))
        reply = read_input("review> ")
        result = graph.invoke(Command(resume=reply), config)


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
        if line.startswith("/learn"):
            run_learn(line)
            continue
        run_turn(graph, config, make_input(line))

    conn.close()


if __name__ == "__main__":
    main()
