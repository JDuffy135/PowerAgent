"""Chat tab: the CLI REPL's browser equivalent, plus drag-and-drop ingestion.

The graph, thread, and interrupt round-trip are identical to `src/cli.py`; the
only difference is that an interrupt spans Streamlit reruns instead of blocking
on `input()` (see `src/ui/driver.py`). Quick-reply buttons send `yes`/`no` --
tokens every confirm parser (ingest review, stat confirm, draft confirm)
accepts -- and free text still works for corrections/block picks.

File uploads replace the CLI's `/ingest` + `/learn` path arguments: the file is
saved under `data/uploads/` (the graph input carries a path, not bytes) and
routed to the HITL ingest flow or the knowledge base.
"""
from __future__ import annotations

import uuid
from pathlib import Path

import streamlit as st

from src.cli import make_input
from src.ingest.knowledge import KnowledgeDoc, ingest_knowledge_file
from src.ui.driver import drive_turn, resume_payload

UPLOADS_DIR = Path(__file__).parent.parent.parent / "data" / "uploads"


def _state() -> None:
    ss = st.session_state
    ss.setdefault("chat_history", [])       # {"role", "content", "kind"} dicts
    ss.setdefault("thread_id", uuid.uuid4().hex)
    ss.setdefault("chat_printed", 0)        # messages consumed on this thread
    ss.setdefault("pending_interrupt", False)


def _config() -> dict:
    return {
        "configurable": {"thread_id": st.session_state.thread_id},
        "recursion_limit": 100,
    }


def _invoke(graph, payload) -> None:
    """One graph call; fold replies + any interrupt prompt into the history."""
    ss = st.session_state
    try:
        with st.spinner("thinking..."):
            result = drive_turn(graph, _config(), payload, printed=ss.chat_printed)
    except Exception as exc:  # a dead Ollama server shouldn't kill the app
        ss.chat_history.append(
            {"role": "assistant", "content": f"error: {exc}", "kind": "error"}
        )
        ss.pending_interrupt = False
        return

    for reply in result.replies:
        ss.chat_history.append({"role": "assistant", "content": reply, "kind": "chat"})
    if result.interrupt_prompt:
        ss.chat_history.append(
            {"role": "assistant", "content": result.interrupt_prompt, "kind": "review"}
        )
    ss.chat_printed = result.printed
    ss.pending_interrupt = result.interrupt_prompt is not None


def _send(graph, text: str) -> None:
    st.session_state.chat_history.append({"role": "user", "content": text, "kind": "chat"})
    if st.session_state.pending_interrupt:
        _invoke(graph, resume_payload(text))
    else:
        _invoke(graph, make_input(text))


def _save_upload(uploaded) -> Path:
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    # The upload name is client-supplied: keep only the basename so a crafted
    # name (e.g. containing "../") can never write outside data/uploads/.
    safe_name = Path(uploaded.name).name
    if not safe_name or safe_name in {".", ".."}:
        safe_name = "upload"
    dest = UPLOADS_DIR / safe_name
    dest.write_bytes(uploaded.getbuffer())
    return dest


def _sidebar(graph) -> None:
    with st.sidebar:
        st.subheader("Files")
        uploaded = st.file_uploader(
            "Training log or reference material", type=["txt", "xlsx", "pdf"]
        )
        if uploaded is not None:
            if st.button("Ingest as training log (HITL review)", width="stretch"):
                path = _save_upload(uploaded)
                _send(graph, f"/ingest {path}")
                st.rerun()
            with st.expander("Add to knowledge base instead"):
                topic = st.text_input("Topic (optional)", key="learn_topic")
                title = st.text_input("Title (optional)", key="learn_title")
                st.caption("Anything left blank is guessed by the local model.")
                if st.button("Learn (no review)", width="stretch"):
                    path = _save_upload(uploaded)
                    doc = KnowledgeDoc(topic=topic or None, title=title or None)
                    try:
                        with st.spinner("embedding..."):
                            n = ingest_knowledge_file(str(path), doc=doc)
                        st.success(f"Learned {uploaded.name} ({n} chunks).")
                    except Exception as exc:
                        st.error(f"Could not learn {uploaded.name}: {exc}")

        st.divider()
        if st.button("New conversation", width="stretch"):
            st.session_state.thread_id = uuid.uuid4().hex
            st.session_state.chat_history = []
            st.session_state.chat_printed = 0
            st.session_state.pending_interrupt = False
            st.rerun()


def render(conn, graph) -> None:
    _state()
    _sidebar(graph)

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            if msg["kind"] == "review":
                st.text(msg["content"])  # rendered batches are preformatted text
            elif msg["kind"] == "error":
                st.error(msg["content"])
            else:
                st.markdown(msg["content"])

    if st.session_state.pending_interrupt:
        col_yes, col_no, _pad = st.columns([1, 1, 4])
        if col_yes.button("✅ Approve / yes"):
            _send(graph, "yes")
            st.rerun()
        if col_no.button("❌ Reject / no"):
            _send(graph, "no")
            st.rerun()
        st.caption("...or type a correction / answer below.")

    placeholder = (
        "your reply (correction, block pick, yes/no)..."
        if st.session_state.pending_interrupt
        else "ask about your training, report a stat, request a program..."
    )
    if text := st.chat_input(placeholder):
        _send(graph, text)
        st.rerun()
