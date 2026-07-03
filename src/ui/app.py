"""Streamlit entry point (Stage 9).

Run from the repo root:

    streamlit run src/ui/app.py

Wires the same resources as the CLI (live training DB, checkpointer, compiled
graph) and lays out the tabs. Resources are `st.cache_resource`-shared across
reruns/sessions; per-browser-session state (chat thread, pending interrupt)
lives in `st.session_state` (see `chat_tab.py`). The training connection is
opened with `check_same_thread=False` because Streamlit reruns hop threads;
reruns within a session are serialized, so this is safe for a single-user
local app.
"""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

# `streamlit run src/ui/app.py` executes this file as a script, so the repo
# root isn't on sys.path the way `python -m` puts it there.
_ROOT = Path(__file__).parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.agent.graph import build_graph, get_checkpointer  # noqa: E402
from src.agent.llm_provider import CONFIG_PATH, load_config  # noqa: E402
from src.db.connection import get_conn, init_db  # noqa: E402
from src.ui import backfill_tab, chat_tab, devtools_tab, organizer_tab, trends_tab  # noqa: E402

st.set_page_config(page_title="Powerlifting Coach", page_icon="🏋️", layout="wide")


@st.cache_resource
def _resources():
    cfg = load_config()
    db_path = Path(cfg.get("db_path", "data/training.db"))
    if not db_path.is_absolute():
        db_path = CONFIG_PATH.parent / db_path
    db_path.parent.mkdir(parents=True, exist_ok=True)

    conn = get_conn(db_path, check_same_thread=False)
    init_db(conn)
    graph = build_graph(conn, checkpointer=get_checkpointer())
    return conn, graph


conn, graph = _resources()
# Presentation unit from config (§2): the chat flow reads it via `make_input`;
# the read-only tabs take it as a parameter. Storage is always canonical lb.
display_unit = load_config().get("display_unit", "lb")

st.title("🏋️ Powerlifting Coach")

tab_chat, tab_trends, tab_organizer, tab_backfill, tab_dev = st.tabs(
    ["💬 Chat", "📈 Trends", "🗂️ Organizer", "📥 Backfill", "🛠️ Dev Tools"]
)
with tab_chat:
    chat_tab.render(conn, graph)
with tab_trends:
    trends_tab.render(conn, display_unit)
with tab_organizer:
    organizer_tab.render(conn)
with tab_backfill:
    backfill_tab.render(conn, display_unit)
with tab_dev:
    devtools_tab.render(conn)
