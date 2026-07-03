"""Historical backfill tab: bulk-ingest years of training history at once.

Front-end for `src/ingest/backfill.py`. Two modes, per the relaxed-bulk-HITL
decision:

- **Stage for review** (default): every chunk lands as a `pending_review`
  batch; the "Pending batches" section below renders each one for per-batch
  approve/reject -- the normal HITL contract, amortized.
- **Commit everything**: the run button itself is the one explicit approval for
  the whole archive; every chunk is still audited in `ingest_batch`.
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from src.ingest.backfill import DEFAULT_MAX_CHARS, run_backfill, split_archive
from src.ingest.commit import commit_batch, reject_batch
from src.ingest.loaders import parse_upload
from src.ingest.review import render_batch
from src.ingest.stage import get_pending_batch
from src.tools.admin import list_batches
from src.tools.organize import list_blocks
from src.ui.chat_tab import _save_upload


def _archive_text() -> str:
    """Paste box or file upload (txt/xlsx/pdf all go through parse_upload)."""
    pasted = st.text_area(
        "Paste training history",
        height=220,
        placeholder="2023-01-05 w1d1\nsquat 315x5x3\nbench 225x5x3\n\n2023-01-07 w1d2\n...",
    )
    uploaded = st.file_uploader(
        "...or upload an archive file", type=["txt", "xlsx", "pdf"], key="backfill_upload"
    )
    if uploaded is not None:
        try:
            return parse_upload(_save_upload(uploaded))
        except Exception as exc:
            st.error(f"Could not read {uploaded.name}: {exc}")
            return ""
    return pasted


def _run_section(conn) -> None:
    text = _archive_text()

    col1, col2, col3 = st.columns(3)
    with col1:
        mode = st.radio(
            "Mode",
            ["Stage for review", "Commit everything"],
            help="Stage: each chunk waits for per-batch approval below. "
                 "Commit: this button is the one approval for the whole archive.",
        )
    with col2:
        blocks = list_blocks(conn)
        block_labels = ["(leave unattached)"] + [
            f"#{b.block_id} {b.name} — {b.program_name}" for b in blocks
        ]
        block_pick = st.selectbox(
            "Attach committed sessions to",
            block_labels,
            disabled=mode != "Commit everything",
        )
    with col3:
        max_chars = st.number_input(
            "Chunk size (chars)", min_value=1000, max_value=30000,
            value=DEFAULT_MAX_CHARS, step=1000,
            help="~4 chars/token of log text per extraction call.",
        )
        embed = st.checkbox("Embed session prose (Chroma)", value=True)

    n_chunks = len(split_archive(text, max_chars=int(max_chars))) if text.strip() else 0
    st.caption(f"{len(text)} chars -> {n_chunks} extraction chunk(s).")

    if st.button("Run backfill", type="primary", disabled=n_chunks == 0):
        auto_commit = mode == "Commit everything"
        block_id = (
            blocks[block_labels.index(block_pick) - 1].block_id
            if auto_commit and block_pick != "(leave unattached)"
            else None
        )
        bar = st.progress(0.0, text="extracting...")
        results = run_backfill(
            conn,
            text,
            source="backfill",
            auto_commit=auto_commit,
            block_id=block_id,
            embed_prose=embed,
            max_chars=int(max_chars),
            progress=lambda done, total, r: bar.progress(
                done / total, text=f"chunk {done}/{total}: {r.status}"
            ),
        )
        bar.empty()
        st.dataframe(
            pd.DataFrame([r.model_dump() for r in results]),
            hide_index=True, width="stretch",
        )
        failed = [r for r in results if r.status == "failed"]
        if failed:
            st.warning(f"{len(failed)} chunk(s) failed -- see errors above; the rest went through.")
        else:
            st.success(f"All {len(results)} chunk(s) {'committed' if auto_commit else 'staged'}.")


def _pending_section(conn, display_unit: str = "lb") -> None:
    pending = list_batches(conn, status="pending_review")
    st.subheader(f"Pending batches ({len(pending)})")
    if not pending:
        st.caption("Nothing awaiting review.")
        return

    blocks = list_blocks(conn)
    block_labels = ["(leave unattached)"] + [
        f"#{b.block_id} {b.name} — {b.program_name}" for b in blocks
    ]
    for batch in pending:
        header = (
            f"batch #{batch.batch_id} — {batch.source_file or '(no source)'} — "
            f"{batch.n_sessions} session(s), {batch.n_sets} set(s)"
        )
        with st.expander(header):
            try:
                st.text(render_batch(get_pending_batch(conn, batch.batch_id), display_unit))
            except Exception as exc:
                st.error(f"Could not render batch: {exc}")
                continue
            pick = st.selectbox("Attach to", block_labels, key=f"batch_block_{batch.batch_id}")
            col_ok, col_no, _ = st.columns([1, 1, 4])
            if col_ok.button("✅ Commit", key=f"commit_{batch.batch_id}"):
                block_id = (
                    None if pick == "(leave unattached)"
                    else blocks[block_labels.index(pick) - 1].block_id
                )
                try:
                    result = commit_batch(conn, batch.batch_id, block_id=block_id)
                except Exception as exc:
                    st.error(f"Commit failed: {exc}")
                else:
                    st.success(f"Committed: {result.sessions_created} session(s), "
                               f"{result.sets_created} set(s).")
                    st.rerun()
            if col_no.button("❌ Reject", key=f"reject_{batch.batch_id}"):
                reject_batch(conn, batch.batch_id)
                st.rerun()


def render(conn, display_unit: str = "lb") -> None:
    st.subheader("Backfill an archive")
    _run_section(conn)
    st.divider()
    _pending_section(conn, display_unit)
