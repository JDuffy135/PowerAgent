"""Program/block organizer tab: thin rendering over `src/tools/organize.py`.

Fixes ingest-time assignment mistakes after the fact: reattach sessions,
rename/merge/move programs and blocks, and start a drafted program. Every
button maps 1:1 to one organize-function call; errors surface inline.
"""
from __future__ import annotations

import datetime as dt

import pandas as pd
import streamlit as st

from src.tools.organize import (
    OrganizeError,
    get_block_review,
    get_program_review,
    list_blocks,
    list_programs,
    list_sessions,
    merge_blocks,
    merge_programs,
    move_block,
    reattach_session,
    rename_block,
    rename_program,
    set_block_review,
    set_program_review,
    start_draft,
)


def _run(action, success: str) -> None:
    try:
        action()
    except OrganizeError as exc:
        st.error(str(exc))
    else:
        st.success(success)
        st.rerun()


def _save_review(save, target: str) -> None:
    """Run a review save (SQLite write + Chroma embed). The prose is committed
    before embedding, so an embed failure (e.g. Ollama down) is a warning, not a
    loss."""
    try:
        save()
    except OrganizeError as exc:
        st.error(str(exc))
    except Exception as exc:  # embedder/Chroma failure -- prose is already saved
        st.warning(f"Saved {target} review to the database, but embedding failed: {exc}")
    else:
        st.success(f"Saved {target} review.")
        st.rerun()


def _program_label(p) -> str:
    return f"#{p.program_id} {p.name} ({p.status})"


def _block_label(b) -> str:
    return f"#{b.block_id} {b.name} — {b.program_name}"


def _review_section(conn, programs, blocks) -> None:
    """Text areas to write a block/program review; saving embeds it for retrieval
    (the coach can cite past reviews when it writes new programs)."""
    st.subheader("Reviews")
    st.caption(
        "Block and macrocycle reviews are saved to the training DB *and* embedded "
        "so the coach can recall them (e.g. when drafting your next block)."
    )
    col1, col2 = st.columns(2)
    with col1:
        if blocks:
            block = st.selectbox("Block", blocks, format_func=_block_label, key="review_block_pick")
            current = get_block_review(conn, block.block_id) or ""
            text = st.text_area("Block review", value=current, key=f"review_block_{block.block_id}",
                                height=160, placeholder="How did this block go? What worked / didn't?")
            if st.button("Save block review"):
                _save_review(lambda: set_block_review(conn, block.block_id, text), "block")
        else:
            st.info("No blocks to review yet.")
    with col2:
        if programs:
            program = st.selectbox("Program", programs, format_func=_program_label, key="review_prog_pick")
            current = get_program_review(conn, program.program_id) or ""
            text = st.text_area("Macrocycle review", value=current, key=f"review_prog_{program.program_id}",
                                height=160, placeholder="Overall review of this program / prep.")
            if st.button("Save program review"):
                _save_review(lambda: set_program_review(conn, program.program_id, text), "program")
        else:
            st.info("No programs to review yet.")


def render(conn) -> None:
    programs = list_programs(conn)
    blocks = list_blocks(conn)

    # ------------------------------------------------------------- programs
    st.subheader("Programs")
    if programs:
        st.dataframe(
            pd.DataFrame([p.model_dump() for p in programs]),
            hide_index=True, width="stretch",
        )
    else:
        st.info("No programs yet.")

    with st.expander("Rename / merge / start a program"):
        if programs:
            col1, col2 = st.columns(2)
            with col1:
                target = st.selectbox("Program", programs, format_func=_program_label, key="prog_rename_pick")
                new_name = st.text_input("New name", value=target.name, key="prog_rename_name")
                if st.button("Rename program"):
                    _run(lambda: rename_program(conn, target.program_id, new_name),
                         f"Renamed program #{target.program_id}.")
            with col2:
                if len(programs) >= 2:
                    src = st.selectbox("Merge this program...", programs, format_func=_program_label, key="prog_merge_src")
                    dst = st.selectbox("...into", [p for p in programs if p.program_id != src.program_id],
                                       format_func=_program_label, key="prog_merge_dst")
                    st.caption("Moves every block over, then deletes the source program row "
                               "(its goals/review prose is dropped).")
                    if st.button("Merge programs"):
                        _run(lambda: merge_programs(conn, src.program_id, dst.program_id),
                             f"Merged #{src.program_id} into #{dst.program_id}.")

        drafts = [p for p in programs if p.status == "draft"]
        if drafts:
            st.divider()
            draft = st.selectbox("Start this draft", drafts, format_func=_program_label, key="draft_pick")
            start = st.date_input("Start date", value=dt.date.today(), key="draft_start")
            st.caption("Flips the draft to `incomplete` so its sessions count in analysis.")
            if st.button("Start draft"):
                _run(lambda: start_draft(conn, draft.program_id, start.isoformat()),
                     f"Started {draft.name}.")

    # --------------------------------------------------------------- blocks
    st.subheader("Blocks")
    if blocks:
        st.dataframe(
            pd.DataFrame([b.model_dump() for b in blocks]),
            hide_index=True, width="stretch",
        )

        with st.expander("Rename / merge / move a block"):
            col1, col2 = st.columns(2)
            with col1:
                target = st.selectbox("Block", blocks, format_func=_block_label, key="block_rename_pick")
                new_name = st.text_input("New name", value=target.name, key="block_rename_name")
                if st.button("Rename block"):
                    _run(lambda: rename_block(conn, target.block_id, new_name),
                         f"Renamed block #{target.block_id}.")

                if programs:
                    dest_prog = st.selectbox("Move it to program", programs,
                                             format_func=_program_label, key="block_move_dst")
                    if st.button("Move block"):
                        _run(lambda: move_block(conn, target.block_id, dest_prog.program_id),
                             f"Moved block #{target.block_id}.")
            with col2:
                if len(blocks) >= 2:
                    src = st.selectbox("Merge this block...", blocks, format_func=_block_label, key="block_merge_src")
                    dst = st.selectbox("...into", [b for b in blocks if b.block_id != src.block_id],
                                       format_func=_block_label, key="block_merge_dst")
                    st.caption("Moves its sessions + programmed slots, then deletes the source block.")
                    if st.button("Merge blocks"):
                        _run(lambda: merge_blocks(conn, src.block_id, dst.block_id),
                             f"Merged #{src.block_id} into #{dst.block_id}.")
    else:
        st.info("No blocks yet.")

    # -------------------------------------------------------------- reviews
    _review_section(conn, programs, blocks)

    # ------------------------------------------------------------- sessions
    st.subheader("Sessions")
    scope_options = ["Unattached only", "All (latest 200)"] + [_block_label(b) for b in blocks]
    scope = st.selectbox("Show", scope_options, key="session_scope")
    if scope == "Unattached only":
        sessions = list_sessions(conn, unattached_only=True)
    elif scope == "All (latest 200)":
        sessions = list_sessions(conn)
    else:
        sessions = list_sessions(conn, blocks[scope_options.index(scope) - 2].block_id)

    if not sessions:
        st.info("No sessions in this scope.")
        return

    st.dataframe(
        pd.DataFrame([s.model_dump() for s in sessions]),
        hide_index=True, width="stretch",
    )

    picked = st.multiselect(
        "Reattach session(s)",
        sessions,
        format_func=lambda s: f"#{s.session_id} {s.date} {s.day_label or ''} ({s.set_count} sets)",
        key="session_reattach_pick",
    )
    dest_labels = ["(unattached)"] + [_block_label(b) for b in blocks]
    dest = st.selectbox("...to block", dest_labels, key="session_reattach_dst")
    if picked and st.button("Reattach"):
        block_id = None if dest == "(unattached)" else blocks[dest_labels.index(dest) - 1].block_id
        def _move():
            for s in picked:
                reattach_session(conn, s.session_id, block_id)
        _run(_move, f"Reattached {len(picked)} session(s).")
