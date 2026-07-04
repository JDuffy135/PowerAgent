"""Dev Tools tab: direct table editing, DB backup, ingest-batch audit browser.

The editable grid is `st.data_editor(num_rows="dynamic")` over one allowlisted
table; "Apply changes" diffs the edited grid against what was loaded
(`src/ui/editing.py`) and replays the diff through `src/tools/admin.py`. This
is the developer maintenance surface -- it bypasses the agent's HITL flow by
design, because every write here is an explicit hand edit.

Two destructive actions get their own confirmation step on top of that: an
"Apply changes" that includes a row delete pauses for an explicit
Confirm/Cancel before it runs, and wiping the whole training DB (the "Danger
zone") requires typing the exact confirmation phrase before the button even
enables.
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from src.tools.admin import (
    CONFIRMATION_PHRASE,
    EDITABLE_TABLES,
    AdminError,
    backup_db,
    clear_all_data,
    count_rows,
    delete_row,
    fetch_table,
    get_batch_json,
    insert_row,
    list_batches,
    table_columns,
    update_row,
)
from src.ui.editing import diff_rows

BACKUPS_DIR = "data/backups"


def _flash(message: str) -> None:
    """Queue a success message to display after the next rerun. `st.success`
    followed by `st.rerun()` would discard the message before it renders."""
    st.session_state["devtools_flash"] = message


def _show_flash() -> None:
    message = st.session_state.pop("devtools_flash", None)
    if message:
        st.success(message)


def _table_editor(conn) -> None:
    st.subheader("Tables")
    st.caption(
        "Weights here are the canonical stored values in **pounds (lb)** — the "
        "`display_unit` kg setting is presentation-only and never applies to this "
        "grid, so edits round-trip without conversion drift."
    )
    table = st.selectbox("Table", sorted(EDITABLE_TABLES), key="dev_table")
    pk = EDITABLE_TABLES[table]
    total = count_rows(conn, table)
    limit = st.number_input(
        "Rows to load (newest first)", min_value=10, max_value=5000,
        value=min(500, max(10, total)), step=50,
    )

    original = fetch_table(conn, table, limit=int(limit))
    st.caption(f"{total} row(s) total; editing the latest {len(original)}. "
               f"Add rows at the bottom (leave `{pk}` blank), edit cells in "
               "place, or select rows and delete them -- then apply. Deletes "
               "ask for a confirm before they run.")

    if original:
        df = pd.DataFrame(original)
    else:
        # Empty table: still offer an editable grid with the right columns.
        df = pd.DataFrame(columns=table_columns(conn, table))

    edited = st.data_editor(
        df, num_rows="dynamic", key=f"editor_{table}", width="stretch", hide_index=True
    )

    pending_key = f"pending_delete_plan_{table}"

    if st.button("Apply changes", type="primary"):
        plan = diff_rows(original, edited.to_dict("records"), pk)
        if plan.empty:
            st.info("No changes to apply.")
        elif plan.deletes:
            # Deletes pause for an explicit confirm; inserts/updates apply right away.
            st.session_state[pending_key] = plan
        else:
            _apply_plan(conn, table, pk, plan)

    pending = st.session_state.get(pending_key)
    if pending:
        st.warning(
            f"This will delete {len(pending.deletes)} row(s) ({pk}="
            f"{pending.deletes}), plus any other pending edits. This cannot be undone."
        )
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Confirm and apply", key=f"confirm_apply_{table}", type="primary"):
                st.session_state.pop(pending_key, None)
                _apply_plan(conn, table, pk, pending)
        with col2:
            if st.button("Cancel", key=f"cancel_apply_{table}"):
                st.session_state.pop(pending_key, None)
                st.rerun()


def _apply_plan(conn, table: str, pk: str, plan) -> None:
    applied, errors = [], []
    for pk_value in plan.deletes:
        try:
            delete_row(conn, table, pk_value)
            applied.append(f"deleted {pk}={pk_value}")
        except AdminError as exc:
            errors.append(str(exc))
    for pk_value, changes in plan.updates:
        try:
            update_row(conn, table, pk_value, changes)
            applied.append(f"updated {pk}={pk_value}")
        except AdminError as exc:
            errors.append(str(exc))
    for values in plan.inserts:
        try:
            new_pk = insert_row(conn, table, values)
            applied.append(f"inserted {pk}={new_pk}")
        except AdminError as exc:
            errors.append(str(exc))
    for err in errors:
        st.error(err)
    if applied:
        if errors:
            # Partial result: show it in place next to the errors, no rerun.
            st.success("; ".join(applied))
        else:
            _flash("; ".join(applied))
            st.rerun()


def _danger_zone(conn) -> None:
    st.subheader("Danger zone")
    st.caption(
        "Wipes every row in every training table -- programs, blocks, sessions, "
        "sets, PRs, bodyweight, injuries, measurements, the ingest audit "
        "trail, and the embedded personal notes (the reference knowledge base "
        "is kept). This cannot be undone; back up first."
    )
    # A widget's session_state key can't be written after the widget exists,
    # so the wipe queues this flag and the input is reset here, pre-widget,
    # on the following run.
    if st.session_state.pop("clear_all_reset", False):
        st.session_state.pop("clear_all_phrase", None)
    phrase = st.text_input(
        f'Type "{CONFIRMATION_PHRASE}" to enable the button below',
        key="clear_all_phrase",
    )
    disabled = phrase.strip().lower() != CONFIRMATION_PHRASE
    if st.button("Clear all training data", type="primary", disabled=disabled):
        # Best-effort Chroma handle: a missing/broken vector store must not
        # block the SQLite wipe (the embedded notes are flagged below instead).
        try:
            from src.ingest.embed import get_chroma_client

            chroma_client = get_chroma_client()
        except Exception:
            chroma_client = None
        try:
            counts = clear_all_data(conn, phrase, chroma_client=chroma_client)
            st.session_state["clear_all_reset"] = True
            message = f"Cleared {sum(counts.values())} row(s) across {len(counts)} table(s)."
            if chroma_client is None:
                message += (
                    " (Embedded personal notes could NOT be cleared -- the vector "
                    "store was unreachable; run the wipe again or clear data/chroma.)"
                )
            _flash(message)
            st.rerun()
        except AdminError as exc:
            st.error(str(exc))


def _backup_section(conn) -> None:
    st.subheader("Backup")
    st.caption("Consistent snapshot of the live DB via SQLite's online backup API.")
    if st.button("Back up training DB"):
        dest = backup_db(conn, BACKUPS_DIR)
        st.success(f"Backed up to `{dest}`.")


def _audit_section(conn) -> None:
    st.subheader("Ingest audit trail")
    status = st.selectbox(
        "Status", ["(all)", "pending_review", "committed", "rejected"], key="audit_status"
    )
    batches = list_batches(conn, status=None if status == "(all)" else status)
    if not batches:
        st.caption("No batches.")
        return
    st.dataframe(
        pd.DataFrame([b.model_dump() for b in batches]),
        hide_index=True, width="stretch",
    )
    batch_id = st.number_input(
        "Inspect batch (parsed JSON)", min_value=0, value=0, step=1, key="audit_batch_id"
    )
    if batch_id:
        try:
            st.code(get_batch_json(conn, int(batch_id)), language="json")
        except AdminError as exc:
            st.error(str(exc))


def render(conn) -> None:
    _show_flash()
    _table_editor(conn)
    st.divider()
    col1, col2 = st.columns([1, 2])
    with col1:
        _backup_section(conn)
    with col2:
        _audit_section(conn)
    st.divider()
    _danger_zone(conn)
