"""Dev Tools tab: direct table editing, DB backup, ingest-batch audit browser.

The editable grid is `st.data_editor(num_rows="dynamic")` over one allowlisted
table; "Apply changes" diffs the edited grid against what was loaded
(`src/ui/editing.py`) and replays the diff through `src/tools/admin.py`. This
is the developer maintenance surface -- it bypasses the agent's HITL flow by
design, because every write here is an explicit hand edit.
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from src.tools.admin import (
    EDITABLE_TABLES,
    AdminError,
    backup_db,
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


def _table_editor(conn) -> None:
    st.subheader("Tables")
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
               "place, or select rows and delete them -- then apply.")

    if original:
        df = pd.DataFrame(original)
    else:
        # Empty table: still offer an editable grid with the right columns.
        df = pd.DataFrame(columns=table_columns(conn, table))

    edited = st.data_editor(
        df, num_rows="dynamic", key=f"editor_{table}", width="stretch", hide_index=True
    )

    if st.button("Apply changes", type="primary"):
        plan = diff_rows(original, edited.to_dict("records"), pk)
        if plan.empty:
            st.info("No changes to apply.")
            return
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
            st.success("; ".join(applied))
            st.rerun()


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
    _table_editor(conn)
    st.divider()
    col1, col2 = st.columns([1, 2])
    with col1:
        _backup_section(conn)
    with col2:
        _audit_section(conn)
