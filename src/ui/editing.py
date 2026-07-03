"""Diffing for the Dev Tools data editor (streamlit-free, unit-testable).

`st.data_editor(num_rows="dynamic")` hands back the whole edited table; this
module turns (original rows, edited rows) into explicit insert/update/delete
ops for `src/tools/admin.py`. Rows are plain dicts keyed by column name.

Conventions:
- a row whose pk is None/NaN/"" is an **insert** (the editor's added rows have
  no pk yet);
- a pk present in original but absent from edited is a **delete**;
- a pk present in both with any changed cell is an **update** carrying only the
  changed columns.
"""
from __future__ import annotations

import math
from typing import Any

from pydantic import BaseModel, Field


class EditPlan(BaseModel):
    inserts: list[dict] = Field(default_factory=list)
    updates: list[tuple[Any, dict]] = Field(default_factory=list)  # (pk_value, changed cols)
    deletes: list[Any] = Field(default_factory=list)               # pk values

    @property
    def empty(self) -> bool:
        return not (self.inserts or self.updates or self.deletes)


def _is_missing(value) -> bool:
    if value is None or value == "":
        return True
    return isinstance(value, float) and math.isnan(value)


def _clean(row: dict) -> dict:
    """pandas hands NaN for empty cells; normalize those to None so SQLite gets
    NULL, not the float nan."""
    return {k: (None if _is_missing(v) else v) for k, v in row.items()}


def diff_rows(original: list[dict], edited: list[dict], pk: str) -> EditPlan:
    """Compute the admin ops that turn `original` into `edited` (see module
    docstring for the insert/update/delete conventions)."""
    plan = EditPlan()
    original_by_pk = {row[pk]: _clean(row) for row in original}
    seen: set = set()

    for raw in edited:
        row = _clean(raw)
        pk_value = row.get(pk)
        if _is_missing(pk_value):
            values = {k: v for k, v in row.items() if k != pk and v is not None}
            if values:  # a fully-blank added row is noise, not an insert
                plan.inserts.append(values)
            continue
        seen.add(pk_value)
        before = original_by_pk.get(pk_value)
        if before is None:
            # pk typed by hand for a new row: treat as insert with explicit pk
            plan.inserts.append(row)
            continue
        changed = {k: v for k, v in row.items() if k != pk and before.get(k) != v}
        if changed:
            plan.updates.append((pk_value, changed))

    plan.deletes = [pk_value for pk_value in original_by_pk if pk_value not in seen]
    return plan
