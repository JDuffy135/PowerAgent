"""Stage 1 + Stage 4 query tools wrapped as LangChain tools for the ANALYZE
ReAct loop (ARCHITECTURE.md §5).

`make_analyze_tools(conn, ...)` returns a list of `StructuredTool`s with the DB
connection (and embedder/Chroma client for note search) closed over, so the
LLM-facing schema never exposes `conn`. Design goals for small local models:

- **Tight, few-arg signatures** with explicit `YYYY-MM-DD` date formats and enum
  (`Literal`) choices, so the generated JSON schema guides the model.
- **Every tool returns a plain JSON-able dict** (Pydantic results are
  `model_dump()`-ed here) -- the ReAct loop stores these verbatim as `evidence`.
- **`ExerciseNotFound` is caught and returned as `{"error": ...}`** rather than
  raised, so the model can recover (try another name, ask the user) instead of
  crashing the turn.

The ANALYZE prompt tells the model to prefer the typed tools and reach for
`run_readonly_sql` only when none fit (ARCHITECTURE.md §5.2).
"""
from __future__ import annotations

import sqlite3
from typing import Literal

from langchain_core.tools import StructuredTool, tool

from src.tools import queries as q
from src.tools.queries import ExerciseNotFound
from src.tools.sql import ReadonlySQLError, ReadonlySQLTimeout, run_readonly_sql
from src.tools.vector import search_notes


def _dump(result) -> dict:
    """Normalize a tool result (Pydantic model | list | None) into a JSON-able dict."""
    if result is None:
        return {"result": None}
    if isinstance(result, list):
        return {"result": [r.model_dump() for r in result]}
    return result.model_dump()


def make_analyze_tools(conn: sqlite3.Connection, *, embedder=None, chroma_client=None) -> list[StructuredTool]:
    """Build the ANALYZE toolset bound to `conn` (+ optional note-search seams)."""

    @tool
    def get_best_set(exercise: str, date_from: str, date_to: str, min_reps: int = 1) -> dict:
        """Heaviest single set of an exercise in a date window (dates are YYYY-MM-DD, inclusive).
        Use for "what was my best bench in March?". `min_reps` filters to sets of at least that many reps."""
        try:
            return _dump(q.get_best_set(conn, exercise, date_from, date_to, min_reps))
        except ExerciseNotFound as exc:
            return {"error": str(exc)}

    @tool
    def get_lifts(exercise: str, date_from: str, date_to: str, top_sets_only: bool = False) -> dict:
        """Every logged set of an exercise in a window (YYYY-MM-DD), grouped by session.
        Set `top_sets_only=true` to see only heavy top singles/doubles, not backoff work."""
        try:
            return _dump(q.get_lifts(conn, exercise, date_from, date_to, top_sets_only))
        except ExerciseNotFound as exc:
            return {"error": str(exc)}

    @tool
    def get_e1rm_trend(exercise: str, date_from: str, date_to: str, by: Literal["week", "block"] = "week") -> dict:
        """Estimated-1RM (Epley) trend for an exercise over a window (YYYY-MM-DD), bucketed by 'week' or 'block'.
        Each point cites the source set so the estimate is auditable. Use for "how has my squat e1RM trended?"."""
        try:
            return _dump(q.get_e1rm_trend(conn, exercise, date_from, date_to, by))
        except ExerciseNotFound as exc:
            return {"error": str(exc)}

    @tool
    def get_volume_trend(exercise_or_muscle_group: str, date_from: str, date_to: str, by: Literal["week", "block"] = "week") -> dict:
        """Hard-set count + tonnage per bucket for one exercise OR a whole muscle group (e.g. 'quads', 'chest').
        Window is YYYY-MM-DD; bucket by 'week' or 'block'."""
        try:
            return _dump(q.get_volume_trend(conn, exercise_or_muscle_group, date_from, date_to, by))
        except ExerciseNotFound as exc:
            return {"error": str(exc)}

    @tool
    def get_frequency(exercise: str, date_from: str, date_to: str, by: Literal["week", "block"] = "week") -> dict:
        """How many distinct sessions an exercise was trained in per bucket (YYYY-MM-DD window; 'week' or 'block')."""
        try:
            return _dump(q.get_frequency(conn, exercise, date_from, date_to, by))
        except ExerciseNotFound as exc:
            return {"error": str(exc)}

    @tool
    def get_bodyweight_trend(date_from: str, date_to: str) -> dict:
        """Bodyweight entries in a window (YYYY-MM-DD) with first/last/delta/min/max summary."""
        return _dump(q.get_bodyweight_trend(conn, date_from, date_to))

    @tool
    def get_sessions(date_from: str, date_to: str, session_type: Literal["lifting", "cardio", "other"] | None = None) -> dict:
        """All sessions in a window (YYYY-MM-DD), chronological. Optionally filter by session_type."""
        return _dump(q.get_sessions(conn, date_from, date_to, session_type=session_type))

    @tool
    def get_prs(exercise: str | None = None, date_from: str | None = None, date_to: str | None = None) -> dict:
        """Recorded personal records. Optionally scope to one exercise and/or a date window (YYYY-MM-DD)."""
        try:
            return _dump(q.get_prs(conn, exercise, date_from, date_to))
        except ExerciseNotFound as exc:
            return {"error": str(exc)}

    @tool
    def get_injuries(active_only: bool = False, area: str | None = None) -> dict:
        """Logged injuries. `active_only=true` returns only ongoing ones; `area` filters by body area (e.g. 'right knee')."""
        return _dump(q.get_injuries(conn, active_only, area))

    @tool
    def get_measurements(site: str | None = None, date_from: str | None = None, date_to: str | None = None) -> dict:
        """Body measurements (inches). Optionally filter by `site` (e.g. 'arm') and/or a date window (YYYY-MM-DD)."""
        return _dump(q.get_measurements(conn, site, date_from, date_to))

    @tool
    def get_programs(status: Literal["complete", "incomplete", "draft"] | None = None) -> dict:
        """List training programs. Only tool that can surface `draft` programs; pass a `status` to filter."""
        return _dump(q.get_programs(conn, status))

    @tool
    def get_block_outline(block_id: int) -> dict:
        """The programmed (planned) slots for a block id, in program order."""
        return _dump(q.get_block_outline(conn, block_id))

    @tool
    def compare_programmed_vs_actual(block_id: int, exercise: str | None = None) -> dict:
        """Compare planned vs performed work for a block id. Reports programmed-but-skipped and
        performed-but-unplanned work explicitly. Optionally scope to one exercise."""
        try:
            return _dump(q.compare_programmed_vs_actual(conn, block_id, exercise))
        except ExerciseNotFound as exc:
            return {"error": str(exc)}

    @tool
    def search_training_notes(
        query: str,
        date_from: str | None = None,
        date_to: str | None = None,
        doc_type: str | None = None,
    ) -> dict:
        """Semantic search over personal session notes / block reviews for a topic (e.g. 'knee pain').
        REQUIRES at least one scope: a date window (YYYY-MM-DD) and/or a doc_type. Use for prose, not numbers."""
        try:
            results = search_notes(
                query,
                date_from=date_from,
                date_to=date_to,
                doc_type=doc_type,
                embedder=embedder,
                client=chroma_client,
            )
        except ValueError as exc:
            return {"error": str(exc)}
        return {"result": [r.model_dump() for r in results]}

    @tool
    def run_sql(query: str) -> dict:
        """ESCAPE HATCH: run one read-only SELECT when no typed tool fits. Prefer the typed tools.
        Rejects anything that isn't a single SELECT. Weights are stored in lb."""
        try:
            return run_readonly_sql(conn, query).model_dump()
        except (ReadonlySQLError, ReadonlySQLTimeout) as exc:
            return {"error": str(exc)}

    return [
        get_best_set,
        get_lifts,
        get_e1rm_trend,
        get_volume_trend,
        get_frequency,
        get_bodyweight_trend,
        get_sessions,
        get_prs,
        get_injuries,
        get_measurements,
        get_programs,
        get_block_outline,
        compare_programmed_vs_actual,
        search_training_notes,
        run_sql,
    ]
