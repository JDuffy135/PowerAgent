"""Historical backfill: bulk-ingest a big chunk of training history at once
(Stage 9).

The normal `/ingest` flow is one file -> one batch -> one HITL review. That's
the right friction for a weekly log, and the wrong friction for pasting three
years of archive. Backfill splits the archive into extraction-sized chunks on
session boundaries and runs each chunk through the SAME pipeline
(`extract_training_data` -> `stage_batch` -> `commit_batch`), so every chunk
still leaves an `ingest_batch` audit row.

**[DECISION] Relaxed-confirmation bulk mode:** with `auto_commit=True` the one
explicit user action that starts the run ("run and commit all") stands in for
per-batch approval -- the HITL invariant's *approval* happens once, up front,
instead of per chunk. With `auto_commit=False` every chunk is merely staged
`pending_review`; the UI then offers per-batch render/commit/reject.

A chunk that fails extraction is recorded and skipped -- one garbled month
never aborts the run.
"""
from __future__ import annotations

import re
import sqlite3
from typing import Callable

from pydantic import BaseModel

from src.ingest.commit import commit_batch
from src.ingest.extract import extract_training_data
from src.ingest.stage import stage_batch

# ~6k chars ≈ 1.5k tokens of log text per extraction call: big enough to keep
# multi-session context, small enough for the local extraction model.
DEFAULT_MAX_CHARS = 6000

# Lines that *start* a new session entry in typical training archives:
# ISO dates, US dates, or week/day labels ("w3d2", "Week 3 Day 2").
_SESSION_START = re.compile(
    r"^\s*("
    r"\d{4}-\d{2}-\d{2}"                  # 2026-03-14
    r"|\d{1,2}/\d{1,2}(/\d{2,4})?"        # 3/14, 3/14/26
    r"|w(eek\s*)?\d+\s*d(ay\s*)?\d+"      # w3d2, week 3 day 2
    r")\b",
    re.IGNORECASE,
)


def _session_blocks(text: str) -> list[str]:
    """Split archive text into per-session blocks: a new block starts at a
    blank-line gap or at a date/week-label line. Never splits mid-session."""
    blocks: list[str] = []
    current: list[str] = []

    def flush() -> None:
        block = "\n".join(current).strip()
        if block:
            blocks.append(block)
        current.clear()

    blank_run = False
    for line in text.splitlines():
        if not line.strip():
            blank_run = True
            current.append(line)
            continue
        if _SESSION_START.match(line) and any(l.strip() for l in current):
            flush()
        elif blank_run and _looks_like_header(line) and any(l.strip() for l in current):
            flush()
        blank_run = False
        current.append(line)
    flush()
    return blocks


def _looks_like_header(line: str) -> bool:
    """After a blank-line gap, a short line (day label, 'CARDIO', a heading)
    plausibly starts a new session even without a recognizable date."""
    return len(line.strip()) <= 60


def split_archive(text: str, max_chars: int = DEFAULT_MAX_CHARS) -> list[str]:
    """Split a big archive into extraction-sized chunks along session
    boundaries. Sessions are greedily packed up to `max_chars`; a single
    session longer than `max_chars` is hard-split at line boundaries (rare --
    it means one workout's prose alone exceeds the window)."""
    if not text.strip():
        return []

    pieces: list[str] = []
    for block in _session_blocks(text):
        if len(block) <= max_chars:
            pieces.append(block)
            continue
        # Oversized single session: hard-split at line boundaries.
        lines, buf, size = block.splitlines(), [], 0
        for line in lines:
            if buf and size + len(line) + 1 > max_chars:
                pieces.append("\n".join(buf))
                buf, size = [], 0
            buf.append(line)
            size += len(line) + 1
        if buf:
            pieces.append("\n".join(buf))

    chunks: list[str] = []
    buf, size = [], 0
    for piece in pieces:
        if buf and size + len(piece) + 2 > max_chars:
            chunks.append("\n\n".join(buf))
            buf, size = [], 0
        buf.append(piece)
        size += len(piece) + 2
    if buf:
        chunks.append("\n\n".join(buf))
    return chunks


class ChunkResult(BaseModel):
    index: int                      # 1-based chunk number
    status: str                     # 'committed' | 'staged' | 'failed'
    batch_id: int | None = None     # None only when extraction itself failed
    sessions: int = 0
    sets: int = 0
    error: str | None = None


def run_backfill(
    conn: sqlite3.Connection,
    text: str,
    *,
    source: str = "backfill",
    llm=None,
    auto_commit: bool = False,
    block_id: int | None = None,
    embedder=None,
    chroma_client=None,
    embed_prose: bool = True,
    max_chars: int = DEFAULT_MAX_CHARS,
    progress: Callable[[int, int, ChunkResult], None] | None = None,
) -> list[ChunkResult]:
    """Run the whole archive through extract -> stage (-> commit) per chunk.

    `auto_commit=False` stages every chunk `pending_review` for per-batch
    review; `auto_commit=True` is the relaxed bulk mode (see module docstring).
    `block_id` (bulk mode only) attaches every committed session to that block.
    `progress(done, total, result)` fires after each chunk for UI progress bars.
    Extraction/commit failures are captured per chunk, never raised.
    """
    chunks = split_archive(text, max_chars=max_chars)
    results: list[ChunkResult] = []

    for i, chunk in enumerate(chunks, start=1):
        batch_id: int | None = None
        try:
            parsed = extract_training_data(chunk, conn, llm)
            batch_id = stage_batch(conn, parsed, source_file=f"{source} [chunk {i}/{len(chunks)}]")
            result = ChunkResult(
                index=i,
                status="staged",
                batch_id=batch_id,
                sessions=len(parsed.sessions),
                sets=sum(len(s.sets) for s in parsed.sessions),
            )
            if auto_commit:
                commit = commit_batch(
                    conn,
                    batch_id,
                    block_id=block_id,
                    embedder=embedder,
                    chroma_client=chroma_client,
                    embed_prose=embed_prose,
                )
                result.status = "committed"
                result.sessions = commit.sessions_created
                result.sets = commit.sets_created
        except Exception as exc:  # isolate the bad chunk, keep going
            # batch_id is non-None here iff staging succeeded but the commit
            # failed -- the batch stays pending_review for manual follow-up.
            result = ChunkResult(
                index=i,
                status="failed",
                batch_id=batch_id,
                error=f"{type(exc).__name__}: {exc}",
            )
        results.append(result)
        if progress is not None:
            progress(i, len(chunks), result)

    return results
