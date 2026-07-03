"""Historical backfill: archive splitter + bulk stage/commit (Stage 9).

No live models: the extraction LLM is a stub emitting one minimal session per
call (echoing which chunk it saw), same seam as the extract tests.
"""
from __future__ import annotations

import json

from src.ingest.backfill import ChunkResult, run_backfill, split_archive

ARCHIVE = """\
2026-01-05 w1d1
squat 315x5x3
bench 225x5x3

2026-01-07 w1d2
deadlift 405x3
rows 135x10x3

2026-01-09 w1d3
bench 230x4x4
"""


# ---------------------------------------------------------------------------
# split_archive
# ---------------------------------------------------------------------------

def test_split_empty_text():
    assert split_archive("") == []
    assert split_archive("   \n\n  ") == []


def test_split_small_archive_is_one_chunk():
    chunks = split_archive(ARCHIVE)
    assert len(chunks) == 1
    assert "w1d1" in chunks[0] and "w1d3" in chunks[0]


def test_split_respects_session_boundaries():
    # Force tiny chunks: each session block must stay intact.
    chunks = split_archive(ARCHIVE, max_chars=60)
    assert len(chunks) == 3
    assert chunks[0].startswith("2026-01-05")
    assert chunks[1].startswith("2026-01-07")
    assert chunks[2].startswith("2026-01-09")
    # No session got split in half.
    assert "squat" in chunks[0] and "bench 225" in chunks[0]
    assert "deadlift" in chunks[1] and "rows" in chunks[1]


def test_split_date_line_starts_new_session_without_blank_gap():
    text = "2026-02-01\nsquat 300x5\n2026-02-03\nbench 200x5"
    chunks = split_archive(text, max_chars=30)
    assert len(chunks) == 2
    assert chunks[0].startswith("2026-02-01")
    assert chunks[1].startswith("2026-02-03")


def test_split_oversized_single_session_hard_splits():
    text = "2026-03-01\n" + "\n".join(f"accessory line {i} 100x10" for i in range(100))
    chunks = split_archive(text, max_chars=400)
    assert len(chunks) > 1
    assert sum(c.count("accessory line") for c in chunks) == 100


def test_split_coverage_no_lost_lines():
    chunks = split_archive(ARCHIVE, max_chars=80)
    rejoined = "\n".join(chunks)
    for line in ARCHIVE.strip().splitlines():
        if line.strip():
            assert line in rejoined


# ---------------------------------------------------------------------------
# run_backfill
# ---------------------------------------------------------------------------

def _stub_extract_llm(fail_on: set[str] = frozenset()):
    """Extraction stub: one session per chunk whose date comes from the chunk's
    first line; raises for chunks containing any `fail_on` marker."""
    def _call(prompt: str) -> str:
        for marker in fail_on:
            if marker in prompt:
                raise RuntimeError(f"stub refused chunk containing {marker!r}")
        date = prompt.strip().splitlines()[0].split()[0]
        return json.dumps({
            "sessions": [{
                "date": date,
                "raw_note": prompt.strip(),
                "sets": [{
                    "exercise_raw": "bench press",  # resolves via seeded alias
                    "set_index": 1,
                    "weight_lb": 225.0,
                    "reps": 5,
                    "raw_text": "bench 225x5",
                }],
            }],
            "new_exercise_candidates": [],
        })
    return _call


def test_backfill_stage_only(conn):
    results = run_backfill(conn, ARCHIVE, llm=_stub_extract_llm(), max_chars=60)
    assert [r.status for r in results] == ["staged"] * 3
    assert all(r.batch_id is not None for r in results)
    statuses = [
        conn.execute(
            "SELECT status, source_file FROM ingest_batch WHERE batch_id = ?",
            (r.batch_id,),
        ).fetchone()
        for r in results
    ]
    assert all(s["status"] == "pending_review" for s in statuses)
    assert statuses[0]["source_file"] == "backfill [chunk 1/3]"


def test_backfill_auto_commit_writes_sessions(conn):
    before = conn.execute("SELECT COUNT(*) FROM session").fetchone()[0]
    seen = []
    results = run_backfill(
        conn,
        ARCHIVE,
        llm=_stub_extract_llm(),
        auto_commit=True,
        embed_prose=False,
        max_chars=60,
        progress=lambda done, total, r: seen.append((done, total, r.status)),
    )
    assert [r.status for r in results] == ["committed"] * 3
    assert sum(r.sessions for r in results) == 3
    after = conn.execute("SELECT COUNT(*) FROM session").fetchone()[0]
    assert after == before + 3
    assert seen == [(1, 3, "committed"), (2, 3, "committed"), (3, 3, "committed")]


def test_backfill_auto_commit_attaches_block(conn):
    block_id = conn.execute("SELECT block_id FROM block LIMIT 1").fetchone()[0]
    results = run_backfill(
        conn, ARCHIVE, llm=_stub_extract_llm(), auto_commit=True,
        embed_prose=False, block_id=block_id, max_chars=60,
    )
    dates = ("2026-01-05", "2026-01-07", "2026-01-09")
    rows = conn.execute(
        f"SELECT block_id FROM session WHERE date IN {dates}"
    ).fetchall()
    assert len(rows) == 3 and all(r[0] == block_id for r in rows)
    assert all(r.status == "committed" for r in results)


def test_backfill_failed_chunk_is_isolated(conn):
    results = run_backfill(
        conn,
        ARCHIVE,
        llm=_stub_extract_llm(fail_on={"deadlift"}),
        auto_commit=True,
        embed_prose=False,
        max_chars=60,
    )
    assert [r.status for r in results] == ["committed", "failed", "committed"]
    failed = results[1]
    assert failed.batch_id is None  # extraction itself failed -> nothing staged
    assert "stub refused" in failed.error
    assert isinstance(failed, ChunkResult)


def test_backfill_empty_text_is_noop(conn):
    assert run_backfill(conn, "", llm=_stub_extract_llm()) == []
