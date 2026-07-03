"""Program/block organizer ops (Stage 9) -- pure SQL, seeded in-memory DB."""
from __future__ import annotations

import pytest

from src.tools.organize import (
    OrganizeError,
    list_blocks,
    list_programs,
    list_sessions,
    merge_blocks,
    merge_programs,
    move_block,
    reattach_session,
    rename_block,
    rename_program,
    start_draft,
)


def _program_id(conn) -> int:
    return conn.execute("SELECT program_id FROM program LIMIT 1").fetchone()[0]


def _block_ids(conn) -> list[int]:
    return [r[0] for r in conn.execute("SELECT block_id FROM block ORDER BY block_id")]


# ---------------------------------------------------------------------------
# Listings
# ---------------------------------------------------------------------------

def test_list_programs_counts(conn):
    programs = list_programs(conn)
    assert len(programs) >= 1
    prep = next(p for p in programs if p.name == "2026 Meet 1 Prep")
    assert prep.block_count == 3
    assert prep.session_count > 0


def test_list_blocks_scoped_to_program(conn):
    program_id = _program_id(conn)
    blocks = list_blocks(conn, program_id)
    assert {b.name for b in blocks} >= {"Hypertrophy Phase 1", "Strength Block 1"}
    assert all(b.program_id == program_id for b in blocks)


def test_list_sessions_by_block_and_unattached(conn):
    blocks = list_blocks(conn)
    with_sessions = next(b for b in blocks if b.session_count > 0)
    sessions = list_sessions(conn, with_sessions.block_id)
    assert sessions and all(s.block_id == with_sessions.block_id for s in sessions)

    conn.execute("INSERT INTO session (date, session_type) VALUES ('2026-07-01', 'lifting')")
    conn.commit()
    unattached = list_sessions(conn, unattached_only=True)
    assert any(s.date == "2026-07-01" and s.block_id is None for s in unattached)


# ---------------------------------------------------------------------------
# Renames / reattach / move
# ---------------------------------------------------------------------------

def test_rename_program_and_block(conn):
    program_id = _program_id(conn)
    rename_program(conn, program_id, "  Meet 1 Prep (renamed)  ")
    assert conn.execute(
        "SELECT name FROM program WHERE program_id = ?", (program_id,)
    ).fetchone()[0] == "Meet 1 Prep (renamed)"

    block_id = _block_ids(conn)[0]
    rename_block(conn, block_id, "Hypertrophy A")
    assert conn.execute(
        "SELECT name FROM block WHERE block_id = ?", (block_id,)
    ).fetchone()[0] == "Hypertrophy A"


def test_rename_rejects_empty_and_unknown(conn):
    with pytest.raises(OrganizeError):
        rename_program(conn, _program_id(conn), "   ")
    with pytest.raises(OrganizeError):
        rename_block(conn, 99999, "x")


def test_reattach_session_moves_only_the_fk(conn):
    session = list_sessions(conn)[0]
    target = next(b for b in list_blocks(conn) if b.block_id != session.block_id)
    sets_before = conn.execute(
        "SELECT COUNT(*) FROM lift_set WHERE session_id = ?", (session.session_id,)
    ).fetchone()[0]

    reattach_session(conn, session.session_id, target.block_id)
    row = conn.execute(
        "SELECT block_id FROM session WHERE session_id = ?", (session.session_id,)
    ).fetchone()
    assert row[0] == target.block_id
    sets_after = conn.execute(
        "SELECT COUNT(*) FROM lift_set WHERE session_id = ?", (session.session_id,)
    ).fetchone()[0]
    assert sets_after == sets_before

    reattach_session(conn, session.session_id, None)  # detach
    assert conn.execute(
        "SELECT block_id FROM session WHERE session_id = ?", (session.session_id,)
    ).fetchone()[0] is None


def test_reattach_session_unknown_block(conn):
    session = list_sessions(conn)[0]
    with pytest.raises(OrganizeError):
        reattach_session(conn, session.session_id, 99999)


def test_move_block_to_other_program(conn):
    other = conn.execute(
        "INSERT INTO program (name, status) VALUES ('Off-season', 'incomplete')"
    ).lastrowid
    block_id = _block_ids(conn)[0]
    move_block(conn, block_id, other)
    assert conn.execute(
        "SELECT program_id FROM block WHERE block_id = ?", (block_id,)
    ).fetchone()[0] == other


# ---------------------------------------------------------------------------
# Merges
# ---------------------------------------------------------------------------

def test_merge_blocks_moves_sessions_and_slots(conn):
    blocks = list_blocks(conn)
    src = next(b for b in blocks if b.session_count > 0)
    dst = next(b for b in blocks if b.block_id != src.block_id)
    src_sessions = src.session_count
    slot_count = conn.execute(
        "SELECT COUNT(*) FROM programmed_slot WHERE block_id = ?", (src.block_id,)
    ).fetchone()[0]

    moved = merge_blocks(conn, src.block_id, dst.block_id)
    assert moved == src_sessions
    assert conn.execute(
        "SELECT COUNT(*) FROM block WHERE block_id = ?", (src.block_id,)
    ).fetchone()[0] == 0
    assert conn.execute(
        "SELECT COUNT(*) FROM session WHERE block_id = ?", (src.block_id,)
    ).fetchone()[0] == 0
    if slot_count:
        assert conn.execute(
            "SELECT COUNT(*) FROM programmed_slot WHERE block_id = ?", (dst.block_id,)
        ).fetchone()[0] >= slot_count


def test_merge_blocks_rejects_self_merge(conn):
    block_id = _block_ids(conn)[0]
    with pytest.raises(OrganizeError):
        merge_blocks(conn, block_id, block_id)


def test_merge_programs_moves_blocks(conn):
    src = _program_id(conn)
    dst = conn.execute(
        "INSERT INTO program (name, status) VALUES ('Archive', 'complete')"
    ).lastrowid
    n_blocks = len(list_blocks(conn, src))

    moved = merge_programs(conn, src, dst)
    assert moved == n_blocks
    assert conn.execute(
        "SELECT COUNT(*) FROM program WHERE program_id = ?", (src,)
    ).fetchone()[0] == 0
    assert len(list_blocks(conn, dst)) == n_blocks


# ---------------------------------------------------------------------------
# Start-a-draft flow
# ---------------------------------------------------------------------------

def test_start_draft_flips_status_and_stamps_date(conn):
    draft = conn.execute(
        "INSERT INTO program (name, status) VALUES ('Draft Strength Cycle', 'draft')"
    ).lastrowid
    start_draft(conn, draft, "2026-07-06")
    row = conn.execute(
        "SELECT status, start_date FROM program WHERE program_id = ?", (draft,)
    ).fetchone()
    assert row["status"] == "incomplete"
    assert row["start_date"] == "2026-07-06"


def test_start_draft_rejects_non_draft(conn):
    with pytest.raises(OrganizeError):
        start_draft(conn, _program_id(conn), "2026-07-06")
