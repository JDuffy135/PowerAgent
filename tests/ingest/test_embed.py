"""Tests for the Chroma prose-embedding seam (Step 3).

All tests use a deterministic fake embedder + an in-memory Chroma client, so no
Ollama server or on-disk store is required (mirroring the extraction tests'
stub-LLM approach).
"""
from __future__ import annotations

import pytest

from src.ingest.embed import (
    PERSONAL_NOTES_COLLECTION,
    SessionNote,
    embed_session_notes,
    get_embedder,
)


def test_embed_writes_notes_with_linking_metadata(fake_embedder, chroma_client):
    notes = [
        SessionNote(
            session_id=42,
            date="2026-07-01",
            raw_note="Squat felt strong today",
            exercises=["Low Bar Squat"],
        )
    ]
    n = embed_session_notes(notes, embedder=fake_embedder, client=chroma_client)
    assert n == 1

    collection = chroma_client.get_collection(PERSONAL_NOTES_COLLECTION)
    got = collection.get(include=["metadatas", "documents"])
    assert got["ids"] == ["session_42"]
    meta = got["metadatas"][0]
    assert meta["session_id"] == 42
    assert meta["date"] == "2026-07-01"
    assert meta["doc_type"] == "session_note"
    assert meta["exercises"] == "Low Bar Squat"
    assert got["documents"][0] == "Squat felt strong today"


def test_blank_notes_are_skipped(fake_embedder, chroma_client):
    notes = [
        SessionNote(session_id=1, date="2026-07-01", raw_note="   ", exercises=[]),
        SessionNote(session_id=2, date="2026-07-01", raw_note="", exercises=[]),
    ]
    assert embed_session_notes(notes, embedder=fake_embedder, client=chroma_client) == 0


def test_empty_input_returns_zero(fake_embedder, chroma_client):
    assert embed_session_notes([], embedder=fake_embedder, client=chroma_client) == 0


def test_upsert_is_idempotent_per_session(fake_embedder, chroma_client):
    note = SessionNote(session_id=7, date="2026-07-01", raw_note="deadlift day", exercises=["Deadlift"])
    embed_session_notes([note], embedder=fake_embedder, client=chroma_client)
    embed_session_notes([note], embedder=fake_embedder, client=chroma_client)

    collection = chroma_client.get_collection(PERSONAL_NOTES_COLLECTION)
    assert collection.count() == 1  # re-embedding overwrote, didn't duplicate


def test_multiple_exercises_joined_in_metadata(fake_embedder, chroma_client):
    note = SessionNote(
        session_id=9,
        date="2026-07-01",
        raw_note="bench + triceps",
        exercises=["Bench Press", "Standing Overhead Tricep Extensions"],
    )
    embed_session_notes([note], embedder=fake_embedder, client=chroma_client)
    meta = chroma_client.get_collection(PERSONAL_NOTES_COLLECTION).get(include=["metadatas"])[
        "metadatas"
    ][0]
    assert meta["exercises"] == "Bench Press, Standing Overhead Tricep Extensions"


def test_get_embedder_non_local_provider_raises(monkeypatch):
    import src.ingest.embed as embed_mod

    monkeypatch.setattr(embed_mod, "_node_config", lambda node: {"provider": "cloud"})
    with pytest.raises(NotImplementedError):
        get_embedder()


def test_get_embedder_returns_callable_for_local(monkeypatch):
    import src.ingest.embed as embed_mod

    monkeypatch.setattr(embed_mod, "_node_config", lambda node: {"provider": "local"})
    assert callable(get_embedder())
