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


# ---------------------------------------------------------------------------
# embed_review (Stage 11b) -- block/program reviews + form cues
# ---------------------------------------------------------------------------

def test_embed_review_writes_block_review(fake_embedder, chroma_client):
    from src.ingest.embed import BLOCK_REVIEW_DOC_TYPE, block_review_id, embed_review

    doc_id = embed_review(
        "Great strength block; deadlift moved well.",
        block_review_id(5),
        BLOCK_REVIEW_DOC_TYPE,
        date="2026-06-30",
        block_id=5,
        program_id=2,
        embedder=fake_embedder,
        client=chroma_client,
    )
    assert doc_id == "block_review_5"
    collection = chroma_client.get_collection(PERSONAL_NOTES_COLLECTION)
    got = collection.get(ids=["block_review_5"], include=["metadatas", "documents"])
    meta = got["metadatas"][0]
    assert meta["doc_type"] == "block_review"
    assert meta["block_id"] == 5 and meta["program_id"] == 2
    assert meta["date"] == "2026-06-30" and meta["session_id"] == 0


def test_embed_review_upsert_is_idempotent(fake_embedder, chroma_client):
    from src.ingest.embed import BLOCK_REVIEW_DOC_TYPE, block_review_id, embed_review

    for text in ("first draft", "edited review"):
        embed_review(text, block_review_id(9), BLOCK_REVIEW_DOC_TYPE,
                     embedder=fake_embedder, client=chroma_client)
    collection = chroma_client.get_collection(PERSONAL_NOTES_COLLECTION)
    assert collection.count() == 1
    got = collection.get(ids=["block_review_9"], include=["documents"])
    assert got["documents"][0] == "edited review"


def test_embed_review_blank_deletes(fake_embedder, chroma_client):
    from src.ingest.embed import BLOCK_REVIEW_DOC_TYPE, block_review_id, embed_review

    embed_review("something", block_review_id(1), BLOCK_REVIEW_DOC_TYPE,
                 embedder=fake_embedder, client=chroma_client)
    result = embed_review("   ", block_review_id(1), BLOCK_REVIEW_DOC_TYPE,
                          embedder=fake_embedder, client=chroma_client)
    assert result is None
    collection = chroma_client.get_collection(PERSONAL_NOTES_COLLECTION)
    assert collection.count() == 0


def test_embed_review_form_cue_carries_exercises(fake_embedder, chroma_client):
    from src.ingest.embed import FORM_CUE_DOC_TYPE, embed_review

    embed_review(
        "spread the floor, knees out",
        "form_cue_2_123",
        FORM_CUE_DOC_TYPE,
        exercises=["Low Bar Squat"],
        embedder=fake_embedder,
        client=chroma_client,
    )
    collection = chroma_client.get_collection(PERSONAL_NOTES_COLLECTION)
    meta = collection.get(ids=["form_cue_2_123"], include=["metadatas"])["metadatas"][0]
    assert meta["doc_type"] == "form_cue"
    assert meta["exercises"] == "Low Bar Squat"
