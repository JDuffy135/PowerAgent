"""Tests for knowledge-base ingestion (Stage 8).

All tests use the deterministic fake embedder + in-memory Chroma client (from
`tests/conftest.py`) and a stub metadata-LLM, so no Ollama/Chroma/on-disk store
is required -- same house style as the extraction/embed tests.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from src.ingest.knowledge import (
    DEFAULT_CHUNK_CHARS,
    KNOWLEDGE_COLLECTION,
    KnowledgeDoc,
    chunk_text,
    guess_metadata,
    ingest_knowledge,
    ingest_knowledge_file,
    resolve_metadata,
)

FIXTURES_DIR = Path(__file__).parent / "fixtures"


# ---------------------------------------------------------------------------
# chunker
# ---------------------------------------------------------------------------

def test_chunk_short_text_is_single_chunk():
    assert chunk_text("hello world") == ["hello world"]


def test_chunk_empty_text_is_no_chunks():
    assert chunk_text("   ") == []


def test_chunk_sizes_and_overlap():
    text = "abcdefghij" * 1000  # 10_000 chars
    chunks = chunk_text(text, chunk_chars=1000, overlap_chars=150)
    assert len(chunks) > 1
    # Every chunk (except maybe the last) is exactly chunk_chars long.
    assert all(len(c) == 1000 for c in chunks[:-1])
    # Consecutive chunks overlap by overlap_chars: the tail of one == head of next.
    assert chunks[0][-150:] == chunks[1][:150]


def test_chunk_covers_all_text():
    text = "".join(chr(65 + (i % 26)) for i in range(5000))
    chunks = chunk_text(text, chunk_chars=800, overlap_chars=120)
    # Reconstruct by stripping the overlap from each subsequent chunk.
    step = 800 - 120
    rebuilt = chunks[0] + "".join(c[120:] for c in chunks[1:])
    assert rebuilt == text


def test_chunk_rejects_bad_overlap():
    with pytest.raises(ValueError):
        chunk_text("x" * 100, chunk_chars=100, overlap_chars=100)


def test_default_chunk_chars_reasonable():
    # ~650 tokens * 4 chars/token.
    assert DEFAULT_CHUNK_CHARS == 2600


# ---------------------------------------------------------------------------
# metadata resolution (flags-first, LLM-guessed fallback)
# ---------------------------------------------------------------------------

def _stub_llm(payload_json: str):
    return lambda prompt: payload_json


def test_guess_metadata_parses_llm_json():
    llm = _stub_llm('{"title": "RPE Guide", "topic": "RPE", "author": "J. Coach", "year": 2024}')
    guessed = guess_metadata("some text", llm)
    assert guessed.title == "RPE Guide"
    assert guessed.year == 2024


def test_guess_metadata_tolerates_bad_output():
    guessed = guess_metadata("text", _stub_llm("not json at all"))
    assert guessed.title is None and guessed.year is None


def test_resolve_metadata_flags_win_over_guess():
    provided = KnowledgeDoc(source="paper.pdf", title="My Title", topic="strength")
    llm = _stub_llm('{"title": "GUESSED", "topic": "GUESSED", "author": "Someone", "year": 2020}')
    merged = resolve_metadata("body", provided, llm=llm)
    # Provided fields untouched; only the blanks (author, year) filled from guess.
    assert merged.title == "My Title"
    assert merged.topic == "strength"
    assert merged.author == "Someone"
    assert merged.year == 2020


def test_resolve_metadata_no_llm_leaves_blanks_null():
    provided = KnowledgeDoc(source="x.pdf")
    merged = resolve_metadata("body", provided, llm=None)
    assert merged.title is None and merged.author is None


def test_resolve_metadata_skips_guess_when_complete():
    provided = KnowledgeDoc(title="t", topic="p", author="a", year=2021)
    called = {"n": 0}

    def llm(prompt):
        called["n"] += 1
        return "{}"

    resolve_metadata("body", provided, llm=llm)
    assert called["n"] == 0  # no missing fields -> LLM never invoked


# ---------------------------------------------------------------------------
# ingestion + retrieval round-trip
# ---------------------------------------------------------------------------

def test_ingest_embeds_chunks_with_metadata(fake_embedder, chroma_client):
    text = "RPE autoregulation. " * 500  # long enough to chunk
    n = ingest_knowledge(
        text,
        doc=KnowledgeDoc(source="rpe.pdf", title="RPE", topic="autoregulation"),
        embedder=fake_embedder,
        client=chroma_client,
    )
    assert n > 1
    collection = chroma_client.get_collection(KNOWLEDGE_COLLECTION)
    got = collection.get(include=["metadatas"])
    assert len(got["ids"]) == n
    meta = got["metadatas"][0]
    assert meta["source"] == "rpe.pdf"
    assert meta["topic"] == "autoregulation"
    assert meta["chunk_count"] == n
    assert got["ids"][0] == "knowledge::rpe.pdf::0"


def test_ingest_null_metadata_stored_as_empty(fake_embedder, chroma_client):
    ingest_knowledge(
        "short doc",
        doc=KnowledgeDoc(source="s.txt"),
        embedder=fake_embedder,
        client=chroma_client,
    )
    meta = chroma_client.get_collection(KNOWLEDGE_COLLECTION).get(include=["metadatas"])[
        "metadatas"
    ][0]
    assert meta["title"] == "" and meta["author"] == "" and meta["year"] == 0


def test_ingest_is_idempotent_per_source(fake_embedder, chroma_client):
    text = "deadlift volume landmarks. " * 400
    doc = KnowledgeDoc(source="dl.pdf")
    n1 = ingest_knowledge(text, doc=doc, embedder=fake_embedder, client=chroma_client)
    ingest_knowledge(text, doc=doc, embedder=fake_embedder, client=chroma_client)
    assert chroma_client.get_collection(KNOWLEDGE_COLLECTION).count() == n1  # upsert, no dupes


def test_ingest_empty_text_writes_nothing(fake_embedder, chroma_client):
    assert ingest_knowledge("   ", doc=KnowledgeDoc(source="e.txt"),
                            embedder=fake_embedder, client=chroma_client) == 0


def test_ingest_uses_llm_guess_for_missing_metadata(fake_embedder, chroma_client):
    ingest_knowledge(
        "a study about hypertrophy",
        doc=KnowledgeDoc(source="h.pdf"),
        llm=_stub_llm('{"title": "Hypertrophy Review", "topic": "hypertrophy", "author": null, "year": 2023}'),
        embedder=fake_embedder,
        client=chroma_client,
    )
    meta = chroma_client.get_collection(KNOWLEDGE_COLLECTION).get(include=["metadatas"])[
        "metadatas"
    ][0]
    assert meta["title"] == "Hypertrophy Review"
    assert meta["topic"] == "hypertrophy"
    assert meta["year"] == 2023


def test_ingest_knowledge_file_defaults_source_to_filename(fake_embedder, chroma_client):
    n = ingest_knowledge_file(
        FIXTURES_DIR / "study.pdf",
        embedder=fake_embedder,
        client=chroma_client,
    )
    assert n >= 1
    meta = chroma_client.get_collection(KNOWLEDGE_COLLECTION).get(include=["metadatas"])[
        "metadatas"
    ][0]
    assert meta["source"] == "study.pdf"
