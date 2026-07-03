"""Re-embedding command (Stage 11c) -- build-then-swap with a fake embedder and
an in-memory Chroma client, so no live Ollama/Chroma is required.
"""
from __future__ import annotations

from src.ingest.embed import PERSONAL_NOTES_COLLECTION
from src.ingest.knowledge import KNOWLEDGE_COLLECTION
from src.ingest.reembed import collection_embedder, reembed_all, reembed_collection


def _seed_collection(client, name, n=3):
    collection = client.get_or_create_collection(name)
    collection.add(
        ids=[f"{name}_{i}" for i in range(n)],
        documents=[f"doc {i} in {name}" for i in range(n)],
        embeddings=[[0.0] * 8 for _ in range(n)],
        metadatas=[{"doc_type": "session_note", "n": i} for i in range(n)],
    )
    return collection


def test_reembed_preserves_ids_docs_and_metadata(fake_embedder, chroma_client):
    _seed_collection(chroma_client, PERSONAL_NOTES_COLLECTION, n=3)

    n = reembed_collection(chroma_client, fake_embedder, PERSONAL_NOTES_COLLECTION, "newmodel")
    assert n == 3

    collection = chroma_client.get_collection(PERSONAL_NOTES_COLLECTION)
    got = collection.get(include=["documents", "metadatas", "embeddings"])
    assert sorted(got["ids"]) == [f"{PERSONAL_NOTES_COLLECTION}_{i}" for i in range(3)]
    # Documents + metadata survive; embeddings are now the fake embedder's output.
    assert all(d.startswith("doc ") for d in got["documents"])
    assert {m["doc_type"] for m in got["metadatas"]} == {"session_note"}
    assert collection_embedder(chroma_client, PERSONAL_NOTES_COLLECTION) == "newmodel"


def test_reembed_leaves_no_temp_collection(fake_embedder, chroma_client):
    _seed_collection(chroma_client, PERSONAL_NOTES_COLLECTION, n=2)
    reembed_collection(chroma_client, fake_embedder, PERSONAL_NOTES_COLLECTION, "m2")
    names = {c.name for c in chroma_client.list_collections()}
    assert f"{PERSONAL_NOTES_COLLECTION}__reembed" not in names


def test_reembed_empty_collection_stamps_embedder(fake_embedder, chroma_client):
    chroma_client.get_or_create_collection(PERSONAL_NOTES_COLLECTION)
    n = reembed_collection(chroma_client, fake_embedder, PERSONAL_NOTES_COLLECTION, "m3")
    assert n == 0
    assert collection_embedder(chroma_client, PERSONAL_NOTES_COLLECTION) == "m3"


def test_reembed_all_handles_both_and_skips_missing(fake_embedder, chroma_client):
    _seed_collection(chroma_client, PERSONAL_NOTES_COLLECTION, n=2)
    _seed_collection(chroma_client, KNOWLEDGE_COLLECTION, n=1)

    lines: list[str] = []
    counts = reembed_all(
        embedder=fake_embedder, client=chroma_client,
        new_embedder_name="swapped", write=lines.append,
    )
    assert counts == {PERSONAL_NOTES_COLLECTION: 2, KNOWLEDGE_COLLECTION: 1}
    assert collection_embedder(chroma_client, KNOWLEDGE_COLLECTION) == "swapped"


def test_reembed_all_reports_missing_collection(fake_embedder, chroma_client):
    _seed_collection(chroma_client, PERSONAL_NOTES_COLLECTION, n=1)
    lines: list[str] = []
    counts = reembed_all(
        embedder=fake_embedder, client=chroma_client,
        new_embedder_name="x", write=lines.append,
    )
    assert KNOWLEDGE_COLLECTION not in counts
    assert any("not present" in line for line in lines)
