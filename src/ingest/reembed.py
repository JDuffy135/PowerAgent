"""Re-embedding command: rebuild Chroma collections with a new embedder
(ARCHITECTURE.md §3.2 — changing embedding models requires re-embedding, because
old and new vectors aren't comparable).

Every collection stores its documents + metadata in Chroma, so a re-embed needs
no source files: read `collection.get()`, re-embed the stored `documents` with
the currently configured embedder, and write them back. The rebuild is
**build-then-swap** — a fresh temp collection is populated in full *before* the
original is dropped, so a crash mid-run never leaves a half-embedded collection.
The rebuilt collection's metadata is stamped with the new embedder's name.

Reuses the `get_embedder()`/`get_chroma_client()` seams from `embed.py`, so tests
inject a fake embedder + in-memory client (no live Ollama/Chroma).
"""
from __future__ import annotations

from src.ingest.embed import (
    PERSONAL_NOTES_COLLECTION,
    Embedder,
    embedder_name,
    get_chroma_client,
    get_embedder,
)
from src.ingest.knowledge import KNOWLEDGE_COLLECTION

# The two collections the app writes to (personal notes + reference knowledge).
DEFAULT_COLLECTIONS = (PERSONAL_NOTES_COLLECTION, KNOWLEDGE_COLLECTION)


def collection_embedder(client, name: str) -> str | None:
    """The embedder name stamped on a collection's metadata, or None if unknown
    (collections created before the stamp existed carry nothing)."""
    try:
        collection = client.get_collection(name)
    except Exception:
        return None
    return (collection.metadata or {}).get("embedder")


def _existing_names(client) -> set[str]:
    return {c.name for c in client.list_collections()}


def reembed_collection(
    client,
    embedder: Embedder,
    name: str,
    new_embedder_name: str,
) -> int:
    """Rebuild one collection's vectors with `embedder`; returns the doc count.

    Build-then-swap: the re-embedded docs are written to `<name>__reembed` first,
    then the original is dropped and recreated (stamped with `new_embedder_name`)
    from the in-memory snapshot, then the temp collection is removed. An empty
    collection is left in place but still gets its embedder metadata stamped.
    """
    source = client.get_or_create_collection(name)
    data = source.get(include=["documents", "metadatas"])
    ids = data.get("ids") or []
    documents = data.get("documents") or []
    metadatas = data.get("metadatas") or []

    if not ids:
        # Nothing to re-embed; just make sure the embedder stamp is current.
        client.delete_collection(name)
        client.create_collection(name, metadata={"embedder": new_embedder_name})
        return 0

    new_embeddings = embedder(documents)

    tmp_name = f"{name}__reembed"
    if tmp_name in _existing_names(client):
        client.delete_collection(tmp_name)  # clear a stale temp from an aborted run
    tmp = client.create_collection(tmp_name, metadata={"embedder": new_embedder_name})
    tmp.add(ids=ids, documents=documents, embeddings=new_embeddings, metadatas=metadatas)

    # Only now, with a full temp collection on disk, drop and rebuild the original.
    client.delete_collection(name)
    final = client.create_collection(name, metadata={"embedder": new_embedder_name})
    final.add(ids=ids, documents=documents, embeddings=new_embeddings, metadatas=metadatas)
    client.delete_collection(tmp_name)
    return len(ids)


def reembed_all(
    embedder: Embedder | None = None,
    client=None,
    new_embedder_name: str | None = None,
    collections: tuple[str, ...] = DEFAULT_COLLECTIONS,
    write=print,
) -> dict[str, int]:
    """Re-embed every collection with the currently configured embedder.

    Returns `{collection_name: doc_count}`. `write` is the progress sink (defaults
    to `print` for the CLI; tests pass a list's `.append`). Missing collections
    are skipped with a note rather than treated as an error.
    """
    if embedder is None:
        embedder = get_embedder()
    if client is None:
        client = get_chroma_client()
    if new_embedder_name is None:
        new_embedder_name = embedder_name()

    existing = _existing_names(client)
    counts: dict[str, int] = {}
    for name in collections:
        if name not in existing:
            write(f"  {name}: not present, skipped")
            continue
        old = collection_embedder(client, name)
        n = reembed_collection(client, embedder, name, new_embedder_name)
        note = f" (was {old!r})" if old and old != new_embedder_name else ""
        write(f"  {name}: re-embedded {n} doc(s) with {new_embedder_name!r}{note}")
        counts[name] = n
    return counts
