"""Semantic search over the Chroma `personal_notes` collection (ARCHITECTURE.md §3.2).

Per §3.2, metadata `where` filters are mandatory for `personal_notes` queries so
semantic search always respects a time window / doc type / exercise scope rather
than searching the whole history blind. Reuses the `get_embedder()` /
`get_chroma_client()` seams from `src.ingest.embed` so tests can inject a fake
embedder + in-memory client (no live Ollama/Chroma required).
"""
from __future__ import annotations

from typing import Sequence

from pydantic import BaseModel

from src.ingest.embed import PERSONAL_NOTES_COLLECTION, Embedder, get_chroma_client, get_embedder
from src.ingest.knowledge import KNOWLEDGE_COLLECTION


class NoteResult(BaseModel):
    session_id: int
    date: str | None
    doc_type: str | None
    text: str
    exercises: list[str]
    distance: float | None


def search_notes(
    query: str,
    date_from: str | None = None,
    date_to: str | None = None,
    exercises: Sequence[str] | None = None,
    doc_type: str | None = None,
    n_results: int = 10,
    embedder: Embedder | None = None,
    client=None,
) -> list[NoteResult]:
    """Semantic search over `personal_notes`, scoped by at least one metadata filter.

    Raises `ValueError` if none of `date_from`/`date_to`/`exercises`/`doc_type`
    are given -- per §3.2 an unscoped similarity search over all personal notes
    is not allowed.

    `exercises` matches client-side (Chroma metadata stores the mentioned
    exercises as a comma-joined string, not a list, so substring containment
    can't be expressed as a native `where` clause).
    """
    if date_from is None and date_to is None and not exercises and doc_type is None:
        raise ValueError(
            "search_notes requires at least one metadata filter "
            "(date_from, date_to, exercises, or doc_type)"
        )

    # Chroma's $gte/$lte require numeric operands, so date filtering goes
    # through the numeric `date_ordinal` metadata mirror (see `embed.py`),
    # not the display-string `date` field.
    where_clauses = []
    if date_from is not None:
        where_clauses.append({"date_ordinal": {"$gte": int(date_from.replace("-", ""))}})
    if date_to is not None:
        where_clauses.append({"date_ordinal": {"$lte": int(date_to.replace("-", ""))}})
    if doc_type is not None:
        where_clauses.append({"doc_type": doc_type})

    if len(where_clauses) == 1:
        where = where_clauses[0]
    elif len(where_clauses) > 1:
        where = {"$and": where_clauses}
    else:
        where = None

    if embedder is None:
        embedder = get_embedder()
    if client is None:
        client = get_chroma_client()

    collection = client.get_or_create_collection(PERSONAL_NOTES_COLLECTION)

    query_embedding = embedder([query])[0]
    result = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        where=where,
    )

    ids = result.get("ids", [[]])[0]
    documents = result.get("documents", [[]])[0]
    metadatas = result.get("metadatas", [[]])[0]
    distances = (result.get("distances") or [[]])[0]

    notes: list[NoteResult] = []
    for i, doc in enumerate(documents):
        metadata = metadatas[i] or {}
        note_exercises = [e.strip() for e in (metadata.get("exercises") or "").split(",") if e.strip()]

        if exercises:
            wanted = {e.lower() for e in exercises}
            if not any(e.lower() in wanted for e in note_exercises):
                continue

        notes.append(
            NoteResult(
                session_id=metadata.get("session_id"),
                date=metadata.get("date") or None,
                doc_type=metadata.get("doc_type"),
                text=doc,
                exercises=note_exercises,
                distance=distances[i] if i < len(distances) else None,
            )
        )

    return notes


class KnowledgeResult(BaseModel):
    source: str | None
    title: str | None
    topic: str | None
    author: str | None
    year: int | None
    text: str
    distance: float | None


def search_knowledge(
    query: str,
    topic: str | None = None,
    n_results: int = 5,
    embedder: Embedder | None = None,
    client=None,
) -> list[KnowledgeResult]:
    """Semantic search over the reference `knowledge` collection (§3.2).

    Unlike `search_notes`, a scope filter is *optional* here: knowledge is
    reference material, not time-windowed personal history, so an unscoped
    similarity search is legitimate. Passing `topic` narrows to matching docs via
    a native Chroma `where` clause. Reuses the same embedder/client seams so tests
    inject a fake embedder + in-memory client.
    """
    where = {"topic": topic} if topic else None

    if embedder is None:
        embedder = get_embedder()
    if client is None:
        client = get_chroma_client()

    collection = client.get_or_create_collection(KNOWLEDGE_COLLECTION)

    query_embedding = embedder([query])[0]
    result = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        where=where,
    )

    documents = result.get("documents", [[]])[0]
    metadatas = result.get("metadatas", [[]])[0]
    distances = (result.get("distances") or [[]])[0]

    hits: list[KnowledgeResult] = []
    for i, doc in enumerate(documents):
        metadata = metadatas[i] or {}
        hits.append(
            KnowledgeResult(
                source=metadata.get("source") or None,
                title=metadata.get("title") or None,
                topic=metadata.get("topic") or None,
                author=metadata.get("author") or None,
                year=metadata.get("year") or None,
                text=doc,
                distance=distances[i] if i < len(distances) else None,
            )
        )

    return hits
