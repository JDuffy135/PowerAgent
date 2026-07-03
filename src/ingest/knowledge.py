"""Knowledge-base ingestion: reference docs -> Chroma `knowledge` collection
(ARCHITECTURE.md §3.2).

This is the *reference material* path (studies, articles, PDFs, video
transcripts), distinct from the training-log path in `extract.py`/`commit.py`:

- **No HITL review.** Reference material isn't training data -- there's no
  schema to validate, no numbers to commit to SQLite. Chunks are embedded
  directly into the `knowledge` collection.
- **[DECISION] Character-approximation chunker.** ~500-800-token chunks at
  ~15% overlap, approximated as characters (~4 chars/token) so there's no
  tokenizer dependency. See `chunk_text`.
- **[DECISION] Metadata is flags-first, LLM-guessed as fallback.** The caller
  may pass any of `source/title/topic/author/year` explicitly; a small LLM pass
  fills in whichever were left blank (guessing from the document text), and
  anything still unknown defaults to NULL. Flags always win over the guess.

Reuses the `get_embedder()`/`get_chroma_client()` seams from `embed.py`, so
tests inject a fake embedder + in-memory client (no live Ollama/Chroma).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from pydantic import BaseModel, ValidationError

from src.ingest.embed import Embedder, get_chroma_client, get_embedder

KNOWLEDGE_COLLECTION = "knowledge"

# Character-approximation chunker knobs (**[DECISION]**, ARCHITECTURE.md §3.2).
# ~4 chars/token, ~650-token target chunk, ~15% overlap. Kept as characters so
# no tokenizer dependency is pulled in.
CHARS_PER_TOKEN = 4
DEFAULT_CHUNK_TOKENS = 650
DEFAULT_OVERLAP_FRACTION = 0.15
DEFAULT_CHUNK_CHARS = DEFAULT_CHUNK_TOKENS * CHARS_PER_TOKEN            # 2600
DEFAULT_OVERLAP_CHARS = int(DEFAULT_CHUNK_CHARS * DEFAULT_OVERLAP_FRACTION)  # 390

# Fields the LLM metadata-guess pass is allowed to fill (ARCHITECTURE.md §3.2).
METADATA_FIELDS = ("source", "title", "topic", "author", "year")

METADATA_SYSTEM_PROMPT = """You are cataloguing a reference document (a study, article, or transcript) \
for a powerlifting knowledge base. From the document text, infer bibliographic metadata. \
Output ONLY a single JSON object -- no prose, no markdown code fences -- with these keys:
- "title": the document's title, or null if you cannot tell.
- "topic": a short subject tag (e.g. "hypertrophy", "RPE", "deadlift technique"), or null.
- "author": the author(s), or null.
- "year": the publication year as an integer, or null.
Use null for any field you cannot determine with reasonable confidence. Do not guess wildly."""


class KnowledgeDoc(BaseModel):
    """One reference document's metadata. All fields optional -- missing values
    stay NULL (stored as empty string / 0 in Chroma scalar metadata)."""

    source: str | None = None   # e.g. filename or URL
    title: str | None = None
    topic: str | None = None
    author: str | None = None
    year: int | None = None


class GuessedMetadata(BaseModel):
    """Schema for the LLM metadata-guess pass (source is never guessed -- it's the
    upload provenance, always known to the caller)."""

    title: str | None = None
    topic: str | None = None
    author: str | None = None
    year: int | None = None


def chunk_text(
    text: str,
    chunk_chars: int = DEFAULT_CHUNK_CHARS,
    overlap_chars: int = DEFAULT_OVERLAP_CHARS,
) -> list[str]:
    """Split `text` into overlapping character-window chunks.

    Character-approximation of a token chunker (**[DECISION]**): ~4 chars/token,
    so the defaults target ~650-token chunks with ~15% overlap. The window slides
    by `chunk_chars - overlap_chars` each step. Whitespace-only chunks are
    dropped; short inputs yield a single chunk. Raises `ValueError` if
    `overlap_chars >= chunk_chars` (the window would never advance).
    """
    if chunk_chars <= 0:
        raise ValueError("chunk_chars must be positive")
    if overlap_chars < 0 or overlap_chars >= chunk_chars:
        raise ValueError("overlap_chars must be in [0, chunk_chars)")

    text = text.strip()
    if not text:
        return []
    if len(text) <= chunk_chars:
        return [text]

    step = chunk_chars - overlap_chars
    chunks: list[str] = []
    start = 0
    while start < len(text):
        piece = text[start : start + chunk_chars].strip()
        if piece:
            chunks.append(piece)
        if start + chunk_chars >= len(text):
            break
        start += step
    return chunks


def get_metadata_llm(node: str = "ingest_extract"):
    """Build the `prompt -> raw JSON string` callable for the metadata-guess pass.

    Reuses the extraction provider seam (`extract.get_llm`) with the metadata
    prompt + `GuessedMetadata` schema, so it honors the same local/cloud config
    routing as everything else. Imported lazily to keep this module import-light.
    """
    from src.ingest.extract import get_llm

    return get_llm(
        node,
        system_prompt=METADATA_SYSTEM_PROMPT,
        schema=GuessedMetadata.model_json_schema(),
    )


def guess_metadata(text: str, llm) -> GuessedMetadata:
    """Run the LLM metadata-guess pass over the document text.

    `llm` is a `prompt -> raw JSON string` callable (the same seam as
    `extract.get_llm`); the caller builds it with the metadata prompt + schema.
    Returns an all-null `GuessedMetadata` if the model output is unusable, so a
    flaky guess never blocks ingestion -- the fields simply stay NULL.
    """
    # Cap the text fed to the guesser; the head of a document carries the
    # title/author/year, and this keeps the prompt cheap.
    excerpt = text[:6000]
    try:
        raw = llm(excerpt)
        payload = json.loads(raw)
        return GuessedMetadata.model_validate(payload)
    except (ValueError, ValidationError, KeyError):
        return GuessedMetadata()


def resolve_metadata(
    text: str,
    provided: KnowledgeDoc,
    llm=None,
) -> KnowledgeDoc:
    """Merge caller-provided metadata with an LLM guess for the missing fields.

    **[DECISION]** Flags win: any field set on `provided` is kept verbatim. Only
    the still-missing fields (among title/topic/author/year) trigger the guess,
    and only if `llm` is supplied. `source` is never guessed -- it's the upload
    provenance. Anything still unknown stays NULL.
    """
    missing = [f for f in ("title", "topic", "author", "year") if getattr(provided, f) is None]
    if not missing or llm is None:
        return provided

    guessed = guess_metadata(text, llm)
    merged = provided.model_copy()
    for field in missing:
        value = getattr(guessed, field)
        if value is not None:
            setattr(merged, field, value)
    return merged


def _metadata_dict(doc: KnowledgeDoc, chunk_index: int, chunk_count: int) -> dict:
    """Chroma metadata scalars for one chunk. NULLs become '' / 0 so every key is
    always present (Chroma `where` filters can't match a missing key)."""
    return {
        "source": doc.source or "",
        "title": doc.title or "",
        "topic": doc.topic or "",
        "author": doc.author or "",
        "year": doc.year or 0,
        "chunk_index": chunk_index,
        "chunk_count": chunk_count,
    }


def _doc_id(doc: KnowledgeDoc, chunk_index: int) -> str:
    """Stable per-chunk id so re-ingesting the same source upserts, not
    duplicates. Keyed on source (falls back to title) + chunk index."""
    base = doc.source or doc.title or "untitled"
    return f"knowledge::{base}::{chunk_index}"


def ingest_knowledge(
    text: str,
    doc: KnowledgeDoc | None = None,
    llm=None,
    embedder: Embedder | None = None,
    client=None,
    collection_name: str = KNOWLEDGE_COLLECTION,
    chunk_chars: int = DEFAULT_CHUNK_CHARS,
    overlap_chars: int = DEFAULT_OVERLAP_CHARS,
) -> int:
    """Chunk `text`, resolve its metadata, and embed it into the `knowledge`
    collection. Returns the number of chunks written.

    Direct embed path -- no HITL (reference material, not training data). Metadata
    is flags-first with an optional LLM guess for the blanks (`resolve_metadata`).
    Idempotent per source: chunk ids are `knowledge::<source>::<i>` and we
    `upsert`, so re-ingesting a source overwrites rather than duplicates.
    """
    doc = doc or KnowledgeDoc()
    chunks = chunk_text(text, chunk_chars=chunk_chars, overlap_chars=overlap_chars)
    if not chunks:
        return 0

    doc = resolve_metadata(text, doc, llm=llm)

    if embedder is None:
        embedder = get_embedder()
    if client is None:
        client = get_chroma_client()

    collection = client.get_or_create_collection(collection_name)

    ids = [_doc_id(doc, i) for i in range(len(chunks))]
    metadatas = [_metadata_dict(doc, i, len(chunks)) for i in range(len(chunks))]
    embeddings = embedder(chunks)

    collection.upsert(
        ids=ids,
        documents=chunks,
        embeddings=embeddings,
        metadatas=metadatas,
    )
    return len(chunks)


def ingest_knowledge_file(
    path: str | Path,
    doc: KnowledgeDoc | None = None,
    llm=None,
    embedder: Embedder | None = None,
    client=None,
    **kwargs,
) -> int:
    """Load a file via `parse_upload` and ingest it. `source` defaults to the
    file name when the caller didn't set it explicitly."""
    from src.ingest.loaders import parse_upload

    path = Path(path)
    doc = doc or KnowledgeDoc()
    if doc.source is None:
        doc = doc.model_copy(update={"source": path.name})

    text = parse_upload(path)
    return ingest_knowledge(
        text, doc=doc, llm=llm, embedder=embedder, client=client, **kwargs
    )
