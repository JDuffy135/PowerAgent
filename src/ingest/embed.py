"""Chroma prose embedding for session notes (ARCHITECTURE.md §3.2).

Session `raw_note` prose is embedded into the `personal_notes` collection so the
agent can later hop from a retrieved note to the exact numbers of that day (the
`session_id` metadata links prose back to the SQLite rows).

Two seams keep this testable without a live model, mirroring `get_llm` in
`extract.py`:

- `get_embedder(node)` returns a `texts -> vectors` callable, reading
  `config.yaml` `nodes.<node>` (defaults to local Ollama `nomic-embed-text`).
- `get_chroma_client(path)` returns a persistent client at `data/chroma/`.

Tests inject a deterministic fake embedder and an in-memory Chroma client, so no
Ollama server or on-disk store is required.
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Callable, Iterable, Sequence

import yaml
from pydantic import BaseModel

CONFIG_PATH = Path(__file__).parent.parent.parent / "config.yaml"

DEFAULT_OLLAMA_HOST = "http://localhost:11434"
DEFAULT_EMBED_MODEL = "nomic-embed-text"
DEFAULT_CHROMA_PATH = Path(__file__).parent.parent.parent / "data" / "chroma"

PERSONAL_NOTES_COLLECTION = "personal_notes"
ANALYSIS_DOC_TYPE = "analysis"  # SYNTHESIZE output stored via the "store this analysis?" offer

# Stage 11b doc types: block/program reviews and per-exercise form cues, all
# embedded into `personal_notes` so `search_notes(doc_type=...)` (and GENERATE's
# block-review retrieval) can pull them back.
BLOCK_REVIEW_DOC_TYPE = "block_review"
PROGRAM_REVIEW_DOC_TYPE = "program_review"
FORM_CUE_DOC_TYPE = "form_cue"

# A list of texts in, one embedding vector per text out.
Embedder = Callable[[Sequence[str]], list[list[float]]]


class SessionNote(BaseModel):
    """One session's prose plus the metadata that links it back to SQLite."""

    session_id: int
    date: str | None
    raw_note: str
    exercises: list[str] = []  # canonical exercise names mentioned that day


def _load_config() -> dict:
    if not CONFIG_PATH.exists():
        return {}
    return yaml.safe_load(CONFIG_PATH.read_text()) or {}


def _node_config(node: str) -> dict:
    return _load_config().get("nodes", {}).get(node, {}) or {}


def get_embedder(node: str = "ingest_embed") -> Embedder:
    """Return a `texts -> vectors` callable for the given graph node.

    Reads `config.yaml`'s `nodes.<node>` (`provider`/`model`/`host`); defaults to
    a local Ollama `/api/embed` call. Flipping providers is a config edit.
    """
    cfg = _node_config(node)
    provider = cfg.get("provider", "local")
    if provider != "local":
        raise NotImplementedError(
            f"Provider {provider!r} is not wired up yet (Step 3 embeddings are local-only)"
        )

    model = cfg.get("model", DEFAULT_EMBED_MODEL)
    host = cfg.get("host", DEFAULT_OLLAMA_HOST)

    def _embed(texts: Sequence[str]) -> list[list[float]]:
        payload = {"model": model, "input": list(texts)}
        request = urllib.request.Request(
            f"{host}/api/embed",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=180) as response:
                body = json.loads(response.read())
        except urllib.error.URLError as exc:
            raise RuntimeError(
                f"Ollama embed request to {host} failed (is `ollama serve` running?): {exc}"
            ) from exc
        return body["embeddings"]

    return _embed


def embedder_name(node: str = "ingest_embed") -> str:
    """The configured embedding model name for `node` (defaults to
    `nomic-embed-text`). Stamped into collection metadata by the re-embed command
    so a later embedder swap is detectable (ARCHITECTURE.md §3.2)."""
    return _node_config(node).get("model", DEFAULT_EMBED_MODEL)


def _chroma_path() -> Path:
    cfg = _load_config()
    raw = cfg.get("chroma_path")
    if raw:
        path = Path(raw)
        if not path.is_absolute():
            path = CONFIG_PATH.parent / path
        return path
    return DEFAULT_CHROMA_PATH


def get_chroma_client(path: str | Path | None = None):
    """Return a persistent Chroma client rooted at `data/chroma/` (or `path`).

    Imported lazily so the rest of the ingest pipeline doesn't pull in chromadb
    unless prose actually gets embedded.
    """
    import chromadb

    resolved = Path(path) if path is not None else _chroma_path()
    resolved.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(resolved))


def embed_session_notes(
    notes: Iterable[SessionNote],
    embedder: Embedder | None = None,
    client=None,
    collection_name: str = PERSONAL_NOTES_COLLECTION,
) -> int:
    """Embed each non-empty session note into the `personal_notes` collection.

    Idempotent per session: ids are `session_<id>` and we `upsert`, so
    re-embedding a session overwrites rather than duplicates. Notes with blank
    prose are skipped. Returns the number of notes embedded.

    Chroma metadata values must be scalars, so the `exercises` list is stored as
    a comma-joined string under the `exercises` key.
    """
    records = [n for n in notes if n.raw_note and n.raw_note.strip()]
    if not records:
        return 0

    if embedder is None:
        embedder = get_embedder()
    if client is None:
        client = get_chroma_client()

    collection = client.get_or_create_collection(collection_name)

    documents = [n.raw_note for n in records]
    embeddings = embedder(documents)
    ids = [f"session_{n.session_id}" for n in records]
    metadatas = [
        {
            "date": n.date or "",
            # Numeric mirror of `date` (YYYY-MM-DD -> YYYYMMDD int) so Chroma's
            # `where` range operators ($gte/$lte), which require int/float
            # operands, can filter by date. `date` stays the display string.
            "date_ordinal": int(n.date.replace("-", "")) if n.date else 0,
            "session_id": n.session_id,
            "doc_type": "session_note",
            "exercises": ", ".join(n.exercises),
        }
        for n in records
    ]

    collection.upsert(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas,
    )
    return len(records)


def embed_review(
    text: str,
    doc_id: str,
    doc_type: str,
    *,
    date: str | None = None,
    block_id: int = 0,
    program_id: int = 0,
    exercises: Sequence[str] = (),
    embedder: Embedder | None = None,
    client=None,
    collection_name: str = PERSONAL_NOTES_COLLECTION,
) -> str | None:
    """Upsert one block/program review or form cue into `personal_notes` (Stage 11b).

    Single-document (unchunked) and idempotent on `doc_id`, so re-saving an edited
    review overwrites in place rather than duplicating. Blank `text` deletes any
    existing doc under `doc_id` and returns None (clearing a review un-embeds it).
    `session_id` is 0 (these aren't tied to a logged session); `block_id` /
    `program_id` / `exercises` metadata let retrieval scope back to the source.
    """
    if embedder is None:
        embedder = get_embedder()
    if client is None:
        client = get_chroma_client()

    collection = client.get_or_create_collection(collection_name)

    if not text or not text.strip():
        collection.delete(ids=[doc_id])
        return None

    metadata = {
        "date": date or "",
        "date_ordinal": int(date.replace("-", "")) if date else 0,
        "session_id": 0,
        "doc_type": doc_type,
        "exercises": ", ".join(exercises),
        "block_id": block_id,
        "program_id": program_id,
    }
    collection.upsert(
        ids=[doc_id],
        documents=[text],
        embeddings=[embedder([text])[0]],
        metadatas=[metadata],
    )
    return doc_id


def block_review_id(block_id: int) -> str:
    return f"block_review_{block_id}"


def program_review_id(program_id: int) -> str:
    return f"program_review_{program_id}"


def embed_analysis(
    text: str,
    date: str | None = None,
    embedder: Embedder | None = None,
    client=None,
    collection_name: str = PERSONAL_NOTES_COLLECTION,
) -> str | None:
    """Embed one SYNTHESIZE analysis into `personal_notes` under `doc_type='analysis'`.

    Unlike session notes, an analysis isn't tied to a logged session, so it gets a
    unique time-based id (`analysis_<ms>`) and `session_id=0`. Returns the id, or
    None if `text` is blank. Reuses the same embedder/client seams as everything
    else so tests inject a fake embedder + in-memory client.
    """
    if not text or not text.strip():
        return None

    if embedder is None:
        embedder = get_embedder()
    if client is None:
        client = get_chroma_client()

    collection = client.get_or_create_collection(collection_name)

    doc_id = f"analysis_{int(time.time() * 1000)}"
    metadata = {
        "date": date or "",
        "date_ordinal": int(date.replace("-", "")) if date else 0,
        "session_id": 0,
        "doc_type": ANALYSIS_DOC_TYPE,
        "exercises": "",
    }
    collection.upsert(
        ids=[doc_id],
        documents=[text],
        embeddings=[embedder([text])[0]],
        metadatas=[metadata],
    )
    return doc_id
