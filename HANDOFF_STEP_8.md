# Handoff — Step 8: xlsx/pdf loaders + knowledge-base ingestion

For the next Claude session. Read `ARCHITECTURE.md` (full design) and the rest of
this file for what's built. `IMPLEMENTATION_ROADMAP.md` has the remaining-work
plan; **only the optional Stage 9 polish remains** — the core system (Stages
1–8) is complete.

## Where things stand (Steps 1–8 — DONE, 198 tests passing)

Steps 1–7 unchanged (data spine, LLM extraction, HITL staging/commit, full tool
layer, LangGraph core + CLI, ANALYZE/SYNTHESIZE/UPDATE_STATS, GENERATE + cloud
offload). Step 8 completes the file-loader surface and adds the reference
knowledge base:

- **`src/ingest/loaders.py`** — `parse_upload` now handles `.xlsx` and `.pdf`
  alongside `.txt` (the `NotImplementedError` stubs are gone):
  - `_load_xlsx` (openpyxl, `read_only=True, data_only=True`): **[DECISION] no
    structural assumptions** — one text block per sheet with a `=== Sheet:
    <name> ===` header, each non-empty row tab-joined verbatim, fully-blank rows
    and trailing-blank cells dropped. `data_only=True` yields cached formula
    values, not formula strings. Cell text is preserved **uncleaned** (mixed
    lb/kg, pin settings, emoji) — that mess is what `extract_training_data` is
    built for; pre-normalizing would throw away signal.
  - `_load_pdf` (pypdf): one block per page, `=== Page N ===` headers; pages
    with no extractable text (scanned images — OCR is out of scope) are skipped.
  - Deps added to `pyproject.toml`: `openpyxl>=3.1`, `pypdf>=6.0` (installed in
    `.venv`).
- **`src/ingest/knowledge.py`** (new) — the reference-material ingest path,
  distinct from the training-log path (no schema, no SQLite, **no HITL** — it's
  reference material, embedded directly):
  - `chunk_text` — **[DECISION] character-approximation chunker** (~4 chars/token,
    `DEFAULT_CHUNK_CHARS = 2600` ≈ 650 tokens, `DEFAULT_OVERLAP_CHARS = 390` ≈
    15%). Window slides by `chunk_chars - overlap_chars`; short input → one chunk;
    raises `ValueError` if `overlap >= chunk`.
  - `KnowledgeDoc` (Pydantic) — metadata `source/title/topic/author/year`, all
    optional (missing → NULL). `GuessedMetadata` is the LLM-guess schema (no
    `source` — that's upload provenance, always known).
  - `guess_metadata(text, llm)` / `resolve_metadata(text, provided, llm)` —
    **[DECISION] flags-first, LLM-guessed fallback**: any field the caller set
    wins; only the still-blank fields (title/topic/author/year) trigger the guess,
    and only if `llm` is supplied; anything still unknown stays NULL. `llm` is the
    same `prompt -> raw JSON` seam as `extract.get_llm`; `get_metadata_llm()`
    builds it with `METADATA_SYSTEM_PROMPT` + `GuessedMetadata` schema.
    `guess_metadata` swallows bad LLM output (returns all-null) so a flaky guess
    never blocks ingestion.
  - `ingest_knowledge(text, doc, llm, embedder, client)` — chunk → resolve
    metadata → `upsert` into the `knowledge` collection (`KNOWLEDGE_COLLECTION`).
    Idempotent per source (ids `knowledge::<source>::<i>`). NULL metadata stored
    as `''`/`0` (Chroma scalars; `where` can't match a missing key). Reuses the
    `get_embedder()`/`get_chroma_client()` seams from `embed.py`.
  - `ingest_knowledge_file(path, ...)` — `parse_upload` + `ingest_knowledge`;
    `source` defaults to the file name.
- **`src/tools/vector.py`** — `search_knowledge(query, topic=None)` +
  `KnowledgeResult`. **Unlike `search_notes`, a scope filter is optional** —
  reference material isn't time-windowed personal history, so unscoped similarity
  search is legitimate; `topic` narrows via a native Chroma `where` clause.
- **`src/agent/tools.py`** — `search_knowledge` registered as
  `search_knowledge_base` in `make_analyze_tools` (docstring steers it to
  external/theory questions, not the user's own logs). **Both ANALYZE and
  GENERATE pick it up automatically** — they share the toolset.
- **`src/cli.py`** — `/learn <path> [--source/--title/--topic/--author/--year
  ...]` command (`parse_learn` + `run_learn`), wired into the REPL loop before
  the router. Uses `shlex` for quoted flag values; a bad `--year` is dropped.
  `run_learn(llm=...)` is injectable (`_UNSET` sentinel → `get_metadata_llm()`
  by default) so tests pass a stub or `None` (skip guessing). Loader/embed
  failures print a message instead of killing the REPL. Banner updated.
- **Fixtures** (committed, authored once): `tests/ingest/fixtures/training_log.xlsx`
  (two sheets: strength + cardio, mixed lb/kg, `385x1, 315x4`, `Reps: N/A`, a
  blank row, emoji) and `tests/ingest/fixtures/study.pdf` (two text pages).
- Tests (27 new, 198 total): `tests/ingest/test_loaders.py` (xlsx one-block-per-
  sheet, verbatim messy cells, blank-row drop, pdf per-page, unsupported type),
  `tests/ingest/test_knowledge.py` (chunk sizes/overlap/coverage/edge cases,
  metadata flags-win + null fallback + skip-when-complete, ingest round-trip +
  idempotency + LLM-guess), `search_knowledge` tests in `tests/test_vector.py`
  (unscoped allowed, topic filter, empty collection), `/learn` parsing/handler
  in `tests/test_cli.py`.

## Decisions made this step (given by the user, now locked in)

1. **Chunker**: character-approximation (~4 chars/token), zero tokenizer deps.
2. **Knowledge ingest UX**: `/learn` CLI command, direct embed (no HITL);
   metadata flags-first with an LLM guess for omitted fields, NULL if not found.
3. **xlsx structure**: no layout assumptions — the loader emits the raw grid
   (one block per sheet) and lets the extractor infer shape.

## Implementation notes / gotchas for whoever builds on this

- **No tokenizer dependency by design.** If a future step wants exact token
  counts, the upgrade path is a tokenizer (e.g. tiktoken) swapped into
  `chunk_text`; the signature stays `text -> list[str]`.
- **`reportlab` was used only to author `study.pdf`** and was uninstalled — it is
  **not** a project/runtime dependency. The PDF fixture is committed; loading it
  needs only `pypdf`. Regenerating it needs reportlab reinstalled.
- **The knowledge collection has no HITL door** — this is intentional per the
  decision (reference material). If block-review/form-cue ingestion ever needs
  review, that goes through the training-log HITL path, not here.
- **`search_knowledge` is unscoped-friendly on purpose.** Keep the asymmetry with
  `search_notes` (which *requires* a filter): the two collections have different
  privacy/recall semantics.
- **xlsx reads with `data_only=True`**, so a workbook opened/saved by a tool that
  didn't compute formulas will yield `None` for those cells. Real coach exports
  from Excel/Sheets carry cached values, so this is the right default; note it if
  a fixture ever shows blank computed cells.

## Must handle / preserve (carried forward, still true)

- No live models in tests — every LLM/embedder/Chroma dependency stays behind an
  injectable seam (`get_llm`/`get_metadata_llm`, model factories, `get_embedder`,
  `get_chroma_client`, `_anthropic_client`). Knowledge tests use the fake
  embedder + `EphemeralClient`; metadata-guess tests use stub LLMs.
- HITL invariant unchanged: nothing durable to `session`/`lift_set`/`pr`/etc.
  outside an approved interrupt branch. **Knowledge ingest is exempt by decision**
  (reference material, its own collection) — it never touches SQLite or the
  `personal_notes` collection.
- lb canonical; unit conversion only at presentation.
- Draft programs excluded from every analysis tool by default.
- Seeder only touches `data/sample.db`; `data/training.db` is live; checkpoints
  in `data/checkpoints.db`.
- Never hard-require an API key when `provider: local`.

## What comes after (context only — do NOT build now)

Per `IMPLEMENTATION_ROADMAP.md`, **Stage 9 (optional polish)** is all that
remains — a grab-bag: Streamlit/Gradio UI over the same graph; historical
backfill (`--bulk` ingest of the real training archive); `display_unit: kg`
end-to-end pass; block-review / form-cue embedding paths
(`doc_type='block_review'|'form_cue'` exist in the design but nothing writes
them yet); ops niceties (DB backup, `ingest_batch` audit browser, re-embedding
command for embedder swaps); and the **program/block organizer** (reattach
sessions, rename/merge programs/blocks, fix ingest-time assignment mistakes) —
plus the "start this draft" flow (flip a draft to `incomplete`, attach sessions
as logged). Pick what matters; none of it is load-bearing for the core system.
