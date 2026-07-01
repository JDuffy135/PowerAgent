# Handoff — Step 2: LLM Extraction Pipeline

For the next Claude session. Read `ARCHITECTURE.md` (full design) and
`CLAUDE_CODE_STEP_1.md` (what's already built) before starting.

## Where things stand (Step 1 — DONE)

Data spine is complete and tested (`pytest` → 15 passing):
- `src/db/schema.sql` + `src/db/connection.py` — full SQLite schema, WAL + FK pragmas, idempotent `init_db`.
- `src/tools/resolve.py` — `resolve_exercise()` (exact alias → fuzzy → None) + `add_exercise()`.
- `src/tools/queries.py` — `get_best_set`, `get_lifts`, `get_e1rm_trend`, `get_bodyweight_trend` (all Pydantic-typed, draft programs excluded, `ExerciseNotFound` on miss).
- `src/seed.py` — "2026 Meet 1 Prep" sample dataset; also used as the test fixture.

**Env note:** no system `pip`/`venv`; a manual `.venv` was bootstrapped (gitignored).
Activate with `source .venv/bin/activate`, then `python -m pytest -q`.

## Scope of Step 2 (and nothing more)

Build the **LLM extraction pipeline**: raw log text → schema-validated Pydantic
`ParsedBatch`. Per ARCHITECTURE.md §5.3. Still **no** LangGraph, Chroma, HITL
commit, or CLI — those are Steps 3–4.

1. **Pydantic models** (`src/ingest/models.py`): `ParsedBatch` mirroring the schema —
   sessions → sets/cardio, plus new-exercise candidates (name/tier/muscle_group guesses)
   that Step 3's HITL will confirm. Include per-field `confidence` where the parser is unsure.
2. **File loaders** (`src/ingest/loaders.py`): `parse_upload(path)` → raw text blocks.
   Support `.txt` now; xlsx (openpyxl/pandas) and pdf can be stubbed or minimal.
3. **Extraction node** (`src/ingest/extract.py`): `extract_training_data(text) -> ParsedBatch`.
   LLM-as-parser via Ollama (Qwen3 14B, structured output). Keep the provider behind a thin
   `get_llm()` seam (ARCHITECTURE.md §6.3) so it's cloud-flippable later.
4. **Golden-file tests** (`tests/ingest/`): raw log fixtures → expected `ParsedBatch` JSON.

## Parser must handle (all seen in real data — ARCHITECTURE.md §5.3)

- kg→lb mixed in one cell → normalize to lb, keep original in `raw_text`.
- projected vs actual in one cell (`1x3 @ 170KG (actually used 375 pounds)`) → `programmed_slot` vs `lift_set`.
- pin/plate configs (`143x1, 121x2`, `35KGx2`) → `equipment_note` + best-effort weight.
- top single + backoffs on one line (`385x1, 315x4`) → set rows with `is_top_set`.
- skipped exercises (`Reps: N/A`) → no set rows; note still preserved.
- slang/Spanish/emoji → irrelevant to numbers, preserved in prose.

## Reuse, don't rebuild

- Resolve exercise names with `resolve.resolve_exercise()`; unknown names become
  new-exercise candidates in `ParsedBatch` (do **not** auto-insert — that's HITL in Step 3).
- All weights normalize to **lb** at parse time (canonical unit). Preserve raw strings.
- Keep `extract_training_data` pure (text in → `ParsedBatch` out); no DB writes this step.

## Definition of done

- `extract_training_data(sample_log)` returns a valid `ParsedBatch` for the golden fixtures.
- Golden-file tests green; existing 15 Step-1 tests still green.
- No DB mutation, no Chroma, no LangGraph, no CLI yet.

## Known future cleanup (not Step 2, just be aware)

`src/seed.py` wipe-and-reloads `data/training.db` — the same file real ingested
logs will use. Before Step 3's commit path writes real data, retarget the seeder
to a separate `data/sample.db` (or guard it) so re-seeding can't erase real logs.
