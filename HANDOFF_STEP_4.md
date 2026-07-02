# Handoff — Step 4: Complete the Tool Layer

For the next Claude session. Read `ARCHITECTURE.md` (full design) and the rst of this file below for what's built.
`IMPLEMENTATION_ROADMAP.md` has the full remaining-work plan (Stages 5–9).

## Where things stand (Steps 1–4 — DONE, 107 tests passing)

**Step 1 — data spine:** `src/db/schema.sql` + `connection.py`,
`resolve_exercise`/`add_exercise`, `get_best_set`/`get_lifts`/`get_e1rm_trend`/
`get_bodyweight_trend`, `src/seed.py` (targets `data/sample.db`).

**Step 2 — LLM extraction:** `ParsedBatch` models, `.txt` loader,
`extract_training_data` with the `get_llm(node)` provider seam.

**Step 3 — HITL staging + commit:** `stage_batch`/`get_pending_batch`,
`render_batch`, transactional `commit_batch`/`reject_batch`, Chroma
`personal_notes` embedding via `get_embedder()`/`get_chroma_client()`.

**Step 4 — the rest of the tool layer** (this step):
- `src/tools/queries.py` gained: `get_sessions`, `get_frequency`,
  `get_volume_trend`, `get_prs`, `find_recent_prs`, `commit_prs`,
  `get_injuries`, `get_measurements`, `get_programs`, `get_block_outline`,
  `compare_programmed_vs_actual`.
- `src/tools/vector.py` (new file): `search_notes` — semantic search over the
  `personal_notes` Chroma collection, mandatory metadata filter.
- `src/tools/sql.py` (new file): `run_readonly_sql` — the gated
  text-to-SQL escape hatch, validated via `sqlglot` (now a dependency).
- `src/seed.py` extended with `measurement` rows and `programmed_slot` rows for
  Strength Block 1 (one exact match, one variance case, one programmed-but-
  never-performed case — see below), so the new tools have fixture data.
- README + `IMPLEMENTATION_ROADMAP.md` updated; this handoff doc written.

## Decisions made this step (given by the user, now locked in)

1. **`get_volume_trend`** returns both `hard_sets` (count) and `tonnage_lb` per
   bucket. For bodyweight-only sets (`weight_lb IS NULL` — e.g. plain
   bodyweight pullups with no added weight), tonnage estimates load using the
   single most-recently-logged bodyweight in the whole `bodyweight` table
   (**not** scoped to the set's date), defaulting to `0` if no bodyweight has
   ever been recorded. See `_latest_bodyweight()` in `queries.py`. Note:
   Weighted Pullups in the seed data store the *added* weight in `weight_lb`
   (non-NULL), so they don't hit this path in the seeded fixtures — the
   fallback is tested directly in `test_get_volume_trend_bodyweight_only_set_*`.
2. **PR auto-derivation is a separate tool.** `get_prs` only reads the
   manually-recorded `pr` table. `find_recent_prs(conn, date_from, date_to,
   exercise=None)` is read-only and returns `PRCandidate`s: sets within the
   window whose Epley e1RM beats every prior set for that exercise (compared
   against all-time history, not just the window). `commit_prs(conn,
   candidates)` inserts user-accepted candidates with `context='auto-derived'`,
   and is idempotent (skips a candidate if an identical
   `exercise_id`+`date`+`weight_lb`+`reps` row already exists).
3. **`compare_programmed_vs_actual` fallback behavior**, per the user's
   explicit design: nothing is ever silently dropped.
   - Match key: `(week_number, day_number, exercise_id)` within the block.
   - A programmed slot with no matching performed session still appears in
     `rows`, with every `actual_*` field `None`.
   - Performed work with no matching programmed slot appears in
     `unmatched_actual` (includes the session's `raw_note` and top set).
   - If the block has **zero** `programmed_slot` rows at all, `rows=[]` and
     `note` explicitly says "No programmed data found for this block ... "
     while `unmatched_actual` still surfaces everything that was actually
     performed.
   - If there's **no data at all** (no programmed slots, no performed work),
     `note` says so and both lists are empty.
4. **`sqlglot`** accepted as a dependency (now in `pyproject.toml`) for
   `run_readonly_sql`'s SELECT-only validation.

## Implementation notes / gotchas for whoever builds on this

- **Chroma `where` range filters need numeric operands.** `$gte`/`$lte` on a
  string `date` field raises a `ValueError` at the chromadb layer. Fixed by
  adding a `date_ordinal` (int, `YYYYMMDD`) metadata mirror in
  `embed_session_notes` (`src/ingest/embed.py`) alongside the existing string
  `date` field. `search_notes` filters on `date_ordinal`; `date` is still what
  gets returned for display. If you add more Chroma collections/date filters
  later (e.g. the `knowledge` collection in Stage 8), reuse this pattern.
- **`exercises` filtering in `search_notes` is client-side.** Chroma metadata
  values must be scalars, so `exercises` is stored as a comma-joined string,
  not a list — a native `where` clause can't do substring containment. The
  fix: query Chroma with only date/doc_type in `where`, then filter the
  returned batch in Python by whether any requested exercise name appears in
  the comma-joined string. This means `n_results` limits the *pre-filter* pool,
  so a narrow `exercises` filter combined with a small `n_results` could
  under-return; bump `n_results` if that becomes a problem in practice.
- **`run_readonly_sql`'s timeout is best-effort and POSIX-only** (uses
  `signal.SIGALRM`/`setitimer`; silently becomes a no-op if unavailable, e.g.
  on Windows). Fine for local dev on Linux; revisit if this ever needs to run
  somewhere without POSIX signals.
- **A test-isolation bug was found and fixed in Step 3's test suite**: the
  existing `test_programmed_slots_skipped_but_preserved_in_audit` in
  `tests/ingest/test_commit.py` asserted `programmed_slot` table count `== 0`,
  which broke once the seeder started inserting `programmed_slot` rows. Fixed
  to compare before/after counts instead (the actual invariant being tested —
  that commit doesn't insert programmed slots — still holds).
- `get_frequency` counts **distinct sessions** an exercise appears in per
  bucket, not total sets — that's the natural reading of "frequency."

## Must handle / preserve (carried forward, still true)

- No live models in tests — everything LLM/embedder/Chroma-dependent stays
  behind an injectable seam (`get_llm`, `get_embedder`, `get_chroma_client`).
- HITL invariant: nothing durable happens outside the Step 3 commit path.
- lb canonical; unit conversion only at presentation (not built yet — that's
  SYNTHESIZE, Stage 6).
- Draft programs excluded from every analysis tool by default (`get_programs`
  is still the only one that can list them).
- Seeder only ever touches `data/sample.db`.

## What comes after (context only — do NOT build now)

Per `IMPLEMENTATION_ROADMAP.md`:
- **Stage 5** — LangGraph core: state, provider module, ROUTER, INGEST node
  wiring, checkpointer, minimal CLI REPL. Highest-risk stage (recommend
  opus + high / fable + medium). Has open decisions about block-assignment at
  ingest time and LangChain coupling — read that stage's "⚠️ Decisions" section
  before starting.
- **Stage 6** — ANALYZE ReAct loop + SYNTHESIZE + UPDATE_STATS (registers the
  tools built in this step).
- **Stage 7** — GENERATE program writer + cloud provider branch.
- **Stage 8** — xlsx/pdf loaders + knowledge-base ingestion + `search_knowledge`
  (independent of Stages 5–7, can run any time).
- **Stage 9 (optional)** — UI, backfill, ops polish.

Still no LangGraph, no router, no ANALYZE/GENERATE/UPDATE_STATS nodes, no CLI.
