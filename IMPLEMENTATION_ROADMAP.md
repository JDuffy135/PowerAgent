# Implementation Roadmap — Remaining Work (Stages 4–9)

Master plan for finishing the Powerlifting Coach per `ARCHITECTURE.md`. Steps 1–3
are **done** (67 tests passing as of 2026-07-02):

- **Step 1 — data spine:** full SQLite schema, `resolve_exercise`/`add_exercise`,
  `get_best_set` / `get_lifts` / `get_e1rm_trend` / `get_bodyweight_trend`, seeder
  (targets `data/sample.db`).
- **Step 2 — LLM extraction:** `ParsedBatch` models, `.txt` loader,
  `extract_training_data` with the `get_llm(node)` provider seam (local Ollama,
  structured JSON output). Golden-file tests with stub LLMs.
- **Step 3 — HITL staging + commit:** `stage_batch` / `get_pending_batch`,
  `render_batch` (confidence-flagged review text), transactional `commit_batch` /
  `reject_batch`, Chroma `personal_notes` embedding behind `get_embedder()` /
  `get_chroma_client()` seams.

Each stage below is sized for **one Claude Code session**. Stages 4 and 8 are
independent of each other; 5 → 6 → 7 are sequential. Every stage must follow the
established conventions: Pydantic-typed returns, seams for anything requiring a
live model (tests use fakes/stubs — no live Ollama in the dev environment),
golden-file tests where parsing is involved, lb-canonical weights, draft-program
exclusion, and a handoff-doc update at the end.

**Model notation:** `model + reasoning` is the *minimum effective* configuration;
using one notch above is always safe. (Scale: haiku < sonnet < opus < fable;
reasoning low < medium < high.)

---

## Stage 4 — Complete the tool layer ✅ DONE (2026-07-02)

**Minimum effective model: sonnet + medium.** This is pattern-following work —
`src/tools/queries.py` already establishes the idiom (resolve → SQL → Pydantic).
The only genuinely tricky function is `compare_programmed_vs_actual`.

**Status:** implemented. 107 tests passing (67 prior + 40 new). See
`HANDOFF_STEP_9.md` (the current handoff) for current state,
and the "Full tool layer (Step 4)"
section of `README.md` for usage. Resolved decisions (see that section for the
"why"): `get_volume_trend` returns both hard-set count and tonnage, bodyweight
sets estimate load off the single latest bodyweight entry (0 if none exists);
`find_recent_prs` + `commit_prs` is the separate auto-derivation path;
`compare_programmed_vs_actual` surfaces every mismatch explicitly rather than
dropping it; `sqlglot` was added as a dependency for `run_readonly_sql`.
`search_knowledge` remains deferred to Stage 8 as originally planned.

### Scope

Remaining typed query tools from ARCHITECTURE.md §5.1, in `src/tools/queries.py`
(or split into `queries.py` + `vector.py` + `sql.py` if the file gets long):

1. `get_sessions(date_from, date_to, block_id=None, session_type=None)`
2. `get_volume_trend(exercise_or_muscle_group, by='week'|'block', date_from, date_to)`
3. `get_frequency(exercise, by='week', date_from, date_to)`
4. `get_prs(exercise=None, date_from=None, date_to=None)`
5. `get_injuries(active_only=False, area=None)`
6. `get_measurements(site=None, date_from=None, date_to=None)`
7. `get_programs(status=None)` — the one tool allowed to show drafts
8. `get_block_outline(block_id)` — `programmed_slot` rows
9. `compare_programmed_vs_actual(block_id, exercise=None)`
10. `search_notes(query, date_from=None, date_to=None, exercises=None, doc_type=None)`
    — Chroma `personal_notes` similarity search with **mandatory metadata `where`
    filters** (§3.2); reuse `get_embedder()`/`get_chroma_client()` from
    `src/ingest/embed.py`. Tests use the same fake embedder + in-memory client
    pattern as `tests/ingest/test_embed.py`.
11. `run_readonly_sql(query)` — the gated escape hatch (§5.2): SELECT-only
    validation, read-only connection (`PRAGMA query_only`), row limit, timeout.

(`search_knowledge` is deferred to Stage 8, where the `knowledge` collection is
actually created.)

Also: extend `src/seed.py` with sample rows for the currently-unseeded tables
(`pr`, `injury`, `measurement`, `programmed_slot` if missing) so the new tools
have fixture data.

### Definition of done

- All new tools Pydantic-typed, draft-exclusion enforced, `ExerciseNotFound` on
  unresolvable names, unit tests against the seeded in-memory DB.
- `run_readonly_sql` rejects INSERT/UPDATE/DELETE/PRAGMA/ATTACH/multi-statement
  input, tested.
- `search_notes` tested with the fake embedder.
- README + a `HANDOFF_STEP_4.md` (or renamed equivalent) updated.

### ⚠️ Decisions you need to make (or explicitly delegate)

- **Volume definition** for `get_volume_trend`: tonnage (Σ weight×reps), hard-set
  count, or both? Recommendation: return both per bucket; but confirm. Also: how
  to bucket bodyweight-only sets (weight NULL) — exclude from tonnage, count in
  set count?
- **PR semantics** for `get_prs`: read only the manually-recorded `pr` table, or
  also derive all-time bests from `lift_set` on the fly? (Schema comment implies
  the `pr` table is authoritative; auto-derivation could be a separate tool.)
- **`compare_programmed_vs_actual` matching rule**: how a `programmed_slot`
  is paired with performed sessions — by `(block_id, week_number, day_number,
  exercise_id)`? What to return when day numbers are missing? Define the
  mismatch/skip semantics before implementing.
- **SQL validation dependency**: ARCHITECTURE.md suggests `sqlglot`. Accept the
  dependency, or use a stricter hand-rolled allowlist? (sqlglot is the safer
  parse; it's a pure-Python dep.)

---

## Stage 5 — LangGraph core: state, provider, ROUTER, INGEST node, minimal CLI ✅ DONE (2026-07-02)

**Minimum effective model: opus + high** (fable + medium if available). This is
the highest-risk stage: LangGraph `interrupt()`/resume semantics with a
`SqliteSaver` checkpointer, graph topology, and testing all of it without a live
LLM. Getting the HITL contract wrong here poisons Stages 6–7.

**Status:** implemented. 141 tests passing (107 prior + 34 new). See
`HANDOFF_STEP_9.md` (the current handoff) for the full writeup and the
"LangGraph agent core + CLI"
section of `README.md` for usage. All five open decisions resolved — see the
"Decisions" section below, now recording the final calls.

### Scope

Create `src/agent/` per ARCHITECTURE.md §4 and §7:

1. **Dependencies:** add `langgraph`, `langgraph-checkpoint-sqlite`, and
   whatever chat-model client is chosen (see decisions) to `pyproject.toml`.
2. **`src/agent/llm_provider.py`** — generalize `get_llm(node)` out of
   `src/ingest/extract.py` into the shared provider (§6.3). Keep
   `extract.get_llm` as a thin re-export or migrate call sites; don't break the
   existing tests. Local Ollama path only; the cloud branch lands in Stage 7
   (keep the `provider: cloud` raise-with-clear-message).
3. **`src/agent/state.py`** — `AgentState` TypedDict (§4.3): `messages`,
   `intent`, `evidence`, `pending_batch_id`, `display_unit`.
4. **`src/agent/nodes/router.py`** — intent classification into
   `ingest / analyze / generate / update_stats / chat`, structured output
   (Pydantic), local model, tiny prompt. Testable with an injected stub LLM.
5. **`src/agent/nodes/ingest.py`** — wire the existing pipeline:
   `parse_upload → extract_training_data → stage_batch → interrupt()`(showing
   `render_batch` output)` → commit_batch | reject_batch`. On free-text
   corrections: a small correction-application LLM pass that edits the staged
   `ParsedBatch`, re-renders, re-interrupts (§4.4). Nothing durable before
   approval — that invariant is already enforced by Step 3; don't bypass it.
6. **`src/agent/nodes/chitchat.py`** — trivial fallback node.
7. **`src/agent/graph.py`** — build the graph with `SqliteSaver` checkpointer
   (sibling DB file, e.g. `data/checkpoints.db`); ANALYZE/GENERATE/UPDATE_STATS
   registered as placeholder nodes that reply "not implemented yet".
8. **`src/cli.py`** — minimal REPL: read line → invoke graph with a thread id →
   print response; handle the interrupt/resume round-trip (`approve` / `reject`
   / free-text corrections). No streaming/rich UI required yet.
9. Tests: graph-level tests with stub LLMs — route each intent; run a full
   ingest→interrupt→approve flow against an in-memory DB and assert rows landed;
   ingest→reject writes nothing; correction pass round-trip with a stub.

### Definition of done

- `python -m src.cli` starts a REPL that (with a live Ollama) can ingest a .txt
  log end-to-end with HITL approval; all graph paths covered by stub-LLM tests;
  existing 67 tests still green.

### ✅ Decisions (resolved by the user, 2026-07-02 — now locked in)

- **LangChain coupling**: adopted `langchain-ollama` (`ChatOllama` via
  `llm_provider.get_chat_model(node)`) for the agent graph; the Step 2 raw-
  `urllib` `get_llm` callable seam stays as-is for structured-output pipelines
  (extraction + correction) and is re-exported from `src/agent/llm_provider.py`,
  the canonical import point going forward.
- **Correction-pass contract**: full re-emit — the correction LLM receives the
  original batch JSON + the user's free text and re-emits the complete
  `ParsedBatch`, which re-validates in one shot (`src/ingest/correct.py`).
  The correct→re-render loop is capped at **5 rounds**
  (`MAX_CORRECTION_ROUNDS`); at the cap only approve/reject are accepted.
- **Block assignment at ingest**: option (a) — after approval, the HITL flow
  asks which program/block the batch belongs to (existing block id, `new
  <program> / <block>` created on the fly with status `incomplete`, or `none`
  to leave unattached). Assigning a block unblocks `programmed_slot` insertion.
  Unattached/misfiled batches can be reorganized later — a "review and organize
  programs/blocks" feature is planned (Stage 9 grab-bag) so ingest-time
  mistakes are never permanent.
- **How INGEST receives files**: a dedicated CLI command, `/ingest <path>`,
  which presets `intent='ingest'` and skips the router (more reliable with
  small local models). The eventual UI replaces this with drag-and-drop / an
  "import from my PC" file picker; the graph input shape is already
  UI-agnostic, so only the front-end changes.
- **Checkpointer location**: separate `data/checkpoints.db` (`checkpoints_db`
  in `config.yaml`), keeping `training.db` purely domain data.

---

## Stage 6 — ANALYZE ReAct loop + SYNTHESIZE + UPDATE_STATS ✅ DONE (2026-07-02)

**Minimum effective model: opus + medium** (sonnet + high workable if Stage 5
left clean scaffolding). The code volume is moderate, but making a *local Qwen3
14B* drive a ReAct tool loop reliably is prompt/tool-schema engineering that
benefits from stronger judgment.

**Status:** implemented. 161 tests passing (141 prior + 20 new). See
`HANDOFF_STEP_9.md` for the full writeup and the "ANALYZE + SYNTHESIZE +
UPDATE_STATS (Step 6)" section of `README.md` for usage. All four open decisions
resolved — see the "✅ Decisions" section below, now recording the final calls.

### Scope

1. **Tool registration:** wrap all Stage 1 + Stage 4 tools as LangGraph tools
   with tight docstrings/arg schemas (small-model-friendly: few args, enums,
   explicit date formats). Include `search_notes` and gated `run_readonly_sql`
   (prompt says: prefer typed tools).
2. **`src/agent/nodes/analyze.py`** — ReAct loop: bounded iterations (e.g. max
   8 tool calls), accumulates `evidence` in state, handles `ExerciseNotFound`
   gracefully (ask user / try fuzzy alternatives), hands off to SYNTHESIZE.
   Cloud-flippable via `llm_provider` (config only; cloud impl is Stage 7).
3. **`src/agent/nodes/synthesize.py`** — compose final answer from `evidence`;
   apply `display_unit` (lb→kg conversion at presentation only, §2); cite the
   source sets behind every e1RM/PR claim (tools already return them);
   optional `interrupt()` to offer storing an analysis/review (can be stubbed to
   "not offered yet" — see decisions).
4. **`src/agent/nodes/update_stats.py`** — parse "bodyweight was 146 this
   morning" / "hit a 405 deadlift PR" into single-row inserts to
   `bodyweight`/`pr`/`injury`/`measurement`, with lightweight confirm-before-write
   HITL (reuse the interrupt pattern from Stage 5).
5. Tests: scripted-stub-LLM tests that walk the ReAct loop (stub emits tool
   calls, harness executes real tools against the seeded DB, stub then emits the
   final answer); unit conversion tests; UPDATE_STATS confirm/insert/decline paths.

### Definition of done

- With live Ollama: "what was my best bench in March?", "show my deadlift e1RM
  trend this prep", "any knee-pain mentions in the last two blocks?" all answered
  with tool-backed evidence. Stub-LLM tests green without a model.

### ✅ Decisions (resolved by the user, 2026-07-02 — now locked in)

- **ANALYZE/SYNTHESIZE model**: the **Qwen3.6 35B-A3B MoE** (config
  `nodes.analyze` / `nodes.synthesize`, both `qwen3.6:35b-a3b`). Step down to
  `qwen3:14b` later if latency bottlenecks — pure config change; the ReAct prompt
  is tuned for a capable local model. UPDATE_STATS stays on the 14B workhorse.
- **Evidence overflow**: the ANALYZE loop is capped at `MAX_TOOL_CALLS = 8`
  (model turns). On overflow it sets `evidence_truncated`; SYNTHESIZE answers
  with the **partial evidence + a fixed disclaimer that also tells the user to
  narrow the question scope** (specific exercise / shorter range / one metric).
  Per-evidence-item serialization is capped at `MAX_EVIDENCE_CHARS = 2000`.
- **"Store this analysis?" interrupt**: **implemented now.** SYNTHESIZE stashes
  the answer + flags `offer_store`; a dedicated `store_offer` node interrupts
  ("store this? yes/no") and, on yes, embeds the analysis into Chroma
  `personal_notes` under `doc_type='analysis'` (`embed_analysis`, id
  `analysis_<ms>`, `session_id=0`). Only offered when there's real evidence.
- **UPDATE_STATS scope**: **`bodyweight` + `pr` only** for now (injury/measurement
  deferred). The parse LLM auto-detects which of the two was reported and
  normalizes the weight to lb; unknown PR exercises are declined at parse time.

---

## Stage 7 — GENERATE (program writer) + cloud offload ✅ DONE (2026-07-02)

**Minimum effective model: sonnet + high** for the plumbing; the *value* of this
stage is in the GENERATE system prompt, and drafting/iterating that prompt is
where opus + medium pays for itself. Recommendation: opus + medium.

**Status:** implemented. 171 tests passing (161 prior + 10 net new). See
`HANDOFF_STEP_9.md` for the full writeup and the "GENERATE (program writer) +
cloud offload (Step 7)" section of `README.md` for usage. All four open
decisions resolved — see the "✅ Decisions" section below, now recording the
final calls.

### Scope

1. **Cloud provider branch in `llm_provider.py`** — implement
   `provider: cloud`: read model + API key env var from `config.yaml`, return a
   chat model. Flipping ANALYZE/GENERATE to cloud becomes the promised
   zero-code-change config edit. Never hard-require a key when
   `provider: local`.
2. **`src/agent/nodes/generate.py`** — program/block writer (§4.2): pulls
   history through the same tools (recent e1RMs, `get_volume_trend`,
   `get_injuries(active_only=True)`, block reviews via `search_notes`,
   `compare_programmed_vs_actual` for what progressions actually held up), then
   drafts a block/macrocycle.
3. **Persist drafts**: write the accepted draft as `program(status='draft')` +
   `block` + `programmed_slot` rows — via HITL confirm (reuse interrupt
   pattern). Draft exclusion (§8.2) already keeps these out of analysis.
4. Tests: stub-LLM graph tests (evidence-gathering calls happen, draft rows
   land only on approval); cloud provider unit test with a fake transport (no
   real API calls in CI).

### Definition of done

- "Write me a 4-week strength block based on this prep" produces a reviewable
  draft grounded in queried history; on approval it exists as a `draft` program
  queryable via `get_programs('draft')` / `get_block_outline`.

### ✅ Decisions (resolved by the user, 2026-07-02 — now locked in)

- **Cloud vendor + SDK**: the **Anthropic API, default model `claude-sonnet-5`**.
  `get_chat_model` uses `langchain-anthropic` (`ChatAnthropic`); the raw
  `get_llm` seam uses the `anthropic` SDK directly, with the target JSON schema
  embedded in the system prompt (arbitrary Pydantic schemas aren't guaranteed
  strict-structured-output-compatible; downstream Pydantic validation stays the
  contract, same as the Ollama path). API key env var: the name is configurable
  per node via `nodes.<node>.api_key_env`, **default `ANTHROPIC_API_KEY`**; a
  key is never required when `provider: local`. GENERATE defaults to
  `provider: cloud` in `config.yaml`.
- **Privacy option**: **deferred** — cloud nodes receive the raw evidence.
  The anonymized-summary mode remains a future option (§6.3).
- **Draft output format**: **structured output** — `DraftProgram`/`DraftSlot`
  Pydantic models reusing the `ParsedProgrammedSlot` shape, so an approved
  draft is machine-insertable into `program`/`block`/`programmed_slot` with no
  second extraction pass.
- **Generation guardrails**: the user's training philosophy is encoded verbatim
  as `generate.TRAINING_PHILOSOPHY` and injected into both GENERATE prompts:
  (1) easy-moderate SBD work, accessories pushed to failure at recoverable
  volume; (2) 4-week SBD waves with ramping top-set RPE (wk1 easy → wk4
  near-max, e.g. deadlift single RPE 5→9); (3) weak-point SBD variations ≥1x/wk
  in every block except peaking blocks (comp-specific there); (4) 4-5 training
  days/wk unless specified; (5) program around active injuries (substitute
  patterns for a whole block, trim volume/intensity as needed); (6) unknown
  accessory volume → start ~10 sets/wk per muscle group, counting only 7+-rep
  SBD sets toward it; (7) unknown SBD volume → ~7-9 weekly deadlift, 8-10
  squat, 10-15 bench sets (variations included); (8) default frequency: squat/
  deadlift 2x/wk (primary + secondary), bench 3x/wk (heavy/light/moderate).

---

## Stage 8 — File loaders (xlsx/pdf/txt) + knowledge base ingestion ✅ DONE (2026-07-02)

**Minimum effective model: sonnet + medium.** Well-trodden libraries and an
established pipeline shape. Independent of Stages 5–7; can run any time after
Stage 3 (only `search_knowledge` registration depends on Stage 6's tool wiring).

**Status:** implemented. 198 tests passing (171 prior + 27 new). See
`HANDOFF_STEP_9.md` for the full writeup and the "File loaders + knowledge base
(Step 8)" section of `README.md` for usage. All three open decisions resolved —
see the "✅ Decisions" section below, now recording the final calls.

### Scope

1. **`src/ingest/loaders.py`** — implement the stubbed `.xlsx` (openpyxl;
   preserve cell text verbatim — the messy strings are what the extractor is
   built for, don't pre-clean) and `.pdf` (pypdf text extraction) branches of
   `parse_upload`. Multi-sheet/tab handling: emit one text block per sheet with
   a header line. Golden fixtures: at least one real-ish workbook + PDF.
2. **Knowledge ingestion** (`src/ingest/knowledge.py`): studies/articles/PDFs →
   `knowledge` Chroma collection (§3.2): ~500–800-token chunks, ~15 % overlap,
   metadata `source/title/topic/author/year`. Reuses `get_embedder()`/
   `get_chroma_client()`. This is a *direct* embed path — no HITL review needed
   (it's reference material, not training data) unless you decide otherwise.
3. **`search_knowledge(query, topic=None)`** tool + registration in the ANALYZE
   toolset (if Stage 6 is done; otherwise leave as a plain function like the
   Stage 4 tools).
4. Tests: loader golden files; chunker unit tests (sizes/overlap); fake-embedder
   ingestion + retrieval round-trip.

### ✅ Decisions (resolved by the user, 2026-07-02 — now locked in)

- **Chunker**: **character-approximation** — `src/ingest/knowledge.py::chunk_text`
  approximates a token chunker at ~4 chars/token (`DEFAULT_CHUNK_CHARS = 2600`,
  ~650 tokens; ~15% overlap = `DEFAULT_OVERLAP_CHARS = 390`), so no tokenizer
  dependency is pulled in. The window slides by `chunk_chars - overlap_chars`;
  short inputs yield one chunk.
- **Knowledge ingest UX**: a **CLI command, `/learn <path> [--source/--title/
  --topic/--author/--year ...]`**, embedding directly (no HITL — reference
  material, not training data). **Metadata is flags-first with an LLM fallback**:
  any flag the user passes wins; a small LLM pass (`guess_metadata`, reusing the
  `get_llm` seam via `get_metadata_llm`) fills whichever fields were omitted from
  the document text; anything still unknown **defaults to NULL** (stored as
  `''`/`0` in Chroma scalar metadata). `source` is never guessed — it's upload
  provenance, defaulting to the file name.
- **xlsx structure assumption**: **none — the loader is built around guesses.**
  Every coach's workbook differs (blocks-per-sheet, weeks-as-columns, free-form),
  so `_load_xlsx` makes no structural assumptions: one text block per sheet with a
  `=== Sheet: <name> ===` header, each non-empty row tab-joined verbatim (blank
  rows/trailing-blank cells dropped), `data_only=True` so formula cells yield
  cached values. The messy cell strings are preserved uncleaned for the LLM
  extractor. Fixture: `tests/ingest/fixtures/training_log.xlsx` (two sheets, mixed
  lb/kg, `Reps: N/A`, emoji). PDF path (`_load_pdf`, pypdf) mirrors it: one block
  per page, `=== Page N ===` headers, empty/scanned pages skipped (OCR out of
  scope). Fixture: `tests/ingest/fixtures/study.pdf`.

---

## Stage 9 (optional) — Polish: Streamlit/Gradio UI, backfill, ops ✅ DONE (2026-07-03)

**Minimum effective model: sonnet + medium.**

**Status:** implemented. 240 tests passing (198 prior + 42 new). See
`HANDOFF_STEP_9.md` for the full writeup and the "Streamlit UI + backfill +
organizer + dev tools (Step 9)" section of `README.md` for usage. Built from
the grab-bag: the Streamlit UI (`streamlit run src/ui/app.py` — chat with
interrupt round-trips, organizer, backfill, and dev-tools tabs), the
program/block organizer (`src/tools/organize.py`, incl. the "start this draft"
flow), historical backfill (`src/ingest/backfill.py`), and the ops niceties
(DB backup, `ingest_batch` audit browser, direct table CRUD in
`src/tools/admin.py`). Deliberately *not* built (still open, none
load-bearing): the `display_unit: kg` end-to-end pass, block-review /
form-cue embedding paths, and a re-embedding command for embedder swaps.

Only start once the CLI flow is stable (§7 roadmap). Grab-bag, pick what matters:

- Streamlit or Gradio chat UI over the same graph (checkpointer thread per
  browser session); render `render_batch` reviews and approve/reject buttons.
- **Historical backfill**: batch-ingest your real training archive through the
  pipeline (mostly your time in HITL review, not code — but may motivate a
  `--bulk` mode with relaxed per-session confirmation).
- `display_unit: kg` end-to-end pass; block-review / form-cue embedding paths
  (`doc_type='block_review'|'form_cue'` are in the schema design but nothing
  writes them yet — they'd come from macrocycle-review ingestion).
- Ops niceties: DB backup command, `ingest_batch` audit browser, re-embedding
  command for embedder swaps (§3.2 notes embedder changes require re-embedding).
- **Program/block organizer** (committed to during Stage 5's block-assignment
  decision): a review flow to reattach sessions to a different block, rename or
  merge programs/blocks, and fix ingest-time assignment mistakes — so block
  assignment at ingest never has to be perfect.

### ✅ Decisions (resolved by the user, 2026-07-03 — now locked in)

- **Streamlit vs. Gradio**: **Streamlit** (user's call, matching the
  recommendation). Tab layout per the user's spec — Chat (main), Organizer,
  Dev Tools — plus a dedicated **Backfill** tab (historical backfill is a
  data-entry workflow, not a dev QoL tool, so it got its own tab).
- **Bulk-backfill HITL**: two modes. *Stage for review* keeps the per-batch
  approval contract (chunks land `pending_review`, reviewed/committed one by
  one in the Backfill tab); *Commit everything* is the relaxed `--bulk`
  equivalent — the single explicit "run and commit" click approves the whole
  archive, with every chunk still audited in `ingest_batch`.
- **Dev-tools CRUD sits outside the HITL interrupt flow** by design: every
  write is an explicit hand edit in the grid, and the surface is never exposed
  to the LLM as a tool.
- **Block reviews do NOT get their own ingestion path yet** (the second open
  decision): deferred; nothing writes `doc_type='block_review'|'form_cue'`.

---

## Cross-stage conventions (bind every session to these)

1. **No live models in tests.** Every LLM/embedder/Chroma dependency goes behind
   an injectable seam; tests use stubs/fakes/in-memory clients. This is already
   the house style — keep it.
2. **HITL invariant:** nothing touches `session`/`lift_set`/Chroma/`pr`/etc.
   without explicit user approval through an interrupt (Stage 3's commit path is
   the only door).
3. **lb canonical, conversion only in SYNTHESIZE/presentation.**
4. **Draft programs excluded from analysis by default.**
5. **Seeder touches `data/sample.db` only; `data/training.db` is live.**
6. Each session ends by updating `README.md` and writing `HANDOFF_STEP_<n+1>.md`
   in the style of `HANDOFF_STEP_3.md` (scope of next step, must-preserve
   invariants, definition of done).
