# Handoff — Step 11: Deferred polish (kg, reviews/cues, re-embed)

For the next Claude session. Read `ARCHITECTURE.md` (full design) and the rest
of this file for what's built. `IMPLEMENTATION_ROADMAP.md` Stages 1–11 are now
**all complete** — the roadmap is finished. There is no planned remaining stage;
what's left is genuinely optional (see "Ideas, if you want more" below).

## Where things stand (Steps 1–11 — DONE, 278 tests passing)

Steps 1–10 unchanged (data spine, LLM extraction, HITL staging/commit, full tool
layer, LangGraph core + CLI, ANALYZE/SYNTHESIZE/UPDATE_STATS, GENERATE + cloud
offload, file loaders + knowledge base, Streamlit UI + organizer + backfill +
dev tools, Trends tab). Step 11 closes the three long-deferred polish items:

### 11a — `display_unit: kg` end-to-end

lb stays canonical; conversion is presentation-only (§2). Helpers in
`src/agent/units.py` (`to_display_weight`, `format_weight`) were already applied
by SYNTHESIZE, UPDATE_STATS, and GENERATE draft rendering. This step closed the
gaps:

- **`src/ingest/review.py`**: `render_batch` / `_render_session` / `_render_set`
  / `_render_slot` / `_fmt_weight` now take a `unit` (default `"lb"`) and format
  via `format_weight`. Callers: the `ingest_review` node
  (`src/agent/nodes/ingest.py`) passes `state["display_unit"]`; the Backfill
  tab passes the configured unit.
- **Trends** (`src/ui/trends.py`): frame builders take `unit=` and convert the
  weight columns (`bodyweight_frame`, `e1rm_frame`, `pr_frame`, `volume_frame` —
  `tonnage_lb` only, `hard_sets` is a count). `measurement_frame` is never
  converted (inches). New `weight_label(unit)` → `"(kg)"` for axis titles. The
  tab (`trends_tab.py`) takes `display_unit`, converts the bodyweight
  first/last/… metrics, and labels every axis/tooltip with the unit. **Note the
  weight column *names* keep their historical spelling (`weight_lb`, `tonnage_lb`)
  even when holding kg — they're the display series, chosen so the Altair
  encodings stay stable; don't read them as "always lb".**
- **Config plumbing**: `make_input` (`src/cli.py`) reads
  `load_config()["display_unit"]` into the fresh graph state, so the chat/CLI
  flow carries it (checkpointer preserves it across resume). `src/ui/app.py`
  reads it once and passes it to `trends_tab.render` / `backfill_tab.render`.
- **Dev Tools** grid stays lb by design, with a caption saying so.

### 11b — Block-review / form-cue embedding paths

- **`src/ingest/embed.py`**: `embed_review(text, doc_id, doc_type, *, date,
  block_id, program_id, exercises, embedder, client)` — single-document
  (unchunked), idempotent on `doc_id`, **blank text deletes the doc** (clearing a
  review un-embeds it). Constants `BLOCK_REVIEW_DOC_TYPE` / `PROGRAM_REVIEW_DOC_TYPE`
  / `FORM_CUE_DOC_TYPE`; id helpers `block_review_id` / `program_review_id`.
  Metadata carries `session_id=0`, `doc_type`, `block_id`, `program_id`,
  `exercises`, `date`/`date_ordinal` (so `search_notes`' existing filters work).
- **`src/tools/organize.py`**: `set_block_review` / `set_program_review` write
  `review_text` to SQLite (committed first), then embed via the seams;
  `get_block_review` / `get_program_review` read it back. `embed=False` skips
  Chroma (used by the offline test). Chroma import is lazy so organize.py stays
  import-light.
- **Organizer tab**: a "Reviews" section — pick a block/program, edit its review
  in a text area, save. `_save_review` commits then embeds, so an embed failure
  (Ollama down) is a *warning* (prose already saved), not a loss.
- **Chat path** (extends UPDATE_STATS — **no new graph topology**): `StatUpdate`
  gained kinds `block_review` / `form_cue` and a `text` field. The parse node
  resolves a form cue's exercise and attaches a block review to `_latest_block`
  (most recently dated). The confirm node embeds the cue / calls
  `set_block_review` on "yes". `make_update_stats_confirm_node` now takes
  `embedder` / `chroma_client` / `embed_reviews`, wired from `build_graph`
  (`embed_reviews=True`, gated off in offline tests). The router prompt lists
  reviews/cues under `update_stats`.
- Retrieval works for free: `search_notes(doc_type='block_review'|'form_cue')`
  and GENERATE's `search_training_notes` already filter `doc_type`.

### 11c — Re-embedding command

- **`src/ingest/reembed.py`**: `reembed_collection(client, embedder, name,
  new_embedder_name)` reads `collection.get()`, re-embeds the stored `documents`,
  and does a **build-then-swap** (populate `<name>__reembed` fully → drop
  original → recreate → copy → drop temp), stamping `{"embedder": name}` into
  collection metadata. `reembed_all(...)` iterates `personal_notes` + `knowledge`,
  skipping any that don't exist; `collection_embedder(client, name)` reads the
  stamp. `embedder_name(node)` in `embed.py` reads the configured model name.
- **`/reembed`** CLI command (`run_reembed` in `src/cli.py`). No source
  re-parsing needed — documents are all stored in Chroma.

### Tests (25 new, 278 total)

- `tests/ingest/test_review.py`: kg rendering (sets + slots convert; default lb).
- `tests/test_trends.py`: frame-builder kg conversion (tonnage yes, hard_sets no,
  source tooltip carries unit); `weight_label`.
- `tests/ingest/test_embed.py`: `embed_review` upsert / blank-deletes / metadata.
- `tests/test_organize.py`: `set_block_review` / `set_program_review` (+ getters,
  blank-clears, `embed=False`, unknown-id raises) with fake embedder + in-memory
  Chroma.
- `tests/agent/test_graph_update_stats.py`: form-cue and block-review chat flows
  (confirm → embed / SQLite write); block-review declines with no blocks.
- `tests/ingest/test_reembed.py`: round-trip preserves ids/docs/metadata, stamps
  embedder, leaves no temp collection, handles empty + missing collections.
- `tests/test_cli.py`: `run_reembed` reports counts (seams monkeypatched).

## Decisions made this step (now locked in)

1. **kg scope: all user-facing surfaces** (chat, HITL reviews/confirms, generate
   drafts, Trends, backfill). Dev Tools grid excluded — it edits canonical lb.
2. **Review/form-cue entry: both** the Organizer tab and a chat path.
3. **Chat routing: extend `update_stats`** (new kinds) rather than a new router
   intent — reuses the parse→confirm→write wiring untouched.
4. **Re-embed surface: CLI-only** (`/reembed`); re-embedding needs a live
   embedder and can be slow, so a blocking Streamlit button was rejected.
5. **Reviews are single-doc, not chunked.** Simpler idempotent upsert/delete on a
   fixed id; reviews here are short. If reviews ever get long enough to want
   per-section chunking, delete-then-re-add all chunks for that id (stale-chunk
   cleanup is why single-doc was chosen first).
6. **Block reviews from chat attach to the latest block.** The Organizer tab is
   the path for reviewing an *older* block by name.

## Implementation notes / gotchas for whoever builds on this

- **Presentation-only conversion is load-bearing.** Every kg path converts at the
  very end (renderer / frame builder / confirm prompt). Never convert before
  storage, and never write a converted value back to SQLite/Chroma.
- **`_fmt_weight` still returns `"BW"`** for a `None` weight (bodyweight-only
  set) regardless of unit — don't route that through `format_weight`.
- **`set_block_review` commits SQLite before embedding** on purpose: a dead
  embedder must not lose the prose. Callers should treat an embed exception as a
  warning, not a failed save (the Organizer tab does).
- **`embed_reviews=False`** on `build_graph` disables the Chroma writes in the
  UPDATE_STATS confirm node for offline tests; the SQLite `review_text` write
  (block reviews) still happens via `set_block_review(embed=False)`.
- **reembed is build-then-swap** so a crash can't leave a half-embedded
  collection; if you change it, preserve that ordering (temp fully built before
  the original is dropped).
- The preview `launch.json` runs against the **live** `data/training.db`.
  Verified the kg pass live by temporarily flipping `display_unit: kg` (Trends
  metrics showed kg, axes `(kg)`), then restoring lb.

## Must handle / preserve (carried forward, still true)

- No live models in tests — every LLM/embedder/Chroma dependency stays behind an
  injectable seam; tests use stubs/fakes/in-memory clients.
- HITL invariant: nothing durable to `session`/`lift_set`/`pr`/Chroma outside an
  approved interrupt — the Dev Tools CRUD, bulk-backfill commit-all, knowledge
  ingest, and the *user-initiated* Organizer review save are the explicit
  user-action exemptions. The chat review/cue path goes through the confirm
  interrupt.
- lb canonical; conversion only at presentation.
- Draft programs excluded from analysis by default.
- Seeder touches `data/sample.db` only; `data/training.db` is live; checkpoints
  in `data/checkpoints.db`.
- Never hard-require an API key when every node is `provider: local`.

## Ideas, if you want more (nothing load-bearing)

- Per-section chunking for long reviews (see decision 5).
- A privacy/anonymized-summary mode for cloud nodes (ARCHITECTURE.md §6.3,
  deferred at Stage 7).
- Injury/measurement capture in the chat UPDATE_STATS path (still bodyweight /
  PR / review / cue only).
- A Trends unit toggle in the tab itself (currently config-driven only).
