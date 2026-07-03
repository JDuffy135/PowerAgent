# Handoff — Step 9: Streamlit UI, organizer, historical backfill, dev tools

For the next Claude session. Read `ARCHITECTURE.md` (full design) and the rest
of this file for what's built. `IMPLEMENTATION_ROADMAP.md` is now **fully
complete** (Stages 1–9); only a few explicitly-deferred polish items remain
(listed at the bottom).

## Where things stand (Steps 1–9 — DONE, 240 tests passing)

Steps 1–8 unchanged (data spine, LLM extraction, HITL staging/commit, full tool
layer, LangGraph core + CLI, ANALYZE/SYNTHESIZE/UPDATE_STATS, GENERATE + cloud
offload, file loaders + knowledge base). Step 9 adds the browser UI and the
organizational/ops layer around it:

- **`src/ui/`** — Streamlit app, `streamlit run src/ui/app.py`. Same resources
  as the CLI (live `training.db`, `checkpoints.db` SqliteSaver, compiled
  graph), wired in `app.py` behind `st.cache_resource`; four tabs:
  - **Chat** (`chat_tab.py`): the REPL in a browser. The interrupt round-trip
    spans Streamlit reruns instead of blocking on `input()` — the logic lives
    in the **streamlit-free** `src/ui/driver.py` (`drive_turn` /
    `resume_payload` / frozen `TurnResult`). Quick-reply buttons send
    `yes`/`no` (tokens accepted by *all three* confirm parsers: ingest review
    `_APPROVE_WORDS`/`_REJECT_WORDS`, update-stats `_YES_WORDS`, generate
    `_YES_WORDS`/`_NO_WORDS`); free text still handles corrections and block
    picks. Sidebar uploader saves to `data/uploads/` then sends `/ingest
    <path>` through `make_input` (graph input carries a path, not bytes) or
    embeds via `ingest_knowledge_file`. `chat_printed` (message count consumed
    on the thread) persists across the whole session and is updated after
    every call — do NOT reset it per turn; messages accumulate on the
    checkpointer thread.
  - **Organizer** (`organizer_tab.py`): thin veneer over
    **`src/tools/organize.py`** (new) — `list_programs/list_blocks/
    list_sessions` (draft-inclusive; this is the organizer, not analysis),
    `rename_program/rename_block`, `reattach_session` (FK move only, sets ride
    along), `move_block`, `merge_blocks` (sessions + programmed_slots move,
    src row deleted), `merge_programs` (blocks move, src prose dropped), and
    `start_draft` (the "start this draft" flow: `draft` → `incomplete` +
    start date). All raise `OrganizeError` on bad preconditions; merges are
    transactional (rollback on failure).
  - **Backfill** (`backfill_tab.py`): front-end for
    **`src/ingest/backfill.py`** (new). `split_archive(text, max_chars=6000)`
    cuts an archive along session boundaries — new block at date/week-label
    line starts (`_SESSION_START` regex: ISO dates, US dates, `w3d2`-style) or
    blank-line gaps followed by a short header line — then greedily packs
    sessions into chunks; a single oversized session hard-splits at line
    boundaries. `run_backfill` pipes each chunk through the SAME pipeline
    (`extract_training_data → stage_batch → commit_batch`), so every chunk
    gets an `ingest_batch` audit row. `auto_commit=False` (default) stages
    everything `pending_review`; the tab's "Pending batches" section renders
    each batch (`render_batch`) with per-batch commit/reject + optional block
    attach. `auto_commit=True` is the relaxed bulk mode. Failed chunks are
    isolated (`ChunkResult.status='failed'`, error captured), never abort the
    run. Paste box or file upload (txt/xlsx/pdf via `parse_upload`).
  - **Dev Tools** (`devtools_tab.py`): front-end for **`src/tools/admin.py`**
    (new) — allowlisted-table CRUD (`EDITABLE_TABLES` maps table → pk;
    `ingest_batch` deliberately absent = read-only audit), `backup_db` (SQLite
    online backup API → `data/backups/<prefix>-backup-<utc>.db`), and
    `list_batches`/`get_batch_json` for the audit browser. The editable grid
    is `st.data_editor(num_rows="dynamic")`; "Apply changes" diffs edited vs
    loaded rows via the **streamlit-free** `src/ui/editing.py::diff_rows`
    (blank-pk row → insert, missing pk → delete, changed cells → update with
    only the changed columns; NaN normalized to None so pandas NaN never hits
    SQLite and NULL↔NaN doesn't produce spurious updates).
- **`src/db/connection.py`** — `get_conn(..., check_same_thread=False)` param
  added for the UI (Streamlit reruns hop threads; access is serialized).
- **`.claude/launch.json`** — dev-server config for previewing the UI.
- Deps added: `streamlit>=1.40`, `pandas>=2.0`.
- Tests (42 new, 240 total): `tests/test_organize.py` (13),
  `tests/test_admin.py` (8), `tests/ingest/test_backfill.py` (11, stub
  extraction LLM), `tests/test_ui.py` (10: driver with a fake graph incl.
  interrupt/resume round-trip, `diff_rows` cases, tab-module import smoke).

## Decisions made this step (given by the user, now locked in)

1. **Streamlit** (over Gradio), with the user's tab spec — Chat / Organizer /
   Dev Tools — plus a dedicated **Backfill** tab (it's a data-entry workflow,
   not a dev tool).
2. **Bulk-backfill HITL**: stage-for-review keeps per-batch approval; "Commit
   everything" relaxes it to one explicit approval for the whole run, still
   fully audited.
3. **Dev-tools CRUD bypasses the HITL interrupt flow by design** — every write
   is an explicit hand edit and the surface is never exposed to the LLM.
4. **Block-review/form-cue ingestion stays unbuilt** (was Stage 9's second open
   decision — deferred).

## Implementation notes / gotchas for whoever builds on this

- **UI logic vs. rendering split**: anything decision-shaped lives in
  streamlit-free modules (`driver.py`, `editing.py`, plus `organize.py` /
  `admin.py` / `backfill.py`) with unit tests; the `*_tab.py` modules are
  veneers and only import-smoke-tested. Keep new UI logic on the testable side
  of that line.
- **`drive_turn`'s `printed` contract**: pass the previous `TurnResult.printed`
  back on every call on the same thread. The CLI's `run_turn` resets its
  counter per turn because it re-reads the full result each invoke within one
  loop; the UI must NOT reset across turns or old replies re-render.
- **The chat quick-reply tokens must stay `yes`/`no`.** If a confirm parser's
  word sets ever change, keep `yes`/`no` in all of them or update
  `chat_tab.py`.
- **`st.cache_resource` shares conn+graph across browser sessions**; per-session
  state (thread_id, history, pending-interrupt flag) lives in
  `st.session_state`. A second browser tab = new thread on the same graph.
- **Backfill chunking is heuristic** (`_SESSION_START` + short-header rule).
  If the user's real archive has an exotic layout, tune the regex — the
  greedy packer and coverage tests (`test_split_coverage_no_lost_lines`) make
  regressions visible.
- **`diff_rows` treats a hand-typed pk on a new editor row as an insert with
  explicit pk**; SQLite honors explicit INTEGER PRIMARY KEY values.
- **Deleting a parent row in Dev Tools fails with `AdminError`** (FK ON) —
  children first. This is intentional; no cascade.
- The preview `launch.json` runs against the **live** `data/training.db` — the
  UI is the production surface, not a sandbox.

## Must handle / preserve (carried forward, still true)

- No live models in tests — every LLM/embedder/Chroma dependency stays behind
  an injectable seam. Backfill tests inject a stub extraction LLM; UI driver
  tests use a fake graph object.
- HITL invariant unchanged for the *agent's* write paths: nothing durable to
  `session`/`lift_set`/`pr`/etc. outside an approved interrupt branch. The two
  Stage 9 exemptions are explicit user-action surfaces: dev-tools CRUD
  (decision 3) and bulk-backfill commit-all (decision 2, one up-front
  approval, fully audited). Knowledge ingest remains exempt per Stage 8.
- lb canonical; unit conversion only at presentation.
- Draft programs excluded from every analysis tool by default (`get_programs`
  and the organizer listings are the draft-inclusive exceptions).
- Seeder only touches `data/sample.db`; `data/training.db` is live; checkpoints
  in `data/checkpoints.db`.
- Never hard-require an API key when `provider: local`.

## What's left (deferred polish — pick up only if the user asks)

- `display_unit: kg` end-to-end pass (config exists; SYNTHESIZE converts, but
  the UI/tools don't surface kg everywhere).
- Block-review / form-cue embedding paths (`doc_type='block_review'|'form_cue'`
  are in the design; nothing writes them).
- Re-embedding command for embedder swaps (§3.2: changing embedders requires
  re-embedding both collections).
