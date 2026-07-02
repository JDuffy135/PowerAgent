# Handoff — Step 5: LangGraph Core (router, INGEST HITL flow, checkpointer, CLI)

For the next Claude session. Read `ARCHITECTURE.md` (full design) and the rest of
this file for what's built. `IMPLEMENTATION_ROADMAP.md` has the remaining-work
plan (Stages 6–9); Stage 6 (ANALYZE + SYNTHESIZE + UPDATE_STATS) is next.

## Where things stand (Steps 1–5 — DONE, 141 tests passing)

**Step 1 — data spine:** `src/db/schema.sql` + `connection.py`,
`resolve_exercise`/`add_exercise`, `get_best_set`/`get_lifts`/`get_e1rm_trend`/
`get_bodyweight_trend`, `src/seed.py` (targets `data/sample.db`).

**Step 2 — LLM extraction:** `ParsedBatch` models, `.txt` loader,
`extract_training_data` with the `get_llm(node)` provider seam.

**Step 3 — HITL staging + commit:** `stage_batch`/`get_pending_batch`,
`render_batch`, transactional `commit_batch`/`reject_batch`, Chroma
`personal_notes` embedding via `get_embedder()`/`get_chroma_client()`.

**Step 4 — full tool layer:** the rest of the typed query tools
(`src/tools/queries.py`), `search_notes` (`src/tools/vector.py`),
`run_readonly_sql` (`src/tools/sql.py`), PR auto-derivation
(`find_recent_prs`/`commit_prs`), extended seeder.

**Step 5 — LangGraph core** (this step):
- **Deps added:** `langgraph`, `langgraph-checkpoint-sqlite`, `langchain-ollama`.
- **`src/agent/llm_provider.py`** — the shared provider (§6.3): re-exports the
  raw `get_llm(node, system_prompt=, schema=)` callable seam (still implemented
  in `src/ingest/extract.py`, which gained the two kwargs; existing tests
  untouched) and adds `get_chat_model(node)` → `ChatOllama` for graph nodes.
  Any `provider != local` raises `NotImplementedError` until Stage 7.
- **`src/agent/state.py`** — `AgentState` (§4.3) plus ingest-flow plumbing:
  `file_path`, `correction_rounds`, `review_decision`, `review_note`.
- **`src/agent/nodes/router.py`** — intent classification into
  `ingest/analyze/generate/update_stats/chat` via JSON structured output;
  falls back to `chat` on malformed output; **skips the LLM entirely when
  `intent` is preset** (the CLI's `/ingest` does this).
- **`src/agent/nodes/ingest.py`** — the INGEST pipeline as THREE nodes
  (`ingest_parse` → `ingest_review` → `ingest_commit`), because `interrupt()`
  replays its node from the top on resume: parse/extract/stage happen once in
  `ingest_parse`; each review round and each block-assignment retry is a fresh
  node execution via loop-back conditional edges.
- **`src/ingest/correct.py`** — correction pass: full `ParsedBatch` re-emit with
  original JSON + user text in the prompt, re-validated, exercise ids
  re-resolved (stale candidates dropped, new unresolved names get candidates).
  `src/ingest/stage.py` gained `update_batch` (pending rows only —
  `BatchNotEditable` otherwise).
- **`src/ingest/commit.py`** — `commit_batch(..., block_id=None)`: with a block,
  sessions attach to it and programmed slots are **inserted** (best-effort
  exercise resolution; unresolvable slot names insert with NULL exercise_id —
  `programmed_slot.exercise_id` is nullable, only sets are strict). Without a
  block, prior behavior (slots skipped, preserved in audit JSON).
  `CommitResult` gained `programmed_slots_created`; `UnknownBlock` raises before
  anything is written.
- **`src/agent/graph.py`** — topology + `SqliteSaver` at `data/checkpoints.db`
  (config key `checkpoints_db`); ANALYZE/GENERATE/UPDATE_STATS are placeholders
  replying "not implemented yet". All model deps injected via factories,
  resolved lazily — building the graph never touches Ollama.
- **`src/cli.py`** — `python -m src.cli` REPL; `/ingest <path>` command;
  `run_turn` services the interrupt/resume round-trips (`Command(resume=...)`).
- **`config.yaml`** gained `checkpoints_db` + `router`/`chitchat`/
  `ingest_correct` node entries.
- Tests: `tests/agent/` (router, provider, full graph flows with stub LLMs +
  `InMemorySaver`), `tests/ingest/test_correct.py`, block-assignment additions
  to `tests/ingest/test_commit.py`, `tests/test_cli.py`.

## Decisions made this step (given by the user, now locked in)

1. **LangChain coupling:** `langchain-ollama`/`ChatOllama` for the agent graph
   (Stage 6 tool-calling needs a `BaseChatModel`); the raw-`urllib` callable
   seam stays for extraction/correction. `llm_provider` is the canonical import.
2. **Correction contract:** full re-emit (original JSON + user text in prompt),
   loop capped at `MAX_CORRECTION_ROUNDS = 5`; at the cap only approve/reject.
3. **Block assignment at ingest:** option (a) — after approval, an interrupt
   asks: existing block id / `new <program> / <block>` (created on the fly,
   program status `incomplete`, `start_date` = earliest session date; program
   names are matched case-insensitively and reused) / `none` (unattached).
   **The user must be able to review/reorganize programs & blocks later** —
   that organizer is now an explicit Stage 9 roadmap item, so ingest-time
   mistakes are recoverable by design.
4. **File intake:** CLI command `/ingest <path>` for now; the future UI gives
   drag-and-drop / a file picker but reuses the same graph input shape
   (`intent='ingest'` + `file_path` preset), so only the front-end changes.
5. **Checkpointer:** separate `data/checkpoints.db` (`SqliteSaver`).

## Implementation notes / gotchas for whoever builds on this

- **`interrupt()` replays the whole node on resume.** That's why INGEST is three
  nodes and why every interrupt sits at the very top of its node, with loop-back
  edges (`ingest_review → ingest_review`, `ingest_commit → ingest_commit`) for
  the correction/invalid-reply loops. If you add interrupts in Stage 6
  (UPDATE_STATS confirm, SYNTHESIZE "store this?"), follow the same shape —
  never put an LLM call before `interrupt()` in the same node.
- **`review_note` is a one-shot state field** prepended to the next interrupt
  prompt (correction failures, invalid block replies, cap notice). Every ingest
  node return path must set it (usually to `None`) or a stale note leaks into
  the next prompt.
- **Conditional edges route on `review_decision`**, set by every ingest node
  return. The router fans out on `intent`; `intent or "chat"` guards against a
  None slipping through.
- **Graph tests drive interrupts** with `graph.invoke(Command(resume=...),
  config)` and assert on `result["__interrupt__"][0].value` (a dict with
  `kind` (`ingest_review` | `block_assign`), `batch_id`, `prompt`). The CLI
  prints `payload["prompt"]` — keep that key if you add interrupt kinds.
- **The CLI sets `recursion_limit: 100`** — each correction round costs graph
  steps; the default 25 could plausibly be hit by a legitimate review session.
- **`StubChatModel`/`scripted_llm`/`golden_batch`** in `tests/agent/conftest.py`
  are the stubbing kit for Stage 6's ReAct-loop tests (a stub emitting tool
  calls will need a richer object, but keep the same no-live-models rule).
- Placeholder nodes (`analyze`, `generate`, `update_stats`) are registered in
  `graph.py` via `make_placeholder_node` — Stage 6/7 replace those
  registrations; the topology/edges are already in place.
- `_parse_review_reply` treats a small word list as approve/reject and
  *everything else* as a correction; a typo'd "aprove" becomes a correction
  round. Harmless (the correction pass no-ops or the user re-replies) but worth
  knowing when testing by hand.

## Must handle / preserve (carried forward, still true)

- No live models in tests — everything LLM/embedder/Chroma-dependent stays
  behind an injectable seam (`get_llm`, `get_chat_model` factories,
  `get_embedder`, `get_chroma_client`). Graph tests use `InMemorySaver`.
- HITL invariant: nothing durable happens outside `commit_batch`/`reject_batch`
  (plus `_create_program_and_block`, which runs only inside the approved-commit
  interrupt flow).
- lb canonical; unit conversion only at presentation (SYNTHESIZE, Stage 6).
- Draft programs excluded from every analysis tool by default.
- Seeder only ever touches `data/sample.db`; `data/training.db` is live;
  checkpoints live in `data/checkpoints.db`.

## What comes after (context only — do NOT build now)

Per `IMPLEMENTATION_ROADMAP.md`:
- **Stage 6** — ANALYZE ReAct loop + SYNTHESIZE + UPDATE_STATS: register the
  Stage 1/4 tools as LangGraph tools (tight, small-model-friendly schemas),
  bounded ReAct loop accumulating `evidence`, unit conversion at presentation,
  confirm-before-write stat updates. Read that stage's "⚠️ Decisions" first
  (ANALYZE model choice, evidence overflow policy, "store this analysis?"
  interrupt, UPDATE_STATS scope).
- **Stage 7** — GENERATE program writer + cloud provider branch in
  `llm_provider.py` (the `provider: cloud` raise is the seam to fill).
- **Stage 8** — xlsx/pdf loaders + knowledge-base ingestion + `search_knowledge`
  (independent; can run any time).
- **Stage 9 (optional)** — UI, backfill, ops polish, and the program/block
  organizer promised by this step's block-assignment decision.
