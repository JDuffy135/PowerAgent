# Handoff — Step 7: GENERATE (program writer) + cloud offload

For the next Claude session. Read `ARCHITECTURE.md` (full design) and the rest of
this file for what's built. `IMPLEMENTATION_ROADMAP.md` has the remaining-work
plan; **Stage 8 (xlsx/pdf loaders + knowledge-base ingestion) is next** (it was
independent of 5–7 all along), then the optional Stage 9 polish.

## Where things stand (Steps 1–7 — DONE, 171 tests passing)

Steps 1–6 unchanged (data spine, LLM extraction, HITL staging/commit, full tool
layer, LangGraph core + CLI, ANALYZE/SYNTHESIZE/UPDATE_STATS). Step 7 adds the
program-writing branch and the cloud provider:

- **Cloud provider branch** — `provider: cloud` now works in both LLM seams:
  - `src/agent/llm_provider.py::get_chat_model` → `langchain_anthropic.ChatAnthropic`
    (lazy import), model default `claude-sonnet-5`, `max_tokens` config key
    (default 16000).
  - `src/ingest/extract.py::get_llm` → `_cloud_llm`: the `anthropic` SDK
    (`_anthropic_client(api_key)` is the module-level seam tests monkeypatch
    with a fake). The target JSON schema is **embedded in the system prompt**,
    not passed as API-enforced structured output — arbitrary Pydantic schemas
    (defaults, no `additionalProperties: false`) aren't guaranteed
    strict-schema-compatible, and downstream Pydantic validation is the real
    contract of this seam anyway (same as the Ollama `format` path). Markdown
    fences are stripped defensively; `stop_reason == "refusal"` / empty text
    raise RuntimeError.
  - API key: env var named by `nodes.<node>.api_key_env` (default
    `ANTHROPIC_API_KEY`), resolved at **build time** via `extract.get_api_key`
    so a missing key fails fast with the variable name in the message. Never
    required when `provider: local`. Unknown providers now raise `ValueError`
    (the old `NotImplementedError` is gone).
  - Deps added: `langchain-anthropic`, `anthropic` (pyproject + installed in
    `.venv`).
- **`src/agent/nodes/generate.py`** — two nodes, split per the interrupt-replay
  contract:
  - `generate`: (1) bounded ReAct evidence loop (`MAX_TOOL_CALLS = 8`) over
    `make_analyze_tools` — same tool set as ANALYZE, node-local scratch, results
    accumulated into `state["evidence"]`; the gather prompt tells the model to
    always check `get_injuries(active_only=true)`, e1RM/volume trends, past
    block outlines, and notes. (2) A structured draft call through the raw
    `get_llm("generate", system_prompt=..., schema=DraftProgram schema)` seam:
    user request + formatted evidence (per-item cap 2000 chars) → JSON →
    `DraftProgram`. Invalid JSON / schema failure / zero slots → error AIMessage,
    `pending_draft=None`, END (no interrupt). Success → rendered draft emitted
    as an AIMessage (so the CLI prints it before the interrupt) and
    `pending_draft` stashed (as a dict — LangGraph state stays JSON-able).
  - `generate_confirm`: `interrupt()` at the very top ("save this draft?
    yes/no"); yes → `persist_draft`; no → discard; anything else → reask
    (loop-back edge, one-shot `review_note`) — same shape as
    `update_stats_confirm`.
- **Draft models** — `DraftSlot`/`DraftProgram` in `generate.py`, mirroring
  `ParsedProgrammedSlot` (**[DECISION]** structured output, machine-insertable):
  `exercise` stays a raw string resolved at save time; `week_number`/`day_number`
  required on every slot; `prescription` free text; `target_weight_lb` canonical
  lb. `render_draft` groups slots week → day and formats targets via
  `format_weight` (display-unit aware).
- **`src/tools/draft.py::persist_draft`** — the flow's only durable write:
  one transaction inserting `program(status='draft')` + `block(focus,
  week_count)` + `programmed_slot` rows; slot exercises resolve best-effort
  (`exercise_id=NULL` when unresolved, names reported in
  `DraftSaveResult.unresolved_exercises` and surfaced in the confirm summary).
  Rollback on any failure.
- **Guardrails** — `generate.TRAINING_PHILOSOPHY` encodes the user's 8-point
  philosophy (ramping-RPE 4-week SBD waves, weak-point variations except in
  peaking blocks, 4-5 days/wk, injury workarounds, ~10-set accessory default
  counting only 7+-rep SBD sets, 7-9/8-10/10-15 weekly DL/SQ/BP starting sets,
  2x SQ/DL + 3x BP frequency). Injected into **both** the gather and draft
  prompts. Edit the constant, not the prompts, to tune philosophy.
- **`src/agent/graph.py`** — placeholder gone; registered `generate` +
  `generate_confirm` with factories `generate_model_factory` (chat, ReAct) and
  `generate_llm_factory` (raw, draft; `None` → config default). Edges:
  `generate` conditional on `pending_draft` (`generate_confirm` | END);
  `generate_confirm` conditional on `review_decision` (`reask` loop | `done`).
- **`src/agent/state.py`** — added `pending_draft`; **`src/cli.py`** resets it
  per turn and the banner now mentions program generation.
- **`config.yaml`** — new `nodes.generate`: `provider: cloud`,
  `model: claude-sonnet-5`, `api_key_env: ANTHROPIC_API_KEY` (**[DECISION]**
  cloud by default; flipping back to local is a config edit).
- Tests: `tests/agent/test_graph_generate.py` (7 tests: evidence loop + draft
  interrupt, approve persists + resolves aliases + flags unresolved, reject
  writes nothing, reask loop, invalid JSON / empty slots end without interrupt,
  saved drafts visible only via `get_programs('draft')`); cloud fake-transport
  tests in `tests/ingest/test_get_llm.py` + `tests/agent/test_llm_provider.py`
  (missing-key raises naming the env var, custom `api_key_env`, request shape,
  fence stripping, unknown-provider ValueError). The Stage 5 placeholder
  routing test was removed with the placeholder.

## Decisions made this step (given by the user, now locked in)

1. **Cloud vendor + SDK:** Anthropic API, default `claude-sonnet-5`;
   `langchain-anthropic` for chat models, `anthropic` SDK for the raw seam;
   key env var configurable per node (`api_key_env`, default
   `ANTHROPIC_API_KEY`).
2. **Privacy option:** deferred — cloud nodes see raw evidence for now.
3. **Draft output format:** structured (`ParsedProgrammedSlot`-style Pydantic),
   directly insertable into `programmed_slot`.
4. **Guardrails:** the 8-point training philosophy above, encoded verbatim in
   `TRAINING_PHILOSOPHY` (see `IMPLEMENTATION_ROADMAP.md` Stage 7 decisions for
   the full text).

## Implementation notes / gotchas for whoever builds on this

- **The `interrupt()`-replays-the-node rule still governs.** All LLM work
  (ReAct loop + draft call) lives in `generate`, which has no interrupt;
  `generate_confirm`'s interrupt is its first statement. Keep this shape.
- **`pending_draft` crosses the interrupt as a dict** (`model_dump()`), and is
  re-validated with `DraftProgram.model_validate` on the confirmed branch. Don't
  stash Pydantic objects in state.
- **Cloud structured output is prompt-enforced, not API-enforced.** If cloud
  extraction/drafting ever proves flaky on JSON validity, the upgrade path is a
  strict-schema sanitizer + `output_config.format` in `_cloud_llm` — the seam
  signature doesn't change.
- **`get_api_key` runs at builder time**, so `build_graph` with default
  factories does NOT hit it (factories are lazy, resolved at node-call time) —
  a keyless local-only setup only fails if a turn actually routes to a cloud
  node.
- **Draft failure paths set `review_decision="none"`** (informational); the
  conditional edge keys on `pending_draft` truthiness, same idiom as
  UPDATE_STATS.
- **`make_placeholder_node`** still exists in `nodes/chitchat.py` but has no
  call sites; reuse or delete at will.
- The GENERATE evidence loop reuses `state["evidence"]` — a turn that runs
  GENERATE overwrites ANALYZE's evidence from a prior turn (fine: CLI resets
  scratch per turn).

## Must handle / preserve (carried forward, still true)

- No live models in tests — every LLM/embedder/Chroma dependency stays behind an
  injectable seam (`get_llm`, `get_chat_model`/model factories, `get_embedder`,
  `get_chroma_client`, `_anthropic_client`). Graph tests use `InMemorySaver` +
  stubs; cloud tests use fake clients; Chroma tests use the fake embedder +
  `EphemeralClient`.
- HITL invariant: nothing durable outside an approved interrupt branch —
  `commit_batch`/`reject_batch` (ingest), `insert_bodyweight`/`insert_pr`
  (update_stats), `embed_analysis` (store_offer), `persist_draft`
  (generate_confirm).
- lb canonical; unit conversion only at presentation (SYNTHESIZE / confirm
  prompts / `render_draft`).
- Draft programs excluded from every analysis tool by default; `get_programs`
  is the only surface that lists them.
- Seeder only touches `data/sample.db`; `data/training.db` is live; checkpoints
  in `data/checkpoints.db`.
- Never hard-require an API key when `provider: local`.

## What comes after (context only — do NOT build now)

Per `IMPLEMENTATION_ROADMAP.md`:
- **Stage 8** — xlsx/pdf loaders (`parse_upload` branches), knowledge-base
  ingestion into the Chroma `knowledge` collection, and the `search_knowledge`
  tool. When done, register `search_knowledge` in `make_analyze_tools` (both
  ANALYZE and GENERATE pick it up automatically — they share the toolset).
  Read that stage's "⚠️ Decisions" first (chunker, ingest UX, xlsx layout
  fixture).
- **Stage 9 (optional)** — Streamlit/Gradio UI, historical backfill, kg
  end-to-end pass, ops niceties, and the program/block organizer (committed to
  during Stage 5). A natural Stage 9 addition now: "start this draft" — flip a
  draft program to `incomplete` and attach sessions as they're logged.
