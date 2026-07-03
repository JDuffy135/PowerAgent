# Handoff — Step 6: ANALYZE ReAct loop + SYNTHESIZE + UPDATE_STATS

For the next Claude session. Read `ARCHITECTURE.md` (full design) and the rest of
this file for what's built. `IMPLEMENTATION_ROADMAP.md` has the remaining-work
plan; **Stage 7 (GENERATE program writer + cloud provider branch) is next.**

## Where things stand (Steps 1–6 — DONE, 161 tests passing)

Steps 1–5 unchanged (data spine, LLM extraction, HITL staging/commit, full tool
layer, LangGraph core + CLI). Step 6 adds the query-answering and stat-recording
branches of the agent graph:

- **`src/agent/tools.py`** — `make_analyze_tools(conn, embedder=, chroma_client=)`
  wraps every Step 1/4 query tool as a LangChain `@tool` (StructuredTool) with the
  DB connection closed over (never in the LLM-facing schema). Tight schemas:
  explicit `YYYY-MM-DD` dates, `Literal` enums (`by='week'|'block'`, statuses,
  session types). Each returns a **plain JSON-able dict** (`_dump` `model_dump()`s
  Pydantic results); `ExerciseNotFound` and SQL/vector errors are caught and
  returned as `{"error": ...}` so the model recovers instead of crashing the turn.
  Tools: the 13 query tools + `search_training_notes` (Chroma) + `run_sql`
  (gated escape hatch).
- **`src/agent/nodes/analyze.py`** — bounded ReAct loop. `model_factory().bind_tools(tools)`,
  then up to `MAX_TOOL_CALLS = 8` model turns: execute requested tool calls,
  append `ToolMessage`s to a **node-local** scratch list, accumulate each result
  into `state["evidence"]`. The ReAct scratch is NOT written back to
  `state["messages"]` (keeps router/history clean). No `interrupt()` here, so the
  loop runs inside one node safely. Loop exhausted while still requesting tools →
  `evidence_truncated=True`. Injects today's date into the prompt (`today=` param
  for test determinism).
- **`src/agent/nodes/synthesize.py`** — two nodes:
  - `synthesize`: composes the answer from `evidence` via the model; **the single
    unit-conversion point** (`src/agent/units.py` converts lb→`display_unit`
    before the evidence hits the prompt). Empty evidence short-circuits to a fixed
    "couldn't find data" message with **no model call**. `evidence_truncated`
    appends `OVERFLOW_DISCLAIMER` (partial-data + narrow-your-scope). Sets
    `offer_store=True` only when there's real evidence; stashes `analysis_text`.
  - `store_offer`: `interrupt()` at the very top asking "store this? yes/no"; on
    yes, `embed_analysis(analysis_text, ...)` into Chroma. Interrupt-before-any-
    durable-work, per the Stage 5 replay contract.
- **`src/agent/units.py`** — `to_display_weight`/`format_weight`/`convert_weights`.
  `convert_weights` deep-walks a dict, converting values under keys ending `_lb`
  (renamed to drop the suffix) or `e1rm`; bools guarded. Known limitation:
  `BodyweightTrend`'s summary fields (`first/last/delta/min/max`) lack the `_lb`
  suffix so they don't auto-convert (rows do); noted in the module docstring.
- **`src/agent/nodes/update_stats.py`** — two nodes (parse + confirm), split for
  the same replay reason as INGEST (LLM parse must run once, before the interrupt):
  - `update_stats_parse`: `get_llm("update_stats", system_prompt=, schema=StatUpdate...)`
    (raw JSON seam) → `StatUpdate` (kind ∈ `bodyweight`|`pr`|`none`, weight
    normalized to lb by the prompt). Resolves the PR exercise at parse time;
    unknown exercise / out-of-scope / unreadable → `_decline` (AIMessage, no
    interrupt, `review_decision="none"`). Otherwise stashes `pending_stat`.
  - `update_stats_confirm`: `interrupt()` with a `display_unit`-formatted summary;
    yes → `insert_bodyweight`/`insert_pr`; no → discard; anything else → reask
    (loop-back edge, one-shot `review_note`).
- **`src/tools/stats.py`** — `insert_bodyweight` / `insert_pr` (the durable writes,
  each commits; only called on the confirmed branch).
- **`src/ingest/embed.py`** — added `embed_analysis(text, date=, ...)` +
  `ANALYSIS_DOC_TYPE = "analysis"`; unique id `analysis_<ms>`, `session_id=0`.
- **`src/agent/graph.py`** — registered `analyze`, `synthesize`, `store_offer`,
  `update_stats_parse`, `update_stats_confirm`; `generate` remains the only
  placeholder. New injectable factories: `analyze_model_factory`,
  `synthesize_model_factory`, `update_stats_llm_factory`, plus `embed_analyses`
  flag (mirrors `embed_prose`). Edges: `analyze → synthesize`; `synthesize`
  conditional on `offer_store` (`store_offer` | END); `store_offer → END`;
  `update_stats_parse` conditional on `pending_stat`; `update_stats_confirm`
  conditional on `review_decision` (`reask` loop | `done`).
- **`src/agent/state.py`** — added `evidence_truncated`, `analysis_text`,
  `offer_store`, `pending_stat`. Reuses `review_decision`/`review_note` for the
  stat-confirm loop.
- **`src/cli.py`** — `run_turn` rewritten to **print new AIMessages as they
  appear** (tracked by index into the running `messages` list) *before* prompting
  on an interrupt, so SYNTHESIZE's analysis shows up right before the "store
  this?" question. `make_input` resets the Stage-6 scratch fields per turn.
- **`config.yaml`** — `nodes.analyze`/`nodes.synthesize` set to `qwen3.6:35b-a3b`
  (the MoE), `nodes.update_stats` to `qwen3:14b`.
- Tests: `tests/agent/test_graph_analyze.py`, `test_graph_update_stats.py`,
  `test_units.py`, `tests/test_stats.py`; `test_graph_ingest.py`'s placeholder
  test trimmed to `generate` only.

## Decisions made this step (given by the user, now locked in)

1. **ANALYZE/SYNTHESIZE model:** Qwen3.6 35B-A3B MoE (config only; step down to
   `qwen3:14b` if latency bottlenecks). UPDATE_STATS stays on the 14B.
2. **Evidence overflow:** answer with **partial evidence + a disclaimer that also
   tells the user to narrow scope**. Cap = `MAX_TOOL_CALLS = 8` model turns;
   per-item serialization cap `MAX_EVIDENCE_CHARS = 2000`.
3. **"Store this analysis?" interrupt:** implemented now — `doc_type='analysis'`
   in `personal_notes`, offered only when evidence is non-empty.
4. **UPDATE_STATS scope:** `bodyweight` + `pr` only; the agent auto-detects which.
   Injury/measurement deferred.

## Implementation notes / gotchas for whoever builds on this

- **The `interrupt()`-replays-the-node rule still governs.** ANALYZE has no
  interrupt so its loop lives in one node; SYNTHESIZE and UPDATE_STATS are each
  split so no LLM/parse runs before their interrupt. Follow this shape in Stage 7
  (GENERATE's draft-confirm interrupt).
- **Tool-calling stub kit** is in `tests/agent/conftest.py`: `ToolCallingStubModel`
  (`.bind_tools` returns self, `.invoke` replays canned AIMessages) +
  `tool_call_message(name, args, id)`. `RaisingChatModel` gained a `bind_tools`
  so it can stand in for an ANALYZE model that must never be called.
- **Evidence is stored in lb**; conversion happens only in `synthesize` via
  `convert_weights`. Don't convert earlier — the tools and `evidence` must stay
  canonical so source-set citations line up.
- **`run_turn` prints by message-index diffing.** If you add nodes that emit
  several AIMessages across interrupts, they'll each print once; keep emitting
  user-facing text as `AIMessage` content.
- **`store_offer`/`update_stats_confirm` treat unrecognized replies** differently:
  store_offer defaults any non-yes to "no" (→ END, no reask); stat-confirm reasks
  on anything that isn't a clear yes/no word. Intentional (storing is optional;
  a stat write should be an explicit yes).
- **`_decline` sets `review_decision="none"`**, which the `update_stats_parse`
  conditional edge treats as "no pending stat → END". The edge actually keys on
  `pending_stat` being falsy, so `review_decision` there is informational.

## Must handle / preserve (carried forward, still true)

- No live models in tests — every LLM/embedder/Chroma dependency stays behind an
  injectable seam (`get_llm`, `get_chat_model`/model factories, `get_embedder`,
  `get_chroma_client`). Graph tests use `InMemorySaver` + stubs; Chroma tests use
  the fake embedder + `EphemeralClient`.
- HITL invariant: nothing durable outside an approved interrupt branch —
  `commit_batch`/`reject_batch` (ingest), `insert_bodyweight`/`insert_pr`
  (update_stats confirm), `embed_analysis` (store_offer yes).
- lb canonical; unit conversion only at presentation (SYNTHESIZE / confirm prompts).
- Draft programs excluded from every analysis tool by default.
- Seeder only touches `data/sample.db`; `data/training.db` is live; checkpoints in
  `data/checkpoints.db`.

## What comes after (context only — do NOT build now)

Per `IMPLEMENTATION_ROADMAP.md`:
- **Stage 7** — GENERATE program writer + the `provider: cloud` branch in
  `llm_provider.py` (the raise is the seam to fill). Reuse the ANALYZE tools to
  gather history; draft-confirm via the interrupt pattern; persist as
  `program(status='draft')`. Read that stage's "⚠️ Decisions" first (cloud
  vendor/SDK, privacy mode, structured-vs-prose draft output, guardrails).
- **Stage 8** — xlsx/pdf loaders + knowledge-base ingestion + `search_knowledge`
  (independent). When done, register `search_knowledge` in `make_analyze_tools`.
- **Stage 9 (optional)** — UI, backfill, ops polish, the program/block organizer.
