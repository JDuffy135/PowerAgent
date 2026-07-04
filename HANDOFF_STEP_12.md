# Handoff — Step 12: LangFuse observability (self-hosted tracing)

For the next Claude session. Read `ARCHITECTURE.md` (full design) and
`HANDOFF_STEP_11.md` (Stages 1–11, all complete) first. This step adds
**tracing/observability only** — no agent behavior changed.

## What was built (299 tests passing, 12 new)

The user wanted to debug the agent against arbitrary chat prompts by seeing
every step of its reasoning. Solution: self-hosted **LangFuse v3** + a single
tracing seam.

### Infrastructure — `observability/`

- `docker-compose.yml`: the **official LangFuse compose file, vendored
  verbatim** (web + worker + Postgres + ClickHouse + Redis + MinIO). Never
  edit it — all customization lives in `observability/.env`, which Compose
  auto-loads (secrets + `LANGFUSE_INIT_*` headless provisioning, so the
  org/project/user/API keys exist deterministically on first boot).
  To upgrade LangFuse, re-download the compose file; `.env` never conflicts.
- `observability/README.md`: start/stop, login, how to read traces.
- UI at `http://localhost:3000`; API keys are in `.env`
  (`LANGFUSE_INIT_PROJECT_PUBLIC_KEY` / `_SECRET_KEY`).

### The tracing seam — `src/agent/tracing.py` (new)

Everything observability goes through this module; **`langfuse` is imported
lazily only after tracing is confirmed enabled**, so the app and tests never
require the package/server. Enabled iff `config.yaml langfuse.enabled: true`
AND the env vars named by `public_key_env`/`secret_key_env` (default
`LANGFUSE_PUBLIC_KEY`/`LANGFUSE_SECRET_KEY`) are set. Any config problem →
one `warnings.warn`, then silent no-op (tracing must never take the coach
down).

- `attach_tracing(config, thread_id=, source=)` — copies a `graph.invoke`
  config, appending the LangFuse LangChain `CallbackHandler` plus
  `metadata={langfuse_session_id: thread_id, langfuse_tags: [source]}`.
  Identity when disabled. This traces **all graph nodes, ChatOllama/
  ChatAnthropic calls, and `tool.invoke` calls** (LangGraph propagates
  callbacks into nodes; py≥3.11 contextvars cover implicit calls).
  The LangGraph `thread_id` doubles as the LangFuse **session id**, so a HITL
  round-trip spanning several `graph.invoke`s reads as one session.
- `traced_llm(call, node=, model=, provider=, system_prompt=)` — wraps a raw
  `prompt -> str` callable in a generation span
  (`start_as_current_observation(as_type="generation")`); records
  input/output/model/error, re-raises untouched. Pass-through when disabled.
  The SDK is OTel-based, so calls inside a traced turn nest under that trace;
  standalone calls (Backfill chunks, `/learn` metadata) become root traces.
- `flush()` — safe always; **reads the cached client only** (never constructs
  one just to flush).
- Test seams: `_set_client_for_tests(client)` and `reset()`.

### Wire-up (three call sites)

- `src/ingest/extract.py::get_llm` — both branches (local `_call`, cloud
  `_cloud_llm`) return through `tracing.traced_llm(...)`, named after the
  node. This one wrapper covers **every** raw-seam caller: extraction, HITL
  correction (`correct.py`), update-stats parse, generate draft call,
  knowledge metadata guessing — including when invoked outside chat
  (Backfill tab, `/learn`). The import of `tracing` is inside `get_llm`
  (circular-import: tracing → llm_provider → extract).
- `src/ui/chat_tab.py::_config()` — wraps the invoke config via
  `attach_tracing(..., source="streamlit")`; `tracing.flush()` after every
  `drive_turn` (and on the error path) so traces are immediately visible.
- `src/cli.py::main()` — same with `source="cli"`; flush after each turn and
  at exit.

### Config & dependencies

- `config.yaml` gained a top-level `langfuse:` section (`enabled`, `host`,
  `public_key_env`, `secret_key_env`). Keys live in env vars only — same
  pattern as `api_key_env`. Currently `enabled: true` (harmless without keys).
- `pyproject.toml`: `langfuse>=3.0` (installed 4.13.0) and `langchain>=1.0`
  (the metapackage — required by `langfuse.langchain.CallbackHandler`).

### Tests — `tests/agent/test_tracing.py` (12 new)

Fake-client only (house rule intact — no live LangFuse, `langfuse` package
never imported by tests): disabled-mode no-ops, warn-once on missing keys,
`attach_tracing` shape + preservation of existing callbacks/metadata,
`traced_llm` generation recording + error re-raise, flush, and two
`get_llm`-integration checks (plain callable when off, `_traced` wrapper when
on).

## Deliberately NOT traced

- The **embedder** (`nomic-embed-text`) — high-volume noise.
- Pure-SQL query tools outside a graph turn (Trends tab) — inside a turn the
  callback handler already captures `tool.invoke`.
- Token usage on the **raw seam**: both callables return only the content
  string (the Ollama/Anthropic usage envelope is discarded before the wrapper
  sees it). LangChain-seam calls DO report tokens via the handler. If you
  want raw-seam usage, thread the response envelope out of
  `extract._call`/`_cloud_llm` first.

## Verification status (2026-07-04)

- `pytest`: **299 passed** with no LangFuse running and keys unset — the
  no-op guarantee holds.
- **Docker is NOT installed on this machine** (no `docker`, no `podman`,
  no passwordless sudo), so the stack itself and live trace inspection could
  not be run end-to-end. First thing to do on a machine with Docker:
  1. `docker compose -f observability/docker-compose.yml up -d`
  2. export the two keys from `observability/.env`
  3. `streamlit run src/ui/app.py`, send one analysis prompt, and confirm in
     the LangFuse UI that the node tree nests correctly (router → analyze →
     tool spans → synthesize) rather than appearing as orphan traces.
- **Contingency if node-internal calls appear as orphan traces**: pass the
  LangGraph-injected `config: RunnableConfig` through node functions to
  `llm.invoke`/`tool.invoke` (accepting `config` as a second node arg) —
  see the plan's contingency note.

## Must handle / preserve (carried forward, still true)

- No live models/services in tests — LangFuse included; use
  `_set_client_for_tests` + `reset()`.
- Tracing must stay **fail-open**: never raise from `tracing.py` toward the
  app; new failure modes should downgrade to `_warn_once`.
- HITL invariant, lb-canonical weights, draft exclusion, seeder → sample.db
  only — all unchanged by this step.
- Never hard-require an API key (Anthropic *or* LangFuse) when running
  local-only.
