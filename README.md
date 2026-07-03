# Powerlifting Coach — Data Foundation + Ingestion + Tools + Agent Graph (Steps 1–6)

SQLite schema, exercise resolver, seed data, and four typed query tools
(Step 1); the LLM extraction pipeline that turns raw log text into a
schema-validated `ParsedBatch` (Step 2); the HITL staging + transactional
commit path plus Chroma prose embedding (Step 3); the rest of the typed
query tools plus Chroma semantic search and a gated read-only SQL escape hatch
(Step 4); the LangGraph agent core — router, INGEST pipeline with
interrupt-based HITL review (corrections + block assignment), SqliteSaver
checkpointing, and a minimal CLI REPL (Step 5); and the ANALYZE ReAct loop,
SYNTHESIZE answer-writer (with unit conversion + a "store this analysis?"
offer), and UPDATE_STATS confirm-before-write path (Step 6). See
`ARCHITECTURE.md` for the full design and `HANDOFF_STEP_6.md` for the latest
handoff. GENERATE (program writer) + the cloud provider branch land in Stage 7.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Seeding

```bash
python -m src.seed
```

Builds `data/sample.db` from scratch. Idempotent via wipe-and-reload: every
run deletes all rows from every table, then re-inserts the sample dataset —
so running it repeatedly always leaves the DB in the same state.

The seeder targets **`data/sample.db`, never `data/training.db`.**
`training.db` is the live database the HITL commit path writes real,
user-approved logs into; keeping the wipe-and-reload sample data in a separate
file makes re-seeding safe by construction.

## Tests

```bash
pytest
```

Tests run against an in-memory SQLite DB seeded fresh per test (see
`tests/conftest.py`).

## Query tool usage

All tools take a `conn` plus validated params and return Pydantic models.
Exercise names are resolved via `resolve_exercise` (exact alias → fuzzy
match → `ExerciseNotFound`). Draft programs are excluded from every tool.

```python
from src.db.connection import get_conn
from src.tools.queries import (
    get_best_set,
    get_lifts,
    get_e1rm_trend,
    get_bodyweight_trend,
)

conn = get_conn("data/training.db")

# Heaviest bench in March 2026
get_best_set(conn, "bench press", "2026-03-01", "2026-03-31")
# -> BestSetResult(exercise='Bench Press', weight_lb=230.0, reps=1, date='2026-03-19', ...)

# All deadlift top singles across the whole log
get_lifts(conn, "deadlift", "2026-01-01", "2026-06-30", top_sets_only=True)
# -> [SessionLifts(session_id=..., date='2026-04-02', sets=[...]), SessionLifts(..., date='2026-06-01', ...)]

# Weekly e1RM trend for deadlift
get_e1rm_trend(conn, "deadlift", "2026-01-01", "2026-06-30", by="week")
# -> [E1RMPoint(bucket='2026-W14', e1rm=357.3, source_weight_lb=335.0, source_reps=2, ...),
#     E1RMPoint(bucket='2026-W23', e1rm=397.8, source_weight_lb=385.0, source_reps=1, ...)]

# Bodyweight trend for the whole prep
get_bodyweight_trend(conn, "2026-01-01", "2026-06-30")
# -> BodyweightTrend(rows=[...], first=138.0, last=146.0, delta=8.0, min=138.0, max=146.0)
```

You can also run the demo block directly:

```bash
python -m src.tools.queries
```

## Ingestion pipeline (Step 2)

```python
from src.ingest.loaders import parse_upload
from src.ingest.extract import extract_training_data

text = parse_upload("some_log.txt")
batch = extract_training_data(text, conn=conn)  # conn optional, read-only exercise resolution
```

- `src/ingest/models.py` — `ParsedBatch` (sessions → sets/cardio/programmed_slots,
  plus `new_exercise_candidates` for names that didn't resolve).
- `src/ingest/loaders.py` — `parse_upload(path)`; `.txt` supported now, `.xlsx`/`.pdf` raise
  `NotImplementedError` (stubbed for a later step).
- `src/ingest/extract.py` — `extract_training_data(text, conn=None, llm=None)`. `get_llm(node)`
  is the provider seam (`config.yaml` → `nodes.<node>`): local Ollama (Qwen3 14B, structured
  JSON output) by default, cloud-flippable later. Pure w.r.t. the DB — `conn` is only used
  read-only via `resolve_exercise`; nothing is written or committed here (that's Step 3).
- `tests/ingest/` — golden-file tests. Since there's no live model in this environment, tests
  inject a stub `llm` callable that returns each fixture's golden JSON, exercising the real
  parse→validate→resolve pipeline without depending on a running Ollama server.

## HITL staging + commit path (Step 3)

Parsing produces a `ParsedBatch`; nothing durable happens until the user
approves. That bridge lives in `src/ingest/`:

```python
from src.ingest.extract import extract_training_data
from src.ingest.stage import stage_batch, get_pending_batch
from src.ingest.review import render_batch
from src.ingest.commit import commit_batch, reject_batch

batch = extract_training_data(text, conn=conn)      # Step 2
batch_id = stage_batch(conn, batch, source_file="log.txt")  # -> ingest_batch(pending_review)

print(render_batch(get_pending_batch(conn, batch_id)))      # readable HITL summary

commit_batch(conn, batch_id)                        # transactional SQLite + Chroma embed
# or: reject_batch(conn, batch_id)                  # writes nothing
```

- `stage.py` — `stage_batch` writes the `pending_review` audit-trail row (the
  serialized batch JSON, no training data); `get_pending_batch` rehydrates it.
- `review.py` — `render_batch(parsed) -> str`, a pure summary with every
  `confidence < 1.0` field flagged. This is what Step 4's `interrupt()` shows.
- `commit.py` — `commit_batch(conn, batch_id)` is **transactional**: it
  resolves/creates exercises (`add_exercise(commit=False)`), inserts
  `session`/`lift_set`/`cardio` rows, and flips the batch to `committed`, all in
  one transaction — a mid-commit failure rolls everything back. Committing an
  already-committed batch is a no-op; `reject_batch` is the terminal
  `rejected` transition. Programmed slots are preserved in the audit JSON but
  not inserted yet (they need block assignment, a later step).
- `embed.py` — session `raw_note` prose is embedded into the Chroma
  `personal_notes` collection (persistent client at `data/chroma/`) as part of
  commit. `get_embedder()` / `get_chroma_client()` are seams (like `get_llm`):
  tests inject a deterministic fake embedder + in-memory client, so no live
  Ollama/`nomic-embed-text` is required. Pass `commit_batch(..., embed_prose=False)`
  to skip Chroma.

## Full tool layer (Step 4)

The rest of ARCHITECTURE.md §5.1's typed tools, plus the vector-search and SQL
escape-hatch tools from §3.2/§5.2:

```python
from src.tools.queries import (
    get_sessions, get_frequency, get_volume_trend, get_prs, find_recent_prs,
    commit_prs, get_injuries, get_measurements, get_programs, get_block_outline,
    compare_programmed_vs_actual,
)
from src.tools.vector import search_notes
from src.tools.sql import run_readonly_sql
```

- `get_volume_trend(exercise_or_muscle_group, date_from, date_to, by='week'|'block')`
  returns both hard-set count and tonnage (lb) per bucket. Bodyweight-only sets
  (`weight_lb IS NULL`) estimate load using the user's most recently logged
  bodyweight (0 lb if none has ever been recorded) — NOT the bodyweight as of
  that specific date, the single latest entry in the table.
- `get_prs` reads only the manually-recorded `pr` table. `find_recent_prs(date_from,
  date_to, exercise=None)` is a separate, read-only tool that auto-derives PR
  candidates (sets whose Epley e1RM beats every prior set for that exercise,
  all-time) without writing anything; pass user-accepted candidates to
  `commit_prs(conn, candidates)` to insert them (idempotent — re-accepting a
  candidate already in `pr` is a no-op).
- `compare_programmed_vs_actual(block_id, exercise=None)` joins `programmed_slot`
  to performed `lift_set` rows by `(week_number, day_number, exercise_id)`.
  Mismatches are surfaced, never silently dropped: a programmed slot with no
  matching session still appears in `rows` with `actual_*` fields `None`;
  performed work with no matching slot appears in `unmatched_actual` (with its
  session's `raw_note`); a block with zero `programmed_slot` rows returns
  `rows=[]` plus a `note` explaining nothing was programmed, while still
  surfacing `unmatched_actual`.
- `search_notes(query, date_from=None, date_to=None, exercises=None, doc_type=None)`
  — semantic search over the `personal_notes` Chroma collection. **At least one
  metadata filter is required** (raises `ValueError` otherwise, per
  ARCHITECTURE.md §3.2). Date filtering uses a numeric `date_ordinal` metadata
  mirror of the display `date` string (Chroma's `$gte`/`$lte` require int/float
  operands); `exercises` filtering is client-side substring containment since
  Chroma stores the mentioned-exercises list as a comma-joined string.
- `run_readonly_sql(conn, query, max_rows=200, timeout_s=5.0)` — the gated
  escape hatch. Validates the query is a single `SELECT` via `sqlglot`
  (rejects multi-statement input and any non-SELECT), runs it under
  `PRAGMA query_only=ON` (always restored after, even on error), and caps rows
  via an outer `LIMIT`.

## LangGraph agent core + CLI (Step 5)

```bash
python -m src.cli
```

Starts a chat REPL over the agent graph (requires a live Ollama server for
routing/extraction/chitchat). `/ingest <path>` ingests a `.txt` training log:
the parse is staged and shown for review — reply `approve`, `reject`, or
describe corrections in plain text (a correction LLM re-emits the full batch;
capped at 5 rounds). On approval you're asked which program/block the batch
belongs to: an existing block id, `new <program> / <block>` (created on the
fly, which also unlocks `programmed_slot` insertion), or `none` to leave it
unattached and organize later. Nothing durable is written before approval.

Key modules: `src/agent/llm_provider.py` (per-node model routing —
`get_chat_model` returns a `ChatOllama`; cloud providers raise until Stage 7),
`src/agent/state.py`, `src/agent/nodes/{router,ingest,chitchat}.py`,
`src/agent/graph.py` (topology + `SqliteSaver` checkpointer in a separate
`data/checkpoints.db`), `src/ingest/correct.py` (HITL correction pass),
`src/cli.py`. All graph paths are covered by stub-LLM tests
(`tests/agent/`) — no live model needed for the suite.

## ANALYZE + SYNTHESIZE + UPDATE_STATS (Step 6)

Plain questions and stat reports are now handled end-to-end in the same REPL:

```text
you> what was my best bench in March?
coach> Your best March bench was 230 lb x1 on 2026-03-19.
Store this analysis to your notes for future reference? (yes/no)
review> no

you> bodyweight was 146 this morning
Record this? bodyweight 146 lb on 2026-07-02. Reply `yes` to save or `no` to discard.
review> yes
coach> Recorded bodyweight 146 lb on 2026-07-02.
```

- **ANALYZE** (`src/agent/nodes/analyze.py`) is a bounded ReAct loop
  (`MAX_TOOL_CALLS = 8`) over the Step 1/4 query tools, wrapped as LangChain
  tools in `src/agent/tools.py` (tight, enum/date-typed schemas;
  `ExerciseNotFound` returned as `{"error": ...}` so the model can recover). It
  accumulates structured `evidence`; the ReAct scratch (tool calls/results)
  stays local to the node so the durable message history stays clean. Hitting
  the cap sets `evidence_truncated`.
- **SYNTHESIZE** (`src/agent/nodes/synthesize.py`) composes the final answer
  from `evidence`. It is the **only** place weights are converted from canonical
  lb to the user's `display_unit` (`src/agent/units.py`). On overflow it appends
  a fixed disclaimer that asks the user to narrow scope. It then offers to store
  the analysis; on `yes` the text is embedded into Chroma `personal_notes` under
  `doc_type='analysis'` (`embed_analysis`).
- **UPDATE_STATS** (`src/agent/nodes/update_stats.py`) parses one reported
  bodyweight or PR (weight normalized to lb), then confirms before writing via
  `interrupt()`; the durable inserts live in `src/tools/stats.py`. Unknown PR
  exercises and out-of-scope reports (injuries/measurements) are declined
  without an interrupt.

All of it is covered by scripted-stub tests (`tests/agent/test_graph_analyze.py`,
`test_graph_update_stats.py`, `test_units.py`, `tests/test_stats.py`) — the
tool-calling stub drives real tools against the seeded DB, no live model.

## What's here vs. what's not

Implemented: `src/db/schema.sql`, `src/db/connection.py`, `src/tools/resolve.py`,
`src/tools/queries.py`, `src/seed.py` (Step 1); `src/ingest/models.py`,
`src/ingest/loaders.py`, `src/ingest/extract.py` (Step 2); `src/ingest/stage.py`,
`src/ingest/review.py`, `src/ingest/commit.py`, `src/ingest/embed.py` (Step 3);
the rest of `src/tools/queries.py`, `src/tools/vector.py`, `src/tools/sql.py`
(Step 4); `src/agent/` (graph, router, INGEST HITL flow, provider),
`src/ingest/correct.py`, `src/cli.py` (Step 5); the ANALYZE ReAct loop
(`src/agent/nodes/analyze.py`, `src/agent/tools.py`), SYNTHESIZE +
"store this analysis?" (`src/agent/nodes/synthesize.py`, `src/agent/units.py`,
`embed_analysis`), and UPDATE_STATS (`src/agent/nodes/update_stats.py`,
`src/tools/stats.py`) (Step 6); full pytest coverage throughout (161 tests).

Explicitly not implemented (future steps per `ARCHITECTURE.md` /
`IMPLEMENTATION_ROADMAP.md`): program generation (GENERATE — still a placeholder
node that replies "not implemented yet"), cloud provider branch, xlsx/pdf
loading, `search_knowledge`/knowledge-base ingestion, Streamlit/Gradio UI.
