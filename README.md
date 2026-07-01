# Powerlifting Coach — Data Foundation + LLM Extraction (Steps 1–2)

SQLite schema, exercise resolver, seed data, and four typed query tools
(Step 1), plus the LLM extraction pipeline that turns raw log text into a
schema-validated `ParsedBatch` (Step 2). See `ARCHITECTURE.md` for the full
design, `CLAUDE_CODE_STEP_1.md` / `HANDOFF_STEP_2.md` for what each step
covered. No LangGraph, Chroma, HITL commit path, or CLI yet.

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

Builds `data/training.db` from scratch. Idempotent via wipe-and-reload: every
run deletes all rows from every table, then re-inserts the sample dataset —
so running it repeatedly always leaves the DB in the same state.

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

## What's here vs. what's not

Implemented: `src/db/schema.sql`, `src/db/connection.py`, `src/tools/resolve.py`,
`src/tools/queries.py`, `src/seed.py` (Step 1); `src/ingest/models.py`,
`src/ingest/loaders.py`, `src/ingest/extract.py` (Step 2); full pytest coverage for both.

Explicitly not implemented (future steps per `ARCHITECTURE.md`): LangGraph
agent graph, Chroma vector store, HITL staging/commit path (`ingest_batch` →
SQLite/Chroma), xlsx/pdf loading, CLI chat, program generation.
