# Claude Code — Step 1: Data Foundation (schema, exercise resolver, typed query tools)

You are implementing **step 1 of a multi-step project**. Read `ARCHITECTURE.md` in this repo for full context, but **do not build the agent, the LLM ingestion pipeline, Chroma, or any LLM calls yet.** This step is purely the data spine: SQLite schema, a seed script, an exercise-name resolver, and the first four typed query tools — with tests. Everything later builds on this, so correctness matters more than breadth.

## Scope of this step (and nothing more)

1. Project skeleton
2. SQLite schema + migrations-lite
3. Exercise resolver (canonical names + aliases + fuzzy fallback)
4. Seed script with realistic sample data (provided below)
5. Four typed query tools: `get_best_set`, `get_lifts`, `get_e1rm_trend`, `get_bodyweight_trend`
6. Pytest coverage for all of the above

**Explicitly out of scope for this step:** LangGraph, Ollama, Chroma, file parsing (xlsx/txt), any LLM usage, CLI chat, program generation. Do not scaffold placeholder modules for them.

## 1. Project skeleton

```
powerlifting-coach/
├── pyproject.toml          # deps: pydantic, pytest; stdlib sqlite3 (no ORM)
├── config.yaml             # just: display_unit: lb, db_path: data/training.db
├── data/                   # gitignored
├── src/
│   ├── db/
│   │   ├── schema.sql
│   │   └── connection.py   # get_conn(): WAL mode, foreign_keys ON, row_factory
│   ├── tools/
│   │   ├── resolve.py      # exercise resolver
│   │   └── queries.py      # the four typed tools
│   └── seed.py
└── tests/
```

Use plain `sqlite3` from the stdlib — no SQLAlchemy. Keep functions small and typed.

## 2. Schema

Implement `schema.sql` exactly as specified in `ARCHITECTURE.md` §3.1 (tables: `program`, `block`, `exercise`, `exercise_alias`, `session`, `lift_set`, `programmed_slot`, `cardio`, `bodyweight`, `pr`, `injury`, `measurement`, `ingest_batch`) plus the listed indexes. Key constraints to preserve:

- `program.status IN ('complete','incomplete','draft')`
- `exercise.tier IN ('competition','variation','accessory')`
- `exercise.muscle_group IN ('chest','triceps','upper back','lower back','biceps','core','front deltoids','side deltoids','rear deltoids','glutes','adductors','abductors','quads','hamstrings','calves','posterior chain')` (NULL allowed)
- All weights stored in **pounds** (`weight_lb` REAL). No kg columns anywhere.
- `connection.py` must set `PRAGMA journal_mode=WAL` and `PRAGMA foreign_keys=ON` on every connection, and provide an `init_db(conn)` that executes schema.sql idempotently (`CREATE TABLE IF NOT EXISTS`).

## 3. Exercise resolver (`resolve.py`)

```python
def resolve_exercise(conn, raw_name: str) -> ResolvedExercise | None
```

- Normalize input (lowercase, strip, collapse whitespace).
- Exact match against `exercise_alias.alias` → return the exercise.
- Else fuzzy match (stdlib `difflib.get_close_matches`, cutoff ~0.85) against aliases AND canonical names; if exactly one confident hit, return it with `matched_via='fuzzy'`.
- Else return `None` (callers decide what to do; later steps will prompt the user).
- `ResolvedExercise` is a Pydantic model: `exercise_id, name, tier, matched_via ('exact'|'fuzzy')`.
- Also provide `add_exercise(conn, name, tier, muscle_group, aliases: list[str])`.

## 4. Seed data (`seed.py`)

Create one program → blocks → sessions → sets so the tools have something real to chew on. Model it on the user's actual training. Insert:

**Program:** `"2026 Meet 1 Prep"`, status `incomplete`, start 2025-12-27.
**Blocks:** `"Hypertrophy Phase 1"` (focus hypertrophy, 13 wk, 2025-12-27→2026-03-29, complete-ish), `"Strength Block 1"` (4 wk, 2026-03-30→2026-04-26), `"Peaking Block"` (4 wk, starting 2026-05-25).

**Exercises (canonical name / tier / muscle_group, with aliases):**

`muscle_group` is restricted to this fixed list (enforced via the CHECK constraint in §2): `chest, triceps, upper back, lower back, biceps, core, front deltoids, side deltoids, rear deltoids, glutes, adductors, abductors, quads, hamstrings, calves, posterior chain`. Note: `posterior chain` is reserved for big compound movements — deadlifts, barbell squats, and variations of these — not for isolation work.

- Bench Press / competition / chest — aliases: "bench press", "competition bench", "comp bench"
- Low Bar Squat / competition / posterior chain — "low bar squat", "squat", "competition squat"
- Deadlift / competition / posterior chain — "deadlift", "deadlifts", "competition deadlift"
- Paused Low Bar Squat / variation / posterior chain — "paused low bar squats", "pause squat"
- Close Grip Bench / variation / triceps — "close grip bench"
- Weighted Pullups / accessory / upper back — "weighted pullups"
- MAG Grip Pulldowns / accessory / upper back — "mag grip pulldowns"
- Standing Overhead Tricep Extensions / accessory / triceps
- Plate Loaded Leg Curls / accessory / hamstrings
- Leg Extensions / accessory / quads

**Sessions + sets:** create ~10 sessions across March–June 2026. Must include, at minimum:
- A deadlift session on 2026-06-01 (block: Peaking, week 2, day 'w2d1'): top single 385x1 (`is_top_set=1`), backoffs 315 for 3,3,3,3; plus MAG Grip Pulldowns 3 sets (14,17,15 @ 143/121 — store weight_lb=143 for set 1 and 121 for sets 2–3, equipment nuance goes in `raw_text`).
- A squat session 2026-05-27: 315x1 top single, 6 backoff sets of 4 @ 255.
- At least two bench sessions in March 2026 with different top sets (e.g. 2026-03-05: 225x1; 2026-03-19: 230x1) — these anchor the "best bench in March" test.
- A cardio session (bike, ~26 min, 2026-05-30).
- Bodyweight entries: ~weekly from 138.0 (2026-01-03) rising to 146.0 (2026-06-01), 8+ rows.
- One PR row (Deadlift 385x1, 2026-06-01, context 'gym', `session_id` linked to the 2026-06-01 deadlift session) and one injury row (right knee, niggle, 2026-02-15→2026-03-10).

Seed must be idempotent (wipe-and-reload or INSERT OR IGNORE — your choice, document it).

## 5. Typed query tools (`queries.py`)

All four take a `conn` plus Pydantic-validated params; all return Pydantic models (not raw rows); all weights in lb. Date params are ISO strings, inclusive on both ends.

```python
get_best_set(conn, exercise: str, date_from: str, date_to: str, min_reps: int = 1) -> BestSetResult | None
    # resolve exercise via resolver; heaviest weight_lb with reps >= min_reps in window.
    # Tie-break: more reps wins, then later date. Returns set details + session date.

get_lifts(conn, exercise: str, date_from: str, date_to: str,
          top_sets_only: bool = False) -> list[SessionLifts]
    # all sets grouped by session, chronological.

get_e1rm_trend(conn, exercise: str, date_from: str, date_to: str,
               by: Literal['week','block'] = 'week') -> list[E1RMPoint]
    # Epley: weight * (1 + reps/30), computed per qualifying set (reps <= 10,
    # is_failed = 0); per bucket return the MAX e1RM and the source set
    # (weight, reps, date) so the number is auditable.

get_bodyweight_trend(conn, date_from: str, date_to: str) -> BodyweightTrend
    # rows in window + summary: first, last, delta, min, max.
```

Design rules:
- Exercise resolution failure → raise a specific `ExerciseNotFound` exception carrying the raw name (the agent will catch this later and ask the user).
- **Sets from `draft` programs are excluded** in every tool: join session→block→program and filter `program.status != 'draft'` (sessions with NULL block are included — they're real logs that just aren't attached yet).
- No unit conversion anywhere in this layer; display formatting is a later step's problem.

## 6. Tests (pytest, in-memory or tmp-path DB seeded per test module)

Minimum assertions:
1. Schema loads; FK violation actually raises (insert `lift_set` with bogus `session_id`).
2. Resolver: exact alias hit; fuzzy hit ("competiton bench" typo → Bench Press); miss returns None; ambiguity doesn't false-positive.
3. `get_best_set("bench press", "2026-03-01", "2026-03-31")` → 230 lb x1 on 2026-03-19 (the March-best anchor case).
4. `get_best_set` respects `min_reps` (min_reps=3 on deadlifts in the window → 315, not the 385 single).
5. `get_lifts(top_sets_only=True)` returns only `is_top_set` rows.
6. `get_e1rm_trend` Epley math is exact for a known set (385x1 → 397.8; 315x3 → 346.5), and buckets by ISO week correctly.
7. `get_bodyweight_trend` delta = last − first over the seeded range (≈ +8.0).
8. Draft exclusion: add a draft program + attached session/set in the test, verify it never appears in any tool's results.

## Definition of done

- `python -m src.seed` builds `data/training.db` from scratch.
- `pytest` fully green.
- A short `README.md` documenting: setup, seeding, and one usage example per query tool (a small `if __name__ == "__main__"` demo in `queries.py` is fine).
- No LLM, LangGraph, Chroma, or parsing code anywhere in the repo yet.

## What comes after (context only — do NOT build now)

Step 2: LLM extraction pipeline (raw log text → Pydantic `ParsedBatch`) with golden-file tests.
Step 3: HITL staging (`ingest_batch`) + commit path + Chroma embedding of prose.
Step 4: LangGraph router + ANALYZE ReAct loop wired to these tools via Ollama.
Step 5: GENERATE (program writer) + cloud offload flag.
