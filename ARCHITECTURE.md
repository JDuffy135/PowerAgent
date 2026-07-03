# Powerlifting Coach / Training-Log Analyst — Architecture Reference

Local, agentic RAG system for powerlifting training analysis and program generation.
Stack: **Python + LangGraph + SQLite + Chroma + Ollama (llama.cpp/Vulkan) + optional cloud LLM offload.**

This document is the single source of truth for architecture decisions. Locked-in user decisions are marked **[DECISION]**.

---

## 1. Core design principles

1. **Hybrid storage.** Quantitative data (sets, reps, weights, RPE, bodyweight, PRs, dates) lives in SQLite and is queried with exact filters and aggregations. Prose (session notes, form cues, block reviews, external studies/articles) lives in Chroma and is queried semantically. The agent routes each question to the right store.
2. **LLM-as-parser ingestion.** Training logs and program outlines are messy, idiosyncratic text (mixed lb/kg in one cell, projected-vs-actual weights together, machine pin settings, slang/emoji). An LLM extraction node converts raw uploads into schema-validated JSON; regex/rules alone will fail.
3. **Human-in-the-loop ingestion.** **[DECISION]** All parsed data is shown to the user for review/correction *before* being committed to SQLite/Chroma. Implemented via LangGraph `interrupt()`.
4. **Typed tools over free-form SQL.** The agent calls a small toolbox of parameterized query functions. This keeps small local models reliable. A gated, read-only text-to-SQL tool exists as an escape hatch only.
5. **Per-node model routing.** Cheap/high-volume nodes always run locally; heavy-reasoning nodes can be flipped to a cloud model via config.

---

## 2. Units & conventions

- **[DECISION] Canonical unit: pounds.** All weights are stored in lb. A `display_unit` user preference (`lb` | `kg`) controls output formatting; conversion happens at the presentation layer only. The ingestion parser must normalize kg-denominated entries (e.g. `1x3 @ 170KG`) to lb on the way in, preserving the original string in `raw_text`.
- Dates stored as ISO-8601 (`YYYY-MM-DD`). Session timestamps may cross midnight (user trains ~11 PM–2 AM); a session's `date` is the calendar date the user wrote in the log header, not the wall-clock date of individual sets.
- Bodyweight in lb; measurements in inches (same display-preference mechanism can convert to cm later if wanted).

---

## 3. Storage layer

### 3.1 SQLite schema

Single file DB (e.g. `data/training.db`). WAL mode on. All FKs enforced.

```sql
-- Programs = macrocycle-level containers (e.g. "2026 Meet 1 Prep")
CREATE TABLE program (
    program_id   INTEGER PRIMARY KEY,
    name         TEXT NOT NULL,
    -- [DECISION] three-state lifecycle:
    --   'complete'   = fully finished, logs exist
    --   'incomplete' = started but not finished (includes the currently-running program)
    --   'draft'      = outlined but never started
    status       TEXT NOT NULL CHECK (status IN ('complete','incomplete','draft')),
    start_date   TEXT,            -- NULL for drafts
    end_date     TEXT,            -- NULL until complete
    goals_text   TEXT,            -- "MAIN GOALS OF THE PROGRAM" prose
    review_text  TEXT,            -- macrocycle review prose (also embedded in Chroma)
    notes        TEXT
);

-- Blocks = mesocycles within a program (hypertrophy block, strength block 1, peaking block...)
CREATE TABLE block (
    block_id     INTEGER PRIMARY KEY,
    program_id   INTEGER NOT NULL REFERENCES program(program_id),
    name         TEXT NOT NULL,
    focus        TEXT,            -- 'hypertrophy' | 'strength' | 'peaking' | ...
    week_count   INTEGER,
    start_date   TEXT,
    end_date     TEXT,
    review_text  TEXT             -- block review prose (also embedded in Chroma)
);

-- Exercise dictionary. Canonical names solve the "MAG GRIP PULLDOWNS" vs
-- "mag grip pulldown" problem; aliases map raw log strings to canonical IDs.
CREATE TABLE exercise (
    exercise_id  INTEGER PRIMARY KEY,
    name         TEXT NOT NULL UNIQUE,      -- canonical, e.g. 'Low Bar Squat'
    -- [DECISION] tier flag lets analysis treat SBD work with more scrutiny
    -- while keeping ONE uniform set-level grain for all exercises:
    tier         TEXT NOT NULL CHECK (tier IN ('competition','variation','accessory')),
    muscle_group TEXT CHECK (muscle_group IN (
                    'chest','triceps','upper back','lower back','biceps','core',
                    'front deltoids','side deltoids','rear deltoids','glutes',
                    'adductors','abductors','quads','hamstrings','calves',
                    'posterior chain')),
                 -- primary target. 'posterior chain' is reserved for big compounds
                 -- (deadlifts, barbell squats, and their variations), not isolation work.
    equipment_note TEXT                      -- pin settings, seat heights, etc. (default setup)
);

CREATE TABLE exercise_alias (
    alias        TEXT PRIMARY KEY,           -- raw string as it appears in logs, lowercased
    exercise_id  INTEGER NOT NULL REFERENCES exercise(exercise_id)
);

-- A session = one workout (or cardio day)
CREATE TABLE session (
    session_id   INTEGER PRIMARY KEY,
    date         TEXT NOT NULL,
    block_id     INTEGER REFERENCES block(block_id),   -- NULL allowed (unattached logs)
    week_number  INTEGER,                    -- within block, if known
    day_number   INTEGER,                    -- within week, if known (w1d3 → 3)
    day_label    TEXT,                       -- raw text as logged: 'w2d1', 'CARDIO', etc.
    duration_min INTEGER,
    session_type TEXT NOT NULL DEFAULT 'lifting'
                 CHECK (session_type IN ('lifting','cardio','other')),
    raw_note     TEXT                        -- full original log text for this session
);

-- [DECISION] Uniform set-level grain for ALL exercises (competition + accessories).
-- Rationale: logs already record per-set reps for accessories, so set grain is the
-- natural arrival format; a single grain avoids dual query paths. The exercise.tier
-- flag provides the competition-vs-accessory distinction analytically.
CREATE TABLE lift_set (
    set_id       INTEGER PRIMARY KEY,
    session_id   INTEGER NOT NULL REFERENCES session(session_id),
    exercise_id  INTEGER NOT NULL REFERENCES exercise(exercise_id),
    set_index    INTEGER NOT NULL,           -- 1-based order within the exercise
    weight_lb    REAL,                       -- canonical lb; NULL for bodyweight-only.
                                             -- For weighted pullups/dips this is the ADDED
                                             -- weight (e.g. 45 = +45 lb), not total load.
    reps         INTEGER,
    rpe          REAL,                       -- NULL if not recorded
    is_paused    INTEGER NOT NULL DEFAULT 0,
    is_amrap     INTEGER NOT NULL DEFAULT 0,
    is_top_set   INTEGER NOT NULL DEFAULT 0, -- heavy single/double before backoffs
    is_failed    INTEGER NOT NULL DEFAULT 0,
    raw_text     TEXT                        -- original substring, incl. any kg notation
);

-- Programmed (planned) work, kept separate from performed work so
-- "projected vs actual" comparisons are a JOIN, not a parsing problem.
CREATE TABLE programmed_slot (
    slot_id      INTEGER PRIMARY KEY,
    block_id     INTEGER NOT NULL REFERENCES block(block_id),
    week_number  INTEGER,
    day_number   INTEGER,                    -- within week, if known
    day_label    TEXT,                       -- raw text: 'MONDAY', 'w1d3', etc.
    exercise_id  INTEGER REFERENCES exercise(exercise_id),
    prescription TEXT NOT NULL,              -- '1x3 @ RPE 7, 4x4 @ RPE 7-8'
    target_weight_lb REAL,                   -- projected weight if specified
    notes        TEXT
);

-- Cardio kept separate from lift_set (different shape entirely)
CREATE TABLE cardio (
    cardio_id    INTEGER PRIMARY KEY,
    session_id   INTEGER NOT NULL REFERENCES session(session_id),
    modality     TEXT,                       -- 'bike', 'run', ...
    distance_mi  REAL,
    duration_min REAL,
    intensity    TEXT,                       -- 'light', 'moderate', ...
    raw_text     TEXT
);

CREATE TABLE bodyweight (
    bw_id        INTEGER PRIMARY KEY,
    date         TEXT NOT NULL,
    weight_lb    REAL NOT NULL,
    note         TEXT
);

CREATE TABLE pr (
    pr_id        INTEGER PRIMARY KEY,
    date         TEXT NOT NULL,
    session_id   INTEGER REFERENCES session(session_id),  -- session where the PR was hit
                                             -- (NULL only if the session isn't in the DB,
                                             --  e.g. a meet or historical PR with no log)
    exercise_id  INTEGER NOT NULL REFERENCES exercise(exercise_id),
    weight_lb    REAL NOT NULL,
    reps         INTEGER NOT NULL,           -- 1 for a true 1RM PR
    context      TEXT                        -- 'gym', 'mock meet', 'meet', notes
);

CREATE TABLE injury (
    injury_id    INTEGER PRIMARY KEY,
    start_date   TEXT NOT NULL,
    end_date     TEXT,                       -- NULL = ongoing
    area         TEXT NOT NULL,              -- 'right knee', 'hip', ...
    severity     TEXT,                       -- 'niggle', 'moderate', 'serious'
    note         TEXT
);

CREATE TABLE measurement (
    m_id         INTEGER PRIMARY KEY,
    date         TEXT NOT NULL,
    site         TEXT NOT NULL,              -- 'arm', 'femur length', ...
    value_in     REAL NOT NULL,
    note         TEXT
);

-- Ingestion audit trail: every upload gets a record; parsed payloads reference it.
CREATE TABLE ingest_batch (
    batch_id     INTEGER PRIMARY KEY,
    created_at   TEXT NOT NULL,
    source_file  TEXT,
    status       TEXT NOT NULL CHECK (status IN ('pending_review','committed','rejected')),
    parsed_json  TEXT                        -- the JSON shown to the user at review time
);
```

Useful indexes: `lift_set(exercise_id, session_id)`, `session(date)`, `session(block_id)`, `bodyweight(date)`, `pr(exercise_id, date)`.

**Derived metrics are computed, not stored:** e1RM (Epley/Brzycki on top sets), weekly tonnage, per-exercise volume — these are functions over `lift_set`, implemented inside the query tools. Storing them invites staleness.

### 3.2 Chroma vector store

Two collections, persistent client (`data/chroma/`):

| Collection | Contents | Chunking | Metadata |
|---|---|---|---|
| `personal_notes` | Per-session note prose, block reviews, "THE GOOD/THE BAD/WHAT I'D CHANGE" text, form-cue updates | One chunk per session note (they're short); block reviews chunked per section | `date`, `block_id`, `session_id`, `doc_type` (`session_note` \| `block_review` \| `form_cue`), `exercises` (list of canonical names mentioned) |
| `knowledge` | Studies, articles, PDFs, video transcripts | ~500–800 token chunks, ~15% overlap | `source`, `title`, `topic`, `author`, `year` |

Key decisions:
- **Metadata `where` filters are mandatory** for `personal_notes` queries so semantic search respects time windows ("knee pain mentions in the last 2 blocks" = filter `block_id IN (...)` then similarity).
- Structured rows and embedded prose are linked via `session_id`/`block_id`, so the agent can hop from a retrieved note to the exact numbers of that day, and vice versa.
- Embedding model: **nomic-embed-text** (or BGE-M3 / Qwen3-Embedding) served via Ollama. Store the embedder name in collection metadata; changing embedders requires re-embedding. *Implemented (Stage 11c):* `src/ingest/reembed.py` rebuilds every collection with the configured embedder (build-then-swap) and stamps its name into collection metadata; exposed as the `/reembed` CLI command. Block/program reviews and form cues are written via `embed_review` (Stage 11b) as single, idempotent `personal_notes` docs (`doc_type` `block_review` / `program_review` / `form_cue`).

---

## 4. Agent architecture (LangGraph)

### 4.1 Graph topology

```
                       ┌──────────────┐
        user input ──▶ │    ROUTER    │  intent classification (structured output)
                       └──────┬───────┘
      ┌───────────┬───────────┼────────────┬──────────────┐
      ▼           ▼           ▼            ▼              ▼
 ┌─────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌───────────┐
 │ INGEST  │ │ ANALYZE  │ │ GENERATE │ │  UPDATE  │ │ CHITCHAT/ │
 │ pipeline│ │ (ReAct   │ │ (program │ │  STATS   │ │ FALLBACK  │
 │         │ │  loop w/ │ │  writer) │ │          │ │           │
 └────┬────┘ │  tools)  │ └────┬─────┘ └────┬─────┘ └─────┬─────┘
      │      └────┬─────┘      │            │             │
      │           └──────┬─────┴────────────┴─────────────┘
      ▼                  ▼
┌────────────┐    ┌─────────────┐
│ interrupt()│    │ SYNTHESIZE  │  final answer composition + unit formatting
│ HITL review│    └──────┬──────┘
└─────┬──────┘           │
      ▼                  ▼
  COMMIT or         (optional) interrupt(): "store this analysis/review?"
  REJECT
```

### 4.2 Node responsibilities

| Node | Job | Model |
|---|---|---|
| **ROUTER** | Classify intent into `ingest` / `analyze` / `generate` / `update_stats` / `chat`. Emits structured output (Pydantic). One job, tiny prompt. | Local (always) |
| **INGEST** | Sub-pipeline: file loader → LLM extraction → Pydantic validation → write `ingest_batch(status='pending_review')` → `interrupt()` for review → on approval, commit to SQLite + embed prose to Chroma. | Local (always) |
| **ANALYZE** | ReAct loop over the typed query tools + vector search. Gathers evidence, computes trends, hands findings to SYNTHESIZE. | Local default; **cloud-flippable** |
| **GENERATE** | Program/block writer. Pulls history via the same tools (recent e1RMs, what progressions worked, injury constraints, block reviews), then drafts a block/macrocycle. Heaviest reasoning in the system. | Local heavy model; **cloud-flippable** |
| **UPDATE_STATS** | Parses "bodyweight was 146 this morning" / "hit a 405 deadlift PR" into single-row inserts. Confirms before writing (lightweight HITL). | Local (always) |
| **SYNTHESIZE** | Compose the final answer from gathered evidence; apply `display_unit` preference; offer to store reviews/analyses via `interrupt()`. | Same model as the calling branch |

### 4.3 State (LangGraph `TypedDict`)

```python
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    intent: str | None                # router output
    evidence: list[dict]              # tool results accumulated by ANALYZE
    pending_batch_id: int | None      # ingest batch awaiting HITL review
    display_unit: str                 # 'lb' | 'kg'
```

Checkpointer: **SqliteSaver** (same DB file or a sibling), required for `interrupt()`/resume to work across turns.

### 4.4 Human-in-the-loop contract **[DECISION]**

- INGEST always pauses with the parsed JSON rendered as a readable summary table (sessions → exercises → sets, plus anything the parser was unsure about flagged with `"confidence": "low"`).
- User can reply: `approve`, `reject`, or supply corrections (free text → a small correction-application LLM pass → re-render → re-confirm).
- Nothing touches `lift_set`/`session`/Chroma until `approve`. The `ingest_batch` row is the audit trail either way.

---

## 5. Tool layer

### 5.1 Typed query tools (the workhorses)

All are plain Python functions with Pydantic-validated args, registered as LangGraph tools. All accept `date_from`/`date_to` and/or `block_ids` filters. All compute in lb and let SYNTHESIZE convert for display.

```
get_sessions(date_from, date_to, block_id=None, session_type=None)
get_lifts(exercise, date_from, date_to, tier=None, top_sets_only=False)
get_best_set(exercise, date_from, date_to, min_reps=1)      # "best bench in March"
get_e1rm_trend(exercise, by='week'|'block', date_from, date_to)
get_volume_trend(exercise_or_muscle_group, by='week'|'block', date_from, date_to)
get_frequency(exercise, by='week', date_from, date_to)
get_bodyweight_trend(date_from, date_to)
get_prs(exercise=None, date_from=None, date_to=None)
get_injuries(active_only=False, area=None)
get_measurements(site=None, date_from=None, date_to=None)
get_programs(status=None)                                    # complete/incomplete/draft
get_block_outline(block_id)                                  # programmed_slot rows
compare_programmed_vs_actual(block_id, exercise=None)
search_notes(query, date_from=None, date_to=None, exercises=None, doc_type=None)
search_knowledge(query, topic=None)
resolve_exercise(raw_name)              # alias lookup + fuzzy match; used by everything
```

Analysis rules baked into tools (not left to the LLM):
- e1RM via Epley (`w * (1 + reps/30)`) computed on qualifying top sets; tool returns both the number and the source set so claims are auditable.
- **Draft programs are excluded from all trend/analysis queries by default** (`status='draft'` filtered out); `get_programs` can still list them explicitly. **[DECISION]** This is what keeps abandoned outlines from polluting analysis.

### 5.2 Escape hatch: gated text-to-SQL

`run_readonly_sql(query)` — `SELECT`-only (validated by sqlglot parse), read-only connection (`PRAGMA query_only`), row limit, timeout. The ANALYZE prompt instructs the model to prefer typed tools and reach for SQL only when none fit.

### 5.3 Ingestion tools

```
parse_upload(path)          # loader (xlsx via openpyxl/pandas, txt, pdf) → raw text blocks
extract_training_data(text) # LLM extraction → ParsedBatch (Pydantic)
stage_batch(parsed)         # writes ingest_batch(pending_review)
commit_batch(batch_id)      # SQLite inserts + Chroma embeds, transactional
```

`ParsedBatch` (Pydantic) mirrors the schema: sessions → sets/cardio, plus new-exercise candidates (which prompt an alias/tier confirmation during HITL review).

Parser must handle (all observed in real data):
- kg↔lb mixed in one cell → normalize to lb, keep original in `raw_text`
- projected vs actual in one cell (`1x3 @ 170KG (actually used 375 pounds)`) → programmed_slot vs lift_set
- pin/plate configs (`143x1, 121x2`, `35KGx2`) → equipment_note, weight best-effort
- top single + backoffs on one line (`385x1, 315x4` / `Reps: 1, 3, 3, 3, 3`) → set rows with `is_top_set`
- skipped exercises (`Reps: N/A`) → no set rows; note still embedded
- slang/Spanish/emoji → irrelevant to numbers, preserved in prose

---

## 6. Models & serving

### 6.1 Runtime

- **Ollama** (wraps llama.cpp) as the local server; on the RX 6750 XT (RDNA2), use the **Vulkan** backend — more reliable than ROCm on consumer AMD. Exposes an OpenAI-compatible endpoint that LangChain/LangGraph consumes via `ChatOllama` / `OpenAI`-compatible client.
- Hardware envelope: 12 GB VRAM, 32 GB DDR5, Ryzen 7 7700X.

### 6.2 Model assignments

| Role | Model | Fit | Notes |
|---|---|---|---|
| Router, UPDATE_STATS, extraction, HITL correction pass | **Qwen3 14B, Q4_K_M** | ~9–10 GB VRAM | Strong tool-calling/structured output for its size; the system workhorse |
| ANALYZE + GENERATE (quality mode) | **Qwen3.6 35B-A3B (MoE), Q4** | ~3B active params; hot experts in VRAM via llama.cpp `-ncmoe`, rest spills to system RAM | Slower but noticeably deeper reasoning; acceptable per user's latency tolerance |
| Embeddings | **nomic-embed-text** (or BGE-M3) | negligible | via Ollama |
| Cloud offload (optional) | **Claude Sonnet / GPT-5.x** | n/a | ANALYZE + GENERATE only |

Practical note: you won't hold the 14B and the 35B-A3B resident simultaneously in 12 GB. Ollama load-swaps models per request automatically; accept the swap latency, or run everything on the 14B initially and introduce the MoE later.

### 6.3 Provider abstraction

```python
def get_llm(node: str) -> BaseChatModel:
    cfg = CONFIG.nodes[node]          # {'provider': 'local'|'cloud', 'model': ...}
    ...
```

Every node fetches its model through this. Flipping ANALYZE/GENERATE to cloud is a config change, zero code change. Privacy option (later): send cloud nodes an anonymized evidence summary rather than raw logs.

---

## 7. Config & layout

```
powerlifting-coach/
├── config.yaml            # node→model map, display_unit, paths
├── data/
│   ├── training.db
│   └── chroma/
├── src/
│   ├── db/                # schema.sql, connection, migrations
│   ├── ingest/            # loaders, extraction, embedding, knowledge base, reembed
│   ├── tools/             # typed query tools, vector search, sql escape hatch
│   ├── agent/             # graph.py, nodes/, state.py, llm_provider.py
│   ├── ui/                # Streamlit app: tab veneers + streamlit-free logic
│   └── cli.py             # terminal REPL alternative
└── tests/                 # golden-file parser tests + tool unit tests
```

UI: Streamlit (`streamlit run src/ui/app.py`), five tabs — Chat, Trends
(time-series charts over the typed query tools), Organizer, Backfill, Dev
Tools. The CLI REPL remains as the terminal alternative.

---

## 8. Locked decisions (quick reference)

1. **Canonical unit: pounds**; `display_unit` preference for output (lb/kg).
2. **Program status:** `complete` / `incomplete` (started, incl. current) / `draft` (never started). Drafts excluded from analysis by default.
3. **Uniform set-level grain** for all exercises; `exercise.tier` (`competition`/`variation`/`accessory`) carries the SBD-vs-accessory distinction.
4. **HITL review before every ingest commit**, via LangGraph `interrupt()`, with an `ingest_batch` audit trail.
5. Hybrid SQLite + Chroma storage; typed tools over free-form SQL; per-node local/cloud model routing.
