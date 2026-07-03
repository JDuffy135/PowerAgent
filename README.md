# Powerlifting Coach

A local-first AI coach for your powerlifting training. Drop in your training
logs (typed, `.txt`, `.xlsx`, or `.pdf`), and it parses them into a structured
history you can then **ask questions about** ("what was my best bench in
March?", "how's my deadlift e1RM trending this prep?"), **report stats to**
("hit a 405 deadlift PR today"), and **get new programs from** (evidence-grounded
training blocks written around your own history and philosophy). It also ingests
reference material — studies, articles, transcripts — so the coach can pull real
theory into its reasoning.

Everything runs on your machine: SQLite for your numbers, a local vector store
for your notes, and local LLMs via [Ollama](https://ollama.com). Nothing about
your training leaves your computer unless you explicitly flip a node to a cloud
model. Every write to your training history is **reviewed by you before it's
saved** — the coach proposes, you approve.

---

## Quick start

### 1. Prerequisites

- **Python 3.11+**
- **[Ollama](https://ollama.com)** running locally, with the default models
  pulled:
  ```bash
  ollama pull qwen3:14b          # routing, extraction, chitchat, stat parsing
  ollama pull qwen3.6:35b-a3b    # analysis + answer writing
  ollama pull nomic-embed-text   # embeddings for semantic note/knowledge search
  ```
- **(Optional) An Anthropic API key** — program generation defaults to the
  cloud (`claude-sonnet-5`) because it's the heaviest reasoning in the app. Set
  it before launching, or switch that node to local (see
  [Configuration](#configuration)):
  ```bash
  export ANTHROPIC_API_KEY=sk-ant-...
  ```

### 2. Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

### 3. Run the app

```bash
streamlit run src/ui/app.py
```

This opens the coach in your browser. Your training data lives in
`data/training.db`, which starts empty — you fill it by ingesting logs (below).
The five tabs are walked through in [Using the app](#using-the-app).

> **Want to see it populated first?** Run `python -m src.seed` to build a
> **separate** sample database (`data/sample.db`) with a realistic prep in it.
> The seeder never touches your live `training.db`.

---

## Using the app

The UI has five tabs.

### 💬 Chat

The main workspace — a chat window over your whole training history. Just type;
the coach figures out what you want:

- **Ask about your history** — "best bench in March?", "deadlift e1RM trend this
  prep", "any knee-pain mentions in the last two blocks?" The coach queries your
  data, answers with the actual numbers, and offers to save noteworthy analyses
  to your notes.
- **Report a stat** — "bodyweight 146 this morning", "hit a 405x1 deadlift PR".
  It shows you what it's about to record and waits for your **yes/no** before
  writing.
- **Ask for a program** — "write me a 4-week strength block based on this prep".
  It studies your history and drafts a full block; you review it and choose to
  save it as a draft (drafts stay out of analysis until you start them) or
  discard it.
- **Just chat** — general training talk.

**Adding files** — use the sidebar uploader for a `.txt`, `.xlsx`, or `.pdf`:

- *Ingest as training log* runs it through review (see below) before anything is
  saved.
- *Add to knowledge base* embeds reference material (a study, an article)
  directly so the coach can cite it later — no review, since it isn't your
  training data. Any metadata you don't fill in (title, author, year, topic) is
  guessed from the document.

**Review before save** — whenever the coach is about to write to your history
(a parsed log, a reported stat, a new draft), it pauses and shows you exactly
what it will do. Click **✅ Approve** / **❌ Reject**, or type a correction in
plain English ("the third set was 315 not 335", "these belong to Strength Block
1") and it re-does the parse. Nothing durable is written until you approve.

### 📈 Trends

Your training at a glance — interactive time-series charts over everything the
coach knows, filtered by a shared date range (defaults to the last 6 months):

- **Bodyweight** — line chart plus first / last / change / min / max summary
  numbers.
- **1RM per lift** — pick a lift (main lifts by default, toggle to show
  accessories): your weekly best **estimated 1RM** as a line, with **recorded
  PRs** overlaid as points (true 1RMs by default; untick to include rep PRs).
  Hover any point to see the exact set behind it. Bucket by block instead to
  compare across mesocycles.
- **Measurements** — limb-circumference / length sites over time, overlaid on
  one chart.
- **Volume** — hard sets and tonnage per week or per block, for a whole muscle
  group or a single exercise.

Draft programs never pollute these charts — only training you actually logged
counts.

### 🗂️ Organizer

Fix how your training is organized after the fact, so getting block assignment
perfect at ingest time never matters:

- **Reattach sessions** to a different block, or detach them entirely.
- **Rename, merge, or move** programs and blocks (e.g. fold a mistakenly-split
  block back together, or move a block to another program).
- **Start a draft** — flip a program the coach drafted from `draft` to active
  (with a start date) so its sessions start counting in your analysis.

Every action is a single button; the tables above show live counts so you can
see what you're working with.

### 📥 Backfill

For loading a lot of history at once — paste (or upload) months or years of
training in one go. The app splits it into chunks along session boundaries and
runs each through the same parse-and-review pipeline. Two modes:

- **Stage for review** *(default)* — every chunk is queued as a pending batch;
  you review and commit each one below, attaching it to a block if you like.
  Same careful review as single-file ingest, just batched.
- **Commit everything** — the trusted-bulk path: one click ingests the whole
  archive at once. Each chunk is still recorded in the audit trail, and a chunk
  that fails to parse is skipped rather than aborting the run.

### 🛠️ Dev Tools

Direct maintenance access to the database, outside the coach's review flow:

- **Edit any table directly** — an editable grid where you can add rows, fix
  cells, or delete rows, then apply the changes. Useful for one-off corrections.
- **Back up the database** — one click writes a consistent snapshot to
  `data/backups/`.
- **Browse the ingestion audit trail** — every upload ever staged, with its
  status (pending / committed / rejected) and the parsed data it produced.

---

## Configuration

Model routing lives in [`config.yaml`](config.yaml). Each stage of the app
(routing, extraction, analysis, program generation, embeddings…) picks its model
independently, so you can tune cost/quality per task or move a single node to
the cloud without touching code:

```yaml
nodes:
  analyze:
    provider: local          # 'local' (Ollama) or 'cloud' (Anthropic)
    model: qwen3.6:35b-a3b
    host: http://localhost:11434
  generate:
    provider: cloud          # program writing defaults to cloud
    model: claude-sonnet-5
    api_key_env: ANTHROPIC_API_KEY
```

- To run **fully offline**, set `generate.provider: local` (and give it a local
  model + host). An API key is **never required** while every node is `local`.
- `display_unit: lb` — weights are stored canonically in pounds and only
  converted for display. Set to `kg` to show kilos.
- `db_path`, `chroma_path`, `checkpoints_db` — where your data lives.

---

## Command-line alternative

The same coach is available as a terminal REPL, if you prefer it to the browser:

```bash
python -m src.cli
```

`/ingest <path>` stages a training log for review; `/learn <path> [--topic ...]`
adds reference material; anything else is chat. The review round-trip works the
same way — approve, reject, or type corrections.

---

## How it works

A brief tour of the design; see [`ARCHITECTURE.md`](ARCHITECTURE.md) for the
full picture.

- **Hybrid storage.** Structured numbers (sessions, sets, PRs, bodyweight,
  programs) live in **SQLite**; prose (session notes, saved analyses, reference
  material) lives in a **Chroma** vector store for semantic search. Quantitative
  questions hit typed SQL tools; "did I ever mention…" questions hit similarity
  search.
- **Agent graph.** A [LangGraph](https://langchain-ai.github.io/langgraph/)
  state machine routes each message to one of five paths — ingest, analyze,
  generate, update-stats, or chat. Analysis and generation are bounded **ReAct
  loops** over a set of typed query tools (best set, e1RM trend, volume,
  frequency, PRs, injuries, programmed-vs-actual, note/knowledge search), so
  every claim is backed by a real query, not a guess.
- **Human-in-the-loop by construction.** Nothing reaches your training tables
  without passing through an `interrupt()` you approve. The graph checkpoints to
  SQLite, so a review can span page reloads. (The Dev Tools tab and bulk-commit
  backfill are the two explicit exceptions — hand edits you're making
  deliberately.)
- **Extraction pipeline.** Raw log text → a schema-validated `ParsedBatch` (via
  a local LLM) → a staged, confidence-flagged review → a transactional commit.
  Loaders handle `.txt`, `.xlsx` (one block per sheet, no layout assumptions),
  and `.pdf` (one block per page).
- **Provider seams.** Every LLM / embedder / vector-store dependency sits behind
  an injectable factory, which is what lets the whole thing be tested with
  stubs and in-memory stores — no live models in the test suite.

---

## Development

```bash
pytest        # 253 tests, in-memory SQLite seeded per test, no live models
```

Project layout:

| Path | What's there |
|------|--------------|
| `src/db/` | SQLite schema + connection helpers |
| `src/tools/` | Typed query tools, semantic search, SQL escape hatch, organizer + admin ops |
| `src/ingest/` | Loaders, LLM extraction, HITL staging/commit, embedding, knowledge base, backfill |
| `src/agent/` | LangGraph graph, nodes (router/analyze/synthesize/generate/…), provider routing |
| `src/ui/` | Streamlit app + tab modules (rendering) and streamlit-free logic (driver, editor diff, chart prep) |
| `src/cli.py` | Terminal REPL |
| `tests/` | Full suite; stubs/fakes for every model dependency |

**Remaining optional polish** (see [`IMPLEMENTATION_ROADMAP.md`](IMPLEMENTATION_ROADMAP.md)
Stage 11): a full `display_unit: kg` pass through the UI, block-review /
form-cue embedding paths, and a re-embedding command for swapping embedders.
