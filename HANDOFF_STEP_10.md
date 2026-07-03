# Handoff — Step 10: Trends tab (UI graphs)

For the next Claude session. Read `ARCHITECTURE.md` (full design) and the rest
of this file for what's built. `IMPLEMENTATION_ROADMAP.md` Stages 1–10 are
**complete**; **Stage 11** (deferred polish: kg pass, review/form-cue
embedding, re-embed command) is the planned remaining work — its full spec is
in the roadmap.

## Where things stand (Steps 1–10 — DONE, 253 tests passing)

Steps 1–9 unchanged (data spine, LLM extraction, HITL staging/commit, full tool
layer, LangGraph core + CLI, ANALYZE/SYNTHESIZE/UPDATE_STATS, GENERATE + cloud
offload, file loaders + knowledge base, Streamlit UI + organizer + backfill +
dev tools). Step 10 adds the **📈 Trends** tab, second in the tab order:

- **`src/ui/trends.py`** (streamlit-free, unit-tested in
  `tests/test_trends.py`): all chart-prep logic —
  - `default_date_range(today, months=6)` → ISO `(from, to)` pair, month
    arithmetic with short-month clamping (`DEFAULT_RANGE_MONTHS = 6` was the
    resolved open decision);
  - `list_exercises(conn, main_lifts_only=True)` (tiers
    `competition`/`variation` by default) and `list_measurement_sites(conn)`
    for the selectors;
  - frame builders `bodyweight_frame` / `e1rm_frame` / `pr_frame` /
    `measurement_frame` / `volume_frame`: query-tool Pydantic outputs →
    chart-ready pandas DataFrames. Empty inputs yield empty frames **with the
    expected columns**, so the veneer branches on `.empty` only.
- **`src/ui/trends_tab.py`** (rendering veneer, import-smoke-tested): shared
  date-range pickers, then four sections —
  - **Bodyweight**: `get_bodyweight_trend` → metrics row
    (first/last/change/min/max) + Altair line;
  - **1RM per lift**: exercise selectbox ("show all exercises" widens past the
    SBD tiers), week/block bucket radio, "1RM PRs only" checkbox. Week mode is
    a **layered chart** — `get_e1rm_trend` line charted on the *source-set
    date* (temporal x via `e1rm_frame`'s `x` column) with `get_prs` scatter
    points on top. Block mode charts e1RM on ordinal block buckets and **skips
    the PR overlay** (block names aren't temporal; the caption says to switch
    to week bucketing for PRs);
  - **Measurements**: site multiselect (default all) → one line per site;
  - **Volume**: muscle-group-or-exercise radio + selectbox, week/block bucket
    → side-by-side hard-sets and tonnage bar charts. Muscle groups come from
    the new public **`MUSCLE_GROUPS`** sorted list in `src/tools/queries.py`
    (the private `_MUSCLE_GROUPS` set stays the membership test inside
    `get_volume_trend`).
- **`src/ui/app.py`**: trends_tab registered ("📈 Trends", second tab).
- **`pyproject.toml`**: `altair>=5.0` declared (it was already a Streamlit
  transitive dep; we now import it directly).
- Tests (13 new, 253 total): `tests/test_trends.py` (date-range math incl.
  clamping and year boundary, selector helpers, every frame builder against
  the seeded DB, empty-frame column contract); trends_tab added to
  `test_ui.py::test_tab_modules_import`.

## Decisions made this step (user-confirmed 2026-07-03, now locked in)

1. **Altair** for charting (layered PR-over-e1RM charts need more than
   `st.line_chart`; zero effective new deps).
2. **1RM graphs = recorded PRs + e1RM overlay** — `pr` rows as points over the
   `get_e1rm_trend` line.
3. **Volume shows both** hard sets and tonnage (tool already returned both).
4. **Default date range: last 6 months** (resolved at implementation time).

## Implementation notes / gotchas for whoever builds on this

- **Keep the logic/veneer split**: anything decision-shaped goes in
  `src/ui/trends.py` (or a sibling streamlit-free module) with tests; the
  `*_tab.py` modules stay import-smoke-only.
- **`e1rm_frame`'s `x` column is mode-dependent** — source-set date (string,
  charted as `:T`) in week mode, block name (charted as `:N`, `sort=None`) in
  block mode. The PR overlay is only valid in week mode; don't layer it onto
  ordinal block buckets.
- **Week buckets sort lexically** (`2026-W05` < `2026-W12`) because
  `_iso_week_bucket` zero-pads — rely on that, don't re-sort frames.
- **Streamlit 1.58 API**: use `width="stretch"` (not the deprecated
  `use_container_width=True`) on charts/dataframes, matching the other tabs.
- **Empty-frame contract**: frame builders always return the full column set
  even when empty; the veneer checks `.empty` and renders `st.info`. Keep that
  contract for any new frame builder.
- Charts read canonical lb straight from the tools — the Stage 11a kg pass is
  expected to convert inside the frame builders via
  `src/agent/units.py::to_display_weight` (measurements stay inches).
- The preview `launch.json` runs against the **live** `data/training.db`.

## Must handle / preserve (carried forward, still true)

- No live models in tests — every LLM/embedder/Chroma dependency stays behind
  an injectable seam. The Trends tab needs none (pure SQL tools).
- HITL invariant unchanged: nothing durable to `session`/`lift_set`/`pr`/etc.
  outside an approved interrupt branch (dev-tools CRUD, bulk-backfill
  commit-all, and knowledge ingest remain the explicit user-action
  exemptions). Trends is read-only.
- lb canonical; unit conversion only at presentation.
- Draft programs excluded from analysis by default — the trend charts inherit
  this from the query tools; don't bypass them with raw SQL in the UI.
- Seeder only touches `data/sample.db`; `data/training.db` is live; checkpoints
  in `data/checkpoints.db`.
- Never hard-require an API key when `provider: local`.

## What's left — Stage 11 (spec'd in `IMPLEMENTATION_ROADMAP.md`)

- **11a** `display_unit: kg` end-to-end (review renderer, generate drafts,
  Trends unit toggle, organizer listings; Dev Tools stays lb).
- **11b** Block-review / form-cue embedding paths (Organizer text areas + an
  UPDATE_STATS-style chat path; `doc_type='block_review'|'form_cue'`).
- **11c** Re-embedding command + embedder name recorded in Chroma collection
  metadata.
