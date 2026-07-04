# LangFuse (self-hosted) — agent tracing

Every chat turn (Streamlit or CLI) and every raw structured-output LLM call is
traced to a local LangFuse instance so you can inspect the agent's reasoning
step-by-step: router classification, ReAct tool calls with their arguments and
results, every LLM generation (prompts, outputs, model, latency), and HITL
interrupt round-trips grouped per conversation.

## Start the stack

Requires Docker with the compose plugin
(`sudo apt install docker.io docker-compose-v2` on Ubuntu, or Docker Desktop).

```bash
docker compose -f observability/docker-compose.yml up -d
```

First boot takes a minute (ClickHouse migrations). The stack is the official
LangFuse v3 compose file (web + worker + Postgres + ClickHouse + Redis +
MinIO), unmodified; all secrets and the headless-init settings live in
[`observability/.env`](.env), which Compose loads automatically. Everything
binds to localhost only.

- **UI:** http://localhost:3000 — log in with
  `jaketduffy@comcast.net` / `poweragent-langfuse` (set in `.env`).
- The org/project and API keys are provisioned automatically on first boot via
  the `LANGFUSE_INIT_*` variables — no clicking through setup.

Stop with `docker compose -f observability/docker-compose.yml down`
(add `-v` to also wipe the trace data volumes).

## Point the app at it

Tracing is controlled by the `langfuse:` section of [`config.yaml`](../config.yaml)
and is a silent no-op when disabled or when the key env vars are missing —
the app (and the test suite) never requires LangFuse to be running.

```bash
export LANGFUSE_PUBLIC_KEY=pk-lf-2e948b29e8285746bbc7f1f9e0f96c6f   # from .env
export LANGFUSE_SECRET_KEY=sk-lf-91a386f2badd0c26820c05b391d67ca9   # from .env
streamlit run src/ui/app.py       # or: python -m src.cli
```

## Reading traces

- **Traces** view: one trace per graph turn. Expand it to see the node tree —
  `router` → `analyze`/`generate`/… — with each `ChatOllama`/`ChatAnthropic`
  generation and each tool call (args + JSON result) as children. Raw
  structured-output calls appear as generations named after their node
  (`ingest_extract`, `ingest_correct`, `update_stats_parse`, `generate_draft`,
  `knowledge_metadata`).
- **Sessions** view: a whole conversation. The LangGraph `thread_id` is the
  LangFuse session id, so a HITL round-trip (parse → interrupt → your
  approve/reject → commit) reads as one timeline even though it spans several
  graph invocations.
- **Tags**: every trace is tagged `streamlit` or `cli` by entry point.
- Standalone LLM work outside a chat turn (Backfill tab chunks, `/learn`
  metadata guessing) shows up as its own top-level traces.

## Updating LangFuse

`docker-compose.yml` is vendored verbatim from
<https://github.com/langfuse/langfuse/blob/main/docker-compose.yml>; to
upgrade, re-download it and `docker compose ... up -d` again. Local overrides
live only in `.env`, so upstream updates never conflict.
