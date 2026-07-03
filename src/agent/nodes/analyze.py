"""ANALYZE node: a bounded ReAct loop over the typed query tools (ARCHITECTURE.md §4.2).

The model is given the tool-bound question and drives a gather-evidence loop:
each turn it either calls tools (whose results are appended as `ToolMessage`s and
accumulated into `evidence`) or emits a final message, at which point the loop
stops and control passes to SYNTHESIZE.

**No `interrupt()` lives here**, so the loop can run entirely inside one node --
unlike INGEST, ANALYZE never pauses mid-loop, so there is no replay hazard. The
ReAct scratch messages (tool calls + tool results) are kept *local* to the node
and are NOT written back to `state["messages"]`: only structured `evidence` and
the final answer (composed by SYNTHESIZE) belong in the durable history, keeping
the router's view of the conversation clean.

**[DECISION] Evidence overflow:** the loop is capped at `MAX_TOOL_CALLS`. If the
model is still requesting tools at the cap, we stop and set
`evidence_truncated=True`; SYNTHESIZE then answers with the partial evidence,
adds a disclaimer, and suggests narrowing the question.

**[DECISION] Model:** ANALYZE runs on the Qwen3.6 35B-A3B MoE (config
`nodes.analyze`); step down to the 14B if latency bottlenecks. The model is
resolved lazily via a factory so tests inject a scripted tool-calling stub.
"""
from __future__ import annotations

import json
import sqlite3
from datetime import date as date_cls
from typing import Callable

from langchain_core.messages import SystemMessage, ToolMessage

from src.agent.state import AgentState
from src.agent.tools import make_analyze_tools

MAX_TOOL_CALLS = 8  # [DECISION] evidence-gathering cap; see module docstring

ANALYZE_SYSTEM_PROMPT = """You are the analysis engine for a powerlifting-coach assistant. \
Answer the user's question about their training history by calling the provided tools to \
gather evidence -- never invent numbers.

Rules:
- Today's date is {today}. All dates are ISO YYYY-MM-DD. Weights in the database are pounds.
- Prefer the typed tools (get_best_set, get_e1rm_trend, get_volume_trend, ...). Only use \
run_sql when no typed tool fits.
- Resolve relative time ("this prep", "last two blocks", "in March") to concrete date \
windows before calling a tool. When unsure of the span, prefer a wider window.
- If a tool returns {{"error": ...}} because an exercise name didn't resolve, try an \
obvious alternative name once; if it still fails, stop and say so.
- Call tools one or a few at a time. When you have enough evidence, STOP calling tools and \
reply with a brief note that you're done -- a separate step writes the final answer.
"""


def make_analyze_node(
    conn: sqlite3.Connection,
    model_factory: Callable[[], object],
    *,
    embedder=None,
    chroma_client=None,
    today: str | None = None,
):
    tools = make_analyze_tools(conn, embedder=embedder, chroma_client=chroma_client)
    tools_by_name = {t.name: t for t in tools}

    def analyze(state: AgentState) -> dict:
        today_str = today or date_cls.today().isoformat()
        llm = model_factory().bind_tools(tools)

        scratch = [
            SystemMessage(content=ANALYZE_SYSTEM_PROMPT.format(today=today_str)),
            *state["messages"],
        ]

        evidence: list[dict] = []
        truncated = False

        for _ in range(MAX_TOOL_CALLS):
            ai = llm.invoke(scratch)
            scratch.append(ai)

            tool_calls = getattr(ai, "tool_calls", None) or []
            if not tool_calls:
                break

            for call in tool_calls:
                name, args, call_id = call["name"], call.get("args", {}), call.get("id")
                tool = tools_by_name.get(name)
                if tool is None:
                    result = {"error": f"unknown tool {name!r}"}
                else:
                    try:
                        result = tool.invoke(args)
                    except Exception as exc:  # bad args, etc. -- feed back, don't crash the turn
                        result = {"error": f"{type(exc).__name__}: {exc}"}
                evidence.append({"tool": name, "args": args, "result": result})
                scratch.append(
                    ToolMessage(content=json.dumps(result, default=str), tool_call_id=call_id)
                )
        else:
            # Loop exhausted while the model was still requesting tools.
            truncated = True

        return {"evidence": evidence, "evidence_truncated": truncated}

    return analyze
