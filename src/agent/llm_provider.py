"""Shared per-node model provider (ARCHITECTURE.md §6.3).

Every agent-graph node fetches its model here, so flipping a node between local
Ollama and a cloud provider is a `config.yaml` edit (`nodes.<node>.provider`),
never a call-site change. The cloud branch lands in Stage 7; until then any
`provider` other than `local` raises with a clear message.

Two seams, matching the two kinds of LLM use in the system:

- `get_llm(node, system_prompt=..., schema=...)` -- the raw `prompt -> JSON str`
  callable used by structured-output pipelines (extraction, HITL correction).
  Re-exported from `src.ingest.extract`, which established it in Step 2; this
  module is the canonical import point going forward.
- `get_chat_model(node)` -- a LangChain `BaseChatModel` (**[DECISION]**
  `langchain-ollama`'s `ChatOllama`) for graph nodes that converse or will
  drive tool-calling in Stage 6 (router, chitchat, ANALYZE, GENERATE).

Tests never call either: node factories accept injected stubs.
"""
from __future__ import annotations

from pathlib import Path

import yaml

from src.ingest.extract import get_llm  # noqa: F401  (re-export: the raw callable seam)

CONFIG_PATH = Path(__file__).parent.parent.parent / "config.yaml"

DEFAULT_OLLAMA_HOST = "http://localhost:11434"
DEFAULT_CHAT_MODEL = "qwen3:14b"


def load_config() -> dict:
    if not CONFIG_PATH.exists():
        return {}
    return yaml.safe_load(CONFIG_PATH.read_text()) or {}


def _node_config(node: str) -> dict:
    return load_config().get("nodes", {}).get(node, {}) or {}


def get_chat_model(node: str):
    """Return a `BaseChatModel` for the given graph node (local Ollama only for now).

    Imported lazily so stub-injected tests never touch langchain-ollama.
    """
    cfg = _node_config(node)
    provider = cfg.get("provider", "local")
    if provider != "local":
        raise NotImplementedError(
            f"Provider {provider!r} is not wired up yet (cloud lands in Stage 7)"
        )

    from langchain_ollama import ChatOllama

    return ChatOllama(
        model=cfg.get("model", DEFAULT_CHAT_MODEL),
        base_url=cfg.get("host", DEFAULT_OLLAMA_HOST),
    )
