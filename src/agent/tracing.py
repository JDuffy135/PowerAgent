"""LangFuse tracing seam (see observability/README.md).

Everything observability-related goes through this module, and everything in
it is a **silent no-op** when tracing is off — `langfuse` is only imported
once tracing is confirmed enabled, so the app and the test suite never
require the package to be importable or the server to be running (house
rule: no live services in tests).

Tracing is on when all three hold:
- `config.yaml` has `langfuse.enabled: true`;
- the env vars named by `langfuse.public_key_env` / `langfuse.secret_key_env`
  (default `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY`) are set;
- the `langfuse` package imports.

Two integration points, matching the two LLM seams:

- `attach_tracing(config, thread_id=..., source=...)` — adds the LangChain
  `CallbackHandler` + session metadata to a `graph.invoke` config, so every
  graph node, chat-model call, and `tool.invoke` is traced. The LangGraph
  `thread_id` doubles as the LangFuse session id, grouping a whole HITL
  conversation (multiple invokes) into one session timeline.
- `traced_llm(call, node=..., ...)` — wraps a raw `prompt -> str` callable
  (the `get_llm` seam) in a generation span. The SDK is OTel-based, so a
  wrapped call made *inside* a traced graph turn nests under that trace;
  standalone calls (Backfill chunks, `/learn` metadata) become their own
  root traces.

Tests inject a fake client via `_set_client_for_tests`.
"""
from __future__ import annotations

import os
import warnings
from typing import Callable

DEFAULT_PUBLIC_KEY_ENV = "LANGFUSE_PUBLIC_KEY"
DEFAULT_SECRET_KEY_ENV = "LANGFUSE_SECRET_KEY"
DEFAULT_HOST = "http://localhost:3000"

_UNSET = object()
_client = _UNSET  # resolved Langfuse client, None (disabled), or _UNSET (not yet tried)
_handler = None
_warned = False


def _langfuse_config() -> dict:
    # Imported lazily: llm_provider itself imports src.ingest.extract, which
    # imports this module — a top-level import here would be circular.
    from src.agent.llm_provider import load_config

    return load_config().get("langfuse", {}) or {}


def _warn_once(message: str) -> None:
    global _warned
    if not _warned:
        warnings.warn(f"LangFuse tracing disabled: {message}", stacklevel=3)
        _warned = True


def _set_client_for_tests(client) -> None:
    """Test seam: force the resolved client (and reset the cached handler)."""
    global _client, _handler
    _client = client
    _handler = None


def reset() -> None:
    """Forget the cached client/handler (tests; config reloads)."""
    global _client, _handler, _warned
    _client = _UNSET
    _handler = None
    _warned = False


def get_langfuse():
    """The singleton Langfuse client, or None when tracing is off.

    Resolution happens once; a config/env problem downgrades to a one-time
    warning, never an exception — tracing must not take the coach down.
    """
    global _client
    if _client is not _UNSET:
        return _client

    cfg = _langfuse_config()
    if not cfg.get("enabled"):
        _client = None
        return None

    public_key = os.environ.get(cfg.get("public_key_env", DEFAULT_PUBLIC_KEY_ENV))
    secret_key = os.environ.get(cfg.get("secret_key_env", DEFAULT_SECRET_KEY_ENV))
    if not public_key or not secret_key:
        _warn_once(
            "langfuse.enabled is true but the key env vars are unset "
            f"({cfg.get('public_key_env', DEFAULT_PUBLIC_KEY_ENV)} / "
            f"{cfg.get('secret_key_env', DEFAULT_SECRET_KEY_ENV)}; "
            "see observability/README.md)."
        )
        _client = None
        return None

    try:
        from langfuse import Langfuse
    except ImportError as exc:
        _warn_once(f"the langfuse package is not installed ({exc}).")
        _client = None
        return None

    _client = Langfuse(
        public_key=public_key,
        secret_key=secret_key,
        host=cfg.get("host", DEFAULT_HOST),
    )
    return _client


def tracing_enabled() -> bool:
    return get_langfuse() is not None


def get_callback_handler():
    """The (cached) LangChain callback handler, or None when tracing is off."""
    global _handler
    if get_langfuse() is None:
        return None
    if _handler is None:
        from langfuse.langchain import CallbackHandler

        _handler = CallbackHandler()
    return _handler


def attach_tracing(config: dict, *, thread_id: str, source: str) -> dict:
    """Return a copy of a `graph.invoke` config with tracing attached.

    Identity (same dict back) when tracing is off. `thread_id` becomes the
    LangFuse session id; `source` ('streamlit' | 'cli') is a trace tag.
    """
    handler = get_callback_handler()
    if handler is None:
        return config
    traced = dict(config)
    traced["callbacks"] = list(config.get("callbacks", [])) + [handler]
    traced["metadata"] = {
        **config.get("metadata", {}),
        "langfuse_session_id": thread_id,
        "langfuse_tags": [source],
    }
    return traced


def traced_llm(
    call: Callable[[str], str],
    *,
    node: str,
    model: str,
    provider: str,
    system_prompt: str | None = None,
) -> Callable[[str], str]:
    """Wrap a raw `prompt -> str` callable in a LangFuse generation span.

    Pass-through when tracing is off. Errors are recorded on the span and
    re-raised untouched.
    """
    if get_langfuse() is None:
        return call

    def _traced(prompt: str) -> str:
        client = get_langfuse()
        if client is None:  # keys removed after wrap; stay safe
            return call(prompt)
        with client.start_as_current_observation(
            name=node,
            as_type="generation",
            model=model,
            input={"system": system_prompt, "prompt": prompt},
            metadata={"node": node, "provider": provider},
        ) as generation:
            try:
                output = call(prompt)
            except Exception as exc:
                generation.update(level="ERROR", status_message=str(exc))
                raise
            generation.update(output=output)
            return output

    return _traced


def flush() -> None:
    """Flush buffered spans (end of a Streamlit turn / CLI exit). No-op when off."""
    client = _client if _client is not _UNSET else None
    if client is not None:
        client.flush()
