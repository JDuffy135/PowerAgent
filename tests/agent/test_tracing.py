"""LangFuse tracing seam (`src/agent/tracing.py`) — fakes only, no live LangFuse.

The house rule holds: nothing here imports the `langfuse` package or talks to
a server. The client is injected via `_set_client_for_tests`; the disabled
paths are exercised by monkeypatching the config loader.
"""
from __future__ import annotations

import warnings

import pytest

from src.agent import tracing


class FakeGeneration:
    def __init__(self):
        self.updates = []

    def update(self, **kwargs):
        self.updates.append(kwargs)


class FakeObservationCM:
    def __init__(self, client, kwargs):
        self.client = client
        self.kwargs = kwargs
        self.generation = FakeGeneration()

    def __enter__(self):
        return self.generation

    def __exit__(self, *exc):
        return False


class FakeLangfuse:
    def __init__(self):
        self.observations = []
        self.flushed = 0

    def start_as_current_observation(self, **kwargs):
        cm = FakeObservationCM(self, kwargs)
        self.observations.append(cm)
        return cm

    def flush(self):
        self.flushed += 1


@pytest.fixture(autouse=True)
def clean_state():
    tracing.reset()
    yield
    tracing.reset()


def _config_off(monkeypatch):
    monkeypatch.setattr(
        "src.agent.llm_provider.load_config", lambda: {"langfuse": {"enabled": False}}
    )


def _config_on(monkeypatch):
    monkeypatch.setattr(
        "src.agent.llm_provider.load_config", lambda: {"langfuse": {"enabled": True}}
    )


# ---------------------------------------------------------------- disabled --

def test_disabled_by_config(monkeypatch):
    _config_off(monkeypatch)
    assert tracing.get_langfuse() is None
    assert not tracing.tracing_enabled()
    assert tracing.get_callback_handler() is None


def test_enabled_without_keys_warns_once_and_noops(monkeypatch):
    _config_on(monkeypatch)
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert tracing.get_langfuse() is None
        assert tracing.get_langfuse() is None  # cached; no second warning
    assert len(caught) == 1
    assert "env vars are unset" in str(caught[0].message)


def test_attach_tracing_is_identity_when_disabled(monkeypatch):
    _config_off(monkeypatch)
    config = {"configurable": {"thread_id": "t1"}}
    assert tracing.attach_tracing(config, thread_id="t1", source="cli") is config


def test_traced_llm_is_passthrough_when_disabled(monkeypatch):
    _config_off(monkeypatch)

    def call(prompt):
        return "raw"

    assert tracing.traced_llm(call, node="n", model="m", provider="local") is call


def test_flush_noop_when_never_enabled(monkeypatch):
    _config_off(monkeypatch)
    tracing.flush()  # must not raise, must not build a client


# ----------------------------------------------------------------- enabled --

def test_attach_tracing_adds_handler_and_metadata():
    fake = FakeLangfuse()
    tracing._set_client_for_tests(fake)
    handler = object()
    tracing._handler = handler  # bypass the real CallbackHandler import

    config = {"configurable": {"thread_id": "t1"}, "recursion_limit": 100}
    traced = tracing.attach_tracing(config, thread_id="t1", source="streamlit")

    assert traced is not config  # original untouched
    assert "callbacks" not in config
    assert traced["callbacks"] == [handler]
    assert traced["metadata"]["langfuse_session_id"] == "t1"
    assert traced["metadata"]["langfuse_tags"] == ["streamlit"]
    assert traced["configurable"] == config["configurable"]
    assert traced["recursion_limit"] == 100


def test_attach_tracing_preserves_existing_callbacks_and_metadata():
    fake = FakeLangfuse()
    tracing._set_client_for_tests(fake)
    handler = object()
    tracing._handler = handler
    prior_cb = object()

    traced = tracing.attach_tracing(
        {"callbacks": [prior_cb], "metadata": {"keep": 1}},
        thread_id="t2",
        source="cli",
    )
    assert traced["callbacks"] == [prior_cb, handler]
    assert traced["metadata"]["keep"] == 1
    assert traced["metadata"]["langfuse_session_id"] == "t2"


def test_traced_llm_records_generation():
    fake = FakeLangfuse()
    tracing._set_client_for_tests(fake)

    wrapped = tracing.traced_llm(
        lambda prompt: f"out:{prompt}",
        node="ingest_extract",
        model="qwen3:14b",
        provider="local",
        system_prompt="sys",
    )
    assert wrapped("hello") == "out:hello"

    assert len(fake.observations) == 1
    cm = fake.observations[0]
    assert cm.kwargs["name"] == "ingest_extract"
    assert cm.kwargs["as_type"] == "generation"
    assert cm.kwargs["model"] == "qwen3:14b"
    assert cm.kwargs["input"] == {"system": "sys", "prompt": "hello"}
    assert cm.kwargs["metadata"]["provider"] == "local"
    assert cm.generation.updates == [{"output": "out:hello"}]


def test_traced_llm_records_error_and_reraises():
    fake = FakeLangfuse()
    tracing._set_client_for_tests(fake)

    def boom(prompt):
        raise RuntimeError("ollama down")

    wrapped = tracing.traced_llm(boom, node="n", model="m", provider="local")
    with pytest.raises(RuntimeError, match="ollama down"):
        wrapped("x")

    (cm,) = fake.observations
    (update,) = cm.generation.updates
    assert update["level"] == "ERROR"
    assert "ollama down" in update["status_message"]


def test_flush_flushes_injected_client():
    fake = FakeLangfuse()
    tracing._set_client_for_tests(fake)
    tracing.flush()
    assert fake.flushed == 1


# ----------------------------------------------------- get_llm integration --

def test_get_llm_passthrough_when_tracing_disabled(monkeypatch, tmp_path):
    """The raw seam still returns a plain working callable with tracing off."""
    _config_off(monkeypatch)
    from src.ingest import extract

    monkeypatch.setattr(extract, "_node_config", lambda node: {"provider": "local"})
    llm = extract.get_llm("ingest_extract")
    assert callable(llm)


def test_get_llm_wrapped_when_tracing_enabled(monkeypatch):
    """With a client active, get_llm returns the traced wrapper around _call."""
    fake = FakeLangfuse()
    tracing._set_client_for_tests(fake)
    from src.ingest import extract

    monkeypatch.setattr(extract, "_node_config", lambda node: {"provider": "local"})
    llm = extract.get_llm("ingest_extract")
    # Not invoked (that would hit Ollama) -- but the wrapper identity differs
    # from a plain local _call, and its closure carries the traced generation.
    assert llm.__name__ == "_traced"
