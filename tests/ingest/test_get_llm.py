"""Tests for the get_llm() provider seam (ARCHITECTURE.md §6.3).

No live Ollama server is required: `urllib.request.urlopen` is monkeypatched
so these tests check request shape/config routing, not model output.
"""
from __future__ import annotations

import json

import pytest

from src.ingest import extract


class _FakeResponse:
    def __init__(self, body: dict):
        self._body = json.dumps(body).encode("utf-8")

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_get_llm_local_calls_ollama_chat_endpoint(monkeypatch):
    captured = {}

    def fake_urlopen(request, timeout=None):
        captured["url"] = request.full_url
        captured["payload"] = json.loads(request.data)
        return _FakeResponse({"message": {"content": '{"sessions": []}'}})

    monkeypatch.setattr(extract.urllib.request, "urlopen", fake_urlopen)

    call = extract.get_llm("ingest_extract")
    result = call("some raw log text")

    assert result == '{"sessions": []}'
    assert captured["url"] == "http://localhost:11434/api/chat"
    assert captured["payload"]["model"] == "qwen3:14b"
    assert captured["payload"]["messages"][-1]["content"] == "some raw log text"
    assert "format" in captured["payload"]  # structured-output schema passed through


def test_get_llm_unknown_node_falls_back_to_defaults(monkeypatch):
    def fake_urlopen(request, timeout=None):
        return _FakeResponse({"message": {"content": "{}"}})

    monkeypatch.setattr(extract.urllib.request, "urlopen", fake_urlopen)

    call = extract.get_llm("some_node_not_in_config")
    call("text")  # should not raise; falls back to local/default model/host


def _cloud_config(monkeypatch, tmp_path, extra: str = ""):
    config = tmp_path / "config.yaml"
    config.write_text(
        "nodes:\n  ingest_extract:\n    provider: cloud\n    model: claude-sonnet-5\n" + extra
    )
    monkeypatch.setattr(extract, "CONFIG_PATH", config)


def test_get_llm_cloud_missing_key_raises_at_build_time(monkeypatch, tmp_path):
    _cloud_config(monkeypatch, tmp_path)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        extract.get_llm("ingest_extract")


def test_get_llm_cloud_calls_anthropic_with_fake_client(monkeypatch, tmp_path):
    """Cloud branch unit test with a fake transport — no real API calls."""
    _cloud_config(monkeypatch, tmp_path)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-123")

    captured = {}

    class _Block:
        type = "text"
        text = '```json\n{"sessions": []}\n```'  # fences must be stripped

    class _Response:
        stop_reason = "end_turn"
        content = [_Block()]

    class _FakeMessages:
        def create(self, **kwargs):
            captured.update(kwargs)
            return _Response()

    class _FakeClient:
        messages = _FakeMessages()

    def fake_client_factory(api_key):
        captured["api_key"] = api_key
        return _FakeClient()

    monkeypatch.setattr(extract, "_anthropic_client", fake_client_factory)

    call = extract.get_llm("ingest_extract")
    result = call("some raw log text")

    assert result == '{"sessions": []}'
    assert captured["api_key"] == "sk-test-123"
    assert captured["model"] == "claude-sonnet-5"
    assert captured["messages"] == [{"role": "user", "content": "some raw log text"}]
    # The schema rides in the system prompt (downstream Pydantic is the contract).
    assert "JSON schema" in captured["system"]
    assert "sessions" in captured["system"]


def test_get_llm_cloud_custom_api_key_env(monkeypatch, tmp_path):
    _cloud_config(monkeypatch, tmp_path, extra="    api_key_env: MY_CLOUD_KEY\n")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("MY_CLOUD_KEY", raising=False)

    with pytest.raises(RuntimeError, match="MY_CLOUD_KEY"):
        extract.get_llm("ingest_extract")


def test_get_llm_connection_error_raises_runtime_error(monkeypatch):
    import urllib.error

    def fake_urlopen(request, timeout=None):
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr(extract.urllib.request, "urlopen", fake_urlopen)

    call = extract.get_llm("ingest_extract")
    with pytest.raises(RuntimeError):
        call("text")
