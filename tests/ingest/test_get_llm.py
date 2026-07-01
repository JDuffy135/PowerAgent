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


def test_get_llm_cloud_provider_not_implemented(monkeypatch, tmp_path):
    config = tmp_path / "config.yaml"
    config.write_text("nodes:\n  ingest_extract:\n    provider: cloud\n")
    monkeypatch.setattr(extract, "CONFIG_PATH", config)

    with pytest.raises(NotImplementedError):
        extract.get_llm("ingest_extract")


def test_get_llm_connection_error_raises_runtime_error(monkeypatch):
    import urllib.error

    def fake_urlopen(request, timeout=None):
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr(extract.urllib.request, "urlopen", fake_urlopen)

    call = extract.get_llm("ingest_extract")
    with pytest.raises(RuntimeError):
        call("text")
