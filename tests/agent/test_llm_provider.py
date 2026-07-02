"""Provider tests: config-driven chat-model construction, cloud raises, re-export."""
from __future__ import annotations

import pytest

from src.agent import llm_provider


def test_get_chat_model_reads_node_config(monkeypatch, tmp_path):
    config = tmp_path / "config.yaml"
    config.write_text(
        "nodes:\n  router:\n    provider: local\n    model: qwen3:4b\n"
        "    host: http://example:1234\n"
    )
    monkeypatch.setattr(llm_provider, "CONFIG_PATH", config)

    model = llm_provider.get_chat_model("router")
    assert model.model == "qwen3:4b"
    assert model.base_url == "http://example:1234"


def test_get_chat_model_defaults_for_unknown_node(monkeypatch, tmp_path):
    monkeypatch.setattr(llm_provider, "CONFIG_PATH", tmp_path / "missing.yaml")
    model = llm_provider.get_chat_model("nonexistent_node")
    assert model.model == llm_provider.DEFAULT_CHAT_MODEL


def test_get_chat_model_cloud_not_implemented(monkeypatch, tmp_path):
    config = tmp_path / "config.yaml"
    config.write_text("nodes:\n  router:\n    provider: cloud\n")
    monkeypatch.setattr(llm_provider, "CONFIG_PATH", config)

    with pytest.raises(NotImplementedError):
        llm_provider.get_chat_model("router")


def test_get_llm_reexported():
    from src.ingest.extract import get_llm

    assert llm_provider.get_llm is get_llm
