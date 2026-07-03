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


def test_get_chat_model_cloud_builds_chat_anthropic(monkeypatch, tmp_path):
    config = tmp_path / "config.yaml"
    config.write_text(
        "nodes:\n  generate:\n    provider: cloud\n    model: claude-sonnet-5\n"
    )
    monkeypatch.setattr(llm_provider, "CONFIG_PATH", config)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-123")

    model = llm_provider.get_chat_model("generate")
    assert type(model).__name__ == "ChatAnthropic"
    assert model.model == "claude-sonnet-5"
    assert model.anthropic_api_key.get_secret_value() == "sk-test-123"


def test_get_chat_model_cloud_missing_key_raises(monkeypatch, tmp_path):
    config = tmp_path / "config.yaml"
    config.write_text("nodes:\n  generate:\n    provider: cloud\n")
    monkeypatch.setattr(llm_provider, "CONFIG_PATH", config)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        llm_provider.get_chat_model("generate")


def test_get_chat_model_unknown_provider_raises(monkeypatch, tmp_path):
    config = tmp_path / "config.yaml"
    config.write_text("nodes:\n  router:\n    provider: mainframe\n")
    monkeypatch.setattr(llm_provider, "CONFIG_PATH", config)

    with pytest.raises(ValueError, match="mainframe"):
        llm_provider.get_chat_model("router")


def test_get_llm_reexported():
    from src.ingest.extract import get_llm

    assert llm_provider.get_llm is get_llm
