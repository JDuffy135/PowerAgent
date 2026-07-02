"""CLI input parsing + interrupt round-trip driver tests (no live models)."""
from __future__ import annotations

from src.cli import make_input


def test_make_input_ingest_command_presets_intent_and_path():
    graph_input = make_input("/ingest logs/w3d2.txt")
    assert graph_input["intent"] == "ingest"
    assert graph_input["file_path"] == "logs/w3d2.txt"


def test_make_input_ingest_without_path():
    graph_input = make_input("/ingest")
    assert graph_input["intent"] == "ingest"
    assert graph_input["file_path"] is None


def test_make_input_plain_message_goes_to_router():
    graph_input = make_input("what was my best bench in March?")
    assert graph_input["intent"] is None
    assert graph_input["file_path"] is None
    assert graph_input["messages"][0].content == "what was my best bench in March?"
