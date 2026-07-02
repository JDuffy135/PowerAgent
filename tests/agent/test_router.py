"""Router node tests: classification parsing, fallback, and preset-intent skip."""
from __future__ import annotations

from langchain_core.messages import HumanMessage

from src.agent.nodes.router import classify_intent, make_router_node
from tests.agent.conftest import RaisingChatModel, StubChatModel


def test_classify_each_intent():
    for intent in ("ingest", "analyze", "generate", "update_stats", "chat"):
        model = StubChatModel(f'{{"intent": "{intent}"}}')
        assert classify_intent("some message", model) == intent


def test_classify_tolerates_fenced_json():
    model = StubChatModel('Sure!\n```json\n{"intent": "analyze"}\n```')
    assert classify_intent("how's my bench?", model) == "analyze"


def test_classify_falls_back_to_chat_on_garbage():
    assert classify_intent("hi", StubChatModel("not json at all")) == "chat"
    assert classify_intent("hi", StubChatModel('{"intent": "banana"}')) == "chat"
    assert classify_intent("hi", StubChatModel('{"wrong_key": 1}')) == "chat"


def test_router_node_skips_llm_when_intent_preset():
    node = make_router_node(lambda: RaisingChatModel())
    result = node({"intent": "ingest", "messages": [HumanMessage(content="/ingest x.txt")]})
    assert result == {}


def test_router_node_classifies_last_message():
    node = make_router_node(lambda: StubChatModel('{"intent": "generate"}'))
    result = node({"intent": None, "messages": [HumanMessage(content="write me a block")]})
    assert result == {"intent": "generate"}
