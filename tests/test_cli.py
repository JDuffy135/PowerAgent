"""CLI input parsing + interrupt round-trip driver tests (no live models)."""
from __future__ import annotations

from pathlib import Path

from src.cli import make_input, parse_learn, run_learn

FIXTURES_DIR = Path(__file__).parent / "ingest" / "fixtures"


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


# ---------------------------------------------------------------------------
# /learn parsing + handler
# ---------------------------------------------------------------------------

def test_parse_learn_path_and_flags():
    path, doc = parse_learn("/learn study.pdf --topic hypertrophy --year 2024 --author 'J. Coach'")
    assert path == "study.pdf"
    assert doc.topic == "hypertrophy"
    assert doc.year == 2024
    assert doc.author == "J. Coach"
    assert doc.title is None  # omitted -> stays None for the LLM guess pass


def test_parse_learn_bad_year_is_dropped():
    path, doc = parse_learn("/learn x.pdf --year notanumber")
    assert path == "x.pdf"
    assert doc.year is None


def test_parse_learn_no_path():
    path, _ = parse_learn("/learn")
    assert path is None


def test_run_learn_embeds_file(fake_embedder, chroma_client):
    from src.ingest.knowledge import KNOWLEDGE_COLLECTION

    written = []
    run_learn(
        f"/learn {FIXTURES_DIR / 'study.pdf'} --topic testtopic",
        llm=None,  # skip the guess pass -> no live model
        embedder=fake_embedder,
        chroma_client=chroma_client,
        write=written.append,
    )
    assert any("learned" in line for line in written)
    collection = chroma_client.get_collection(KNOWLEDGE_COLLECTION)
    assert collection.count() >= 1
    meta = collection.get(include=["metadatas"])["metadatas"][0]
    assert meta["topic"] == "testtopic"
    assert meta["source"] == "study.pdf"


def test_run_learn_missing_path_prints_usage():
    written = []
    run_learn("/learn", write=written.append)
    assert any("usage" in line for line in written)


# ---------------------------------------------------------------------------
# /reembed command (Stage 11c)
# ---------------------------------------------------------------------------

def test_run_reembed_reports_counts(fake_embedder, chroma_client, monkeypatch):
    import src.ingest.reembed as reembed_mod
    from src.cli import run_reembed
    from src.ingest.embed import PERSONAL_NOTES_COLLECTION

    col = chroma_client.get_or_create_collection(PERSONAL_NOTES_COLLECTION)
    col.add(ids=["x"], documents=["squat day"], embeddings=[[0.0] * 8],
            metadatas=[{"doc_type": "session_note"}])

    # Route the real seams to the in-memory fakes -- no live Ollama/Chroma.
    monkeypatch.setattr(reembed_mod, "get_embedder", lambda: fake_embedder)
    monkeypatch.setattr(reembed_mod, "get_chroma_client", lambda: chroma_client)
    monkeypatch.setattr(reembed_mod, "embedder_name", lambda: "fake-model")

    written: list[str] = []
    run_reembed(write=written.append)
    assert any("re-embedded" in line for line in written)
    assert any("done" in line for line in written)
