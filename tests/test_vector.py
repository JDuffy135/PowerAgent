import pytest

from src.ingest.embed import SessionNote, embed_session_notes
from src.ingest.knowledge import KnowledgeDoc, ingest_knowledge
from src.tools.vector import search_knowledge, search_notes


@pytest.fixture()
def seeded_notes(fake_embedder, chroma_client):
    notes = [
        SessionNote(
            session_id=1,
            date="2026-02-15",
            raw_note="Knee felt a bit off during squats today, backed off the last set.",
            exercises=["Low Bar Squat"],
        ),
        SessionNote(
            session_id=2,
            date="2026-03-01",
            raw_note="Bench felt strong today, no issues at all.",
            exercises=["Bench Press"],
        ),
        SessionNote(
            session_id=3,
            date="2026-06-01",
            raw_note="Hit a deadlift PR, felt great, pulldowns pin dropped mid-set.",
            exercises=["Deadlift", "MAG Grip Pulldowns"],
        ),
    ]
    embed_session_notes(notes, embedder=fake_embedder, client=chroma_client)
    return fake_embedder, chroma_client


def test_search_notes_requires_a_metadata_filter(seeded_notes):
    fake_embedder, chroma_client = seeded_notes
    with pytest.raises(ValueError):
        search_notes("knee pain", embedder=fake_embedder, client=chroma_client)


def test_search_notes_date_range_filter(seeded_notes):
    fake_embedder, chroma_client = seeded_notes
    results = search_notes(
        "knee pain",
        date_from="2026-01-01",
        date_to="2026-02-28",
        embedder=fake_embedder,
        client=chroma_client,
    )
    assert len(results) == 1
    assert results[0].session_id == 1


def test_search_notes_excludes_out_of_range_dates(seeded_notes):
    fake_embedder, chroma_client = seeded_notes
    results = search_notes(
        "anything",
        date_from="2026-05-01",
        date_to="2026-06-30",
        embedder=fake_embedder,
        client=chroma_client,
    )
    session_ids = {r.session_id for r in results}
    assert session_ids == {3}


def test_search_notes_exercise_filter(seeded_notes):
    fake_embedder, chroma_client = seeded_notes
    results = search_notes(
        "anything",
        exercises=["MAG Grip Pulldowns"],
        embedder=fake_embedder,
        client=chroma_client,
    )
    assert len(results) == 1
    assert results[0].session_id == 3


def test_search_notes_doc_type_filter(seeded_notes):
    fake_embedder, chroma_client = seeded_notes
    results = search_notes(
        "anything",
        doc_type="session_note",
        embedder=fake_embedder,
        client=chroma_client,
    )
    assert len(results) == 3
    assert all(r.doc_type == "session_note" for r in results)


# ---------------------------------------------------------------------------
# search_knowledge (reference collection)
# ---------------------------------------------------------------------------

@pytest.fixture()
def seeded_knowledge(fake_embedder, chroma_client):
    ingest_knowledge(
        "Autoregulation via RPE lets lifters adjust load to daily readiness.",
        doc=KnowledgeDoc(source="rpe.pdf", title="RPE Guide", topic="autoregulation"),
        embedder=fake_embedder,
        client=chroma_client,
    )
    ingest_knowledge(
        "Deadlift weekly volume landmarks typically fall between 7 and 9 hard sets.",
        doc=KnowledgeDoc(source="dl.pdf", title="DL Volume", topic="volume"),
        embedder=fake_embedder,
        client=chroma_client,
    )
    return fake_embedder, chroma_client


def test_search_knowledge_unscoped_is_allowed(seeded_knowledge):
    fake_embedder, chroma_client = seeded_knowledge
    # Unlike search_notes, no metadata filter is required for reference material.
    results = search_knowledge("readiness", embedder=fake_embedder, client=chroma_client)
    assert len(results) == 2


def test_search_knowledge_topic_filter(seeded_knowledge):
    fake_embedder, chroma_client = seeded_knowledge
    results = search_knowledge(
        "anything", topic="volume", embedder=fake_embedder, client=chroma_client
    )
    assert len(results) == 1
    assert results[0].source == "dl.pdf"
    assert results[0].title == "DL Volume"


def test_search_knowledge_empty_collection(fake_embedder, chroma_client):
    assert search_knowledge("x", embedder=fake_embedder, client=chroma_client) == []
