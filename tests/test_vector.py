import pytest

from src.ingest.embed import SessionNote, embed_session_notes
from src.tools.vector import search_notes


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
