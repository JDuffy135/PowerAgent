import pytest

from src.db.connection import get_conn, init_db
from src.seed import seed


@pytest.fixture()
def conn():
    connection = get_conn(":memory:")
    init_db(connection)
    seed(connection)
    yield connection
    connection.close()


@pytest.fixture()
def fake_embedder():
    """Deterministic 8-dim embedder: no Ollama, no model. Same text -> same vector."""
    def _embed(texts):
        return [[float((len(t) + i * 7) % 97) for i in range(8)] for t in texts]

    return _embed


@pytest.fixture()
def chroma_client():
    """In-memory Chroma client so embed tests need no on-disk store or model.

    `EphemeralClient` instances share one process-global store, so we clear any
    existing collections up front to keep each test isolated.
    """
    import chromadb

    client = chromadb.EphemeralClient()
    for collection in client.list_collections():
        client.delete_collection(collection.name)
    return client
