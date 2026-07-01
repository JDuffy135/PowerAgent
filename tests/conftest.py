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
