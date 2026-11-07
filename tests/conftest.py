"""Shared fixtures: one in-memory SQLite session per medallion layer, plus Qdrant.

Each layer is already an independent SQLAlchemy `Base`/database in the real
system (see `src/db/*_models.py`), and nothing in the
transform code is MySQL-specific, so a separate in-memory SQLite engine per
layer is a faithful, dependency-free stand-in for the real MySQL databases.
"""

from __future__ import annotations

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from db import bronze_models, gold_models, raw_models, silver_models


def _sqlite_session(base):
    engine = create_engine("sqlite:///:memory:", future=True)
    base.metadata.create_all(engine)
    return sessionmaker(bind=engine, future=True)()


@pytest.fixture
def raw_session():
    session = _sqlite_session(raw_models.Base)
    yield session
    session.close()


@pytest.fixture
def bronze_session():
    session = _sqlite_session(bronze_models.Base)
    yield session
    session.close()


@pytest.fixture
def silver_session():
    session = _sqlite_session(silver_models.Base)
    yield session
    session.close()


@pytest.fixture
def gold_session():
    session = _sqlite_session(gold_models.Base)
    yield session
    session.close()


@pytest.fixture
def vector_store():
    """In-process Qdrant (local mode): real filter, alias and payload semantics.

    The role in-memory SQLite plays for MySQL. Server-only behaviour (payload
    indexes, the exact-search parameter) is covered by the opt-in `qdrant`
    marker in `tests/test_qdrant_server.py` instead.
    """
    from qdrant_client import QdrantClient

    from db.vector_store import QdrantVectorStore

    client = QdrantClient(":memory:")
    yield QdrantVectorStore(client)
    client.close()
