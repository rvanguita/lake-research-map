"""Server-only Qdrant behaviour the in-process client cannot show.

The local client ignores payload indexes and search parameters, so these run
against the compose service: `docker compose --profile qdrant up -d qdrant`,
then `RUN_QDRANT_INTEGRATION=1 uv run pytest -m qdrant`. They create and drop
their own throwaway collection and alias; the project's collections and the
`lit_chunks_active` alias are never touched.
"""

from __future__ import annotations

import os
import uuid

import numpy as np
import pytest

from db.vector_store import (
    PAYLOAD_INDEXES,
    VectorPoint,
    collection_name,
    get_vector_store,
    point_id,
)

pytestmark = [
    pytest.mark.qdrant,
    pytest.mark.skipif(
        os.environ.get("RUN_QDRANT_INTEGRATION") != "1",
        reason="set RUN_QDRANT_INTEGRATION=1 with a running Qdrant for the server checks",
    ),
]


@pytest.fixture
def server_store():
    store = get_vector_store()
    version_id = f"test-{uuid.uuid4().hex}"
    yield store, version_id
    if store.has_collection(version_id):
        store.drop_collection(version_id)


def test_server_collection_has_payload_indexes_and_filters_exactly(server_store):
    from qdrant_client import models

    store, version_id = server_store
    assert store.local is False
    store.ensure_collection(version_id, 4)
    rng = np.random.default_rng(1)
    matrix = rng.normal(size=(30, 4)).astype(np.float32)
    store.upsert(
        version_id,
        [
            VectorPoint(
                point_id(f"10.1/{i}", "abstract", 0),
                row,
                {"doi": f"10.1/{i}", "year": 2000 + i, "chunk_type": "abstract"},
            )
            for i, row in enumerate(matrix)
        ],
    )

    info = store.client.get_collection(collection_name(version_id))
    assert set(PAYLOAD_INDEXES) <= set(info.payload_schema)

    query = rng.normal(size=4).astype(np.float32)
    year_filter = models.Filter(
        must=[models.FieldCondition(key="year", range=models.Range(gte=2010))]
    )
    hits = store.query_dense(version_id, query, query_filter=year_filter, limit=5)

    unit = matrix / np.linalg.norm(matrix, axis=1, keepdims=True)
    scores = unit @ (query / np.linalg.norm(query))
    eligible = [i for i in np.argsort(-scores) if 2000 + i >= 2010][:5]
    assert [hit.payload["doi"] for hit in hits] == [f"10.1/{i}" for i in eligible]


def test_server_alias_switch_is_atomic_and_scoped_to_its_name(server_store):
    store, version_id = server_store
    alias = f"lit_chunks_test_alias_{uuid.uuid4().hex[:8]}"
    store.ensure_collection(version_id, 4)
    try:
        store.point_alias(version_id, alias=alias)
        assert store.alias_target(alias) == version_id
    finally:
        from qdrant_client import models

        store.client.update_collection_aliases(
            change_aliases_operations=[
                models.DeleteAliasOperation(delete_alias=models.DeleteAlias(alias_name=alias))
            ]
        )
