"""The Qdrant store's own contract, against the in-process local client."""

from __future__ import annotations

import numpy as np
import pytest

from lake_research_map.db.vector_store import (
    ACTIVE_ALIAS,
    QdrantVectorStore,
    VectorPoint,
    VectorStoreUnavailable,
    collection_name,
    get_vector_store,
    point_id,
    version_of,
)


def _unit(values) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float32)
    return vector / np.linalg.norm(vector)


def _point(doi: str, vector, chunk_type: str = "abstract", seq: int = 0, **payload) -> VectorPoint:
    return VectorPoint(
        point_id(doi, chunk_type, seq),
        _unit(vector),
        {"doi": doi, "chunk_type": chunk_type, "seq": seq, **payload},
    )


def test_point_ids_are_deterministic_per_chunk_key():
    assert point_id("10.1/a", "abstract", 0) == point_id("10.1/a", "abstract", 0)
    assert point_id("10.1/a", "abstract", 0) != point_id("10.1/a", "fulltext", 0)
    assert point_id("10.1/a", "fulltext", 1) != point_id("10.1/a", "fulltext", 2)
    assert version_of(collection_name("v1")) == "v1"
    assert version_of(ACTIVE_ALIAS) is None


def test_ensure_collection_is_idempotent_and_rejects_another_dimension(vector_store):
    assert vector_store.ensure_collection("v1", 3) is True
    assert vector_store.ensure_collection("v1", 3) is False
    with pytest.raises(ValueError, match="incompatible vectors"):
        vector_store.ensure_collection("v1", 4)
    assert vector_store.versions() == ["v1"]


def test_upsert_and_fetch_round_trip_vectors_and_payload(vector_store):
    vector_store.ensure_collection("v1", 3)
    vector_store.upsert("v1", [_point("10.1/a", [1, 0, 0], text_sha256="h")])

    stored = vector_store.fetch("v1", [point_id("10.1/a", "abstract", 0), "missing-id"])

    assert list(stored) == [point_id("10.1/a", "abstract", 0)]
    assert stored[point_id("10.1/a", "abstract", 0)].payload["text_sha256"] == "h"
    with_vector = vector_store.fetch(
        "v1", [point_id("10.1/a", "abstract", 0)], with_vectors=True, payload_fields=False
    )
    np.testing.assert_allclose(next(iter(with_vector.values())).vector, [1.0, 0.0, 0.0], atol=1e-6)


def test_find_by_text_hash_requires_the_same_model_revision(vector_store):
    vector_store.ensure_collection("v1", 3)
    vector_store.upsert(
        "v1",
        [
            _point("10.1/a", [1, 0, 0], text_sha256="h1", embed_model="m", embed_revision="r1"),
            _point("10.1/b", [0, 1, 0], text_sha256="h2", embed_model="m", embed_revision="r0"),
        ],
    )

    found = vector_store.find_by_text_hash("v1", ["h1", "h2", "h3"], model="m", revision="r1")

    assert set(found) == {"h1"}
    assert vector_store.find_by_text_hash("absent", ["h1"], model="m", revision="r1") == {}


def test_exact_dense_query_matches_the_numpy_cosine_order(vector_store):
    rng = np.random.default_rng(0)
    matrix = rng.normal(size=(40, 8)).astype(np.float32)
    vector_store.ensure_collection("v1", 8)
    vector_store.upsert("v1", [_point(f"10.1/{i}", row) for i, row in enumerate(matrix)])
    query = rng.normal(size=8).astype(np.float32)

    hits = vector_store.query_dense("v1", query, limit=10)

    unit_rows = matrix / np.linalg.norm(matrix, axis=1, keepdims=True)
    expected = np.argsort(-(unit_rows @ (query / np.linalg.norm(query))))[:10]
    assert [hit.payload["doi"] for hit in hits] == [f"10.1/{i}" for i in expected]


def test_alias_moves_atomically_between_versions(vector_store):
    for version in ("v1", "v2"):
        vector_store.ensure_collection(version, 3)
    assert vector_store.alias_target() is None

    vector_store.point_alias("v1")
    assert vector_store.alias_target() == "v1"
    vector_store.point_alias("v2")
    assert vector_store.alias_target() == "v2"


def test_set_payloads_merges_one_payload_into_each_group(vector_store):
    vector_store.ensure_collection("v1", 3)
    vector_store.upsert(
        "v1",
        [
            _point("10.1/a", [1, 0, 0]),
            _point("10.1/a", [0, 1, 0], chunk_type="fulltext", seq=1),
            _point("10.1/b", [0, 0, 1]),
        ],
    )
    group_a = [point_id("10.1/a", "abstract", 0), point_id("10.1/a", "fulltext", 1)]

    vector_store.set_payloads("v1", [(group_a, {"theme_id": 3})])

    stored = vector_store.fetch("v1", [*group_a, point_id("10.1/b", "abstract", 0)])
    assert [stored[key].payload.get("theme_id") for key in group_a] == [3, 3]
    assert "theme_id" not in stored[point_id("10.1/b", "abstract", 0)].payload
    assert stored[group_a[0]].payload["doi"] == "10.1/a"


def test_abstract_vectors_are_ordered_by_doi_and_skip_fulltext(vector_store):
    vector_store.ensure_collection("v1", 3)
    vector_store.upsert(
        "v1",
        [
            _point("10.1/b", [0, 1, 0]),
            _point("10.1/a", [1, 0, 0]),
            _point("10.1/a", [0, 0, 1], chunk_type="fulltext", seq=1),
        ],
    )

    dois, matrix = vector_store.abstract_vectors("v1")

    assert dois == ["10.1/a", "10.1/b"]
    np.testing.assert_allclose(matrix, [[1, 0, 0], [0, 1, 0]], atol=1e-6)


def test_delete_points_removes_only_the_named_points(vector_store):
    vector_store.ensure_collection("v1", 3)
    vector_store.upsert("v1", [_point("10.1/a", [1, 0, 0]), _point("10.1/b", [0, 1, 0])])

    vector_store.delete_points("v1", [point_id("10.1/a", "abstract", 0)])

    assert vector_store.count("v1") == 1


def test_an_unreachable_server_raises_vector_store_unavailable():
    from qdrant_client import QdrantClient

    # Port 9 (discard) is closed on any normal machine: the request is refused
    # locally, so this never leaves the host.
    store = QdrantVectorStore(
        QdrantClient(url="http://127.0.0.1:9", timeout=2, check_compatibility=False)
    )

    with pytest.raises(VectorStoreUnavailable, match="docker compose --profile qdrant"):
        store.has_collection("v1")


def test_get_vector_store_requires_qdrant_url(monkeypatch):
    monkeypatch.setenv("QDRANT_URL", "")

    with pytest.raises(VectorStoreUnavailable, match="QDRANT_URL is not set"):
        get_vector_store()
