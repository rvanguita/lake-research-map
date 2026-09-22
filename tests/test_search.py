import numpy as np
import pandas as pd

from lake_research_map.dashboard.search import (
    _rank_by_similarity,
    build_vector_index,
    search_vector_index,
)


def _vector(*values: float) -> bytes:
    return np.array(values, dtype=np.float32).tobytes()


def _chunks_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"doi": "10.1/a", "text": "close match", "embedding_bin": _vector(1.0, 0.0, 0.0)},
            {"doi": "10.1/b", "text": "far match", "embedding_bin": _vector(0.0, 1.0, 0.0)},
            {"doi": "10.1/c", "text": "no embedding yet", "embedding_bin": None},
        ]
    )


def test_rank_by_similarity_ranks_closest_vector_first():
    query = np.array([1.0, 0.0, 0.0])
    result = _rank_by_similarity(query, _chunks_df(), top_k=10)

    assert list(result["doi"]) == ["10.1/a", "10.1/b"]
    assert result.iloc[0]["score"] > result.iloc[1]["score"]


def test_rank_by_similarity_excludes_rows_without_embedding():
    query = np.array([1.0, 0.0, 0.0])
    result = _rank_by_similarity(query, _chunks_df(), top_k=10)

    assert "10.1/c" not in set(result["doi"])


def test_rank_by_similarity_respects_top_k():
    query = np.array([1.0, 0.0, 0.0])
    result = _rank_by_similarity(query, _chunks_df(), top_k=1)

    assert len(result) == 1
    assert result.iloc[0]["doi"] == "10.1/a"


def test_rank_by_similarity_no_embedded_rows_returns_empty():
    df = pd.DataFrame([{"doi": "10.1/c", "text": "no embedding yet", "embedding_bin": None}])
    result = _rank_by_similarity(np.array([1.0, 0.0, 0.0]), df, top_k=10)
    assert result.empty


def test_rank_by_similarity_no_embedding_column_returns_empty():
    df = pd.DataFrame([{"doi": "10.1/c", "text": "no embedding column"}])
    result = _rank_by_similarity(np.array([1.0, 0.0, 0.0]), df, top_k=10)
    assert result.empty


def test_rank_by_similarity_ignores_the_retired_json_mirror():
    """Binary is the only canonical vector representation.

    The JSON column is no longer written, so a frame carrying only JSON is a
    frame with no usable vectors -- silently parsing it would hide a chunk that
    never passed `embed_contract`.
    """
    df = pd.DataFrame([{"doi": "10.1/a", "text": "json only", "embedding": [1.0, 0.0, 0.0]}])
    result = _rank_by_similarity(np.array([1.0, 0.0, 0.0]), df, top_k=10)
    assert result.empty


def test_build_and_search_vector_index():
    matrix = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.9, 0.1, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    idx = build_vector_index(matrix)
    assert idx is not None

    query = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    indices, sims = search_vector_index(idx, query, top_k=2)
    assert len(indices) == 2
    assert indices[0] == 0
    assert indices[1] == 1
    assert sims[0] >= sims[1]
    assert np.isclose(sims[0], 1.0, atol=1e-4)


def test_build_vector_index_empty():
    empty = np.zeros((0, 3), dtype=np.float32)
    idx = build_vector_index(empty)
    assert idx is None
    indices, sims = search_vector_index(idx, np.array([1.0, 0.0, 0.0]))
    assert len(indices) == 0


def test_bm25_search_ranks_matching_keywords():
    from lake_research_map.dashboard.search import bm25_search

    df = pd.DataFrame(
        [
            {"id": 1, "text": "Optimal distribution planning with solar PV systems"},
            {"id": 2, "text": "Electric vehicle charging stations and battery storage"},
            {"id": 3, "text": "Unrelated supply chain logistics optimization"},
        ]
    )
    res = bm25_search("distribution planning", df, top_k=5)
    assert not res.empty
    assert res.iloc[0]["id"] == 1
    assert res.iloc[0]["bm25_score"] > 0


def test_hybrid_search_rrf_merges_dense_and_sparse():
    from lake_research_map.dashboard.search import hybrid_search_rrf

    df = pd.DataFrame(
        [
            {
                "id": 1,
                "text": "Solar power distribution planning",
                "embedding_bin": np.array([1.0, 0.0, 0.0], dtype=np.float32).tobytes(),
            },
            {
                "id": 2,
                "text": "Battery storage coordination",
                "embedding_bin": np.array([0.0, 1.0, 0.0], dtype=np.float32).tobytes(),
            },
        ]
    )
    res = hybrid_search_rrf(
        "solar distribution", df, top_k=2, query_vector=np.array([1.0, 0.0, 0.0])
    )
    assert not res.empty
    assert "score" in res.columns
    assert "dense_score" in res.columns
    assert "bm25_score" in res.columns
    assert "dense_rank" in res.columns
    assert "bm25_rank" in res.columns
    assert res.iloc[0]["dense_rank"] == 1
