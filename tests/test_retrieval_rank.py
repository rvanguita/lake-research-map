"""Pure ranking pieces of the retrieval package."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from lake_research_map.retrieval.rank import (
    BM25Index,
    excerpt,
    first_per_key,
    rrf_fuse,
    tokenize,
)

TEXTS = [
    "Optimal placement of distributed generation in radial distribution networks",
    "Distribution network expansion planning under load growth uncertainty",
    "Cold chain logistics and facility location for food distribution",
    "MILP model for distribution system planning with IEEE 33-bus feeder",
    "Distribution distribution distribution",
]


def _single_shot_bm25(query: str, texts: list[str], k1: float = 1.5, b: float = 0.75):
    """The scoring `dashboard/search.py::bm25_search` applied, written out."""
    docs = [tokenize(text) for text in texts]
    lengths = np.array([max(len(doc), 1) for doc in docs], dtype=float)
    avgdl = lengths.mean()
    scores = np.zeros(len(docs))
    for term in (token for token in tokenize(query) if len(token) > 1):
        df = sum(1 for doc in docs if term in doc)
        if not df:
            continue
        idf = np.log(1 + (len(docs) - df + 0.5) / (df + 0.5))
        for i, doc in enumerate(docs):
            tf = doc.count(term)
            if tf:
                scores[i] += idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * lengths[i] / avgdl))
    return scores


@pytest.mark.parametrize(
    "query",
    ["distribution planning", "IEEE 33-bus MILP", "logistics", "a distribution distribution"],
)
def test_bm25_index_scores_equal_the_single_shot_definition(query):
    np.testing.assert_allclose(BM25Index(TEXTS).scores(query), _single_shot_bm25(query, TEXTS))


def test_bm25_index_matches_the_dashboard_bm25_search_on_the_same_population():
    from lake_research_map.dashboard.search import bm25_search

    frame = pd.DataFrame({"id": range(len(TEXTS)), "text": TEXTS})
    legacy = bm25_search("distribution planning uncertainty", frame, top_k=10)
    scores = BM25Index(TEXTS).scores("distribution planning uncertainty")

    np.testing.assert_allclose(
        legacy["bm25_score"].to_numpy(), scores[legacy["id"].to_numpy()], rtol=1e-12
    )


def test_bm25_index_is_reusable_and_scores_zero_without_matches():
    index = BM25Index(TEXTS)

    assert index.scores("transmission").sum() == 0
    assert index.scores("logistics").argmax() == 2
    assert index.scores("radial").argmax() == 0
    assert BM25Index([]).scores("anything").shape == (0,)


def test_rrf_fuse_sums_reciprocal_ranks_over_one_based_positions():
    fused = rrf_fuse([["a", "b", "c"], ["b", "d"]], k=60)

    assert fused["b"] == pytest.approx(1 / 62 + 1 / 61)
    assert fused["a"] == pytest.approx(1 / 61)
    assert fused["d"] == pytest.approx(1 / 62)
    assert max(fused, key=fused.get) == "b"


def test_first_per_key_keeps_each_articles_best_chunk_in_rank_order():
    ranked = [("10.1/a", 0.9), ("10.1/b", 0.8), ("10.1/a", 0.7), ("10.1/c", 0.6)]

    assert first_per_key(ranked, limit=10) == [("10.1/a", 0.9), ("10.1/b", 0.8), ("10.1/c", 0.6)]
    assert first_per_key(ranked, limit=2) == [("10.1/a", 0.9), ("10.1/b", 0.8)]


def test_excerpt_centres_on_the_first_query_term():
    text = "Lead-in words. " * 30 + "The hosting capacity of feeders is assessed." + " Tail." * 30

    window = excerpt(text, "hosting capacity", width=120)

    assert "hosting capacity" in window
    assert window.startswith("…") and window.endswith("…")
    assert excerpt("short text", "query") == "short text"
