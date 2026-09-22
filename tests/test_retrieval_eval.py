from __future__ import annotations

import pytest

from lake_research_map.dashboard.retrieval_eval import (
    benchmark_retrieval_modes,
    evaluate_rankings,
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank,
)


def test_hand_computed_binary_ranking_metrics():
    ranked = ["d0", "d2", "d1", "d3"]
    relevant = {"d1", "d2", "d4"}

    assert recall_at_k(ranked, relevant, 3) == pytest.approx(2 / 3)
    assert reciprocal_rank(ranked, relevant) == pytest.approx(1 / 2)
    expected_dcg = 1 / 1.584962500721156 + 1 / 2
    ideal_dcg = 1 + 1 / 1.584962500721156 + 1 / 2
    assert ndcg_at_k(ranked, relevant, 3) == pytest.approx(expected_dcg / ideal_dcg)


def test_metrics_handle_empty_judgements_and_duplicate_results():
    assert recall_at_k(["a"], [], 10) == 0
    assert reciprocal_rank(["a"], []) == 0
    assert ndcg_at_k(["a"], [], 10) == 0
    assert recall_at_k(["a", "a", "b"], {"b"}, 2) == 1


def test_rankings_are_macro_averaged_on_one_judgement_set():
    rankings = {
        "dense": {"q1": ["a", "b"], "q2": ["x", "c"]},
        "bm25": {"q1": ["b", "a"], "q2": ["c", "x"]},
    }
    relevant = {"q1": {"a"}, "q2": {"c"}}
    metrics = evaluate_rankings(rankings, relevant, k=1)

    assert metrics["dense"] == {
        "recall@1": 0.5,
        "mrr": 0.75,
        "ndcg@1": 0.5,
        "queries": 2.0,
    }
    assert metrics["bm25"]["recall@1"] == 0.5
    assert metrics["bm25"]["mrr"] == 0.75


def test_benchmark_runs_identical_queries_and_reports_latency():
    ticks = iter([0.0, 0.010, 0.010, 0.030, 0.030, 0.035, 0.035, 0.045])
    queries = [("q1", "one"), ("q2", "two")]
    modes = {
        "dense": lambda text, _k: [f"dense-{text}"],
        "bm25": lambda text, _k: [f"bm25-{text}"],
    }

    rankings, latency = benchmark_retrieval_modes(
        modes, queries, top_k=5, clock=lambda: next(ticks)
    )

    assert rankings["dense"] == {"q1": ["dense-one"], "q2": ["dense-two"]}
    assert latency["dense"]["mean_ms"] == pytest.approx(15.0)
    assert latency["bm25"]["max_ms"] == pytest.approx(10.0)
