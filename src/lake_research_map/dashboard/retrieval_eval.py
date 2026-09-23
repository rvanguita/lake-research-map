"""Pure ranking metrics and latency measurement for retrieval evaluation."""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Iterable, Mapping, Sequence


def _unique_prefix(ranked_ids: Sequence[str], k: int) -> list[str]:
    return list(dict.fromkeys(item for item in ranked_ids if item))[: max(0, int(k))]


def recall_at_k(ranked_ids: Sequence[str], relevant_ids: Iterable[str], k: int) -> float:
    """Fraction of known relevant documents retrieved within the first ``k`` ranks."""
    relevant = {item for item in relevant_ids if item}
    if not relevant:
        return 0.0
    retrieved = set(_unique_prefix(ranked_ids, k))
    return len(retrieved & relevant) / len(relevant)


def reciprocal_rank(ranked_ids: Sequence[str], relevant_ids: Iterable[str]) -> float:
    """Reciprocal rank of the first relevant document, or zero when none is found."""
    relevant = {item for item in relevant_ids if item}
    for rank, item in enumerate(_unique_prefix(ranked_ids, len(ranked_ids)), start=1):
        if item in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(ranked_ids: Sequence[str], relevant_ids: Iterable[str], k: int) -> float:
    """Binary normalized discounted cumulative gain at ``k``."""
    relevant = {item for item in relevant_ids if item}
    if not relevant or k <= 0:
        return 0.0
    ranked = _unique_prefix(ranked_ids, k)
    dcg = sum(
        1.0 / math.log2(rank + 1) for rank, item in enumerate(ranked, start=1) if item in relevant
    )
    ideal_hits = min(len(relevant), k)
    ideal = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / ideal


def benchmark_retrieval_modes(
    modes: Mapping[str, Callable[[str, int], Sequence[str]]],
    queries: Sequence[tuple[str, str]],
    *,
    top_k: int,
    clock: Callable[[], float] = time.perf_counter,
) -> tuple[dict[str, dict[str, list[str]]], dict[str, dict[str, float]]]:
    """Run identical queries through each mode and measure end-to-end latency.

    Returns rankings by ``mode -> query_id`` plus per-mode mean, median-like p50,
    maximum, and total milliseconds. Model warm-up is deliberately outside this
    helper so callers can choose whether cold-start latency belongs in a report.
    """
    rankings: dict[str, dict[str, list[str]]] = {}
    latency: dict[str, dict[str, float]] = {}
    for mode, retrieve in modes.items():
        mode_rankings: dict[str, list[str]] = {}
        elapsed_ms: list[float] = []
        for query_id, text in queries:
            started = clock()
            mode_rankings[query_id] = list(retrieve(text, top_k))
            elapsed_ms.append((clock() - started) * 1000.0)
        ordered = sorted(elapsed_ms)
        midpoint = len(ordered) // 2
        p50 = (
            0.0
            if not ordered
            else ordered[midpoint]
            if len(ordered) % 2
            else (ordered[midpoint - 1] + ordered[midpoint]) / 2.0
        )
        total = sum(elapsed_ms)
        rankings[mode] = mode_rankings
        latency[mode] = {
            "mean_ms": total / len(elapsed_ms) if elapsed_ms else 0.0,
            "p50_ms": p50,
            "max_ms": max(elapsed_ms, default=0.0),
            "total_ms": total,
        }
    return rankings, latency


def evaluate_rankings(
    rankings: Mapping[str, Mapping[str, Sequence[str]]],
    relevant_by_query: Mapping[str, Iterable[str]],
    *,
    k: int,
) -> dict[str, dict[str, float]]:
    """Macro-average Recall@k, MRR, and nDCG@k over judged queries."""
    query_ids = sorted(relevant_by_query)
    metrics: dict[str, dict[str, float]] = {}
    for mode, by_query in rankings.items():
        recalls = []
        reciprocal_ranks = []
        ndcgs = []
        for query_id in query_ids:
            ranked = by_query.get(query_id, ())
            relevant = relevant_by_query[query_id]
            recalls.append(recall_at_k(ranked, relevant, k))
            reciprocal_ranks.append(reciprocal_rank(ranked, relevant))
            ndcgs.append(ndcg_at_k(ranked, relevant, k))
        denominator = len(query_ids)
        metrics[mode] = {
            f"recall@{k}": sum(recalls) / denominator if denominator else 0.0,
            "mrr": sum(reciprocal_ranks) / denominator if denominator else 0.0,
            f"ndcg@{k}": sum(ndcgs) / denominator if denominator else 0.0,
            "queries": float(denominator),
        }
    return metrics
