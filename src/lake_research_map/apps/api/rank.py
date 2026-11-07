"""Pure ranking pieces: an Okapi BM25 index built once, fusion, collapsing.

No model, no database, no Streamlit: everything here is a function of its
arguments, which is what lets the ranking logic be tested with hand-built
inputs. Dense scores come from Qdrant (`db/vector_store.py`); lexical scores,
their fusion and the per-article collapse happen here.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from collections.abc import Hashable, Iterable, Sequence

import numpy as np

_TOKEN = re.compile(r"\b\w+\b")


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


class BM25Index:
    """Okapi BM25 over a fixed list of texts, tokenized once.

    The scoring is the one the chunk-level dashboard search used before WP-28
    (same tokenizer, k1=1.5, b=0.75, IDF `log(1 + (N - df + 0.5) / (df + 0.5))`,
    query tokens shorter than two characters ignored, a repeated query token
    counted again), but the corpus statistics are computed once over the whole
    scope instead of on every query over whatever subset was passed in. A filter
    therefore changes which documents may rank, never how a document scores.
    """

    def __init__(self, texts: Sequence[str], *, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        counters = [Counter(tokenize(text or "")) for text in texts]
        self.n_docs = len(counters)
        self.doc_lens = np.array([max(sum(c.values()), 1) for c in counters], dtype=float)
        self.avgdl = float(self.doc_lens.mean()) if self.n_docs else 1.0
        postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for doc, counter in enumerate(counters):
            for term, frequency in counter.items():
                postings[term].append((doc, frequency))
        self._postings = {
            term: (
                np.fromiter((doc for doc, _ in pairs), dtype=np.int64, count=len(pairs)),
                np.fromiter((tf for _, tf in pairs), dtype=float, count=len(pairs)),
            )
            for term, pairs in postings.items()
        }

    def scores(self, query: str) -> np.ndarray:
        """One score per indexed text; zero where no query term occurs."""
        scores = np.zeros(self.n_docs, dtype=float)
        for term in (token for token in tokenize(query) if len(token) > 1):
            posting = self._postings.get(term)
            if posting is None:
                continue
            docs, tf = posting
            df = len(docs)
            idf = float(np.log(1.0 + (self.n_docs - df + 0.5) / (df + 0.5)))
            denominator = tf + self.k1 * (1.0 - self.b + self.b * self.doc_lens[docs] / self.avgdl)
            scores[docs] += idf * tf * (self.k1 + 1.0) / np.maximum(denominator, 1e-9)
        return scores


def rrf_fuse(rankings: Iterable[Sequence[Hashable]], *, k: int = 60) -> dict[Hashable, float]:
    """Reciprocal Rank Fusion (Cormack et al., SIGIR 2009) over 1-based ranks."""
    fused: dict[Hashable, float] = {}
    for ranking in rankings:
        for rank, key in enumerate(ranking, start=1):
            fused[key] = fused.get(key, 0.0) + 1.0 / (k + rank)
    return fused


def first_per_key(ordered: Iterable[tuple[Hashable, ...]], *, limit: int) -> list[tuple]:
    """Keep the first (best-ranked) item per key, in order, up to `limit` keys.

    Items are tuples whose first element is the key (a DOI). This is how a
    chunk ranking becomes an article ranking: an article ranks where its best
    chunk ranks, and its other chunks do not take further places.
    """
    seen: set[Hashable] = set()
    kept: list[tuple] = []
    for item in ordered:
        if item[0] in seen:
            continue
        seen.add(item[0])
        kept.append(item)
        if len(kept) >= limit:
            break
    return kept


def excerpt(text: str, query: str, *, width: int = 240) -> str:
    """A window of `text` around the first query term it contains."""
    text = " ".join((text or "").split())
    if len(text) <= width:
        return text
    lowered = text.lower()
    positions = [
        lowered.find(term) for term in tokenize(query) if len(term) > 2 and term in lowered
    ]
    center = min(positions) if positions else 0
    start = max(0, min(center - width // 3, len(text) - width))
    end = start + width
    return ("…" if start else "") + text[start:end].strip() + ("…" if end < len(text) else "")
