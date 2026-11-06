"""`Retriever`: rank the published articles for a query, screen them, cite them.

One article per hit. By default only the abstract chunks rank -- one per
article, so every article competes on equal terms. Full text is 97 of 3,115
articles supplying 4,437 chunks before cleaning; letting those chunks rank
would favour articles for having a PDF, not for being relevant. `scope="all"`
does rank every chunk (an article then ranks where its best chunk ranks), and
says so in its output; `evidence()` is the intended use of full text.

Dense candidates come from the version's Qdrant collection with the filters
applied server-side; lexical candidates come from BM25 over the same filtered
Gold rows (`retrieval/filters.py` keeps the two populations identical); hybrid
fuses them by Reciprocal Rank Fusion (k=60), and an optional cross-encoder
reorders the fused candidates. Ties break by DOI, so a query returns the same
list on every run.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd

from lake_research_map.db.vector_store import (
    ACTIVE_ALIAS,
    QdrantVectorStore,
    VectorStoreUnavailable,
    collection_name,
)
from lake_research_map.retrieval.filters import (
    ArticleFilters,
    article_filter_mask,
    to_qdrant_filter,
)
from lake_research_map.retrieval.index import Corpus
from lake_research_map.retrieval.rank import excerpt, first_per_key, rrf_fuse

logger = logging.getLogger(__name__)

MODES = ("hybrid", "dense", "bm25")
SCOPES = ("abstract", "all")
RRF_K = 60

EmbedFn = Callable[[str], Sequence[float]]
RerankFn = Callable[[str, Sequence[str]], Sequence[float]]
TokenLengthFn = Callable[[Sequence[str]], Sequence[int]]

# A chunk ranking entry: (doi, chunk_type, seq, score).
_Ranked = tuple[str, str, int, float]


class RetrievalUnavailable(RuntimeError):
    """Dense retrieval needs the vector store and the embedding model."""


def _clean(value: Any) -> Any:
    """JSON-safe scalar: numpy to Python, NaN to None."""
    if value is None:
        return None
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def _year(value: Any) -> int | None:
    value = _clean(value)
    return None if value is None else int(value)


@dataclass(frozen=True)
class ArticleHit:
    rank: int
    doi: str
    title: str | None
    year: int | None
    venue: str | None
    source: str
    url: str | None
    score: float
    mode: str
    scope: str
    chunk_type: str
    chunk_seq: int
    dense_score: float | None
    dense_rank: int | None
    bm25_score: float | None
    bm25_rank: int | None
    rerank_score: float | None
    rerank_rank: int | None
    relevance_margin: float | None
    theme_label: str | None
    has_pdf: bool
    abstract_missing: bool
    abstract_truncated: bool | None
    excerpt: str

    def to_dict(self) -> dict[str, Any]:
        return {key: _clean(value) for key, value in asdict(self).items()}


@dataclass(frozen=True)
class Passage:
    doi: str
    seq: int
    score: float
    text: str

    def to_dict(self) -> dict[str, Any]:
        return {key: _clean(value) for key, value in asdict(self).items()}


class Retriever:
    """Every retrieval operation over one published version.

    Models are resolved on first use, so constructing a retriever, or running a
    lexical search, never loads one. Tests inject `embed_fn`, `rerank_fn` and
    `token_len_fn` instead of downloading models.
    """

    def __init__(
        self,
        corpus: Corpus,
        store: QdrantVectorStore | None,
        *,
        embed_fn: EmbedFn | None = None,
        rerank_fn: RerankFn | None = None,
        token_len_fn: TokenLengthFn | None = None,
        context_window: int | None = None,
    ) -> None:
        self.corpus = corpus
        self.store = store
        self._embed_fn = embed_fn
        self._rerank_fn = rerank_fn
        self._token_len_fn = token_len_fn
        self._context_window = context_window
        self._abstract_token_lengths: dict[str, int] | None = None

    # -- models ------------------------------------------------------------

    def _embed(self, text: str) -> np.ndarray:
        if self._embed_fn is None:
            from lake_research_map.retrieval.models import embed_query

            self._embed_fn = embed_query
        try:
            return np.asarray(self._embed_fn(text), dtype=np.float32)
        except Exception as exc:
            raise RetrievalUnavailable(f"the embedding model is unavailable ({exc})") from exc

    def _rerank(self, query: str, texts: Sequence[str]) -> list[float]:
        if self._rerank_fn is None:
            from lake_research_map.retrieval.models import rerank_scores

            self._rerank_fn = rerank_scores
        try:
            return [float(score) for score in self._rerank_fn(query, texts)]
        except Exception as exc:
            raise RetrievalUnavailable(f"the reranker is unavailable ({exc})") from exc

    def context_window(self) -> int:
        if self._context_window is None:
            from lake_research_map.retrieval.models import context_window_tokens

            self._context_window = context_window_tokens()
        return self._context_window

    def _token_lengths(self, texts: Sequence[str]) -> list[int] | None:
        if self._token_len_fn is None:
            from lake_research_map.retrieval.models import token_lengths

            self._token_len_fn = token_lengths
        try:
            return [int(length) for length in self._token_len_fn(texts)]
        except Exception:
            return None

    def abstract_token_lengths(self) -> dict[str, int] | None:
        """Token count of every abstract chunk, measured once (None if unmeasurable)."""
        if self._abstract_token_lengths is None:
            abstracts = self.corpus.abstracts
            lengths = self._token_lengths(abstracts["text"].tolist())
            if lengths is None:
                return None
            self._abstract_token_lengths = dict(zip(abstracts["doi"], lengths, strict=True))
        return self._abstract_token_lengths

    # -- candidates ----------------------------------------------------------

    def _require_store(self) -> QdrantVectorStore:
        if self.store is None:
            raise RetrievalUnavailable("the vector store is not configured or did not answer")
        return self.store

    def _dense(
        self, query: str, filters: ArticleFilters, scope: str, pool: int, allowed: set[str]
    ) -> list[_Ranked]:
        store = self._require_store()
        vector = self._embed(query)
        # In the all-chunks scope an article can hold many of the nearest
        # chunks; ask for enough to still reach `pool` distinct articles.
        limit = pool if scope == "abstract" else min(pool * 8, 2000)
        try:
            hits = store.query_dense(
                self.corpus.version_id,
                vector,
                query_filter=to_qdrant_filter(
                    filters, chunk_type="abstract" if scope == "abstract" else None
                ),
                limit=limit,
                payload_fields=["doi", "chunk_type", "seq"],
            )
        except VectorStoreUnavailable as exc:
            raise RetrievalUnavailable(str(exc)) from exc
        # The payload filter and the Gold-side mask select the same articles
        # (tests/test_retrieval_filters.py); intersecting again keeps a drifted
        # payload from ever fusing two different populations.
        ranked = [
            (hit.payload["doi"], hit.payload["chunk_type"], int(hit.payload["seq"]), hit.score)
            for hit in hits
            if hit.payload["doi"] in allowed
        ]
        ranked.sort(key=lambda item: (-item[3], item[0], item[1], item[2]))
        return first_per_key(ranked, limit=pool)

    def _lexical(self, query: str, allowed: set[str], scope: str, pool: int) -> list[_Ranked]:
        rows = self.corpus.abstracts if scope == "abstract" else self.corpus.chunks
        index = self.corpus.abstract_index if scope == "abstract" else self.corpus.chunk_index
        scores = index.scores(query)
        dois = rows["doi"].to_numpy()
        candidates = [int(i) for i in np.flatnonzero(scores > 0) if dois[i] in allowed]
        candidates.sort(key=lambda i: (-scores[i], dois[i]))
        chunk_types = rows["chunk_type"].to_numpy()
        seqs = rows["seq"].to_numpy()
        return first_per_key(
            ((dois[i], chunk_types[i], int(seqs[i]), float(scores[i])) for i in candidates),
            limit=pool,
        )

    # -- operations ---------------------------------------------------------

    def search(
        self,
        query: str,
        *,
        top_k: int = 10,
        mode: str = "hybrid",
        scope: str = "abstract",
        filters: ArticleFilters | None = None,
        rerank: bool = False,
        candidates: int = 50,
    ) -> list[ArticleHit]:
        """The `top_k` most relevant eligible articles, one hit per article."""
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, not {mode!r}")
        if scope not in SCOPES:
            raise ValueError(f"scope must be one of {SCOPES}, not {scope!r}")
        query = (query or "").strip()
        if not query:
            return []
        filters = filters or ArticleFilters()
        top_k = max(1, int(top_k))
        pool = max(int(candidates), top_k)
        articles = self.corpus.articles
        allowed = set(articles.index[article_filter_mask(articles, filters)])
        if not allowed:
            return []

        dense = (
            self._dense(query, filters, scope, pool, allowed) if mode in ("dense", "hybrid") else []
        )
        lexical = self._lexical(query, allowed, scope, pool) if mode in ("bm25", "hybrid") else []
        dense_by = {item[0]: (rank, item) for rank, item in enumerate(dense, start=1)}
        lexical_by = {item[0]: (rank, item) for rank, item in enumerate(lexical, start=1)}

        if mode == "hybrid":
            fused = rrf_fuse([[i[0] for i in dense], [i[0] for i in lexical]], k=RRF_K)
            order = sorted(fused, key=lambda doi: (-fused[doi], doi))
            score_of = fused
        else:
            ranked = dense if mode == "dense" else lexical
            order = [item[0] for item in ranked]
            score_of = {item[0]: item[3] for item in ranked}

        rerank_by: dict[str, tuple[int, float]] = {}
        if rerank and order:
            head = order[:pool]
            texts = [self._matched_text(doi, dense_by, lexical_by) for doi in head]
            scores = self._rerank(query, texts)
            reordered = sorted(zip(head, scores, strict=True), key=lambda pair: (-pair[1], pair[0]))
            rerank_by = {doi: (rank, score) for rank, (doi, score) in enumerate(reordered, 1)}
            order = [doi for doi, _ in reordered]

        # Truncation is measured with the embedder's own tokenizer, which means
        # loading that model: worth it when the query already did (dense,
        # hybrid) or a measurement exists, not for a lexical-only search.
        measure = (
            mode != "bm25"
            or self._token_len_fn is not None
            or self._abstract_token_lengths is not None
        )
        lengths = self._hit_token_lengths(order[:top_k]) if measure else None
        window = self.context_window() if lengths else None
        hits = []
        for rank, doi in enumerate(order[:top_k], start=1):
            chunk = (dense_by.get(doi) or lexical_by.get(doi))[1]
            article = articles.loc[doi]
            dense_rank, dense_item = dense_by.get(doi, (None, None))
            lexical_rank, lexical_item = lexical_by.get(doi, (None, None))
            rerank_rank, rerank_score = rerank_by.get(doi, (None, None))
            truncated = None
            if lengths is not None and doi in lengths and window is not None:
                truncated = lengths[doi] > window
            hits.append(
                ArticleHit(
                    rank=rank,
                    doi=doi,
                    title=article["title"],
                    year=_year(article["year"]),
                    venue=article["venue"],
                    source=article["source"],
                    url=article["url"],
                    score=float(rerank_score if rerank_score is not None else score_of[doi]),
                    mode=mode + ("+rerank" if rerank_by else ""),
                    scope=scope,
                    chunk_type=chunk[1],
                    chunk_seq=int(chunk[2]),
                    dense_score=None if dense_item is None else float(dense_item[3]),
                    dense_rank=dense_rank,
                    bm25_score=None if lexical_item is None else float(lexical_item[3]),
                    bm25_rank=lexical_rank,
                    rerank_score=rerank_score,
                    rerank_rank=rerank_rank,
                    relevance_margin=_clean(article["relevance_margin"]),
                    theme_label=_clean(article["theme_label"]),
                    has_pdf=bool(article["has_pdf"]),
                    abstract_missing=bool(article["abstract_missing"]),
                    abstract_truncated=truncated,
                    excerpt=excerpt(
                        self.corpus.text_by_key.get((doi, chunk[1], int(chunk[2])), ""), query
                    ),
                )
            )
        return hits

    def _matched_text(self, doi: str, dense_by: dict, lexical_by: dict) -> str:
        chunk = (dense_by.get(doi) or lexical_by.get(doi))[1]
        return self.corpus.text_by_key.get((doi, chunk[1], int(chunk[2])), "")

    def _hit_token_lengths(self, dois: Sequence[str]) -> dict[str, int] | None:
        if self._abstract_token_lengths is not None:
            return self._abstract_token_lengths
        texts = [self.corpus.text_by_key.get((doi, "abstract", 0), "") for doi in dois]
        lengths = self._token_lengths(texts)
        if lengths is None:
            return None
        return dict(zip(dois, lengths, strict=True))

    def screen(
        self,
        subject: str,
        *,
        off_subject: str | None = None,
        filters: ArticleFilters | None = None,
        limit: int | None = None,
    ) -> pd.DataFrame:
        """Score every eligible article's abstract against a subject.

        With `off_subject`, the ranking key is the margin between the two
        similarities -- the contrastive design of `transform/semantics.py`,
        whose zero means "closer to the off-subject than to the subject".
        """
        filters = filters or ArticleFilters()
        articles = self.corpus.articles
        eligible = articles[article_filter_mask(articles, filters)]
        columns = ["doi", "title", "year", "venue", "source", "relevance_margin", "theme_label"]
        frame = eligible[columns].reset_index(drop=True)
        if frame.empty:
            return frame.assign(rank=pd.Series(dtype=int), subject_score=pd.Series(dtype=float))
        frame["subject_score"] = frame["doi"].map(self._similarities(subject, filters, len(frame)))
        sort_by = "subject_score"
        if off_subject:
            frame["off_subject_score"] = frame["doi"].map(
                self._similarities(off_subject, filters, len(frame))
            )
            frame["subject_margin"] = frame["subject_score"] - frame["off_subject_score"]
            sort_by = "subject_margin"
        frame = frame.sort_values([sort_by, "doi"], ascending=[False, True], na_position="last")
        frame.insert(0, "rank", range(1, len(frame) + 1))
        frame = frame.reset_index(drop=True)
        return frame.head(limit) if limit else frame

    def _similarities(self, text: str, filters: ArticleFilters, limit: int) -> dict[str, float]:
        store = self._require_store()
        try:
            hits = store.query_dense(
                self.corpus.version_id,
                self._embed(text),
                query_filter=to_qdrant_filter(filters, chunk_type="abstract"),
                limit=limit,
                payload_fields=["doi"],
            )
        except VectorStoreUnavailable as exc:
            raise RetrievalUnavailable(str(exc)) from exc
        return {hit.payload["doi"]: hit.score for hit in hits}

    def evidence(self, doi: str, query: str, *, top_k: int = 3) -> list[Passage]:
        """The full-text passages of one article that best answer `query`.

        Passages, not documents: the PDF text stays local (SDD §11), and a
        client cites the DOI. Lexical-only when the vector store is down.
        """
        chunks = self.corpus.chunks
        rows = chunks[(chunks["doi"] == doi) & (chunks["chunk_type"] == "fulltext")]
        if rows.empty or not (query or "").strip():
            return []
        positions = rows.index.to_numpy()
        lexical_scores = self.corpus.chunk_index.scores(query)[positions]
        lexical = [
            int(seq)
            for seq, score in sorted(
                zip(rows["seq"], lexical_scores, strict=True), key=lambda p: (-p[1], p[0])
            )
            if score > 0
        ]
        rankings = [lexical]
        dense_scores: dict[int, float] = {}
        try:
            store = self._require_store()
            hits = store.query_dense(
                self.corpus.version_id,
                self._embed(query),
                query_filter=to_qdrant_filter(
                    ArticleFilters(exclude_non_articles=False), chunk_type="fulltext", doi=doi
                ),
                limit=len(rows),
                payload_fields=["seq"],
            )
            dense_scores = {int(hit.payload["seq"]): hit.score for hit in hits}
            rankings.append(sorted(dense_scores, key=lambda seq: (-dense_scores[seq], seq)))
        except RetrievalUnavailable:
            pass
        fused = rrf_fuse(rankings, k=RRF_K)
        best = sorted(fused, key=lambda seq: (-fused[seq], seq))[: max(1, int(top_k))]
        text_of = dict(zip(rows["seq"], rows["text"], strict=True))
        return [Passage(doi=doi, seq=int(seq), score=fused[seq], text=text_of[seq]) for seq in best]

    def article(self, doi: str) -> dict[str, Any] | None:
        articles = self.corpus.articles
        if doi not in articles.index:
            return None
        row = articles.loc[doi]
        fulltext = int(
            (
                (self.corpus.chunks["doi"] == doi)
                & (self.corpus.chunks["chunk_type"] == "fulltext")
            ).sum()
        )
        fields = (
            "doi", "title", "authors", "year", "venue", "source", "publication_category",
            "url", "abstract", "citation_count", "relevance_score", "offtopic_score",
            "relevance_margin", "theme_label", "has_pdf", "abstract_missing", "is_non_article",
        )  # fmt: skip
        record = {field: _clean(row[field]) for field in fields}
        record["year"] = _year(row["year"])
        record["authors"] = list(row["authors"] or [])
        record["fulltext_chunks"] = fulltext
        return record

    def status(self, *, measure_truncation: bool = False) -> dict[str, Any]:
        from lake_research_map.transform.embeddings import EMBED_MODEL_NAME, EMBED_MODEL_REVISION

        corpus = self.corpus
        chunks = corpus.chunks
        report: dict[str, Any] = {
            "version_id": corpus.version_id,
            "articles": len(corpus.articles),
            "abstract_chunks": int((chunks["chunk_type"] == "abstract").sum()),
            "fulltext_chunks": int((chunks["chunk_type"] == "fulltext").sum()),
            "fulltext_articles": corpus.fulltext_articles,
            "abstract_missing": int(corpus.articles["abstract_missing"].sum()),
            "embed_model": EMBED_MODEL_NAME,
            "embed_revision": EMBED_MODEL_REVISION,
            "collection": collection_name(corpus.version_id),
            "alias": ACTIVE_ALIAS,
            "store_reachable": False,
            "points": None,
            "alias_target": None,
        }
        if self.store is not None:
            try:
                report["points"] = self.store.count(corpus.version_id)
                report["alias_target"] = self.store.alias_target()
                report["store_reachable"] = True
            except VectorStoreUnavailable:
                pass
        report["alias_in_sync"] = report["alias_target"] == corpus.version_id
        if measure_truncation:
            lengths = self.abstract_token_lengths()
            if lengths is not None:
                window = self.context_window()
                report["context_window_tokens"] = window
                report["abstract_truncated"] = sum(
                    1 for length in lengths.values() if length > window
                )
        return report


def build_retriever(version_id: str | None = None) -> Retriever:
    """A retriever over the published version (or `version_id`).

    Lexical-only when the vector store does not answer: dense and hybrid
    searches then raise `RetrievalUnavailable`, which every caller turns into a
    message rather than a crash.
    """
    from lake_research_map.db.engines import get_session
    from lake_research_map.db.vector_store import get_vector_store
    from lake_research_map.retrieval.index import load_corpus

    session = get_session("gold")
    try:
        corpus = load_corpus(session, version_id)
    finally:
        session.close()
    if corpus is None:
        raise ValueError("no Gold version is published yet; run the pipeline first")
    try:
        store = get_vector_store()
    except VectorStoreUnavailable as exc:
        logger.warning("dense retrieval unavailable, only lexical search will answer: %s", exc)
        store = None
    return Retriever(corpus, store)


# The modes compared by the retrieval benchmark (RQ-08, WP-13) and pooled for
# relevance judgements. The all-chunks rows are the modes the stored v1
# judgements were pooled from; the abstract rows are the article-level default.
BENCHMARK_MODES: dict[str, dict[str, Any]] = {
    "Dense, all chunks": {"mode": "dense", "scope": "all"},
    "BM25, all chunks": {"mode": "bm25", "scope": "all"},
    "Hybrid RRF, all chunks": {"mode": "hybrid", "scope": "all"},
    "Dense, abstracts": {"mode": "dense", "scope": "abstract"},
    "BM25, abstracts": {"mode": "bm25", "scope": "abstract"},
    "Hybrid RRF, abstracts": {"mode": "hybrid", "scope": "abstract"},
    "Hybrid RRF + rerank, abstracts": {"mode": "hybrid", "scope": "abstract", "rerank": True},
}


def mode_runner(retriever: Retriever, settings: dict[str, Any]) -> Callable[[str, int], list[str]]:
    """`(query, depth) -> [doi]` for one benchmark mode, over every article.

    No eligibility filter: the judgements were pooled over the whole corpus,
    non-articles included, and a mode must be scored on the population it was
    judged on.
    """
    unfiltered = ArticleFilters(exclude_non_articles=False)

    def run(text: str, depth: int) -> list[str]:
        return [
            hit.doi for hit in retriever.search(text, top_k=depth, filters=unfiltered, **settings)
        ]

    return run
