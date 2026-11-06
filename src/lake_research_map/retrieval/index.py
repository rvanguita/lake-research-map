"""The published version's articles and chunk texts, loaded once per process.

Vectors stay in Qdrant (ADR-08). What is loaded here is what filtering, lexical
scoring and rendering a result need: one row per article with its metadata and
screening signals, and every chunk's text keyed by its point id. Reading goes
through a Gold session, so the whole package runs against the in-memory
SQLite fixture in tests.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from lake_research_map.db.gold_models import DatasetArticle, DatasetChunk, DatasetSemantics
from lake_research_map.db.vector_store import point_id
from lake_research_map.retrieval.rank import BM25Index
from lake_research_map.transform.vector_payloads import abstract_is_missing, primary_source
from lake_research_map.transform.versioned_gold import active_version_id

_ARTICLE_COLUMNS = (
    DatasetArticle.doi,
    DatasetArticle.title,
    DatasetArticle.authors,
    DatasetArticle.year,
    DatasetArticle.venue,
    DatasetArticle.sources,
    DatasetArticle.publication_category,
    DatasetArticle.is_non_article,
    DatasetArticle.has_pdf,
    DatasetArticle.abstract,
    DatasetArticle.url,
    DatasetArticle.citation_count,
)
_SEMANTIC_COLUMNS = (
    DatasetSemantics.doi,
    DatasetSemantics.relevance_score,
    DatasetSemantics.offtopic_score,
    DatasetSemantics.theme_id,
    DatasetSemantics.theme_label,
)


@dataclass
class Corpus:
    """One published version: `articles` indexed by DOI, `chunks` in key order."""

    version_id: str
    articles: pd.DataFrame
    chunks: pd.DataFrame

    @cached_property
    def abstracts(self) -> pd.DataFrame:
        return self.chunks[self.chunks["chunk_type"] == "abstract"].reset_index(drop=True)

    @cached_property
    def abstract_index(self) -> BM25Index:
        return BM25Index(self.abstracts["text"].tolist())

    @cached_property
    def chunk_index(self) -> BM25Index:
        # Built only when a query asks for the all-chunks scope.
        return BM25Index(self.chunks["text"].tolist())

    @cached_property
    def text_by_key(self) -> dict[tuple[str, str, int], str]:
        return dict(
            zip(
                zip(self.chunks["doi"], self.chunks["chunk_type"], self.chunks["seq"], strict=True),
                self.chunks["text"],
                strict=True,
            )
        )

    @cached_property
    def fulltext_articles(self) -> int:
        return int(self.chunks.loc[self.chunks["chunk_type"] == "fulltext", "doi"].nunique())


def _frame(session: Session, statement, columns: list[str]) -> pd.DataFrame:
    rows = session.execute(statement).all()
    return pd.DataFrame([tuple(row) for row in rows], columns=columns)


def load_corpus(session: Session, version_id: str | None = None) -> Corpus | None:
    """Load a version (the published one by default); None when there is nothing to load."""
    version_id = version_id or active_version_id(session)
    if version_id is None:
        return None
    articles = _frame(
        session,
        select(*_ARTICLE_COLUMNS).where(DatasetArticle.dataset_version_id == version_id),
        [column.key for column in _ARTICLE_COLUMNS],
    )
    if articles.empty:
        return None
    semantics = _frame(
        session,
        select(*_SEMANTIC_COLUMNS).where(DatasetSemantics.dataset_version_id == version_id),
        [column.key for column in _SEMANTIC_COLUMNS],
    )
    articles = articles.merge(semantics, on="doi", how="left")
    articles["source"] = articles["sources"].map(lambda value: primary_source(value or []))
    articles["abstract_missing"] = articles["abstract"].map(abstract_is_missing)
    # NaN wherever either score is missing -- the payload's null, and what the
    # margin filter keeps.
    articles["relevance_margin"] = pd.to_numeric(
        articles["relevance_score"], errors="coerce"
    ) - pd.to_numeric(articles["offtopic_score"], errors="coerce")
    articles = articles.set_index("doi", drop=False).sort_index()

    chunks = _frame(
        session,
        select(DatasetChunk.doi, DatasetChunk.chunk_type, DatasetChunk.seq, DatasetChunk.text)
        .where(DatasetChunk.dataset_version_id == version_id)
        .order_by(DatasetChunk.doi, DatasetChunk.chunk_type, DatasetChunk.seq),
        ["doi", "chunk_type", "seq", "text"],
    )
    chunks["seq"] = chunks["seq"].astype(int)
    chunks["point_id"] = [
        point_id(doi, chunk_type, seq)
        for doi, chunk_type, seq in zip(
            chunks["doi"], chunks["chunk_type"], chunks["seq"], strict=True
        )
    ]
    return Corpus(version_id=version_id, articles=articles, chunks=chunks)
