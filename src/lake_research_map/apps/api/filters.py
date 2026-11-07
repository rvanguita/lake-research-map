"""One definition of "which articles are eligible", for pandas and for Qdrant.

The dashboard's sidebar, the CLI flags and the MCP tool arguments all build an
`ArticleFilters`. Lexical scoring applies it as a pandas mask over Gold rows;
dense search sends it to Qdrant as a server-side filter over the payload
written by `transform/vector_payloads.py`. Both must select the same articles,
or a hybrid ranking would fuse two different populations --
`tests/test_retrieval_filters.py` holds them to it.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass(frozen=True, slots=True)
class ArticleFilters:
    """Eligibility rules; the defaults keep every real article.

    `min_margin` keeps articles with no score: a missing signal must never
    silently shrink the corpus (the same rule as the sidebar's margin filter).
    """

    year_range: tuple[int, int] | None = None
    sources: tuple[str, ...] = ()
    venues: tuple[str, ...] = ()
    publication_categories: tuple[str, ...] = ()
    min_margin: float | None = None
    exclude_non_articles: bool = True
    require_abstract: bool = False


def _year_bounds(year_range: Any) -> tuple[int, int] | None:
    if year_range is None:
        return None
    if isinstance(year_range, int | float):
        return int(year_range), int(year_range)
    low, high = year_range
    return int(low), int(high)


def article_filter_mask(df: pd.DataFrame, filters: ArticleFilters) -> pd.Series:
    """Boolean mask over an article frame; a filter on an absent column is skipped."""
    mask = pd.Series(True, index=df.index)
    if filters.min_margin is not None:
        # `relevance_margin` needs `offtopic_score`, which a database whose last
        # semantic run predates the contrastive anchor does not have.
        column = "relevance_margin" if "relevance_margin" in df.columns else "relevance_score"
        if column in df.columns:
            mask &= df[column].isna() | df[column].ge(filters.min_margin)
    bounds = _year_bounds(filters.year_range)
    if bounds and "year" in df.columns:
        years = pd.to_numeric(df["year"], errors="coerce")
        mask &= years.ge(bounds[0]) & years.le(bounds[1])
    for values, column in (
        (filters.sources, "source"),
        (filters.venues, "venue"),
        (filters.publication_categories, "publication_category"),
    ):
        if values and column in df.columns:
            mask &= df[column].isin(values)
    if filters.exclude_non_articles and "is_non_article" in df.columns:
        mask &= ~df["is_non_article"].fillna(False).astype(bool)
    if filters.require_abstract and "abstract_missing" in df.columns:
        mask &= ~df["abstract_missing"].fillna(True).astype(bool)
    return mask


def apply_article_filters(df: pd.DataFrame, filters: ArticleFilters) -> pd.DataFrame:
    return df.loc[article_filter_mask(df, filters)]


def to_qdrant_filter(
    filters: ArticleFilters,
    *,
    chunk_type: str | None = None,
    doi: str | None = None,
):
    """The same rules as `article_filter_mask`, as a Qdrant payload filter."""
    from qdrant_client import models

    must: list[Any] = []
    must_not: list[Any] = []
    if chunk_type is not None:
        must.append(
            models.FieldCondition(key="chunk_type", match=models.MatchValue(value=chunk_type))
        )
    if doi is not None:
        must.append(models.FieldCondition(key="doi", match=models.MatchValue(value=doi)))
    if filters.min_margin is not None:
        must.append(
            models.Filter(
                should=[
                    models.FieldCondition(
                        key="relevance_margin", range=models.Range(gte=filters.min_margin)
                    ),
                    models.IsNullCondition(is_null=models.PayloadField(key="relevance_margin")),
                    models.IsEmptyCondition(is_empty=models.PayloadField(key="relevance_margin")),
                ]
            )
        )
    bounds = _year_bounds(filters.year_range)
    if bounds:
        must.append(
            models.FieldCondition(key="year", range=models.Range(gte=bounds[0], lte=bounds[1]))
        )
    for values, key in (
        (filters.sources, "source"),
        (filters.venues, "venue"),
        (filters.publication_categories, "publication_category"),
    ):
        if values:
            must.append(models.FieldCondition(key=key, match=models.MatchAny(any=list(values))))
    if filters.exclude_non_articles:
        must_not.append(
            models.FieldCondition(key="is_non_article", match=models.MatchValue(value=True))
        )
    if filters.require_abstract:
        must_not.append(
            models.FieldCondition(key="abstract_missing", match=models.MatchValue(value=True))
        )
    if not must and not must_not:
        return None
    return models.Filter(must=must or None, must_not=must_not or None)


def filters_from_args(
    *,
    year_min: int | None = None,
    year_max: int | None = None,
    sources: Sequence[str] | None = None,
    venues: Sequence[str] | None = None,
    publication_categories: Sequence[str] | None = None,
    min_margin: float | None = None,
    require_abstract: bool = False,
    include_non_articles: bool = False,
) -> ArticleFilters:
    """Build filters from flat CLI/MCP arguments; an open year bound stays open."""
    year_range = None
    if year_min is not None or year_max is not None:
        year_range = (
            int(year_min) if year_min is not None else -(10**9),
            int(year_max) if year_max is not None else 10**9,
        )
    return ArticleFilters(
        year_range=year_range,
        sources=tuple(sources or ()),
        venues=tuple(venues or ()),
        publication_categories=tuple(publication_categories or ()),
        min_margin=min_margin,
        exclude_non_articles=not include_non_articles,
        require_abstract=require_abstract,
    )
