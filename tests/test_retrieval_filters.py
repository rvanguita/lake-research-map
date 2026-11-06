"""Article filters: one definition, the same articles in pandas and in Qdrant."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from lake_research_map.db.vector_store import VectorPoint, point_id
from lake_research_map.retrieval.filters import (
    ArticleFilters,
    apply_article_filters,
    filters_from_args,
    to_qdrant_filter,
)
from lake_research_map.transform.vector_payloads import (
    abstract_is_missing,
    article_payload,
    primary_source,
    relevance_margin,
    semantic_payload,
)

ARTICLES = [
    # doi, sources, year, venue, category, non_article, abstract, relevance, offtopic
    ("10.1/a", ["ieee"], 2018, "TPWRS", "journal", False, "A" * 80, 0.80, 0.60),
    ("10.1/b", ["elsevier"], 2021, "IJEPES", "journal", False, "B" * 80, 0.62, 0.70),
    ("10.1/c", ["ieee", "elsevier"], 2023, "ISGT", "conference", False, "short", 0.75, 0.65),
    ("10.1/d", ["elsevier"], None, "IJEPES", "journal", False, "D" * 80, None, None),
    ("10.1/e", [], 2024, None, "other", True, "E" * 80, 0.70, None),
    ("10.1/f", ["ieee"], 2027, "TPWRS", "review", False, "F" * 80, 0.78, 0.50),
]


def _frame() -> pd.DataFrame:
    rows = []
    for doi, sources, year, venue, category, non_article, abstract, rel, off in ARTICLES:
        rows.append(
            {
                "doi": doi,
                "source": primary_source(sources),
                "year": year,
                "venue": venue,
                "publication_category": category,
                "is_non_article": non_article,
                "abstract_missing": abstract_is_missing(abstract),
                "relevance_score": rel,
                "offtopic_score": off,
                "relevance_margin": relevance_margin(rel, off),
            }
        )
    return pd.DataFrame(rows)


def _store_with_payloads(vector_store) -> None:
    vector_store.ensure_collection("v1", 2)
    points = []
    for index, (doi, sources, year, venue, category, non_article, abstract, rel, off) in enumerate(
        ARTICLES
    ):
        payload = {"doi": doi, "chunk_type": "abstract"}
        payload.update(
            article_payload(
                SimpleNamespace(
                    sources=sources,
                    title=doi,
                    year=year,
                    venue=venue,
                    publication_category=category,
                    is_non_article=non_article,
                    has_pdf=False,
                    abstract=abstract,
                    url=None,
                )
            )
        )
        if rel is not None:
            payload.update(
                semantic_payload(
                    SimpleNamespace(
                        relevance_score=rel, offtopic_score=off, theme_id=0, theme_label="T"
                    )
                )
            )
        angle = index / len(ARTICLES)
        points.append(
            VectorPoint(
                point_id(doi, "abstract", 0),
                np.array([np.cos(angle), np.sin(angle)], dtype=np.float32),
                payload,
            )
        )
    vector_store.upsert("v1", points)


FILTER_CASES = [
    ArticleFilters(),
    ArticleFilters(exclude_non_articles=False),
    ArticleFilters(min_margin=0.0),
    ArticleFilters(min_margin=0.1, exclude_non_articles=False),
    ArticleFilters(year_range=(2019, 2024)),
    ArticleFilters(year_range=(2018, 2027), sources=("ieee",)),
    ArticleFilters(venues=("IJEPES",), publication_categories=("journal",)),
    ArticleFilters(require_abstract=True, exclude_non_articles=False),
    filters_from_args(year_min=2020, min_margin=-0.05),
]


@pytest.mark.parametrize("filters", FILTER_CASES, ids=lambda f: repr(f)[:60])
def test_pandas_and_qdrant_select_the_same_articles(filters, vector_store):
    _store_with_payloads(vector_store)

    expected = set(apply_article_filters(_frame(), filters)["doi"])
    stored = vector_store.scroll(
        "v1", query_filter=to_qdrant_filter(filters, chunk_type="abstract"), payload_fields=["doi"]
    )

    assert {point.payload["doi"] for point in stored} == expected


def test_the_margin_filter_keeps_unscored_articles():
    kept = set(apply_article_filters(_frame(), ArticleFilters(min_margin=0.0))["doi"])

    # b is closer to logistics (negative margin); d has no score at all.
    assert kept == {"10.1/a", "10.1/c", "10.1/d", "10.1/f"}


def test_non_articles_are_excluded_by_default_and_kept_on_request():
    default = set(apply_article_filters(_frame(), ArticleFilters())["doi"])
    everything = set(
        apply_article_filters(_frame(), ArticleFilters(exclude_non_articles=False))["doi"]
    )

    assert "10.1/e" not in default
    assert "10.1/e" in everything


def test_require_abstract_uses_the_fifty_character_rule():
    kept = set(apply_article_filters(_frame(), ArticleFilters(require_abstract=True))["doi"])

    assert "10.1/c" not in kept
    assert abstract_is_missing("x" * 49) and not abstract_is_missing("x" * 50)


def test_filters_from_args_leaves_an_open_year_bound_open():
    filters = filters_from_args(year_max=2020, sources=["ieee"], include_non_articles=True)

    assert filters.year_range[1] == 2020 and filters.year_range[0] < 1900
    assert filters.sources == ("ieee",)
    assert filters.exclude_non_articles is False
    assert to_qdrant_filter(ArticleFilters(exclude_non_articles=False)) is None


def test_missing_values_from_pandas_count_as_missing():
    # pandas 3 hands a missing string or list over as NaN, not None.
    nan = float("nan")

    assert abstract_is_missing(nan) and abstract_is_missing(None)
    assert primary_source(nan) == "unknown" and primary_source([]) == "unknown"
    assert primary_source(["elsevier", "ieee"]) == "elsevier"
