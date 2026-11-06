"""Payload stored beside each chunk vector in Qdrant (ADR-08).

The payload is what lets a retrieval client filter and cite without a second
round-trip to MySQL: the chunk's own text and embedding provenance, the article
fields the dashboard's sidebar already filters on, and -- once the semantic
stage has run -- the screening signals. Every field is derived from Gold rows of
the same dataset version, so the payload of a published collection is as
immutable as those rows.

The derivations here (`primary_source`, `abstract_is_missing`, the contrastive
margin) are the single definitions the dashboard and the retrieval package use
too; a filter that means one thing in pandas and another in a Qdrant query would
silently return different articles for the same request.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from typing import Any

from lake_research_map.transform.embeddings import EMBED_MODEL_NAME, EMBED_MODEL_REVISION

# Below this an "abstract" is a placeholder or a stray fragment, and the
# abstract chunk is effectively the title plus keywords: 95 articles of the
# 3,115-article corpus measured on 2026-11-06.
ABSTRACT_MIN_CHARS = 50


# Both helpers take whatever a reader hands them: `None` from a database row,
# but NaN from a pandas 3 column, where a missing string or list is a float.


def primary_source(sources: Sequence[str] | None) -> str:
    """The publisher an article is attributed to: the first of its sources."""
    if isinstance(sources, list | tuple) and sources:
        return sources[0]
    return "unknown"


def abstract_is_missing(abstract: str | None) -> bool:
    if not isinstance(abstract, str):
        return True
    return len(abstract.strip()) < ABSTRACT_MIN_CHARS


def relevance_margin(relevance_score: float | None, offtopic_score: float | None) -> float | None:
    """Distance to the review's topic minus distance to the logistics reading.

    Zero is the screening threshold (CLAUDE.md, "Why the `semantic` stage
    exists"). Undefined when the off-topic anchor was never scored.
    """
    if relevance_score is None or offtopic_score is None:
        return None
    return float(relevance_score) - float(offtopic_score)


def article_payload(article: Any) -> dict[str, Any]:
    sources = list(article.sources or [])
    return {
        "title": article.title,
        "year": article.year,
        "venue": article.venue,
        "sources": sources,
        "source": primary_source(sources),
        "publication_category": article.publication_category,
        "is_non_article": bool(article.is_non_article),
        "has_pdf": bool(article.has_pdf),
        "abstract_missing": abstract_is_missing(article.abstract),
        "url": article.url,
    }


def semantic_payload(row: Any) -> dict[str, Any]:
    return {
        "relevance_score": float(row.relevance_score),
        "offtopic_score": None if row.offtopic_score is None else float(row.offtopic_score),
        "relevance_margin": relevance_margin(row.relevance_score, row.offtopic_score),
        "theme_id": int(row.theme_id),
        "theme_label": row.theme_label,
    }


def chunk_payload(
    *,
    version_id: str,
    doi: str,
    chunk_type: str,
    seq: int,
    text: str,
    text_sha256: str,
    embedded_at: datetime | None,
    article: Any | None = None,
    semantics: Any | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "dataset_version_id": version_id,
        "doi": doi,
        "chunk_type": chunk_type,
        "seq": int(seq),
        "text": text,
        "text_sha256": text_sha256,
        "char_len": len(text),
        "embed_model": EMBED_MODEL_NAME,
        "embed_revision": EMBED_MODEL_REVISION,
        "embedded_at": embedded_at.isoformat() if embedded_at else None,
    }
    if article is not None:
        payload.update(article_payload(article))
    if semantics is not None:
        payload.update(semantic_payload(semantics))
    return payload
