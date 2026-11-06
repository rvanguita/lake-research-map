"""MCP server: the retrieval operations as tools for an LLM client (stdio).

Register it with Claude Code:

    claude mcp add lake-research-map -- \\
        uv run --directory /path/to/lake-research-map lake-research-map mcp

The tools are plain functions returning JSON-serializable dicts, so they are
tested without a server; `build_server` only registers them. The retriever is
built on the first call (the published Gold version, Qdrant for dense search)
and reused for the life of the process. Nothing may be printed to stdout: on
the stdio transport it carries the protocol.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from typing import Any

from lake_research_map.retrieval.api import RetrievalUnavailable, Retriever, build_retriever
from lake_research_map.retrieval.filters import filters_from_args

INSTRUCTIONS = (
    "Systematic-review corpus on electric power distribution system planning "
    "(IEEE Xplore + ScienceDirect exports). Use search_articles to find the most "
    "relevant articles for a subject; it returns one hit per article, ranked over the "
    "abstracts by default. Use screen_articles to score every article against a subject "
    "(optionally against a competing reading, e.g. the logistics sense of 'distribution'). "
    "Use get_evidence for short full-text passages of one article, and cite the DOI. "
    "Scores rank; they are not probabilities. No retrieval mode is validated against "
    "human relevance labels yet."
)

_retriever: Retriever | None = None


def configure(retriever: Retriever | None) -> None:
    """Inject a retriever (tests), or reset to build one on the next call."""
    global _retriever
    _retriever = retriever


def _get() -> Retriever:
    global _retriever
    if _retriever is None:
        _retriever = build_retriever()
    return _retriever


def search_articles(
    query: str,
    top_k: int = 10,
    mode: str = "hybrid",
    scope: str = "abstract",
    rerank: bool = False,
    year_min: int | None = None,
    year_max: int | None = None,
    sources: list[str] | None = None,
    venues: list[str] | None = None,
    publication_categories: list[str] | None = None,
    min_margin: float | None = None,
    require_abstract: bool = False,
    include_non_articles: bool = False,
) -> dict[str, Any]:
    """Rank the published articles for `query`, one hit per article.

    mode: "hybrid" (dense + BM25 fused by reciprocal rank, the default), "dense"
    (embedding cosine) or "bm25" (lexical; exact terms such as "IEEE 33-bus").
    scope: "abstract" (default; one abstract per article, every article on equal
    terms) or "all" (full-text chunks rank too; only ~3% of the articles have a
    PDF, so this favours them). rerank: reorder candidates with a local
    cross-encoder; experimental. min_margin: drop articles semantically closer to
    the logistics reading of "distribution" than this (0 is the screening
    threshold; unscored articles are kept). Each hit carries the DOI, title,
    year, venue, the component ranks and scores, a short excerpt, and flags for a
    missing or truncated abstract.
    """
    filters = filters_from_args(
        year_min=year_min,
        year_max=year_max,
        sources=sources,
        venues=venues,
        publication_categories=publication_categories,
        min_margin=min_margin,
        require_abstract=require_abstract,
        include_non_articles=include_non_articles,
    )
    retriever = _get()
    try:
        hits = retriever.search(
            query, top_k=top_k, mode=mode, scope=scope, filters=filters, rerank=rerank
        )
    except (RetrievalUnavailable, ValueError) as exc:
        return {"error": str(exc), "hits": []}
    return {
        "version_id": retriever.corpus.version_id,
        "filters": asdict(filters),
        "hits": [hit.to_dict() for hit in hits],
    }


def screen_articles(
    subject: str,
    off_subject: str | None = None,
    limit: int = 50,
    year_min: int | None = None,
    year_max: int | None = None,
    sources: list[str] | None = None,
    min_margin: float | None = None,
    include_non_articles: bool = False,
) -> dict[str, Any]:
    """Score every eligible article's abstract against `subject` (cosine similarity).

    With `off_subject`, articles are ranked by subject_score minus
    off_subject_score; a negative margin means the abstract is closer to the
    competing reading. Returns the `limit` best (0 for all).
    """
    filters = filters_from_args(
        year_min=year_min,
        year_max=year_max,
        sources=sources,
        min_margin=min_margin,
        include_non_articles=include_non_articles,
    )
    retriever = _get()
    try:
        frame = retriever.screen(
            subject, off_subject=off_subject, filters=filters, limit=limit or None
        )
    except RetrievalUnavailable as exc:
        return {"error": str(exc), "articles": []}
    return {
        "version_id": retriever.corpus.version_id,
        "articles": json.loads(frame.to_json(orient="records")),
    }


def get_article(doi: str) -> dict[str, Any]:
    """Metadata, abstract, screening signals and full-text availability of one DOI."""
    record = _get().article(doi)
    return record if record is not None else {"error": f"unknown DOI {doi}"}


def get_evidence(doi: str, query: str, top_k: int = 3) -> dict[str, Any]:
    """The full-text passages of one article that best answer `query`.

    Empty when the article has no linked PDF. Passages are short chunks of the
    paper's own text (reference lists are removed before chunking).
    """
    passages = _get().evidence(doi, query, top_k=top_k)
    return {"doi": doi, "passages": [passage.to_dict() for passage in passages]}


def corpus_status(measure_truncation: bool = False) -> dict[str, Any]:
    """Published version, counts, models, vector-store state and coverage caveats.

    measure_truncation counts abstracts longer than the embedding model's
    512-token window (their tail does not reach the vector); it loads a tokenizer.
    """
    return _get().status(measure_truncation=measure_truncation)


TOOLS = (search_articles, screen_articles, get_article, get_evidence, corpus_status)


def build_server():
    # mcp 2.x: `FastMCP` became `MCPServer` (same decorator and stdio `run`).
    from mcp.server.mcpserver import MCPServer

    server = MCPServer("lake-research-map", instructions=INSTRUCTIONS)
    for tool in TOOLS:
        server.add_tool(tool)
    return server


def serve_stdio() -> None:
    build_server().run(transport="stdio")
