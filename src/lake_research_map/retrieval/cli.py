"""`lake-research-map search | screen | mcp`.

All three only read: none takes the pipeline lock, so a long-running MCP
server never blocks a pipeline run (`pipeline._mutates_published_inputs`).
With `--json`, stdout carries exactly one JSON document and nothing else --
logs go to stderr -- so the output can be piped into another program.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict

from lake_research_map.retrieval.api import MODES, SCOPES, Retriever, build_retriever
from lake_research_map.retrieval.filters import ArticleFilters, filters_from_args

logger = logging.getLogger(__name__)

COMMANDS = ("search", "screen", "mcp")


def _filter_arguments(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("article filters (applied before ranking)")
    group.add_argument("--year-min", type=int)
    group.add_argument("--year-max", type=int)
    group.add_argument(
        "--source", action="append", dest="sources", help="repeatable: ieee, elsevier"
    )
    group.add_argument("--venue", action="append", dest="venues", help="repeatable")
    group.add_argument(
        "--category",
        action="append",
        dest="publication_categories",
        help="repeatable: journal, conference, review, other",
    )
    group.add_argument(
        "--min-margin",
        type=float,
        help="drop articles closer to the logistics reading than this margin (0 is the "
        "screening threshold); articles with no score are kept",
    )
    group.add_argument(
        "--require-abstract",
        action="store_true",
        help="drop articles whose abstract is missing (under 50 characters)",
    )
    group.add_argument(
        "--include-non-articles",
        action="store_true",
        help="keep front matter and other records flagged as non-articles",
    )


def configure(subparsers) -> None:
    search = subparsers.add_parser(
        "search", help="Rank the published articles for a query (one hit per article)"
    )
    search.add_argument("query")
    search.add_argument("--top-k", type=int, default=10)
    search.add_argument("--mode", choices=MODES, default="hybrid")
    search.add_argument(
        "--scope",
        choices=SCOPES,
        default="abstract",
        help="abstract: one chunk per article (default); all: every chunk, biased towards "
        "articles that have a PDF",
    )
    search.add_argument(
        "--rerank",
        action="store_true",
        help="reorder the candidates with a local cross-encoder (experimental; not an "
        "approved default without human relevance labels)",
    )
    search.add_argument("--candidates", type=int, default=50)
    _filter_arguments(search)
    search.add_argument("--json", action="store_true", help="print one JSON document")

    screen = subparsers.add_parser(
        "screen", help="Score every published article's abstract against a subject"
    )
    screen.add_argument("subject")
    screen.add_argument(
        "--off-subject",
        help="a competing reading of the subject; ranks by the margin between the two",
    )
    screen.add_argument("--limit", type=int, default=50, help="0 for every article")
    _filter_arguments(screen)
    screen.add_argument("--json", action="store_true", help="print one JSON document")

    subparsers.add_parser(
        "mcp", help="Serve search, screening and evidence as MCP tools over stdio"
    )


def filters_from_namespace(args: argparse.Namespace) -> ArticleFilters:
    return filters_from_args(
        year_min=args.year_min,
        year_max=args.year_max,
        sources=args.sources,
        venues=args.venues,
        publication_categories=args.publication_categories,
        min_margin=args.min_margin,
        require_abstract=args.require_abstract,
        include_non_articles=args.include_non_articles,
    )


def search_payload(retriever: Retriever, args: argparse.Namespace) -> dict:
    filters = filters_from_namespace(args)
    hits = retriever.search(
        args.query,
        top_k=args.top_k,
        mode=args.mode,
        scope=args.scope,
        filters=filters,
        rerank=args.rerank,
        candidates=args.candidates,
    )
    return {
        "version_id": retriever.corpus.version_id,
        "query": args.query,
        "mode": args.mode + ("+rerank" if args.rerank else ""),
        "scope": args.scope,
        "filters": asdict(filters),
        "hits": [hit.to_dict() for hit in hits],
    }


def screen_payload(retriever: Retriever, args: argparse.Namespace) -> dict:
    filters = filters_from_namespace(args)
    frame = retriever.screen(
        args.subject,
        off_subject=args.off_subject,
        filters=filters,
        limit=args.limit or None,
    )
    records = json.loads(frame.to_json(orient="records"))
    return {
        "version_id": retriever.corpus.version_id,
        "subject": args.subject,
        "off_subject": args.off_subject,
        "filters": asdict(filters),
        "articles": records,
    }


def render_hits(payload: dict) -> str:
    lines = [f"version {payload['version_id'][:12]}  {payload['mode']} over {payload['scope']}"]
    for hit in payload["hits"]:
        title = (hit["title"] or "")[:72]
        lines.append(
            f"{hit['rank']:>3}  {hit['score']:.4f}  {hit['year'] or '----'}  {hit['doi']:<40} {title}"
        )
    return "\n".join(lines)


def render_screen(payload: dict) -> str:
    key = "subject_margin" if payload["off_subject"] else "subject_score"
    lines = [f"version {payload['version_id'][:12]}  ranked by {key}"]
    for row in payload["articles"]:
        value = row.get(key)
        shown = "  n/a " if value is None else f"{value:+.4f}"
        lines.append(f"{row['rank']:>4}  {shown}  {row['doi']:<40} {(row['title'] or '')[:64]}")
    return "\n".join(lines)


def run(args: argparse.Namespace) -> None:
    # A request log line per Qdrant call drowns the output a person reads.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    if args.command == "mcp":
        from lake_research_map.retrieval.mcp_server import serve_stdio

        serve_stdio()
        return
    retriever = build_retriever()
    if args.command == "search":
        payload = search_payload(retriever, args)
        print(json.dumps(payload, ensure_ascii=False) if args.json else render_hits(payload))
    else:
        payload = screen_payload(retriever, args)
        print(json.dumps(payload, ensure_ascii=False) if args.json else render_screen(payload))
