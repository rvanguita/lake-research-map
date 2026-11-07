"""`enrichment refresh-openalex|refresh-citations|crossref-references|resolve-references`.

The OpenAlex and Crossref metadata crawls. Only `refresh-openalex` feeds a stage, so
only it takes the pipeline lock (`cli.main._mutates_published_inputs`).
"""

from __future__ import annotations

import argparse
import logging
import os
from datetime import UTC, datetime

from sqlalchemy import select

from db.bootstrap import bootstrap
from runs import _layer_sessions

logger = logging.getLogger(__name__)


def configure(subparsers) -> None:
    enrichment = subparsers.add_parser(
        "enrichment", help="Append reproducible external metadata observations"
    )
    actions = enrichment.add_subparsers(dest="enrichment_action", required=True)
    refresh = actions.add_parser("refresh-openalex", help="Refresh the active DOI population")
    refresh.add_argument("--max-fetch", type=int, default=100)
    # 0.1s is exactly OpenAlex's stated 10 req/s ceiling, so a live crawl on
    # 2026-09-22 was throttled outright after ~1,100 requests and never
    # recovered. Headroom costs minutes over 3,115 DOIs; being blocked costs
    # the whole run.
    refresh.add_argument("--delay", type=float, default=0.25)
    refresh.add_argument(
        "--refresh-all",
        action="store_true",
        help="Re-observe DOIs that already succeeded, for a deliberate as-of snapshot "
        "instead of gap-filling",
    )
    refresh.add_argument(
        "--commit-every",
        type=int,
        default=25,
        help="Persist progress every N fetches so an interrupt keeps what it already got",
    )
    refresh.add_argument(
        "--store-payload",
        action="store_true",
        help="Also store each raw Work JSON; adds hundreds of MB over the corpus and "
        "nothing reads it back",
    )
    citations = actions.add_parser(
        "refresh-citations",
        help="Crawl incoming citation edges for works already observed in Bronze",
    )
    citations.add_argument("--max-works", type=int, default=50)
    citations.add_argument("--delay", type=float, default=0.2)
    citations.add_argument(
        "--refresh-all",
        action="store_true",
        help="Re-crawl works already carrying a crawl timestamp",
    )
    citations.add_argument(
        "--commit-every",
        type=int,
        default=25,
        help="Persist progress every N works so an interrupt keeps what it already got",
    )
    citations.add_argument(
        "--max-pages",
        type=int,
        default=None,
        help="Page budget per batch; a batch that exhausts it marks its works truncated",
    )
    citations.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="Works OR-joined into one cites: query (1 = one query per work)",
    )
    crossref = actions.add_parser(
        "crossref-references",
        help="Collect Crossref reference deposits and date their references (no OpenAlex quota)",
    )
    crossref.add_argument("--delay", type=float, default=0.1)
    references = actions.add_parser(
        "resolve-references",
        help="Record the publication year of every work the corpus cites (Price, reference age)",
    )
    references.add_argument("--max-fetch", type=int, default=60_000)
    references.add_argument("--batch-size", type=int, default=100)
    references.add_argument("--delay", type=float, default=0.25)


def _require_openalex_identity() -> str:
    """OpenAlex has no API key; what it asks for is a contact address.

    The gate used to demand `OPENALEX_API_KEY` as well, which no OpenAlex
    account issues for the public corpus, so every refresh was unreachable by
    construction -- the same shape of defect as the database roles WP-08
    removed. The polite-pool address is mandatory because an anonymous crawl
    of this size is what gets a client rate-limited; a key is accepted when
    one exists and never required.
    """
    email = (os.environ.get("OPENALEX_EMAIL") or "").strip()
    if not email:
        raise ValueError(
            "OPENALEX_EMAIL must be set: OpenAlex identifies polite-pool clients by a contact "
            "address, and an anonymous crawl of the whole corpus will be throttled. "
            "OPENALEX_API_KEY is optional and only applies to a premium account."
        )
    return email


def _run_crossref_references(args: argparse.Namespace) -> None:
    """Collect Crossref reference lists and date their references (WP-23).

    Kept out of the OpenAlex gate on purpose: Crossref reads its own
    `CROSSREF_EMAIL`, because consent to send a contact address is per service.
    Without one the public pool is used, at a lower rate, rather than failing.
    """
    from db.gold_models import DatasetArticle, PublicationState
    from ingest.crossref import collect_crossref_references

    bootstrap()
    with _layer_sessions("gold", "bronze") as (gold, bronze):
        state = gold.get(PublicationState, 1)
        if state is None or not state.active_version_id:
            raise ValueError("no active dataset version")
        dois = gold.scalars(
            select(DatasetArticle.doi).where(
                DatasetArticle.dataset_version_id == state.active_version_id
            )
        ).all()
        stats = collect_crossref_references(bronze, list(dois), delay=args.delay)
        print(stats)
        if stats.get("stopped_early"):
            print(
                f"Stopped early ({stats['stopped_early']}). Everything fetched is committed; "
                "re-run to continue from where this run stopped."
            )


def run(args: argparse.Namespace) -> None:
    from db.gold_models import DatasetArticle, PublicationState
    from ingest.openalex import refresh_openalex_observations

    if args.enrichment_action == "crossref-references":
        _run_crossref_references(args)
        return
    _require_openalex_identity()
    bootstrap()
    with _layer_sessions("gold", "bronze") as (gold_session, bronze_session):
        # The forward crawl reads Bronze only. Requiring an active Gold version
        # for it would fail with "no active dataset version", which says
        # nothing about the command the operator actually ran.
        if args.enrichment_action == "resolve-references":
            from ingest.openalex import resolve_reference_years

            ref_stats = resolve_reference_years(
                bronze_session,
                max_fetch=args.max_fetch,
                batch_size=args.batch_size,
                delay=args.delay,
            )
            print(ref_stats)
            if ref_stats.get("stopped_early"):
                print(
                    f"Stopped early ({ref_stats['stopped_early']}). Progress is committed; "
                    "re-run after the quota window resets to continue."
                )
            elif ref_stats["remaining"]:
                print(f"{ref_stats['remaining']} cited works still unresolved. Re-run to continue.")
            return
        if args.enrichment_action == "refresh-citations":
            citing_stats = _refresh_citation_edges(bronze_session, args)
            print(citing_stats)
            if citing_stats.get("stopped_early"):
                print(
                    "Stopped early: five consecutive works came back throttled. Progress is "
                    "committed and a re-run resumes from the first uncrawled work."
                )
            elif citing_stats["remaining"]:
                print(f"{citing_stats['remaining']} works still uncrawled. Re-run to continue.")
            return
        state = gold_session.get(PublicationState, 1)
        if state is None or not state.active_version_id:
            raise ValueError("no active dataset version")
        dois = gold_session.scalars(
            select(DatasetArticle.doi)
            .where(DatasetArticle.dataset_version_id == state.active_version_id)
            .order_by(DatasetArticle.doi)
        ).all()
        stats = refresh_openalex_observations(
            bronze_session,
            dois,
            max_fetch=args.max_fetch,
            delay=args.delay,
            refresh_all=args.refresh_all,
            commit_every=args.commit_every,
            store_payload=args.store_payload,
        )
        print(stats)
        if stats.get("stopped_early"):
            print(
                f"Stopped early: OpenAlex returned {stats['stopped_early']} five times in a "
                "row, so the run halted instead of spending the rest of the batch on an API "
                "that had stopped answering. Everything fetched is committed. Check the "
                "logged X-RateLimit headers above: when the limit is a per-window quota "
                "rather than a rate, a larger --delay buys nothing and the only fix is to "
                "wait for the reset. Re-running then resumes from here."
            )
        elif stats["remaining"]:
            print(
                f"{stats['remaining']} DOIs still unobserved. Re-run the same command to "
                "continue; it resumes from where this run stopped."
            )


def _refresh_citation_edges(bronze_session, args: argparse.Namespace) -> dict:
    """Collect forward citation edges for works `refresh-openalex` already saw.

    The crawl is driven from `lit_external_works` rather than from DOIs,
    because `cites:` filters on an OpenAlex work id: a DOI the backward pass
    never resolved has no id to crawl, and asking for one would be a second
    lookup per work for no extra evidence.

    Truncation is recorded, not smoothed over. A work whose page budget ran
    out has an incomplete forward set, and `citation_graph_coverage` has to be
    able to exclude it -- otherwise a disruption index would be computed over
    a citation tail that was silently cut off.
    """
    import time

    from db.bronze_models import ExternalWork
    from ingest.openalex import (
        CITING_BATCH_MAX_PAGES,
        CITING_BATCH_SIZE,
        CONSECUTIVE_FAILURE_LIMIT,
        fetch_openalex_citing_batch,
        interleave_by_registrant,
        persist_incoming_edges,
    )

    observed_at = datetime.now(UTC).replace(tzinfo=None)
    works = bronze_session.scalars(
        select(ExternalWork).where(ExternalWork.provider == "openalex")
    ).all()
    if not works:
        raise ValueError(
            "no OpenAlex works have been observed yet; run `enrichment refresh-openalex` first"
        )

    # Same resumability rule as the backward pass: a work is "done" when it
    # carries a crawl timestamp, not when it has an edge -- a work nobody
    # cites produces no edge and would otherwise be retried forever.
    pending = [w for w in works if getattr(w, "citing_crawled_at", None) is None]
    if getattr(args, "refresh_all", False):
        pending = list(works)
    # Publisher-proportional order (`ADR-07`), so a crawl the quota stops
    # part-way is a representative sample rather than one publisher's block.
    by_doi = {w.doi: w for w in pending if w.doi}
    ordered = [by_doi[doi] for doi in interleave_by_registrant(sorted(by_doi))]
    ordered += [w for w in pending if not w.doi]
    selected = ordered[: args.max_works]
    batch_size = max(int(getattr(args, "batch_size", CITING_BATCH_SIZE) or 1), 1)
    max_pages = args.max_pages or CITING_BATCH_MAX_PAGES
    logger.info(
        "openalex citing crawl: %d works, %d already crawled, %d pending, crawling %d "
        "in batches of %d",
        len(works),
        len(works) - len(pending),
        len(pending),
        len(selected),
        batch_size,
    )

    edges = truncated = crawled = throttled = 0
    consecutive_throttles = 0
    stopped_early = None
    for start in range(0, len(selected), batch_size):
        chunk = selected[start : start + batch_size]
        results = fetch_openalex_citing_batch(
            [w.provider_work_id for w in chunk], max_pages=max_pages
        )
        chunk_throttled = any(r["throttled"] or r["error"] for r in results.values())
        if chunk_throttled:
            # A throttled or rejected query says nothing about these works.
            # Marking them crawled would make the resume skip them forever and
            # leave them recorded as "truncated" for no reason -- which is
            # what the per-work loop used to do to every throttled work.
            throttled += len(chunk)
            consecutive_throttles += 1
            error = next((r["error"] for r in results.values() if r["error"]), None)
            if error or consecutive_throttles >= CONSECUTIVE_FAILURE_LIMIT:
                stopped_early = "error" if error else "rate_limited"
                logger.warning(
                    "openalex citing crawl: stopping at %d/%d works (%s)",
                    start,
                    len(selected),
                    error or f"{consecutive_throttles} consecutive throttled batches",
                )
                break
            continue
        consecutive_throttles = 0
        for work in chunk:
            result = results[work.provider_work_id]
            edges += persist_incoming_edges(
                bronze_session, work.provider_work_id, result["citing_work_ids"], observed_at
            )
            work.citing_crawled_at = observed_at
            work.citing_truncated = bool(result["truncated"])
            truncated += int(result["truncated"])
            crawled += 1
        bronze_session.commit()
        logger.info("openalex citing crawl: %d/%d works, %d edges", crawled, len(selected), edges)
        if args.delay:
            time.sleep(args.delay)
    bronze_session.commit()
    return {
        "works_known": len(works),
        "already_crawled": len(works) - len(pending),
        "works_crawled": crawled,
        "remaining": max(len(pending) - crawled, 0),
        "throttled_works": throttled,
        "stopped_early": stopped_early,
        "edges_inserted": edges,
        "truncated_works": truncated,
        "observed_at": observed_at.isoformat(),
    }
