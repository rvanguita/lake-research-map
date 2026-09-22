"""OpenAlex REST API client for automated citation & reference count enrichment.

Provides free, public bibliographic enrichment without requiring an API key.
OpenAlex allows up to 10 requests/second without authentication (polite pool
when including an email in headers or query params). Responses are persisted
as append-only observations rather than rewriting a local cache file.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from datetime import UTC, datetime
from urllib.parse import quote

import requests
from sqlalchemy import select
from sqlalchemy.orm import Session

from lake_research_map.db.bronze_models import (
    AccessObservation,
    CitationEdge,
    CitationYearCount,
    EnrichmentObservation,
    ExternalWork,
)
from lake_research_map.ingest.enrichment import _normalize_doi

logger = logging.getLogger(__name__)

OPENALEX_BASE_URL = "https://api.openalex.org/works"
PROJECT_URL = "https://github.com/rvanguita/lake-research-map"
DEFAULT_USER_AGENT = f"lake-research-map/1.0 ({PROJECT_URL})"


def _user_agent(email: str | None) -> str:
    """Identify the client honestly, or not at all.

    The header used to carry a fixed `mailto:researcher@example.com`. An
    address nobody reads is worse than no address: it is what OpenAlex would
    contact about a misbehaving crawl, and it made every run look identically
    anonymous while claiming otherwise. The contact address now comes from the
    environment, and the header simply omits it when there is none.
    """
    contact = (email or os.environ.get("OPENALEX_EMAIL") or "").strip()
    if not contact:
        return DEFAULT_USER_AGENT
    return f"lake-research-map/1.0 ({PROJECT_URL}; mailto:{contact})"


# A crawl that has been told to stop must stop. The corpus refresh on
# 2026-09-22 collected 73 consecutive HTTP 429s while the loop kept going,
# spending four requests and seven seconds of backoff per DOI against an API
# that had already refused -- roughly 8,000 futile requests had it run to the
# end. Five, rather than one: an isolated 429 is noise, five in a row is a
# policy.
CONSECUTIVE_FAILURE_LIMIT = 5
# `Retry-After` is honoured but capped: a header is a hint from a service, not
# a licence to park the process for an hour.
MAX_RETRY_AFTER_SECONDS = 60.0
_THROTTLE_HEADERS = (
    "retry-after",
    "x-ratelimit-limit",
    "x-ratelimit-remaining",
    "x-ratelimit-reset",
)


def retry_after_seconds(response) -> float | None:
    """What the server asked us to wait, or None when it did not say."""
    try:
        raw = response.headers.get("Retry-After")
    except AttributeError:  # a fake response in a test may carry no headers
        return None
    if not raw:
        return None
    try:
        return max(float(str(raw).strip()), 0.0)
    except ValueError:
        # The header also allows an HTTP-date. Falling back to the guess beats
        # carrying a second parser for a value we only compare against a cap.
        return None


def retrying_is_futile(response) -> bool:
    """True when the server's own wait exceeds anything worth sleeping through.

    OpenAlex answered the exhausted quota with `Retry-After: 19587` -- 5.4
    hours. Sleeping the capped 60s and trying again three times just spends
    three minutes to be refused three more times. When the server names a wait
    that long, the honest move is to surface the refusal immediately and let
    the circuit breaker end the batch.
    """
    asked = retry_after_seconds(response)
    return asked is not None and asked > MAX_RETRY_AFTER_SECONDS


def backoff_seconds(response, retry_count: int) -> float:
    """How long to wait before retrying, preferring what the server said.

    The previous version always guessed `min(2**n, 8)` and discarded the
    `Retry-After` header in which OpenAlex states the answer exactly. Guessing
    short is what turns a brief throttle into a sustained one.
    """
    asked = retry_after_seconds(response)
    if asked is not None:
        return min(asked, MAX_RETRY_AFTER_SECONDS)
    return float(min(2**retry_count, 8))


def _log_throttle_headers(response) -> None:
    """Say what the server actually reported, once per process.

    Without this the only evidence of a block is a column of 429s, which
    cannot distinguish a burst limit from a daily quota from a `mailto` that
    never reached the polite pool -- and that distinction is what decides the
    delay to re-run with.
    """
    global _THROTTLE_LOGGED
    if _THROTTLE_LOGGED:
        return
    try:
        headers = {
            name: value
            for name, value in response.headers.items()
            if name.lower() in _THROTTLE_HEADERS
        }
    except AttributeError:
        return
    _THROTTLE_LOGGED = True
    logger.warning(
        "OpenAlex throttled this client (HTTP %s); headers: %s",
        getattr(response, "status_code", "?"),
        headers or "none returned",
    )


_THROTTLE_LOGGED = False


def fetch_openalex_work(
    doi: str,
    *,
    timeout: float = 8.0,
    email: str | None = None,
) -> dict[str, int | None] | None:
    """Fetch citation count and reference count from OpenAlex for a given DOI.

    Returns dict with 'citation_count' and 'reference_count', or None if not found/failed.
    """
    observation = fetch_openalex_observation(doi, timeout=timeout, email=email)
    if observation["status"] != "success":
        return None
    return {
        "citation_count": observation["citation_count"],
        "reference_count": observation["reference_count"],
    }


def fetch_openalex_observation(
    doi: str,
    *,
    timeout: float = 8.0,
    email: str | None = None,
    api_key: str | None = None,
    max_retries: int = 3,
) -> dict:
    """Fetch one DOI and retain response provenance instead of collapsing failures."""
    clean_doi = _normalize_doi(doi)
    if not clean_doi:
        return {
            "doi": "",
            "status": "invalid_doi",
            "http_status": None,
            "retry_count": 0,
            "error_message": "blank DOI after normalization",
        }

    url = f"{OPENALEX_BASE_URL}/https://doi.org/{quote(clean_doi, safe='')}"
    params = {}
    resolved_email = email or os.environ.get("OPENALEX_EMAIL")
    headers = {"User-Agent": _user_agent(resolved_email)}
    resolved_key = api_key or os.environ.get("OPENALEX_API_KEY")
    if resolved_email:
        params["mailto"] = resolved_email
    if resolved_key:
        params["api_key"] = resolved_key

    for retry_count in range(max_retries + 1):
        try:
            response = requests.get(url, headers=headers, params=params, timeout=timeout)
            if response.status_code == 404:
                return {
                    "doi": clean_doi,
                    "status": "not_found",
                    "http_status": 404,
                    "retry_count": retry_count,
                    "error_message": None,
                }
            if response.status_code == 429 or response.status_code >= 500:
                _log_throttle_headers(response)
                if retry_count < max_retries and not retrying_is_futile(response):
                    time.sleep(backoff_seconds(response, retry_count))
                    continue
                return {
                    "doi": clean_doi,
                    "status": "rate_limited" if response.status_code == 429 else "error",
                    "http_status": response.status_code,
                    "retry_count": retry_count,
                    "error_message": f"OpenAlex returned HTTP {response.status_code}",
                }
            response.raise_for_status()
            data = response.json()
            payload = json.dumps(data, sort_keys=True, separators=(",", ":"))
            referenced = data.get("referenced_works") or []
            cited_by = data.get("cited_by_count")
            return {
                "doi": clean_doi,
                "provider_work_id": data.get("id"),
                "status": "success",
                "http_status": response.status_code,
                "citation_count": int(cited_by) if cited_by is not None else None,
                "reference_count": len(referenced),
                "response_sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
                "retry_count": retry_count,
                "payload": data,
                "error_message": None,
            }
        except (requests.RequestException, ValueError) as exc:
            if retry_count < max_retries:
                time.sleep(min(2**retry_count, 8))
                continue
            logger.warning("OpenAlex request failed for %s: %s", clean_doi, exc)
            return {
                "doi": clean_doi,
                "status": "error",
                "http_status": getattr(getattr(exc, "response", None), "status_code", None),
                "retry_count": retry_count,
                "error_message": str(exc),
            }
    raise AssertionError("retry loop must return")


def refresh_openalex_observations(
    session: Session,
    dois: list[str],
    *,
    observed_at: datetime | None = None,
    max_fetch: int = 100,
    delay: float = 0.1,
    refresh_all: bool = False,
    commit_every: int = 25,
    store_payload: bool = False,
) -> dict[str, int]:
    """Append a reproducible observation batch to Bronze, resumably.

    Resumability is the point. The skip used to compare `observed_at` against
    *this run's* `batch_time`, which a previous run can never equal, while the
    work list was always `sorted(dois)[:max_fetch]` -- the same leading slice
    every time. A corpus of 3,115 DOIs at the default `max_fetch=100` therefore
    re-fetched the same first hundred on every invocation and never reached the
    hundred-and-first. Skipping DOIs that already carry a *successful*
    observation is what lets run N+1 continue where run N stopped.

    Failed and not-found observations are retried rather than skipped: a 429 or
    a timeout says nothing about the DOI, and OpenAlex does add records over
    time. `refresh_all=True` re-observes everything, which is the right mode for
    a deliberate as-of snapshot rather than gap-filling.

    Progress is committed every `commit_every` rows. The single commit at the
    end meant an interrupt at request 3,000 discarded all 3,000, which on a
    crawl this long is the likely outcome rather than the unlucky one.
    """
    batch_time = (observed_at or datetime.now(UTC)).replace(tzinfo=None)
    normalized = sorted({_normalize_doi(doi) for doi in dois if _normalize_doi(doi)})

    already: set[str] = set()
    if not refresh_all:
        already = set(
            session.scalars(
                select(EnrichmentObservation.doi).where(
                    EnrichmentObservation.provider == "openalex",
                    EnrichmentObservation.status == "success",
                )
            ).all()
        )
    pending = [doi for doi in normalized if doi not in already]
    batch = pending[:max_fetch]

    logger.info(
        "openalex refresh: %d DOIs, %d already observed, %d pending, fetching %d",
        len(normalized),
        len(already & set(normalized)),
        len(pending),
        len(batch),
    )

    inserted = success = 0
    consecutive_failures = 0
    stopped_early: str | None = None
    for position, doi in enumerate(batch, start=1):
        result = fetch_openalex_observation(doi)
        session.add(
            EnrichmentObservation(
                provider="openalex",
                doi=doi,
                provider_work_id=result.get("provider_work_id"),
                observed_at=batch_time,
                status=result["status"],
                http_status=result.get("http_status"),
                citation_count=result.get("citation_count"),
                reference_count=result.get("reference_count"),
                response_sha256=result.get("response_sha256"),
                retry_count=result.get("retry_count", 0),
                # The whole Work JSON is tens of kilobytes and nothing in the
                # project reads it back: `_persist_openalex_evidence` lifts the
                # annual counts, reference edges and access status into their
                # own tables, and `response_sha256` already proves what was
                # received. Storing it for 3,115 works would add hundreds of
                # megabytes to a MySQL server this project shares with
                # unrelated ones, so it is opt-in.
                payload=result.get("payload") if store_payload else None,
                error_message=result.get("error_message"),
            )
        )
        if result["status"] == "success":
            _persist_openalex_evidence(session, result, batch_time)
        inserted += 1
        success += int(result["status"] == "success")

        if result["status"] in {"rate_limited", "error"}:
            consecutive_failures += 1
        else:
            consecutive_failures = 0
        if consecutive_failures >= CONSECUTIVE_FAILURE_LIMIT:
            stopped_early = result["status"]
            logger.warning(
                "openalex refresh: stopping after %d consecutive %s responses at %d/%d; "
                "progress is committed and the next run resumes here",
                consecutive_failures,
                stopped_early,
                position,
                len(batch),
            )
            break

        if commit_every and position % commit_every == 0:
            session.commit()
            logger.info("openalex refresh: %d/%d fetched, %d ok", position, len(batch), success)
        if delay:
            time.sleep(delay)
    session.commit()
    return {
        "requested": len(normalized),
        "already_observed": len(already & set(normalized)),
        "pending": len(pending),
        "fetched": inserted,
        "inserted": inserted,
        "success": success,
        "stopped_early": stopped_early,
        "remaining": max(len(pending) - success, 0),
    }


def _persist_openalex_evidence(session: Session, result: dict, observed_at: datetime) -> None:
    """Persist longitudinal, graph, and access fields from one successful Work."""
    payload = result.get("payload") or {}
    work_id = result.get("provider_work_id")
    if not work_id:
        return
    work = session.scalar(
        select(ExternalWork).where(
            ExternalWork.provider == "openalex",
            ExternalWork.provider_work_id == work_id,
        )
    )
    if work is None:
        work = ExternalWork(
            provider="openalex",
            provider_work_id=work_id,
            doi=result["doi"],
            publication_year=payload.get("publication_year"),
            title=payload.get("display_name"),
            first_observed_at=observed_at,
            last_observed_at=observed_at,
        )
        session.add(work)
    else:
        work.last_observed_at = observed_at

    for value in payload.get("counts_by_year") or []:
        year = value.get("year")
        count = value.get("cited_by_count")
        if year is not None and count is not None:
            session.add(
                CitationYearCount(
                    provider="openalex",
                    provider_work_id=work_id,
                    year=int(year),
                    citation_count=int(count),
                    observed_at=observed_at,
                )
            )
    for cited_work_id in payload.get("referenced_works") or []:
        if cited_work_id and cited_work_id != work_id:
            session.add(
                CitationEdge(
                    provider="openalex",
                    citing_work_id=work_id,
                    cited_work_id=str(cited_work_id),
                    observed_at=observed_at,
                    discovered_via="referenced_works",
                )
            )
    access = payload.get("open_access") or {}
    location = payload.get("best_oa_location") or {}
    session.add(
        AccessObservation(
            provider="openalex",
            provider_work_id=work_id,
            observed_at=observed_at,
            is_oa=access.get("is_oa"),
            oa_status=access.get("oa_status"),
            license=location.get("license"),
            landing_page_url=location.get("landing_page_url"),
        )
    )


# ---------------------------------------------------------------------------
# WP-24: incoming citation edges and graph coverage
# ---------------------------------------------------------------------------

# OpenAlex caps a page at 200 and will keep paginating a highly-cited work for
# a long time. The cap exists so one popular paper cannot consume an entire
# refresh budget; a truncated crawl is recorded as truncated rather than
# silently treated as "no more citations".
CITING_PAGE_SIZE = 200
CITING_MAX_PAGES = 5
# Same retry budget as the backward pass, so one throttled page does not
# discard a work's whole forward set.
CITING_MAX_RETRIES = 3


def fetch_openalex_citing_works(
    work_id: str,
    *,
    session_factory=None,
    email: str | None = None,
    api_key: str | None = None,
    max_pages: int = CITING_MAX_PAGES,
) -> dict:
    """List the OpenAlex works that cite `work_id`, following `cites:` pages.

    Returns ``{"citing_work_ids": [...], "truncated": bool, "pages": int}``.
    `truncated` is the field that matters downstream: without it an incomplete
    crawl is indistinguishable from a work with few citations, and the
    disruption index that WP-24 exists to enable would be computed on a
    forward-citation set that is quietly missing its tail.

    Network-bound. `session_factory` is injected so the pagination logic can be
    tested without calling OpenAlex.
    """
    import requests

    get = session_factory or requests.get
    resolved_email = email or os.environ.get("OPENALEX_EMAIL")
    resolved_key = api_key or os.environ.get("OPENALEX_API_KEY")

    citing: list[str] = []
    cursor = "*"
    pages = 0
    truncated = False
    while pages < max_pages:
        params = {
            "filter": f"cites:{work_id}",
            "per-page": CITING_PAGE_SIZE,
            "cursor": cursor,
            "select": "id",
        }
        if resolved_email:
            params["mailto"] = resolved_email
        if resolved_key:
            params["api_key"] = resolved_key
        # The forward pass had no 429 handling at all -- a bare
        # `raise_for_status()` -- so the first throttled page killed a crawl of
        # a thousand works. It was never exercised against the live API, which
        # is exactly why that went unnoticed.
        response = None
        for retry_count in range(CITING_MAX_RETRIES + 1):
            response = get(
                OPENALEX_BASE_URL,
                params=params,
                headers={"User-Agent": _user_agent(resolved_email)},
                timeout=30,
            )
            status = getattr(response, "status_code", 200)
            if status == 429 or status >= 500:
                _log_throttle_headers(response)
                if retry_count < CITING_MAX_RETRIES and not retrying_is_futile(response):
                    time.sleep(backoff_seconds(response, retry_count))
                    continue
                # Out of retries: report the partial set as truncated rather
                # than as a complete crawl that found nothing more.
                return {
                    "citing_work_ids": list(dict.fromkeys(citing)),
                    "truncated": True,
                    "pages": pages,
                    "throttled": True,
                }
            break
        response.raise_for_status()
        payload = response.json()
        pages += 1
        for item in payload.get("results") or []:
            identifier = item.get("id")
            if identifier and identifier != work_id:
                citing.append(str(identifier))
        cursor = (payload.get("meta") or {}).get("next_cursor")
        if not cursor or not (payload.get("results") or []):
            break
    else:
        truncated = True

    # `while ... else` runs the else only when the loop was never broken out
    # of, which is exactly the "ran out of page budget" case.
    return {
        "citing_work_ids": list(dict.fromkeys(citing)),
        "truncated": truncated,
        "pages": pages,
        "throttled": False,
    }


def persist_incoming_edges(
    session: Session,
    work_id: str,
    citing_work_ids: list[str],
    observed_at: datetime,
) -> int:
    """Store `cites:` results as edges, tagged with how they were discovered.

    Direction needs no column: an incoming edge is one whose `cited_work_id`
    is ours. What does need recording is the provenance, because a crawl that
    stopped early and a work that is genuinely uncited look identical
    afterwards.
    """
    inserted = 0
    for citing in dict.fromkeys(citing_work_ids):
        if not citing or citing == work_id:
            continue
        session.add(
            CitationEdge(
                provider="openalex",
                citing_work_id=str(citing),
                cited_work_id=work_id,
                observed_at=observed_at,
                discovered_via="cites_query",
            )
        )
        inserted += 1
    return inserted


def citation_graph_coverage(session: Session, work_ids: list[str]) -> dict:
    """Report how much of the corpus has usable forward and backward edges.

    WP-24's gate says CD/disruption stays unavailable unless coverage is
    adequate, which cannot be evaluated without measuring it. Backward
    coverage is trustworthy when present because a reference list is complete
    per work; forward coverage is only trustworthy where the crawl was not
    truncated, so the two are reported separately rather than averaged into
    one reassuring number.
    """
    wanted = [str(w) for w in work_ids if w]
    if not wanted:
        return {
            "population": 0,
            "with_backward": 0,
            "with_forward": 0,
            "backward_coverage": 0.0,
            "forward_coverage": 0.0,
            "usable_for_disruption": 0,
        }

    rows = session.execute(
        select(CitationEdge.citing_work_id, CitationEdge.cited_work_id, CitationEdge.discovered_via)
    ).all()
    backward: set[str] = set()
    forward_edges: set[str] = set()
    for citing, cited, via in rows:
        if via == "referenced_works" and citing in wanted:
            backward.add(citing)
        elif via == "cites_query" and cited in wanted:
            forward_edges.add(cited)

    # Forward coverage is a property of the crawl, not of the edges it found.
    # Counting works that have an incoming edge silently excludes every work
    # nobody cites -- for which the correct answer, "zero citing works", is
    # known and usable. A truncated crawl is the opposite case: it produced
    # edges but its tail is missing, so it is excluded from the usable set.
    crawled: set[str] = set()
    truncated: set[str] = set()
    try:
        for work_id, crawled_at, was_truncated in session.execute(
            select(
                ExternalWork.provider_work_id,
                ExternalWork.citing_crawled_at,
                ExternalWork.citing_truncated,
            ).where(ExternalWork.provider == "openalex")
        ).all():
            if work_id in wanted and crawled_at is not None:
                crawled.add(work_id)
                if was_truncated:
                    truncated.add(work_id)
    except Exception:  # pragma: no cover - pre-migration database
        # Before the additive migration the columns do not exist; fall back to
        # edge presence so the audit degrades instead of failing.
        logger.warning("forward-crawl columns unavailable; falling back to edge presence")
        crawled = set(forward_edges)

    forward = crawled or forward_edges
    population = len(set(wanted))
    both = (backward & forward) - truncated
    return {
        "population": population,
        "with_backward": len(backward),
        "with_forward": len(forward),
        "with_forward_edges": len(forward_edges),
        "truncated": len(truncated),
        "backward_coverage": len(backward) / population,
        "forward_coverage": len(forward) / population,
        # The disruption index needs both directions for the same work, so the
        # usable population is the intersection, never the larger of the two,
        # and never a work whose forward tail was cut off.
        "usable_for_disruption": len(both),
        "disruption_coverage": len(both) / population,
    }
