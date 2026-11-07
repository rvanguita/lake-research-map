"""OpenAlex REST API client for automated citation & reference count enrichment.

Provides free, public bibliographic enrichment without requiring an API key.
OpenAlex allows up to 10 requests/second without authentication (polite pool
when including an email in headers or query params). Responses are persisted
as append-only observations rather than rewriting a local cache file.

This module holds the HTTP client (identity, throttling, retry/backoff), the
batched DOI refresh, the forward `cites:` crawl, reference-year resolution and
the persistence of what they return. The audits that read that evidence back --
coverage, graph integrity, access validation, `contradicted_zeros` -- are in
`ingest/openalex_audits.py`.
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

from db.bronze_models import (
    AccessObservation,
    CitationEdge,
    CitationYearCount,
    EnrichmentObservation,
    ExternalWork,
)
from ingest.doi import normalize_doi

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
    clean_doi = normalize_doi(doi)
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


# OpenAlex OR-joins up to 50 values in one filter, so a DOI population can be
# fetched in ceil(n/50) requests instead of n. That is not a micro-optimisation
# here: the quota is 1,000 requests per window, so one-request-per-DOI made the
# 3,115-DOI corpus a four-window, ~22-hour job, while batching makes it 63
# requests. It is also the polite way to ask -- the same data for a sixtieth of
# the load.
OPENALEX_FILTER_BATCH = 50


def fetch_openalex_batch(
    dois: list[str],
    *,
    timeout: float = 30.0,
    email: str | None = None,
    api_key: str | None = None,
    max_retries: int = 3,
    session_factory=None,
) -> dict[str, dict]:
    """Fetch a batch of DOIs in one request, keyed by normalized DOI.

    Returns the same per-DOI result shape as `fetch_openalex_observation`, so
    the persistence path does not care which one produced it. A DOI the
    response does not carry is reported `not_found`: OpenAlex simply omits
    unknown works from a filtered result, and treating an omission as an error
    would retry it forever.
    """
    get = session_factory or requests.get
    clean = [doi for doi in (normalize_doi(value) for value in dois) if doi]
    if not clean:
        return {}

    resolved_email = email or os.environ.get("OPENALEX_EMAIL")
    resolved_key = api_key or os.environ.get("OPENALEX_API_KEY")
    params = {
        "filter": "doi:" + "|".join(f"https://doi.org/{doi}" for doi in clean),
        "per-page": len(clean),
    }
    if resolved_email:
        params["mailto"] = resolved_email
    if resolved_key:
        params["api_key"] = resolved_key

    def _failure(status: str, http_status, retry_count: int, message: str) -> dict[str, dict]:
        return {
            doi: {
                "doi": doi,
                "status": status,
                "http_status": http_status,
                "retry_count": retry_count,
                "error_message": message,
            }
            for doi in clean
        }

    for retry_count in range(max_retries + 1):
        try:
            response = get(
                OPENALEX_BASE_URL,
                params=params,
                headers={"User-Agent": _user_agent(resolved_email)},
                timeout=timeout,
            )
            status_code = getattr(response, "status_code", 200)
            if status_code == 429 or status_code >= 500:
                _log_throttle_headers(response)
                if retry_count < max_retries and not retrying_is_futile(response):
                    time.sleep(backoff_seconds(response, retry_count))
                    continue
                # The whole batch shares one verdict, which is what lets the
                # circuit breaker see a throttle as a throttle rather than as
                # fifty unrelated failures.
                failed = _failure(
                    "rate_limited" if status_code == 429 else "error",
                    status_code,
                    retry_count,
                    f"OpenAlex returned HTTP {status_code}",
                )
                # A wait longer than the cap is an exhausted quota: no retry
                # inside this run can succeed, so the caller stops at once.
                if retrying_is_futile(response):
                    for value in failed.values():
                        value["futile"] = True
                return failed
            response.raise_for_status()
            payload = response.json()
        except (requests.RequestException, ValueError) as exc:
            if retry_count < max_retries:
                time.sleep(min(2**retry_count, 8))
                continue
            logger.warning("OpenAlex batch request failed: %s", exc)
            return _failure("error", None, retry_count, str(exc))

        results: dict[str, dict] = {}
        for work in payload.get("results") or []:
            doi = normalize_doi(work.get("doi"))
            if not doi:
                continue
            body = json.dumps(work, sort_keys=True, separators=(",", ":"))
            referenced = work.get("referenced_works") or []
            cited_by = work.get("cited_by_count")
            results[doi] = {
                "doi": doi,
                "provider_work_id": work.get("id"),
                "status": "success",
                "http_status": status_code,
                "citation_count": int(cited_by) if cited_by is not None else None,
                "reference_count": len(referenced),
                "response_sha256": hashlib.sha256(body.encode("utf-8")).hexdigest(),
                "retry_count": retry_count,
                "payload": work,
                "error_message": None,
            }
        for doi in clean:
            results.setdefault(
                doi,
                {
                    "doi": doi,
                    "status": "not_found",
                    "http_status": status_code,
                    "retry_count": retry_count,
                    "error_message": None,
                },
            )
        return results
    raise AssertionError("retry loop must return")


# A DOI's registrant prefix is its publisher, so walking DOIs in sorted order
# walks publishers in blocks. That looks like neutral iteration and is not: the
# first live crawl stopped at its quota having observed 989 Elsevier works
# (10.1016) and **zero** IEEE ones (10.1109, 1,340 articles), because `10.1016`
# sorts entirely before `10.1109`. Every coverage figure measured on that
# subset described one publisher while reading as though it described the
# corpus.
REGISTRANT_LABELS = {
    "10.1016": "Elsevier",
    "10.1109": "IEEE",
    "10.1049": "IET",
    "10.1002": "Wiley",
    "10.3390": "MDPI",
    "10.1007": "Springer",
}


def registrant_prefix(doi: str) -> str:
    """The DOI registrant, which identifies the publisher."""
    return str(doi).split("/", 1)[0]


def registrant_label(doi_or_prefix: str) -> str:
    """A human name for a registrant, falling back to the prefix itself."""
    prefix = registrant_prefix(doi_or_prefix)
    return REGISTRANT_LABELS.get(prefix, prefix)


def interleave_by_registrant(dois: list[str]) -> list[str]:
    """Order DOIs so that *any* prefix of the result is proportional by publisher.

    Each DOI is positioned at `(index + 0.5) / group_size` within its own
    registrant, and the whole list is sorted by that fraction. Stopping after
    the first N therefore yields roughly N * (group_size / total) from every
    group instead of exhausting one publisher before reaching the next -- which
    is what makes a partial crawl a usable sample rather than a biased one.

    Deterministic, with no random seed: the same corpus always produces the
    same order, which is the reproducibility `NFR-01` requires of anything a
    published figure rests on.
    """
    groups: dict[str, list[str]] = {}
    for doi in dois:
        groups.setdefault(registrant_prefix(doi), []).append(doi)

    positioned: list[tuple[float, str, str]] = []
    for prefix, items in groups.items():
        size = len(items)
        for index, doi in enumerate(sorted(items)):
            positioned.append(((index + 0.5) / size, prefix, doi))
    # The prefix and DOI join the sort key only to break ties deterministically
    # when two groups land on the same fraction.
    positioned.sort()
    return [doi for _, _, doi in positioned]


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
    batch_size: int = OPENALEX_FILTER_BATCH,
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
    normalized = interleave_by_registrant(
        sorted({normalize_doi(doi) for doi in dois if normalize_doi(doi)})
    )

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

    # One request per `OPENALEX_FILTER_BATCH` DOIs rather than per DOI. The
    # loop below still walks DOIs one at a time so the breaker, the commit
    # cadence and the stats keep counting subjects, not requests.
    def _resolve(chunk: list[str]) -> dict[str, dict]:
        """One request for the whole chunk, or the per-DOI path when batching is off."""
        if batch_size > 1:
            return fetch_openalex_batch(chunk)
        return {doi: fetch_openalex_observation(doi) for doi in chunk}

    position = 0
    for chunk_start in range(0, len(batch), max(batch_size, 1)):
        chunk = batch[chunk_start : chunk_start + max(batch_size, 1)]
        resolved = _resolve(chunk)
        if delay:
            time.sleep(delay)

        for doi in chunk:
            position += 1
            result = resolved[doi]
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
                    # The whole Work JSON is tens of kilobytes and nothing in
                    # the project reads it back: `_persist_openalex_evidence`
                    # lifts the annual counts, reference edges and access
                    # status into their own tables, and `response_sha256`
                    # already proves what was received. Storing it for 3,115
                    # works would add hundreds of megabytes to a MySQL server
                    # this project shares with unrelated ones, so it is opt-in.
                    payload=result.get("payload") if store_payload else None,
                    error_message=result.get("error_message"),
                )
            )
            if result["status"] == "success":
                _persist_openalex_evidence(session, result, batch_time)
            inserted += 1
            success += int(result["status"] == "success")

            if commit_every and position % commit_every == 0:
                session.commit()
                logger.info("openalex refresh: %d/%d fetched, %d ok", position, len(batch), success)

        # The breaker counts failed *requests*. Counting DOIs made one failed
        # 50-DOI batch look like fifty consecutive failures and stopped the
        # whole crawl on a single transient 500. With batching off, a request
        # is one DOI and the behaviour is unchanged.
        outcomes = [resolved[doi] for doi in chunk]
        if outcomes and all(o["status"] in {"rate_limited", "error"} for o in outcomes):
            consecutive_failures += 1
            futile = any(o.get("futile") for o in outcomes)
            if futile or consecutive_failures >= CONSECUTIVE_FAILURE_LIMIT:
                stopped_early = outcomes[-1]["status"]
                logger.warning(
                    "openalex refresh: stopping at %d/%d (%s); progress is committed and "
                    "the next run resumes here",
                    position,
                    len(batch),
                    "quota exhausted" if futile else f"{consecutive_failures} failed requests",
                )
        else:
            consecutive_failures = 0
        if stopped_early:
            break

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


# ---------------------------------------------------------------------------
# Batched graph collection: incoming edges and reference years
# ---------------------------------------------------------------------------

# One `cites:` query per work cost 2,254 requests for the 2,215 works still
# uncrawled after the first window -- more than two quota windows. OR-joining
# 50 works into one filter and paging the union costs 233. Each citing work in
# the union is attributed back to the batch members it cites through its own
# `referenced_works`, so nothing is lost by asking once.
CITING_BATCH_SIZE = 50
# Budget per batch, not per work. A union of 50 works averages ~950 citing
# works in this corpus, i.e. five pages; 50 pages leave room for the rare batch
# holding several heavily cited papers.
CITING_BATCH_MAX_PAGES = 50
REFERENCE_BATCH_SIZE = 100
# The width proven against the live API for DOI filters; used if the
# provider rejects the wider one.
REFERENCE_BATCH_FALLBACK = 50


def short_work_id(work_id: str) -> str:
    """`https://openalex.org/W123` -> `W123`, the form OpenAlex filters accept."""
    return str(work_id).rstrip("/").rsplit("/", 1)[-1]


def hash_order(values: list[str]) -> list[str]:
    """A deterministic order uncorrelated with anything the values encode.

    OpenAlex work ids are assigned over time, so sorting them orders by
    ingestion era, which tracks publication year -- exactly the property the
    reference-year resolution exists to measure. Resolving in that order and
    stopping at the quota would reproduce `ADR-07`'s bias in a new dimension.
    Ordering by a hash of the id is stable across runs and neutral with
    respect to year, publisher and everything else.
    """
    return sorted(values, key=lambda value: hashlib.sha256(str(value).encode("utf-8")).hexdigest())


def _paged_filter(
    params: dict,
    *,
    get,
    email: str | None,
    max_pages: int,
    timeout: float = 30.0,
) -> dict:
    """Page a filtered `/works` query, with the throttle policy of the single-DOI path.

    Returns ``{"results", "count", "pages", "throttled", "error"}``. ``count`` is
    OpenAlex's own total for the filter, which is what decides truncation
    precisely: a crawl is complete when it holds ``count`` results, not when it
    happened to stop before its page budget.
    """
    results: list[dict] = []
    cursor = "*"
    pages = 0
    total: int | None = None
    headers = {"User-Agent": _user_agent(email)}
    while pages < max_pages and cursor:
        page_params = dict(params, cursor=cursor)
        response = None
        for retry_count in range(CITING_MAX_RETRIES + 1):
            try:
                response = get(
                    OPENALEX_BASE_URL, params=page_params, headers=headers, timeout=timeout
                )
            except (requests.RequestException, ValueError) as exc:
                if retry_count < CITING_MAX_RETRIES:
                    time.sleep(min(2**retry_count, 8))
                    continue
                return {
                    "results": results,
                    "count": total,
                    "pages": pages,
                    "throttled": False,
                    "error": str(exc),
                }
            status = getattr(response, "status_code", 200)
            if status == 429 or status >= 500:
                _log_throttle_headers(response)
                if retry_count < CITING_MAX_RETRIES and not retrying_is_futile(response):
                    time.sleep(backoff_seconds(response, retry_count))
                    continue
                return {
                    "results": results,
                    "count": total,
                    "pages": pages,
                    "throttled": True,
                    "error": None,
                }
            break
        if getattr(response, "status_code", 200) >= 400:
            # A 4xx is not a throttle: most likely the filter itself was
            # rejected. Reported, never retried, so a malformed query fails
            # loudly in one request instead of quietly per work.
            return {
                "results": results,
                "count": total,
                "pages": pages,
                "throttled": False,
                "error": f"OpenAlex returned HTTP {response.status_code}",
            }
        payload = response.json()
        pages += 1
        meta = payload.get("meta") or {}
        if total is None and meta.get("count") is not None:
            total = int(meta["count"])
        batch = payload.get("results") or []
        results.extend(batch)
        cursor = meta.get("next_cursor") if batch else None
    return {"results": results, "count": total, "pages": pages, "throttled": False, "error": None}


def fetch_openalex_citing_batch(
    work_ids: list[str],
    *,
    session_factory=None,
    email: str | None = None,
    api_key: str | None = None,
    max_pages: int = CITING_BATCH_MAX_PAGES,
) -> dict[str, dict]:
    """Incoming edges for up to `CITING_BATCH_SIZE` works in one paged query.

    Returns ``{work_id: {"citing_work_ids", "truncated", "throttled", "error"}}``
    keyed by the ids exactly as passed. Truncation is decided by OpenAlex's own
    result count, and applies to the whole batch: when the union was cut off
    there is no way to tell which member lost edges, so every member is
    flagged rather than any being certified complete.
    """
    get = session_factory or requests.get
    resolved_email = email or os.environ.get("OPENALEX_EMAIL")
    resolved_key = api_key or os.environ.get("OPENALEX_API_KEY")
    by_short = {short_work_id(work_id): work_id for work_id in work_ids}
    if not by_short:
        return {}
    params = {
        "filter": "cites:" + "|".join(sorted(by_short)),
        "per-page": CITING_PAGE_SIZE,
        "select": "id,referenced_works",
    }
    if resolved_email:
        params["mailto"] = resolved_email
    if resolved_key:
        params["api_key"] = resolved_key

    fetched = _paged_filter(params, get=get, email=resolved_email, max_pages=max_pages)
    citing: dict[str, list[str]] = {work_id: [] for work_id in work_ids}
    for item in fetched["results"]:
        citing_id = item.get("id")
        if not citing_id:
            continue
        for reference in item.get("referenced_works") or []:
            target = by_short.get(short_work_id(reference))
            if target is not None and short_work_id(citing_id) != short_work_id(target):
                citing[target].append(str(citing_id))

    incomplete = (
        fetched["throttled"]
        or fetched["error"] is not None
        or (fetched["count"] is not None and len(fetched["results"]) < fetched["count"])
        or (fetched["count"] is None and fetched["pages"] >= max_pages)
    )
    return {
        work_id: {
            "citing_work_ids": list(dict.fromkeys(citing[work_id])),
            "truncated": bool(incomplete),
            "throttled": bool(fetched["throttled"]),
            "error": fetched["error"],
        }
        for work_id in work_ids
    }


def resolve_reference_years(
    session: Session,
    *,
    max_fetch: int = 60_000,
    batch_size: int = REFERENCE_BATCH_SIZE,
    delay: float = 0.25,
    session_factory=None,
    email: str | None = None,
) -> dict:
    """Record the publication year of every work the corpus cites.

    Candidates are the distinct `cited_work_id`s of `referenced_works` edges
    that are neither corpus works (whose year `lit_external_works` already
    holds) nor already resolved. Resumable, hash-ordered (`hash_order`), and
    stopped by the same five-in-a-row breaker as the other passes.
    """
    from db.bronze_models import ReferenceWork

    get = session_factory or requests.get
    resolved_email = email or os.environ.get("OPENALEX_EMAIL")
    observed_at = datetime.now(UTC).replace(tzinfo=None)

    cited = set(
        session.scalars(
            select(CitationEdge.cited_work_id)
            .where(CitationEdge.discovered_via == "referenced_works")
            .distinct()
        ).all()
    )
    in_corpus = set(session.scalars(select(ExternalWork.provider_work_id)).all())
    done = set(session.scalars(select(ReferenceWork.provider_work_id)).all())
    pending = hash_order(sorted(cited - in_corpus - done))
    batch = pending[:max_fetch]
    logger.info(
        "reference years: %d cited works, %d in corpus, %d resolved, %d pending, resolving %d",
        len(cited),
        len(cited & in_corpus),
        len(done),
        len(pending),
        len(batch),
    )

    resolved = not_found = consecutive_failures = requests_made = 0
    stopped_early = None
    size = max(batch_size, 1)
    start = 0
    while start < len(batch):
        chunk = batch[start : start + size]
        by_short = {short_work_id(work_id): work_id for work_id in chunk}
        params = {
            "filter": "ids.openalex:" + "|".join(sorted(by_short)),
            "per-page": len(by_short),
            "select": "id,publication_year",
        }
        if resolved_email:
            params["mailto"] = resolved_email
        fetched = _paged_filter(params, get=get, email=resolved_email, max_pages=1)
        requests_made += max(fetched["pages"], 1)
        if fetched["error"] and size > REFERENCE_BATCH_FALLBACK:
            # OpenAlex documents OR filters of up to 100 values, and 100 halves
            # the requests (564 instead of 1,130 for this corpus) -- enough to
            # fit the whole resolution into one quota window. If the provider
            # rejects the wider filter, fall back to the width already proven
            # in production and retry the same chunk, instead of stopping.
            logger.warning(
                "reference years: %d-value filter rejected (%s); falling back to %d",
                size,
                fetched["error"],
                REFERENCE_BATCH_FALLBACK,
            )
            size = REFERENCE_BATCH_FALLBACK
            continue
        if fetched["throttled"] or fetched["error"]:
            consecutive_failures += 1
            if fetched["error"]:
                logger.warning("reference years: batch rejected: %s", fetched["error"])
            if consecutive_failures >= CONSECUTIVE_FAILURE_LIMIT or fetched["error"]:
                stopped_early = "rate_limited" if fetched["throttled"] else "error"
                break
            start += len(chunk)
            continue
        consecutive_failures = 0
        years = {
            short_work_id(item.get("id")): item.get("publication_year")
            for item in fetched["results"]
            if item.get("id")
        }
        for short, work_id in by_short.items():
            found = short in years
            session.add(
                ReferenceWork(
                    provider="openalex",
                    provider_work_id=work_id,
                    publication_year=years.get(short),
                    status="success" if found else "not_found",
                    observed_at=observed_at,
                )
            )
            resolved += int(found)
            not_found += int(not found)
        session.commit()
        start += len(chunk)
        if delay:
            time.sleep(delay)
    session.commit()
    done_now = resolved + not_found
    return {
        "cited_works": len(cited),
        "in_corpus": len(cited & in_corpus),
        "pending": len(pending),
        "resolved": resolved,
        "not_found": not_found,
        "requests": requests_made,
        "batch_size": size,
        "stopped_early": stopped_early,
        "remaining": max(len(pending) - done_now, 0),
    }
