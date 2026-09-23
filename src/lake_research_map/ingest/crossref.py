"""Crossref as a second, quota-free source of cited-reference years (WP-23).

Price's index needs the publication year of every reference a corpus work
cites. The OpenAlex route to those years runs through a quota of 1,000
requests per window that the forward citation crawl also needs, and 56,330
cited works had never been dated. Crossref has no per-window quota, and a
publisher's own reference deposit carries most years inline -- 88% of
Elsevier's and 45% of IEEE's in a sample -- with a DOI for most of the rest.

Two passes, both batched, resumable and stopped by the same five-in-a-row
circuit breaker as the OpenAlex client:

1. **Reference lists** for corpus DOIs, 20 per request.
2. **Year lookups** for references deposited with a DOI but no year, 50 per
   request.

Nothing here merges the two providers reference by reference. Each work uses
one list -- Crossref when deposited, OpenAlex otherwise -- because Price's
index is a property of a citing work's own list, and splicing two lists that
enumerate the same references differently would count some twice.
"""

from __future__ import annotations

import logging
import os
import re
import time
from datetime import UTC, datetime

import requests
from sqlalchemy import select
from sqlalchemy.orm import Session

from lake_research_map.db.bronze_models import (
    CitationEdge,
    CrossrefReference,
    CrossrefReferenceList,
    DoiYear,
    EnrichmentObservation,
    ExternalWork,
    ReferenceWork,
)
from lake_research_map.ingest.enrichment import _normalize_doi
from lake_research_map.ingest.openalex import (
    CONSECUTIVE_FAILURE_LIMIT,
    PROJECT_URL,
    _log_throttle_headers,
    backoff_seconds,
    hash_order,
    interleave_by_registrant,
    registrant_prefix,
    retrying_is_futile,
)

logger = logging.getLogger(__name__)

CROSSREF_URL = "https://api.crossref.org/works"
# Reference arrays make list responses large, so lists travel 20 at a time;
# year lookups carry one date each and travel 50 at a time.
LIST_BATCH = 20
YEAR_BATCH = 50
MAX_RETRIES = 3
_YEAR = re.compile(r"(1[6-9]\d\d|20\d\d|2100)")


def _email(email: str | None) -> str:
    # A separate variable from OPENALEX_EMAIL on purpose: consent to send a
    # contact address is per service, and reusing one would extend it silently.
    return (email or os.environ.get("CROSSREF_EMAIL") or "").strip()


def _user_agent(email: str) -> str:
    base = f"lake-research-map/1.0 ({PROJECT_URL}"
    return f"{base}; mailto:{email})" if email else f"{base})"


def parse_reference_year(value: object) -> int | None:
    """The year a deposit wrote, or None.

    Deposits carry free text: `2019`, `2019a` for a second work that year,
    `in press`. Only a plausible four-digit year counts.
    """
    if value is None:
        return None
    match = _YEAR.search(str(value))
    return int(match.group(1)) if match else None


def issued_year(item: dict) -> int | None:
    parts = ((item.get("issued") or {}).get("date-parts") or [[None]])[0]
    return int(parts[0]) if parts and parts[0] else None


def _request(params: dict, *, get, email: str, timeout: float = 60.0) -> dict:
    """One filtered query with the OpenAlex client's throttle policy."""
    headers = {"User-Agent": _user_agent(email)}
    if email:
        params = dict(params, mailto=email)
    for retry_count in range(MAX_RETRIES + 1):
        try:
            response = get(CROSSREF_URL, params=params, headers=headers, timeout=timeout)
        except (requests.RequestException, ValueError) as exc:
            if retry_count < MAX_RETRIES:
                time.sleep(min(2**retry_count, 8))
                continue
            return {"items": [], "status": "error", "error": str(exc)}
        status = getattr(response, "status_code", 200)
        if status == 429 or status >= 500:
            _log_throttle_headers(response)
            if retry_count < MAX_RETRIES and not retrying_is_futile(response):
                time.sleep(backoff_seconds(response, retry_count))
                continue
            return {"items": [], "status": "throttled", "error": None}
        if status >= 400:
            # A rejected query is not retried: it would fail the same way.
            return {"items": [], "status": "error", "error": f"Crossref returned HTTP {status}"}
        return {"items": (response.json().get("message") or {}).get("items") or [], "status": "ok"}
    raise AssertionError("retry loop must return")


def _doi_filter(dois: list[str]) -> str:
    return ",".join(f"doi:{doi}" for doi in dois)


def _queryable(doi: str) -> bool:
    # A comma inside a DOI would split the OR filter; such DOIs are skipped
    # and counted rather than silently mangled.
    return bool(doi) and "," not in doi


def fetch_crossref_reference_lists(
    dois: list[str], *, session_factory=None, email: str | None = None
) -> dict:
    """Reference lists for a batch of DOIs, keyed by normalized DOI."""
    get = session_factory or requests.get
    clean = [
        doi for doi in dict.fromkeys(_normalize_doi(d) for d in dois) if doi and _queryable(doi)
    ]
    if not clean:
        return {"status": "ok", "lists": {}}
    fetched = _request(
        {"filter": _doi_filter(clean), "rows": len(clean), "select": "DOI,reference,issued"},
        get=get,
        email=_email(email),
    )
    if fetched["status"] != "ok":
        return {"status": fetched["status"], "error": fetched.get("error"), "lists": {}}
    lists: dict[str, list[dict] | None] = {doi: None for doi in clean}
    for item in fetched["items"]:
        doi = _normalize_doi(item.get("DOI"))
        if doi in lists:
            lists[doi] = [
                {
                    "doi": _normalize_doi(reference.get("DOI")),
                    "year": parse_reference_year(reference.get("year")),
                }
                for reference in item.get("reference") or []
            ]
    return {"status": "ok", "lists": lists}


def fetch_crossref_years(
    dois: list[str], *, session_factory=None, email: str | None = None
) -> dict:
    """Publication years for a batch of DOIs; a DOI Crossref lacks maps to None."""
    get = session_factory or requests.get
    clean = [
        doi for doi in dict.fromkeys(_normalize_doi(d) for d in dois) if doi and _queryable(doi)
    ]
    if not clean:
        return {"status": "ok", "years": {}}
    fetched = _request(
        {"filter": _doi_filter(clean), "rows": len(clean), "select": "DOI,issued"},
        get=get,
        email=_email(email),
    )
    if fetched["status"] != "ok":
        return {"status": fetched["status"], "error": fetched.get("error"), "years": {}}
    years: dict[str, int | None] = {doi: None for doi in clean}
    found: set[str] = set()
    for item in fetched["items"]:
        doi = _normalize_doi(item.get("DOI"))
        if doi in years:
            years[doi] = issued_year(item)
            found.add(doi)
    return {"status": "ok", "years": years, "found": found}


def collect_crossref_references(
    session: Session,
    corpus_dois: list[str],
    *,
    delay: float = 0.1,
    list_batch: int = LIST_BATCH,
    year_batch: int = YEAR_BATCH,
    session_factory=None,
    email: str | None = None,
    observed_at: datetime | None = None,
) -> dict:
    """Collect deposited reference lists, then date the references that need it.

    Resumable in both passes: a citing DOI with any recorded list observation
    is not re-fetched, and a reference DOI with any recorded year lookup is not
    looked up again. Throttled batches are never recorded, so they are retried.
    """
    at = (observed_at or datetime.now(UTC)).replace(tzinfo=None)
    corpus = sorted({d for d in (_normalize_doi(value) for value in corpus_dois) if d})
    corpus_set = set(corpus)

    seen = set(session.scalars(select(CrossrefReferenceList.doi)).all())
    pending = interleave_by_registrant([doi for doi in corpus if doi not in seen])
    stats = {
        "corpus": len(corpus),
        "lists_already": len(corpus_set & seen),
        "lists_fetched": 0,
        "lists_deposited": 0,
        "lists_not_found": 0,
        "references_stored": 0,
        "years_requested": 0,
        "years_found": 0,
        "requests": 0,
        "stopped_early": None,
    }
    logger.info("crossref lists: %d corpus DOIs, %d pending", len(corpus), len(pending))

    failures = 0
    for start in range(0, len(pending), max(list_batch, 1)):
        chunk = pending[start : start + max(list_batch, 1)]
        result = fetch_crossref_reference_lists(chunk, session_factory=session_factory, email=email)
        stats["requests"] += 1
        if result["status"] != "ok":
            failures += 1
            if result["status"] == "error" or failures >= CONSECUTIVE_FAILURE_LIMIT:
                stats["stopped_early"] = result["status"] if result["status"] != "ok" else None
                logger.warning("crossref lists: stopping (%s)", result.get("error") or "throttled")
                break
            continue
        failures = 0
        for doi, references in result["lists"].items():
            status = "success" if references is not None else "not_found"
            session.add(
                CrossrefReferenceList(
                    doi=doi, observed_at=at, status=status, deposited=len(references or [])
                )
            )
            stats["lists_fetched"] += 1
            stats["lists_not_found"] += int(references is None)
            stats["lists_deposited"] += int(bool(references))
            for position, reference in enumerate(references or []):
                session.add(
                    CrossrefReference(
                        citing_doi=doi,
                        position=position,
                        reference_doi=reference["doi"],
                        year=reference["year"],
                        observed_at=at,
                    )
                )
                stats["references_stored"] += 1
        session.commit()
        if stats["lists_fetched"] % 500 < max(list_batch, 1):
            logger.info("crossref lists: %d fetched", stats["lists_fetched"])
        if delay:
            time.sleep(delay)

    if stats["stopped_early"]:
        return stats

    # Year lookups: references deposited with a DOI but without a year, that
    # are not corpus works (whose year is already known) and not yet looked up.
    undated = set(
        session.scalars(
            select(CrossrefReference.reference_doi)
            .where(CrossrefReference.year.is_(None), CrossrefReference.reference_doi.is_not(None))
            .distinct()
        ).all()
    )
    looked_up = set(
        session.scalars(select(DoiYear.doi).where(DoiYear.provider == "crossref")).all()
    )
    queue = hash_order(sorted(doi for doi in undated - corpus_set - looked_up if _queryable(doi)))
    stats["years_requested"] = len(queue)
    stats["unqueryable_dois"] = sum(1 for doi in undated if not _queryable(doi))
    logger.info("crossref years: %d reference DOIs to date", len(queue))

    failures = 0
    for start in range(0, len(queue), max(year_batch, 1)):
        chunk = queue[start : start + max(year_batch, 1)]
        result = fetch_crossref_years(chunk, session_factory=session_factory, email=email)
        stats["requests"] += 1
        if result["status"] != "ok":
            failures += 1
            if result["status"] == "error" or failures >= CONSECUTIVE_FAILURE_LIMIT:
                stats["stopped_early"] = result["status"]
                logger.warning("crossref years: stopping (%s)", result.get("error") or "throttled")
                break
            continue
        failures = 0
        for doi, year in result["years"].items():
            found = doi in result["found"]
            session.add(
                DoiYear(
                    provider="crossref",
                    doi=doi,
                    year=year,
                    status="success" if found else "not_found",
                    observed_at=at,
                )
            )
            stats["years_found"] += int(found and year is not None)
        session.commit()
        if delay:
            time.sleep(delay)
    return stats


def reference_year_coverage(session: Session, corpus_years: dict[str, int | None]) -> dict:
    """Cited-reference years over one reference list per corpus work.

    Each work uses its Crossref deposit when one exists, and its OpenAlex
    `referenced_works` otherwise. A work both providers report as citing
    nothing has a known empty list and leaves the denominator; a work with no
    list from either is counted separately as unenumerated, because its
    references cannot be counted at all -- neither as dated nor as missing.
    """
    corpus = {
        _normalize_doi(doi): year for doi, year in corpus_years.items() if _normalize_doi(doi)
    }

    # Latest Crossref list per DOI.
    latest: dict[str, tuple[datetime, str, int]] = {}
    for doi, observed_at, status, deposited in session.execute(
        select(
            CrossrefReferenceList.doi,
            CrossrefReferenceList.observed_at,
            CrossrefReferenceList.status,
            CrossrefReferenceList.deposited,
        )
    ).all():
        if doi not in latest or observed_at > latest[doi][0]:
            latest[doi] = (observed_at, status, deposited)
    crossref_refs: dict[str, list[tuple[str | None, int | None]]] = {}
    for citing, observed_at, ref_doi, year in session.execute(
        select(
            CrossrefReference.citing_doi,
            CrossrefReference.observed_at,
            CrossrefReference.reference_doi,
            CrossrefReference.year,
        )
    ).all():
        if citing in latest and observed_at == latest[citing][0]:
            crossref_refs.setdefault(citing, []).append((ref_doi, year))
    looked_up = {
        doi: year
        for doi, year, status in session.execute(
            select(DoiYear.doi, DoiYear.year, DoiYear.status).where(DoiYear.provider == "crossref")
        ).all()
        if status == "success" and year is not None
    }

    # OpenAlex side: DOI -> work id, the work's references, and years.
    work_of: dict[str, str] = {}
    year_of: dict[str, int] = {}
    for work_id, doi, year in session.execute(
        select(ExternalWork.provider_work_id, ExternalWork.doi, ExternalWork.publication_year)
    ).all():
        if doi:
            work_of[_normalize_doi(doi)] = work_id
        if year is not None:
            year_of[work_id] = year
    for work_id, year, status in session.execute(
        select(ReferenceWork.provider_work_id, ReferenceWork.publication_year, ReferenceWork.status)
    ).all():
        if status == "success" and year is not None:
            year_of.setdefault(work_id, year)
    openalex_refs: dict[str, set[str]] = {}
    for citing, cited in session.execute(
        select(CitationEdge.citing_work_id, CitationEdge.cited_work_id).where(
            CitationEdge.discovered_via == "referenced_works"
        )
    ).all():
        openalex_refs.setdefault(citing, set()).add(cited)
    openalex_count: dict[str, int] = {}
    for work_id, count in session.execute(
        select(EnrichmentObservation.provider_work_id, EnrichmentObservation.reference_count).where(
            EnrichmentObservation.status == "success"
        )
    ).all():
        if work_id is not None and count is not None:
            openalex_count[work_id] = count

    from lake_research_map.ingest.openalex import contradicted_zeros

    disputed = contradicted_zeros(session)["references"]
    source_of: dict[str, str] = {}
    references = dated = checked = consistent = 0
    known_empty = unenumerated = 0
    listed: set[str] = set()
    mostly_dated: set[str] = set()
    count_pairs = count_agree = 0
    for doi, citing_year in corpus.items():
        work_id = work_of.get(doi)
        entry = latest.get(doi)
        years: list[int | None]
        if entry and entry[1] == "success" and entry[2] > 0:
            source_of[doi] = "crossref"
            years = [
                year
                if year is not None
                else looked_up.get(ref_doi) or (corpus.get(ref_doi) if ref_doi else None)
                for ref_doi, year in crossref_refs.get(doi, [])
            ]
            if work_id in openalex_count and openalex_count[work_id] > 0:
                count_pairs += 1
                oa = openalex_count[work_id]
                count_agree += int(abs(entry[2] - oa) <= max(2, 0.1 * oa))
        elif work_id and openalex_refs.get(work_id):
            source_of[doi] = "openalex"
            years = [year_of.get(cited) for cited in openalex_refs[work_id]]
        elif work_id and openalex_count.get(work_id) == 0 and work_id not in disputed:
            known_empty += 1
            continue
        else:
            unenumerated += 1
            continue
        listed.add(doi)
        references += len(years)
        work_dated = sum(1 for year in years if year is not None)
        dated += work_dated
        if years and work_dated / len(years) >= 0.8:
            mostly_dated.add(doi)
        if citing_year is not None:
            for year in years:
                if year is not None:
                    checked += 1
                    consistent += int(year <= citing_year + 1)

    by_source = {"crossref": 0, "openalex": 0}
    for source in source_of.values():
        by_source[source] += 1
    return {
        "corpus": len(corpus),
        "works_by_source": by_source,
        "known_empty": known_empty,
        "unenumerated": unenumerated,
        "references": references,
        "dated": dated,
        "coverage": dated / references if references else 0.0,
        "temporal_checked": checked,
        "temporal_inconsistent": checked - consistent,
        "temporal_consistency": consistent / checked if checked else None,
        "count_pairs": count_pairs,
        "count_agreement": count_agree / count_pairs if count_pairs else None,
        "listed_dois": listed,
        "mostly_dated_dois": mostly_dated,
        "registrant": {doi: registrant_prefix(doi) for doi in corpus},
    }
