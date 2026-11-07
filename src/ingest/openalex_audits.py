"""Audits of the OpenAlex evidence already stored in Bronze: coverage, integrity, validation.

Nothing here talks to the API -- the HTTP client, the fetch/crawl loops and the
persistence of what they return stay in `ingest/openalex.py`. These functions only
read the observation, edge and access tables and report how far the citation
graph, its annual trajectories and the access status can be trusted. They back the
`audit` command (`cli/audit.py`) and the Crossref reconciliation, which reuses
`contradicted_zeros`.

An OpenAlex zero is not always an answer: `reference_count` is
`len(referenced_works)`, empty when OpenAlex has no reference list as well as when
the work cites nothing, so every coverage figure below treats a zero as known only
when no other source contradicts it (`contradicted_zeros`).
"""

from __future__ import annotations

import logging

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from db.bronze_models import (
    AccessObservation,
    CitationEdge,
    CitationYearCount,
    EnrichmentObservation,
    ExternalWork,
)
from ingest.doi import normalize_doi
from ingest.openalex import short_work_id

logger = logging.getLogger(__name__)


def contradicted_zeros(session: Session) -> dict[str, set[str]]:
    """Work ids whose OpenAlex zero another source contradicts.

    OpenAlex reports `reference_count = len(referenced_works)`, and an empty
    list is sometimes OpenAlex lacking the list rather than the work citing
    nothing: of 324 works it reported at zero references, 81 have references
    deposited in Crossref and 18 a positive count in the corpus's own
    metadata. A zero is only a known answer when no source disagrees.
    """
    from db.bronze_models import Article as BronzeArticle
    from db.bronze_models import CrossrefReferenceList

    work_of = {
        normalize_doi(doi): work_id
        for work_id, doi in session.execute(
            select(ExternalWork.provider_work_id, ExternalWork.doi)
        ).all()
        if doi
    }
    references: set[str] = set()
    citations: set[str] = set()
    for doi, deposited in session.execute(
        select(CrossrefReferenceList.doi, CrossrefReferenceList.deposited)
    ).all():
        if deposited and normalize_doi(doi) in work_of:
            references.add(work_of[normalize_doi(doi)])
    for doi, reference_count, citation_count in session.execute(
        select(BronzeArticle.doi, BronzeArticle.reference_count, BronzeArticle.citation_count)
    ).all():
        work_id = work_of.get(normalize_doi(doi)) if doi else None
        if work_id is None:
            continue
        if (reference_count or 0) > 0:
            references.add(work_id)
        if (citation_count or 0) > 0:
            citations.add(work_id)
    return {"references": references, "citations": citations}


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
            "backward_ids": set(),
            "forward_ids": set(),
            "usable_ids": set(),
        }

    wanted_set = set(wanted)
    rows = session.execute(
        select(CitationEdge.citing_work_id, CitationEdge.cited_work_id, CitationEdge.discovered_via)
    ).all()
    backward: set[str] = set()
    forward_edges: set[str] = set()
    for citing, cited, via in rows:
        if via == "referenced_works" and citing in wanted_set:
            backward.add(citing)
        elif via == "cites_query" and cited in wanted_set:
            forward_edges.add(cited)

    # A work whose own observation reports zero references has a *known*,
    # empty reference list. It produces no edge, so counting edge presence
    # alone scored all 324 such works in the corpus as uncovered and reported
    # backward coverage at 89.5% when the true figure was 100%. The same
    # mistake had already been fixed for the forward direction; this is its
    # mirror image.
    backward_with_edges = len(backward)
    disputed = contradicted_zeros(session)["references"]
    for work_id, reference_count in session.execute(
        select(EnrichmentObservation.provider_work_id, EnrichmentObservation.reference_count).where(
            EnrichmentObservation.provider == "openalex",
            EnrichmentObservation.status == "success",
        )
    ).all():
        if work_id in wanted_set and reference_count == 0 and work_id not in disputed:
            backward.add(work_id)

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
            if work_id in wanted_set and crawled_at is not None:
                crawled.add(work_id)
                if was_truncated:
                    truncated.add(work_id)
    except Exception:  # pragma: no cover - pre-migration database
        # Before the additive migration the columns do not exist; fall back to
        # edge presence so the audit degrades instead of failing.
        logger.warning("forward-crawl columns unavailable; falling back to edge presence")
        crawled = set(forward_edges)

    forward = crawled or forward_edges
    population = len(wanted_set)
    both = (backward & forward) - truncated
    return {
        "population": population,
        "with_backward": len(backward),
        "with_backward_edges": backward_with_edges,
        # The sets themselves, so an audit can break coverage out by publisher
        # without re-deriving (and possibly re-mis-deriving) them.
        "backward_ids": backward,
        "forward_ids": forward,
        "usable_ids": both,
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


def citation_graph_integrity(session: Session) -> dict:
    """Structural checks, plus agreement between OpenAlex's two edge indexes.

    OpenAlex exposes each citation twice: in the citing work's
    `referenced_works` and in a `cites:` query on the cited work. For a pair
    where *both* ends are corpus works, and the cited work's forward crawl is
    complete, the two must agree. The agreement rate is therefore a direct
    measure of how far the graph can be trusted, independent of any coverage
    figure.
    """
    works = {
        work_id: (crawled_at, truncated)
        for work_id, crawled_at, truncated in session.execute(
            select(
                ExternalWork.provider_work_id,
                ExternalWork.citing_crawled_at,
                ExternalWork.citing_truncated,
            ).where(ExternalWork.provider == "openalex")
        ).all()
    }
    complete_forward = {w for w, (at, trunc) in works.items() if at is not None and not trunc}

    backward: set[tuple[str, str]] = set()
    forward: set[tuple[str, str]] = set()
    self_loops = dangling = 0
    for citing, cited, via in session.execute(
        select(CitationEdge.citing_work_id, CitationEdge.cited_work_id, CitationEdge.discovered_via)
    ).all():
        if short_work_id(citing) == short_work_id(cited):
            self_loops += 1
        if via == "referenced_works":
            backward.add((citing, cited))
        elif via == "cites_query":
            forward.add((citing, cited))
            if cited not in works:
                dangling += 1

    # Backward -> forward: every corpus-internal reference to a fully crawled
    # work should reappear in that work's `cites:` result.
    checkable_b = {pair for pair in backward if pair[0] in works and pair[1] in complete_forward}
    confirmed_b = checkable_b & forward
    # Forward -> backward: every in-corpus citer found by the crawl should list
    # the cited work among its own references.
    checkable_f = {pair for pair in forward if pair[0] in works and pair[1] in complete_forward}
    confirmed_f = checkable_f & backward

    checkable = len(checkable_b) + len(checkable_f)
    agreement = (len(confirmed_b) + len(confirmed_f)) / checkable if checkable else None
    return {
        "self_loops": self_loops,
        "dangling_forward": dangling,
        "internal_pairs_checked": checkable,
        "backward_confirmed": len(confirmed_b),
        "backward_checkable": len(checkable_b),
        "forward_confirmed": len(confirmed_f),
        "forward_checkable": len(checkable_f),
        "agreement": agreement,
    }


# IEEE CSV licence values that *imply* open access. `IEEE` (publisher
# copyright) does not imply closed: a repository copy makes an article green
# OA, so that disagreement is legitimate and excluded from the test.
IEEE_OPEN_LICENSES = {"CCBY": "cc-by", "CCBYNCND": "cc-by-nc-nd", "OAPA": None}


def access_validation(session: Session, ieee_licenses: dict[str, str]) -> dict:
    """Check OpenAlex access status against the IEEE CSV, an independent source.

    Two implications are testable. An article the IEEE export licenses under
    Creative Commons or OAPA must be open access in OpenAlex. And where both
    sources name a CC licence, it must be the same one. Everything else --
    including an `IEEE`-copyright article that OpenAlex finds green -- is not
    a contradiction and is not counted as one.
    """
    work_by_doi = {
        normalize_doi(doi): work_id
        for work_id, doi in session.execute(
            select(ExternalWork.provider_work_id, ExternalWork.doi).where(
                ExternalWork.provider == "openalex"
            )
        ).all()
        if doi
    }
    latest: dict[str, tuple] = {}
    for work_id, observed_at, is_oa, oa_status, license_ in session.execute(
        select(
            AccessObservation.provider_work_id,
            AccessObservation.observed_at,
            AccessObservation.is_oa,
            AccessObservation.oa_status,
            AccessObservation.license,
        ).where(AccessObservation.provider == "openalex")
    ).all():
        if work_id not in latest or observed_at > latest[work_id][0]:
            latest[work_id] = (observed_at, is_oa, oa_status, license_)

    observed = sum(1 for row in latest.values() if row[1] is not None)
    implied_open = open_confirmed = licence_pairs = licence_agree = 0
    contradictions: list[str] = []
    for doi, ieee_license in ieee_licenses.items():
        work_id = work_by_doi.get(normalize_doi(doi))
        if work_id is None or work_id not in latest or ieee_license not in IEEE_OPEN_LICENSES:
            continue
        _, is_oa, _, oa_license = latest[work_id]
        implied_open += 1
        if is_oa:
            open_confirmed += 1
        else:
            contradictions.append(doi)
        expected = IEEE_OPEN_LICENSES[ieee_license]
        if expected and oa_license:
            licence_pairs += 1
            licence_agree += int(oa_license == expected)
    return {
        "works_with_access": observed,
        "works_observed": len(latest),
        "ieee_open_licensed": implied_open,
        "open_confirmed": open_confirmed,
        "open_agreement": open_confirmed / implied_open if implied_open else None,
        "licence_pairs": licence_pairs,
        "licence_agreement": licence_agree / licence_pairs if licence_pairs else None,
        "contradictions": contradictions[:20],
    }


def citation_year_coverage(session: Session, corpus_years: dict[str, int] | None = None) -> dict:
    """WP-23's two measurements: annual trajectories, and cited-reference years.

    **Trajectories.** OpenAlex's `counts_by_year` lists only years with at
    least one citation, so a never-cited work arrives with an empty series. Its
    trajectory is known -- zero every year -- and counting it as uncovered is
    what reported 76.7% on a corpus whose true figure was 98.6%: 677 of the 721
    "missing" series belonged to works with no citations at all.

    The series also starts in a fixed year (2012 for this provider), so a work
    published earlier has a *left-censored* history even when its series is
    present. Longevity and Sleeping Beauty need the whole history from
    publication, so those works are reported separately rather than folded
    into coverage.

    **Reference years.** Price's index needs the publication year of each
    cited reference. A reference is dated when the cited work is in the corpus
    (`lit_external_works`) or has been resolved (`lit_reference_works`).
    """
    from db.bronze_models import ReferenceWork

    works = {
        work_id: (doi, year)
        for work_id, doi, year in session.execute(
            select(
                ExternalWork.provider_work_id, ExternalWork.doi, ExternalWork.publication_year
            ).where(ExternalWork.provider == "openalex")
        ).all()
    }
    with_series = set(
        session.scalars(
            select(CitationYearCount.provider_work_id)
            .where(CitationYearCount.provider == "openalex")
            .distinct()
        ).all()
    ) & set(works)
    never_cited = {
        work_id
        for work_id, count in session.execute(
            select(
                EnrichmentObservation.provider_work_id, EnrichmentObservation.citation_count
            ).where(
                EnrichmentObservation.provider == "openalex",
                EnrichmentObservation.status == "success",
            )
        ).all()
        if work_id in works and (count or 0) == 0
    } - contradicted_zeros(session)["citations"]
    known = with_series | never_cited
    series_start = session.scalar(
        select(func.min(CitationYearCount.year)).where(CitationYearCount.provider == "openalex")
    )
    left_censored = (
        {w for w, (_, year) in works.items() if year is not None and year < series_start}
        if series_start is not None
        else set()
    )
    complete_history = known - left_censored

    year_of = {w: year for w, (_, year) in works.items() if year is not None}
    for work_id, year, status in session.execute(
        select(ReferenceWork.provider_work_id, ReferenceWork.publication_year, ReferenceWork.status)
    ).all():
        if status == "success" and year is not None:
            year_of.setdefault(work_id, year)
    references = session.execute(
        select(CitationEdge.citing_work_id, CitationEdge.cited_work_id).where(
            CitationEdge.discovered_via == "referenced_works"
        )
    ).all()
    reference_pairs = {(citing, cited) for citing, cited in references}
    distinct_cited = {cited for _, cited in reference_pairs}
    dated_pairs = sum(1 for _, cited in reference_pairs if cited in year_of)

    # Validation, as opposed to coverage: are the years themselves right?
    # (1) The provider's year for a corpus work against the corpus's own
    # metadata. A one-year gap is the online-first versus issue-year
    # difference, not an error, so agreement is scored within +/-1.
    agreement_pairs = agreement_within_one = 0
    if corpus_years:
        normalized_corpus = {
            normalize_doi(doi): year for doi, year in corpus_years.items() if year is not None
        }
        for doi, year in ((d, y) for d, y in works.values() if d and y is not None):
            corpus_year = normalized_corpus.get(normalize_doi(doi))
            if corpus_year is not None:
                agreement_pairs += 1
                agreement_within_one += int(abs(year - corpus_year) <= 1)
    # (2) Temporal consistency: a reference cannot be published after the
    # work citing it, beyond the one year an in-press citation allows.
    consistent_pairs = checked_pairs = 0
    for citing, cited in reference_pairs:
        citing_year = year_of.get(citing)
        cited_year = year_of.get(cited)
        if citing_year is not None and cited_year is not None:
            checked_pairs += 1
            consistent_pairs += int(cited_year <= citing_year + 1)

    population = len(works)
    return {
        "population": population,
        "with_series": len(with_series),
        "never_cited": len(never_cited - with_series),
        "known": len(known),
        "known_coverage": len(known) / population if population else 0.0,
        "unexplained": population - len(known),
        "series_start": series_start,
        "left_censored": len(left_censored),
        "complete_history": len(complete_history),
        "reference_pairs": len(reference_pairs),
        "dated_reference_pairs": dated_pairs,
        "reference_year_coverage": dated_pairs / len(reference_pairs) if reference_pairs else 0.0,
        "distinct_cited": len(distinct_cited),
        "distinct_cited_dated": len(distinct_cited & set(year_of)),
        "year_agreement_pairs": agreement_pairs,
        "year_agreement": agreement_within_one / agreement_pairs if agreement_pairs else None,
        "temporal_checked": checked_pairs,
        "temporal_inconsistent": checked_pairs - consistent_pairs,
        "temporal_consistency": consistent_pairs / checked_pairs if checked_pairs else None,
        "known_ids": known,
        "complete_history_ids": complete_history,
        "work_dois": {w: doi for w, (doi, _) in works.items()},
    }
