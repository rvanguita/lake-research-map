"""`audit reference-corpus|citation-graph|citation-years`: read-only acceptance audits."""

from __future__ import annotations

import argparse

from sqlalchemy import func, select

from db.bootstrap import bootstrap
from db.engines import get_session
from quality import (
    ACCESS_AGREEMENT_GATE,
    COVERAGE_GATE,
    INTEGRITY_AGREEMENT_GATE,
    TEMPORAL_GATE,
    YEAR_AGREEMENT_GATE,
)


def configure(subparsers) -> None:
    audit = subparsers.add_parser("audit", help="Run read-only corpus acceptance audits")
    actions = audit.add_subparsers(dest="audit_action", required=True)
    actions.add_parser("reference-corpus", help="Report active version and contract coverage")
    actions.add_parser(
        "citation-graph",
        help="Measure forward/backward citation coverage and whether disruption is usable",
    )
    actions.add_parser(
        "citation-years",
        help="Measure annual citation-count coverage behind Price/longevity analyses",
    )


def _verdict(passed: bool | None) -> str:
    return "n/a (nothing checkable)" if passed is None else ("PASS" if passed else "FAIL")


def _audit_citation_graph() -> None:
    """Report WP-24's two gates -- graph integrity and access validation -- plus coverage.

    Completion reads: "Graph integrity and access-validation audits pass;
    CD/OACA remain unavailable if coverage or confounding control is
    inadequate." So integrity and access have to pass for the package to
    close, while the disruption index may legitimately stay unavailable; the
    two are reported as separate verdicts for that reason.
    """
    from db.bronze_models import ExternalWork
    from db.gold_models import DatasetArticle, PublicationState
    from ingest.openalex_audits import (
        access_validation,
        citation_graph_coverage,
        citation_graph_integrity,
    )

    bootstrap()
    session = get_session("bronze")
    try:
        works = dict(
            session.execute(
                select(ExternalWork.provider_work_id, ExternalWork.doi).where(
                    ExternalWork.provider == "openalex"
                )
            ).all()
        )
        if not works:
            print(
                "No OpenAlex works observed, so citation coverage is 0 by absence rather than "
                "by measurement. Run `enrichment refresh-openalex` and then "
                "`enrichment refresh-citations`."
            )
            return
        coverage = citation_graph_coverage(session, list(works))
        population = coverage["population"]
        print("Coverage")
        print(f"  Population:        {population}")
        print(
            f"  Backward known:    {coverage['with_backward']} ({coverage['backward_coverage']:.1%})"
            f" -- {coverage['with_backward_edges']} with reference edges, the rest report "
            "zero references"
        )
        print(
            f"  Forward crawled:   {coverage['with_forward']} ({coverage['forward_coverage']:.1%})"
            f", {coverage['truncated']} truncated, {coverage['with_forward_edges']} with citers"
        )
        print(
            f"  Usable for CD:     {coverage['usable_for_disruption']} "
            f"({coverage['disruption_coverage']:.1%}) -- both directions known, not truncated"
        )

        integrity = citation_graph_integrity(session)
        agreement = integrity["agreement"]
        integrity_ok = (
            None
            if agreement is None
            else integrity["self_loops"] == 0
            and integrity["dangling_forward"] == 0
            and agreement >= INTEGRITY_AGREEMENT_GATE
        )
        print("\nGraph integrity")
        print(f"  Self-loops:        {integrity['self_loops']}")
        print(
            f"  Dangling forward:  {integrity['dangling_forward']} (cites: edges to a non-corpus work)"
        )
        print(
            f"  Cross-index check: {integrity['backward_confirmed']}/{integrity['backward_checkable']}"
            f" references re-found by cites:, {integrity['forward_confirmed']}/"
            f"{integrity['forward_checkable']} citers listing the reference"
        )
        if agreement is not None:
            print(f"  Agreement:         {agreement:.1%} (gate {INTEGRITY_AGREEMENT_GATE:.0%})")
        print(f"  Verdict:           {_verdict(integrity_ok)}")

        gold = get_session("gold")
        try:
            state = gold.get(PublicationState, 1)
            licenses = (
                dict(
                    gold.execute(
                        select(DatasetArticle.doi, DatasetArticle.license).where(
                            DatasetArticle.dataset_version_id == state.active_version_id,
                            DatasetArticle.license.is_not(None),
                        )
                    ).all()
                )
                if state and state.active_version_id
                else {}
            )
        finally:
            gold.close()
        access = access_validation(session, licenses)
        access_ok = (
            None
            if access["open_agreement"] is None
            else access["open_agreement"] >= ACCESS_AGREEMENT_GATE
        )
        print("\nAccess validation (OpenAlex against the IEEE CSV licence, an independent source)")
        print(
            f"  Access status observed: {access['works_with_access']} / {access['works_observed']} works"
        )
        if access["open_agreement"] is not None:
            print(
                f"  CC/OAPA-licensed and open in OpenAlex: {access['open_confirmed']}/"
                f"{access['ieee_open_licensed']} ({access['open_agreement']:.1%}, gate "
                f"{ACCESS_AGREEMENT_GATE:.0%})"
            )
        if access["licence_agreement"] is not None:
            print(
                f"  Same CC licence named by both: {access['licence_agreement']:.1%} of "
                f"{access['licence_pairs']} pairs"
            )
        if access["contradictions"]:
            print(
                f"  Open-licensed but closed in OpenAlex: {', '.join(access['contradictions'][:5])}"
            )
        print(f"  Verdict: {_verdict(access_ok)}")

        cd_ok = coverage["disruption_coverage"] >= COVERAGE_GATE
        print(
            f"\nDisruption index: {'available' if cd_ok else 'unavailable'} -- usable coverage "
            f"{coverage['disruption_coverage']:.1%} against a {COVERAGE_GATE:.0%} gate"
        )

        _print_registrant_coverage(
            {
                "backward known": {works[w] for w in coverage["backward_ids"] if works.get(w)},
                "forward crawled": {works[w] for w in coverage["forward_ids"] if works.get(w)},
                "usable for CD": {works[w] for w in coverage["usable_ids"] if works.get(w)},
            },
            "Citation-graph coverage",
        )
    finally:
        session.close()


def _print_registrant_coverage(columns: dict, title: str) -> None:
    """Break external-data coverage out by publisher, against the whole corpus.

    An aggregate percentage over a single-publisher sample is precisely the
    number that hid the sampling bias found on 2026-09-22. Each column is one
    named measurement, so the table can never again show *observed* works under
    a heading that claims *covered* ones -- which is what the first version of
    this table did.
    """
    from db.gold_models import DatasetArticle, PublicationState
    from ingest.openalex import registrant_label, registrant_prefix

    gold = get_session("gold")
    try:
        state = gold.get(PublicationState, 1)
        if state is None or not state.active_version_id:
            return
        corpus = gold.scalars(
            select(DatasetArticle.doi).where(
                DatasetArticle.dataset_version_id == state.active_version_id
            )
        ).all()
    finally:
        gold.close()
    if not corpus:
        return

    totals: dict[str, int] = {}
    for doi in corpus:
        totals[registrant_prefix(doi)] = totals.get(registrant_prefix(doi), 0) + 1
    counted = {
        name: {
            prefix: sum(1 for doi in dois if registrant_prefix(doi) == prefix) for prefix in totals
        }
        for name, dois in columns.items()
    }
    names = list(columns)
    print(f"\n{title} by publisher (denominator is the corpus):")
    print("  " + f"{'publisher':10} {'corpus':>6}  " + "  ".join(f"{name:>18}" for name in names))
    for prefix, total in sorted(totals.items(), key=lambda item: -item[1])[:6]:
        cells = "  ".join(
            f"{counted[name][prefix]:>6} ({counted[name][prefix] / total:6.1%})    "
            for name in names
        )
        flag = "  <- not reached" if counted[names[0]][prefix] == 0 else ""
        print(f"  {registrant_label(prefix):10} {total:>6}  {cells}{flag}")


def _audit_citation_years() -> None:
    """Report WP-23's gate: annual trajectories and cited-reference years.

    Completion reads: "Coverage and validation gates pass; only then may Price,
    longevity, or Sleeping Beauty panels enter WP-21 review." Both halves of
    the deliverable are measured here, and left-censoring is reported beside
    them, because a trajectory that starts in 2012 is complete for Price's
    index and incomplete for any analysis that needs the whole history.
    """
    from db.gold_models import DatasetArticle, PublicationState
    from ingest.openalex_audits import citation_year_coverage

    bootstrap()
    gold = get_session("gold")
    try:
        state = gold.get(PublicationState, 1)
        corpus_years = (
            dict(
                gold.execute(
                    select(DatasetArticle.doi, DatasetArticle.year).where(
                        DatasetArticle.dataset_version_id == state.active_version_id
                    )
                ).all()
            )
            if state and state.active_version_id
            else {}
        )
    finally:
        gold.close()
    session = get_session("bronze")
    try:
        cov = citation_year_coverage(session, corpus_years)
        if not cov["population"]:
            print(
                "No OpenAlex works observed, so annual-count coverage is 0 by absence rather "
                "than by measurement. Run `enrichment refresh-openalex` first."
            )
            return
        population = cov["population"]
        print("Annual citation trajectories")
        print(f"  Observed works:        {population}")
        print(
            f"  Trajectory known:      {cov['known']} ({cov['known_coverage']:.1%}) -- "
            f"{cov['with_series']} with a series, {cov['never_cited']} never cited "
            "(an empty series is a known zero, not a gap)"
        )
        print(f"  Unexplained gaps:      {cov['unexplained']}")
        if cov["series_start"] is not None:
            print(
                f"  Left-censored:         {cov['left_censored']} works published before "
                f"{cov['series_start']}, where the provider's series begins"
            )
            print(
                f"  Complete from publication: {cov['complete_history']} "
                f"({cov['complete_history'] / population:.1%}) -- the population longevity and "
                "Sleeping Beauty may use"
            )

        from ingest.crossref import reference_year_coverage

        refs = reference_year_coverage(session, corpus_years)
        print("\nCited-reference publication years (Price's index, reference age)")
        print(
            f"  Reference lists:       {refs['works_by_source']['crossref']} from Crossref deposits, "
            f"{refs['works_by_source']['openalex']} from OpenAlex (one list per work, never spliced)"
        )
        print(
            f"  Known empty:           {refs['known_empty']}; unenumerated (no list from either "
            f"provider): {refs['unenumerated']}"
        )
        print(
            f"  Dated references:      {refs['dated']} / {refs['references']} "
            f"({refs['coverage']:.1%})"
        )

        print("\nValidation of the years themselves")
        if cov["year_agreement"] is not None:
            print(
                f"  Provider year vs corpus metadata: {cov['year_agreement']:.1%} within one year "
                f"over {cov['year_agreement_pairs']} works (gate {YEAR_AGREEMENT_GATE:.0%}; a one-"
                "year gap is online-first versus issue year)"
            )
        if refs["temporal_consistency"] is not None:
            print(
                f"  Reference not newer than its citer (+1 in-press): "
                f"{refs['temporal_consistency']:.2%} over {refs['temporal_checked']} dated "
                f"references, {refs['temporal_inconsistent']} inconsistent (gate {TEMPORAL_GATE:.0%})"
            )
        if refs["count_agreement"] is not None:
            print(
                f"  Crossref vs OpenAlex reference count, within 10% or 2: "
                f"{refs['count_agreement']:.1%} of {refs['count_pairs']} works (reported, not gated)"
            )

        trajectories_ok = cov["known_coverage"] >= COVERAGE_GATE
        references_ok = refs["coverage"] >= COVERAGE_GATE
        validation_ok = (
            None
            if cov["year_agreement"] is None or refs["temporal_consistency"] is None
            else cov["year_agreement"] >= YEAR_AGREEMENT_GATE
            and refs["temporal_consistency"] >= TEMPORAL_GATE
        )
        print(
            f"\nVerdict: trajectories {_verdict(trajectories_ok)}, reference years "
            f"{_verdict(references_ok)} (coverage gate {COVERAGE_GATE:.0%} each), "
            f"validation {_verdict(validation_ok)}"
        )
        if trajectories_ok and references_ok and validation_ok:
            print("  WP-23 gates met: Price, longevity and Sleeping Beauty may enter WP-21 review.")
        if not references_ok:
            print(
                "  Price's index stays unavailable until reference years are dated: run "
                "`enrichment crossref-references`, then `enrichment resolve-references`."
            )

        dois = cov["work_dois"]
        _print_registrant_coverage(
            {
                "trajectory known": {dois[w] for w in cov["known_ids"] if dois.get(w)},
                "complete history": {dois[w] for w in cov["complete_history_ids"] if dois.get(w)},
                "refs >=80% dated": refs["mostly_dated_dois"],
            },
            "Annual-trajectory coverage",
        )
    finally:
        session.close()


def run(args: argparse.Namespace) -> None:
    if args.audit_action == "citation-graph":
        _audit_citation_graph()
        return
    if args.audit_action == "citation-years":
        _audit_citation_years()
        return

    from db.gold_models import (
        DatasetArticle,
        DatasetChunk,
        DatasetSemantics,
        PublicationState,
        QualityResult,
    )

    bootstrap()
    session = get_session("gold")
    try:
        state = session.get(PublicationState, 1)
        version_id = state.active_version_id if state else None
        if not version_id:
            raise ValueError("no active dataset version")
        counts = {
            "version_id": version_id,
            "articles": session.scalar(
                select(func.count())
                .select_from(DatasetArticle)
                .where(DatasetArticle.dataset_version_id == version_id)
            ),
            "chunks": session.scalar(
                select(func.count())
                .select_from(DatasetChunk)
                .where(DatasetChunk.dataset_version_id == version_id)
            ),
            "semantics": session.scalar(
                select(func.count())
                .select_from(DatasetSemantics)
                .where(DatasetSemantics.dataset_version_id == version_id)
            ),
            "failed_quality_checks": session.scalar(
                select(func.count())
                .select_from(QualityResult)
                .where(
                    QualityResult.dataset_version_id == version_id,
                    QualityResult.severity == "error",
                    QualityResult.passed.is_(False),
                )
            ),
        }
        print(counts)
    finally:
        session.close()
