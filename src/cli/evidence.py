"""`evidence pdf-matches|rejections|authors|taxonomy|retrieval`: review subjects as a CSV."""

from __future__ import annotations

import argparse
import logging

from db.engines import get_session

logger = logging.getLogger(__name__)


def configure(subparsers) -> None:
    evidence = subparsers.add_parser(
        "evidence", help="Generate stratified candidate sets for human review"
    )
    actions = evidence.add_subparsers(dest="evidence_action", required=True)
    for name, helptext in (
        ("pdf-matches", "PDF-to-article pairings, banded around the linking threshold"),
        ("rejections", "Records excluded at silver, sampled per rejection reason"),
        ("authors", "Author-name pairs at risk of a wrong merge or split"),
        ("taxonomy", "Articles per taxonomy class, plus an unclassified stratum"),
        ("retrieval", "Pooled top-k results for the versioned technical query set"),
    ):
        action = actions.add_parser(name, help=helptext)
        action.add_argument("--output", required=True, help="CSV path to write the subjects to")
        action.add_argument("--limit", type=int, default=None, help="Cap on subjects per stratum")
        action.add_argument("--seed", type=int, default=0)


def run(args: argparse.Namespace) -> None:
    """Write a labelled-ready CSV of review subjects.

    Read-only against the corpus: generating a sample must never mutate what
    it is sampling. The output feeds `reviews setup --subject-file`, or is
    labelled directly and imported.
    """
    import csv
    from pathlib import Path

    from transform import evidence_samples

    action = args.evidence_action
    limit = args.limit
    with get_session("silver") as silver:
        if action == "pdf-matches":
            subjects, stats = evidence_samples.pdf_match_candidates(
                silver, per_band=limit or 15, seed=args.seed
            )
            workflow = "pdf"
        elif action == "rejections":
            subjects, stats = evidence_samples.rejection_candidates(
                silver, limit=limit or 30, seed=args.seed
            )
            workflow = "screening"
        elif action == "authors":
            subjects, stats = evidence_samples.author_ambiguity_candidates(
                silver, limit=limit or 40
            )
            workflow = "author"
        elif action == "taxonomy":
            subjects, stats = evidence_samples.taxonomy_candidates(
                silver, per_class=limit or 12, seed=args.seed
            )
            workflow = "taxonomy"
        else:
            # Retrieval pools every benchmarked mode -- the chunk-level ones the
            # v1 judgements came from and the article-level ones -- over the
            # published version, so it needs the embedded corpus, not silver.
            from apps.api.api import (
                BENCHMARK_MODES,
                RetrievalUnavailable,
                build_retriever,
                mode_runner,
            )

            retriever = build_retriever()
            runners = {
                name: mode_runner(retriever, settings) for name, settings in BENCHMARK_MODES.items()
            }
            pooled_modes: list[str] = []
            skipped_modes: dict[str, str] = {}

            def _retrieve(text: str, depth: int) -> list[str]:
                rankings = []
                for name, run_mode in runners.items():
                    if name in skipped_modes:
                        continue
                    try:
                        rankings.append(run_mode(text, depth))
                    except RetrievalUnavailable as exc:
                        # A mode that cannot run (no reranker download, Qdrant
                        # down) is left out of the pool and said so, rather
                        # than failing the whole sample.
                        skipped_modes[name] = str(exc)
                        logger.warning("evidence retrieval: mode %r skipped: %s", name, exc)
                        continue
                    if name not in pooled_modes:
                        pooled_modes.append(name)
                # Preserve each mode's rank order before de-duplicating the
                # pooled judgement candidates. No mode supplies the ground
                # truth for its competitors.
                return list(dict.fromkeys(doi for ranking in rankings for doi in ranking))

            subjects, stats = evidence_samples.retrieval_candidates(_retrieve, depth=limit or 10)
            stats = {**stats, "pooled_modes": pooled_modes, "skipped_modes": skipped_modes}
            workflow = "retrieval"

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("workflow", "subject_id", "label", "rationale"))
        writer.writeheader()
        for subject in subjects:
            writer.writerow(
                {"workflow": workflow, "subject_id": subject, "label": "", "rationale": ""}
            )
    logger.info("evidence %s: %d subjects -> %s (%s)", action, len(subjects), output, stats)
