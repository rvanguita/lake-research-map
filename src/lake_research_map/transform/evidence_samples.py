"""Stratified candidate sets for the human-evidence work packages.

`review_workflows.py` already owns the durable half of human review -- protocols,
dual assignments, append-only labels, adjudication. What it cannot do is decide
*what* to put in front of a reviewer: `assign_dataset_articles` takes whole
articles by DOI, while `WP-05`, `WP-11`, `WP-12` and `WP-13` each review
something else entirely -- a PDF-to-article pair, a pair of author names, a
label on one article, or one result for one query.

This module generates those subjects. Each function returns
``(subject_ids, stats)``; the ids feed straight into
``assign_dataset_articles(subject_ids=...)`` so no new review machinery exists.

The sampling is stratified rather than uniform on purpose. A uniform draw from
a corpus where almost every fuzzy PDF match scores above 95 spends the
reviewer's whole budget confirming easy matches and never reaches the band
where the threshold actually decides anything.

Nothing here writes: generating a candidate set must never mutate the corpus
it is sampling from.
"""

from __future__ import annotations

import logging
import re
from collections import defaultdict

import numpy as np
import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from lake_research_map.db.silver_models import Article as SilverArticle
from lake_research_map.db.silver_models import RejectedArticle

logger = logging.getLogger(__name__)

# The fuzzy-title threshold silver links PDFs at (see transform/silver_articles).
# The bands below straddle it because a validation sample that only contains
# confident matches cannot measure where the threshold should sit.
PDF_MATCH_THRESHOLD = 85.0
PDF_SCORE_BANDS: tuple[tuple[float, float], ...] = (
    (PDF_MATCH_THRESHOLD, 90.0),
    (90.0, 95.0),
    (95.0, 100.01),
)

# WP-13 needs a query set that is versioned, not improvised per run: a
# Recall@k measured against a different set of questions each time is not a
# measurement. These are the technical questions the corpus is meant to answer.
RETRIEVAL_QUERIES: tuple[tuple[str, str], ...] = (
    ("q01", "optimal placement and sizing of distributed generation in distribution networks"),
    ("q02", "distribution network expansion planning under load growth uncertainty"),
    ("q03", "reliability-oriented planning of radial distribution feeders"),
    ("q04", "mixed-integer linear programming model for distribution system planning"),
    ("q05", "electric vehicle charging infrastructure impact on distribution planning"),
    ("q06", "distribution network reconfiguration for loss minimization"),
    ("q07", "energy storage siting for voltage regulation in distribution grids"),
    ("q08", "multi-objective optimization of distribution expansion with renewables"),
    ("q09", "resilience planning of distribution systems against extreme weather"),
    ("q10", "stochastic programming for distribution planning with renewable uncertainty"),
)

_INITIAL = re.compile(r"\b([a-z])\b")


def _rows_to_frame(rows, columns: list[str]) -> pd.DataFrame:
    return pd.DataFrame([{c: getattr(r, c, None) for c in columns} for r in rows], columns=columns)


# --------------------------------------------------------------------------
# WP-05 -- PDF matching and no-DOI rejection
# --------------------------------------------------------------------------


def stratify_pdf_matches(
    frame: pd.DataFrame, *, per_band: int = 15, seed: int = 0
) -> tuple[list[str], dict]:
    """Sample linked PDFs in score bands straddling the linking threshold.

    Pure half of :func:`pdf_match_candidates`. The subject id is
    ``doi::pdf_path`` because the reviewer is judging the *pairing*, not the
    article: the same article can be correct and its PDF still wrong.
    """
    required = {"doi", "pdf_path", "pdf_match_score"}
    if frame.empty or not required.issubset(frame.columns):
        return [], {"bands": {}, "total": 0, "reason": "no_linked_pdfs"}

    linked = frame.dropna(subset=["doi", "pdf_path", "pdf_match_score"]).copy()
    linked["pdf_match_score"] = pd.to_numeric(linked["pdf_match_score"], errors="coerce")
    linked = linked.dropna(subset=["pdf_match_score"])
    rng = np.random.default_rng(seed)

    subjects: list[str] = []
    bands: dict[str, int] = {}
    for low, high in PDF_SCORE_BANDS:
        in_band = linked[(linked["pdf_match_score"] >= low) & (linked["pdf_match_score"] < high)]
        label = f"{low:g}-{high if high <= 100 else 100:g}"
        bands[label] = int(len(in_band))
        if in_band.empty:
            continue
        take = min(per_band, len(in_band))
        chosen = in_band.iloc[rng.choice(len(in_band), size=take, replace=False)]
        subjects.extend(f"{row.doi}::{row.pdf_path}" for row in chosen.itertuples())

    return sorted(set(subjects)), {
        "bands": bands,
        "total": len(set(subjects)),
        "linked_population": int(len(linked)),
    }


def pdf_match_candidates(
    silver_session: Session, *, per_band: int = 15, seed: int = 0
) -> tuple[list[str], dict]:
    """Stratified PDF-pairing subjects for the `pdf` review workflow."""
    rows = silver_session.scalars(
        select(SilverArticle).where(SilverArticle.has_pdf.is_(True))
    ).all()
    frame = _rows_to_frame(rows, ["doi", "title", "pdf_path", "pdf_match_score"])
    return stratify_pdf_matches(frame, per_band=per_band, seed=seed)


def rejection_candidates(
    silver_session: Session, *, limit: int = 30, seed: int = 0
) -> tuple[list[str], dict]:
    """Sample rejected records so the no-DOI exclusion can be checked for bias.

    A record dropped for having no DOI is invisible to every downstream
    analysis, so whether those records were in scope is a question only a
    human can answer -- and one the review has to answer to claim the
    exclusion is unbiased.
    """
    rows = silver_session.scalars(select(RejectedArticle)).all()
    if not rows:
        return [], {"total": 0, "by_reason": {}}

    by_reason: dict[str, list] = defaultdict(list)
    for row in rows:
        by_reason[row.reason].append(row)

    rng = np.random.default_rng(seed)
    subjects: list[str] = []
    counts: dict[str, int] = {}
    per_reason = max(1, limit // max(1, len(by_reason)))
    for reason, group in sorted(by_reason.items()):
        counts[reason] = len(group)
        take = min(per_reason, len(group))
        for index in rng.choice(len(group), size=take, replace=False):
            record = group[int(index)]
            subjects.append(f"reject::{record.source}::{record.source_id}")

    return sorted(set(subjects)), {"total": len(set(subjects)), "by_reason": counts}


# --------------------------------------------------------------------------
# WP-11 -- author identity
# --------------------------------------------------------------------------


def author_ambiguity_pairs(author_names: list[str], *, limit: int = 40) -> tuple[list[str], dict]:
    """Name pairs that heuristic canonicalization may have merged or split.

    Two error modes, and they pull in opposite directions:

    - **merge risk** -- distinct people collapsed into one canonical name,
      which inflates a ranking and invents collaborations. Detected as several
      raw spellings mapping onto one canonical form.
    - **split risk** -- one person spread over several canonical names, which
      deflates their output. Detected as same surname with compatible but
      unequal initials.

    A sample drawn from only one mode would measure only half the error.
    """
    from lake_research_map.dashboard.analytics import canonical_author

    unique = sorted({name.strip() for name in author_names if name and name.strip()})
    if len(unique) < 2:
        return [], {"merge_risk": 0, "split_risk": 0, "population": len(unique)}

    canonical_groups: dict[str, list[str]] = defaultdict(list)
    for name in unique:
        canonical_groups[canonical_author(name)].append(name)

    merge_pairs: list[tuple[str, str]] = []
    for variants in canonical_groups.values():
        if len(variants) < 2:
            continue
        for i, left in enumerate(variants):
            for right in variants[i + 1 :]:
                merge_pairs.append((left, right))

    # Split risk: same surname, different initial sets that do not contradict.
    by_surname: dict[str, list[str]] = defaultdict(list)
    for canonical in canonical_groups:
        parts = canonical.split()
        if parts:
            by_surname[parts[-1]].append(canonical)

    split_pairs: list[tuple[str, str]] = []
    for surname, names in by_surname.items():
        if len(names) < 2 or not surname:
            continue
        for i, left in enumerate(sorted(names)):
            for right in sorted(names)[i + 1 :]:
                left_initials = set(_INITIAL.findall(left.lower()))
                right_initials = set(_INITIAL.findall(right.lower()))
                # One initial set containing the other is the classic "J. Silva"
                # vs "J. C. Silva" case; disjoint sets are probably two people.
                if (
                    left_initials
                    and right_initials
                    and (left_initials <= right_initials or right_initials <= left_initials)
                ):
                    split_pairs.append((left, right))

    # Interleave so a truncated sample still covers both error modes.
    subjects: list[str] = []
    for merge, split in zip(merge_pairs, split_pairs, strict=False):
        subjects.append(f"{merge[0]}||{merge[1]}")
        subjects.append(f"{split[0]}||{split[1]}")
    leftover = merge_pairs[len(split_pairs) :] + split_pairs[len(merge_pairs) :]
    subjects.extend(f"{a}||{b}" for a, b in leftover)

    return subjects[:limit], {
        "merge_risk": len(merge_pairs),
        "split_risk": len(split_pairs),
        "population": len(unique),
    }


def author_ambiguity_candidates(
    silver_session: Session, *, limit: int = 40
) -> tuple[list[str], dict]:
    """Ambiguous author-name pairs for the `author` review workflow."""
    rows = silver_session.scalars(select(SilverArticle)).all()
    names: list[str] = []
    for row in rows:
        for name in row.authors or []:
            if isinstance(name, str):
                names.append(name)
    return author_ambiguity_pairs(names, limit=limit)


# --------------------------------------------------------------------------
# WP-12 -- engineering taxonomy
# --------------------------------------------------------------------------


def stratify_taxonomy(
    frame: pd.DataFrame, *, per_class: int = 12, seed: int = 0
) -> tuple[list[str], dict]:
    """Sample DOIs per taxonomy class, with an explicit unclassified stratum.

    The unclassified stratum is not optional. Precision measured only on
    articles the regex already matched cannot see false negatives, and a
    taxonomy that silently drops a third of the corpus is the failure mode
    most worth catching.
    """
    from lake_research_map.dashboard.analytics import optimization_methods_taxonomy

    if frame.empty or "doi" not in frame.columns:
        return [], {"classes": {}, "unclassified": 0}

    working = frame.copy()
    for column, default in (("title", ""), ("abstract", ""), ("keywords", None)):
        if column not in working.columns:
            working[column] = [[] for _ in range(len(working))] if default is None else default
    # The classifier reads `year` and `citation_count` for its summary columns.
    # Padding them keeps sampling independent of which layer the frame came
    # from -- the stratification only needs the text fields.
    if "year" not in working.columns:
        working["year"] = 2020
    if "citation_count" not in working.columns:
        working["citation_count"] = 0

    try:
        taxonomy = optimization_methods_taxonomy(working)
    except (KeyError, ValueError):
        logger.debug("taxonomy sampling: classifier failed", exc_info=True)
        return [], {"classes": {}, "unclassified": 0, "reason": "classifier_failed"}

    haystack = (
        working["title"].fillna("").astype(str)
        + " "
        + working["abstract"].fillna("").astype(str)
        + " "
        + working["keywords"].apply(
            lambda value: " ".join(value) if isinstance(value, list) else str(value or "")
        )
    ).str.lower()

    rng = np.random.default_rng(seed)
    subjects: list[str] = []
    class_counts: dict[str, int] = {}
    matched_any = pd.Series(False, index=working.index)

    summary = taxonomy.get("summary")
    labels = list(summary["method"]) if isinstance(summary, pd.DataFrame) else []
    for label in labels:
        # Match on the label's own words; the classifier's internal patterns
        # are not exported, and a reviewer only needs a stratum, not the rule.
        tokens = [token for token in re.split(r"[^a-z0-9]+", label.lower()) if len(token) > 3]
        if not tokens:
            continue
        hit = haystack.apply(lambda text, tk=tokens: any(token in text for token in tk))
        matched_any |= hit
        pool = working[hit]
        class_counts[label] = int(len(pool))
        if pool.empty:
            continue
        take = min(per_class, len(pool))
        chosen = pool.iloc[rng.choice(len(pool), size=take, replace=False)]
        subjects.extend(f"{label}::{doi}" for doi in chosen["doi"])

    unclassified = working[~matched_any]
    if not unclassified.empty:
        take = min(per_class, len(unclassified))
        chosen = unclassified.iloc[rng.choice(len(unclassified), size=take, replace=False)]
        subjects.extend(f"unclassified::{doi}" for doi in chosen["doi"])

    return sorted(set(subjects)), {
        "classes": class_counts,
        "unclassified": int(len(unclassified)),
        "population": int(len(working)),
    }


def taxonomy_candidates(
    silver_session: Session, *, per_class: int = 12, seed: int = 0
) -> tuple[list[str], dict]:
    """Per-class taxonomy subjects for the `taxonomy` review workflow."""
    rows = silver_session.scalars(select(SilverArticle)).all()
    frame = _rows_to_frame(rows, ["doi", "title", "abstract", "keywords", "year"])
    return stratify_taxonomy(frame, per_class=per_class, seed=seed)


# --------------------------------------------------------------------------
# WP-13 -- retrieval evaluation
# --------------------------------------------------------------------------


def pool_retrieval_results(
    results_by_query: dict[str, list[str]], *, depth: int = 10
) -> tuple[list[str], dict]:
    """Pool the top-k of every retrieval mode into one judgement set.

    Pooling is what makes the three modes comparable: judging only what the
    current default returns would score every alternative against a ground
    truth built from the default's own hits, which guarantees it wins. The
    union is judged once and every mode is then scored on the same labels.
    """
    subjects: list[str] = []
    per_query: dict[str, int] = {}
    for query_id, dois in sorted(results_by_query.items()):
        unique_dois = list(dict.fromkeys(doi for doi in dois if doi))[:depth]
        per_query[query_id] = len(unique_dois)
        subjects.extend(f"{query_id}::{doi}" for doi in unique_dois)
    return sorted(set(subjects)), {
        "queries": len(per_query),
        "judgements": len(set(subjects)),
        "per_query": per_query,
        "depth": depth,
    }


def retrieval_candidates(
    retrieve, *, queries: tuple[tuple[str, str], ...] = RETRIEVAL_QUERIES, depth: int = 10
) -> tuple[list[str], dict]:
    """Pooled retrieval subjects for the `retrieval` review workflow.

    `retrieve` is injected as ``(query_text, depth) -> list[doi]`` so the
    pooling logic is testable without embeddings, and so the caller decides
    which modes to pool.
    """
    results: dict[str, list[str]] = {}
    for query_id, text in queries:
        try:
            results[query_id] = list(retrieve(text, depth))
        except Exception:
            logger.debug("retrieval sampling: query %r failed", query_id, exc_info=True)
            results[query_id] = []
    subjects, stats = pool_retrieval_results(results, depth=depth)
    stats["query_set_size"] = len(queries)
    return subjects, stats
