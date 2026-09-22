"""WP-05/11/12/13: the candidate sets that unblock human review.

These generators are what stands between a delivered review workflow and a
reviewer being able to start. The tests below check the property that makes
each sample worth labelling — stratification — rather than just that a list
comes back.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from lake_research_map.transform.evidence_samples import (
    PDF_MATCH_THRESHOLD,
    RETRIEVAL_QUERIES,
    author_ambiguity_pairs,
    pool_retrieval_results,
    stratify_pdf_matches,
    stratify_taxonomy,
)


def test_pdf_sample_straddles_the_linking_threshold():
    """A sample of only confident matches cannot validate the threshold.

    The corpus links almost everything above 95, so a uniform draw would spend
    the whole review budget confirming easy matches and never reach the band
    where 85 actually decides anything.
    """
    frame = pd.DataFrame(
        {
            "doi": [f"10.1000/{i}" for i in range(60)],
            "pdf_path": [f"/data/articles/{i}.pdf" for i in range(60)],
            "pdf_match_score": list(np.linspace(PDF_MATCH_THRESHOLD, 100.0, 60)),
        }
    )

    subjects, stats = stratify_pdf_matches(frame, per_band=5, seed=1)

    assert len(subjects) == 15  # five from each of the three bands
    assert set(stats["bands"]) == {"85-90", "90-95", "95-100"}
    assert all(count > 0 for count in stats["bands"].values())
    # The subject is the pairing, not the article: the same DOI can be right
    # while its PDF is wrong.
    assert all("::" in subject for subject in subjects)
    assert all(subject.split("::")[1].endswith(".pdf") for subject in subjects)

    # Determinism, so two reviewers are given the same sample.
    assert subjects == stratify_pdf_matches(frame, per_band=5, seed=1)[0]

    empty, empty_stats = stratify_pdf_matches(pd.DataFrame())
    assert empty == [] and empty_stats["reason"] == "no_linked_pdfs"


def test_author_sample_covers_both_merge_and_split_risk():
    """Sampling one error mode measures half the error.

    A wrong merge inflates a ranking and invents collaborations; a wrong split
    deflates someone's output. They pull in opposite directions, so a sample
    drawn from only one cannot bound identity error.
    """
    names = [
        "J. Silva",
        "Joao Silva",
        "J. C. Silva",
        "Maria Souza",
        "M. Souza",
        "Ana Lima",
    ]

    subjects, stats = author_ambiguity_pairs(names)

    assert stats["population"] == len(names)
    assert stats["merge_risk"] > 0
    assert all("||" in subject for subject in subjects)
    assert len(subjects) == len(set(subjects))

    # A corpus with nothing ambiguous yields nothing to review, rather than
    # padding the sample with unambiguous names.
    quiet, quiet_stats = author_ambiguity_pairs(["Ana Lima", "Bruno Costa"])
    assert quiet == []
    assert quiet_stats["merge_risk"] == 0


def test_taxonomy_sample_populates_real_classes_and_an_unclassified_stratum():
    """Precision measured only on matched articles cannot see false negatives.

    The named strata matter as much as the unclassified one: an earlier version
    re-derived the match from the words in each label, matched nothing, and
    reported the entire corpus as unclassified -- a sample that asks a reviewer
    to label classes the classifier never assigns. Asserting only that the
    unclassified stratum exists passes in that broken state, so this asserts
    the named classes are populated too.
    """
    frame = pd.DataFrame(
        {
            "doi": [f"10.1000/{i}" for i in range(40)],
            "title": ["Genetic algorithm for distribution planning"] * 10
            + ["Mixed-integer linear programming expansion"] * 10
            + ["Multi-objective pareto planning of feeders"] * 10
            + ["An unrelated study of cold chain logistics"] * 10,
            "abstract": ["Planning of distribution networks."] * 40,
            "keywords": [["distribution planning"] for _ in range(40)],
            "year": [2020] * 40,
        }
    )

    subjects, stats = stratify_taxonomy(frame, per_class=3, seed=2)

    assert stats["population"] == 40
    populated = {label for label, count in stats["classes"].items() if count}
    assert populated >= {
        "Genetic Algorithms (GA)",
        "Mixed-Integer Linear Programming (MILP)",
        "Multi-objective Optimization",
    }
    assert any(not subject.startswith("unclassified::") for subject in subjects)
    assert any(subject.startswith("unclassified::") for subject in subjects)
    assert stats["unclassified"] == 10
    assert all("::" in subject for subject in subjects)

    assert stratify_taxonomy(pd.DataFrame())[0] == []


def test_taxonomy_strata_are_the_classifier_s_own_classes():
    """A stratum the dashboard does not recognise is not reviewable evidence."""
    from lake_research_map.dashboard.analytics import OPTIMIZATION_METHOD_PATTERNS

    frame = pd.DataFrame(
        {
            "doi": ["10.1000/1"],
            "title": ["Particle swarm optimization"],
            "abstract": [""],
        }
    )
    _subjects, stats = stratify_taxonomy(frame, per_class=1)
    assert set(stats["classes"]) == set(OPTIMIZATION_METHOD_PATTERNS)


def test_retrieval_sampling_fails_loudly_when_every_query_errors():
    """A backend that cannot run is a hard failure, not an empty result set.

    Swallowing it produced "0 judgements" with nothing to explain it, which is
    exactly how an unusable sample reaches a reviewer.
    """
    from lake_research_map.transform.evidence_samples import retrieval_candidates

    def broken(_text, _depth):
        raise AttributeError("module has no attribute 'hybrid_search'")

    with pytest.raises(RuntimeError, match="every retrieval query failed"):
        retrieval_candidates(broken)

    # A backend that works reports no failures and real judgements.
    subjects, stats = retrieval_candidates(lambda _t, d: ["10.1/a", "10.1/b"][:d], depth=2)
    assert stats["failed_queries"] == {}
    assert len(subjects) == 2 * stats["queries"]


def test_retrieval_pooling_judges_every_mode_on_the_same_labels():
    """Judging only the default's hits guarantees the default wins.

    Pooling the union across modes is what makes Recall@k comparable at all,
    so duplicates collapse and every mode is scored on one judgement set.
    """
    results = {
        "q01": ["10.1/a", "10.1/b", "10.1/a", "10.1/c"],  # dense, with a repeat
        "q02": ["10.1/b", "10.1/d"],  # lexical, overlapping q01
    }

    subjects, stats = pool_retrieval_results(results, depth=3)

    assert stats["queries"] == 2
    assert stats["judgements"] == len(subjects) == 5
    assert stats["per_query"] == {"q01": 3, "q02": 2}
    # The same DOI under two queries is two judgements, not one.
    assert "q01::10.1/b" in subjects and "q02::10.1/b" in subjects
    assert len(subjects) == len(set(subjects))


def test_retrieval_query_set_is_versioned_and_unique():
    """A Recall@k measured against different questions each run is not a metric."""
    ids = [query_id for query_id, _text in RETRIEVAL_QUERIES]
    assert len(ids) == len(set(ids)) >= 10
    assert all(text.strip() for _qid, text in RETRIEVAL_QUERIES)
