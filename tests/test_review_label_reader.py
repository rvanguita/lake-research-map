"""WP-09/WP-10: the join between persisted labels and the calibration.

`reviews import` wrote reviewer decisions to the database while the
calibration only ever saw a CSV uploaded in the browser, so a durable label
could never reach the threshold it exists to validate. These tests cover that
crossing and the two digests that let an approval name its evidence.
"""

from __future__ import annotations

import pandas as pd

from lake_research_map.db.gold_models import ReviewLabel
from lake_research_map.transform.review_workflows import (
    ADJUDICATED_REVIEWER,
    adjudicate,
    assign_dataset_articles,
    assignment_progress,
    collect_labels,
    ensure_protocol,
)
from lake_research_map.transform.screening_calibration import (
    label_set_digest,
    resolve_review_consensus,
    screening_model_digest,
)

VERSION = "v-2026-09-22"


def _round(session, subjects=("10.1000/a", "10.1000/b", "10.1000/c")):
    protocol = ensure_protocol(
        session,
        workflow="screening",
        protocol_version="v1",
        instructions="Include distribution-system-planning work.",
    )
    assign_dataset_articles(
        session,
        workflow="screening",
        dataset_version_id=VERSION,
        protocol=protocol,
        reviewer_ids=["ana", "bruno"],
        subject_ids=list(subjects),
    )
    return protocol


def _label(session, subject_id, reviewer_id, label, revision=1):
    from sqlalchemy import select

    from lake_research_map.db.gold_models import ReviewAssignment

    assignment = session.scalar(
        select(ReviewAssignment).where(
            ReviewAssignment.subject_id == subject_id,
            ReviewAssignment.reviewer_id == reviewer_id,
        )
    )
    session.add(
        ReviewLabel(
            assignment_id=assignment.id,
            label_revision=revision,
            label=label,
            payload_sha256="0" * 64,
        )
    )
    session.commit()


def test_collect_labels_returns_only_the_newest_revision(gold_session):
    _round(gold_session)
    _label(gold_session, "10.1000/a", "ana", "exclude", revision=1)
    _label(gold_session, "10.1000/a", "ana", "include", revision=2)

    rows = collect_labels(gold_session, workflow="screening", dataset_version_id=VERSION)

    assert rows == [
        {
            "doi": "10.1000/a",
            "subject_id": "10.1000/a",
            "reviewer": "ana",
            "manual_label": "include",
            "protocol_version": "v1",
        }
    ]


def test_an_adjudication_reaches_the_calibrator_without_erasing_the_raw_labels(gold_session):
    _round(gold_session)
    _label(gold_session, "10.1000/a", "ana", "include")
    _label(gold_session, "10.1000/a", "bruno", "exclude")
    adjudicate(
        gold_session,
        workflow="screening",
        dataset_version_id=VERSION,
        subject_id="10.1000/a",
        final_label="include",
        adjudicator_id="carla",
        rationale="in scope on full text",
    )

    rows = collect_labels(gold_session, workflow="screening", dataset_version_id=VERSION)
    reviewers = {row["reviewer"] for row in rows}
    assert reviewers == {"ana", "bruno", ADJUDICATED_REVIEWER}

    resolved = resolve_review_consensus(pd.DataFrame(rows))
    assert resolved.loc[0, "resolution"] == "adjudicated"
    assert bool(resolved.loc[0, "y_true"]) is True


def test_collect_labels_is_scoped_to_its_workflow_and_version(gold_session):
    _round(gold_session)
    _label(gold_session, "10.1000/a", "ana", "include")

    assert collect_labels(gold_session, workflow="screening", dataset_version_id="other") == []
    assert collect_labels(gold_session, workflow="pdf", dataset_version_id=VERSION) == []


def test_progress_separates_the_reviewer_who_has_not_started(gold_session):
    _round(gold_session)
    _label(gold_session, "10.1000/a", "ana", "include")
    _label(gold_session, "10.1000/b", "ana", "exclude")

    progress = assignment_progress(gold_session, workflow="screening", dataset_version_id=VERSION)

    assert progress["assignments"] == 6
    assert progress["labelled"] == 2
    assert progress["outstanding"] == 4
    # The point of the split: 2 of 6 hides that one reviewer did nothing.
    assert progress["labelled_by_reviewer"] == {"ana": 2}
    assert progress["outstanding_by_reviewer"] == {"ana": 1, "bruno": 3}


def test_label_set_digest_distinguishes_label_sets_that_share_a_consensus():
    unanimous = pd.DataFrame(
        [
            {"doi": "10.1000/a", "reviewer": "ana", "manual_label": "include"},
            {"doi": "10.1000/a", "reviewer": "bruno", "manual_label": "include"},
        ]
    )
    single = pd.DataFrame([{"doi": "10.1000/a", "reviewer": "ana", "manual_label": "include"}])

    # Both resolve to the same decision; an approval that cannot tell them
    # apart has not recorded the evidence it was granted on.
    assert resolve_review_consensus(unanimous).loc[0, "manual_label"] == "include"
    assert resolve_review_consensus(single).loc[0, "manual_label"] == "include"
    assert label_set_digest(unanimous) != label_set_digest(single)


def test_label_set_digest_ignores_row_order():
    rows = [
        {"doi": "10.1000/b", "reviewer": "bruno", "manual_label": "exclude"},
        {"doi": "10.1000/a", "reviewer": "ana", "manual_label": "include"},
    ]
    assert label_set_digest(pd.DataFrame(rows)) == label_set_digest(
        pd.DataFrame(list(reversed(rows)))
    )


def test_label_set_digest_tolerates_an_empty_or_malformed_frame():
    assert len(label_set_digest(pd.DataFrame())) == 64
    assert len(label_set_digest(pd.DataFrame([{"doi": "10.1000/a"}]))) == 64


def test_the_model_digest_covers_every_input_that_changes_the_rule():
    base = {
        "embed_model": "BAAI/bge-small-en-v1.5",
        "anchor_text": "distribution system planning",
        "off_anchor_text": "supply chain logistics",
        "threshold": 0.0,
        "min_recall": 0.98,
    }
    reference = screening_model_digest(**base)
    assert screening_model_digest(**base) == reference
    for field, value in (
        ("embed_model", "other/model"),
        ("anchor_text", "something else"),
        ("off_anchor_text", "something else"),
        ("threshold", 0.01),
        ("min_recall", 0.95),
    ):
        assert screening_model_digest(**{**base, field: value}) != reference, field


def test_a_stored_round_calibrates_end_to_end(gold_session):
    """setup -> import -> collect -> resolve -> calibrate, with no upload.

    The chain is the deliverable. Each link was already tested in isolation and
    the whole ran nowhere, because the calibrator's only input was a browser
    upload.
    """
    from lake_research_map.transform.screening_calibration import (
        calibrate_screening_threshold,
        validate_review_labels,
    )

    # 60 subjects with a clean margin separation, 30 per class: above the
    # calibrator's floor of 40 resolved with 10 in each class.
    subjects = [f"10.1000/{index:03d}" for index in range(60)]
    _round(gold_session, subjects=subjects)
    for index, doi in enumerate(subjects):
        decision = "include" if index < 30 else "exclude"
        for reviewer in ("ana", "bruno"):
            _label(gold_session, doi, reviewer, decision)

    rows = collect_labels(gold_session, workflow="screening", dataset_version_id=VERSION)
    assert len(rows) == 120

    scored = pd.DataFrame(
        {
            "doi": subjects,
            "relevance_margin": [0.05 + index * 0.001 for index in range(30)]
            + [-0.05 - index * 0.001 for index in range(30)],
        }
    )
    labels, issues = validate_review_labels(pd.DataFrame(rows), known_dois=set(subjects))
    assert issues.empty
    resolved = resolve_review_consensus(labels)
    assert int(resolved["resolved"].sum()) == 60

    result = calibrate_screening_threshold(resolved, scored, n_bootstrap=50, seed=9)

    assert result["valid"] is True
    assert result["n_resolved"] == 60
    assert result["metrics"]["recall"] == 1.0
    assert 0.0 <= result["threshold"] <= 0.1
    assert set(result["confidence_intervals"]) >= {"recall", "specificity", "precision"}

    # The digest names this exact body of evidence, so `reviews approve` can
    # bind the recorded threshold to it.
    assert len(label_set_digest(labels)) == 64


def test_a_reject_subject_is_reported_rather_than_silently_dropped(gold_session):
    """Screening subjects are not all DOIs.

    `evidence rejections` emits `reject::<source>::<id>` for records dropped
    before they ever had a margin. They cannot take part in a threshold
    evaluation, and the validation has to say so -- a round whose labels mostly
    vanish at this join otherwise looks like a round nobody labelled.
    """
    from lake_research_map.transform.screening_calibration import validate_review_labels

    _round(gold_session, subjects=["10.1000/a", "reject::ieee::991"])
    _label(gold_session, "10.1000/a", "ana", "include")
    _label(gold_session, "reject::ieee::991", "ana", "exclude")

    rows = collect_labels(gold_session, workflow="screening", dataset_version_id=VERSION)
    labels, issues = validate_review_labels(pd.DataFrame(rows), known_dois={"10.1000/a"})

    assert list(labels["doi"]) == ["10.1000/a"]
    assert list(issues["code"]) == ["unknown_doi"]
    assert issues.loc[0, "detail"] == "outside the current corpus"


def test_the_calibrate_command_prints_both_digests_and_applies_nothing(
    gold_session, capsys, tmp_path, monkeypatch
):
    """Exercise the CLI handler itself, not only the functions beneath it."""
    import argparse

    from lake_research_map import pipeline as pipeline_module
    from lake_research_map.db.gold_models import DatasetSemantics

    subjects = [f"10.1000/{index:03d}" for index in range(60)]
    _round(gold_session, subjects=subjects)
    for index, doi in enumerate(subjects):
        for reviewer in ("ana", "bruno"):
            _label(gold_session, doi, reviewer, "include" if index < 30 else "exclude")
        gold_session.add(
            DatasetSemantics(
                dataset_version_id=VERSION,
                doi=doi,
                relevance_score=0.6 if index < 30 else 0.4,
                offtopic_score=0.4 if index < 30 else 0.6,
                theme_id=0,
                theme_label="theme",
                map_x=0.0,
                map_y=0.0,
            )
        )
    gold_session.commit()

    output = tmp_path / "calibration.json"
    args = argparse.Namespace(version_id=VERSION, min_recall=0.98, output_json=str(output))
    pipeline_module._print_screening_calibration(gold_session, args)

    printed = capsys.readouterr().out
    assert "Label set SHA-256:" in printed
    assert "Model SHA-256:" in printed
    assert "Nothing was applied." in printed
    assert "reviews approve --workflow screening" in printed

    import json

    payload = json.loads(output.read_text())
    assert len(payload["label_set_sha256"]) == 64
    assert len(payload["model_sha256"]) == 64
    assert payload["metrics"]["recall"] == 1.0
    assert payload["n_holdout"] > 0


def test_calibrate_refuses_a_version_that_has_no_contrastive_margin(gold_session):
    import argparse

    import pytest

    from lake_research_map import pipeline as pipeline_module
    from lake_research_map.db.gold_models import DatasetSemantics

    _round(gold_session, subjects=["10.1000/a"])
    _label(gold_session, "10.1000/a", "ana", "include")
    gold_session.add(
        DatasetSemantics(
            dataset_version_id=VERSION,
            doi="10.1000/a",
            relevance_score=0.6,
            offtopic_score=None,
            theme_id=0,
            theme_label="theme",
            map_x=0.0,
            map_y=0.0,
        )
    )
    gold_session.commit()

    args = argparse.Namespace(version_id=VERSION, min_recall=0.98, output_json=None)
    with pytest.raises(ValueError, match="contrastive margin"):
        pipeline_module._print_screening_calibration(gold_session, args)
