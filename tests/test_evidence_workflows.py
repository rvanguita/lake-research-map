from __future__ import annotations

import csv
from datetime import datetime

from lake_research_map.db.bronze_models import (
    AccessObservation,
    CitationEdge,
    CitationYearCount,
    EnrichmentObservation,
    ExternalWork,
)
from lake_research_map.db.gold_models import (
    DatasetArticle,
    ReviewAdjudication,
    ReviewAssignment,
    ReviewLabel,
)
from lake_research_map.ingest.enrichment import load_enrichment_observations
from lake_research_map.ingest.openalex import _persist_openalex_evidence
from lake_research_map.transform.review_workflows import (
    adjudicate,
    assign_dataset_articles,
    ensure_protocol,
    export_assignments,
    import_labels,
)


def test_enrichment_observations_are_selected_as_of_without_rewriting_history(bronze_session):
    bronze_session.add_all(
        [
            EnrichmentObservation(
                provider="openalex",
                doi="10.1/a",
                observed_at=datetime(2025, 1, 1),
                status="success",
                citation_count=3,
                reference_count=8,
            ),
            EnrichmentObservation(
                provider="openalex",
                doi="10.1/a",
                observed_at=datetime(2026, 1, 1),
                status="success",
                citation_count=7,
                reference_count=8,
            ),
            EnrichmentObservation(
                provider="openalex",
                doi="10.1/a",
                observed_at=datetime(2026, 2, 1),
                status="error",
                error_message="timeout",
            ),
        ]
    )
    bronze_session.commit()

    old = load_enrichment_observations(bronze_session, as_of=datetime(2025, 6, 1))
    current = load_enrichment_observations(bronze_session)

    assert old["10.1/a"]["citation_count"] == 3
    assert current["10.1/a"]["citation_count"] == 7
    assert bronze_session.query(EnrichmentObservation).count() == 3


def test_openalex_evidence_persists_history_graph_and_access(bronze_session):
    observed_at = datetime(2026, 1, 1)
    _persist_openalex_evidence(
        bronze_session,
        {
            "doi": "10.1/a",
            "provider_work_id": "https://openalex.org/W1",
            "payload": {
                "display_name": "Study",
                "publication_year": 2020,
                "counts_by_year": [{"year": 2025, "cited_by_count": 4}],
                "referenced_works": ["https://openalex.org/W2"],
                "open_access": {"is_oa": True, "oa_status": "gold"},
                "best_oa_location": {
                    "license": "cc-by",
                    "landing_page_url": "https://example.test/article",
                },
            },
        },
        observed_at,
    )
    bronze_session.commit()

    assert bronze_session.query(ExternalWork).one().publication_year == 2020
    assert bronze_session.query(CitationYearCount).one().citation_count == 4
    assert bronze_session.query(CitationEdge).one().cited_work_id.endswith("W2")
    assert bronze_session.query(AccessObservation).one().oa_status == "gold"


def test_dual_review_round_trip_preserves_raw_labels(gold_session, tmp_path):
    version_id = "v1"
    gold_session.add_all(
        [
            DatasetArticle(dataset_version_id=version_id, doi="10.1/a", sources=[]),
            DatasetArticle(dataset_version_id=version_id, doi="10.1/b", sources=[]),
        ]
    )
    protocol = ensure_protocol(
        gold_session,
        workflow="screening",
        protocol_version="1",
        instructions="Include power-distribution planning studies.",
    )
    assert (
        assign_dataset_articles(
            gold_session,
            workflow="screening",
            dataset_version_id=version_id,
            protocol=protocol,
            reviewer_ids=["reviewer-b", "reviewer-a"],
        )
        == 4
    )

    output = tmp_path / "reviewer-a.csv"
    assert (
        export_assignments(
            gold_session,
            workflow="screening",
            dataset_version_id=version_id,
            reviewer_id="reviewer-a",
            output_path=output,
        )
        == 2
    )
    records = list(csv.DictReader(output.open(encoding="utf-8")))
    for record in records:
        record["label"] = "include" if record["subject_id"] == "10.1/a" else "exclude"
        record["rationale"] = "reviewed"
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)

    assert (
        import_labels(
            gold_session,
            workflow="screening",
            reviewer_id="reviewer-a",
            input_path=output,
        )
        == 2
    )
    first_hashes = {row.payload_sha256 for row in gold_session.query(ReviewLabel).all()}
    assert len(first_hashes) == 2

    import_labels(
        gold_session,
        workflow="screening",
        reviewer_id="reviewer-a",
        input_path=output,
    )
    assert gold_session.query(ReviewLabel).count() == 4
    assert {row.label_revision for row in gold_session.query(ReviewLabel)} == {1, 2}
    assert gold_session.query(ReviewAssignment).count() == 4

    adjudicate(
        gold_session,
        workflow="screening",
        dataset_version_id=version_id,
        subject_id="10.1/a",
        final_label="include",
        adjudicator_id="lead",
        rationale="Consensus after discussion",
    )
    assert gold_session.query(ReviewAdjudication).one().final_label == "include"
    assert gold_session.query(ReviewLabel).count() == 4


def test_subject_file_scopes_the_assignment_to_the_sample(tmp_path):
    """The evidence CSVs were a dead end before this: `reviews setup` could only
    assign the whole dataset version, so a stratified sample had nowhere to go."""
    import csv

    from lake_research_map.pipeline import _read_subject_file

    path = tmp_path / "sample.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("workflow", "subject_id", "label", "rationale"))
        writer.writeheader()
        for subject in ("10.1/b::/a/2.pdf", "10.1/a::/a/1.pdf", "10.1/b::/a/2.pdf", ""):
            writer.writerow(
                {"workflow": "pdf", "subject_id": subject, "label": "", "rationale": ""}
            )

    subjects = _read_subject_file(str(path))

    # Deduplicated, sorted, blanks dropped -- the same contract
    # `assign_dataset_articles` already applies to its subject list.
    assert subjects == ["10.1/a::/a/1.pdf", "10.1/b::/a/2.pdf"]


def test_subject_file_refuses_a_csv_it_cannot_use(tmp_path):
    import pytest

    from lake_research_map.pipeline import _read_subject_file

    wrong = tmp_path / "wrong.csv"
    wrong.write_text("doi,label\n10.1/a,include\n", encoding="utf-8")
    with pytest.raises(ValueError, match="subject_id"):
        _read_subject_file(str(wrong))

    empty = tmp_path / "empty.csv"
    empty.write_text("workflow,subject_id,label,rationale\n", encoding="utf-8")
    with pytest.raises(ValueError, match="no subject_id"):
        _read_subject_file(str(empty))
