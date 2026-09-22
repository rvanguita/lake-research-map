"""An exported review queue must carry the evidence needed to judge it.

The export wrote `assignment_id` and an opaque `subject_id` and nothing else,
which is why `review/rene-pdf.csv` sat on disk containing only its header: the
reviewer opened it, found `10.1002/…::data/.lake_research_map/objects/ae/ae97…`
and had nothing to compare. Four packages were recorded as blocked on human
labels that no human could physically supply.
"""

from __future__ import annotations

import csv
from pathlib import Path

from lake_research_map.db.bronze_models import Article as BronzeArticle
from lake_research_map.db.gold_models import DatasetArticle
from lake_research_map.db.raw_models import PdfFile
from lake_research_map.db.silver_models import Article as SilverArticle
from lake_research_map.transform.evidence_samples import SUBJECT_COLUMNS, describe_subjects
from lake_research_map.transform.review_workflows import (
    assign_dataset_articles,
    ensure_protocol,
    export_assignments,
)

VERSION = "v-export"


def test_screening_subject_resolves_to_the_rejected_record(bronze_session):
    bronze_session.add(
        BronzeArticle(
            source="ieee",
            source_id="bib:abc:991",
            record_type="inproceedings",
            title="Regional load profile subclasses",
            year=2013,
            venue="ICUE 2013",
            abstract="Distribution network planning needs consolidated regional views.",
        )
    )
    bronze_session.commit()

    described = describe_subjects(
        "screening",
        ["reject::ieee::bib:abc:991"],
        dataset_version_id=VERSION,
        bronze_session=bronze_session,
    )

    row = described["reject::ieee::bib:abc:991"]
    assert row["title"] == "Regional load profile subclasses"
    assert row["source"] == "ieee"
    assert row["year"] == "2013"
    assert "consolidated regional views" in row["abstract"]


def test_pdf_subject_resolves_the_content_addressed_blob_to_a_filename(gold_session, raw_session):
    blob = "data/.lake_research_map/objects/ae/ae97ce4b"
    gold_session.add(
        DatasetArticle(
            dataset_version_id=VERSION,
            doi="10.1002/x.ch3",
            title="Distribution System Planning",
            year=2019,
        )
    )
    gold_session.commit()
    raw_session.add(
        PdfFile(
            filename="Distribution System Planning.pdf",
            path="data/articles/Distribution System Planning.pdf",
            sha256="0" * 64,
            size_bytes=1,
            archive_path=blob,
        )
    )
    raw_session.commit()

    described = describe_subjects(
        "pdf",
        [f"10.1002/x.ch3::{blob}"],
        dataset_version_id=VERSION,
        gold_session=gold_session,
        raw_session=raw_session,
    )

    row = described[f"10.1002/x.ch3::{blob}"]
    # Both halves of the comparison must be readable, or the reviewer cannot
    # answer whether this PDF is this article.
    assert row["article_title"] == "Distribution System Planning"
    assert row["pdf_file"] == "Distribution System Planning.pdf"


def test_retrieval_subject_resolves_the_query_text_not_just_its_id(gold_session):
    gold_session.add(
        DatasetArticle(
            dataset_version_id=VERSION,
            doi="10.1016/j.est.1",
            title="Optimal siting and sizing of DG",
            abstract="An integrated model for optimal siting.",
        )
    )
    gold_session.commit()

    described = describe_subjects(
        "retrieval",
        ["q01::10.1016/j.est.1"],
        dataset_version_id=VERSION,
        gold_session=gold_session,
    )

    row = described["q01::10.1016/j.est.1"]
    assert row["query"].startswith("optimal placement and sizing")
    assert row["article_title"] == "Optimal siting and sizing of DG"


def test_taxonomy_subject_keeps_the_class_beside_the_abstract(gold_session):
    gold_session.add(
        DatasetArticle(
            dataset_version_id=VERSION,
            doi="10.1016/j.ap.1",
            title="Distributed energy storage planning",
            abstract="Integration of high-penetration distributed generators.",
        )
    )
    gold_session.commit()

    subject = "Conical / Convex Relaxation (SOCP)::10.1016/j.ap.1"
    row = describe_subjects(
        "taxonomy", [subject], dataset_version_id=VERSION, gold_session=gold_session
    )[subject]

    assert row["taxonomy_class"] == "Conical / Convex Relaxation (SOCP)"
    assert "high-penetration" in row["abstract"]


def test_author_subject_carries_titles_for_both_spellings(silver_session):
    silver_session.add_all(
        [
            SilverArticle(
                doi="10.1/a",
                record_type="article",
                title="Smart grid diagnostics",
                authors=["A Abaide"],
            ),
            SilverArticle(
                doi="10.1/b",
                record_type="article",
                title="Impacts of distributed generation",
                authors=["A da Rosa Abaide"],
            ),
        ]
    )
    silver_session.commit()

    subject = "a abaide||a da rosa abaide"
    row = describe_subjects(
        "author", [subject], dataset_version_id=VERSION, silver_session=silver_session
    )[subject]

    # The names alone never settle a homonym; what decides it is what each
    # spelling publishes.
    assert row["variant_a"] == "a abaide"
    assert row["variant_b"] == "a da rosa abaide"
    assert "Smart grid diagnostics" in row["papers_a"]
    assert "Impacts of distributed generation" in row["papers_b"]


def test_export_writes_the_context_columns_with_label_last(gold_session, bronze_session, tmp_path):
    protocol = ensure_protocol(
        gold_session, workflow="screening", protocol_version="v2", instructions="criteria"
    )
    assign_dataset_articles(
        gold_session,
        workflow="screening",
        dataset_version_id=VERSION,
        protocol=protocol,
        reviewer_ids=["rene", "revisor2"],
        subject_ids=["reject::ieee::bib:abc:991"],
    )
    bronze_session.add(
        BronzeArticle(
            source="ieee",
            source_id="bib:abc:991",
            record_type="article",
            title="A rejected record",
            abstract="Its abstract.",
        )
    )
    bronze_session.commit()

    output = tmp_path / "rene-screening.csv"
    count = export_assignments(
        gold_session,
        workflow="screening",
        dataset_version_id=VERSION,
        reviewer_id="rene",
        output_path=output,
        bronze_session=bronze_session,
    )

    assert count == 1
    rows = list(csv.DictReader(output.open()))
    assert rows[0]["title"] == "A rejected record"
    header = list(rows[0])
    assert header[:2] == ["assignment_id", "subject_id"]
    # What the reviewer types into stays rightmost, after everything they read.
    assert header[-2:] == ["label", "rationale"]
    assert set(SUBJECT_COLUMNS["screening"]).issubset(header)


def test_export_still_works_without_any_context_session(gold_session, tmp_path):
    """A missing layer must leave blanks, never fail the whole queue."""
    protocol = ensure_protocol(
        gold_session, workflow="screening", protocol_version="v2", instructions="criteria"
    )
    assign_dataset_articles(
        gold_session,
        workflow="screening",
        dataset_version_id=VERSION,
        protocol=protocol,
        reviewer_ids=["rene", "revisor2"],
        subject_ids=["reject::ieee::missing"],
    )

    output = Path(tmp_path / "queue.csv")
    assert (
        export_assignments(
            gold_session,
            workflow="screening",
            dataset_version_id=VERSION,
            reviewer_id="rene",
            output_path=output,
        )
        == 1
    )
    row = next(iter(csv.DictReader(output.open())))
    assert row["subject_id"] == "reject::ieee::missing"
    assert row["title"] == ""


def test_an_enriched_queue_still_imports(gold_session, bronze_session, tmp_path):
    """The added columns must not break the path the reviewer's file comes back on."""
    from lake_research_map.transform.review_workflows import collect_labels, import_labels

    protocol = ensure_protocol(
        gold_session, workflow="screening", protocol_version="v2", instructions="criteria"
    )
    assign_dataset_articles(
        gold_session,
        workflow="screening",
        dataset_version_id=VERSION,
        protocol=protocol,
        reviewer_ids=["rene", "revisor2"],
        subject_ids=["reject::ieee::bib:abc:991"],
    )
    bronze_session.add(
        BronzeArticle(
            source="ieee",
            source_id="bib:abc:991",
            record_type="article",
            title="A rejected record",
            abstract="Its abstract.",
        )
    )
    bronze_session.commit()

    queue = tmp_path / "rene-screening.csv"
    export_assignments(
        gold_session,
        workflow="screening",
        dataset_version_id=VERSION,
        reviewer_id="rene",
        output_path=queue,
        bronze_session=bronze_session,
    )

    # Fill it in the way a reviewer would: type into `label`, leave the
    # context columns untouched.
    rows = list(csv.DictReader(queue.open()))
    rows[0]["label"] = "include"
    rows[0]["rationale"] = "in scope on title"
    with queue.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    assert (
        import_labels(gold_session, workflow="screening", reviewer_id="rene", input_path=queue) == 1
    )

    stored = collect_labels(gold_session, workflow="screening", dataset_version_id=VERSION)
    assert stored == [
        {
            "doi": "reject::ieee::bib:abc:991",
            "subject_id": "reject::ieee::bib:abc:991",
            "reviewer": "rene",
            "manual_label": "include",
            "protocol_version": "v2",
        }
    ]
