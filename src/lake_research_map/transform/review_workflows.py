"""Persistent, dataset-versioned human review workflows.

The module deliberately contains no Streamlit calls.  Raw labels are
append-only; consensus and adjudication are derived without overwriting the
reviewers' evidence.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from lake_research_map.db.gold_models import (
    DatasetArticle,
    ModelApproval,
    ReviewAdjudication,
    ReviewAssignment,
    ReviewLabel,
    ReviewProtocol,
)

WORKFLOW_LABELS = {
    "screening": frozenset({"include", "exclude", "uncertain"}),
    "author": frozenset({"same", "different", "uncertain"}),
    "taxonomy": frozenset({"present", "absent", "ambiguous"}),
    "retrieval": frozenset({"relevant", "not_relevant", "uncertain"}),
    "pdf": frozenset({"match", "mismatch", "uncertain"}),
}


def ensure_protocol(
    session: Session,
    *,
    workflow: str,
    protocol_version: str,
    instructions: str,
) -> ReviewProtocol:
    if workflow not in WORKFLOW_LABELS:
        raise ValueError(f"unknown workflow {workflow!r}")
    digest = hashlib.sha256(instructions.encode("utf-8")).hexdigest()
    protocol = session.scalar(
        select(ReviewProtocol).where(
            ReviewProtocol.workflow == workflow,
            ReviewProtocol.protocol_version == protocol_version,
        )
    )
    if protocol is None:
        protocol = ReviewProtocol(
            workflow=workflow,
            protocol_version=protocol_version,
            instructions_sha256=digest,
            instructions=instructions,
        )
        session.add(protocol)
        session.flush()
    elif protocol.instructions_sha256 != digest:
        raise ValueError("a protocol version is immutable; use a new protocol version")
    return protocol


def assign_dataset_articles(
    session: Session,
    *,
    workflow: str,
    dataset_version_id: str,
    protocol: ReviewProtocol,
    reviewer_ids: list[str],
    subject_ids: list[str] | None = None,
) -> int:
    reviewers = sorted({value.strip() for value in reviewer_ids if value.strip()})
    if len(reviewers) < 2:
        raise ValueError("at least two distinct reviewers are required")
    subjects = (
        subject_ids
        or session.scalars(
            select(DatasetArticle.doi)
            .where(DatasetArticle.dataset_version_id == dataset_version_id)
            .order_by(DatasetArticle.doi)
        ).all()
    )
    inserted = 0
    for subject_id in sorted(set(subjects)):
        for reviewer_id in reviewers:
            exists = session.scalar(
                select(ReviewAssignment.id).where(
                    ReviewAssignment.workflow == workflow,
                    ReviewAssignment.dataset_version_id == dataset_version_id,
                    ReviewAssignment.subject_id == subject_id,
                    ReviewAssignment.reviewer_id == reviewer_id,
                )
            )
            if exists is None:
                session.add(
                    ReviewAssignment(
                        workflow=workflow,
                        dataset_version_id=dataset_version_id,
                        protocol_id=protocol.id,
                        subject_id=subject_id,
                        reviewer_id=reviewer_id,
                    )
                )
                inserted += 1
    session.commit()
    return inserted


def export_assignments(
    session: Session,
    *,
    workflow: str,
    dataset_version_id: str,
    reviewer_id: str,
    output_path: Path,
) -> int:
    assignments = session.scalars(
        select(ReviewAssignment)
        .where(
            ReviewAssignment.workflow == workflow,
            ReviewAssignment.dataset_version_id == dataset_version_id,
            ReviewAssignment.reviewer_id == reviewer_id,
        )
        .order_by(ReviewAssignment.subject_id)
    ).all()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("assignment_id", "subject_id", "label", "rationale"),
        )
        writer.writeheader()
        for assignment in assignments:
            writer.writerow(
                {
                    "assignment_id": assignment.id,
                    "subject_id": assignment.subject_id,
                    "label": "",
                    "rationale": "",
                }
            )
    return len(assignments)


def import_labels(
    session: Session,
    *,
    workflow: str,
    reviewer_id: str,
    input_path: Path,
) -> int:
    allowed = WORKFLOW_LABELS.get(workflow)
    if allowed is None:
        raise ValueError(f"unknown workflow {workflow!r}")
    inserted = 0
    with input_path.open(newline="", encoding="utf-8") as handle:
        for record in csv.DictReader(handle):
            assignment_id = int(record["assignment_id"])
            assignment = session.get(ReviewAssignment, assignment_id)
            if (
                assignment is None
                or assignment.workflow != workflow
                or assignment.reviewer_id != reviewer_id
                or assignment.subject_id != record["subject_id"].strip()
            ):
                raise ValueError(f"assignment {assignment_id} does not match this import")
            label = record["label"].strip().casefold()
            if label not in allowed:
                raise ValueError(f"invalid {workflow} label {label!r}")
            rationale = record.get("rationale", "").strip() or None
            canonical = f"{assignment_id}\n{label}\n{rationale or ''}"
            revision = (
                session.scalar(
                    select(func.max(ReviewLabel.label_revision)).where(
                        ReviewLabel.assignment_id == assignment_id
                    )
                )
                or 0
            ) + 1
            session.add(
                ReviewLabel(
                    assignment_id=assignment_id,
                    label_revision=revision,
                    label=label,
                    rationale=rationale,
                    payload_sha256=hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
                )
            )
            inserted += 1
    session.commit()
    return inserted


def adjudicate(
    session: Session,
    *,
    workflow: str,
    dataset_version_id: str,
    subject_id: str,
    final_label: str,
    adjudicator_id: str,
    rationale: str,
) -> ReviewAdjudication:
    if final_label not in WORKFLOW_LABELS.get(workflow, ()):
        raise ValueError(f"invalid {workflow} label {final_label!r}")
    row = session.scalar(
        select(ReviewAdjudication).where(
            ReviewAdjudication.workflow == workflow,
            ReviewAdjudication.dataset_version_id == dataset_version_id,
            ReviewAdjudication.subject_id == subject_id,
        )
    )
    if row is None:
        row = ReviewAdjudication(
            workflow=workflow,
            dataset_version_id=dataset_version_id,
            subject_id=subject_id,
            final_label=final_label,
            adjudicator_id=adjudicator_id,
            rationale=rationale,
        )
        session.add(row)
    else:
        row.final_label = final_label
        row.adjudicator_id = adjudicator_id
        row.rationale = rationale
    session.commit()
    return row


def approve_model(
    session: Session,
    *,
    workflow: str,
    dataset_version_id: str,
    model_sha256: str,
    label_set_sha256: str,
    parameters: dict,
    metrics: dict,
    approved_by: str,
    approved: bool,
) -> ModelApproval:
    if len(model_sha256) != 64 or len(label_set_sha256) != 64:
        raise ValueError("model and label-set hashes must be SHA-256 hex digests")
    canonical = json.dumps(
        {"parameters": parameters, "metrics": metrics}, sort_keys=True, separators=(",", ":")
    )
    if not canonical:
        raise ValueError("approval payload cannot be empty")
    row = ModelApproval(
        workflow=workflow,
        dataset_version_id=dataset_version_id,
        model_sha256=model_sha256,
        parameters=parameters,
        metrics=metrics,
        label_set_sha256=label_set_sha256,
        approved=approved,
        approved_by=approved_by,
    )
    session.add(row)
    session.commit()
    return row
