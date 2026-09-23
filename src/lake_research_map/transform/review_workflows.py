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
    gold_session: Session | None = None,
    silver_session: Session | None = None,
    bronze_session: Session | None = None,
    raw_session: Session | None = None,
) -> int:
    """Write one reviewer's queue, with the evidence needed to judge it.

    The extra sessions are optional and the export works without them, but a
    queue exported without any of them is the one this function used to
    produce: `assignment_id` plus an opaque `subject_id`, which asks a human
    to decide whether `reject::ieee::bib:03f7a342cbc9:6761637` belonged in the
    corpus. Four packages sat blocked on human labels that nobody could
    physically supply.

    `label` and `rationale` stay last so the columns a reviewer types into are
    the rightmost ones in a spreadsheet, after everything they must read.
    """
    assignments = session.scalars(
        select(ReviewAssignment)
        .where(
            ReviewAssignment.workflow == workflow,
            ReviewAssignment.dataset_version_id == dataset_version_id,
            ReviewAssignment.reviewer_id == reviewer_id,
        )
        .order_by(ReviewAssignment.subject_id)
    ).all()

    from lake_research_map.transform.evidence_samples import SUBJECT_COLUMNS, describe_subjects

    context_columns = SUBJECT_COLUMNS.get(workflow, ())
    context = describe_subjects(
        workflow,
        [assignment.subject_id for assignment in assignments],
        dataset_version_id=dataset_version_id,
        gold_session=gold_session if gold_session is not None else session,
        silver_session=silver_session,
        bronze_session=bronze_session,
        raw_session=raw_session,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("assignment_id", "subject_id", *context_columns, "label", "rationale"),
        )
        writer.writeheader()
        for assignment in assignments:
            described = context.get(assignment.subject_id, {})
            writer.writerow(
                {
                    "assignment_id": assignment.id,
                    "subject_id": assignment.subject_id,
                    **{column: described.get(column, "") for column in context_columns},
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


# Reserved reviewer id. `screening_calibration.resolve_review_consensus`
# already treats a row under this name as the decision that outranks a
# disagreement, so an adjudication reaches the calibrator as an extra row
# rather than by overwriting the raw labels it was meant to preserve.
ADJUDICATED_REVIEWER = "adjudicated"


def collect_labels(
    session: Session,
    *,
    workflow: str,
    dataset_version_id: str,
) -> list[dict[str, str]]:
    """Return the durable labels in the long form the calibration reads.

    This is the join the two halves of the workflow were missing: `import`
    wrote reviewer decisions to the database, while calibration only ever saw
    a CSV uploaded in the browser, so a persisted label could never reach the
    threshold it exists to validate.

    Only the highest revision of each assignment is returned. The table is
    append-only, so a reviewer who corrects a decision adds a row instead of
    editing one, and every earlier revision is history rather than a second
    opinion.
    """
    if workflow not in WORKFLOW_LABELS:
        raise ValueError(f"unknown workflow {workflow!r}")

    label_rows = session.execute(
        select(
            ReviewLabel.assignment_id,
            ReviewLabel.label,
            ReviewAssignment.subject_id,
            ReviewAssignment.reviewer_id,
            ReviewProtocol.protocol_version,
        )
        .join(ReviewAssignment, ReviewAssignment.id == ReviewLabel.assignment_id)
        .join(ReviewProtocol, ReviewProtocol.id == ReviewAssignment.protocol_id)
        .where(
            ReviewAssignment.workflow == workflow,
            ReviewAssignment.dataset_version_id == dataset_version_id,
        )
        .order_by(ReviewLabel.assignment_id, ReviewLabel.label_revision.desc())
    ).all()

    rows: list[dict[str, str]] = []
    seen: set[int] = set()
    protocol_by_subject: dict[str, str] = {}
    for assignment_id, label, subject_id, reviewer_id, protocol_version in label_rows:
        protocol_by_subject.setdefault(subject_id, protocol_version)
        if assignment_id in seen:
            continue
        seen.add(assignment_id)
        rows.append(
            {
                "doi": subject_id,
                "subject_id": subject_id,
                "reviewer": reviewer_id,
                "manual_label": label,
                "protocol_version": protocol_version,
            }
        )

    for row in session.scalars(
        select(ReviewAdjudication).where(
            ReviewAdjudication.workflow == workflow,
            ReviewAdjudication.dataset_version_id == dataset_version_id,
        )
    ):
        rows.append(
            {
                "doi": row.subject_id,
                "subject_id": row.subject_id,
                "reviewer": ADJUDICATED_REVIEWER,
                "manual_label": row.final_label,
                "protocol_version": protocol_by_subject.get(row.subject_id, "unversioned"),
            }
        )

    return sorted(rows, key=lambda row: (row["subject_id"], row["reviewer"]))


def assignment_progress(
    session: Session,
    *,
    workflow: str,
    dataset_version_id: str,
) -> dict[str, int]:
    """Count how much of a review round has actually been labelled.

    Reported per reviewer as well as in total, because a round where one
    reviewer finished and the other has not started produces plenty of labels
    and zero agreement, and a single completion percentage hides that.
    """
    assignments = session.execute(
        select(ReviewAssignment.id, ReviewAssignment.reviewer_id).where(
            ReviewAssignment.workflow == workflow,
            ReviewAssignment.dataset_version_id == dataset_version_id,
        )
    ).all()
    labelled = set(
        session.scalars(
            select(ReviewLabel.assignment_id).where(
                ReviewLabel.assignment_id.in_([row.id for row in assignments])
            )
        ).all()
        if assignments
        else []
    )
    per_reviewer: dict[str, int] = {}
    outstanding: dict[str, int] = {}
    for assignment_id, reviewer_id in assignments:
        if assignment_id in labelled:
            per_reviewer[reviewer_id] = per_reviewer.get(reviewer_id, 0) + 1
        else:
            outstanding[reviewer_id] = outstanding.get(reviewer_id, 0) + 1
    return {
        "assignments": len(assignments),
        "labelled": len(labelled),
        "outstanding": len(assignments) - len(labelled),
        "labelled_by_reviewer": dict(sorted(per_reviewer.items())),
        "outstanding_by_reviewer": dict(sorted(outstanding.items())),
    }
