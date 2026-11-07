"""`reviews setup|export|import|adjudicate|approve|status|calibrate`: human evidence."""

from __future__ import annotations

import argparse
import json

from sqlalchemy import select

from db.bootstrap import bootstrap
from db.engines import get_session
from runs import _layer_sessions


def configure(subparsers) -> None:
    reviews = subparsers.add_parser("reviews", help="Manage persistent human-evidence workflows")
    actions = reviews.add_subparsers(dest="review_action", required=True)
    setup = actions.add_parser("setup", help="Create protocol and dual-review assignments")
    setup.add_argument("--workflow", required=True)
    setup.add_argument("--version-id", required=True)
    setup.add_argument("--protocol-version", required=True)
    setup.add_argument("--instructions-file", required=True)
    setup.add_argument("--reviewer", action="append", required=True)
    setup.add_argument(
        "--subject-file",
        default=None,
        help=(
            "CSV from `evidence ...` whose subject_id column scopes the assignment. "
            "Without it every article in the dataset version is assigned, which is "
            "rarely what a stratified review wants."
        ),
    )
    export = actions.add_parser("export", help="Export one reviewer's assignments")
    export.add_argument("--workflow", required=True)
    export.add_argument("--version-id", required=True)
    export.add_argument("--reviewer", required=True)
    export.add_argument("--output", required=True)
    import_parser = actions.add_parser("import", help="Import append-only reviewer labels")
    import_parser.add_argument("--workflow", required=True)
    import_parser.add_argument("--reviewer", required=True)
    import_parser.add_argument("--input", required=True)
    adjudicate_parser = actions.add_parser("adjudicate", help="Record a final adjudication")
    adjudicate_parser.add_argument("--workflow", required=True)
    adjudicate_parser.add_argument("--version-id", required=True)
    adjudicate_parser.add_argument("--subject-id", required=True)
    adjudicate_parser.add_argument("--label", required=True)
    adjudicate_parser.add_argument("--adjudicator", required=True)
    adjudicate_parser.add_argument("--reason", required=True)
    approve = actions.add_parser("approve", help="Persist an evaluated model decision")
    approve.add_argument("--workflow", required=True)
    approve.add_argument("--version-id", required=True)
    approve.add_argument("--model-sha256", required=True)
    approve.add_argument("--label-set-sha256", required=True)
    approve.add_argument("--parameters-json", required=True)
    approve.add_argument("--metrics-json", required=True)
    approve.add_argument("--approved-by", required=True)
    approve.add_argument("--reject", action="store_true")
    status = actions.add_parser(
        "status", help="Report labelling progress, agreement, and the disagreement queue"
    )
    status.add_argument("--workflow", required=True)
    status.add_argument("--version-id", required=True)
    status.add_argument(
        "--queue-limit",
        type=int,
        default=20,
        help="How many unresolved subjects to list (0 lists none)",
    )
    calibrate = actions.add_parser(
        "calibrate",
        help="Evaluate the screening threshold against the labels stored in the database",
    )
    calibrate.add_argument("--version-id", required=True)
    calibrate.add_argument(
        "--min-recall",
        type=float,
        default=0.98,
        help="Sensitivity the selected threshold must reach on the calibration split",
    )
    calibrate.add_argument(
        "--output-json", help="Write the full result, including both digests, to this path"
    )


def _read_subject_file(path: str) -> list[str]:
    """Read the `subject_id` column of a CSV written by the `evidence` commands.

    Without this the review CLI could only ever assign the whole dataset
    version, so a stratified sample had nowhere to go: the generators wrote a
    CSV that nothing consumed.
    """
    import csv
    from pathlib import Path

    handle = Path(path)
    with handle.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None or "subject_id" not in reader.fieldnames:
            raise ValueError(f"{path}: expected a `subject_id` column, got {reader.fieldnames}")
        subjects = [
            row["subject_id"].strip() for row in reader if (row.get("subject_id") or "").strip()
        ]
    if not subjects:
        raise ValueError(f"{path}: no subject_id values to assign")
    return sorted(dict.fromkeys(subjects))


def _print_review_status(session, args: argparse.Namespace) -> None:
    """Report a review round from the durable tables rather than an upload.

    The agreement report was only ever reachable by uploading a CSV to the
    dashboard, so the labels the CLI persists had no reader. Progress is
    broken out per reviewer because a round where one reviewer finished and
    the other has not started still produces labels, and a single completion
    figure hides exactly the case that makes kappa meaningless.
    """
    import pandas as pd

    from transform.review_workflows import (
        assignment_progress,
        collect_labels,
    )
    from transform.screening_calibration import (
        label_set_digest,
        resolve_review_consensus,
        reviewer_agreement,
    )

    progress = assignment_progress(
        session, workflow=args.workflow, dataset_version_id=args.version_id
    )
    print(f"Assignments: {progress['assignments']}")
    print(f"Labelled:    {progress['labelled']} ({progress['outstanding']} outstanding)")
    for reviewer, count in progress["labelled_by_reviewer"].items():
        pending = progress["outstanding_by_reviewer"].get(reviewer, 0)
        print(f"  {reviewer}: {count} labelled, {pending} outstanding")

    rows = collect_labels(session, workflow=args.workflow, dataset_version_id=args.version_id)
    if not rows:
        print("No labels have been imported for this workflow and version yet.")
        return
    labels = pd.DataFrame(rows)
    print(f"Label set SHA-256: {label_set_digest(labels)}")

    agreement = reviewer_agreement(labels)
    if agreement.empty:
        print("Agreement needs at least two reviewers with binary decisions in common.")
    else:
        for row in agreement.itertuples(index=False):
            kappa = "n/a" if pd.isna(row.kappa) else f"{row.kappa:.3f}"
            raw = "n/a" if pd.isna(row.raw_agreement) else f"{row.raw_agreement:.1%}"
            print(
                f"  {row.reviewer_a} vs {row.reviewer_b}: n={row.n_overlap} "
                f"agreement={raw} kappa={kappa} ({row.status})"
            )

    resolved = resolve_review_consensus(labels)
    unresolved = resolved[~resolved["resolved"]]
    print(f"Resolved: {int(resolved['resolved'].sum())}, unresolved: {len(unresolved)}")
    if args.queue_limit and not unresolved.empty:
        print("Disagreement queue (adjudicate these):")
        for row in unresolved.head(args.queue_limit).itertuples(index=False):
            print(f"  {row.doi} [{row.resolution}] reviewers={row.reviewer_count}")


def _print_screening_calibration(session, args: argparse.Namespace) -> None:
    """Calibrate the screening margin against the labels stored in the database.

    Nothing is applied: the command reports the evidence and the two digests
    that `reviews approve` needs, and the approval stays a human decision.
    That separation is the point of the package -- a threshold that excludes
    work is not allowed to select itself.
    """
    import pandas as pd

    from db.gold_models import DatasetSemantics
    from transform.embeddings import EMBED_MODEL_NAME
    from transform.review_workflows import collect_labels
    from transform.screening_calibration import (
        calibrate_screening_threshold,
        label_set_digest,
        resolve_review_consensus,
        screening_model_digest,
        validate_review_labels,
    )
    from transform.semantics import ANCHOR_TEXT, OFF_ANCHOR_TEXT

    rows = collect_labels(session, workflow="screening", dataset_version_id=args.version_id)
    if not rows:
        raise ValueError(
            "no screening labels are stored for this version; import reviewer decisions first"
        )

    scored = pd.DataFrame(
        session.execute(
            select(
                DatasetSemantics.doi,
                DatasetSemantics.relevance_score,
                DatasetSemantics.offtopic_score,
            ).where(DatasetSemantics.dataset_version_id == args.version_id)
        ).all(),
        columns=["doi", "relevance_score", "offtopic_score"],
    )
    if scored.empty or scored["offtopic_score"].isna().all():
        raise ValueError(
            "this version carries no contrastive margin; re-run `--stage semantic` before "
            "calibrating, because a threshold on the single anchor is not what is approved"
        )
    scored["relevance_margin"] = scored["relevance_score"] - scored["offtopic_score"]

    labels, issues = validate_review_labels(
        pd.DataFrame(rows), known_dois=set(scored["doi"].dropna().astype(str))
    )
    # A `reject::` subject is a record dropped before it ever had a margin, so
    # it cannot take part in a threshold evaluation. It is reported rather than
    # dropped in silence: a round whose labels mostly vanish at this join looks
    # identical to a round that was never labelled.
    if not issues.empty:
        for row in issues.itertuples(index=False):
            print(f"  [{row.severity}] {row.code}: {row.doi or ''} {row.detail}")
    if labels.empty:
        raise ValueError("no stored label survived validation against the active margins")

    resolved = resolve_review_consensus(labels)
    result = calibrate_screening_threshold(resolved, scored, min_recall=args.min_recall)
    labels_digest = label_set_digest(labels)

    print(f"Labels used: {len(labels)} rows over {resolved['doi'].nunique()} subjects")
    print(f"Label set SHA-256: {labels_digest}")
    if not result["valid"]:
        print(f"Not calibratable yet: {result['reason']} (resolved={result['n_resolved']})")
        print(
            "The calibration needs at least 40 resolved subjects with 10 in each class. "
            "Automatic exclusion stays disabled until then, which is the documented "
            "failure mode rather than an error."
        )
        return

    threshold = float(result["threshold"])
    model_digest = screening_model_digest(
        embed_model=EMBED_MODEL_NAME,
        anchor_text=ANCHOR_TEXT,
        off_anchor_text=OFF_ANCHOR_TEXT,
        threshold=threshold,
        min_recall=args.min_recall,
    )
    metrics = result["metrics"]
    intervals = result["confidence_intervals"]
    print(f"Threshold: margin >= {threshold:+.4f}")
    print(f"Model SHA-256: {model_digest}")
    print(f"Fitted on {result['n_calibration']}, evaluated on {result['n_holdout']} held out.")
    for name in ("recall", "specificity", "precision", "f2", "workload_reduction"):
        low, high = intervals[name]
        print(f"  {name}: {metrics[name]:.1%} (95% CI {low:.1%}-{high:.1%})")
    print(
        "Nothing was applied. Record the decision with `reviews approve --workflow screening "
        f"--version-id {args.version_id} --model-sha256 {model_digest} "
        f"--label-set-sha256 {labels_digest} ...`."
    )

    if args.output_json:
        from pathlib import Path

        payload = {
            "dataset_version_id": args.version_id,
            "threshold": threshold,
            "min_recall": args.min_recall,
            "model_sha256": model_digest,
            "label_set_sha256": labels_digest,
            "metrics": metrics,
            "confidence_intervals": {k: list(v) for k, v in intervals.items()},
            "n_calibration": result["n_calibration"],
            "n_holdout": result["n_holdout"],
            "n_resolved": result["n_resolved"],
        }
        path = Path(args.output_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        print(f"Wrote {path}.")


def run(args: argparse.Namespace) -> None:
    from pathlib import Path

    from transform.review_workflows import (
        adjudicate,
        approve_model,
        assign_dataset_articles,
        ensure_protocol,
        export_assignments,
        import_labels,
    )

    bootstrap()
    session = get_session("gold")
    try:
        if args.review_action == "setup":
            instructions = Path(args.instructions_file).read_text(encoding="utf-8")
            protocol = ensure_protocol(
                session,
                workflow=args.workflow,
                protocol_version=args.protocol_version,
                instructions=instructions,
            )
            subject_ids = _read_subject_file(args.subject_file) if args.subject_file else None
            count = assign_dataset_articles(
                session,
                workflow=args.workflow,
                dataset_version_id=args.version_id,
                protocol=protocol,
                reviewer_ids=args.reviewer,
                subject_ids=subject_ids,
            )
            scope = f"{len(subject_ids)} subjects" if subject_ids else "every article"
            print(f"Created {count} assignments over {scope}.")
        elif args.review_action == "export":
            # Each workflow reviews a different object, so the context comes
            # from a different layer: rejected records live in Bronze, author
            # spellings in Silver, PDF filenames in Raw. They are opened here
            # and closed when the block exits, because transform builders take
            # sessions and never open their own. `_layer_sessions` also closes
            # the first ones if a later open fails, which the dict literal
            # this replaced did not.
            with _layer_sessions("silver", "bronze", "raw") as (silver, bronze, raw):
                count = export_assignments(
                    session,
                    workflow=args.workflow,
                    dataset_version_id=args.version_id,
                    reviewer_id=args.reviewer,
                    output_path=Path(args.output),
                    silver_session=silver,
                    bronze_session=bronze,
                    raw_session=raw,
                )
            print(f"Exported {count} assignments to {args.output}.")
        elif args.review_action == "import":
            count = import_labels(
                session,
                workflow=args.workflow,
                reviewer_id=args.reviewer,
                input_path=Path(args.input),
            )
            print(f"Imported {count} immutable labels.")
        elif args.review_action == "adjudicate":
            adjudicate(
                session,
                workflow=args.workflow,
                dataset_version_id=args.version_id,
                subject_id=args.subject_id,
                final_label=args.label,
                adjudicator_id=args.adjudicator,
                rationale=args.reason,
            )
            print(f"Adjudicated {args.subject_id} as {args.label}.")
        elif args.review_action == "status":
            _print_review_status(session, args)
        elif args.review_action == "calibrate":
            _print_screening_calibration(session, args)
        else:
            approve_model(
                session,
                workflow=args.workflow,
                dataset_version_id=args.version_id,
                model_sha256=args.model_sha256,
                label_set_sha256=args.label_set_sha256,
                parameters=json.loads(args.parameters_json),
                metrics=json.loads(args.metrics_json),
                approved_by=args.approved_by,
                approved=not args.reject,
            )
            print("Recorded model approval decision.")
    finally:
        session.close()
