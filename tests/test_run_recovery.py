"""WP-08: an abandoned run must stop reading as `running` on its own.

Until this landed, the only thing that ever cleared a killed run was someone
remembering to type `maintenance recover-stale`, so the execution history --
and the dashboard status that reads it -- claimed a run was in progress that
had not existed for days.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from sqlalchemy import select

from lake_research_map.db.gold_models import PipelineExecution, PipelineRun
from lake_research_map.pipeline import recover_abandoned_executions

NOW = datetime.now(UTC).replace(tzinfo=None)


def _execution(session, execution_id, *, status="running", started=None, heartbeat=None):
    session.add(
        PipelineExecution(
            execution_id=execution_id,
            requested_stage="all",
            trigger="cli",
            status=status,
            started_at=started or NOW,
            heartbeat_at=heartbeat,
        )
    )
    session.commit()


def _run(session, execution_id, *, stage="embed", status="running", started=None):
    # `_begin_stage` writes a live row with `finished_at == started_at` and a
    # zero duration, because both columns are NOT NULL; the fixture matches it.
    begin = started or NOW
    session.add(
        PipelineRun(
            execution_id=execution_id,
            stage=stage,
            status=status,
            started_at=begin,
            finished_at=begin,
            duration_seconds=0.0,
        )
    )
    session.commit()


def test_holding_the_lock_makes_every_running_row_recoverable(gold_session):
    _execution(gold_session, "abandoned", heartbeat=NOW)
    _run(gold_session, "abandoned")

    # No cutoff: the caller is asserting exclusivity, so a fresh heartbeat is
    # not evidence of life -- the process that wrote it is gone.
    recovered = recover_abandoned_executions(gold_session)

    assert recovered == ["abandoned"]
    execution = gold_session.get(PipelineExecution, "abandoned")
    assert execution.status == "error"
    assert execution.finished_at is not None
    run_row = gold_session.scalar(select(PipelineRun))
    assert run_row.status == "error"
    assert run_row.duration_seconds >= 0.0


def test_a_cutoff_spares_a_run_that_is_still_beating(gold_session):
    _execution(gold_session, "alive", heartbeat=NOW)
    _execution(gold_session, "stale", started=NOW - timedelta(hours=3))

    recovered = recover_abandoned_executions(gold_session, cutoff=NOW - timedelta(minutes=30))

    # `stale` has no heartbeat at all, which is the state a process killed
    # before its first batch leaves behind; it must count as abandoned.
    assert recovered == ["stale"]
    assert gold_session.get(PipelineExecution, "alive").status == "running"


def test_a_finished_execution_is_never_rewritten(gold_session):
    _execution(gold_session, "done", status="success", started=NOW - timedelta(days=2))
    _run(gold_session, "done", status="success")

    assert recover_abandoned_executions(gold_session) == []
    assert gold_session.get(PipelineExecution, "done").status == "success"
    assert gold_session.scalar(select(PipelineRun)).status == "success"


def test_recovery_reports_the_reason_it_was_triggered_by(gold_session):
    _execution(gold_session, "abandoned")

    recover_abandoned_executions(gold_session, reason="recovered automatically")

    assert gold_session.get(PipelineExecution, "abandoned").error_message == (
        "recovered automatically"
    )


def test_the_status_command_reports_progress_and_the_disagreement_queue(gold_session, capsys):
    """WP-09's agreement report had no reader outside the dashboard upload."""
    import argparse

    from sqlalchemy import select as sa_select

    from lake_research_map import pipeline as pipeline_module
    from lake_research_map.db.gold_models import ReviewAssignment, ReviewLabel
    from lake_research_map.transform.review_workflows import (
        assign_dataset_articles,
        ensure_protocol,
    )

    version = "v-status"
    protocol = ensure_protocol(
        gold_session,
        workflow="screening",
        protocol_version="v1",
        instructions="criteria",
    )
    assign_dataset_articles(
        gold_session,
        workflow="screening",
        dataset_version_id=version,
        protocol=protocol,
        reviewer_ids=["ana", "bruno"],
        subject_ids=["10.1000/a", "10.1000/b"],
    )
    for subject, reviewer, label in (
        ("10.1000/a", "ana", "include"),
        ("10.1000/a", "bruno", "exclude"),
        ("10.1000/b", "ana", "include"),
    ):
        assignment = gold_session.scalar(
            sa_select(ReviewAssignment).where(
                ReviewAssignment.subject_id == subject,
                ReviewAssignment.reviewer_id == reviewer,
                ReviewAssignment.dataset_version_id == version,
            )
        )
        gold_session.add(
            ReviewLabel(
                assignment_id=assignment.id,
                label_revision=1,
                label=label,
                payload_sha256="0" * 64,
            )
        )
    gold_session.commit()

    args = argparse.Namespace(workflow="screening", version_id=version, queue_limit=10)
    pipeline_module._print_review_status(gold_session, args)

    printed = capsys.readouterr().out
    assert "Assignments: 4" in printed
    assert "Labelled:    3 (1 outstanding)" in printed
    assert "ana: 2 labelled, 0 outstanding" in printed
    assert "bruno: 1 labelled, 1 outstanding" in printed
    assert "Label set SHA-256:" in printed
    # One clean disagreement, which is exactly what adjudication is for.
    assert "Disagreement queue" in printed
    assert "10.1000/a [disagreement]" in printed
