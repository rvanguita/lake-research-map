"""CLI orchestrator for the medallion pipeline.

Usage:
    uv run lake-research-map --stage raw
    uv run lake-research-map --stage bronze
    uv run lake-research-map --stage silver
    uv run lake-research-map --stage gold
    uv run lake-research-map --stage embed
    uv run lake-research-map --stage semantic
    uv run lake-research-map --stage all       # default
    uv run lake-research-map duplicates list
    uv run lake-research-map duplicates merge --canonical-doi <doi> --duplicate-doi <doi> --reason <text>
    uv run lake-research-map duplicates keep --doi-a <doi> --doi-b <doi> --reason <text>
    uv run lake-research-map duplicates undo --doi-a <doi> --doi-b <doi>
"""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import os
import signal
import sys
import uuid
from datetime import UTC, datetime, timedelta

from sqlalchemy import func, select, text

from lake_research_map.db.bootstrap import bootstrap
from lake_research_map.db.engines import get_session
from lake_research_map.ingest.raw_bib import load_bib_entries
from lake_research_map.ingest.raw_config import load_configs
from lake_research_map.ingest.raw_csv import load_ieee_csv
from lake_research_map.ingest.raw_pdfs import load_pdf_inventory
from lake_research_map.ingest.reconciliation import (
    active_manifest,
    persist_snapshot,
    remove_absent_sources,
)
from lake_research_map.ingest.snapshots import ScannedSource, build_fingerprint, scan_sources
from lake_research_map.quality import (
    ContractViolation,
    assert_contract,
    bronze_contract,
    embed_contract,
    gold_contract,
    persist_results,
    raw_contract,
    semantic_contract,
    silver_contract,
)
from lake_research_map.transform.bronze_articles import build_bronze_articles
from lake_research_map.transform.silver_articles import build_silver_articles
from lake_research_map.transform.versioned_gold import (
    build_dataset_embeddings,
    build_dataset_gold,
    build_dataset_semantics,
    materialize_version,
)

STAGES = ("raw", "bronze", "silver", "gold", "embed", "semantic", "all")

logger = logging.getLogger(__name__)

PIPELINE_LOCK_NAME = "lake_research_map_pipeline"
PIPELINE_LOG_ENV = "LAKE_RESEARCH_MAP_LOG"


class PipelineBusyError(RuntimeError):
    """Raised when another process already owns the medallion pipeline lock."""


class PipelineInterruptedError(RuntimeError):
    """Raised for terminal/process signals so stage failure is persisted."""


def _install_signal_handlers() -> None:
    """Turn normal terminal shutdown signals into a recorded pipeline error."""

    def _handle(signum, _frame):
        name = signal.Signals(signum).name
        raise PipelineInterruptedError(
            f"pipeline interrupted by {name}; completed embedding batches are resumable"
        )

    for name in ("SIGINT", "SIGTERM", "SIGHUP"):
        signum = getattr(signal, name, None)
        if signum is not None:
            signal.signal(signum, _handle)


def _configure_logging() -> None:
    """Configure terminal and persistent logging for the CLI entry point."""
    root = logging.getLogger()
    if any(getattr(handler, "_lake_research_map", False) for handler in root.handlers):
        return

    log_path = os.environ.get(PIPELINE_LOG_ENV, str(scan_log_path()))
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stderr)]
    try:
        log_file = os.path.abspath(log_path)
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
    except OSError:
        # A read-only data directory must not prevent the pipeline from running.
        pass
    for handler in handlers:
        handler._lake_research_map = True
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=handlers,
    )


def scan_log_path():
    """Return the default log path without importing configuration at module load."""
    from lake_research_map.config import DATA_DIR

    return DATA_DIR / ".lake_research_map" / "pipeline.log"


@contextlib.contextmanager
def _pipeline_lock(execution_id: str):
    """Serialize mutating pipeline processes with a MySQL advisory lock.

    The lock is held by a dedicated Gold connection for the complete CLI
    invocation. SQLite fixtures intentionally bypass it because each test has
    an isolated in-memory database and the MySQL GET_LOCK function is absent.
    """
    lock_session = get_session("gold")
    acquired = False
    try:
        dialect = lock_session.get_bind().dialect.name
        if dialect == "mysql":
            acquired = bool(
                lock_session.execute(
                    text("SELECT GET_LOCK(:lock_name, 0)"),
                    {"lock_name": PIPELINE_LOCK_NAME},
                ).scalar()
            )
            if not acquired:
                raise PipelineBusyError(
                    "another pipeline execution is already running; "
                    f"execution {execution_id} was not started. "
                    "Wait for it to finish and retry."
                )
        # Yielding the flag is what lets the caller sweep abandoned runs
        # safely: holding this lock is proof that no other writer is live, so
        # anything still marked `running` was left behind by a process that
        # died. Under SQLite the flag is False and no sweep happens, because
        # nothing was proven.
        yield acquired
    finally:
        if acquired:
            try:
                lock_session.execute(
                    text("SELECT RELEASE_LOCK(:lock_name)"),
                    {"lock_name": PIPELINE_LOCK_NAME},
                )
                lock_session.rollback()
            except Exception:
                logger.warning("could not release pipeline advisory lock", exc_info=True)
        lock_session.close()


_STAGE_SEQUENCE = {stage: index for index, stage in enumerate(STAGES[:-1], start=1)}


def _curation_records(session) -> list[dict]:
    from lake_research_map.db.gold_models import DuplicateOverride

    return [
        {
            "doi_a": row.doi_a,
            "doi_b": row.doi_b,
            "decision": row.decision,
            "canonical_doi": row.canonical_doi,
            "reason": row.reason,
        }
        for row in session.scalars(
            select(DuplicateOverride).order_by(DuplicateOverride.doi_a, DuplicateOverride.doi_b)
        )
    ]


def _ensure_execution(
    session,
    execution_id: str,
    *,
    workflow: str,
    trigger: str,
    version_id: str | None = None,
):
    from lake_research_map.db.gold_models import PipelineExecution

    execution = session.get(PipelineExecution, execution_id)
    if execution is None:
        execution = PipelineExecution(
            execution_id=execution_id,
            requested_stage=workflow,
            trigger=trigger,
            dataset_version_id=version_id,
            status="running",
        )
        session.add(execution)
    elif version_id and execution.dataset_version_id not in (None, version_id):
        raise ValueError(
            f"execution {execution_id!r} is already bound to {execution.dataset_version_id}"
        )
    if version_id:
        execution.dataset_version_id = version_id
    execution.status = "running"
    execution.error_message = None
    execution.heartbeat_at = datetime.now(UTC).replace(tzinfo=None)
    session.flush()
    return execution


def _working_version(session, execution_id: str | None = None) -> str:
    from lake_research_map.db.gold_models import PipelineExecution, PublicationState

    if execution_id:
        execution = session.get(PipelineExecution, execution_id)
        if execution and execution.dataset_version_id:
            return execution.dataset_version_id
    state = session.get(PublicationState, 1)
    if state and state.working_version_id:
        return state.working_version_id
    raise RuntimeError("no working dataset version; run the raw stage first")


def _is_unchanged_full_run(session, version_id: str, workflow: str) -> bool:
    from lake_research_map.db.gold_models import DatasetVersion

    version = session.get(DatasetVersion, version_id)
    return workflow == "all" and version is not None and version.status == "active"


def _complete_skipped_stage(
    session, run_id: int, started: datetime, execution_id: str, version_id: str, stage: str
) -> dict:
    stats = {"no_change": True, "dataset_version_id": version_id}
    _finish_stage(session, run_id, started, stats=stats, status="skipped")
    if stage == "semantic":
        _finalize_execution(session, execution_id, "success")
    session.commit()
    print(f"[{stage}] {stats}")
    return stats


def _begin_stage(session, stage: str, execution_id: str, version_id: str):
    from lake_research_map.db.gold_models import PipelineRun

    started = datetime.now(UTC)
    attempt = (
        session.scalar(
            select(func.max(PipelineRun.attempt)).where(
                PipelineRun.execution_id == execution_id, PipelineRun.stage == stage
            )
        )
        or 0
    ) + 1
    row = PipelineRun(
        execution_id=execution_id,
        dataset_version_id=version_id,
        input_version_id=version_id,
        output_version_id=version_id,
        sequence=_STAGE_SEQUENCE[stage],
        attempt=attempt,
        stage=stage,
        started_at=started,
        finished_at=started,
        duration_seconds=0.0,
        status="running",
    )
    session.add(row)
    session.commit()
    return row.id, started


def _finish_stage(
    session,
    run_id: int,
    started: datetime,
    *,
    stats: dict | None,
    status: str,
    error: str | None = None,
) -> None:
    from lake_research_map.db.gold_models import PipelineRun

    row = session.get(PipelineRun, run_id)
    finished = datetime.now(UTC)
    row.finished_at = finished
    row.duration_seconds = (finished - started).total_seconds()
    row.stats = stats
    row.status = status
    row.error_message = error
    from lake_research_map.db.gold_models import PipelineExecution

    execution = session.get(PipelineExecution, row.execution_id) if row.execution_id else None
    if execution is not None:
        execution.heartbeat_at = finished.replace(tzinfo=None)
    session.flush()


def _finalize_execution(session, execution_id: str, status: str, error: str | None = None) -> None:
    from lake_research_map.db.gold_models import DatasetVersion, PipelineExecution

    execution = session.get(PipelineExecution, execution_id)
    if execution is None:
        return
    execution.status = status
    execution.finished_at = datetime.now(UTC)
    execution.error_message = error
    if error and execution.dataset_version_id:
        version = session.get(DatasetVersion, execution.dataset_version_id)
        if version is not None and version.status != "active":
            version.status = "failed"
            version.failed_at = datetime.now(UTC)
            version.failure_reason = error
    session.flush()


def _record_results(
    session,
    *,
    execution_id: str,
    run_id: int,
    version_id: str,
    stage: str,
    results,
) -> None:
    persist_results(
        session,
        execution_id=execution_id,
        stage_run_id=run_id,
        dataset_version_id=version_id,
        stage=stage,
        results=results,
    )


def _handle_stage_failure(
    session,
    run_id: int,
    started: datetime,
    execution_id: str,
    exc: Exception,
) -> None:
    session.rollback()
    _finish_stage(session, run_id, started, stats=None, status="error", error=str(exc))
    _finalize_execution(session, execution_id, "error", str(exc))
    session.commit()


def _append_effective_sources(raw_session, scanned: list[ScannedSource]) -> list[ScannedSource]:
    """Retain archived active paths that are absent from an additive download batch."""
    from lake_research_map.db.raw_models import SourceBlob, SourceFile, SourceRevision

    effective = {source.path: source for source in scanned}
    for row in raw_session.scalars(select(SourceFile)).all():
        if row.path in effective:
            continue
        revision = (
            raw_session.get(SourceRevision, row.source_revision_id)
            if row.source_revision_id
            else raw_session.scalar(
                select(SourceRevision).where(
                    SourceRevision.path == row.path, SourceRevision.sha256 == row.sha256
                )
            )
        )
        blob = raw_session.get(SourceBlob, row.sha256)
        if revision is None or blob is None:
            raise RuntimeError(
                f"append policy cannot retain {row.path!r}: archived revision is incomplete"
            )
        effective[row.path] = ScannedSource(
            path=row.path,
            source=row.source,
            kind=row.kind,
            sha256=row.sha256,
            size_bytes=row.size_bytes,
            mtime=row.mtime,
            revision_id=revision.revision_id,
            archive_path=blob.archive_path,
        )
    return sorted(effective.values(), key=lambda source: source.path)


def _prepare_raw_version(execution_id: str, workflow: str, trigger: str, source_policy: str):
    from lake_research_map.db.gold_models import DatasetVersion, PublicationState

    sources = scan_sources()
    raw_session = get_session("raw")
    try:
        if source_policy == "append":
            sources = _append_effective_sources(raw_session, sources)
    finally:
        raw_session.close()
    gold_session = get_session("gold")
    try:
        fingerprint = build_fingerprint(sources, curation_records=_curation_records(gold_session))
        state = gold_session.get(PublicationState, 1)
        if state is None:
            state = PublicationState(id=1)
            gold_session.add(state)
        version = gold_session.get(DatasetVersion, fingerprint.version_id)
        if version is None:
            version = DatasetVersion(
                version_id=fingerprint.version_id,
                parent_version_id=state.active_version_id,
                source_manifest_sha256=fingerprint.source_manifest_sha256,
                config_sha256=fingerprint.config_sha256,
                code_revision=fingerprint.code_revision,
                code_sha256=fingerprint.code_sha256,
                curation_sha256=fingerprint.curation_sha256,
                status="candidate",
            )
            gold_session.add(version)
        elif version.status == "failed":
            version.status = "candidate"
            version.failed_at = None
            version.failure_reason = None
        state.working_version_id = fingerprint.version_id
        state.execution_id = execution_id
        _ensure_execution(
            gold_session,
            execution_id,
            workflow=workflow,
            trigger=trigger,
            version_id=fingerprint.version_id,
        )
        gold_session.commit()
        return sources, fingerprint.version_id, version.status == "active"
    finally:
        gold_session.close()


def run_raw(
    *,
    execution_id: str,
    workflow: str = "raw",
    trigger: str = "cli",
    source_policy: str = "append",
) -> dict:
    sources, version_id, already_active = _prepare_raw_version(
        execution_id, workflow, trigger, source_policy
    )
    gold_session = get_session("gold")
    raw_session = get_session("raw")
    run_id, started = _begin_stage(gold_session, "raw", execution_id, version_id)
    try:
        if already_active and workflow == "all":
            return _complete_skipped_stage(
                gold_session, run_id, started, execution_id, version_id, "raw"
            )
        previous = active_manifest(raw_session)
        removed = (
            remove_absent_sources(raw_session, {source.path for source in sources})
            if source_policy == "snapshot"
            else 0
        )
        stats = {
            "source_policy": source_policy,
            "config": load_configs(raw_session),
            "ieee_csv_rows": load_ieee_csv(raw_session),
            "bib_entries": load_bib_entries(raw_session),
            "pdf_files": load_pdf_inventory(raw_session),
        }
        changes = persist_snapshot(
            raw_session,
            sources,
            version_id=version_id,
            execution_id=execution_id,
            previous=previous,
        )
        stats.update(changes)
        stats["removed_active_paths"] = removed
        results = raw_contract(raw_session, version_id)
        _record_results(
            gold_session,
            execution_id=execution_id,
            run_id=run_id,
            version_id=version_id,
            stage="raw",
            results=results,
        )
        gold_session.commit()
        assert_contract("raw", results)
        raw_session.commit()
        _finish_stage(gold_session, run_id, started, stats=stats, status="success")
        if workflow != "all":
            _finalize_execution(gold_session, execution_id, "success")
        gold_session.commit()
        print(f"[raw] {stats}")
        return stats
    except Exception as exc:
        raw_session.rollback()
        _handle_stage_failure(gold_session, run_id, started, execution_id, exc)
        raise
    finally:
        raw_session.close()
        gold_session.close()


def run_bronze(*, execution_id: str, workflow: str = "bronze", trigger: str = "cli") -> dict:
    gold_session = get_session("gold")
    version_id = _working_version(gold_session, execution_id)
    _ensure_execution(
        gold_session, execution_id, workflow=workflow, trigger=trigger, version_id=version_id
    )
    gold_session.commit()
    run_id, started = _begin_stage(gold_session, "bronze", execution_id, version_id)
    raw_session = get_session("raw")
    bronze_session = get_session("bronze")
    try:
        if _is_unchanged_full_run(gold_session, version_id, workflow):
            return _complete_skipped_stage(
                gold_session, run_id, started, execution_id, version_id, "bronze"
            )
        stats = build_bronze_articles(raw_session, bronze_session, version_id)
        results = bronze_contract(raw_session, bronze_session, version_id)
        _record_results(
            gold_session,
            execution_id=execution_id,
            run_id=run_id,
            version_id=version_id,
            stage="bronze",
            results=results,
        )
        gold_session.commit()
        assert_contract("bronze", results)
        bronze_session.commit()
        _finish_stage(gold_session, run_id, started, stats=stats, status="success")
        if workflow != "all":
            _finalize_execution(gold_session, execution_id, "success")
        gold_session.commit()
        print(f"[bronze] {stats}")
        return stats
    except Exception as exc:
        bronze_session.rollback()
        _handle_stage_failure(gold_session, run_id, started, execution_id, exc)
        raise
    finally:
        raw_session.close()
        bronze_session.close()
        gold_session.close()


def run_silver(*, execution_id: str, workflow: str = "silver", trigger: str = "cli") -> dict:
    gold_session = get_session("gold")
    version_id = _working_version(gold_session, execution_id)
    _ensure_execution(
        gold_session, execution_id, workflow=workflow, trigger=trigger, version_id=version_id
    )
    gold_session.commit()
    run_id, started = _begin_stage(gold_session, "silver", execution_id, version_id)
    raw_session = get_session("raw")
    bronze_session = get_session("bronze")
    silver_session = get_session("silver")
    try:
        if _is_unchanged_full_run(gold_session, version_id, workflow):
            return _complete_skipped_stage(
                gold_session, run_id, started, execution_id, version_id, "silver"
            )
        stats = build_silver_articles(bronze_session, silver_session, raw_session, version_id)
        results = silver_contract(bronze_session, silver_session, raw_session, version_id)
        _record_results(
            gold_session,
            execution_id=execution_id,
            run_id=run_id,
            version_id=version_id,
            stage="silver",
            results=results,
        )
        gold_session.commit()
        assert_contract("silver", results)
        silver_session.commit()
        _finish_stage(gold_session, run_id, started, stats=stats, status="success")
        if workflow != "all":
            _finalize_execution(gold_session, execution_id, "success")
        gold_session.commit()
        print(f"[silver] {stats}")
        return stats
    except Exception as exc:
        silver_session.rollback()
        _handle_stage_failure(gold_session, run_id, started, execution_id, exc)
        raise
    finally:
        raw_session.close()
        bronze_session.close()
        silver_session.close()
        gold_session.close()


def run_gold(*, execution_id: str, workflow: str = "gold", trigger: str = "cli") -> dict:
    gold_session = get_session("gold")
    version_id = _working_version(gold_session, execution_id)
    _ensure_execution(
        gold_session, execution_id, workflow=workflow, trigger=trigger, version_id=version_id
    )
    gold_session.commit()
    run_id, started = _begin_stage(gold_session, "gold", execution_id, version_id)
    silver_session = get_session("silver")
    try:
        if _is_unchanged_full_run(gold_session, version_id, workflow):
            return _complete_skipped_stage(
                gold_session, run_id, started, execution_id, version_id, "gold"
            )
        stats = build_dataset_gold(silver_session, gold_session, version_id)
        results = gold_contract(gold_session, version_id)
        _record_results(
            gold_session,
            execution_id=execution_id,
            run_id=run_id,
            version_id=version_id,
            stage="gold",
            results=results,
        )
        gold_session.commit()
        assert_contract("gold", results)
        _finish_stage(gold_session, run_id, started, stats=stats, status="success")
        if workflow != "all":
            _finalize_execution(gold_session, execution_id, "success")
        gold_session.commit()
        print(f"[gold] {stats}")
        return stats
    except Exception as exc:
        _handle_stage_failure(gold_session, run_id, started, execution_id, exc)
        raise
    finally:
        silver_session.close()
        gold_session.close()


def run_embed(*, execution_id: str, workflow: str = "embed", trigger: str = "cli") -> dict:
    gold_session = get_session("gold")
    version_id = _working_version(gold_session, execution_id)
    _ensure_execution(
        gold_session, execution_id, workflow=workflow, trigger=trigger, version_id=version_id
    )
    gold_session.commit()
    run_id, started = _begin_stage(gold_session, "embed", execution_id, version_id)
    try:
        if _is_unchanged_full_run(gold_session, version_id, workflow):
            return _complete_skipped_stage(
                gold_session, run_id, started, execution_id, version_id, "embed"
            )
        gold_session.info["execution_id"] = execution_id
        stats = build_dataset_embeddings(gold_session, version_id)
        results = embed_contract(gold_session, version_id)
        _record_results(
            gold_session,
            execution_id=execution_id,
            run_id=run_id,
            version_id=version_id,
            stage="embed",
            results=results,
        )
        gold_session.commit()
        assert_contract("embed", results)
        _finish_stage(gold_session, run_id, started, stats=stats, status="success")
        if workflow != "all":
            _finalize_execution(gold_session, execution_id, "success")
        gold_session.commit()
        print(f"[embed] {stats}")
        return stats
    except Exception as exc:
        _handle_stage_failure(gold_session, run_id, started, execution_id, exc)
        raise
    finally:
        gold_session.close()


def _version_ready(session, version_id: str) -> bool:
    from lake_research_map.db.gold_models import PipelineRun, QualityResult

    for stage in STAGES[:-1]:
        run_id = session.scalar(
            select(PipelineRun.id)
            .where(
                PipelineRun.dataset_version_id == version_id,
                PipelineRun.stage == stage,
                PipelineRun.status == "success",
            )
            .order_by(PipelineRun.id.desc())
            .limit(1)
        )
        if run_id is None:
            return False
        blocking_failure = session.scalar(
            select(func.count())
            .select_from(QualityResult)
            .where(
                QualityResult.stage_run_id == run_id,
                QualityResult.severity == "error",
                QualityResult.passed.is_(False),
            )
        )
        if blocking_failure:
            return False
    return True


def run_semantic(*, execution_id: str, workflow: str = "semantic", trigger: str = "cli") -> dict:
    gold_session = get_session("gold")
    version_id = _working_version(gold_session, execution_id)
    _ensure_execution(
        gold_session, execution_id, workflow=workflow, trigger=trigger, version_id=version_id
    )
    gold_session.commit()
    run_id, started = _begin_stage(gold_session, "semantic", execution_id, version_id)
    try:
        if _is_unchanged_full_run(gold_session, version_id, workflow):
            return _complete_skipped_stage(
                gold_session, run_id, started, execution_id, version_id, "semantic"
            )
        stats = build_dataset_semantics(gold_session, version_id)
        results = semantic_contract(gold_session, version_id)
        _record_results(
            gold_session,
            execution_id=execution_id,
            run_id=run_id,
            version_id=version_id,
            stage="semantic",
            results=results,
        )
        gold_session.commit()
        assert_contract("semantic", results)
        _finish_stage(gold_session, run_id, started, stats=stats, status="success")
        gold_session.flush()
        if not _version_ready(gold_session, version_id):
            raise ContractViolation("publication", [])
        materialize_version(gold_session, version_id, execution_id)
        _finalize_execution(gold_session, execution_id, "success")
        gold_session.commit()
        print(f"[semantic] {stats}")
        return stats
    except Exception as exc:
        _handle_stage_failure(gold_session, run_id, started, execution_id, exc)
        raise
    finally:
        gold_session.close()


def run_all(
    *,
    execution_id: str | None = None,
    trigger: str = "cli",
    source_policy: str = "append",
) -> dict:
    execution_id = execution_id or str(uuid.uuid4())
    bootstrap()
    return {
        "raw": run_raw(
            execution_id=execution_id,
            workflow="all",
            trigger=trigger,
            source_policy=source_policy,
        ),
        "bronze": run_bronze(execution_id=execution_id, workflow="all", trigger=trigger),
        "silver": run_silver(execution_id=execution_id, workflow="all", trigger=trigger),
        "gold": run_gold(execution_id=execution_id, workflow="all", trigger=trigger),
        "embed": run_embed(execution_id=execution_id, workflow="all", trigger=trigger),
        "semantic": run_semantic(execution_id=execution_id, workflow="all", trigger=trigger),
    }


def run(
    stage: str,
    *,
    execution_id: str | None = None,
    trigger: str = "cli",
    workflow: str | None = None,
    source_policy: str = "append",
) -> None:
    execution_id = execution_id or str(uuid.uuid4())
    workflow = workflow or stage
    with _pipeline_lock(execution_id) as exclusive:
        bootstrap()
        if exclusive:
            _recover_before_run(execution_id)
        if stage == "all":
            run_all(execution_id=execution_id, trigger=trigger, source_policy=source_policy)
            return
        runners = {
            "raw": run_raw,
            "bronze": run_bronze,
            "silver": run_silver,
            "gold": run_gold,
            "embed": run_embed,
            "semantic": run_semantic,
        }
        if stage == "raw":
            run_raw(
                execution_id=execution_id,
                workflow=workflow,
                trigger=trigger,
                source_policy=source_policy,
            )
        else:
            runners[stage](execution_id=execution_id, workflow=workflow, trigger=trigger)


def _add_pair_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--doi-a", required=True, help="First DOI in the reviewed pair")
    parser.add_argument("--doi-b", required=True, help="Second DOI in the reviewed pair")


def _configure_duplicate_commands(subparsers) -> None:
    duplicates = subparsers.add_parser(
        "duplicates", help="Review persistent cross-DOI near-duplicate decisions"
    )
    actions = duplicates.add_subparsers(dest="duplicate_action", required=True)

    list_parser = actions.add_parser("list", help="List unresolved duplicate candidates")
    list_parser.add_argument("--all", action="store_true", help="Include resolved decision history")

    merge = actions.add_parser("merge", help="Merge a duplicate into an explicit canonical DOI")
    merge.add_argument("--canonical-doi", required=True)
    merge.add_argument("--duplicate-doi", required=True)
    merge.add_argument("--reason", required=True)

    keep = actions.add_parser("keep", help="Keep a reviewed pair as distinct publications")
    _add_pair_arguments(keep)
    keep.add_argument("--reason", required=True)

    undo = actions.add_parser("undo", help="Remove a previous duplicate decision")
    _add_pair_arguments(undo)


def _configure_version_commands(subparsers) -> None:
    versions = subparsers.add_parser("versions", help="Inspect or reactivate dataset versions")
    actions = versions.add_subparsers(dest="version_action", required=True)
    actions.add_parser("list", help="List dataset versions and publication status")
    activate = actions.add_parser("activate", help="Reactivate a validated Gold snapshot")
    activate.add_argument("--version-id", required=True)


def _configure_review_commands(subparsers) -> None:
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

    from lake_research_map.transform.review_workflows import (
        assignment_progress,
        collect_labels,
    )
    from lake_research_map.transform.screening_calibration import (
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

    from lake_research_map.db.gold_models import DatasetSemantics
    from lake_research_map.transform.embeddings import EMBED_MODEL_NAME
    from lake_research_map.transform.review_workflows import collect_labels
    from lake_research_map.transform.screening_calibration import (
        calibrate_screening_threshold,
        label_set_digest,
        resolve_review_consensus,
        screening_model_digest,
        validate_review_labels,
    )
    from lake_research_map.transform.semantics import ANCHOR_TEXT, OFF_ANCHOR_TEXT

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


def _run_review_command(args: argparse.Namespace) -> None:
    from pathlib import Path

    from lake_research_map.transform.review_workflows import (
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
            count = export_assignments(
                session,
                workflow=args.workflow,
                dataset_version_id=args.version_id,
                reviewer_id=args.reviewer,
                output_path=Path(args.output),
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


def _configure_enrichment_commands(subparsers) -> None:
    enrichment = subparsers.add_parser(
        "enrichment", help="Append reproducible external metadata observations"
    )
    actions = enrichment.add_subparsers(dest="enrichment_action", required=True)
    refresh = actions.add_parser("refresh-openalex", help="Refresh the active DOI population")
    refresh.add_argument("--max-fetch", type=int, default=100)
    refresh.add_argument("--delay", type=float, default=0.1)
    citations = actions.add_parser(
        "refresh-citations",
        help="Crawl incoming citation edges for works already observed in Bronze",
    )
    citations.add_argument("--max-works", type=int, default=50)
    citations.add_argument("--delay", type=float, default=0.2)
    citations.add_argument(
        "--max-pages",
        type=int,
        default=None,
        help="Page budget per work; a work that exhausts it is recorded as truncated",
    )


def _require_openalex_identity() -> str:
    """OpenAlex has no API key; what it asks for is a contact address.

    The gate used to demand `OPENALEX_API_KEY` as well, which no OpenAlex
    account issues for the public corpus, so every refresh was unreachable by
    construction -- the same shape of defect as the database roles WP-08
    removed. The polite-pool address is mandatory because an anonymous crawl
    of this size is what gets a client rate-limited; a key is accepted when
    one exists and never required.
    """
    email = (os.environ.get("OPENALEX_EMAIL") or "").strip()
    if not email:
        raise ValueError(
            "OPENALEX_EMAIL must be set: OpenAlex identifies polite-pool clients by a contact "
            "address, and an anonymous crawl of the whole corpus will be throttled. "
            "OPENALEX_API_KEY is optional and only applies to a premium account."
        )
    return email


def _run_enrichment_command(args: argparse.Namespace) -> None:
    from lake_research_map.db.gold_models import DatasetArticle, PublicationState
    from lake_research_map.ingest.openalex import refresh_openalex_observations

    _require_openalex_identity()
    bootstrap()
    gold_session = get_session("gold")
    bronze_session = get_session("bronze")
    try:
        # The forward crawl reads Bronze only. Requiring an active Gold version
        # for it would fail with "no active dataset version", which says
        # nothing about the command the operator actually ran.
        if args.enrichment_action == "refresh-citations":
            print(_refresh_citation_edges(bronze_session, args))
            return
        state = gold_session.get(PublicationState, 1)
        if state is None or not state.active_version_id:
            raise ValueError("no active dataset version")
        dois = gold_session.scalars(
            select(DatasetArticle.doi)
            .where(DatasetArticle.dataset_version_id == state.active_version_id)
            .order_by(DatasetArticle.doi)
        ).all()
        stats = refresh_openalex_observations(
            bronze_session,
            dois,
            max_fetch=args.max_fetch,
            delay=args.delay,
        )
        print(stats)
    finally:
        bronze_session.close()
        gold_session.close()


def _refresh_citation_edges(bronze_session, args: argparse.Namespace) -> dict:
    """Collect forward citation edges for works `refresh-openalex` already saw.

    The crawl is driven from `lit_external_works` rather than from DOIs,
    because `cites:` filters on an OpenAlex work id: a DOI the backward pass
    never resolved has no id to crawl, and asking for one would be a second
    lookup per work for no extra evidence.

    Truncation is recorded, not smoothed over. A work whose page budget ran
    out has an incomplete forward set, and `citation_graph_coverage` has to be
    able to exclude it -- otherwise a disruption index would be computed over
    a citation tail that was silently cut off.
    """
    import time

    from lake_research_map.db.bronze_models import ExternalWork
    from lake_research_map.ingest.openalex import (
        CITING_MAX_PAGES,
        fetch_openalex_citing_works,
        persist_incoming_edges,
    )

    observed_at = datetime.now(UTC).replace(tzinfo=None)
    work_ids = bronze_session.scalars(
        select(ExternalWork.provider_work_id)
        .where(ExternalWork.provider == "openalex")
        .order_by(ExternalWork.provider_work_id)
    ).all()
    if not work_ids:
        raise ValueError(
            "no OpenAlex works have been observed yet; run `enrichment refresh-openalex` first"
        )

    max_pages = args.max_pages or CITING_MAX_PAGES
    edges = truncated = crawled = 0
    for work_id in work_ids[: args.max_works]:
        result = fetch_openalex_citing_works(work_id, max_pages=max_pages)
        edges += persist_incoming_edges(
            bronze_session, work_id, result["citing_work_ids"], observed_at
        )
        truncated += int(result["truncated"])
        crawled += 1
        if args.delay:
            time.sleep(args.delay)
    bronze_session.commit()
    return {
        "works_known": len(work_ids),
        "works_crawled": crawled,
        "edges_inserted": edges,
        "truncated_works": truncated,
        "observed_at": observed_at.isoformat(),
    }


def _configure_evidence_commands(subparsers) -> None:
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


def _run_evidence_command(args: argparse.Namespace) -> None:
    """Write a labelled-ready CSV of review subjects.

    Read-only against the corpus: generating a sample must never mutate what
    it is sampling. The output feeds `reviews setup --subject-id-file`, or is
    labelled directly and imported.
    """
    import csv
    from pathlib import Path

    from lake_research_map.db.engines import get_session
    from lake_research_map.transform import evidence_samples

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
            # Retrieval pools the live search modes, so it needs the embedded
            # corpus rather than the silver metadata the others read.
            from lake_research_map.dashboard import loaders
            from lake_research_map.dashboard.search import hybrid_search_rrf

            # Loaded once: every query pools over the same chunk population,
            # and re-reading it per query would be ten full table scans.
            chunk_frame = loaders.chunk_search_data()
            if chunk_frame.empty:
                # `chunk_search_data` tolerates an unreachable database and an
                # absent table alike, both as an empty frame, so this message
                # must not assert a cause it cannot tell apart.
                raise ValueError(
                    "no chunks available to retrieve over. Either the `embed` stage has not "
                    "run, or the database is unreachable -- check the log above for a "
                    "connection error before re-running `--stage embed`."
                )

            def _retrieve(text: str, depth: int) -> list[str]:
                hits = hybrid_search_rrf(text, chunk_frame, top_k=depth)
                return [str(doi) for doi in hits["doi"]] if "doi" in hits else []

            subjects, stats = evidence_samples.retrieval_candidates(_retrieve, depth=limit or 10)
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


def _configure_audit_commands(subparsers) -> None:
    audit = subparsers.add_parser("audit", help="Run read-only corpus acceptance audits")
    actions = audit.add_subparsers(dest="audit_action", required=True)
    actions.add_parser("reference-corpus", help="Report active version and contract coverage")
    actions.add_parser(
        "citation-graph",
        help="Measure forward/backward citation coverage and whether disruption is usable",
    )


def _audit_citation_graph() -> None:
    """Report WP-24's coverage gate instead of asserting it was met.

    The gate says CD/disruption stays unavailable unless coverage is adequate,
    which nothing could evaluate while `citation_graph_coverage` had no caller.
    The usable population is the intersection of the two directions, so this
    prints the number that decides the gate rather than the larger of the two
    that would flatter it.
    """
    from lake_research_map.db.bronze_models import ExternalWork
    from lake_research_map.ingest.openalex import citation_graph_coverage

    bootstrap()
    session = get_session("bronze")
    try:
        work_ids = session.scalars(
            select(ExternalWork.provider_work_id).where(ExternalWork.provider == "openalex")
        ).all()
        if not work_ids:
            print(
                "No OpenAlex works observed, so citation coverage is 0 by absence rather than "
                "by measurement. Run `enrichment refresh-openalex` and then "
                "`enrichment refresh-citations`."
            )
            return
        coverage = citation_graph_coverage(session, list(work_ids))
        print(f"Population:          {coverage['population']}")
        print(
            f"Backward (refs):     {coverage['with_backward']} "
            f"({coverage['backward_coverage']:.1%})"
        )
        print(
            f"Forward (cites:):    {coverage['with_forward']} ({coverage['forward_coverage']:.1%})"
        )
        print(
            f"Usable for CD:       {coverage['usable_for_disruption']} "
            f"({coverage['disruption_coverage']:.1%}) -- works with both directions"
        )
        if coverage["disruption_coverage"] < 0.80:
            print(
                "Below 80% in both directions: the disruption index stays unavailable, which is "
                "WP-24's documented outcome for inadequate coverage, not a failure to fix in code."
            )
    finally:
        session.close()


def _run_audit_command(args: argparse.Namespace) -> None:
    if args.audit_action == "citation-graph":
        _audit_citation_graph()
        return

    from lake_research_map.db.gold_models import (
        DatasetArticle,
        DatasetChunk,
        DatasetSemantics,
        PublicationState,
        QualityResult,
    )

    bootstrap()
    session = get_session("gold")
    try:
        state = session.get(PublicationState, 1)
        version_id = state.active_version_id if state else None
        if not version_id:
            raise ValueError("no active dataset version")
        counts = {
            "version_id": version_id,
            "articles": session.scalar(
                select(func.count())
                .select_from(DatasetArticle)
                .where(DatasetArticle.dataset_version_id == version_id)
            ),
            "chunks": session.scalar(
                select(func.count())
                .select_from(DatasetChunk)
                .where(DatasetChunk.dataset_version_id == version_id)
            ),
            "semantics": session.scalar(
                select(func.count())
                .select_from(DatasetSemantics)
                .where(DatasetSemantics.dataset_version_id == version_id)
            ),
            "failed_quality_checks": session.scalar(
                select(func.count())
                .select_from(QualityResult)
                .where(
                    QualityResult.dataset_version_id == version_id,
                    QualityResult.severity == "error",
                    QualityResult.passed.is_(False),
                )
            ),
        }
        print(counts)
    finally:
        session.close()


def _configure_maintenance_commands(subparsers) -> None:
    maintenance = subparsers.add_parser("maintenance", help="Safe pipeline recovery operations")
    actions = maintenance.add_subparsers(dest="maintenance_action", required=True)
    recover = actions.add_parser("recover-stale", help="Fail stale running executions")
    recover.add_argument("--older-than-minutes", type=int, default=30)


def recover_abandoned_executions(
    session,
    *,
    cutoff: datetime | None = None,
    reason: str = "recovered after stale heartbeat",
) -> list[str]:
    """Fail every execution still marked `running` that nothing is driving.

    Callers must already hold the pipeline lock. With `cutoff` set, only runs
    whose last signal predates it are touched; with `cutoff` None the caller is
    asserting exclusivity -- the advisory lock is held, so no live writer
    exists and every `running` row is by definition abandoned.

    Returns the recovered execution ids so the caller can report them; a
    recovery that happens silently is indistinguishable from one that never
    ran.
    """
    from lake_research_map.db.gold_models import PipelineExecution, PipelineRun

    # A missing heartbeat has to count as stale, not as "still alive".
    # `heartbeat_at` is NULL for every execution that predates the column
    # and for any process killed before its first batch, so requiring it
    # to be non-null made the recovery a no-op on exactly the abandoned
    # runs it exists to clear (NULL < cutoff is NULL, never true).
    # `started_at` is always present, so fall back to it.
    last_signal = func.coalesce(PipelineExecution.heartbeat_at, PipelineExecution.started_at)
    query = select(PipelineExecution).where(PipelineExecution.status == "running")
    if cutoff is not None:
        query = query.where(last_signal < cutoff)
    executions = session.scalars(query).all()
    for execution in executions:
        execution.status = "error"
        execution.finished_at = datetime.now(UTC).replace(tzinfo=None)
        execution.error_message = reason
        for run_row in session.scalars(
            select(PipelineRun).where(
                PipelineRun.execution_id == execution.execution_id,
                PipelineRun.status == "running",
            )
        ):
            run_row.status = "error"
            run_row.finished_at = execution.finished_at
            run_row.duration_seconds = max(
                0.0, (execution.finished_at - run_row.started_at).total_seconds()
            )
            run_row.error_message = execution.error_message
    session.commit()
    return [execution.execution_id for execution in executions]


def _recover_before_run(execution_id: str) -> None:
    """Clear runs abandoned by an earlier crash, before this one starts.

    The manual `maintenance recover-stale` command was the only way a killed
    run ever stopped reading as `running`, so until someone remembered to type
    it the execution history claimed a run was in progress that had not existed
    for days -- and the dashboard's own status came from those rows. This runs
    under the lock we already hold, which is what makes it safe to treat every
    surviving `running` row as abandoned rather than waiting out a timeout.
    """
    session = get_session("gold")
    try:
        recovered = recover_abandoned_executions(
            session, reason="recovered automatically at the start of a later run"
        )
        if recovered:
            logger.warning(
                "recovered %d abandoned execution(s) before starting %s: %s",
                len(recovered),
                execution_id,
                ", ".join(recovered),
            )
    except Exception:
        # Recovery is housekeeping. A pipeline run that is otherwise ready
        # must not be blocked by it.
        logger.warning("could not recover abandoned executions", exc_info=True)
        session.rollback()
    finally:
        session.close()


def _run_maintenance_command(args: argparse.Namespace) -> None:
    if args.older_than_minutes < 1:
        raise ValueError("--older-than-minutes must be positive")
    execution_id = f"recovery-{uuid.uuid4()}"
    with _pipeline_lock(execution_id):
        bootstrap()
        session = get_session("gold")
        try:
            cutoff = datetime.now(UTC).replace(tzinfo=None) - timedelta(
                minutes=args.older_than_minutes
            )
            recovered = recover_abandoned_executions(session, cutoff=cutoff)
            print({"recovered": len(recovered), "cutoff": cutoff.isoformat()})
        finally:
            session.close()


def _run_duplicate_command(args: argparse.Namespace) -> None:
    from lake_research_map.transform.duplicate_resolution import (
        list_review_rows,
        record_decision,
        undo_decision,
    )

    bootstrap()
    gold_session = get_session("gold")
    silver_session = None
    try:
        if args.duplicate_action == "list":
            rows = list_review_rows(gold_session, include_resolved=args.all)
            if not rows:
                print("No near-duplicate review rows found.")
                return
            for row in rows:
                similarity = (
                    f" similarity={row['similarity']:.4f}" if row["similarity"] is not None else ""
                )
                decision = (
                    f" decision={row['decision']} canonical={row['canonical_doi'] or '-'}"
                    if row["decision"]
                    else ""
                )
                reason = f" reason={row['reason']}" if row["reason"] else ""
                print(
                    f"{row['kind']} doi_a={row['doi_a']} doi_b={row['doi_b']}"
                    f"{similarity}{decision}{reason}"
                )
            return

        if args.duplicate_action == "merge":
            silver_session = get_session("silver")
            override = record_decision(
                gold_session,
                silver_session,
                decision="merge",
                doi_a=args.canonical_doi,
                doi_b=args.duplicate_doi,
                canonical_doi=args.canonical_doi,
                reason=args.reason,
            )
            print(
                f"Recorded merge: {args.duplicate_doi} -> {override.canonical_doi}. "
                "Run --stage all to build and publish a newly versioned dataset."
            )
            return

        if args.duplicate_action == "keep":
            silver_session = get_session("silver")
            record_decision(
                gold_session,
                silver_session,
                decision="keep",
                doi_a=args.doi_a,
                doi_b=args.doi_b,
                reason=args.reason,
            )
            print(
                "Recorded keep-separate decision. The pair is now excluded from the review queue."
            )
            return

        removed = undo_decision(gold_session, doi_a=args.doi_a, doi_b=args.doi_b)
        if removed:
            print(
                "Removed duplicate decision. Run --stage all to build and publish a newly "
                "versioned dataset."
            )
        else:
            print("No matching duplicate decision exists.")
    finally:
        gold_session.close()
        if silver_session is not None:
            silver_session.close()


def _run_version_command(args: argparse.Namespace) -> None:
    from lake_research_map.db.gold_models import DatasetVersion, PipelineExecution

    bootstrap()
    session = get_session("gold")
    try:
        if args.version_action == "list":
            versions = session.scalars(
                select(DatasetVersion).order_by(DatasetVersion.created_at.desc())
            ).all()
            if not versions:
                print("No dataset versions found.")
                return
            for version in versions:
                print(
                    f"{version.version_id} status={version.status} "
                    f"created={version.created_at} published={version.published_at or '-'}"
                )
            return

        version = session.get(DatasetVersion, args.version_id)
        if version is None:
            raise ValueError(f"unknown dataset version {args.version_id}")
        is_legacy = bool(version.stats and version.stats.get("kind") == "legacy_import")
        if not is_legacy and not _version_ready(session, version.version_id):
            raise ValueError("the requested version has not passed every required quality gate")
        execution_id = str(uuid.uuid4())
        session.add(
            PipelineExecution(
                execution_id=execution_id,
                requested_stage="activate",
                trigger="cli",
                dataset_version_id=version.version_id,
                status="running",
            )
        )
        session.flush()
        materialize_version(session, version.version_id, execution_id)
        _finalize_execution(session, execution_id, "success")
        session.commit()
        print(f"Activated dataset version {version.version_id}.")
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def main(argv: list[str] | None = None) -> None:
    invoked_as_cli = argv is None
    _configure_logging()
    if invoked_as_cli:
        _install_signal_handlers()
    parser = argparse.ArgumentParser(description="lake-research-map medallion pipeline")
    parser.add_argument("--stage", choices=STAGES, default="all", help="pipeline stage to run")
    parser.add_argument(
        "--execution-id", help="parent execution identifier shared by orchestrators"
    )
    parser.add_argument("--trigger", choices=("cli", "airflow", "dashboard"), default="cli")
    parser.add_argument("--workflow", choices=STAGES, help="parent workflow for stage correlation")
    parser.add_argument(
        "--source-policy",
        choices=("append", "snapshot"),
        default="append",
        help="retain absent archived sources (append) or mirror the files currently on disk",
    )
    subparsers = parser.add_subparsers(dest="command")
    _configure_duplicate_commands(subparsers)
    _configure_version_commands(subparsers)
    _configure_review_commands(subparsers)
    _configure_enrichment_commands(subparsers)
    _configure_evidence_commands(subparsers)
    _configure_audit_commands(subparsers)
    _configure_maintenance_commands(subparsers)
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])
    try:
        if args.command == "duplicates":
            from lake_research_map.transform.duplicate_resolution import DuplicateResolutionError

            try:
                _run_duplicate_command(args)
            except DuplicateResolutionError as exc:
                parser.error(str(exc))
        elif args.command == "versions":
            try:
                _run_version_command(args)
            except ValueError as exc:
                parser.error(str(exc))
        elif args.command == "reviews":
            try:
                _run_review_command(args)
            except (OSError, ValueError) as exc:
                parser.error(str(exc))
        elif args.command == "enrichment":
            try:
                _run_enrichment_command(args)
            except ValueError as exc:
                parser.error(str(exc))
        elif args.command == "evidence":
            try:
                _run_evidence_command(args)
            except (OSError, ValueError) as exc:
                parser.error(str(exc))
        elif args.command == "audit":
            try:
                _run_audit_command(args)
            except ValueError as exc:
                parser.error(str(exc))
        elif args.command == "maintenance":
            try:
                _run_maintenance_command(args)
            except ValueError as exc:
                parser.error(str(exc))
        else:
            if args.execution_id is None and args.trigger == "cli" and args.workflow is None:
                if args.source_policy == "append":
                    run(args.stage)
                else:
                    run(args.stage, source_policy=args.source_policy)
            else:
                run(
                    args.stage,
                    execution_id=args.execution_id,
                    trigger=args.trigger,
                    workflow=args.workflow,
                    source_policy=args.source_policy,
                )
    except PipelineBusyError as exc:
        logger.error("pipeline not started: %s", exc)
        if invoked_as_cli:
            raise SystemExit(2) from exc
        raise
    except Exception as exc:
        logger.exception("pipeline failed: %s", exc)
        if invoked_as_cli:
            raise SystemExit(1) from exc
        raise


if __name__ == "__main__":
    main()
