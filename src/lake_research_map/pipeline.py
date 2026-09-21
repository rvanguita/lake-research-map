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
import logging
import os
import sys
import uuid
from datetime import UTC, datetime

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
        yield
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
    with _pipeline_lock(execution_id):
        bootstrap()
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
