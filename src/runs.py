"""Process infrastructure and run bookkeeping for the medallion pipeline.

Everything a stage needs around its transform, and nothing a stage computes: the
signal and logging setup of a CLI process, the MySQL advisory lock, one session per
layer, the `lit_pipeline_executions` / `lit_pipeline_runs` rows a stage begins and
finishes, and the recovery of executions an earlier process abandoned. The stages
themselves are `pipeline.py`; the command line that wires both is `cli/`. This
module imports no application and no stage, so either can depend on it.
"""

from __future__ import annotations

import contextlib
import logging
import os
import signal
import sys
from datetime import UTC, datetime

from sqlalchemy import func, select, text

from db.engines import get_session
from quality import persist_results

# The medallion stages in execution order. `pipeline.STAGES` is this plus "all";
# it is defined here, once, because `_STAGE_SEQUENCE` numbers the run rows and
# this module must not import `pipeline` (which imports it).
STAGE_ORDER = ("raw", "bronze", "silver", "gold", "embed", "semantic")

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
    from config import DATA_DIR

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


@contextlib.contextmanager
def _layer_sessions(*layers: str):
    """Open one session per layer and close every one of them on exit.

    The stage functions used to open their sessions and then call
    `_working_version`, `_ensure_execution` and `_begin_stage` before entering
    their `try/finally`, so an exception there (no working version, an
    execution bound to another version, a failed commit) leaked every session
    already open. Each session is registered for closing the moment it is
    opened, so a failure anywhere after -- including a later `get_session` --
    still closes the ones before it. Sessions are opened through the
    module-level `get_session`, the injection point the tests monkeypatch.
    """
    with contextlib.ExitStack() as stack:
        sessions = []
        for layer in layers:
            session = get_session(layer)
            stack.callback(session.close)
            sessions.append(session)
        yield tuple(sessions)


_STAGE_SEQUENCE = {stage: index for index, stage in enumerate(STAGE_ORDER, start=1)}


def _ensure_execution(
    session,
    execution_id: str,
    *,
    workflow: str,
    trigger: str,
    version_id: str | None = None,
):
    from db.gold_models import PipelineExecution

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
    # A resumed execution is not finished; a stale timestamp left by an earlier
    # recovery would otherwise show a live run as having ended.
    execution.finished_at = None
    execution.heartbeat_at = datetime.now(UTC).replace(tzinfo=None)
    session.flush()
    return execution


def _working_version(session, execution_id: str | None = None) -> str:
    from db.gold_models import PipelineExecution, PublicationState

    if execution_id:
        execution = session.get(PipelineExecution, execution_id)
        if execution and execution.dataset_version_id:
            return execution.dataset_version_id
    state = session.get(PublicationState, 1)
    if state and state.working_version_id:
        return state.working_version_id
    raise RuntimeError("no working dataset version; run the raw stage first")


def _is_unchanged_full_run(session, version_id: str, workflow: str) -> bool:
    from db.gold_models import DatasetVersion

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
    from db.gold_models import PipelineRun

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
    from db.gold_models import PipelineRun

    row = session.get(PipelineRun, run_id)
    finished = datetime.now(UTC)
    row.finished_at = finished
    row.duration_seconds = (finished - started).total_seconds()
    row.stats = stats
    row.status = status
    row.error_message = error
    from db.gold_models import PipelineExecution

    execution = session.get(PipelineExecution, row.execution_id) if row.execution_id else None
    if execution is not None:
        execution.heartbeat_at = finished.replace(tzinfo=None)
    session.flush()


def _finalize_execution(session, execution_id: str, status: str, error: str | None = None) -> None:
    from db.gold_models import DatasetVersion, PipelineExecution

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


def recover_abandoned_executions(
    session,
    *,
    cutoff: datetime | None = None,
    reason: str = "recovered after stale heartbeat",
    exclude: set[str] | None = None,
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
    from db.gold_models import PipelineExecution, PipelineRun

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
    if exclude:
        # A workflow run stage by stage (`--execution-id` with `--workflow`)
        # runs each stage as its own process under one execution id, and every
        # stage but the last leaves that execution `running` on purpose. Holding the lock proves no *other* writer is
        # alive; it says nothing about the execution this process is about to
        # continue, which the sweep used to mark as crashed between every task.
        query = query.where(PipelineExecution.execution_id.not_in(sorted(exclude)))
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
            session,
            reason="recovered automatically at the start of a later run",
            exclude={execution_id},
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
