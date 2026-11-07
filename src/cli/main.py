"""`lake-research-map`: the argument parser, the lock policy and the dispatch.

Usage is in `pipeline`'s docstring and in `uv run lake-research-map --help`.
Every command group is registered here and nowhere else; the stage runner is
`pipeline.run`. The application commands (`vectors`, `search`, `screen`, `mcp`)
are wired here because the core may not import an application.
"""

from __future__ import annotations

import argparse
import contextlib
import logging
import sys
import uuid

# `_pipeline_lock` is looked up on `runs` at call time, so a test can replace it.
import runs
from apps.api import cli as api_cli
from apps.qdrant import cli as qdrant_cli
from cli import audit, duplicates, enrichment, evidence, maintenance, reviews, versions
from db.vector_store import VectorStoreUnavailable
from pipeline import STAGES, run
from runs import (
    PipelineBusyError,
    PipelineInterruptedError,
    _configure_logging,
    _install_signal_handlers,
)

logger = logging.getLogger(__name__)


def _mutates_published_inputs(args: argparse.Namespace) -> bool:
    """Whether a CLI command changes what the next publication would contain.

    These take the same advisory lock as a stage run. Without it a duplicate
    merge recorded while bronze and silver were running was folded in by the
    gold stage of a version whose id had hashed the decisions *before* it,
    and `versions activate` could race `materialize_version` over the live
    tables. The citation-edge, reference-year and Crossref crawls feed only
    audits, not any stage, so they stay unlocked rather than blocking the
    pipeline for the hours a crawl can take.
    """
    if args.command == "duplicates":
        return args.duplicate_action != "list"
    if args.command == "versions":
        return args.version_action == "activate"
    if args.command == "enrichment":
        return args.enrichment_action == "refresh-openalex"
    if args.command == "vectors":
        # Repairs write collections a running embed stage or a publication may
        # be using; only the read-only status runs alongside them.
        return args.vectors_action != "status"
    # `search`, `screen` and `mcp` only read; a long-running MCP server must
    # never hold the lock a pipeline run needs.
    return False


def main(argv: list[str] | None = None) -> None:
    invoked_as_cli = argv is None
    _configure_logging()
    if invoked_as_cli:
        _install_signal_handlers()
    parser = argparse.ArgumentParser(description="lake-research-map medallion pipeline")
    parser.add_argument("--stage", choices=STAGES, default="all", help="pipeline stage to run")
    parser.add_argument(
        "--execution-id",
        help="run one stage of a workflow under a shared execution id (with --workflow), so a "
        "workflow can run as separate stage processes and still be recorded as one execution",
    )
    parser.add_argument("--workflow", choices=STAGES, help="parent workflow for stage correlation")
    parser.add_argument(
        "--source-policy",
        choices=("append", "snapshot"),
        default="append",
        help="retain absent archived sources (append) or mirror the files currently on disk",
    )
    subparsers = parser.add_subparsers(dest="command")
    duplicates.configure(subparsers)
    versions.configure(subparsers)
    reviews.configure(subparsers)
    enrichment.configure(subparsers)
    evidence.configure(subparsers)
    audit.configure(subparsers)
    maintenance.configure(subparsers)
    qdrant_cli.configure(subparsers)
    # `search`, `screen` and `mcp` (read-only: none takes the pipeline lock).
    api_cli.configure(subparsers)
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])
    lock = (
        runs._pipeline_lock(f"cli-{args.command}-{uuid.uuid4()}")
        if _mutates_published_inputs(args)
        else contextlib.nullcontext()
    )
    try:
        lock.__enter__()
    except PipelineBusyError as exc:
        logger.error("command not started: %s", exc)
        if invoked_as_cli:
            raise SystemExit(2) from exc
        raise
    try:
        if args.command == "duplicates":
            from transform.duplicate_resolution import DuplicateResolutionError

            try:
                duplicates.run(args)
            except DuplicateResolutionError as exc:
                parser.error(str(exc))
        elif args.command == "versions":
            try:
                versions.run(args)
            except ValueError as exc:
                parser.error(str(exc))
        elif args.command == "reviews":
            try:
                reviews.run(args)
            except (OSError, ValueError) as exc:
                parser.error(str(exc))
        elif args.command == "enrichment":
            try:
                enrichment.run(args)
            except ValueError as exc:
                parser.error(str(exc))
        elif args.command == "evidence":
            try:
                evidence.run(args)
            except (OSError, ValueError) as exc:
                parser.error(str(exc))
        elif args.command == "audit":
            try:
                audit.run(args)
            except ValueError as exc:
                parser.error(str(exc))
        elif args.command == "maintenance":
            try:
                maintenance.run(args)
            except ValueError as exc:
                parser.error(str(exc))
        elif args.command == "vectors":
            try:
                qdrant_cli.run(args)
            except (ValueError, VectorStoreUnavailable) as exc:
                parser.error(str(exc))
        elif args.command in ("search", "screen", "mcp"):
            from apps.api.api import RetrievalUnavailable

            try:
                api_cli.run(args)
            except (ValueError, RetrievalUnavailable) as exc:
                parser.error(str(exc))
            except PipelineInterruptedError:
                # Ctrl-C or SIGTERM is how an MCP client stops the server; it
                # is not a pipeline failure.
                logger.info("%s stopped", args.command)
        else:
            if args.execution_id is None and args.workflow is None:
                if args.source_policy == "append":
                    run(args.stage)
                else:
                    run(args.stage, source_policy=args.source_policy)
            else:
                run(
                    args.stage,
                    execution_id=args.execution_id,
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
    finally:
        lock.__exit__(None, None, None)


if __name__ == "__main__":
    main()
