"""`versions list|activate`: inspect dataset versions, reactivate a published snapshot."""

from __future__ import annotations

import argparse
import uuid

from sqlalchemy import select

from db.bootstrap import bootstrap
from db.engines import get_session
from db.vector_store import get_vector_store
from pipeline import _version_ready
from runs import _finalize_execution
from transform.publication import materialize_version, publish_vector_alias


def configure(subparsers) -> None:
    versions = subparsers.add_parser("versions", help="Inspect or reactivate dataset versions")
    actions = versions.add_subparsers(dest="version_action", required=True)
    actions.add_parser("list", help="List dataset versions and publication status")
    activate = actions.add_parser("activate", help="Reactivate a validated Gold snapshot")
    activate.add_argument("--version-id", required=True)


def run(args: argparse.Namespace) -> None:
    from db.gold_models import DatasetVersion, PipelineExecution

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
        if version.stats and version.stats.get("kind") == "legacy_import":
            # This used to skip the readiness check for legacy snapshots, but
            # `materialize_version` re-asserts the publication contract anyway,
            # and a legacy import cannot pass it: its chunks carry no embedding
            # revision or text hash and it has no semantic run. The bypass could
            # only ever end in a contract violation, so say so up front.
            raise ValueError(
                "a legacy_import version cannot be re-activated: it predates the embedding "
                "and semantic contracts every publication must pass"
            )
        if not _version_ready(session, version.version_id):
            raise ValueError("the requested version has not passed every required quality gate")
        # Before any write: reactivation re-checks the version's vectors.
        store = get_vector_store()
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
        materialize_version(session, version.version_id, execution_id, store=store)
        _finalize_execution(session, execution_id, "success")
        session.commit()
        publish_vector_alias(store, version.version_id)
        print(f"Activated dataset version {version.version_id}.")
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
