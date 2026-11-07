"""`maintenance recover-stale`: fail executions that stopped heartbeating."""

from __future__ import annotations

import argparse
import uuid
from datetime import UTC, datetime, timedelta

import runs
from db.bootstrap import bootstrap
from db.engines import get_session
from runs import recover_abandoned_executions


def configure(subparsers) -> None:
    maintenance = subparsers.add_parser("maintenance", help="Safe pipeline recovery operations")
    actions = maintenance.add_subparsers(dest="maintenance_action", required=True)
    recover = actions.add_parser("recover-stale", help="Fail stale running executions")
    recover.add_argument("--older-than-minutes", type=int, default=30)


def run(args: argparse.Namespace) -> None:
    if args.older_than_minutes < 1:
        raise ValueError("--older-than-minutes must be positive")
    execution_id = f"recovery-{uuid.uuid4()}"
    # Looked up on `runs` at call time, so a test can replace the lock.
    with runs._pipeline_lock(execution_id):
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
