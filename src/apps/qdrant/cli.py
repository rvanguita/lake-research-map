"""`vectors status|backfill|rebuild|sync-alias`: inspect and repair the Qdrant collections.

The repairs write only Qdrant, never Gold rows (a published snapshot is immutable), and every
action except `status` takes the pipeline lock -- that rule lives with the other commands' lock
rules in `pipeline._mutates_published_inputs`. The store client itself (`db/vector_store.py`) stays in
the core: the embed and semantic stages and publication depend on it, as they do on MySQL.
"""

from __future__ import annotations

import argparse

from sqlalchemy import func, select

from db.bootstrap import bootstrap
from db.engines import get_session
from db.vector_store import get_vector_store
from quality import embed_contract
from transform.versioned_gold import publish_vector_alias


def configure(subparsers) -> None:
    from db.vector_store import ACTIVE_ALIAS

    vectors = subparsers.add_parser(
        "vectors", help="Inspect and repair the Qdrant collections that hold the chunk vectors"
    )
    actions = vectors.add_subparsers(dest="vectors_action", required=True)
    actions.add_parser("status", help="Compare each version's collection with its Gold chunks")
    actions.add_parser(
        "backfill",
        help="Copy vectors still stored as MySQL BLOBs into Qdrant, without re-embedding",
    )
    rebuild = actions.add_parser(
        "rebuild",
        help="Restore a version's collection from its Gold text (embeds only what is missing)",
    )
    rebuild.add_argument(
        "--version", dest="version_id", help="dataset version to restore (default: the active one)"
    )
    actions.add_parser("sync-alias", help=f"Point {ACTIVE_ALIAS} at the active version")


def _backfill_targets(session) -> tuple[list[str], list[tuple[str, int, int]]]:
    """Versions whose every chunk still holds a reusable pre-ADR-08 BLOB.

    Reusable means the provenance the embed contract demands: the pinned model
    and revision plus a text hash. A version with BLOBs but without that
    provenance (the legacy import) is reported, not copied -- it could not pass
    the contract, so its vectors would only look valid.
    """
    from sqlalchemy import and_, case

    from db.gold_models import DatasetChunk
    from transform.embeddings import EMBED_MODEL_NAME, EMBED_MODEL_REVISION

    reusable = and_(
        DatasetChunk.embedding_bin.is_not(None),
        DatasetChunk.embed_model == EMBED_MODEL_NAME,
        DatasetChunk.embed_revision == EMBED_MODEL_REVISION,
        DatasetChunk.text_sha256.is_not(None),
    )
    rows = session.execute(
        select(
            DatasetChunk.dataset_version_id,
            func.count(),
            func.sum(case((reusable, 1), else_=0)),
            func.sum(case((DatasetChunk.embedding_bin.is_not(None), 1), else_=0)),
        ).group_by(DatasetChunk.dataset_version_id)
    ).all()
    eligible = sorted(version for version, total, ok, _ in rows if total and int(ok or 0) == total)
    skipped = sorted(
        (version, int(total), int(ok or 0))
        for version, total, ok, blobs in rows
        if int(blobs or 0) and int(ok or 0) != total
    )
    return eligible, skipped


def run(args: argparse.Namespace) -> None:
    from db.gold_models import DatasetChunk, DatasetVersion
    from db.vector_store import ACTIVE_ALIAS, collection_name
    from transform.versioned_gold import (
        active_version_id,
        fill_vector_collection,
    )

    bootstrap()
    store = get_vector_store()
    session = get_session("gold")
    try:
        active = active_version_id(session)
        action = args.vectors_action
        if action == "status":
            alias = store.alias_target()
            in_sync = "in sync" if alias and alias == active else "OUT OF SYNC"
            print(f"active version: {active or '-'}")
            print(f"{ACTIVE_ALIAS} -> {alias or '-'} ({in_sync})")
            stored = set(store.versions())
            chunk_counts = dict(
                session.execute(
                    select(DatasetChunk.dataset_version_id, func.count()).group_by(
                        DatasetChunk.dataset_version_id
                    )
                ).all()
            )
            for version in session.scalars(
                select(DatasetVersion).order_by(DatasetVersion.created_at.desc())
            ).all():
                chunks = int(chunk_counts.get(version.version_id, 0))
                points = store.count(version.version_id) if version.version_id in stored else None
                state = (
                    "no collection"
                    if points is None
                    else "complete"
                    if points == chunks
                    else f"{points}/{chunks} points"
                )
                print(
                    f"{version.version_id} status={version.status} chunks={chunks} vectors={state}"
                )
            return
        if action == "sync-alias":
            if active is None:
                raise ValueError("no dataset version is published yet")
            if not store.has_collection(active):
                raise ValueError(
                    f"the active version {active} has no collection; run `vectors backfill` "
                    "or `vectors rebuild` first"
                )
            store.point_alias(active)
            print(f"{ACTIVE_ALIAS} -> {collection_name(active)}")
            return

        if action == "backfill":
            targets, skipped = _backfill_targets(session)
            for version_id, total, reusable in skipped:
                print(
                    f"{version_id} skipped: {reusable}/{total} chunks carry a reusable vector "
                    "(pinned model, revision and text hash); this version cannot pass the "
                    "embed contract"
                )
            if not targets:
                print("No version holds a complete set of reusable MySQL vectors.")
        else:
            version_id = args.version_id or active
            if version_id is None:
                raise ValueError("no dataset version is published yet; pass --version")
            if session.get(DatasetVersion, version_id) is None:
                raise ValueError(f"unknown dataset version {version_id}")
            targets = [version_id]
        alias_ready = False
        for version_id in targets:
            stats = fill_vector_collection(
                session, store, version_id, allow_compute=action == "rebuild"
            )
            results = embed_contract(session, version_id, store)
            failed = [r.check_id for r in results if r.severity == "error" and not r.passed]
            verdict = "passed" if not failed else "FAILED: " + ", ".join(failed)
            print(f"{version_id} {stats} embed contract {verdict}")
            if version_id == active and not failed:
                alias_ready = True
        if alias_ready and publish_vector_alias(store, active):
            print(f"{ACTIVE_ALIAS} -> {collection_name(active)}")
    finally:
        session.close()
