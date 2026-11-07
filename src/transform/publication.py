"""Publish and reactivate immutable Gold snapshots.

Publication is the only step that writes the live Gold projections
(`lit_articles`, `lit_chunks`, `lit_semantics`, `lit_duplicate_pairs`): it
copies a validated `lit_dataset_*` snapshot over them and moves the
publication pointer, in one transaction. The version's Qdrant collection is
part of the publication contract; the external alias moves only after the
commit (`publish_vector_alias`). The builders that produce a candidate version
-- Gold rows, vectors, semantics -- live in `transform/versioned_gold.py`, and
import `_state` from here because the pointer row belongs to publication.
"""

from __future__ import annotations

import hashlib
import json
import logging

from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session

from db.gold_models import (
    Article,
    Chunk,
    DatasetArticle,
    DatasetChunk,
    DatasetDuplicatePair,
    DatasetSemantics,
    DatasetVersion,
    DuplicatePair,
    PublicationState,
    Semantics,
)
from db.vector_store import ACTIVE_ALIAS, QdrantVectorStore

logger = logging.getLogger(__name__)

# A version in either state has been served to readers and must never be
# rewritten: `materialize_version` leaves `published_at` set when a later
# publication supersedes a version, so a superseded snapshot is exactly as
# published as the active one -- and exact rollback (`versions activate`)
# depends on it still being what readers saw.
PUBLISHED_STATUSES = frozenset({"active", "superseded"})


class PublishedVersionError(ValueError):
    """A write was aimed at a version that has already been published."""


def is_published(version: DatasetVersion) -> bool:
    """Whether `version` has ever been served to readers.

    Either signal is enough on its own: a legacy import is `active` with no
    timestamp, and a status nobody expects with `published_at` set still
    names a snapshot readers once saw.
    """
    return version.published_at is not None or version.status in PUBLISHED_STATUSES


def is_mutable(version: DatasetVersion | None) -> bool:
    """Whether a builder may rewrite the version's `lit_dataset_*` rows.

    The one predicate behind every refusal -- the Gold, embed and semantic
    builders, the working-version lookup, the raw stage's fingerprint reuse
    and the failure bookkeeping all ask this and nothing else, so a snapshot
    cannot be mutable for one of them and immutable for another. A version id
    with no row is a candidate nothing has published.
    """
    return version is None or not is_published(version)


def activate_hint(version_id: str) -> str:
    """The command that serves a published version again, as every refusal names it."""
    return f"`lake-research-map versions activate --version-id {version_id}`"


def published_version_error(version: DatasetVersion) -> PublishedVersionError:
    """The error every write refused by `is_mutable` raises, naming the way out."""
    return PublishedVersionError(
        f"dataset version {version.version_id} is already published (status "
        f"{version.status}); published snapshots are immutable. Run the complete pipeline "
        "to build a new candidate from changed inputs, or use "
        f"{activate_hint(version.version_id)} to serve this one again"
    )


def _state(session: Session) -> PublicationState:
    state = session.get(PublicationState, 1)
    if state is None:
        state = PublicationState(id=1)
        session.add(state)
        session.flush()
    return state


def _copy_live_to_dataset(session: Session, version_id: str) -> None:
    for row in session.scalars(select(Article)).all():
        session.add(
            DatasetArticle(
                dataset_version_id=version_id,
                **{
                    column.name: getattr(row, column.name)
                    for column in Article.__table__.columns
                    if column.name != "id"
                },
            )
        )
    for row in session.scalars(select(Chunk)).all():
        session.add(
            DatasetChunk(
                dataset_version_id=version_id,
                **{
                    column.name: getattr(row, column.name)
                    for column in Chunk.__table__.columns
                    if column.name != "id"
                },
            )
        )
    for row in session.scalars(select(Semantics)).all():
        session.add(
            DatasetSemantics(
                dataset_version_id=version_id,
                **{
                    column.name: getattr(row, column.name)
                    for column in Semantics.__table__.columns
                    if column.name != "id"
                },
            )
        )
    for row in session.scalars(select(DuplicatePair)).all():
        session.add(
            DatasetDuplicatePair(
                dataset_version_id=version_id,
                doi_a=min(row.doi_a, row.doi_b),
                doi_b=max(row.doi_a, row.doi_b),
                similarity=row.similarity,
                created_at=row.created_at,
            )
        )


def adopt_legacy_publication(session: Session) -> str | None:
    state = _state(session)
    if state.active_version_id or not session.scalar(select(func.count()).select_from(Article)):
        return state.active_version_id
    payload = {
        "articles": [row[0] for row in session.execute(select(Article.doi).order_by(Article.doi))],
        "chunks": [
            tuple(row)
            for row in session.execute(
                select(Chunk.doi, Chunk.chunk_type, Chunk.seq, Chunk.text).order_by(
                    Chunk.doi, Chunk.chunk_type, Chunk.seq
                )
            )
        ],
    }
    version_id = hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode()
    ).hexdigest()
    if session.get(DatasetVersion, version_id) is None:
        session.add(
            DatasetVersion(
                version_id=version_id,
                status="active",
                stats={"kind": "legacy_import", "provenance_complete": False},
            )
        )
        _copy_live_to_dataset(session, version_id)
    state.active_version_id = version_id
    session.flush()
    return version_id


def materialize_version(
    session: Session, version_id: str, execution_id: str, *, store: QdrantVectorStore
) -> None:
    """Atomically replace live Gold projections with a validated snapshot.

    The version's Qdrant collection is part of the publication contract, so a
    pointer can never move to a version whose vectors are missing or stale.
    Moving the external alias is left to the caller, after its commit
    (`publish_vector_alias`): the alias must never point at a version Gold has
    not published.
    """
    from quality import assert_contract, publication_contract

    version = session.get(DatasetVersion, version_id)
    if version is None:
        raise ValueError(f"unknown dataset version {version_id}")
    assert_contract("publication", publication_contract(session, version_id, store))
    articles = session.scalars(
        select(DatasetArticle).where(DatasetArticle.dataset_version_id == version_id)
    ).all()
    if not articles:
        raise ValueError(f"dataset version {version_id} has no Gold article snapshot")

    state = _state(session)
    if state.active_version_id is None:
        adopt_legacy_publication(session)
    previous_id = state.active_version_id

    session.execute(delete(Semantics))
    session.execute(delete(DuplicatePair))
    session.execute(delete(Chunk))
    session.execute(delete(Article))
    session.add_all(
        [
            Article(
                **{
                    column.name: getattr(row, column.name)
                    for column in Article.__table__.columns
                    if column.name != "id"
                }
            )
            for row in articles
        ]
    )
    chunks = session.scalars(
        select(DatasetChunk).where(DatasetChunk.dataset_version_id == version_id)
    ).all()
    session.add_all(
        [
            Chunk(
                **{
                    column.name: getattr(row, column.name)
                    for column in Chunk.__table__.columns
                    if column.name != "id"
                }
            )
            for row in chunks
        ]
    )
    semantic_rows = session.scalars(
        select(DatasetSemantics).where(DatasetSemantics.dataset_version_id == version_id)
    ).all()
    session.add_all(
        [
            Semantics(
                **{
                    column.name: getattr(row, column.name)
                    for column in Semantics.__table__.columns
                    if column.name != "id"
                }
            )
            for row in semantic_rows
        ]
    )
    pair_rows = session.scalars(
        select(DatasetDuplicatePair).where(DatasetDuplicatePair.dataset_version_id == version_id)
    ).all()
    session.add_all(
        [
            DuplicatePair(
                doi_a=row.doi_a,
                doi_b=row.doi_b,
                similarity=row.similarity,
                created_at=row.created_at,
            )
            for row in pair_rows
        ]
    )
    if previous_id and previous_id != version_id:
        previous = session.get(DatasetVersion, previous_id)
        if previous is not None:
            previous.status = "superseded"
    version.status = "active"
    from db.time import naive_utc_now

    version.published_at = naive_utc_now()
    version.failed_at = None
    version.failure_reason = None
    state.active_version_id = version_id
    state.working_version_id = version_id
    state.execution_id = execution_id
    session.flush()


def publish_vector_alias(store: QdrantVectorStore, version_id: str) -> bool:
    """Point the external alias at a version Gold has just published.

    Called after the publication commit. A failure is logged, never raised:
    the publication already happened, internal readers resolve collections
    through the Gold pointer rather than the alias, and `vectors sync-alias`
    repairs it.
    """
    try:
        store.point_alias(version_id)
    except Exception:
        logger.exception(
            "alias %s was not moved to version %s; run `lake-research-map vectors sync-alias`",
            ACTIVE_ALIAS,
            version_id,
        )
        return False
    return True
