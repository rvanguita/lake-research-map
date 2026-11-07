"""Published snapshots are immutable (Wave P0-1).

A version readers have seen -- active or superseded, `published_at` set -- is
never rewritten. Three paths used to reach one: the builders refused only the
*active* version, `runs._working_version` returned whatever version an
`--execution-id` was bound to, and the raw stage pointed the working version
at any existing version whose fingerprint matched, a superseded one included.
Every one of them now refuses through the same predicate
(`transform.publication.is_mutable`) with a message naming the version and
`versions activate`, which remains the way to serve a published version again.
"""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from types import SimpleNamespace

import numpy as np
import pytest
from sqlalchemy import create_engine, delete, func, select
from sqlalchemy.orm import sessionmaker

import pipeline
import runs
from cli import versions as cli_versions
from db import bronze_models, gold_models, raw_models, silver_models
from db.gold_models import (
    Article,
    DatasetArticle,
    DatasetChunk,
    DatasetDuplicatePair,
    DatasetSemantics,
    DatasetVersion,
    PipelineExecution,
    PublicationState,
)
from db.raw_models import BibEntry
from db.vector_store import VectorPoint, point_id
from ingest.snapshots import ScannedSource, VersionFingerprint
from transform.embeddings import EMBED_MODEL_NAME, EMBED_MODEL_REVISION
from transform.publication import PublishedVersionError, is_mutable, is_published
from transform.vector_payloads import chunk_payload
from transform.versioned_gold import (
    build_dataset_embeddings,
    build_dataset_gold,
    build_dataset_semantics,
    fill_vector_collection,
)

V1 = "1" * 64
V2 = "2" * 64

SOURCE = ScannedSource(
    path="data/elsevier/test.bib",
    source="elsevier",
    kind="bib",
    sha256="a" * 64,
    size_bytes=10,
    mtime=datetime(2026, 9, 21),
    revision_id="b" * 64,
    archive_path="data/.lake_research_map/objects/aa/" + "a" * 64,
)


def _now():
    return datetime.now(UTC).replace(tzinfo=None)


def _fingerprint(version_id: str) -> VersionFingerprint:
    return VersionFingerprint(
        version_id=version_id,
        source_manifest_sha256=hashlib.sha256(version_id.encode()).hexdigest(),
        config_sha256="e" * 64,
        code_revision="commit",
        code_sha256="f" * 64,
        curation_sha256="0" * 64,
    )


def _factories() -> dict:
    factories = {}
    for layer, module in {
        "raw": raw_models,
        "bronze": bronze_models,
        "silver": silver_models,
        "gold": gold_models,
    }.items():
        engine = create_engine("sqlite:///:memory:", future=True)
        module.Base.metadata.create_all(engine)
        factories[layer] = sessionmaker(bind=engine, future=True)
    return factories


class _Corpus:
    """What the fake raw loader ingests, and the fingerprint the raw stage mints for it."""

    def __init__(self) -> None:
        self.fingerprint = _fingerprint(V1)
        self.title = "Planning paper"

    def load_bib(self, session) -> int:
        session.execute(delete(BibEntry))
        session.add(
            BibEntry(
                source="elsevier",
                bib_key="paper",
                entry_type="article",
                source_file=SOURCE.path,
                fields={
                    "title": self.title,
                    "author": "Researcher",
                    "year": "2024",
                    "abstract": f"Distribution network planning study: {self.title}.",
                },
                doi="10.1/paper",
            )
        )
        session.flush()
        return 1


def _fake_embed(session, version_id, *, store):
    """What a successful embed stage leaves behind: points plus Gold provenance."""
    rows = session.scalars(
        select(DatasetChunk)
        .where(DatasetChunk.dataset_version_id == version_id)
        .order_by(DatasetChunk.id)
    ).all()
    store.ensure_collection(version_id, 384)
    now = _now()
    points = []
    for index, row in enumerate(rows, start=1):
        vector = np.full(384, index / 100, dtype=np.float32)
        vector = vector / np.linalg.norm(vector)
        digest = hashlib.sha256(row.text.encode("utf-8")).hexdigest()
        row.text_sha256 = digest
        row.embed_model = EMBED_MODEL_NAME
        row.embed_revision = EMBED_MODEL_REVISION
        row.embedding_dim = 384
        row.embedding_dtype = "float32"
        row.embedding_normalized = True
        row.embedded_at = now
        points.append(
            VectorPoint(
                point_id(row.doi, row.chunk_type, row.seq),
                vector,
                chunk_payload(
                    version_id=version_id,
                    doi=row.doi,
                    chunk_type=row.chunk_type,
                    seq=row.seq,
                    text=row.text,
                    text_sha256=digest,
                    embedded_at=now,
                ),
            )
        )
    store.upsert(version_id, points)
    session.flush()
    return {"embedded": len(rows), "already_embedded": 0, "total_chunks": len(rows)}


def _fake_semantics(session, version_id, *, store):
    chunks = session.scalars(
        select(DatasetChunk)
        .where(DatasetChunk.dataset_version_id == version_id)
        .where(DatasetChunk.chunk_type == "abstract")
    ).all()
    for row in chunks:
        session.add(
            DatasetSemantics(
                dataset_version_id=version_id,
                doi=row.doi,
                relevance_score=0.7,
                offtopic_score=0.2,
                theme_id=0,
                theme_label="Planning",
                map_x=0.0,
                map_y=0.0,
                embed_model=EMBED_MODEL_NAME,
            )
        )
    session.flush()
    return {"articles": len(chunks), "themes": 1, "duplicate_pairs": 0, "embedding_coverage": 1.0}


def _pipeline_harness(monkeypatch, vector_store) -> tuple[dict, _Corpus]:
    """The stages over in-memory layers, with the model-bound builders faked."""
    factories = _factories()
    corpus = _Corpus()
    monkeypatch.setattr(pipeline, "bootstrap", lambda: None)
    monkeypatch.setattr(runs, "get_session", lambda layer: factories[layer]())
    monkeypatch.setattr(pipeline, "scan_sources", lambda: [SOURCE])
    monkeypatch.setattr(pipeline, "build_fingerprint", lambda *args, **kwargs: corpus.fingerprint)
    monkeypatch.setattr(pipeline, "load_configs", lambda session: 0)
    monkeypatch.setattr(pipeline, "load_ieee_csv", lambda session: 0)
    monkeypatch.setattr(pipeline, "load_pdf_inventory", lambda session: 0)
    monkeypatch.setattr(pipeline, "load_bib_entries", corpus.load_bib)
    monkeypatch.setattr(pipeline, "build_dataset_embeddings", _fake_embed)
    monkeypatch.setattr(pipeline, "build_dataset_semantics", _fake_semantics)
    monkeypatch.setattr(pipeline, "get_vector_store", lambda: vector_store)
    # `versions activate` opens its own sessions and store, outside the stages.
    monkeypatch.setattr(cli_versions, "bootstrap", lambda: None)
    monkeypatch.setattr(cli_versions, "get_session", lambda layer: factories[layer]())
    monkeypatch.setattr(cli_versions, "get_vector_store", lambda: vector_store)
    return factories, corpus


def _snapshot(session, version_id: str) -> dict:
    """Row counts of a version's `lit_dataset_*` tables and a digest of its chunk hashes."""
    counts = {
        model.__tablename__: session.scalar(
            select(func.count()).select_from(model).where(model.dataset_version_id == version_id)
        )
        for model in (DatasetArticle, DatasetChunk, DatasetSemantics, DatasetDuplicatePair)
    }
    hashes = sorted(
        session.scalars(
            select(DatasetChunk.text_sha256).where(DatasetChunk.dataset_version_id == version_id)
        ).all()
    )
    assert hashes and all(hashes)
    digest = hashlib.sha256("\n".join(hashes).encode("utf-8")).hexdigest()
    return {**counts, "text_sha256_digest": digest}


def test_is_mutable_is_decided_by_publication_not_by_the_active_flag():
    assert is_mutable(None)
    assert is_mutable(DatasetVersion(version_id="c", status="candidate"))
    assert is_mutable(DatasetVersion(version_id="f", status="failed"))
    assert not is_mutable(DatasetVersion(version_id="a", status="active"))
    assert not is_mutable(DatasetVersion(version_id="s", status="superseded", published_at=_now()))
    # A timestamp alone is enough: readers saw it, whatever the status says.
    assert is_published(DatasetVersion(version_id="odd", status="candidate", published_at=_now()))


def test_builders_and_provenance_writes_refuse_a_superseded_version(
    silver_session, gold_session, vector_store
):
    gold_session.add(DatasetVersion(version_id="v1", status="superseded", published_at=_now()))
    gold_session.add(
        DatasetChunk(
            dataset_version_id="v1",
            doi="10.1/a",
            seq=0,
            chunk_type="abstract",
            text="alpha",
            char_len=5,
        )
    )
    gold_session.commit()

    for write in (
        lambda: build_dataset_gold(silver_session, gold_session, "v1"),
        lambda: build_dataset_embeddings(gold_session, "v1", store=vector_store),
        lambda: build_dataset_semantics(gold_session, "v1", store=vector_store),
        lambda: fill_vector_collection(
            gold_session, vector_store, "v1", write_provenance=True, allow_compute=False
        ),
    ):
        with pytest.raises(PublishedVersionError, match="versions activate --version-id v1"):
            write()

    # The repair path (`vectors backfill|rebuild`) writes Qdrant only, so it
    # may still target a published version.
    stats = fill_vector_collection(gold_session, vector_store, "v1", allow_compute=False)

    assert stats["unresolved"] == 1
    chunk = gold_session.scalars(
        select(DatasetChunk).where(DatasetChunk.dataset_version_id == "v1")
    ).one()
    assert (chunk.text, chunk.embed_model) == ("alpha", None)


def test_working_version_refuses_published_versions_except_the_unchanged_full_run(gold_session):
    gold_session.add_all(
        [
            DatasetVersion(version_id="old", status="superseded", published_at=_now()),
            DatasetVersion(version_id="live", status="active", published_at=_now()),
            DatasetVersion(version_id="next", status="candidate"),
            PipelineExecution(
                execution_id="old-run",
                requested_stage="all",
                trigger="cli",
                dataset_version_id="old",
                status="success",
            ),
            PipelineExecution(
                execution_id="next-run",
                requested_stage="gold",
                trigger="cli",
                dataset_version_id="next",
                status="running",
            ),
            PublicationState(id=1, active_version_id="live", working_version_id="live"),
        ]
    )
    gold_session.commit()

    with pytest.raises(PublishedVersionError, match="version old is already published"):
        runs._working_version(gold_session, "old-run")
    # ...even as a stage of a full run: only the *active* version has an
    # unchanged-run skip, a superseded one has nothing legitimate to do.
    with pytest.raises(PublishedVersionError, match="versions activate --version-id old"):
        runs._working_version(gold_session, "old-run", workflow="all")
    # The working pointer sits on the active version after a publication; a
    # single stage over it would rewrite the snapshot being served.
    with pytest.raises(PublishedVersionError, match="version live is already published"):
        runs._working_version(gold_session, None, workflow="gold")
    assert runs._working_version(gold_session, None, workflow="all") == "live"
    assert runs._working_version(gold_session, "next-run", workflow="gold") == "next"


def test_a_failed_execution_never_marks_a_published_version_failed(gold_session):
    gold_session.add_all(
        [
            DatasetVersion(version_id="old", status="superseded", published_at=_now()),
            PipelineExecution(
                execution_id="stumbled",
                requested_stage="gold",
                trigger="cli",
                dataset_version_id="old",
                status="running",
            ),
        ]
    )
    gold_session.commit()

    runs._finalize_execution(gold_session, "stumbled", "error", "boom")
    gold_session.commit()

    assert gold_session.get(PipelineExecution, "stumbled").status == "error"
    version = gold_session.get(DatasetVersion, "old")
    assert (version.status, version.failed_at, version.failure_reason) == ("superseded", None, None)


def test_a_failed_version_with_the_same_fingerprint_is_still_reset_to_candidate(
    monkeypatch, vector_store
):
    factories, _corpus = _pipeline_harness(monkeypatch, vector_store)
    gold = factories["gold"]()
    gold.add(
        DatasetVersion(version_id=V1, status="failed", failed_at=_now(), failure_reason="boom")
    )
    gold.commit()
    gold.close()

    _, version_id, already_active = pipeline._prepare_raw_version(
        "execution-retry", "raw", "cli", "append"
    )

    assert (version_id, already_active) == (V1, False)
    gold = factories["gold"]()
    try:
        version = gold.get(DatasetVersion, V1)
        assert (version.status, version.failed_at, version.failure_reason) == (
            "candidate",
            None,
            None,
        )
    finally:
        gold.close()


def test_a_superseded_version_survives_every_path_that_used_to_rewrite_it(
    monkeypatch, vector_store
):
    factories, corpus = _pipeline_harness(monkeypatch, vector_store)

    pipeline.run_all(execution_id="execution-v1")
    corpus.fingerprint = _fingerprint(V2)
    corpus.title = "Planning paper, revised"
    pipeline.run_all(execution_id="execution-v2")

    gold = factories["gold"]()
    try:
        assert gold.get(DatasetVersion, V1).status == "superseded"
        assert gold.get(DatasetVersion, V1).published_at is not None
        assert gold.get(DatasetVersion, V2).status == "active"
        before = _snapshot(gold, V1)
    finally:
        gold.close()

    # (c) a stage run under the execution id that built the superseded version.
    with pytest.raises(PublishedVersionError, match="already published") as refused:
        pipeline.run_gold(execution_id="execution-v1")
    assert V1 in str(refused.value)
    assert f"versions activate --version-id {V1}" in str(refused.value)

    # (b) the superseded version's fingerprint minted again by the raw stage.
    corpus.fingerprint = _fingerprint(V1)
    with pytest.raises(
        PublishedVersionError, match="already belongs to published version"
    ) as refused:
        pipeline.run_raw(execution_id="execution-remint")
    assert V1 in str(refused.value)
    assert f"versions activate --version-id {V1}" in str(refused.value)

    gold = factories["gold"]()
    try:
        assert _snapshot(gold, V1) == before
        assert gold.get(DatasetVersion, V1).status == "superseded"
        assert gold.get(DatasetVersion, V2).status == "active"
        state = gold.get(PublicationState, 1)
        assert (state.active_version_id, state.working_version_id) == (V2, V2)
        # Both refusals came before any bookkeeping was written.
        assert gold.get(PipelineExecution, "execution-remint") is None
        assert gold.get(PipelineExecution, "execution-v1").status == "success"
        assert [row.title for row in gold.scalars(select(Article)).all()] == [
            "Planning paper, revised"
        ]
    finally:
        gold.close()

    # `versions activate` is the way to serve the old snapshot again, and it still works.
    cli_versions.run(SimpleNamespace(version_action="activate", version_id=V1))

    gold = factories["gold"]()
    try:
        assert gold.get(PublicationState, 1).active_version_id == V1
        assert gold.get(DatasetVersion, V1).status == "active"
        assert gold.get(DatasetVersion, V2).status == "superseded"
        assert [row.title for row in gold.scalars(select(Article)).all()] == ["Planning paper"]
        assert _snapshot(gold, V1) == before
    finally:
        gold.close()
    assert vector_store.alias_target() == V1
