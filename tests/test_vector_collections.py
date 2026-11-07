"""The embed stage, its contract and the `vectors` commands against Qdrant (ADR-08)."""

from __future__ import annotations

import hashlib
from contextlib import contextmanager
from types import SimpleNamespace

import numpy as np
import pytest
from sqlalchemy import select

from apps.qdrant import cli as qdrant_cli
from cli import main as cli_main
from db.gold_models import (
    DatasetArticle,
    DatasetChunk,
    DatasetVersion,
    PublicationState,
)
from db.vector_store import (
    QdrantVectorStore,
    VectorPoint,
    point_id,
)
from quality import embed_contract
from transform.embeddings import EMBED_MODEL_NAME, EMBED_MODEL_REVISION
from transform.publication import publish_vector_alias
from transform.vector_payloads import chunk_payload
from transform.versioned_gold import (
    build_dataset_embeddings,
    fill_vector_collection,
)

# float32 all the way: dividing by `np.sqrt(384)` (a float64 scalar) would promote
# the vector to float64 and produce a BLOB of the wrong length.
_UNIT = np.full(384, 1 / np.sqrt(384), dtype=np.float32)


class CountingEmbedder:
    """A deterministic stand-in for the model that records what it was asked."""

    def __init__(self) -> None:
        self.texts: list[str] = []

    def __call__(self, texts):
        self.texts.extend(texts)
        for text in texts:
            seed = int(hashlib.sha256(text.encode()).hexdigest()[:8], 16)
            vector = np.random.default_rng(seed).normal(size=384).astype(np.float32)
            yield vector / np.linalg.norm(vector)


def _version(session, version_id: str, texts: dict[tuple[str, str, int], str], status="candidate"):
    session.add(DatasetVersion(version_id=version_id, status=status))
    for doi in sorted({doi for doi, _, _ in texts}):
        session.add(
            DatasetArticle(
                dataset_version_id=version_id,
                doi=doi,
                sources=["ieee"],
                title=f"Title {doi}",
                year=2024,
                abstract="An abstract long enough to count as a real abstract for the payload.",
            )
        )
    for (doi, chunk_type, seq), text in texts.items():
        session.add(
            DatasetChunk(
                dataset_version_id=version_id,
                doi=doi,
                chunk_type=chunk_type,
                seq=seq,
                text=text,
                char_len=len(text),
                text_sha256=hashlib.sha256(text.encode()).hexdigest(),
            )
        )
    session.commit()


def _publish(session, version_id: str) -> None:
    session.get(DatasetVersion, version_id).status = "active"
    session.merge(PublicationState(id=1, active_version_id=version_id))
    session.commit()


def test_embed_stage_writes_vectors_to_qdrant_and_only_provenance_to_gold(
    gold_session, vector_store
):
    _version(
        gold_session, "v1", {("10.1/a", "abstract", 0): "alpha", ("10.1/b", "abstract", 0): "beta"}
    )
    embedder = CountingEmbedder()

    stats = build_dataset_embeddings(gold_session, "v1", store=vector_store, embed_fn=embedder)

    assert stats["embedded"] == 2 and stats["vectors_copied"] == 0
    assert vector_store.count("v1") == 2
    rows = gold_session.scalars(select(DatasetChunk)).all()
    assert all(row.embedding_bin is None and row.embedding is None for row in rows)
    assert {row.embed_revision for row in rows} == {EMBED_MODEL_REVISION}
    assert all(row.embedding_normalized and row.embedded_at for row in rows)
    payload = vector_store.fetch("v1", [point_id("10.1/a", "abstract", 0)])
    payload = next(iter(payload.values())).payload
    assert (payload["title"], payload["year"], payload["source"]) == ("Title 10.1/a", 2024, "ieee")
    assert payload["abstract_missing"] is False
    assert all(result.passed for result in embed_contract(gold_session, "v1", vector_store))

    again = build_dataset_embeddings(gold_session, "v1", store=vector_store, embed_fn=embedder)
    assert again["already_embedded"] == 2 and again["embedded"] == 0


def test_payloads_survive_the_commit_between_batches(gold_session, vector_store, monkeypatch):
    """Each batch commits and expunges; nothing loaded before the loop may detach."""
    from transform import versioned_gold

    monkeypatch.setattr(versioned_gold, "EMBED_BATCH_SIZE", 1)
    _version(
        gold_session,
        "v1",
        {
            ("10.1/a", "abstract", 0): "alpha",
            ("10.1/a", "fulltext", 1): "alpha body",
            ("10.1/b", "abstract", 0): "beta",
        },
    )

    stats = build_dataset_embeddings(
        gold_session, "v1", store=vector_store, embed_fn=CountingEmbedder()
    )

    assert stats["embedded"] == 3
    stored = vector_store.scroll("v1")
    assert {point.payload["title"] for point in stored} == {"Title 10.1/a", "Title 10.1/b"}


def test_a_new_version_reuses_the_active_collection_by_text_hash(gold_session, vector_store):
    _version(gold_session, "v1", {("10.1/a", "abstract", 0): "alpha"})
    build_dataset_embeddings(gold_session, "v1", store=vector_store, embed_fn=CountingEmbedder())
    _publish(gold_session, "v1")
    # Same text under a new version, plus one text no store has seen.
    _version(
        gold_session,
        "v2",
        {("10.1/a", "abstract", 0): "alpha", ("10.1/c", "abstract", 0): "gamma"},
    )
    embedder = CountingEmbedder()

    stats = build_dataset_embeddings(gold_session, "v2", store=vector_store, embed_fn=embedder)

    assert embedder.texts == ["gamma"]
    assert (stats["vectors_copied"], stats["embedded"]) == (1, 1)
    assert all(result.passed for result in embed_contract(gold_session, "v2", vector_store))


def test_a_killed_run_resumes_from_its_own_collection(gold_session, vector_store):
    _version(gold_session, "v1", {("10.1/a", "abstract", 0): "alpha"})
    # The upsert reached Qdrant, the Gold commit did not.
    vector = np.ones(384, dtype=np.float32) / np.sqrt(384)
    vector_store.ensure_collection("v1", 384)
    vector_store.upsert(
        "v1",
        [
            VectorPoint(
                point_id("10.1/a", "abstract", 0),
                vector,
                chunk_payload(
                    version_id="v1",
                    doi="10.1/a",
                    chunk_type="abstract",
                    seq=0,
                    text="alpha",
                    text_sha256=hashlib.sha256(b"alpha").hexdigest(),
                    embedded_at=None,
                ),
            )
        ],
    )

    def refuse(_texts):
        raise AssertionError("a vector already in the collection must not be recomputed")

    stats = build_dataset_embeddings(gold_session, "v1", store=vector_store, embed_fn=refuse)

    assert stats["vectors_copied"] == 1
    assert gold_session.scalar(select(DatasetChunk)).embed_revision == EMBED_MODEL_REVISION


def test_vectors_stored_in_mysql_before_adr_08_are_copied_not_recomputed(
    gold_session, vector_store
):
    # The migration case: the published version still holds its vectors as
    # BLOBs, and no `vectors backfill` has run yet.
    _version(gold_session, "pre-adr08", {("10.1/a", "abstract", 0): "alpha"}, status="active")
    legacy = gold_session.scalar(select(DatasetChunk))
    legacy.embedding_bin = _UNIT.tobytes()
    legacy.embed_model = EMBED_MODEL_NAME
    legacy.embed_revision = EMBED_MODEL_REVISION
    gold_session.merge(PublicationState(id=1, active_version_id="pre-adr08"))
    gold_session.commit()
    _version(gold_session, "v2", {("10.1/a", "abstract", 0): "alpha"})

    def refuse(_texts):
        raise AssertionError("a reusable MySQL vector must not be recomputed")

    stats = build_dataset_embeddings(gold_session, "v2", store=vector_store, embed_fn=refuse)

    assert stats["vectors_copied"] == 1
    stored = vector_store.fetch("v2", [point_id("10.1/a", "abstract", 0)], with_vectors=True)
    np.testing.assert_allclose(
        next(iter(stored.values())).vector,
        _UNIT,
        atol=1e-6,
    )


def test_points_without_a_gold_chunk_are_removed(gold_session, vector_store):
    _version(gold_session, "v1", {("10.1/a", "abstract", 0): "alpha"})
    build_dataset_embeddings(gold_session, "v1", store=vector_store, embed_fn=CountingEmbedder())
    vector_store.upsert(
        "v1",
        [VectorPoint(point_id("10.1/gone", "abstract", 0), np.ones(384, dtype=np.float32), {})],
    )
    assert not next(
        result.passed
        for result in embed_contract(gold_session, "v1", vector_store)
        if result.check_id == "embed.orphans"
    )

    stats = build_dataset_embeddings(
        gold_session, "v1", store=vector_store, embed_fn=CountingEmbedder()
    )

    assert stats["orphans_removed"] == 1
    assert all(result.passed for result in embed_contract(gold_session, "v1", vector_store))


def test_the_contract_fails_on_lost_points_changed_text_and_an_unreachable_store(
    gold_session, vector_store
):
    from qdrant_client import QdrantClient

    _version(
        gold_session,
        "v1",
        {("10.1/a", "abstract", 0): "alpha", ("10.1/b", "abstract", 0): "beta"},
    )
    build_dataset_embeddings(gold_session, "v1", store=vector_store, embed_fn=CountingEmbedder())
    vector_store.delete_points("v1", [point_id("10.1/a", "abstract", 0)])
    changed = gold_session.scalar(select(DatasetChunk).where(DatasetChunk.doi == "10.1/b"))
    changed.text = "beta, revised"

    failed = {
        result.check_id: result
        for result in embed_contract(gold_session, "v1", vector_store)
        if not result.passed
    }

    assert set(failed) == {"embed.complete", "embed.text_hash"}
    assert "vectors rebuild" in failed["embed.complete"].details["hint"]

    offline = QdrantVectorStore(
        QdrantClient(url="http://127.0.0.1:9", timeout=2, check_compatibility=False)
    )
    results = embed_contract(gold_session, "v1", offline)
    assert [(result.check_id, result.passed) for result in results] == [
        ("embed.store_reachable", False)
    ]


def test_the_contract_fails_on_a_non_finite_vector(gold_session, vector_store):
    """FR-06: a NaN in a stored vector blocks publication as `embed.finite`.

    Qdrant refuses a NaN at upsert (local mode and server alike), so the bad
    point is written the way one can actually arrive: an `inf` component, which
    the COSINE collection's normalisation turns into NaN on the way in. Every
    other check -- model, revision, dimension, text hash -- still matches, so
    the failure is the finiteness gate's alone.
    """
    _version(
        gold_session,
        "v1",
        {("10.1/a", "abstract", 0): "alpha", ("10.1/b", "abstract", 0): "beta"},
    )
    build_dataset_embeddings(gold_session, "v1", store=vector_store, embed_fn=CountingEmbedder())
    key = point_id("10.1/a", "abstract", 0)
    point = vector_store.fetch("v1", [key])[key]
    poisoned = _UNIT.copy()
    poisoned[0] = np.inf
    with np.errstate(invalid="ignore"):  # inf / inf inside the store's normalisation
        vector_store.upsert("v1", [VectorPoint(key, poisoned, point.payload)])
    stored = vector_store.fetch("v1", [key], with_vectors=True)[key].vector
    assert not np.isfinite(stored).all(), "the store must hold the non-finite vector"

    results = {
        result.check_id: result for result in embed_contract(gold_session, "v1", vector_store)
    }

    failed = {check_id for check_id, result in results.items() if not result.passed}
    assert failed == {"embed.finite"}
    finite = results["embed.finite"]
    assert finite.severity == "error"
    assert finite.observed == 1
    chunk_id = gold_session.scalar(select(DatasetChunk.id).where(DatasetChunk.doi == "10.1/a"))
    assert finite.details["chunk_ids"] == [chunk_id]


def test_the_alias_is_moved_after_publication_and_a_failure_does_not_raise(vector_store):
    from qdrant_client import QdrantClient

    vector_store.ensure_collection("v1", 384)

    assert publish_vector_alias(vector_store, "v1") is True
    assert vector_store.alias_target() == "v1"
    offline = QdrantVectorStore(
        QdrantClient(url="http://127.0.0.1:9", timeout=2, check_compatibility=False)
    )
    assert publish_vector_alias(offline, "v1") is False


@contextmanager
def _session_scope(session):
    yield session


def _patch_cli(monkeypatch, gold_session, vector_store) -> None:
    monkeypatch.setattr(qdrant_cli, "bootstrap", lambda: None)
    monkeypatch.setattr(qdrant_cli, "get_session", lambda layer: gold_session)
    monkeypatch.setattr(qdrant_cli, "get_vector_store", lambda: vector_store)
    monkeypatch.setattr(gold_session, "close", lambda: None)


def test_backfill_copies_complete_mysql_versions_and_reports_legacy_ones(
    monkeypatch, capsys, gold_session, vector_store
):
    _version(gold_session, "published", {("10.1/a", "abstract", 0): "alpha"}, status="published")
    row = gold_session.scalar(select(DatasetChunk))
    row.embedding_bin = _UNIT.tobytes()
    row.embed_model = EMBED_MODEL_NAME
    row.embed_revision = EMBED_MODEL_REVISION
    row.embedding_dim = 384
    row.embedding_dtype = "float32"
    row.embedding_normalized = True
    gold_session.merge(PublicationState(id=1, active_version_id="published"))
    # The legacy import: vectors without the provenance the contract demands.
    _version(gold_session, "legacy", {("10.1/a", "abstract", 0): "alpha"}, status="superseded")
    legacy = gold_session.scalar(
        select(DatasetChunk).where(DatasetChunk.dataset_version_id == "legacy")
    )
    legacy.embedding_bin = row.embedding_bin
    legacy.text_sha256 = None
    gold_session.commit()
    row_id = row.id
    _patch_cli(monkeypatch, gold_session, vector_store)

    qdrant_cli.run(SimpleNamespace(vectors_action="backfill"))

    output = capsys.readouterr().out
    assert "legacy skipped" in output
    assert "embed contract passed" in output
    assert vector_store.versions() == ["published"]
    assert vector_store.alias_target() == "published"
    # A repair never writes Gold rows: the published snapshot keeps its BLOB.
    assert gold_session.get(DatasetChunk, row_id).embedding_bin is not None


def test_rebuild_restores_a_lost_active_collection(monkeypatch, capsys, gold_session, vector_store):
    _version(gold_session, "v1", {("10.1/a", "abstract", 0): "alpha"})
    build_dataset_embeddings(gold_session, "v1", store=vector_store, embed_fn=CountingEmbedder())
    _publish(gold_session, "v1")
    vector_store.drop_collection("v1")
    _patch_cli(monkeypatch, gold_session, vector_store)
    monkeypatch.setattr("transform.versioned_gold._default_embed_fn", lambda: CountingEmbedder())

    qdrant_cli.run(SimpleNamespace(vectors_action="rebuild", version_id=None))

    assert "embed contract passed" in capsys.readouterr().out
    assert vector_store.count("v1") == 1
    assert vector_store.alias_target() == "v1"


def test_fill_without_compute_leaves_unknown_text_unresolved(gold_session, vector_store):
    _version(gold_session, "v1", {("10.1/a", "abstract", 0): "alpha"})

    stats = fill_vector_collection(gold_session, vector_store, "v1", allow_compute=False)

    assert stats["unresolved"] == 1
    assert vector_store.count("v1") == 0


@pytest.mark.parametrize(
    ("action", "locked"),
    [("status", False), ("backfill", True), ("rebuild", True), ("sync-alias", True)],
)
def test_vector_repairs_take_the_pipeline_lock_and_status_does_not(action, locked):
    args = SimpleNamespace(command="vectors", vectors_action=action)

    assert cli_main._mutates_published_inputs(args) is locked
