from __future__ import annotations

import hashlib
import os
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

from lake_research_map import pipeline
from lake_research_map.db import bronze_models, gold_models, raw_models, silver_models
from lake_research_map.db.bronze_models import Article as BronzeArticle
from lake_research_map.db.gold_models import (
    Article,
    Chunk,
    DatasetArticle,
    DatasetChunk,
    DatasetSemantics,
    DatasetVersion,
    PublicationState,
    SemanticRun,
    Semantics,
)
from lake_research_map.db.raw_models import BibEntry, IeeeCsvRow
from lake_research_map.db.silver_models import Article as SilverArticle
from lake_research_map.ingest.snapshots import (
    ScannedSource,
    VersionFingerprint,
    build_fingerprint,
    diff_manifests,
    scan_sources,
)
from lake_research_map.quality import (
    ContractViolation,
    assert_contract,
    embed_contract,
    semantic_contract,
)
from lake_research_map.transform.bronze_articles import build_bronze_articles
from lake_research_map.transform.embeddings import EMBED_MODEL_NAME, EMBED_MODEL_REVISION
from lake_research_map.transform.versioned_gold import (
    build_dataset_embeddings,
    build_dataset_gold,
    build_dataset_semantics,
    materialize_version,
)


def _write_source_tree(root):
    data = root / "data"
    (data / "ieee").mkdir(parents=True)
    (data / "elsevier").mkdir()
    (data / "articles").mkdir()
    (data / "ieee" / "config.csv").write_text('query "distribution planning"')
    (data / "ieee" / "records.bib").write_text("@article{x,title={X}}")
    (data / "articles" / "paper.pdf").write_bytes(b"%PDF-test")
    (root / "src").mkdir()
    (root / "src" / "pipeline.py").write_text("VERSION = 1\n")
    (root / "pyproject.toml").write_text("[project]\nname='fixture'\n")
    return data


def test_source_snapshot_is_content_addressed_and_logically_stable(tmp_path):
    data = _write_source_tree(tmp_path)
    archive = data / ".lake_research_map" / "objects"

    first = scan_sources(data, repo_root=tmp_path, archive_dir=archive)
    fingerprint_a = build_fingerprint(first, repo_root=tmp_path)
    objects_a = {path for path in archive.rglob("*") if path.is_file()}

    config = data / "ieee" / "config.csv"
    os.utime(config, (config.stat().st_atime, config.stat().st_mtime + 10))
    second = scan_sources(data, repo_root=tmp_path, archive_dir=archive)
    fingerprint_b = build_fingerprint(second, repo_root=tmp_path)

    assert fingerprint_a.version_id == fingerprint_b.version_id
    assert {path for path in archive.rglob("*") if path.is_file()} == objects_a

    config.write_text('query "distribution planning"\nfilter open-access')
    third = scan_sources(data, repo_root=tmp_path, archive_dir=archive)
    fingerprint_c = build_fingerprint(third, repo_root=tmp_path)
    assert fingerprint_c.version_id != fingerprint_a.version_id
    assert len({path for path in archive.rglob("*") if path.is_file()}) == len(objects_a) + 1


def test_manifest_diff_recognizes_rename_without_duplicating_content(tmp_path):
    data = _write_source_tree(tmp_path)
    archive = data / ".lake_research_map" / "objects"
    first = scan_sources(data, repo_root=tmp_path, archive_dir=archive)
    previous = {row.path: (row.revision_id, row.sha256) for row in first}

    original = data / "ieee" / "records.bib"
    original.rename(data / "ieee" / "renamed.bib")
    second = scan_sources(data, repo_root=tmp_path, archive_dir=archive)
    changes = diff_manifests(previous, {row.path: row for row in second})

    renamed = [change for change in changes if change["change_type"] == "renamed"]
    assert len(renamed) == 1
    assert renamed[0]["path"] == "data/ieee/renamed.bib"


def test_append_policy_retains_archived_sources_absent_from_new_batch(raw_session):
    archived_time = datetime(2026, 1, 1, tzinfo=UTC).replace(tzinfo=None)
    raw_session.add(
        raw_models.SourceBlob(
            sha256="a" * 64,
            archive_path="data/.lake_research_map/objects/aa/archived",
            size_bytes=12,
        )
    )
    raw_session.add(
        raw_models.SourceRevision(
            revision_id="r" * 64,
            path="data/references/IEEE Xplore/previous.bib",
            source="ieee",
            kind="bib",
            sha256="a" * 64,
            size_bytes=12,
            mtime=archived_time,
        )
    )
    raw_session.add(
        raw_models.SourceFile(
            path="data/references/IEEE Xplore/previous.bib",
            source="ieee",
            kind="bib",
            sha256="a" * 64,
            size_bytes=12,
            mtime=archived_time,
            source_revision_id="r" * 64,
        )
    )
    raw_session.commit()

    current = ScannedSource(
        path="data/references/IEEE Xplore/current.bib",
        source="ieee",
        kind="bib",
        sha256="b" * 64,
        size_bytes=24,
        mtime=archived_time,
        revision_id="s" * 64,
        archive_path="data/.lake_research_map/objects/bb/current",
    )
    effective = pipeline._append_effective_sources(raw_session, [current])

    assert [source.path for source in effective] == [
        "data/references/IEEE Xplore/current.bib",
        "data/references/IEEE Xplore/previous.bib",
    ]


def test_bronze_rebuild_uses_file_qualified_keys_and_propagates_removal(
    raw_session, bronze_session
):
    fields = {
        "Document Title": "Paper",
        "Publication Year": "2024",
        "Authors": "Author",
    }
    raw_session.add_all(
        [
            IeeeCsvRow(
                row_index=0,
                source_file="data/ieee/export-a.csv",
                fields={**fields, "DOI": "10.1/a"},
                doi="10.1/a",
            ),
            IeeeCsvRow(
                row_index=0,
                source_file="data/ieee/export-b.csv",
                fields={**fields, "DOI": "10.1/b"},
                doi="10.1/b",
            ),
        ]
    )
    raw_session.commit()

    build_bronze_articles(raw_session, bronze_session, "v1")
    bronze_session.commit()
    rows = bronze_session.scalars(select(BronzeArticle)).all()
    assert len(rows) == 2
    assert len({row.source_id for row in rows}) == 2
    assert {row.dataset_version_id for row in rows} == {"v1"}

    raw_session.delete(raw_session.scalar(select(IeeeCsvRow).where(IeeeCsvRow.doi == "10.1/a")))
    raw_session.commit()
    build_bronze_articles(raw_session, bronze_session, "v2")
    bronze_session.commit()

    remaining = bronze_session.scalars(select(BronzeArticle)).all()
    assert [row.doi for row in remaining] == ["10.1/b"]
    assert remaining[0].dataset_version_id == "v2"


def test_versioned_gold_build_does_not_mutate_live_publication(silver_session, gold_session):
    silver_session.add(
        SilverArticle(
            dataset_version_id="v1",
            doi="10.1/candidate",
            sources=["ieee"],
            record_type="article",
            publication_category="journal",
            publication_category_basis="journal_record_type",
            title="Candidate",
            authors=[],
            keywords=[],
            countries=[],
            has_abstract=True,
            has_doi=True,
            is_duplicate_merge=False,
            has_pdf=False,
            is_non_article=False,
            bronze_ids=[1],
        )
    )
    silver_session.commit()
    gold_session.add(DatasetVersion(version_id="v1", status="candidate"))
    gold_session.commit()

    stats = build_dataset_gold(silver_session, gold_session, "v1")

    assert stats["articles"] == 1
    assert gold_session.query(DatasetArticle).count() == 1
    snapshot = gold_session.query(DatasetArticle).one()
    assert (snapshot.publication_category, snapshot.publication_category_basis) == (
        "journal",
        "journal_record_type",
    )
    assert gold_session.query(Article).count() == 0

    version = gold_session.get(DatasetVersion, "v1")
    version.status = "active"
    gold_session.commit()
    for rebuild in (
        lambda: build_dataset_gold(silver_session, gold_session, "v1"),
        lambda: build_dataset_embeddings(gold_session, "v1"),
        lambda: build_dataset_semantics(gold_session, "v1"),
    ):
        with pytest.raises(ValueError, match="already published"):
            rebuild()


def test_semantic_run_persists_the_diagnostics_that_produced_its_map(gold_session, monkeypatch):
    from lake_research_map.dashboard import analytics
    from lake_research_map.transform import semantics, versioned_gold

    gold_session.add(DatasetVersion(version_id="v1", status="candidate"))
    vectors = [
        np.array([1.0, 0.0], dtype=np.float32),
        np.array([0.0, 1.0], dtype=np.float32),
    ]
    for index, vector in enumerate(vectors):
        gold_session.add(
            DatasetChunk(
                dataset_version_id="v1",
                doi=f"10.1/{index}",
                seq=0,
                chunk_type="abstract",
                text=f"Article {index}",
                char_len=9,
                embedding_bin=vector.tobytes(),
            )
        )
    gold_session.commit()

    reduced = np.array([[0.0, 0.0], [1.0, 1.0]])
    monkeypatch.setattr(versioned_gold, "reduced_space", lambda _matrix: reduced)
    monkeypatch.setattr(
        versioned_gold,
        "discover_themes",
        lambda _matrix, _texts: (np.array([0, 1]), {0: "A", 1: "B"}),
    )
    monkeypatch.setattr(versioned_gold, "project_2d", lambda _matrix: reduced)
    monkeypatch.setattr(versioned_gold, "near_duplicate_pairs", lambda *_args: [])
    monkeypatch.setattr(
        analytics,
        "semantic_stability_diagnostics",
        lambda *_args: {"valid": True, "bootstrap_ari_mean": 0.9},
    )
    monkeypatch.setattr(
        semantics,
        "theme_sweep",
        lambda _matrix: [{"k": 2, "silhouette": 0.5}],
    )

    versioned_gold.build_dataset_semantics(
        gold_session,
        "v1",
        anchor_vectors=(vectors[0], vectors[1]),
    )

    run = gold_session.query(SemanticRun).one()
    assert run.stability == {
        "valid": True,
        "bootstrap_ari_mean": 0.9,
        "k_sweep": [{"k": 2, "silhouette": 0.5}],
    }


def _candidate(session, version_id: str, doi: str, value: float) -> None:
    vector = np.full(384, value, dtype=np.float32)
    session.add(DatasetVersion(version_id=version_id, status="candidate"))
    session.add(
        DatasetArticle(
            dataset_version_id=version_id,
            doi=doi,
            sources=["ieee"],
            title=doi,
            publication_category="journal",
            publication_category_basis="journal_record_type",
        )
    )
    session.add(
        DatasetChunk(
            dataset_version_id=version_id,
            doi=doi,
            seq=0,
            chunk_type="abstract",
            text=doi,
            char_len=len(doi),
            embedding=vector.tolist(),
            embedding_bin=vector.tobytes(),
            embed_model=EMBED_MODEL_NAME,
            text_sha256=hashlib.sha256(doi.encode("utf-8")).hexdigest(),
            embed_revision=EMBED_MODEL_REVISION,
            embedding_dim=384,
            embedding_dtype="float32",
            embedding_normalized=True,
        )
    )
    session.add(
        DatasetSemantics(
            dataset_version_id=version_id,
            doi=doi,
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


def test_quality_gates_reject_incompatible_embedding(gold_session):
    _candidate(gold_session, "a" * 64, "10.1/a", 0.1)
    chunk = gold_session.scalar(select(DatasetChunk))
    chunk.embedding_bin = b"short"
    results = embed_contract(gold_session, "a" * 64)

    assert any(result.check_id == "embed.compatible" and not result.passed for result in results)
    try:
        assert_contract("embed", results)
    except Exception as exc:
        assert "embed.compatible" in str(exc)
    else:
        raise AssertionError("incompatible embeddings must block publication")


def test_materialization_rechecks_publication_contract(gold_session):
    version_id = "a" * 64
    _candidate(gold_session, version_id, "10.1/a", 0.1)
    chunk = gold_session.scalar(select(DatasetChunk))
    chunk.embedding_bin = b"invalid"

    with pytest.raises(ContractViolation, match="embed.compatible"):
        materialize_version(gold_session, version_id, "execution-invalid")

    assert gold_session.query(Article).count() == 0
    assert gold_session.get(DatasetVersion, version_id).status == "candidate"


def test_publication_and_reactivation_restore_exact_gold_snapshot(gold_session):
    first_id = "1" * 64
    second_id = "2" * 64
    _candidate(gold_session, first_id, "10.1/first", 0.1)
    assert all(result.passed for result in semantic_contract(gold_session, first_id))
    materialize_version(gold_session, first_id, "execution-1")
    gold_session.commit()

    _candidate(gold_session, second_id, "10.1/second", 0.2)
    materialize_version(gold_session, second_id, "execution-2")
    gold_session.commit()
    assert [row.doi for row in gold_session.scalars(select(Article)).all()] == ["10.1/second"]

    materialize_version(gold_session, first_id, "execution-rollback")
    gold_session.commit()

    assert [row.doi for row in gold_session.scalars(select(Article)).all()] == ["10.1/first"]
    assert [row.doi for row in gold_session.scalars(select(Chunk)).all()] == ["10.1/first"]
    assert [row.doi for row in gold_session.scalars(select(Semantics)).all()] == ["10.1/first"]
    state = gold_session.get(PublicationState, 1)
    assert state.active_version_id == first_id


@contextmanager
def _null_lock(execution_id):
    """SQLite has no advisory lock; the recovery path takes one unconditionally."""
    yield


def test_versions_cli_dispatches_without_starting_pipeline(monkeypatch):
    called = []
    monkeypatch.setattr(pipeline, "_run_version_command", called.append)

    pipeline.main(["versions", "list"])

    assert len(called) == 1
    assert called[0].version_action == "list"


def test_pipeline_lock_rejects_another_mysql_process(monkeypatch):
    class Dialect:
        name = "mysql"

    class Bind:
        dialect = Dialect()

    class Result:
        def scalar(self):
            return 0

    class FakeSession:
        def __init__(self):
            self.closed = False

        def get_bind(self):
            return Bind()

        def execute(self, *_args, **_kwargs):
            return Result()

        def close(self):
            self.closed = True

    session = FakeSession()
    monkeypatch.setattr(pipeline, "get_session", lambda layer: session)

    with pytest.raises(pipeline.PipelineBusyError, match="another pipeline execution"):
        with pipeline._pipeline_lock("execution-2"):
            raise AssertionError("the lock must reject the second process")

    assert session.closed is True


def test_pipeline_lock_releases_mysql_advisory_lock(monkeypatch):
    class Dialect:
        name = "mysql"

    class Bind:
        dialect = Dialect()

    class Result:
        def __init__(self, value):
            self.value = value

        def scalar(self):
            return self.value

    class FakeSession:
        def __init__(self):
            self.calls = []
            self.closed = False

        def get_bind(self):
            return Bind()

        def execute(self, statement, *_args, **_kwargs):
            self.calls.append(str(statement))
            return Result(1)

        def rollback(self):
            self.calls.append("rollback")

        def close(self):
            self.closed = True

    session = FakeSession()
    monkeypatch.setattr(pipeline, "get_session", lambda layer: session)

    with pipeline._pipeline_lock("execution-1"):
        pass

    assert any("GET_LOCK" in call for call in session.calls)
    assert any("RELEASE_LOCK" in call for call in session.calls)
    assert session.closed is True


def test_full_pipeline_correlates_stages_and_publishes_only_after_gates(monkeypatch):
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

    monkeypatch.setattr(pipeline, "bootstrap", lambda: None)
    monkeypatch.setattr(pipeline, "get_session", lambda layer: factories[layer]())
    source = ScannedSource(
        path="data/elsevier/test.bib",
        source="elsevier",
        kind="bib",
        sha256="a" * 64,
        size_bytes=10,
        mtime=pipeline.datetime(2026, 9, 21),
        revision_id="b" * 64,
        archive_path="data/.lake_research_map/objects/aa/" + "a" * 64,
    )
    fingerprint = VersionFingerprint(
        version_id="c" * 64,
        source_manifest_sha256="d" * 64,
        config_sha256="e" * 64,
        code_revision="commit",
        code_sha256="f" * 64,
        curation_sha256="0" * 64,
    )
    monkeypatch.setattr(pipeline, "scan_sources", lambda: [source])
    monkeypatch.setattr(pipeline, "build_fingerprint", lambda *args, **kwargs: fingerprint)
    monkeypatch.setattr(pipeline, "load_configs", lambda session: 0)
    monkeypatch.setattr(pipeline, "load_ieee_csv", lambda session: 0)
    monkeypatch.setattr(pipeline, "load_pdf_inventory", lambda session: 0)

    def load_bib(session):
        if session.query(BibEntry).count() == 0:
            session.add(
                BibEntry(
                    source="elsevier",
                    bib_key="paper",
                    entry_type="article",
                    source_file=source.path,
                    fields={
                        "title": "Planning paper",
                        "author": "Researcher",
                        "year": "2024",
                        "abstract": "Distribution network planning study.",
                    },
                    doi="10.1/paper",
                )
            )
            session.flush()
        return 1

    monkeypatch.setattr(pipeline, "load_bib_entries", load_bib)

    def embed(session, version_id):
        vector = np.full(384, 0.1, dtype=np.float32)
        rows = session.scalars(
            select(DatasetChunk).where(DatasetChunk.dataset_version_id == version_id)
        ).all()
        for row in rows:
            row.embedding = vector.tolist()
            row.embedding_bin = vector.tobytes()
            row.embed_model = EMBED_MODEL_NAME
            row.text_sha256 = hashlib.sha256(row.text.encode("utf-8")).hexdigest()
            row.embed_revision = EMBED_MODEL_REVISION
            row.embedding_dim = 384
            row.embedding_dtype = "float32"
            row.embedding_normalized = True
        session.flush()
        return {"embedded": len(rows), "already_embedded": 0, "total_chunks": len(rows)}

    def semantics(session, version_id):
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
        return {
            "articles": len(chunks),
            "themes": 1,
            "duplicate_pairs": 0,
            "embedding_coverage": 1.0,
        }

    monkeypatch.setattr(pipeline, "build_dataset_embeddings", embed)
    monkeypatch.setattr(pipeline, "build_dataset_semantics", semantics)

    pipeline.run_all(execution_id="parent-execution")

    gold_session = factories["gold"]()
    runs = gold_session.scalars(select(gold_models.PipelineRun)).all()
    assert [row.stage for row in runs] == ["raw", "bronze", "silver", "gold", "embed", "semantic"]
    assert {row.execution_id for row in runs} == {"parent-execution"}
    assert {row.dataset_version_id for row in runs} == {fingerprint.version_id}
    assert all(row.status == "success" for row in runs)
    assert gold_session.get(PublicationState, 1).active_version_id == fingerprint.version_id
    assert [row.doi for row in gold_session.scalars(select(Article)).all()] == ["10.1/paper"]
    assert gold_session.query(gold_models.QualityResult).count() > 0
    gold_session.close()

    pipeline.run_all(execution_id="parent-unchanged")
    gold_session = factories["gold"]()
    repeated = gold_session.scalars(
        select(gold_models.PipelineRun).where(
            gold_models.PipelineRun.execution_id == "parent-unchanged"
        )
    ).all()
    assert len(repeated) == 6
    assert all(row.status == "skipped" for row in repeated)
    assert {row.dataset_version_id for row in repeated} == {fingerprint.version_id}
    assert gold_session.query(DatasetVersion).count() == 1
    gold_session.close()


def test_golden_corpus_add_edit_rename_remove_and_reactivate(monkeypatch, tmp_path):
    """Exercise reconciliation and publication over a complete mutation sequence."""
    from lake_research_map.ingest import hashing, raw_bib, raw_config, raw_csv, raw_pdfs
    from lake_research_map.transform import bronze_articles

    data = _write_source_tree(tmp_path)
    (data / "articles" / "paper.pdf").unlink()
    first_bib = data / "elsevier" / "first.bib"
    second_bib = data / "elsevier" / "second.bib"

    def write_bib(path: Path, key: str, doi: str, title: str) -> None:
        path.write_text(
            "\n".join(
                [
                    f"@article{{{key},",
                    f"  title={{{title}}},",
                    "  author={Researcher, Ada},",
                    "  year={2024},",
                    "  journal={Grid Journal},",
                    f"  abstract={{Distribution network planning study for {key}.}},",
                    f"  doi={{{doi}}}",
                    "}",
                ]
            ),
            encoding="utf-8",
        )

    write_bib(first_bib, "first", "10.1000/first", "First planning study")
    (data / "ieee" / "records.bib").unlink()

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

    monkeypatch.setattr(pipeline, "bootstrap", lambda: None)
    monkeypatch.setattr(pipeline, "get_session", lambda layer: factories[layer]())
    monkeypatch.setattr(raw_bib, "IEEE_BIB_DIRS", (data / "ieee",))
    monkeypatch.setattr(raw_bib, "ELSEVIER_BIB_DIRS", (data / "elsevier",))
    monkeypatch.setattr(
        raw_config,
        "SEARCH_CONFIG_PATHS",
        (("ieee", data / "ieee" / "config.csv"), ("elsevier", data / "elsevier" / "config.csv")),
    )
    monkeypatch.setattr(raw_csv, "IEEE_DIR", data / "ieee")
    monkeypatch.setattr(raw_pdfs, "ARTICLES_DIR", data / "articles")

    def fixture_relative_path(path) -> str:
        return Path(path).resolve().relative_to(tmp_path).as_posix()

    for module in (hashing, raw_bib, raw_config, raw_csv, raw_pdfs):
        monkeypatch.setattr(module, "relative_path", fixture_relative_path)
    monkeypatch.setattr(bronze_articles, "load_enrichment_cache", lambda: {})
    monkeypatch.setattr(
        pipeline,
        "scan_sources",
        lambda: scan_sources(
            data,
            repo_root=tmp_path,
            archive_dir=data / ".lake_research_map" / "objects",
        ),
    )

    def fixture_fingerprint(sources, *, curation_records=None, enrichment_records=None):
        return build_fingerprint(
            sources,
            curation_records=curation_records,
            enrichment_records=enrichment_records,
            repo_root=tmp_path,
        )

    monkeypatch.setattr(pipeline, "build_fingerprint", fixture_fingerprint)

    def embed(session, version_id):
        rows = session.scalars(
            select(DatasetChunk).where(DatasetChunk.dataset_version_id == version_id)
        ).all()
        for index, row in enumerate(rows, start=1):
            vector = np.full(384, index / 100, dtype=np.float32)
            row.embedding = vector.tolist()
            row.embedding_bin = vector.tobytes()
            row.embed_model = EMBED_MODEL_NAME
            row.text_sha256 = hashlib.sha256(row.text.encode("utf-8")).hexdigest()
            row.embed_revision = EMBED_MODEL_REVISION
            row.embedding_dim = 384
            row.embedding_dtype = "float32"
            row.embedding_normalized = True
        session.flush()
        return {"embedded": len(rows), "already_embedded": 0, "total_chunks": len(rows)}

    def semantics(session, version_id):
        chunks = session.scalars(
            select(DatasetChunk)
            .where(DatasetChunk.dataset_version_id == version_id)
            .where(DatasetChunk.chunk_type == "abstract")
        ).all()
        for index, row in enumerate(chunks):
            session.add(
                DatasetSemantics(
                    dataset_version_id=version_id,
                    doi=row.doi,
                    relevance_score=0.7,
                    offtopic_score=0.2,
                    theme_id=index,
                    theme_label=f"Theme {index}",
                    map_x=float(index),
                    map_y=0.0,
                    embed_model=EMBED_MODEL_NAME,
                )
            )
        session.flush()
        return {
            "articles": len(chunks),
            "themes": len(chunks),
            "duplicate_pairs": 0,
            "embedding_coverage": 1.0,
        }

    monkeypatch.setattr(pipeline, "build_dataset_embeddings", embed)
    monkeypatch.setattr(pipeline, "build_dataset_semantics", semantics)

    def run_and_assert(execution_id: str, expected_dois: set[str]) -> str:
        pipeline.run_all(execution_id=execution_id, source_policy="snapshot")
        raw_session = factories["raw"]()
        bronze_session = factories["bronze"]()
        silver_session = factories["silver"]()
        gold_session = factories["gold"]()
        try:
            assert {row.doi for row in raw_session.scalars(select(BibEntry)).all()} == expected_dois
            assert {
                row.doi for row in bronze_session.scalars(select(BronzeArticle)).all()
            } == expected_dois
            assert {
                row.doi for row in silver_session.scalars(select(SilverArticle)).all()
            } == expected_dois
            assert {row.doi for row in gold_session.scalars(select(Article)).all()} == expected_dois
            state = gold_session.get(PublicationState, 1)
            assert state.active_version_id is not None
            return state.active_version_id
        finally:
            raw_session.close()
            bronze_session.close()
            silver_session.close()
            gold_session.close()

    initial_version = run_and_assert("golden-initial", {"10.1000/first"})

    write_bib(second_bib, "second", "10.1000/second", "Second planning study")
    added_version = run_and_assert("golden-add", {"10.1000/first", "10.1000/second"})

    write_bib(first_bib, "first", "10.1000/first", "First planning study revised")
    edited_version = run_and_assert("golden-edit", {"10.1000/first", "10.1000/second"})
    gold_session = factories["gold"]()
    revised = gold_session.scalar(select(Article).where(Article.doi == "10.1000/first"))
    assert revised.title == "First planning study revised"
    gold_session.close()

    renamed_bib = data / "elsevier" / "renamed-second.bib"
    second_bib.rename(renamed_bib)
    renamed_version = run_and_assert("golden-rename", {"10.1000/first", "10.1000/second"})
    raw_session = factories["raw"]()
    rename_changes = raw_session.scalars(
        select(raw_models.SourceChange).where(
            raw_models.SourceChange.execution_id == "golden-rename"
        )
    ).all()
    assert [
        (row.change_type, row.path) for row in rename_changes if row.change_type == "renamed"
    ] == [("renamed", "data/elsevier/renamed-second.bib")]
    raw_session.close()

    first_bib.unlink()
    removed_version = run_and_assert("golden-remove", {"10.1000/second"})
    raw_session = factories["raw"]()
    removal_changes = raw_session.scalars(
        select(raw_models.SourceChange).where(
            raw_models.SourceChange.execution_id == "golden-remove"
        )
    ).all()
    assert ("removed", "data/elsevier/first.bib") in {
        (row.change_type, row.path) for row in removal_changes
    }
    raw_session.close()

    assert (
        len({initial_version, added_version, edited_version, renamed_version, removed_version}) == 5
    )

    pipeline._run_version_command(
        SimpleNamespace(version_action="activate", version_id=renamed_version)
    )
    gold_session = factories["gold"]()
    try:
        assert gold_session.get(PublicationState, 1).active_version_id == renamed_version
        assert {row.doi for row in gold_session.scalars(select(Article)).all()} == {
            "10.1000/first",
            "10.1000/second",
        }
        revised = gold_session.scalar(select(Article).where(Article.doi == "10.1000/first"))
        assert revised.title == "First planning study revised"
    finally:
        gold_session.close()


def test_recover_stale_clears_executions_with_no_heartbeat(monkeypatch, gold_session):
    """An abandoned run usually has NO heartbeat at all, not an old one.

    Requiring `heartbeat_at IS NOT NULL` made the recovery silently skip every
    execution killed before its first batch (and every one predating the
    column), which is exactly the population it exists to clear.
    """
    from lake_research_map.db.gold_models import PipelineExecution, PipelineRun

    stale_start = datetime.now(UTC).replace(tzinfo=None) - timedelta(hours=6)
    gold_session.add(
        PipelineExecution(
            execution_id="abandoned",
            requested_stage="all",
            trigger="cli",
            status="running",
            started_at=stale_start,
            heartbeat_at=None,
        )
    )
    gold_session.add(
        PipelineRun(
            execution_id="abandoned",
            stage="embed",
            status="running",
            started_at=stale_start,
            # The model declares these NOT NULL, so an in-flight row carries
            # placeholders until the stage finishes; recovery must overwrite them.
            finished_at=stale_start,
            duration_seconds=0.0,
            sequence=1,
            attempt=1,
        )
    )
    gold_session.commit()

    monkeypatch.setattr(pipeline, "bootstrap", lambda: None)
    monkeypatch.setattr(pipeline, "get_session", lambda layer: gold_session)
    monkeypatch.setattr(pipeline, "_pipeline_lock", _null_lock)
    monkeypatch.setattr(gold_session, "close", lambda: None)

    pipeline._run_maintenance_command(
        SimpleNamespace(maintenance_action="recover-stale", older_than_minutes=30)
    )

    execution = gold_session.get(PipelineExecution, "abandoned")
    assert execution.status == "error"
    assert execution.error_message == "recovered after stale heartbeat"
    run_row = gold_session.scalars(
        select(PipelineRun).where(PipelineRun.execution_id == "abandoned")
    ).one()
    assert run_row.status == "error"


def test_new_enrichment_observations_mint_a_new_version(tmp_path):
    """A refresh has to be publishable: its observations are a bronze input."""
    before = build_fingerprint([], repo_root=tmp_path)
    same = build_fingerprint([], enrichment_records=[], repo_root=tmp_path)
    after = build_fingerprint([], enrichment_records=[["10.1/a", 7, 12]], repo_root=tmp_path)
    later = build_fingerprint([], enrichment_records=[["10.1/a", 9, 12]], repo_root=tmp_path)

    # No observations: the id an install without enrichment already had.
    assert same.version_id == before.version_id
    assert after.version_id != before.version_id
    assert later.version_id != after.version_id


def test_a_presentation_only_edit_does_not_mint_a_new_version(tmp_path):
    from lake_research_map.ingest.snapshots import code_fingerprint

    page = tmp_path / "src/lake_research_map/dashboard/pages/overview.py"
    analytics = tmp_path / "src/lake_research_map/dashboard/analytics.py"
    page.parent.mkdir(parents=True)
    page.write_text("CAPTION = 'a'\n")
    analytics.write_text("K = 1\n")
    _, first = code_fingerprint(tmp_path)

    page.write_text("CAPTION = 'b'\n")
    _, caption_edit = code_fingerprint(tmp_path)
    analytics.write_text("K = 2\n")
    _, analytics_edit = code_fingerprint(tmp_path)

    assert caption_edit == first
    # analytics.py feeds the semantic stage's persisted diagnostics.
    assert analytics_edit != first


def test_a_stage_records_the_parent_as_input_and_the_candidate_as_output(gold_session):
    """FR-07: lineage must distinguish what a stage read from what it wrote."""
    from lake_research_map import pipeline
    from lake_research_map.db.gold_models import DatasetVersion, PipelineRun

    gold_session.add_all(
        [
            DatasetVersion(version_id="v-parent", status="published"),
            DatasetVersion(version_id="v-child", parent_version_id="v-parent", status="candidate"),
        ]
    )
    gold_session.commit()

    run_id, _ = pipeline._begin_stage(gold_session, "silver", "exec-1", "v-child")

    run = gold_session.get(PipelineRun, run_id)
    assert (run.input_version_id, run.output_version_id) == ("v-parent", "v-child")


def test_the_first_version_has_no_input_version(gold_session):
    from lake_research_map import pipeline
    from lake_research_map.db.gold_models import DatasetVersion, PipelineRun

    gold_session.add(DatasetVersion(version_id="v-first", status="candidate"))
    gold_session.commit()

    run_id, _ = pipeline._begin_stage(gold_session, "raw", "exec-1", "v-first")

    assert gold_session.get(PipelineRun, run_id).input_version_id is None
