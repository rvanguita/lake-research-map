"""No silent input loss on the enrichment side (Wave P0-5 b and c).

A corrupt `enrichment_cache.json` used to load as `{}` -- indistinguishable
from the missing file a fresh checkout has -- so bronze was rebuilt without a
single Elsevier count and no message; and `pipeline._enrichment_records`
answered every exception with "no observations", so the fingerprint described
inputs the stage never read.
"""

from __future__ import annotations

import logging
import re
from datetime import UTC, datetime

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

import pipeline
import runs
from db.bronze_models import EnrichmentObservation
from db.gold_models import DatasetVersion, PipelineRun, PublicationState
from ingest import enrichment
from ingest.enrichment import EnrichmentCacheError, load_enrichment_cache
from transform.bronze_articles import build_bronze_articles


def test_a_missing_cache_is_an_empty_cache(tmp_path):
    assert load_enrichment_cache(tmp_path / "absent.json") == {}


@pytest.mark.parametrize(
    "content",
    ["{not json", '["a", "list"]', '"a string"', b"\xff\xfe not utf-8"],
    ids=["invalid-json", "list", "string", "undecodable"],
)
def test_a_corrupt_cache_raises_and_names_the_file(tmp_path, content):
    path = tmp_path / "enrichment_cache.json"
    if isinstance(content, bytes):
        path.write_bytes(content)
    else:
        path.write_text(content, encoding="utf-8")

    with pytest.raises(EnrichmentCacheError, match=re.escape(str(path))):
        load_enrichment_cache(path)


def test_an_unreadable_cache_raises(tmp_path):
    # A directory where the file should be: exists, cannot be read (OSError).
    path = tmp_path / "enrichment_cache.json"
    path.mkdir()

    with pytest.raises(EnrichmentCacheError, match="could not be read"):
        load_enrichment_cache(path)


def test_the_bronze_stage_surfaces_a_corrupt_cache(monkeypatch, tmp_path, vector_store):
    """The error reaches the run row and the caller; nothing between swallows it."""
    from db import bronze_models, gold_models, raw_models, silver_models

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
    monkeypatch.setattr(runs, "get_session", lambda layer: factories[layer]())
    path = tmp_path / "enrichment_cache.json"
    path.write_text("{corrupt", encoding="utf-8")
    monkeypatch.setattr(enrichment, "ENRICHMENT_CACHE_PATH", path)
    gold = factories["gold"]()
    gold.add(DatasetVersion(version_id="v" * 64, status="candidate"))
    gold.add(PublicationState(id=1, working_version_id="v" * 64))
    gold.commit()
    gold.close()

    with pytest.raises(EnrichmentCacheError, match="could not be read"):
        pipeline.run_bronze(execution_id="execution-bronze")

    gold = factories["gold"]()
    try:
        run = gold.scalars(select(PipelineRun).where(PipelineRun.stage == "bronze")).one()
        assert run.status == "error"
        assert "enrichment cache" in run.error_message
    finally:
        gold.close()


def test_build_bronze_articles_does_not_catch_a_corrupt_cache(
    raw_session, bronze_session, monkeypatch, tmp_path
):
    path = tmp_path / "enrichment_cache.json"
    path.write_text("[]", encoding="utf-8")
    monkeypatch.setattr(enrichment, "ENRICHMENT_CACHE_PATH", path)

    with pytest.raises(EnrichmentCacheError, match="JSON object keyed by DOI"):
        build_bronze_articles(raw_session, bronze_session, "v1")


def test_enrichment_records_tolerate_only_a_missing_table(monkeypatch, caplog):
    # A bronze database whose enrichment tables do not exist yet.
    engine = create_engine("sqlite:///:memory:", future=True)
    factory = sessionmaker(bind=engine, future=True)
    monkeypatch.setattr(runs, "get_session", lambda layer: factory())
    caplog.set_level(logging.WARNING, logger="pipeline")

    assert pipeline._enrichment_records() == []
    assert "enrichment observations could not be read" in caplog.text


def test_enrichment_records_propagate_any_other_failure(monkeypatch, bronze_session):
    monkeypatch.setattr(runs, "get_session", lambda layer: bronze_session)
    monkeypatch.setattr(bronze_session, "close", lambda: None)

    def _connection_reset(_session):
        raise RuntimeError("connection reset by peer")

    monkeypatch.setattr(enrichment, "load_enrichment_observations", _connection_reset)

    with pytest.raises(RuntimeError, match="connection reset by peer"):
        pipeline._enrichment_records()


def test_enrichment_records_read_the_latest_successful_observation(monkeypatch, bronze_session):
    monkeypatch.setattr(runs, "get_session", lambda layer: bronze_session)
    monkeypatch.setattr(bronze_session, "close", lambda: None)
    bronze_session.add(
        EnrichmentObservation(
            provider="openalex",
            doi="10.1/a",
            observed_at=datetime.now(UTC).replace(tzinfo=None),
            status="success",
            citation_count=7,
            reference_count=12,
        )
    )
    bronze_session.commit()

    assert pipeline._enrichment_records() == [["10.1/a", 7, 12]]
