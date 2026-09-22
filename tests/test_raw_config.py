from __future__ import annotations

from datetime import datetime

from lake_research_map.db.raw_models import BibEntry, Config, DatasetSourceFile, SourceFile
from lake_research_map.ingest import raw_config
from lake_research_map.quality import raw_contract


def test_multiple_search_reports_from_one_publisher_are_preserved(
    raw_session, tmp_path, monkeypatch
):
    first = tmp_path / "batch-a" / "config.csv"
    second = tmp_path / "batch-b" / "config.csv"
    first.parent.mkdir()
    second.parent.mkdir()
    first.write_text('"distribution planning"\nyear 2000-2025\nhttps://example/a')
    second.write_text('"hosting capacity"\nyear 2010-2026\nhttps://example/b')

    monkeypatch.setattr(
        raw_config,
        "SEARCH_CONFIG_PATHS",
        (("ieee", first), ("ieee", second)),
    )
    monkeypatch.setattr(raw_config, "relative_path", lambda path: str(path.relative_to(tmp_path)))

    assert raw_config.load_configs(raw_session) == 2
    raw_session.commit()
    rows = raw_session.query(Config).order_by(Config.source_file).all()
    assert [row.source_file for row in rows] == ["batch-a/config.csv", "batch-b/config.csv"]
    assert [row.query_string for row in rows] == [
        '"distribution planning"',
        '"hosting capacity"',
    ]

    assert raw_config.load_configs(raw_session) == 2
    raw_session.commit()
    assert raw_session.query(Config).count() == 2


def test_raw_contract_warns_until_each_export_directory_has_real_provenance(raw_session):
    version_id = "v1"
    bib_path = "data/references/IEEE Xplore/export.bib"
    config_path = "data/references/IEEE Xplore/config.csv"
    common = {
        "source": "ieee",
        "sha256": "a" * 64,
        "size_bytes": 10,
        "mtime": datetime(2026, 9, 21),
        "dataset_version_id": version_id,
    }
    raw_session.add(SourceFile(path=bib_path, kind="bib", **common))
    raw_session.add(
        DatasetSourceFile(dataset_version_id=version_id, path=bib_path, source_revision_id="r1")
    )
    raw_session.add(
        BibEntry(
            source="ieee",
            bib_key="paper",
            entry_type="article",
            source_file=bib_path,
            fields={"title": "Paper"},
        )
    )
    raw_session.commit()

    result = {check.check_id: check for check in raw_contract(raw_session, version_id)}[
        "raw.search_provenance_coverage"
    ]
    assert result.severity == "warning"
    assert result.passed is False
    assert result.details["missing_source_directories"] == [("ieee", "data/references/IEEE Xplore")]

    raw_session.add(SourceFile(path=config_path, kind="config", **common))
    raw_session.add(
        DatasetSourceFile(dataset_version_id=version_id, path=config_path, source_revision_id="r2")
    )
    raw_session.add(
        Config(
            source="ieee",
            source_file=config_path,
            raw_text="TODO: supply the query and URL",
        )
    )
    raw_session.commit()
    placeholder = {check.check_id: check for check in raw_contract(raw_session, version_id)}[
        "raw.search_provenance_coverage"
    ]
    assert placeholder.passed is False

    raw_session.query(
        Config
    ).one().raw_text = '"distribution system planning"\nyear 2000-2025\nhttps://example/search'
    raw_session.commit()
    complete = {check.check_id: check for check in raw_contract(raw_session, version_id)}[
        "raw.search_provenance_coverage"
    ]
    assert complete.passed is True
