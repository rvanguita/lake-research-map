from __future__ import annotations

import os

import pytest
from sqlalchemy import inspect, text
from sqlalchemy.exc import IntegrityError

from lake_research_map.db.bootstrap import _migrate_config_uniqueness
from lake_research_map.db.engines import get_engine

pytestmark = pytest.mark.mysql


@pytest.mark.skipif(
    os.environ.get("RUN_MYSQL_INTEGRATION") != "1",
    reason="set RUN_MYSQL_INTEGRATION=1 for the explicit MySQL acceptance run",
)
def test_mysql_json_blob_null_transaction_and_advisory_lock():
    engine = get_engine("gold")
    table = "lit_mysql_contract_probe"
    with engine.connect() as connection:
        connection.execute(text(f"DROP TABLE IF EXISTS `{table}`"))
        connection.execute(
            text(
                f"CREATE TABLE `{table}` ("
                "id INT PRIMARY KEY, payload JSON NULL, vector LONGBLOB NULL)"
            )
        )
        connection.commit()
        try:
            transaction = connection.begin()
            connection.execute(
                text(
                    f"INSERT INTO `{table}` (id, payload, vector) VALUES "
                    "(1, JSON_OBJECT('ok', TRUE), :vector), (2, NULL, NULL)"
                ),
                {"vector": bytes(range(32))},
            )
            rows = connection.execute(
                text(
                    f"SELECT id, JSON_EXTRACT(payload, '$.ok'), OCTET_LENGTH(vector) "
                    f"FROM `{table}` ORDER BY id"
                )
            ).all()
            assert rows[0][2] == 32
            assert rows[1][1:] == (None, None)
            transaction.rollback()
            assert connection.execute(text(f"SELECT COUNT(*) FROM `{table}`")).scalar_one() == 0
            assert (
                connection.execute(
                    text("SELECT GET_LOCK('lake_research_map_contract_probe', 0)")
                ).scalar_one()
                == 1
            )
            assert (
                connection.execute(
                    text("SELECT RELEASE_LOCK('lake_research_map_contract_probe')")
                ).scalar_one()
                == 1
            )
        finally:
            connection.execute(text(f"DROP TABLE IF EXISTS `{table}`"))
            connection.commit()


@pytest.mark.skipif(
    os.environ.get("RUN_MYSQL_INTEGRATION") != "1",
    reason="set RUN_MYSQL_INTEGRATION=1 for the explicit MySQL acceptance run",
)
def test_mysql_config_uniqueness_migration_is_idempotent_and_transactional():
    engine = get_engine("raw")
    with engine.connect() as connection:
        before_rows = connection.execute(
            text("SELECT id, source, source_file, raw_text FROM lit_config ORDER BY id")
        ).all()

    _migrate_config_uniqueness(engine)
    _migrate_config_uniqueness(engine)

    unique_constraints = inspect(engine).get_unique_constraints("lit_config")
    assert not any(item.get("column_names") == ["source"] for item in unique_constraints)
    assert any(item.get("column_names") == ["source_file"] for item in unique_constraints)
    indexes = inspect(engine).get_indexes("lit_config")
    assert any(
        item.get("column_names") == ["source"] and not item.get("unique") for item in indexes
    )

    with engine.connect() as connection:
        assert (
            connection.execute(
                text("SELECT id, source, source_file, raw_text FROM lit_config ORDER BY id")
            ).all()
            == before_rows
        )
        connection.rollback()

        transaction = connection.begin()
        connection.execute(
            text(
                "INSERT INTO lit_config (source, source_file, raw_text) VALUES "
                "('ieee', 'contract-probe-a/config.csv', 'probe'), "
                "('ieee', 'contract-probe-b/config.csv', 'probe')"
            )
        )
        assert (
            connection.execute(text("SELECT COUNT(*) FROM lit_config")).scalar_one()
            == len(before_rows) + 2
        )
        with pytest.raises(IntegrityError):
            connection.execute(
                text(
                    "INSERT INTO lit_config (source, source_file, raw_text) "
                    "VALUES ('elsevier', 'contract-probe-a/config.csv', 'duplicate')"
                )
            )
        transaction.rollback()

        assert connection.execute(text("SELECT COUNT(*) FROM lit_config")).scalar_one() == len(
            before_rows
        )
