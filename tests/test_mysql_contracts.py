from __future__ import annotations

import os

import pytest
from sqlalchemy import text

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
