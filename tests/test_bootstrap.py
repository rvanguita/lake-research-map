"""Test db/bootstrap.py additive-column logic and table creation.

The bootstrap runs unattended on every pipeline start, so it must never
drop or rewrite existing data -- this is tested by verifying the additive
column map's invariants and that table creation succeeds on a clean engine.
"""

from __future__ import annotations

from sqlalchemy import create_engine, inspect

from lake_research_map.db import bronze_models, gold_models, raw_models, silver_models
from lake_research_map.db.bootstrap import _ADDITIVE_COLUMNS

VALID_LAYERS = {"raw", "bronze", "silver", "gold"}

_MODELS = {
    "raw": raw_models,
    "bronze": bronze_models,
    "silver": silver_models,
    "gold": gold_models,
}


def test_additive_columns_map_is_well_formed():
    """Every entry has a valid DDL string and references only valid layers."""
    for table, cols in _ADDITIVE_COLUMNS.items():
        assert isinstance(table, str) and len(table) > 0
        for col_name, (ddl, layers) in cols.items():
            assert isinstance(col_name, str) and len(col_name) > 0
            assert isinstance(ddl, str) and len(ddl) > 0, f"{table}.{col_name}: empty DDL"
            assert isinstance(layers, tuple) and len(layers) > 0, (
                f"{table}.{col_name}: layers must be a non-empty tuple"
            )
            for layer in layers:
                assert layer in VALID_LAYERS, f"{table}.{col_name}: unknown layer {layer!r}"


def test_additive_columns_all_nullable_or_have_default():
    """Additive columns must not break existing rows -- they need NULL or a DEFAULT."""
    for table, cols in _ADDITIVE_COLUMNS.items():
        for col_name, (ddl, _layers) in cols.items():
            ddl_upper = ddl.upper()
            assert "NULL" in ddl_upper or "DEFAULT" in ddl_upper, (
                f"{table}.{col_name}: additive column DDL must be nullable or "
                f"have a default, got {ddl!r}"
            )


def test_create_tables_on_sqlite_for_each_layer():
    """create_all should succeed against in-memory SQLite for every layer."""
    for layer, module in _MODELS.items():
        engine = create_engine("sqlite:///:memory:", future=True)
        module.Base.metadata.create_all(engine)

        inspector = inspect(engine)
        tables = inspector.get_table_names()

        # Every layer has at least one table.
        assert len(tables) > 0, f"layer {layer!r} created no tables"

        # Gold must have the pipeline runs table.
        if layer == "gold":
            assert "lit_pipeline_runs" in tables
            assert "lit_chunks" in tables
            assert "lit_semantics" in tables
            assert "lit_duplicate_overrides" in tables
            constraints = {
                constraint["name"]
                for constraint in inspector.get_check_constraints("lit_duplicate_overrides")
            }
            assert constraints == {
                "ck_duplicate_override_canonical",
                "ck_duplicate_override_decision",
                "ck_duplicate_override_order",
            }

        engine.dispose()


def test_gold_chunk_has_embedding_bin_column():
    """The Chunk model must expose `embedding_bin` for the binary migration."""
    engine = create_engine("sqlite:///:memory:", future=True)
    gold_models.Base.metadata.create_all(engine)
    inspector = inspect(engine)
    columns = {c["name"] for c in inspector.get_columns("lit_chunks")}
    assert "embedding_bin" in columns
    engine.dispose()


def test_silver_has_rejected_table():
    """The silver layer must define the lit_rejected table."""
    engine = create_engine("sqlite:///:memory:", future=True)
    silver_models.Base.metadata.create_all(engine)
    inspector = inspect(engine)
    assert "lit_rejected" in inspector.get_table_names()
    engine.dispose()


def test_half_configured_role_credentials_are_refused(monkeypatch):
    """Setting one half of a role's pair silently pairs it with the other's fallback.

    Adding `MYSQL_PIPELINE_PASSWORD` without `MYSQL_PIPELINE_USER` made the
    pipeline connect as `root` with the pipeline password, which surfaces only
    as "Access denied for user 'root'" and sends you looking at the server
    rather than at .env.
    """
    import pytest

    from lake_research_map.config import MySQLSettings

    for key in ("MYSQL_PIPELINE_USER", "MYSQL_PIPELINE_PASSWORD"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("LAKE_RESEARCH_MAP_DB_ROLE", "pipeline")
    monkeypatch.setenv("MYSQL_USER", "root")
    monkeypatch.setenv("MYSQL_PASSWORD", "root-secret")

    # Neither half set: the plain credentials are used, as before.
    assert MySQLSettings.from_env().user == "root"

    monkeypatch.setenv("MYSQL_PIPELINE_PASSWORD", "generated")
    with pytest.raises(RuntimeError, match="MYSQL_PIPELINE_USER is not"):
        MySQLSettings.from_env()

    monkeypatch.setenv("MYSQL_PIPELINE_USER", "lake_pipeline")
    settings = MySQLSettings.from_env()
    assert (settings.user, settings.password) == ("lake_pipeline", "generated")

    # The mirror case is refused too.
    monkeypatch.delenv("MYSQL_PIPELINE_PASSWORD")
    with pytest.raises(RuntimeError, match="MYSQL_PIPELINE_PASSWORD is not"):
        MySQLSettings.from_env()
