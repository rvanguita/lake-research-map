"""Test db/bootstrap.py additive-column logic and table creation.

The bootstrap runs unattended on every pipeline start, so it must never
drop or rewrite existing data -- this is tested by verifying the additive
column map's invariants and that table creation succeeds on a clean engine.
"""

from __future__ import annotations

import re

from sqlalchemy import create_engine, inspect

from db import bronze_models, gold_models, raw_models, silver_models
from db.bootstrap import _ADDITIVE_COLUMNS

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


def test_additive_columns_are_nullable_and_add_no_index():
    """Every additive column is explicitly nullable and adds no index.

    The map runs unattended on every pipeline start: a NOT NULL column can fail
    or rewrite existing rows, and an index rebuilds a possibly large table. A
    plain ``"NULL" in ddl`` check would also accept ``NOT NULL``, so the
    tokens are inspected instead.
    """
    for table, cols in _ADDITIVE_COLUMNS.items():
        for col_name, (ddl, _layers) in cols.items():
            tokens = re.findall(r"[A-Z_]+", ddl.upper())
            pairs = list(zip(tokens, tokens[1:], strict=False))
            assert "NULL" in tokens, f"{table}.{col_name}: must be declared NULL, got {ddl!r}"
            assert ("NOT", "NULL") not in pairs, (
                f"{table}.{col_name}: additive columns must be nullable, got {ddl!r}"
            )
            for forbidden in ("INDEX", "UNIQUE", "KEY", "PRIMARY"):
                assert forbidden not in tokens, (
                    f"{table}.{col_name}: additive migrations add no {forbidden}, got {ddl!r}"
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


def test_text_sha256_is_unindexed_like_the_deployed_schema():
    """The deployed MySQL chunk tables carry no index on ``text_sha256`` (the
    additive migration adds none), so the test schema must not have one either:
    otherwise a lookup by hash alone is fast here and a full scan in production."""
    engine = create_engine("sqlite:///:memory:", future=True)
    gold_models.Base.metadata.create_all(engine)
    inspector = inspect(engine)
    for table in ("lit_chunks", "lit_dataset_chunks"):
        indexed = {
            column for index in inspector.get_indexes(table) for column in index["column_names"]
        }
        assert "text_sha256" not in indexed, f"{table}.text_sha256 must stay unindexed"
    engine.dispose()


# JSON list columns declared ``Mapped[list]`` (so NOT NULL, ``default=list``)
# before this check existed. Their writers always store a list, so they are
# left as they are; the allowlist keeps a *new* mismatch from slipping in.
_KNOWN_NOT_NULL_MODEL_COLUMNS = {
    ("bronze", "lit_articles", "countries"),
    ("silver", "lit_articles", "countries"),
    ("gold", "lit_articles", "countries"),
    ("gold", "lit_articles", "sources"),
}


def test_model_columns_match_their_additive_migration_nullability():
    """A column created by ``create_all`` and the same column added by the
    additive migration must agree: every migrated column is nullable, so the
    model may not declare it NOT NULL."""
    for layer, module in _MODELS.items():
        tables = module.Base.metadata.tables
        for table, cols in _ADDITIVE_COLUMNS.items():
            if table not in tables:
                continue
            for col_name, (_ddl, layers) in cols.items():
                if layer not in layers or col_name not in tables[table].c:
                    continue
                if (layer, table, col_name) in _KNOWN_NOT_NULL_MODEL_COLUMNS:
                    continue
                assert tables[table].c[col_name].nullable, (
                    f"{layer}.{table}.{col_name}: model is NOT NULL but its migration is nullable"
                )
