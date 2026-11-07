---
name: python-testing-conventions
description: pytest conventions for lake-research-map — pure-function-first testing, per-medallion-layer in-memory SQLite fixtures, plain fakes at documented injection points instead of a mocking framework. Use whenever writing or extending tests under tests/, for pipeline transform code or dashboard analytics code.
---

# lake-research-map testing conventions

Applies repo-wide — both `src/{ingest,transform}/` pipeline code and
`src/apps/dashboard/analytics.py`/`forecasting.py` and `apps/api/rank.py` (all pure, no `streamlit`
import, exactly so they're unit-testable). Run with `uv run pytest` (`testpaths = ["tests"]` in
`pyproject.toml`).

## Prefer testing pure functions directly

The dominant pattern in `tests/` is extracting the interesting logic into a small pure function and testing
*that* directly with hand-built inputs, rather than driving a full pipeline stage end-to-end:

- `test_bronze_articles.py` — `normalize_doi`, `_split_bibtex_authors`, `_split_ieee_csv_authors`, `_to_int`
- `test_silver_articles.py` — `normalize_title`, `_merge_group` (plus one full `build_silver_articles` flow)
- `test_gold_articles.py` — `_chunk_text`, `_build_abstract_text`
- `test_retrieval_rank.py` — `BM25Index`, `rrf_fuse`, `first_per_key`, `excerpt`
- `test_retrieval_api.py` — `Retriever` over a hand-built version: SQLite Gold plus the in-process
  Qdrant `vector_store` fixture, with `embed_fn`/`rerank_fn`/`token_len_fn` injected so no model downloads
- `test_analytics_authors.py` — dashboard `analytics.py` functions (`author_year_matrix`,
  `gini_coefficient`, etc.) against hand-built DataFrames

When adding new transform/analytics logic, extract it into a small private function with a clear input/output
contract *specifically so it can be tested this way* — that extraction is part of "done," not a nice-to-have.

## Full-flow tests: one SQLite session per medallion layer

Reach for `tests/conftest.py`'s fixtures (`raw_session`, `bronze_session`, `silver_session`, `gold_session`)
only when a transform stage's end-to-end behavior needs verifying (e.g. `build_silver_articles`'s dedup +
PDF-linking together). Each fixture is a fresh in-memory SQLite engine built from that layer's own
`Base.metadata` — a faithful stand-in for MySQL because nothing in transform code is MySQL-specific. Don't
add a real MySQL dependency outside the `mysql` marker. The `vector_store` fixture is the same idea for Qdrant:
`qdrant-client`'s in-process local mode.

```python
@pytest.fixture
def silver_session():
    session = _sqlite_session(silver_models.Base)
    yield session
    session.close()
```

## No mocking framework: fakes at the injection points

Don't use a mocking framework (`unittest.mock`, `pytest-mock`) in new tests. Four existing files still import
`unittest.mock` (`test_openalex_enrichment.py`, `test_charts.py`, `test_retrieval_benchmark_panel.py`,
`test_population_provenance.py`); they are scheduled to be converted, not copied.

Pure functions and in-memory SQLite sessions cover most of what mocking would be used for. Where a test must
replace a collaborator, pass a plain fake through a documented injection point, or `monkeypatch` that point and
nothing else:

- `session_factory` in `ingest/openalex.py` (the HTTP getter; defaults to `requests.get`);
- `embed_fn` and `store` on the transform builders (`transform/versioned_gold.py`), with the `vector_store`
  fixture as the store;
- `pipeline.get_session`, which `_layer_sessions` calls (`tests/test_pipeline_sessions.py`);
- `transform.versioned_gold._default_embed_fn`, when a stage builds its own embedder
  (`tests/test_vector_collections.py`);
- `embed_fn` / `rerank_fn` / `token_len_fn` on the retriever (`tests/test_retrieval_api.py`).

## Opt-in server markers

Two markers (declared in `pyproject.toml`) select tests that need a real server; the default run skips them:

- `mysql` — `tests/test_mysql_contracts.py`, run with `RUN_MYSQL_INTEGRATION=1 uv run pytest -m mysql`
  against a disposable or explicitly selected MySQL server (touches only `lit_*` tables);
- `qdrant` — `tests/test_qdrant_server.py`, run with `RUN_QDRANT_INTEGRATION=1 uv run pytest -m qdrant` against
  the compose service, for what local mode ignores (payload indexes, exact-search parameters).

Everything else must pass with no MySQL and no Qdrant running.

## `test_raw_bib.py`: a reminder about IEEE's BibTeX gotcha

This file exists specifically to catch the "IEEE `.bib` entries have no separator between them" parsing bug
(see `CLAUDE.md`). Any change to `ingest/raw_bib.py` should keep exercising hand-written `.bib` fixtures that
reproduce that glued-together-entries shape, not just well-formed Elsevier-style fixtures — code that only
passes against clean fixtures can still silently merge/truncate real IEEE records.
