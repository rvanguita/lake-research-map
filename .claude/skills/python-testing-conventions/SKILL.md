---
name: python-testing-conventions
description: pytest conventions for lake-research-map — pure-function-first testing, per-medallion-layer in-memory SQLite fixtures, plain fakes at documented injection points instead of a mocking framework. Use whenever writing or extending tests under tests/, for pipeline transform code or dashboard analytics code.
---

# lake-research-map testing conventions

Applies repo-wide — both `src/{ingest,transform}/` pipeline code and
the `src/apps/dashboard/analytics/` package, `forecasting.py` and `apps/api/rank.py` (all pure, no `streamlit`
import, exactly so they're unit-testable). Run with `uv run pytest` (`testpaths = ["tests"]` in
`pyproject.toml`).

## Prefer testing pure functions directly

The dominant pattern in `tests/` is extracting the interesting logic into a small pure function and testing
*that* directly with hand-built inputs, rather than driving a full pipeline stage end-to-end:

- `test_doi.py` — `ingest/doi.py::normalize_doi`, the one DOI normalizer every caller imports
- `test_bronze_articles.py` — `_split_bibtex_authors`, `_split_ieee_csv_authors`, `_to_int`
- `test_silver_articles.py` — `normalize_title`, `_merge_group` (plus one full `build_silver_articles` flow)
- `test_gold_articles.py` — `_chunk_text`, `_build_abstract_text`
- `test_retrieval_rank.py` — `BM25Index`, `rrf_fuse`, `first_per_key`, `excerpt`
- `test_retrieval_api.py` — `Retriever` over a hand-built version: SQLite Gold plus the in-process
  Qdrant `vector_store` fixture, with `embed_fn`/`rerank_fn`/`token_len_fn` injected so no model downloads
- `test_analytics_authors.py` — `apps/dashboard/analytics/authors.py` functions (`author_year_matrix`,
  `gini_coefficient`, etc.) against hand-built DataFrames
- `test_dashboard_page_analytics.py` — the pure functions extracted from the Topics, Researchers, Quality
  and Production pages, pinned without starting Streamlit

Three structural tests sit beside them and fail on a layout regression rather than a wrong number:

- `test_dashboard_extracted_views.py` — headless renders of each view whose aggregation moved out of a page,
  so the page side of every extracted function (and its cached loader wrapper) runs at least once;
- `test_cli_composition.py` — editing `cli/` mints no dataset version, and `run("all")` bootstraps once;
- `test_import_boundaries.py` — the core imports no application (empty allowlist), only `cli/` imports
  `apps/`, nothing imports `cli`, and the apps do not import each other (except the dashboard using `apps.api`).

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

No mocking framework: `unittest.mock`, `MagicMock`, `patch(...)` and `pytest-mock` appear nowhere under
`tests/`. A test that must replace a collaborator passes a plain fake (a small class or a lambda that records
what it was given) through a documented injection point, or `monkeypatch`es that one point and nothing else.

Pure functions and in-memory SQLite sessions cover most of what mocking would be used for. The injection points
in use:

- `session_factory` in `ingest/openalex.py` (the HTTP getter of the batch and citing-works fetchers; defaults to
  `requests.get`). The single-DOI `fetch_openalex_observation` takes none, so its tests `monkeypatch`
  `openalex.requests.get` and `openalex.time.sleep` (`test_openalex_enrichment.py`, `test_openalex_throttling.py`);
- a dashboard module's own `st` binding -- `components.st` replaced by a fake namespace (`test_charts.py`), or
  single attributes of `loaders.st` / `quality.st` (`test_population_provenance.py`,
  `test_retrieval_benchmark_panel.py`) -- plus that module's data reads (`loaders.active_version_published_at`,
  `quality.loaders.review_labels`, `quality.loaders.retriever`);
- `apps.dashboard.data.get_engine`, bound to a layer fixture's `session.get_bind()`
  (`test_dashboard_publication_pointer.py`, `test_duplicate_resolution.py`);
- `embed_fn` and `store` on the transform builders (`transform/versioned_gold.py`), with the `vector_store`
  fixture as the store;
- `runs.get_session`, which `runs._layer_sessions` calls (`tests/test_pipeline_sessions.py`), and
  `runs._pipeline_lock` (`tests/test_retrieval_cli.py`) -- the bookkeeping moved out of `pipeline.py`, so
  patching `pipeline.get_session` no longer reaches it; a stage's own collaborators (`bootstrap`,
  `build_dataset_*`, `get_vector_store`) are still patched on `pipeline`;
- a command group's module, `cli.<group>` (`cli.versions`, `cli.duplicates`, `cli.evidence`, ...), for its
  `run` dispatcher and the names it imported (`get_session`, `bootstrap`); `cli.main.run` for the stage
  dispatch;
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
