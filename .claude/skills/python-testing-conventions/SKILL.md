---
name: python-testing-conventions
description: pytest conventions for lake-research-map — pure-function-first testing, per-medallion-layer in-memory SQLite fixtures, no mocking. Use whenever writing or extending tests under tests/, for pipeline transform code or dashboard analytics code.
---

# lake-research-map testing conventions

Applies repo-wide — both `src/lake_research_map/{ingest,transform}/` pipeline code and
`src/lake_research_map/apps/dashboard/analytics.py`/`forecasting.py` and `apps/api/rank.py` (all pure, no `streamlit`
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
add a real MySQL dependency to tests.

```python
@pytest.fixture
def silver_session():
    session = _sqlite_session(silver_models.Base)
    yield session
    session.close()
```

## No mocking

There's no mocking framework in this repo (no `unittest.mock`, no `pytest-mock`) — don't introduce one. Pure
functions and in-memory SQLite sessions cover what mocking would otherwise be used for.

## `test_raw_bib.py`: a reminder about IEEE's BibTeX gotcha

This file exists specifically to catch the "IEEE `.bib` entries have no separator between them" parsing bug
(see `CLAUDE.md`). Any change to `ingest/raw_bib.py` should keep exercising hand-written `.bib` fixtures that
reproduce that glued-together-entries shape, not just well-formed Elsevier-style fixtures — code that only
passes against clean fixtures can still silently merge/truncate real IEEE records.
