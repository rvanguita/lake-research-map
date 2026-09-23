# Foundation acceptance evidence — 2026-09-21

## Environment and commands

- MySQL server: 8.4.11.
- Schema migration: `uv run python -m lake_research_map.db.bootstrap`.
- Corpus audit: `uv run lake-research-map audit reference-corpus`.
- Local suite: `uv run pytest -q`.
- MySQL acceptance: `RUN_MYSQL_INTEGRATION=1 uv run pytest -q -m mysql tests/test_mysql_contracts.py`.

## Measured result

- Active version: `31a2270bed86b9fd5058d4de037edb06ed8017e74251ba9edb054cadfaa31f2c`.
- Active Gold articles: 3,115.
- Active Gold chunks: 7,552.
- Active semantic rows: 3,115.
- Persisted blocking quality failures: 0.
- Additive bootstrap completed for all four medallion databases; only `lit_*` objects were inspected or changed.
- The repeatable MySQL acceptance test passed in 0.81 s. Its disposable `gold.lit_mysql_contract_probe` verified JSON, NULL and LONGBLOB round trips, transaction rollback, and `GET_LOCK`/`RELEASE_LOCK`; the probe table was dropped afterward.
- The final local verification passed 236 tests with one opt-in MySQL test skipped in 6.08 s. Correctness passed, but the documented under-five-second target did not; this remains an explicit WP-22 performance gate.

## Residual gates

- The active snapshot predates the new embedding metadata columns. A candidate must be backfilled/re-embedded and pass the stricter contract before it can replace the active version.
- OpenAlex refresh requires `OPENALEX_API_KEY` and `OPENALEX_EMAIL`; no external observations were fabricated.
- Screening, author identity, taxonomy, PDF audit, and retrieval validation require independent human judgments.
- Runtime MySQL users require locally supplied passwords before `python -m lake_research_map.db.roles` can be executed.
- Incoming citation edges and access-confounder validation remain required before CD/OACA results can be exposed.
