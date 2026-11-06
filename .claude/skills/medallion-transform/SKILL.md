---
name: medallion-transform
description: Conventions for adding/editing pipeline code under ingest/, transform/, db/, and pipeline.py — the raw→bronze→silver→gold→embed medallion stages. Use whenever adding an ingest loader, a transform builder, a new model column, or wiring a stage into the CLI/Airflow.
---

# lake-research-map medallion pipeline

This is the authoring-side skill for the pipeline itself (`src/lake_research_map/{ingest,transform,db}/`
and `pipeline.py`) — how to actually *run* it, locally or via Airflow, is the `pipeline-ops` skill instead.
Dashboard code (`dashboard/`) has its own `streamlit-dashboard` skill.

## One database per layer, same table names

Four independent MySQL databases — `raw`, `bronze`, `silver`, `gold`, named plainly after the layer (no
prefix) — each with its own SQLAlchemy `Base` in `db/{raw,bronze,silver,gold}_models.py`. Table names repeat
across layers (`lit_articles`, etc.) but each layer's `Article` model has a different, layer-appropriate
field set — don't assume a column on one layer's model exists on another's.

**These databases are shared with unrelated projects on the same MySQL server** (`raw`/`bronze`/`silver`
already had other tables before this pipeline existed, e.g. `fastf1_results`, `personal_expenses`). Every
table this project owns is `lit_`-prefixed; never touch a table in these databases that isn't.

`db/engines.py`'s `get_engine(layer)` (`@cache`d) / `get_session(layer)` is the one access point — always go
through it, never construct an engine/session directly.

## Vectors live in Qdrant (ADR-08)

`db/vector_store.py`'s `get_vector_store()` is the equivalent access point for the fifth store. Builders take a
`store` argument the way they take sessions (tests pass the in-process `vector_store` fixture). One collection
per dataset version, point ids from `point_id(doi, chunk_type, seq)`; what a point carries is decided in
`transform/vector_payloads.py`, which also owns the shared derivations (`primary_source`,
`abstract_is_missing`, `relevance_margin`). Never write a vector to MySQL, and never write to a published
version's collection except through `fill_vector_collection(write_provenance=False)` (the `vectors` repairs).

Two traps the real data showed and small fixtures hide: the embed stage commits **and expunges** after every
batch, so anything loaded before its loop must be plain result rows, not ORM objects; and
`lit_dataset_chunks.text_sha256` has no index on the deployed schema (additive migrations add none), so query
through `dataset_version_id` first.

## Adding a new column

`db/bootstrap.py`'s `create_tables()` runs `Base.metadata.create_all()` per layer, which only creates
*missing tables*, not new columns on an existing table. For a column on an already-deployed layer, add an
explicit `ALTER TABLE ... ADD COLUMN` block there too (see the existing `reference_count` and `gold.sources`
examples) — otherwise the column only appears for people who drop and recreate their database.

Note the asymmetry: raw/bronze data persists across runs, but **silver and gold are fully truncated and
rebuilt every run** (`silver_session.query(SilverArticle).delete()` in `build_silver_articles`, similarly in
gold) — so a new silver/gold column just needs the builder function updated; only raw/bronze changes need to
worry about migrating already-populated rows.

## Stage-function boilerplate (`pipeline.py`)

Every `run_*()` function follows the same shape:

```python
def run_X() -> dict:
    a_session = get_session("layer_a")
    b_session = get_session("layer_b")
    try:
        stats = build_x(a_session, b_session)
        print(f"[x] {stats}")
        return stats
    finally:
        a_session.close()
        b_session.close()
```

Keep new/changed stages in this shape: always close every session in `finally`, always return and print a
stats dict. Wire a new stage into `STAGES` and `run_all()`/`run()` in `pipeline.py`, and add a matching
`BashOperator` DAG in `airflow/dags/lake_research_map_dags.py` (thin wrapper shelling out to
`uv run lake-research-map --stage X` — DAGs deliberately don't import `lake_research_map` directly).

## Ingest loader idempotency (`ingest/raw_*.py`)

Every raw-layer loader follows the same pattern via `ingest/hashing.py`: hash the source file
(`sha256_file`), check it against `raw.lit_source_files` via `record_source_file()`, and skip re-ingesting an
unchanged file. Follow this pattern for any new `ingest/raw_*.py` loader rather than inventing a new
change-detection scheme.

## The two `normalize_doi` implementations

`transform/bronze_articles.normalize_doi` and `ingest/enrichment._normalize_doi` are **deliberately
duplicated** (comments in both explain this avoids a circular import). If DOI normalization logic changes
(e.g. handling a new URL prefix format), update both — a divergence here silently breaks either bronze
building or citation-count enrichment lookups.

## Gold full text

`gold_articles._extract_pdf_text` runs every PDF through `transform/pdf_text.clean_pdf_text` before chunking
(reference list cut, page furniture dropped, line-break hyphens rejoined, lines reflowed); see CLAUDE.md for the
measured effect. Any change to a cleaner changes the text of every full-text chunk, which mints a new version and
re-embeds the full text once -- the abstracts keep their vectors. Keep each cleaner pure and add its effect to
`PDF_CLEANING_STATS` so the run stats say what it did.

## Silver PDF-linking and dedup

`transform/silver_articles.py` is the reference implementation for "how a transform stage should look":
dedup bronze rows by normalized DOI (`_merge_group`, preferring the record with the longest abstract as a
content-richness proxy), then fuzzy-match PDF filenames to titles via `rapidfuzz` (`normalize_title` +
`PDF_MATCH_THRESHOLD = 85.0`) since filenames are a lossy transform of the title. Follow this file's
structure (small private helpers + one public `build_X(...)  -> dict` entry point) for new transform stages.
