---
name: medallion-transform
description: Conventions for adding/editing pipeline code under ingest/, transform/, db/, and pipeline.py — the raw→bronze→silver→gold→embed→semantic medallion stages. Use whenever adding an ingest loader, a transform builder, a new model column, or wiring a stage into the CLI.
---

# lake-research-map medallion pipeline

This is the authoring-side skill for the pipeline itself (`src/{ingest,transform,db}/`
and `pipeline.py`) — how to actually *run* it is the `pipeline-ops` skill instead.
Dashboard code (`apps/dashboard/`) has its own `streamlit-dashboard` skill.

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
*missing tables*, not new columns on an existing table. For a column on an already-deployed table, add an entry
to `_ADDITIVE_COLUMNS` in `db/bootstrap.py` (keyed by table, then column → `(DDL, layers)`): the DDL must be
`NULL`able and purely additive, and it must not add an index -- it runs unattended on every pipeline start, and
an index on a large table (`lit_dataset_chunks`) would rebuild it there. Never hand-write an `ALTER TABLE` block.
The one index migration that exists, `_migrate_config_uniqueness` (`raw.lit_config`), is a documented exception,
not a precedent.

Every stage except raw rebuilds what it owns, so a new column needs its builder updated, not a backfill of old
rows: bronze deletes `lit_articles` and rebuilds it from raw (`build_bronze_articles`), silver does the same from
bronze, and gold deletes and rebuilds the *candidate version's* rows (`versioned_gold.build_dataset_gold`) --
published versions are immutable snapshots and are never rewritten. Raw is the layer whose rows persist
(content-hash idempotency), so a raw column is the one case where already-ingested rows stay without it until
their source file changes.

## Stage functions (`pipeline.py`)

Do not write a stage from a short template: copy `run_silver` in `pipeline.py`, which is the reference. Every
`run_<stage>()` does, in order:

1. open its sessions with `_layer_sessions(...)` (an `ExitStack`, so a failure anywhere still closes them --
   and the module-level `get_session` it calls is the tests' injection point);
2. resolve the working version (`_working_version`) and record the execution (`_ensure_execution`, commit);
3. `_begin_stage` (a `lit_pipeline_runs` row in `running`);
4. the skip check: `_is_unchanged_full_run` → `_complete_skipped_stage` (an unchanged `--stage all` run of an
   active version does nothing);
5. the builder (`build_*`, sessions and `store` passed in, never opened inside);
6. the stage's `quality.*_contract`, persisted with `_record_results` and committed **before**
   `assert_contract` raises, so a failed gate is still on record;
7. commit the layer session, `_finish_stage(..., status="success")`, `_finalize_execution` when the workflow is
   this stage alone, commit, then `print(f"[stage] {stats}")` and return the stats dict;
8. on any exception: roll back the layer session and call `_handle_stage_failure` (rollback, error status on the
   run and the execution, failed version), then re-raise.

`run_semantic` adds the publication (`_version_ready`, `materialize_version`, then `publish_vector_alias` after
the commit). Wire a new stage into `STAGES`, `_STAGE_SEQUENCE` (derived), `run_all()` and `run()`; the CLI is the
only entry point (there is no orchestrator since ADR-09). A new CLI command that changes what the next
publication contains must also be listed in `_mutates_published_inputs` so it takes the pipeline lock.

## Ingest loader idempotency (`ingest/raw_*.py`)

Every raw-layer loader (`raw_{csv,bib,pdfs,config}.py`) follows the same pattern via `ingest/hashing.py` (on top of the content-addressed snapshot `run_raw` takes through `ingest/snapshots.py`): hash the source file
(`sha256_file`), check it against `raw.lit_source_files` via `record_source_file()`, and skip re-ingesting an
unchanged file. Follow this pattern for any new `ingest/raw_*.py` loader rather than inventing a new
change-detection scheme.

## The two `normalize_doi` implementations

`transform/bronze_articles.normalize_doi` and `ingest/enrichment._normalize_doi` are **deliberately
duplicated** (a comment in `enrichment.py` explains it avoids a circular import). If DOI normalization logic changes
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
