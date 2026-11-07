---
name: pipeline-ops
description: Running and operating the lake-research-map pipeline — the CLI, the Qdrant service, .env/bootstrap setup, stage idempotency, where run stats go. Use whenever running a pipeline stage, debugging a failed run, or repairing the vector store.
---

# Running the lake-research-map pipeline

This is the "how do I actually execute this" skill — for how to *write* ingest/transform code, see
`medallion-transform` instead.

## The CLI is the only way to run it

```bash
uv run lake-research-map --stage all                                 # the chained full run
uv run lake-research-map --stage {raw,bronze,silver,gold,embed,semantic}   # one stage
docker compose --profile qdrant up -d qdrant                         # vector store, port 6333 (loopback)
docker compose up -d                                                 # the read-only dashboard, port ${DASHBOARD_PORT:-8501}
```

The console script is `cli.main:main`; `uv run python -m cli.main ...` is the same CLI without the installed
script.

There is no orchestrator: Airflow (DAGs, container, the dashboard's trigger buttons) was removed on
2026-11-06 (ADR-09), and no page can start a run. Nothing retries a failed stage automatically -- re-run it:
every stage is idempotent, `embed` resumes from its own Qdrant collection, and the next run's start-up sweep
marks the abandoned execution as failed. To run a workflow as separate stage processes (cron, CI), pass the
same `--execution-id` and `--workflow` to each `--stage` call; they are recorded as one execution.

**Qdrant must be running for `embed`, `semantic`, publication (`versions activate` too) and dense search** --
the vectors live there, one collection per dataset version (ADR-08). Without it those stages fail fast with
the compose command in the message; set `QDRANT_URL` in `.env` (`http://127.0.0.1:6333` on the host; the
dashboard container overrides it to `http://qdrant:6333`). `uv run lake-research-map vectors status` compares
every version's collection with its Gold chunks and says whether the `lit_chunks_active` alias is in sync;
`vectors backfill` copies vectors still held as MySQL BLOBs, `vectors rebuild [--version ID]` restores a lost
collection from Gold text, `vectors sync-alias` re-points the alias. Repairs take the pipeline lock, as do stage
runs, `duplicates merge|keep|undo`, `versions activate`, `enrichment refresh-openalex` and
`maintenance recover-stale` (`cli.main._mutates_published_inputs`; full table in SDD §4.3); `search`, `screen`,
`mcp`, `reviews`, `evidence`, `audit` and the other `enrichment` crawls never do.

Embedding is CPU-bound and bounded on purpose (`LAKE_RESEARCH_MAP_EMBED_THREADS=2` in `.env`): re-embedding the
~3,600 full-text chunks after a change to the PDF cleaning takes the better part of an hour at that setting,
while a run that changes no text copies every vector in about a minute.

## Environment setup

`.env` (git-ignored, copy from `.env.example`) needs `MYSQL_HOST/PORT/USER/PASSWORD` and `QDRANT_URL`. Point `FASTEMBED_CACHE_PATH` at persistent storage if `/tmp` is tmpfs (never leave it set
but empty: fastembed then caches into the current directory).
`bootstrap()` (called automatically at the start of every `pipeline.run()`) issues `CREATE DATABASE IF NOT
EXISTS` for all four `raw`/`bronze`/`silver`/`gold` databases plus `create_all()` — no manual DB setup needed
beyond a reachable MySQL server and correct `.env` credentials. `MYSQL_HOST` may point at a shared server
where these database names already host unrelated tables from other projects — `bootstrap()`/the pipeline
only ever create or touch `lit_`-prefixed tables within them, never anything else.

## Stage idempotency, at a glance

| Stage | Re-run behavior |
|---|---|
| raw | Skips unchanged source files via sha256 hash + `raw.lit_source_files` manifest; `--source-policy snapshot` propagates removals |
| bronze | Deletes and rebuilds `bronze.lit_articles` from raw each run, then backfills counts |
| silver | Fully truncated + rebuilt from bronze each run |
| gold | Deletes and rebuilds the **candidate version's** rows (`lit_dataset_articles`/`lit_dataset_chunks`); a published version is never rewritten, and the live tables change only at publication |
| embed | Fills only chunks whose Qdrant point is missing or stale, copying any vector already stored for the same text before embedding — safe to re-run, resumes cleanly if interrupted (each batch is upserted, then Gold commits its provenance) |
| semantic | Rewrites the candidate's `lit_dataset_semantics`/`lit_semantic_runs`/`lit_dataset_duplicate_pairs` and its point payloads, then publishes the version (contract-checked) and moves the `lit_chunks_active` alias |

`embed` is the one stage safe to run repeatedly with no wasted work; the others (except raw) regenerate what they
own from the layer they read. A `--stage all` run whose version is already active skips bronze through semantic
(`no_change`).

## Where run stats actually go

Every `run_*()` in `pipeline.py` prints its stats dict (`print(f"[stage] {stats}")`) and records it: one
`gold.lit_pipeline_executions` row per run, one `gold.lit_pipeline_runs` row per stage with its stats JSON, and
the stage's contract results in `gold.lit_quality_results`. The Pipeline page's audit tab reads those tables.
Its funnel, by contrast, is computed live from current row counts in each database -- a discrepancy there means
"what's in the DB right now", not "what happened on the last run". Run with stdout redirected to a file and the
stage lines appear only at exit (block buffering); read progress from `lit_pipeline_runs` instead.

Log lines go to stderr and to `data/.lake_research_map/pipeline.log` (`LAKE_RESEARCH_MAP_LOG` overrides the
path). The run bookkeeping logs as `runs` and the CLI as `cli.main`, `cli.enrichment` and `cli.evidence`; there
is no `pipeline` logger any more, so filter on those names.
