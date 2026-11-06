---
name: pipeline-ops
description: Running and operating the lake-research-map pipeline — local CLI vs. Docker/Airflow, .env/bootstrap setup, stage idempotency, where run stats go. Use whenever running a pipeline stage, debugging a failed run, or deciding between local and Airflow execution.
---

# Running the lake-research-map pipeline

This is the "how do I actually execute this" skill — for how to *write* ingest/transform code, see
`medallion-transform` instead.

## Local vs. Docker/Airflow

```bash
uv run lake-research-map --stage {raw,bronze,silver,gold,embed,all}   # local, one-shot
docker compose up -d                                                 # Airflow + dashboard, port 8080/8501
docker compose --profile qdrant up -d qdrant                         # vector store, port 6333 (loopback)
```

**Qdrant must be running for `embed`, `semantic`, publication (`versions activate` too) and dense search** --
the vectors live there, one collection per dataset version (ADR-08). Without it those stages fail fast with
the compose command in the message; set `QDRANT_URL` in `.env` (`http://127.0.0.1:6333` on the host; the
compose services override it to `http://qdrant:6333`). `uv run lake-research-map vectors status` compares
every version's collection with its Gold chunks and says whether the `lit_chunks_active` alias is in sync;
`vectors backfill` copies vectors still held as MySQL BLOBs, `vectors rebuild [--version ID]` restores a lost
collection from Gold text, `vectors sync-alias` re-points the alias. Repairs take the pipeline lock.

Use the local CLI for iterating on a single stage during development (fast feedback, no orchestration
overhead). Use `docker compose up -d` when you need DAG run history, per-task logs, or to exercise the
dashboard's "Layers & Pipeline"/"Quality & RAG" pages, which trigger stages through Airflow's REST API
(`dashboard/airflow_client.py`) rather than running the pipeline in-process — those buttons don't work
without Airflow running.

Both paths run the exact same code: `airflow/dags/lake_research_map_dags.py`'s DAGs are thin `BashOperator`
wrappers around `uv run lake-research-map --stage X`, so there's no separate "Airflow version" of the pipeline
logic to keep in sync.

## Environment setup

`.env` (git-ignored, copy from `.env.example`) needs `MYSQL_HOST/PORT/USER/PASSWORD`, `QDRANT_URL` and
`AIRFLOW_BASE_URL`. Point `FASTEMBED_CACHE_PATH` at persistent storage if `/tmp` is tmpfs (never leave it set
but empty: fastembed then caches into the current directory).
`bootstrap()` (called automatically at the start of every `pipeline.run()`) issues `CREATE DATABASE IF NOT
EXISTS` for all four `raw`/`bronze`/`silver`/`gold` databases plus `create_all()` — no manual DB setup needed
beyond a reachable MySQL server and correct `.env` credentials. `MYSQL_HOST` may point at a shared server
where these database names already host unrelated tables from other projects — `bootstrap()`/the pipeline
only ever create or touch `lit_`-prefixed tables within them, never anything else.

## Stage idempotency, at a glance

| Stage | Re-run behavior |
|---|---|
| raw | Skips unchanged source files via sha256 hash + `raw.lit_source_files` manifest |
| bronze | Rebuilt from raw each run |
| silver | Fully truncated + rebuilt from bronze each run |
| gold | Fully truncated + rebuilt from silver each run |
| embed | Fills only chunks whose Qdrant point is missing or stale, copying any vector already stored for the same text before embedding — safe to re-run, resumes cleanly if interrupted (each batch is upserted, then Gold commits its provenance) |

`embed` is the one stage safe to run repeatedly with no wasted work; the others (except raw) simply
regenerate everything downstream of the layer they read from every time.

## Where run stats actually go

There is **no persisted pipeline run-history table**. Every `run_*()` in `pipeline.py` prints its stats dict
(`print(f"[stage] {stats}")`) and returns it — visible in stdout locally, or in the relevant task's log in
the Airflow UI when run via a DAG. The dashboard's "Layers & Pipeline" page computes its funnel/counts live
from current row counts in each database, not from stored history — a discrepancy there means "what's in the
DB right now," not "what happened on the last run."
