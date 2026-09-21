# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

`lake-research-map` — a systematic-literature-review pipeline over bibliographic exports on the topic
*"distribution system planning"* (electric power distribution networks). The corpus is assembled by hand from
publisher search UIs (IEEE Xplore + Elsevier/ScienceDirect), then consolidated by a medallion pipeline
(`src/lake_research_map/`) into MySQL and explored through a Streamlit dashboard (`src/lake_research_map/dashboard/`).

`README.md` is the project overview; `AGENTS.md` provides universal guidelines for all AI agents;
`docs/PRD.md` (why) and `docs/SDD.md` (how) go deeper. This file is the
canonical reference for **source-data quirks** — the other docs cross-reference it rather than repeat it.

## Commands

Managed by [uv](https://docs.astral.sh/uv/) (Python 3.13, `uv_build` backend, src layout).

```bash
uv sync                                          # create/refresh .venv from uv.lock
uv run lake-research-map --stage all             # full pipeline (default stage)
uv run lake-research-map --stage <stage>         # raw | bronze | silver | gold | embed | semantic
uv run python -m lake_research_map.db.bootstrap  # create the 4 databases + tables only, no ingestion
uv run streamlit run main.py                     # dashboard at http://localhost:8501
docker compose up -d                             # Airflow (:8080) + dashboard (:8501), both read .env
uv run pytest                                    # full suite (in-memory SQLite, no MySQL needed)
uv run pytest tests/test_silver_articles.py -k merge   # one file / one test
uv run ruff check --fix && uv run ruff format    # lint + format (config in pyproject.toml)
uv add <pkg>                                     # add a dependency (updates pyproject.toml + uv.lock)
```

MySQL connection settings and `AIRFLOW_BASE_URL` live in `.env` (git-ignored; see `.env.example`).

## Skills

`.claude/skills/` holds the detailed conventions — load the matching one before writing code:

| Skill | Covers |
|---|---|
| `medallion-transform` | authoring `ingest/`, `transform/`, `db/`, `pipeline.py` |
| `pipeline-ops` | *running* stages: local CLI vs. Docker/Airflow, `.env`, idempotency, debugging a run |
| `streamlit-dashboard` | `dashboard/` page contract, chart contract, theme tokens, aggregation rules |
| `python-testing-conventions` | pytest layout, per-layer SQLite fixtures, pure-function-first testing |

## Architecture

### Stage graph

`raw → bronze → silver → gold → embed → semantic`, each a `run_<stage>()` in `pipeline.py` and each a
1:1 Airflow DAG (`airflow/dags/lake_research_map_dags.py`, thin `BashOperator` wrappers around the same CLI —
the DAG file deliberately never imports `lake_research_map`).

- `raw` — verbatim ingestion, one table per source artifact (`ingest/raw_{csv,bib,pdfs,config}.py`).
- `bronze` — IEEE CSV + IEEE `.bib` + Elsevier `.bib` unioned into one typed schema. The IEEE CSV is the
  authoritative record list; `.bib` entries only enter as their own record when their DOI is absent from the CSV.
- `silver` — dedup by normalized DOI, quality flags, fuzzy PDF linking (rapidfuzz, threshold 85).
- `gold` — curated `lit_articles` + `lit_chunks` (RAG unit: one `abstract` chunk per article, plus `fulltext`
  chunks from a linked PDF via pypdf).
- `embed` — fills `lit_chunks.embedding` locally via `fastembed` (`BAAI/bge-small-en-v1.5`, ONNX, no API key).
- `semantic` — reads those embeddings, writes `lit_semantics` + `lit_duplicate_pairs`.

**`gold` rebuilds the articles table but *reconciles* the chunks: a chunk whose text is unchanged keeps its
row and its vector, and only the chunks whose text actually changed are invalidated. (It was a plain
delete-and-rebuild once, which silently threw away every embedding on each run — and with them the basis of
`lit_semantics`.) So after a `gold` run, `embed` fills exactly the chunks that changed, and `semantic` has to
be re-run either way, because it rewrites its own tables from whatever is embedded now.**

### Idempotency, per stage

Each stage has its own re-run contract; preserve it when editing.

- `raw` — content-hash manifest (`ingest/hashing.py` ↔ `lit_source_files`); unchanged files are skipped.
- `bronze` — upsert keyed on `(source, source_id)`; pagination duplicates within a source collapse naturally.
  Note `_upsert` writes field by field, so a `None` literal *overwrites* — that's why `ingest/enrichment.py`
  re-applies the citation/reference-count backfill after every bronze build.
- `silver` — delete-and-rebuild: fully derived from the layer above.
- `gold` — articles delete-and-rebuild; chunks reconcile against the desired set (`(chunk_type, seq)` per
  DOI), so unchanged text keeps its embedding. The run stats say how many were unchanged / invalidated /
  added.
- `embed` — only processes `embedding IS NULL`; running it twice is a no-op.
- `semantic` — truncates the two tables it owns, never touches curated article rows.

### Databases

One MySQL database per layer, named plainly after the layer (`raw`, `bronze`, `silver`, `gold`), each with its
own SQLAlchemy `Base` in `db/<layer>_models.py`. Table names repeat across layers (`lit_articles`) but the
column sets differ — never assume a column on one layer's model exists on another's.

**These databases are shared with unrelated projects on the same MySQL server.** Only ever create or touch
`lit_`-prefixed tables.

`db/bootstrap.py` runs on every pipeline start: `create_all` (new tables only) plus the additive migrations in
`_ADDITIVE_COLUMNS`. Adding a column to an existing table requires an entry there, and every entry must be
nullable and purely additive — it runs unattended.

### Session lifecycle

`pipeline.py` opens one `Session` per layer a stage touches and closes them in a `finally`; transform builders
take those sessions as arguments and never open their own. That's what lets the whole suite run against
in-memory SQLite (`tests/conftest.py`, one session fixture per layer) with no MySQL.

`config.relative_path()` / `absolute_path()`: stored file paths are relative to the repo root because the same
file is `/home/…/data/x.csv` on the host and `/opt/airflow/project/data/x.csv` in the Airflow container —
raw-layer idempotency keys off that string, so absolute paths duplicate every row on a cross-environment run.

### Dashboard

Strict one-way layering — `data.py` (raw SQL → DataFrame, tolerates a missing table) → `loaders.py` (the only
`@st.cache_data` layer, plus list-column/`source` normalization) → `analytics.py` / `forecasting.py` /
`search.py` / `qualis.py` (pure pandas, no `streamlit` import, unit-tested) → `charts.py` (styled Plotly
figures) → `components.py` → `pages/*.py` (one zero-arg `render()`, registered in `app.py`'s `PAGES`).

Pages are read-only, with no exception: pipeline execution goes through Airflow's REST API
(`pipeline_control.py` → `airflow_client.py`). Embedding generation used to be an in-process exception in
`actions.py`; it was removed on 2026-09-21 because it wrote the live `lit_chunks` table that publication
rebuilds from the versioned candidate, so the vectors were silently discarded on the next publish.

The dashboard reads MySQL directly and prefers the **silver** layer (`pick_best_articles_layer`), while
`lit_semantics` lives in gold — hence `loaders.with_semantics()` joins it on `doi` rather than treating it as
a column of the active layer.

`main.py` calls `app.main()` **as a function** on purpose: Streamlit re-executes the entry script on every
rerun, and a module imported for its top-level side effects would only render once.

### Why the `semantic` stage exists

"Distribution system planning" is ambiguous — it also matches logistics/supply-chain papers, and roughly a
tenth of the corpus is facility-location/cold-chain work plus book front matter ingested as articles.
Relevance screening is a core SLR step, so every article is scored against **two** anchors in
`transform/semantics.py` — `ANCHOR_TEXT` (the review's topic) and `OFF_ANCHOR_TEXT` (the logistics reading of
the same query) — and the screening signal is the margin between them, whose **zero is the threshold**:
"closer to logistics than to the review's topic". A single anchor separates the two groups well (ROC AUC 0.96)
but their score distributions overlap, so the percentile cut this used to take also discarded in-scope work;
the margin reaches AUC 0.99 with no overlap. Nothing is auto-deleted, and articles with no score are never
filtered out by `loaders.filter_articles`.

The themes on the same page come from KMeans and the map from t-SNE, both over the **same**
PCA(50) space (`transform/semantics.py::reduced_space`) — sharing it is what keeps a point's color and its
position on the map in agreement. `k` is no longer a fixed constant: `discover_themes` sweeps
`MIN_THEMES`..`MAX_THEMES` (4..12), rejects any solution with a cluster under 2% of the corpus, and breaks
near-ties (within 0.005 silhouette) toward the smaller `k` — because silhouette is nearly flat across that
range on this corpus, so the tie-break, not the maximum, is what actually decides. HDBSCAN finds only two
groups (one continuum plus the logistics island). The Screening page reports bootstrap ARI and projection
trustworthiness beside the map, since a flat silhouette means the boundaries deserve a stability caveat.

## Data corpus (`data/`, gitignored)

`data/` is excluded from git, so it exists only on this machine and paths referenced in code will not resolve for
anyone else. Treat it as read-only input: it is raw publisher output, re-downloading it is manual and tedious.
(A `PreToolUse` hook in `.claude/settings.json` blocks edits to it, except `config.csv`.)

```
data/ieee/       IEEE Xplore export: one metadata CSV + paginated .bib files + bulk-download*.zip of PDFs
data/elsevier/   ScienceDirect export: paginated .bib files only (no CSV, no PDFs)
data/articles/   ~96 PDFs, extracted from the IEEE bulk-download zips
data/sciencedirect.zip            original archive that data/elsevier/ was unpacked from
data/enrichment_cache.json        hand-built {doi: {citation_count, reference_count}}, not produced by any code here
data/classificações_publicadas_*.xlsx   official CAPES/Qualis export (see dashboard/qualis.py)
```

**`config.csv` is provenance, not data.** Each source directory has one, and it holds the free-text record of the
search that produced that export — query string, filters, year range, and the full search URL. It is not a
parseable table; never feed it to `pd.read_csv` expecting columns. When the corpus is refreshed, update the
matching `config.csv` so the search is reproducible.

### The two sources are not interchangeable

Anything that merges IEEE and Elsevier records has to normalize these differences:

| | IEEE Xplore | ScienceDirect / Elsevier |
|---|---|---|
| Metadata | `export*.csv` (28 columns: `Document Title`, `Authors`, `Abstract`, `DOI`, `Author Keywords`, `IEEE Terms`, …) plus `.bib` | `.bib` only |
| Entry type | `@ARTICLE` | `@article` |
| Page size | 25 entries per `.bib` | 100 entries per `.bib` (URL `offset=` drives pagination) |
| `doi` field | bare DOI — `10.1109/TPWRS.2024.3418651` | full URL — `https://doi.org/10.1016/j.ijepes.2020.106042` |
| `keywords` | `;`-separated | `,`-separated |
| Venue field | `journal` | `journal`, plus `url` and sometimes `note` |
| Affiliations, online date, document type, license | yes (CSV only) | none |
| Citation / reference counts | yes (CSV only) | none — backfilled from `enrichment_cache.json` |
| Full text | yes, PDFs in the zips | none |

DOI is the only reliable cross-source join/dedup key — strip the `https://doi.org/` prefix and casefold before
comparing, or the same paper indexed by both publishers will survive deduplication twice. Records with no DOI
at all are dropped at silver (counted as `skipped_no_doi`, never silently).

The IEEE-only fields (`countries`, `online_date`, `document_type`, `license`) carry through bronze → silver →
gold but cover only the IEEE subset — 302 of 1,831 articles, 16.5%. **Any analysis built on them must say it covers the IEEE subset**, not
the whole corpus.

### BibTeX parsing gotcha

The IEEE `.bib` files are written with **no separator between entries** — one entry's closing brace is immediately
followed by the next `@ARTICLE{`, on the same line:

```
  month={Feb},}@ARTICLE{10854892,
```

Line-oriented or naive split-on-`@` parsing will silently merge or truncate records. Use a real BibTeX parser
(e.g. `bibtexparser`) or split on `}@` deliberately. Elsevier's files do put each entry on its own lines, so code
tested only against `data/elsevier/` will appear to work and then fail on IEEE.

### PDF filenames

PDFs in `data/articles/` are named after the article title with punctuation replaced by `-`, e.g.
`Use of Computer Graphics ... in -Electricite de France- -E.D.F.-.pdf` for a title containing quotes and accents.
The mapping back to a title is lossy, so match PDFs to metadata via a normalized/fuzzy title comparison rather
than exact string equality — and prefer DOI-keyed renaming if a linking step is ever added.

### Counts don't line up

Measured on the current corpus (2026-09-17): 304 IEEE CSV rows and 1,815 BibTeX entries across both sources
ingest to 1,836 bronze records; 1,831 survive silver's DOI deduplication (5 dropped as `skipped_no_doi`); 96
have a PDF. The gap between search hits, downloaded entries and retrieved PDFs is a property of how the
corpus was hand-assembled — do not treat a count mismatch as a bug to fix in code. Re-measure before quoting
these numbers; the corpus grows whenever a new export is added.

The split is also lopsided: 1,529 articles come from Elsevier and 302 from IEEE, and **no article is
currently indexed by both** — DOI deduplication is insurance the design needs, not the dominant problem in
this corpus. The 18 near-identical abstracts under distinct DOIs in `lit_duplicate_pairs` are.

## Conventions

- **Language**: everything is in English -- code, comments, docstrings, `docs/`, and every string the
  dashboard renders (page titles, captions, warnings, metric labels, download filenames). The UI was
  Portuguese until the 2026-09-21 migration; `PRD.md` NFR-07 now requires English. Watch for two
  leftovers when editing: values produced in `analytics.py`/`forecasting.py` that are rendered verbatim
  by a page, and multi-line implicit string concatenation, which silently drops the space between
  fragments unless the first one ends with it.
- **Comments explain *why***, and the existing ones encode hard-won decisions (why a chunk boundary is where it
  is, why a palette isn't derived from the brand colors, why `fastembed` is imported lazily). Don't strip them.
- `.claude/settings.json` hooks run on every edit: `ruff check --fix` + `ruff format` on any `.py`, and
  `uv run pytest -q` after touching `ingest/`, `transform/` or `tests/`. Recursive deletion of `data/`,
  force-pushes and hard resets are blocked at the Bash tool.
- `pre-commit` (gitleaks + `scripts/git-hooks/check-docs-branch.sh`) blocks docs-only commits made directly on
  `main` — put documentation changes on a `docs/<topic>` branch. `main` is protected on GitHub.
