# AGENTS.md — AI Assistant Guidelines for `lake-research-map`

This document defines the architecture, design principles, testing protocols, and development rules for AI assistants (such as Antigravity, Claude Code, Cursor, and Codex) working within the `lake-research-map` codebase.

---

## 1. Project Overview & Mission

`lake-research-map` is a production-grade **Medallion Data Lake** and analytical research platform built for a Systematic Literature Review (SLR) on the engineering topic:
> **"Distribution System Planning" (Electric Power Distribution Networks)**

The corpus comprises hand-curated bibliographic exports from **IEEE Xplore** and **Elsevier ScienceDirect** (~1,831 deduplicated articles). The project provides:
1. **Medallion Ingestion & Transform Pipeline** (`src/lake_research_map/`): Multi-tier extraction, normalization, deduplication, chunking, binary vector embedding, and contrastive relevance screening.
2. **Orchestration** (`airflow/`): 1:1 Airflow DAGs mirroring CLI pipeline stages (`raw`, `bronze`, `silver`, `gold`, `embed`, `semantic`, `all`).
3. **Interactive Analytical Dashboard** (`src/lake_research_map/dashboard/`): Multipage Streamlit application featuring 13 deduplicated pages for bibliometric, scientometric, econometric, network, and semantic intelligence.

---

## 2. Architecture & Medallion Data Lake Rules

### 2.1 Database Isolation & Multi-Project Tenancy
The pipeline connects to a MySQL server with 4 discrete databases named plainly after the medallion layers: `raw`, `bronze`, `silver`, and `gold`.
- **CRITICAL**: These MySQL databases are shared with other unrelated projects (e.g., `fastf1_results`, `personal_expenses`).
- **RULE**: Agents must **ONLY** create, query, or modify tables with the `lit_` prefix (e.g., `lit_articles`, `lit_chunks`, `lit_semantics`, `lit_pipeline_runs`, `lit_rejected`). Never inspect or alter non-`lit_` tables.

### 2.2 Medallion Pipeline Stages

```
[Publisher Exports]
  data/ieee/ (CSV + .bib + PDFs)
  data/elsevier/ (.bib)
       │
       ▼
   1. RAW (`raw`)          --> lit_source_files (hash manifest), lit_config, lit_bib_entries, lit_ieee_csv_rows, lit_pdf_files
       │
       ▼
   2. BRONZE (`bronze`)    --> lit_articles (unioned schema, OpenAlex citation/reference backfill)
       │
       ├─────────────────┐
       ▼                 ▼
   3. SILVER (`silver`)   lit_rejected (SLR audit log of dropped non-DOI rows)
       │                  lit_articles (DOI deduplication, non-article tags, fuzzy PDF matching)
       ▼
   4. GOLD (`gold`)        --> lit_articles (curated view)
       │                       lit_chunks (RAG units: abstracts & fulltext; reconciled against text hash)
       │                       lit_pipeline_runs (execution duration, stats JSON, status)
       ▼
   5. EMBED (`embed`)      --> fills lit_chunks.embedding_bin (LargeBinary float32) & embedding (JSON fallback)
       │                       via local ONNX fastembed (BAAI/bge-small-en-v1.5)
       ▼
   6. SEMANTIC (`semantic`)--> lit_semantics (contrastive topic vs logistics margin, KMeans themes, 2D projections)
                               lit_duplicate_pairs (near-duplicate abstracts under distinct DOIs)
```

### 2.3 Stage Idempotency Contracts
- **`raw`**: Uses sha256 hashing (`lit_source_files`). Re-running skips unchanged files.
- **`bronze`**: Natural key upsert `(source, source_id)`. Re-applies OpenAlex enrichment.
- **`silver`**: Delete-and-rebuild derived table. Logs dropped records to `lit_rejected`.
- **`gold`**: Rebuilds `lit_articles` but **reconciles** `lit_chunks`: existing chunks with matching text keep their vector (`embedding_bin`), avoiding redundant embedding recomputation.
- **`embed`**: Only processes chunks where `embedding_bin IS NULL`. Running twice is a zero-op.
- **`semantic`**: Truncates and rebuilds `lit_semantics` and `lit_duplicate_pairs` atomically without modifying articles.
- **`bootstrap`**: `src/lake_research_map/db/bootstrap.py` executes before every pipeline run. Any newly introduced database column must be registered in `_ADDITIVE_COLUMNS` as nullable and additive.

---

## 3. Data Quirks & Ingestion Normalization

When working on `ingest/`, `transform/`, or `loaders.py`, adhere strictly to known corpus quirks:

| Concern | Behavior / Requirement |
|---|---|
| **BibTeX Separators** | IEEE `.bib` files have **no newline or separator** between entries (`month={Feb},}@ARTICLE{10854892,`). Naive parsing breaks. Always use robust parsing (`bibtexparser`). |
| **DOI Normalization** | Elsevier provides full URLs (`https://doi.org/10.1016/...`), IEEE provides bare strings (`10.1109/...`). Always strip `https://doi.org/`, trim whitespace, and lowercase before matching or keying. |
| **IEEE-Only Metadata** | Fields like `countries`, `online_date`, `document_type`, and `license` are only provided by IEEE CSV exports (~16.5% of corpus). Any analysis utilizing them must disclose coverage. |
| **PDF Title Matching** | PDF filenames in `data/articles/` encode titles with dashes for punctuation. Match using `rapidfuzz.fuzz.token_set_ratio` with threshold $\ge 85$. |
| **Non-Article Filtering** | Conference book covers, prefaces, and front matter are tagged via `is_non_article=True` rather than deleted, preserving full corpus auditability. |
| **Search Provenance** | `config.csv` files are free-form search provenance (query strings, timestamps, search URLs), not parseable tabular CSVs. |

---

## 4. Dashboard Architecture & Streamlit Guidelines

The dashboard is structured into 10 workflow-oriented pages in `src/lake_research_map/dashboard/`:

### 4.1 Strict Separation of Concerns
1. **`data.py`**: Raw SQL queries returning pandas DataFrames. Must fail gracefully if tables do not exist.
2. **`loaders.py`**: The **ONLY** layer permitted to use `@st.cache_data`. Handles normalization, schema harmonization, and joins with `lit_semantics`.
3. **`analytics.py` & `forecasting.py`**: **Pure mathematical and statistical functions.** Must never import `streamlit`. Must be 100% testable using standard `pytest`.
4. **`charts.py`**: Pure Plotly figure generators applying theme tokens.
5. **`components.py`**: Reusable Streamlit UI widgets (metrics rows, callout banners, data downloaders).
6. **`pages/*.py`**: Presentation controllers. Each page exports a single zero-argument `render()` function registered in `app.py`.

### 4.2 Deduplication and Tab Hygiene
- **Zero Chart Duplication**: Charts must never be duplicated across tabs within a page or between specialized pages.
- **Role of Visão Geral**: `overview.py` displays high-level macro summaries only. Deep-dive analytical charts belong exclusively to their respective analytical pages.
- **Logical Tab Grouping**:
  - `production.py`: Strictly chronological views (Volume Anual, Crescimento Acumulado, Estratos CAPES/Qualis).
  - `topics.py`: Venue ranking, Bradford Zones, semantic structure, Zipf's Law, c-TF-IDF, descriptive conceptual atypicality, and structural breaks.
  - `highlights.py`: 2 tabs — *Fundamentação Teórica* and *Dinâmica de Citações & Econometria*, including heavy-tail diagnostics, age normalization, and exposure-adjusted count GLM.
  - `researchers.py`: Author productivity, scientific leadership, temporal trajectories, collaboration networks, research lines, and bibliometric laws.
  - `synthesis.py`: 6 selectable engineering-evidence dimensions covering methods, objectives, uncertainty, planning horizons, test systems, and solvers.
  - `forecasting.py`: Complete-year volume forecasts with rolling validation and conformal bands, topic trajectories, Bass diagnostics, and Kleinberg bursts.
  - `semantics.py`: Screening calibration, multi-projection semantic space, themes, semantic isolation, and persistent duplicate-review history.
  - `quality.py`: Metadata/PDF diagnostics, content readiness, anomaly audit, and hybrid retrieval.
  - `pipeline_layers.py`: Medallion flow, quality gates, audit history, source provenance, and Airflow operations.

### 4.3 Visual & Theme Standards
- **Responsive Width**: Always use `width="stretch"` for charts, tables, and containers. **NEVER use deprecated `use_container_width=True`**.
- **Transparent Polar Charts**: For radar and polar charts (e.g., `thematic_radar_chart`), always set `paper_bgcolor="rgba(0,0,0,0)"` and `polar_bgcolor="rgba(0,0,0,0)"` so that the chart adapts seamlessly to Streamlit dark and light themes without an opaque white box.
- **Language Rule**: Code, comments, docstrings, variable names, documentation files
  (`docs/`, `*.md`), and all dashboard UI text are in **English**.
- **Fixed Dark Theme**: The dashboard uses the dark Streamlit theme and matching
  Plotly tokens. Do not add a light-mode toggle or browser-dependent palette.
- **Publication Categories**: Store only the canonical English values `journal`,
  `conference`, `review`, and `other`. Classification precedence is review,
  conference, journal, then other.

---

## 5. Testing & Verification Standards

- **Suíte Size**: The test suite consists of **212 automated tests across 26 test files** in `tests/`.
- **Zero Live MySQL Dependency**: All tests run against in-memory SQLite fixtures (`tests/conftest.py`), creating one isolated session per layer (`raw_session`, `bronze_session`, `silver_session`, `gold_session`).
- **Fast Execution**: The complete suite must pass in under 5 seconds:
  ```bash
  uv run pytest
  ```
- **Linting & Formatting**: Code must conform strictly to Ruff rules:
  ```bash
  uv run ruff check && uv run ruff format --check
  ```

---

## 6. Developer & AI Assistant Commands

```bash
# Environment & Dependencies
uv sync                                    # Sync virtualenv from uv.lock

# Pipeline Execution
uv run lake-research-map --stage all         # Execute full medallion pipeline
uv run lake-research-map --stage <stage>     # Execute single stage (raw|bronze|silver|gold|embed|semantic)
uv run python -m lake_research_map.db.bootstrap  # Bootstrap schema & additive columns

# Dashboard
uv run streamlit run main.py               # Launch Streamlit app on http://localhost:8501

# Orchestration (Airflow + Dashboard)
docker compose up -d                       # Airflow (:8080) + Streamlit (:8501)

# Testing & Verification
uv run pytest                              # Run full test suite (212 tests)
uv run ruff check --fix && uv run ruff format  # Format and lint codebase
```

---

## 7. Golden Rules for Agents

1. **Preserve Architectural Explanations**: Existing comments explain *why* specific choices were made (e.g., why chunk sizes were chosen, why contrastive margins have zero cut, why PCA 50 is shared). Do not strip comments.
2. **Never Write MySQL Data from Pages**: Dashboard pages are read-only. Modifying pipeline data from the UI is prohibited; triggers must go through the Airflow REST API.
3. **Always Run Pytest and Ruff**: Never declare a task complete without executing `uv run pytest` and `uv run ruff check`.
4. **Git Branch Policy**: Direct commits of documentation on `main` are blocked by pre-commit hooks. Always make changes on feature or `docs/*` branches.
