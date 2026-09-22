---
name: streamlit-dashboard
description: Conventions for the lake-research-map Streamlit dashboard (src/lake_research_map/dashboard/) — page contract, chart contract, theme tokens, aggregation rules, and data pitfalls. Use whenever adding or editing a dashboard page, chart, or shared helper.
---

# lake-research-map Streamlit dashboard

This dashboard grew past 2,600 lines with real duplication before a refactor
(see git history around "Refatoração do dashboard + novas análises"). These
rules exist to keep it from drifting back.

## Page contract

- One module per page under `dashboard/pages/`, exposing a single zero-arg
  `render() -> None`.
- Register it in `dashboard/app.py`'s `PAGES` list — `(render_fn, title, icon,
  url_path)` — and it's automatically wired into `st.navigation`.
- Start every page with `articles_df = loaders.require_articles()` (or
  `loaders.filtered_chunks()` for chunk-only pages). This applies the global
  sidebar filters and handles the empty-state / `st.stop()` case — don't
  reimplement either.
- `render_sidebar()` runs before `navigation.run()` in `app.py`, so global
  filter state (`global_year_range`, `global_sources`, `global_venues`) is
  already in `st.session_state` by the time your page renders.
- **Exception: `pages/forecasting.py` calls `loaders.articles()` directly,
  not `require_articles()`.** A time-series forecast needs the full year
  history to fit a trend — applying the sidebar's global year filter would
  silently truncate the training data. It says so in its own `hero_banner`.
  Any other page that needs the *entire* corpus regardless of sidebar state
  should follow the same pattern (and disclose it the same way) rather than
  quietly ignoring `require_articles()`.

## Chart contract

- **Never add a reference/guide line to a chart** — no `add_hline`,
  `add_vline`, `add_shape`, `add_hrect`/`add_vrect`, or benchmark
  annotations. This was ripped out of `theme.py` (`add_source_layers`) on
  purpose: it covered the data with dotted lines and text boxes. If a mean or
  benchmark matters, put it in a `metric_row` card or the chart's `caption`
  instead.
- Build the figure with a helper from `dashboard/charts.py` when the shape
  matches: `source_bars`/`source_lines` for IEEE/Elsevier/Total comparisons,
  `topn_hbar` for ranked top-N bars, `stacked_area` for composition-over-time.
  Add a new builder there rather than hand-rolling `px`/`go` calls in a page
  a second time.
- Compute the data for a chart in `dashboard/analytics.py` (pure pandas, no
  `streamlit` import) — not inline in the page function. Reuse
  `valid_years`, `source_counts_by`, `cumulative_by_source`,
  `cumulative_by_venue`, `source_means`, `explode_authors`,
  `explode_keywords`, `canonical_author` before writing a new groupby.
- Finish every chart with `components.render_chart(fig, caption=..., height=...)`
  — it applies `polish_figure_layout` and calls `st.plotly_chart` +
  `st.caption` in one place. Don't call `polish_figure_layout` /
  `st.plotly_chart` directly in a page.
- **Name both axes.** Every chart with visible axes declares them — via the
  builders' `x_title`/`y_title`, a px `labels={...}`, or `update_layout`.
  Portuguese, sentence case, with the unit in parentheses only when it isn't
  obvious from the name: `Ano de publicação`, `Quantidade de artigos`,
  `Tamanho do chunk (caracteres)`, `Preenchimento (%)`. The category axis of a
  top-N horizontal bar counts too (`Autor`, `Periódico`, `Palavra-chave`).
  A `go.Figure` starts with *no* axis titles and a px figure falls back to the
  raw column name (`count`, `stage`), so neither gives you a usable label for
  free. `render_chart` logs a warning naming any chart that still has a bare
  visible axis — run the app and read the terminal to audit the whole
  dashboard at once.
  An axis whose coordinates genuinely mean nothing — the t-SNE semantic map,
  the co-authorship network — hides its **tick values** (`showticklabels=False`,
  plus `showgrid`/`zeroline`/`ticks` off for a graph canvas) but still carries a
  name (`Dimensão 1 (t-SNE)`, `Posição no layout circular (sem unidade)`), with
  a comment and a caption saying why the numbers are gone. Both used to set
  `title=""`/`visible=False` and shipped as anonymous axes; hiding the *name* is
  never the answer, and silencing `render_chart`'s warning is not a reason to.
- Guard missing columns with `components.require_columns(df, [...], message)`
  instead of a bespoke `st.info(...)`.
- "Total" is a **real series/trace** (`TOTAL_COLOR`, `TOTAL_LABEL` from
  `theme.py`), not a reference line. `source_counts_by`/`cumulative_by_source`
  already return an explicit `total` column for this.

## Theme tokens (`dashboard/theme.py`)

- `SOURCE_COLORS` / `SOURCE_LABELS` — IEEE blue / Elsevier orange, from each
  publisher's brand guidelines. Use these, don't invent new colors for the
  same two publishers.
- `CATEGORICAL_PALETTE` — the dataviz skill's 8-slot CVD-checked palette, for
  charts with many sub-categories (venues, keywords, network nodes).
- `TOTAL_COLOR` / `TOTAL_LABEL` — a vivid magenta/pink (`#ff2e77`), chosen to
  stand out against both `SOURCE_COLORS`. It used to be a dull gray and
  nearly disappeared next to the two saturated brand colors — don't revert
  it to a neutral tone.
- `OTHER_COLOR` — always for the catch-all "Outros" bucket, via
  `venue_color_map(order, others_label=...)`.
- `CHART_HEIGHT` — the shared height for side-by-side chart pairs.
- New chart-design decisions (color formula, mark selection, KPI-tile layout)
  should go through the `dataviz` skill first; this dashboard's palette
  already came from it.

### Fixed dark theme

The dashboard has exactly one theme. The base palette — page and sidebar
backgrounds, text, primary, and the Plotly categorical colors — comes from
`.streamlit/config.toml`, which Streamlit applies itself. No module injects
page CSS.

An earlier version had a sidebar light/dark radio backed by
`st.session_state["dashboard_theme_mode"]`, and an `apply_dashboard_theme()`
that wrote ~200 lines of `<style>`. Both are gone. `PRD.md` NFR-07 now requires
a *fixed* dark theme, and the CSS injector had become dead code that nothing
called while `config.toml` quietly did the real work — it was removed on
2026-09-22 rather than left to be read as the styling entry point.

What remains is the token set `config.toml` cannot reach. `theme._tokens()`
returns `_DARK_TOKENS`, and `theme._active_theme_type()` returns `"dark"`
unconditionally. Call the public `theme_tokens()` from a page or component that
colors its own inline HTML, or a Plotly Indicator (which `polish_figure_layout`
does not touch) — see `components.hero_banner` or
`quality._embedding_readiness`. **Never hardcode a hex color for chart
backgrounds or inline chrome**: add a token instead, so one edit moves every
surface using it. A data color scale (e.g. `color_continuous_scale` on a
heatmap or scatter) is not chrome and is fine to hardcode — it colors marks by
value, not the page or chart background.

The chart canvas itself is the one exception: `theme.CHART_PAPER_BG`
(`rgba(0,0,0,0)`) is a plain module constant, not a token, because a
transparent canvas is the right default — it lets the page's gradient
through instead of laying a flat slab over it. `chart_bg` is still a token and
still solid, for the things that need a real color to stand on (the gauge track
in `quality.py`, the partial-year marker halo in `forecasting.py`, the hover
box). Transparency is also why the shared template names `hoverlabel` and
`modebar` explicitly: Plotly derives both from `paper_bgcolor`, and a
transparent one gives an unreadable tooltip and invisible modebar icons.

**Chart background and font must be set on the figure, not only on the shared
template** — `polish_figure_layout` does this and says why. Streamlit's frontend
runs `layoutWithThemeDefaults` over every Plotly spec, *including* with
`theme=None`, and fills `paper_bgcolor`, `plot_bgcolor` and `font` from
Streamlit's own theme (which follows the browser/system setting, not
`config.toml`) whenever the figure's layout doesn't carry them. It never reads the
template. Declaring them template-only is what made every chart render on a
near-black background; `tests/test_theme.py` guards it.

### Avoiding title/legend overlap

Plotly places a horizontal top legend at a fixed `y` regardless of how many
lines the chart's own title wraps to, so a title + legend combination can
overlap (this happened with `source_bars`/`source_lines` charts once titles
got long). Two rules keep this from recurring:

1. **Don't set a Plotly-level `title` when the page already has an
   `st.subheader`/`st.markdown` header directly above the chart** — that's
   true for most charts in this dashboard. Only give a chart its own title
   when it sits in a column next to a sibling chart under one shared
   subheader (e.g. `production._venue_comparison`'s IEEE/Elsevier pair) or
   the title carries dynamic information the header doesn't (e.g. the
   Pearson-r value in `highlights._references_vs_citations`).
2. `polish_figure_layout()` still defends against the collision when a title
   *is* set: it detects `fig.layout.title.text` and reserves more top margin
   and a higher legend `y` automatically. Don't work around this by hardcoding
   your own `margin`/`legend.y` in a page — extend the function if the
   spacing ever needs to change.

## Aggregation rule

Pure pandas logic lives in `dashboard/analytics.py` (no Streamlit import, so
it's testable/cacheable independently). UI-facing caching wrappers live in
`dashboard/loaders.py` with `@st.cache_data(ttl=60)`; raw SQL against the
medallion databases lives in `dashboard/data.py`. A page function should
mostly be: call a loader, call an analytics function, call a chart builder,
call `render_chart`.

## Data pitfalls specific to this corpus

- `citation_count` / `reference_count` are **NULL for "not collected", never
  0**. Elsevier's counts come from an offline enrichment cache
  (`data/enrichment_cache.json`, applied in `ingest/enrichment.py` during the
  bronze stage) — don't treat a null as zero impact.
- `gold.articles` has a `sources` column (JSON list, mirrors silver) — added
  so gold can be split IEEE/Elsevier like every other layer. If you add a new
  gold column, check whether bronze/silver need the same for source-breakdown
  charts to keep working across all three layers.
- IEEE keyword counts are inflated: bronze concatenates `Author Keywords`
  with `IEEE Terms` without deduplicating (`transform/bronze_articles.py`).
  Don't present "IEEE has richer keywords" as a finding without this caveat.
- Author names are **not the same identity across sources**: IEEE exports
  initials (`J. Liu`), Elsevier full names (`Junyong Liu`), IEEE's own `.bib`
  uses `Last, First`. `analytics.canonical_author` folds these to `"initial
  surname"` — a real simplification that can also merge distinct people who
  share an initial+surname. Any author-identity chart must disclose this
  (see `hero_banner` on the Pesquisadores page for the wording to reuse).
- `year` ranges 1926–2027 (2027 = in-press). Use `analytics.valid_years` with
  an explicit window rather than a bare `pd.to_numeric` — a stray old/future
  row will otherwise dominate any trend/rate calculation.
- There is no pipeline run-history table. Stage stats
  (`pipeline.py`'s `run_raw`/`run_bronze`/etc.) only go to `print` and Airflow
  task logs; silver/gold are truncated and rebuilt every run. Any
  "camadas/funil" chart is computed live from current row counts, not from a
  persisted history — say so in the page rather than implying otherwise.
- The **current year in the corpus is always a partial year** — it's
  collected by hand mid-year, not a closed dataset. `dashboard/forecasting.py`
  (used by `pages/forecasting.py`) treats its `HOLDOUT_YEAR` this way
  explicitly: it's used for model validation but every note/label calls it
  "parcial" rather than presenting it as a finished year. Any new
  year-over-year comparison involving the current year should do the same.
- Forecasting uses plain regression (`sklearn.linear_model.LinearRegression`,
  optionally with `PolynomialFeatures`, plus a log-linear fit for exponential
  growth), not a heavier model — the usable series is short (~16 yearly
  points from `MIN_TRAIN_YEAR = 2010` onward). Don't reach for a model class
  that needs more data than the corpus has just because it's "more ML."
- The semantic signals (`gold.lit_semantics`, joined on `doi` by
  `loaders.with_semantics`) carry **two** scores: `relevance_score` (cosine to
  the review's topic anchor) and `offtopic_score` (cosine to the logistics
  reading of the same query). What screens an article is the derived
  `relevance_margin` between them, and its **zero** is the threshold — not a
  percentile, which the two overlapping distributions made unusable. Both the
  sidebar filter and the Semântica page fall back to the old percentile view
  when `offtopic_score` is absent, i.e. when the database's last `semantic` run
  predates the contrastive anchor; keep that fallback when editing either.
- `lit_gold.chunks.embedding` is filled by the `embed` pipeline stage
  (`transform/embeddings.py`, via `fastembed`'s local ONNX model
  `BAAI/bge-small-en-v1.5`, no API key), not automatically by `--stage gold`.
  It's idempotent — only `embedding IS NULL` rows are processed — so a chunk
  never gets re-embedded, and a partial run (killed mid-way) resumes cleanly
  since each batch commits before the next one starts. If the embeddings
  gauge on Qualidade e RAG ever reads 0% again, it means this stage hasn't
  run yet for the current chunks, not that something is broken.
