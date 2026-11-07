---
name: streamlit-dashboard
description: Conventions for the lake-research-map Streamlit dashboard (src/apps/dashboard/) — page contract, chart contract, theme tokens, aggregation rules, and data pitfalls. Use whenever adding or editing a dashboard page, chart, or shared helper.
---

# lake-research-map Streamlit dashboard

This dashboard once grew past 2,600 lines with real duplication before a
refactor (it predates the repository's current git history). These rules exist
to keep it from drifting back.

## Page contract

- One module per page under `apps/dashboard/pages/`, exposing a single zero-arg
  `render() -> None`.
- Register it in `apps/dashboard/app.py`'s `PAGE_GROUPS` (sidebar section → list of
  `(render_fn, title, icon, url_path)`) — it's then wired into `st.navigation`. `PAGES` is
  a flat view derived from `PAGE_GROUPS`; don't edit it.
- Pages are read-only, with no exception (ADR-06, ADR-09): no database write, no file
  write, no run trigger. Every rendered string is English (`tests/test_dashboard_language.py`).
- Respect page ownership: every chart has exactly one home (CLAUDE.md "Page ownership" and
  the PRD §7 table). Never duplicate a chart on a second page or tab.
- Start every page with `articles_df = loaders.require_articles()`. This applies the
  global sidebar filters, handles the empty-state / `st.stop()` case, shows the
  degraded-mode warning, and renders the dataset-version caption
  (`render_population_provenance`) — don't reimplement any of them.
- `render_sidebar()` runs before `navigation.run()` in `app.py`, so global
  filter state (`global_year_range`, `global_sources`, `global_venues`,
  `global_publication_categories`, and the relevance-margin filter
  `global_min_margin`) is already in `st.session_state` by the time your page
  renders. `loaders.filter_signature()` is the cheap cache key over all five.
- **Exception: `pages/forecasting.py` calls `loaders.articles()` directly,
  not `require_articles()`.** A time-series forecast needs the full year
  history to fit a trend — applying the sidebar's global year filter would
  silently truncate the training data. It says so in its header and `hero_banner`
  ("trained only on complete years"), and still renders the provenance caption.
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
- Build the figure with a helper from `apps/dashboard/charts.py` when the shape
  matches: `source_bars`/`source_lines` for IEEE/Elsevier/Total comparisons,
  `publication_category_bars` for the journal/conference/review/other split,
  `topn_hbar` / `source_topn_hbar` for ranked top-N bars (overall / per source),
  `lorenz_chart` for concentration curves, `stacked_area` for composition-over-time.
  Add a new builder there rather than hand-rolling `px`/`go` calls in a page
  a second time.
- Compute the data for a chart in `apps/dashboard/analytics.py` (pure pandas, no
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
  English, sentence case, with the unit in parentheses only when it isn't
  obvious from the name: `Publication year`, `Articles`,
  `Chunk length (characters)`, `Fill rate (%)`. The category axis of a
  top-N horizontal bar counts too (`Author`, `Venue`, `Keyword`).
  A `go.Figure` starts with *no* axis titles and a px figure falls back to the
  raw column name (`count`, `stage`), so neither gives you a usable label for
  free. `render_chart` logs a warning naming any chart that still has a bare
  visible axis — run the app and read the terminal to audit the whole
  dashboard at once.
  An axis whose coordinates genuinely mean nothing — the t-SNE semantic map,
  the co-authorship network — hides its **tick values** (`showticklabels=False`,
  plus `showgrid`/`zeroline`/`ticks` off for a graph canvas) but still carries a
  name (`Semantic Axis 1 (Latent Space)`, `Position in the circular layout (no unit)`), with
  a comment and a caption saying why the numbers are gone. Both used to set
  `title=""`/`visible=False` and shipped as anonymous axes; hiding the *name* is
  never the answer, and silencing `render_chart`'s warning is not a reason to.
- Guard missing columns with `components.require_columns(df, [...], message)`
  instead of a bespoke `st.info(...)`.
- "Total" is a **real series/trace** (`TOTAL_COLOR`, `TOTAL_LABEL` from
  `theme.py`), not a reference line. `source_counts_by`/`cumulative_by_source`
  already return an explicit `total` column for this.

## Theme tokens (`apps/dashboard/theme.py`)

- `SOURCE_COLORS` / `SOURCE_LABELS` — IEEE blue / Elsevier orange, from each
  publisher's brand guidelines. Use these, don't invent new colors for the
  same two publishers.
- `CATEGORICAL_PALETTE` — the dataviz skill's 8-slot CVD-checked palette, for
  charts with many sub-categories (venues, keywords, network nodes).
- `TOTAL_COLOR` / `TOTAL_LABEL` — a vivid magenta/pink (`#ff2e77`), chosen to
  stand out against both `SOURCE_COLORS`. It used to be a dull gray and
  nearly disappeared next to the two saturated brand colors — don't revert
  it to a neutral tone.
- `OTHER_COLOR` — always for the catch-all "Others" bucket (`analytics.OTHERS_LABEL`),
  via `venue_color_map(order, others_label=OTHERS_LABEL)`.
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
   subheader (e.g. the titled cumulative charts in `production._cumulative_production`) or
   the title carries dynamic information the header doesn't (e.g. the
   Pearson-r value in `highlights._references_vs_citations`).
2. `polish_figure_layout()` still defends against the collision when a title
   *is* set: it detects `fig.layout.title.text` and reserves more top margin
   and a higher legend `y` automatically. Don't work around this by hardcoding
   your own `margin`/`legend.y` in a page — extend the function if the
   spacing ever needs to change.

## Aggregation rule

Pure pandas logic lives in `apps/dashboard/analytics.py`, `forecasting.py`,
`qualis.py` and `retrieval_eval.py` (no Streamlit import, so they're
testable independently). `apps/dashboard/loaders.py` is the only cache layer
(`@st.cache_data` with a TTL, plus `@st.cache_resource` for the retriever); raw
SQL against the medallion databases lives in `apps/dashboard/data.py`. A page
function should mostly be: call a loader, call an analytics function, call a
chart builder, call `render_chart`.

Article search is not a dashboard module: the Quality page's search and its
benchmark call `loaders.retriever()`, which holds one `apps.api.api.Retriever`
per published version as an `@st.cache_resource` (it owns indexes and a Qdrant
client, which must not be pickled per lookup). Don't reimplement ranking in the
dashboard.

## Data pitfalls specific to this corpus

- `citation_count` / `reference_count` are **NULL for "not collected", never
  0**. Elsevier's counts come from an offline enrichment cache
  (`data/enrichment_cache.json` merged with the OpenAlex observations by
  `ingest/enrichment.py`, applied by `transform/bronze_articles.py` during the
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
  (see the `hero_banner` on the Researchers and collaboration page for the
  wording to reuse).
- `year` ranges 1926–2027 (2027 = in-press). Use `analytics.valid_years` with
  an explicit window rather than a bare `pd.to_numeric` — a stray old/future
  row will otherwise dominate any trend/rate calculation.
- Run history lives in `gold.lit_pipeline_executions` / `lit_pipeline_runs`
  (stage stats JSON) and `lit_quality_results`; the Pipeline page's audit tab
  reads them. Bronze/silver are rebuilt every run, so any layer funnel chart is
  computed live from current row counts, not from that history — say so in
  the page rather than implying otherwise. No page starts a run (ADR-09): the
  Pipeline page shows the CLI commands instead.
- The **current year in the corpus is always a partial year** — it's
  collected by hand mid-year, not a closed dataset. `apps/dashboard/forecasting.py`
  (used by `pages/forecasting.py`) treats its `HOLDOUT_YEAR` this way
  explicitly: it's only monitored, never trained on, and every note/label calls
  it "partial" rather than presenting it as a finished year. Any new
  year-over-year comparison involving the current year should do the same.
- Forecasting compares a persistence baseline with plain linear and log-linear
  trends (`sklearn.linear_model.LinearRegression`), selected by rolling-origin
  validation with per-horizon backtests, with conformal bands from held-out
  errors — not a heavier model: the usable series is short (yearly points from
  `MIN_TRAIN_YEAR = 2010` onward). Don't reach for a model class
  that needs more data than the corpus has just because it's "more ML."
- The semantic signals (`gold.lit_semantics`, joined on `doi` by
  `loaders.with_semantics`) carry **two** scores: `relevance_score` (cosine to
  the review's topic anchor) and `offtopic_score` (cosine to the logistics
  reading of the same query). What screens an article is the derived
  `relevance_margin` between them, and its **zero** is the threshold — not a
  percentile, which the two overlapping distributions made unusable. Both the
  sidebar filter and the Screening and discovery page fall back to the old percentile view
  when `offtopic_score` is absent, i.e. when the database's last `semantic` run
  predates the contrastive anchor; keep that fallback when editing either.
- Chunk vectors live in **Qdrant**, one collection per dataset version
  (ADR-08), filled by the `embed` stage (`fastembed`'s local ONNX model
  `BAAI/bge-small-en-v1.5`, no API key) — not by `--stage gold`. Pages get
  the abstract vectors through `data.load_abstract_embeddings_data` (via
  `loaders`), which returns empty when the store does not answer; treat that as "no
  vectors", never as an error. The readiness gauge on Quality and RAG reads
  the *provenance* Gold records (`has_embedding` = `embed_revision IS NOT
  NULL`), so a page render never calls Qdrant. If it reads 0%, the stage
  has not run for the current version, not that something is broken.
