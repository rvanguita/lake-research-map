"""??????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????'?????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????'??????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????'???????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????"""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from lake_research_map.dashboard import loaders
from lake_research_map.dashboard.components import (
    hero_banner,
    metric_row,
    page_header,
    render_chart,
    render_pipeline_controls,
)
from lake_research_map.dashboard.theme import (
    CATEGORICAL_PALETTE,
    OTHER_COLOR,
    SOURCE_COLORS,
    SOURCE_LABELS,
    TOTAL_COLOR,
    hex_to_rgba,
)

LAYER_ORDER = ("raw", "bronze", "silver", "gold")


def _render_search_provenance() -> None:
    """Show immutable search provenance without writing to pipeline tables."""
    st.subheader("Provenance of the search")
    configs = loaders.search_configs()
    if configs.empty:
        st.info("The search configuration has not yet been ingested in the raw layer.")
        return

    st.caption(
        "The records below preserve the queries that produced the exports. "
        "They document the collection and do not represent articles of the corpus."
    )
    for _, row in configs.iterrows():
        source = str(row.get("source") or "Source")
        with st.expander(SOURCE_LABELS.get(source, source.title())):
            st.code(str(row.get("query_string") or "—"), language=None, wrap_lines=True)
            columns = st.columns(2)
            columns[0].metric("Year range", str(row.get("year_range") or "—"))
            columns[1].metric("Filters", str(row.get("filters") or "—"))
            search_url = row.get("search_url")
            if search_url:
                st.link_button(
                    "Open original search", str(search_url), icon=":material/open_in_new:"
                )
            st.caption(f"File: {row.get('source_file') or '—'}")


def render() -> None:
    page_header(
        "🏗️",
        "Pipeline and provenance",
        "The raw → bronze → silver → gold funnel: how much survives each stage, why, and where "
        "metadata coverage improves or degrades.",
    )

    runs_df = loaders.pipeline_runs()
    executions_df = loaders.pipeline_executions()
    versions_df = loaders.dataset_versions()
    publication_df = loaders.publication_state()
    quality_df = loaders.quality_results()
    if runs_df.empty:
        hero_banner(
            "No execution history",
            "Run the pipeline to populate its history. Statistics are recorded automatically "
            "for each execution.",
        )
    else:
        last = runs_df.iloc[0]
        active = None
        if not publication_df.empty:
            active = publication_df.iloc[0].get("active_version_id")
        active_text = f" Active version: <code>{str(active)[:12]}</code>." if active else ""
        hero_banner(
            "Execution history",
            f"Latest run: <b>{last['stage']}</b> at "
            f"<code>{last['finished_at']}</code> — status: <b>{last['status']}</b>."
            f"{active_text}",
        )

    funnel_df = loaders.layer_funnel()
    row_counts = loaders.row_counts()

    if funnel_df.empty or funnel_df[["raw", "bronze", "silver", "gold"]].sum().sum() == 0:
        st.warning(
            "No data found in any layer yet. "
            "(`uv run lake-research-map --stage all`) and reload this page."
        )
        has_pipeline_data = False
    else:
        has_pipeline_data = True
        _headline_metrics(funnel_df)

    tab_flow, tab_quality, tab_audit, tab_operations = st.tabs(
        [
            "Flow and retention",
            "Layer quality",
            "Audit",
            "Implementation and provenance",
        ],
        on_change="rerun",
        key="pipeline_primary_tab",
    )

    if tab_flow.open:
        if has_pipeline_data:
            _sankey_funnel(funnel_df)
            _retention_by_stage(funnel_df)
        else:
            st.info("Run the raw layer to start the funnel.")
    elif tab_quality.open:
        if not quality_df.empty:
            _render_contract_status(quality_df)
        elif has_pipeline_data:
            st.info(
                "There are still no contracts persisted; the diagnosis below is just one "
                "Legacy verification on the loaded tables."
            )
            _drift_check(funnel_df)
        if has_pipeline_data:
            _metadata_coverage_by_layer()
            st.subheader("Raw row count by table")
            st.dataframe(row_counts, hide_index=True, width="stretch")
        else:
            st.info("There are still no layers to compare.")
    elif tab_audit.open:
        _render_version_audit(versions_df, publication_df)
        _render_execution_audit(executions_df, runs_df)
        _render_source_changes()
        st.subheader("Records rejected in the silver layer")
        rejected_df = loaders.rejected_records()
        if rejected_df.empty:
            st.info("Run `--stage silver` for popular.")
        else:
            st.caption(
                f"{len(rejected_df):,} records excluded from the Silver layer — retained here for "
                "Systematic review audit."
            )
            st.dataframe(rejected_df, hide_index=True, width="stretch")
    elif tab_operations.open:
        render_pipeline_controls()
        _render_search_provenance()


def _render_contract_status(quality_df: pd.DataFrame) -> None:
    st.subheader("Executable contracts")
    latest = quality_df.sort_values("checked_at").drop_duplicates(
        ["dataset_version_id", "stage", "check_id"], keep="last"
    )
    failures = latest[(latest["severity"] == "error") & (~latest["passed"].astype(bool))]
    warnings = latest[(latest["severity"] == "warning") & (~latest["passed"].astype(bool))]
    metric_row(
        [
            ("Verifications", f"{len(latest):,}", "last result by contract"),
            ("Blocking failures", f"{len(failures):,}", "error severity"),
            ("Warnings", f"{len(warnings):,}", "non-blocking"),
        ]
    )
    if failures.empty:
        st.success("No blocking failure in the most recent contract run.")
    else:
        st.error(
            f"{len(failures)} blocking contract(s) failed; the candidate version cannot "
            "be published."
        )
    display = latest[
        [
            "stage",
            "check_id",
            "severity",
            "passed",
            "observed",
            "expected",
            "checked_at",
        ]
    ].rename(
        columns={
            "stage": "Stage",
            "check_id": "Contract",
            "severity": "Severity",
            "passed": "Passed",
            "observed": "Observed",
            "expected": "Expected",
            "checked_at": "Checked at",
        }
    )
    st.dataframe(display, hide_index=True, width="stretch")


def _render_version_audit(versions_df: pd.DataFrame, publication_df: pd.DataFrame) -> None:
    st.subheader("Corpus versions")
    if versions_df.empty:
        st.info("No deterministic version was recorded.")
        return
    active = None
    working = None
    if not publication_df.empty:
        active = publication_df.iloc[0].get("active_version_id")
        working = publication_df.iloc[0].get("working_version_id")
    metric_row(
        [
            ("Active version", str(active)[:12] if active else "—", "published Gold"),
            ("Working version", str(working)[:12] if working else "—", "current pipeline"),
            ("Registered versions", f"{len(versions_df):,}", None),
        ]
    )
    columns = [
        "version_id",
        "status",
        "parent_version_id",
        "source_manifest_sha256",
        "code_revision",
        "created_at",
        "published_at",
        "failure_reason",
    ]
    st.dataframe(
        versions_df[[col for col in columns if col in versions_df]],
        hide_index=True,
        width="stretch",
    )


def _render_execution_audit(executions_df: pd.DataFrame, runs_df: pd.DataFrame) -> None:
    st.subheader("Correlated executions and stages")
    if executions_df.empty:
        if runs_df.empty:
            st.info("Run the pipeline to populate its history.")
        else:
            st.dataframe(runs_df, hide_index=True, width="stretch")
        return
    st.dataframe(executions_df, hide_index=True, width="stretch")
    if not runs_df.empty:
        st.caption("Stage attempts linked to the above executions.")
        st.dataframe(runs_df, hide_index=True, width="stretch")


def _render_source_changes() -> None:
    st.subheader("Reconciliation of source files")
    changes = loaders.source_changes()
    if changes.empty:
        st.info("No versioned reconciliation was recorded.")
        return
    st.dataframe(changes, hide_index=True, width="stretch")


def _headline_metrics(funnel_df: pd.DataFrame) -> None:
    totals = funnel_df[["raw", "bronze", "silver", "gold"]].sum()
    dropped = int(funnel_df["dropped_no_doi"].sum())
    metric_row(
        [
            ("📥 Raw", f"{int(totals['raw']):,}", "linhas CSV + entradas bib"),
            ("🥉 Bronze", f"{int(totals['bronze']):,}", None),
            (
                "🥈 Silver",
                f"{int(totals['silver']):,}",
                f"−{dropped} without DOI" if dropped else "no losses",
            ),
            ("🥇 Gold", f"{int(totals['gold']):,}", None),
        ]
    )


def _sankey_funnel(funnel_df: pd.DataFrame) -> None:
    st.subheader("Funil raw → bronze → silver → gold")

    labels = []
    label_index: dict[str, int] = {}

    def _idx(label: str) -> int:
        if label not in label_index:
            label_index[label] = len(labels)
            labels.append(label)
        return label_index[label]

    sources, targets, values, link_colors = [], [], [], []
    for _, row in funnel_df.iterrows():
        src_label = SOURCE_LABELS.get(row["source"], row["source"])
        color = SOURCE_COLORS.get(row["source"], OTHER_COLOR)

        raw_node = _idx(f"Raw ({src_label})")
        bronze_node = _idx(f"Bronze ({src_label})")
        silver_node = _idx(f"Silver ({src_label})")
        gold_node = _idx(f"Gold ({src_label})")
        dropped_node = _idx("Discarded (without DOI)")

        if row["raw"] > 0:
            sources.append(raw_node)
            targets.append(bronze_node)
            values.append(row["raw"])
            link_colors.append(color)

        kept_to_silver = max(row["bronze"] - row["dropped_no_doi"], 0)
        if kept_to_silver > 0:
            sources.append(bronze_node)
            targets.append(silver_node)
            values.append(kept_to_silver)
            link_colors.append(color)
        if row["dropped_no_doi"] > 0:
            sources.append(bronze_node)
            targets.append(dropped_node)
            values.append(row["dropped_no_doi"])
            link_colors.append(OTHER_COLOR)

        if row["gold"] > 0:
            sources.append(silver_node)
            targets.append(gold_node)
            values.append(row["gold"])
            link_colors.append(color)

    fig = go.Figure(
        go.Sankey(
            node=dict(
                label=labels,
                pad=18,
                thickness=16,
                color=CATEGORICAL_PALETTE[0],
                line=dict(color="rgba(255,255,255,0.15)", width=0.5),
            ),
            link=dict(
                source=sources,
                target=targets,
                value=values,
                color=[hex_to_rgba(c, 0.55) for c in link_colors],
            ),
        )
    )
    fig.update_layout(title="Article volume by stage and source, with labeled losses")
    render_chart(
        fig,
        caption="The visible loss ('Discarded (without DOI)') happens in the silver: bronze articles without DOI "
        "normalized DOI never form a Silver record (`silver_articles.py`). Bronze may contain "
        "more lines than the direct sum of the raw because it is upsert-only and never removes orphan lines (see the "
        "drift panel below).",
    )


def _retention_by_stage(funnel_df: pd.DataFrame) -> None:
    st.subheader("Retention by stage and source")
    long_df = funnel_df.melt(
        id_vars="source",
        value_vars=["raw", "bronze", "silver", "gold"],
        var_name="layer",
        value_name="count",
    )
    long_df["layer"] = pd.Categorical(long_df["layer"], categories=LAYER_ORDER, ordered=True)
    long_df = long_df.sort_values("layer")

    fig = go.Figure()
    for src in ("ieee", "elsevier"):
        sub = long_df[long_df["source"] == src]
        fig.add_bar(
            x=sub["layer"],
            y=sub["count"],
            name=SOURCE_LABELS.get(src, src),
            marker_color=SOURCE_COLORS.get(src),
        )
    fig.update_layout(barmode="stack")

    totals = long_df.groupby("layer", observed=True)["count"].sum().reindex(LAYER_ORDER)
    fig.add_trace(
        go.Scatter(
            x=list(LAYER_ORDER),
            y=totals.values,
            name="Total",
            mode="lines+markers",
            line=dict(color=TOTAL_COLOR, width=2.5),
            marker=dict(size=8),
        )
    )
    fig.update_layout(xaxis_title="Layer", yaxis_title="Number of articles")
    render_chart(
        fig,
        caption="Bronze deduplicates only within each source (`(source, source_id)` key); silver and gold "
        "then converge to the same total because there is no DOI overlap between IEEE and Elsevier in this "
        "corpus.",
    )


def _drift_check(funnel_df: pd.DataFrame) -> None:
    st.subheader("Raw-to-Bronze drift")
    drift = funnel_df.copy()
    drift["drift"] = drift["bronze"] - drift["raw"]
    drift_display = drift[["source", "raw", "bronze", "drift"]].copy()
    drift_display["source"] = drift_display["source"].map(lambda s: SOURCE_LABELS.get(s, s))

    has_drift = (drift["drift"] != 0).any()
    st.dataframe(drift_display, hide_index=True, width="stretch")
    if has_drift:
        st.warning(
            "Under the append source policy, Bronze retains records from archived inputs that are "
            "absent from the current download. A positive `bronze > raw` can therefore reflect "
            "retained history rather than corruption; inspect the source-change audit before treating "
            "Bronze as a mirror of the files currently on disk."
        )
    else:
        st.success("Bronze and Raw are aligned to both bases — no drift signal.")


def _metadata_coverage_by_layer() -> None:
    st.subheader("Metadata coverage per layer")
    by_layer = loaders.articles_by_layer()

    fields = ["doi", "abstract", "keywords", "citation_count", "has_pdf"]
    field_labels = {
        "doi": "DOI",
        "abstract": "Abstract",
        "keywords": "Keywords",
        "citation_count": "Citations",
        "has_pdf": "Linked PDF",
    }
    rows = []
    for layer in ("bronze", "silver", "gold"):
        df = by_layer.get(layer, pd.DataFrame())
        if df.empty:
            continue
        for field in fields:
            if field not in df.columns:
                continue
            if field in ("keywords",):
                pct = df[field].apply(lambda v: isinstance(v, list) and len(v) > 0).mean()
            elif field == "has_pdf":
                pct = df[field].fillna(False).astype(bool).mean()
            else:
                pct = df[field].notna().mean()
                if df[field].dtype == object:
                    pct = df[field].fillna("").astype(str).str.strip().ne("").mean()
            rows.append({"layer": layer, "field": field_labels[field], "coverage": pct * 100})

    if not rows:
        st.info("No layer with enough data to compare metadata coverage.")
        return

    coverage_df = pd.DataFrame(rows)
    coverage_df["layer"] = pd.Categorical(
        coverage_df["layer"], categories=["bronze", "silver", "gold"], ordered=True
    )
    fig = px.bar(
        coverage_df.sort_values("layer"),
        x="field",
        y="coverage",
        color="layer",
        barmode="group",
        color_discrete_sequence=[
            CATEGORICAL_PALETTE[1],
            CATEGORICAL_PALETTE[0],
            CATEGORICAL_PALETTE[2],
        ],
        labels={"field": "Field", "coverage": "Completeness (%)", "layer": "Layer"},
    )
    fig.update_traces(hovertemplate="<b>%{x}</b><br>%{data.name}: %{y:.1f}%<extra></extra>")
    render_chart(
        fig,
        caption="Shows what each layer gains and loses. Gold projects Silver without `issn`, `volume`, "
        "`issue`, `pages`, or Silver-only quality flags, while retaining `sources` so the IEEE/Elsevier "
        "breakdown remains available in Gold.",
    )
