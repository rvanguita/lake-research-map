"""👥 Researchers — who publishes, with whom, and who leads each line of research."""

from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from lake_research_map.dashboard import loaders
from lake_research_map.dashboard.analytics import (
    RECENT_WINDOW_YEARS,
    analyze_coauthorship_partners,
    author_count_series,
    author_impact_advanced_indices,
    author_m_quotient_analysis,
    author_productivity_trend,
    coauthorship_community_detection,
    cumulative_researchers,
    explode_keywords,
    gini_coefficient,
    graph_advanced_metrics,
    lorenz_curve,
    output_impact_correlation,
    researchers_by_year,
    source_means,
    valid_years,
)
from lake_research_map.dashboard.charts import (
    lorenz_chart,
    source_bars,
    source_lines,
    source_topn_hbar,
    topn_hbar,
)
from lake_research_map.dashboard.components import (
    article_table,
    hero_banner,
    metric_row,
    page_header,
    render_chart,
)
from lake_research_map.dashboard.theme import CATEGORICAL_PALETTE, theme_tokens

TOP_AUTHORS = 25
MIN_PAPERS_FOR_NETWORK = 4
TOP_NETWORK_AUTHORS = 18


def render() -> None:
    page_header(
        "👥",
        "Researchers and collaboration",
        "Production, collaboration and lines of research of the authors of the corpus.",
    )

    articles_df = loaders.require_articles()
    author_rows = loaders.author_table(loaders.filter_signature())

    if author_rows.empty:
        st.info("'authors' column not available or empty in this layer.")
        return

    hero_banner(
        "...Canonicized names, non-identities verified",
        "The names are normalized for <b>initial + surname</b> (e.g. <code>Junyong Liu</code> and "
        "<code>J. Liu</code> collapse to the same key) because IEEE exports initials and Elsevier exports "
        "full names. This merges spelling variants for the same researcher, but may also merge <b>namesakes "
        "different </b> that share initial and surname — treat the numbers as an approximation, not "
        "como identidade confirmada.",
    )

    n_authors = author_rows["author_key"].nunique()
    per_author_counts = (
        author_rows.groupby("author_key")["doi"].nunique()
        if "doi" in author_rows.columns
        else author_rows.groupby("author_key").size()
    )
    n_5plus = int((per_author_counts >= 5).sum())
    n_3plus = int((per_author_counts >= 3).sum())
    metric_row(
        [
            ("👥 Different authors (canonicalized)", f"{n_authors:,}", None),
            ("🏅 With ≥5 articles", f"{n_5plus:,}", None),
            ("📗 With ≥3 articles", f"{n_3plus:,}", None),
            (
                "🧑‍🤝‍🧑 Average authors per article",
                f"{author_count_series(articles_df).mean():.1f}",
                None,
            ),
        ]
    )

    st.divider()
    section = st.selectbox(
        "Analysis area",
        [
            "Productivity and ranking",
            "Corpus impact",
            "Temporal trajectory",
            "Collaboration and networks",
            "Research lines",
            "Bibliometric laws",
        ],
    )

    if section == "Productivity and ranking":
        sub_prolific, sub_lead = st.tabs(["More Prolific", "🥇 1º/2º Autor"])
        with sub_prolific:
            _top_authors(author_rows)
        with sub_lead:
            _lead_authors_ranking(author_rows)

    elif section == "Corpus impact":
        _render_scientific_leadership_tab(articles_df)

    elif section == "Temporal trajectory":
        (
            sub_active,
            sub_heatmap,
            sub_emerging,
        ) = st.tabs(
            [
                "👥 Annual & Cumulative Volume",
                "🔥 Activity heatmap",
                "🌱 Emerging vs. established",
            ]
        )
        with sub_active:
            active_view = (
                st.segmented_control(
                    "Activity Metric",
                    options=[
                        "Researchers/Year",
                        "Cumulative Researchers",
                        "1st/2nd Authors/Year",
                        "1st/2nd Cumulative",
                    ],
                    default="Researchers/Year",
                    key="res_active_view_selector",
                )
                or "Researchers/Year"
            )
            if active_view == "Researchers/Year":
                _researchers_by_year(articles_df)
            elif active_view == "Cumulative Researchers":
                _cumulative_researchers_chart(articles_df)
            elif active_view == "1st/2nd Authors/Year":
                _lead_authors_by_year(articles_df)
            else:
                _cumulative_lead_authors_chart(articles_df)

        with sub_heatmap:
            _production_heatmap(author_rows)
        with sub_emerging:
            _emerging_vs_established(author_rows)

    elif section == "Collaboration and networks":
        sub_teams, sub_network, sub_cognitive = st.tabs(
            [
                "👥 Teams & size",
                "🕸️ Co-authorship network (Louvain)",
                "🧠 Cognitive distance vs. Impact",
            ]
        )
        with sub_teams:
            _render_team_collaboration_stats(articles_df)
        with sub_network:
            _coauthorship_network(author_rows)
        with sub_cognitive:
            _cognitive_distance_analysis(articles_df, author_rows)

    elif section == "Research lines":
        (
            sub_leaders,
            sub_trend,
            sub_kw_year,
            sub_kw_cum,
            sub_kw_profile,
            sub_kw_shift,
        ) = st.tabs(
            [
                "🔎 Line Leaders",
                "📈 Annual trajectory",
                "👥 Researchers/Year",
                "📈 Cumulative Researchers",
                "🏷️ Keyword profile",
                "🔀 Focus Change",
            ]
        )
        selected_kw, scoped_authors, dois_with_kw = _research_line_selector(
            author_rows, articles_df
        )
        with sub_leaders:
            if selected_kw:
                _research_line_top_authors(selected_kw, scoped_authors)
        with sub_trend:
            if selected_kw:
                _research_line_trend(selected_kw, scoped_authors)
        with sub_kw_year:
            if selected_kw:
                _research_line_researchers_by_year(selected_kw, articles_df, dois_with_kw)
        with sub_kw_cum:
            if selected_kw:
                _research_line_researchers_cumulative(selected_kw, articles_df, dois_with_kw)
                _research_line_articles(articles_df, scoped_authors, selected_kw)

        selected_author = _author_keyword_selector(author_rows)
        working_kw = (
            _author_keyword_working(selected_author, author_rows, articles_df)
            if selected_author
            else None
        )
        with sub_kw_profile:
            if working_kw is not None:
                _author_keyword_overview(selected_author, working_kw)
        with sub_kw_shift:
            if working_kw is not None:
                _author_keyword_shift(working_kw)

    elif section == "Bibliometric laws":
        sub_table, sub_lotka, sub_concentration, sub_trend_table, sub_vs_impact = st.tabs(
            [
                "📋 Complete table",
                "📐 Lotka's law",
                "📉 Concentration (Gini/Lorenz)",
                "📈 Productivity trend",
                "📊 Volume × impact",
            ]
        )
        with sub_table:
            matrix = _full_output_table(articles_df)
        with sub_lotka:
            _lotka_law(author_rows)
        with sub_concentration:
            _concentration_analysis(matrix)
        with sub_trend_table:
            _productivity_trend(matrix)
        with sub_vs_impact:
            _volume_vs_impact(author_rows)


def _render_scientific_leadership_tab(articles_df: pd.DataFrame) -> None:
    st.markdown("### Corpus impact indicators")
    st.caption(
        "It compares researchers only by the articles present in this corpus. "
        "The reading crosses three metrics "
        "canonical bibliometrics: the **$h$-index** (consistency of production and citation), the **$g$-index Egghe** "
        "(that punctuates disproportionate impact articles or blockbusters), the **$e$-index of Zhang** "
        "(which measures the cumulative excess citation beyond the nucleus $h$), and the "
        "**Hirsch $m$-quotient** ($m = h / \\text{years observed in the corpus}$)."
    )

    auth_df = author_impact_advanced_indices(articles_df, min_papers=2)
    m_df = author_m_quotient_analysis(articles_df, min_papers=2)

    if auth_df.empty:
        st.info("Authors with minimum production of 2 articles not found.")
        return

    top_g = auth_df.iloc[0]
    top_m = m_df.iloc[0] if not m_df.empty else None

    metric_row(
        [
            (
                "🥇 Maior g-index",
                f"{top_g['author']} (g={top_g['g_index']})",
                f"h-index: {top_g['h_index']}",
            ),
            (
                "🔥 Maior Excesso Citacional (e)",
                f"{auth_df.sort_values(by='e_index', ascending=False).iloc[0]['author']}",
                f"e={auth_df.sort_values(by='e_index', ascending=False).iloc[0]['e_index']:.1f}",
            ),
            (
                "⚡ Maior Velocidade (m-quotient)",
                f"{top_m['author']} (m={top_m['m_quotient']})" if top_m is not None else "N/A",
                "h-index per year of career",
            ),
            ("👥 Researchers analyzed", str(len(auth_df)), "≥ 2 articles in the corpus"),
        ]
    )

    col_scatter, col_ebar = st.columns([1.1, 0.9])

    with col_scatter:
        st.markdown("### 🎯 h-index vs. g-index (Egghe) dispersion")
        max_val = max(auth_df["g_index"].max(), auth_df["h_index"].max()) + 2

        t = theme_tokens()
        ref_line = t.get("reference_line_subtle", "rgba(180, 180, 180, 0.6)")
        border_color = t.get("point_border", "#FFFFFF")

        fig_hg = go.Figure()
        fig_hg.add_trace(
            go.Scatter(
                x=[0, max_val],
                y=[0, max_val],
                mode="lines",
                name="g = h (Uniform Production)",
                line={"dash": "dash", "color": ref_line, "width": 1.5},
            )
        )
        hover_hg = [
            f"<b>{r['author']}</b><br>• g-index: {r['g_index']}<br>• h-index: {r['h_index']}<br>• Articles: {r['papers']}<br>• Citations: {r['total_citations']:,}<br>• e-index: {r['e_index']}"
            for _, r in auth_df.iterrows()
        ]
        fig_hg.add_trace(
            go.Scatter(
                x=auth_df["h_index"],
                y=auth_df["g_index"],
                mode="markers",
                marker={
                    "size": auth_df["papers"].clip(lower=6, upper=24),
                    "color": auth_df["g_h_diff"],
                    "colorscale": "Plasma",
                    "colorbar": {"title": "g - h"},
                    "opacity": 0.85,
                    "line": {"color": border_color, "width": 1},
                },
                hoverinfo="text",
                hovertext=hover_hg,
                name="Researchers",
            )
        )
        fig_hg.update_layout(
            xaxis_title="h-index (consistency)",
            yaxis_title="g-index (Egghe) - impact with blockbusters",
            height=480,
            margin={"l": 20, "r": 20, "t": 30, "b": 30},
        )
        render_chart(fig_hg)

    with col_ebar:
        st.markdown("#### 🚀 Top 12 by citation excess (Zhang e-index)")
        top_e = auth_df.sort_values(by="e_index", ascending=True).tail(12)
        fig_e = px.bar(
            top_e,
            x="e_index",
            y="author",
            orientation="h",
            color="total_citations",
            color_continuous_scale="Magma",
            labels={
                "e_index": "Zhang e-index",
                "author": "Researcher",
                "total_citations": "Total citations",
            },
        )
        fig_e.update_layout(
            xaxis_title="Zhang e-index",
            yaxis_title="Researcher",
            height=480,
            margin={"l": 20, "r": 20, "t": 30, "b": 30},
        )
        render_chart(fig_e)

    if not m_df.empty:
        st.markdown("### Top 12 in Citational Speed by Career Year (M-quotient of Hirsch)")
        top_m_plot = m_df.head(12).sort_values(by="m_quotient", ascending=True)
        fig_m = px.bar(
            top_m_plot,
            x="m_quotient",
            y="author",
            orientation="h",
            color="papers",
            color_continuous_scale="Tealgrn",
            labels={
                "m_quotient": "m-quotient (h-index / career years)",
                "author": "Researcher",
                "papers": "Articles in Corpus",
            },
        )
        fig_m.update_layout(
            xaxis_title="m-quotient (h / career years)",
            yaxis_title="Researcher",
            height=420,
            margin={"l": 20, "r": 20, "t": 30, "b": 30},
        )
        render_chart(fig_m)

    st.markdown("### 📋 Complete Ranking of Scientific Leadership")
    disp_auth = auth_df.copy()
    if not m_df.empty:
        disp_auth = disp_auth.merge(
            m_df[["author", "first_year", "career_span_years", "m_quotient"]],
            on="author",
            how="left",
        )
        disp_auth.columns = [
            "Researcher",
            "Articles",
            "Total citations",
            "Average citations",
            "h-index",
            "g-index (Egghe)",
            "e-index (Zhang)",
            "Excesso (g - h)",
            "1st Publication Year",
            "Career years",
            "m-quotient",
        ]
    else:
        disp_auth.columns = [
            "Researcher",
            "Articles",
            "Total citations",
            "Average citations",
            "h-index",
            "g-index (Egghe)",
            "e-index (Zhang)",
            "Excesso (g - h)",
        ]
    st.dataframe(disp_auth, hide_index=True, width="stretch")


def _render_team_collaboration_stats(articles_df: pd.DataFrame) -> None:
    st.subheader("👥 Size and Distribution of Authors' Teams")
    with_authors = articles_df.assign(n_authors=author_count_series(articles_df))
    with_authors = with_authors[with_authors["n_authors"] > 0]
    if with_authors.empty:
        st.info("Empty authority column in this layer.")
        return

    means = source_means(with_authors, "n_authors")
    solo_pct = float((with_authors["n_authors"] == 1).mean())

    metric_row(
        [
            (
                "📊 General Mean",
                f"{means['total']:.1f} authors/article",
                f"Mediana: {with_authors['n_authors'].median():.0f}",
            ),
            (
                "📘 Mean IEEE",
                f"{means['ieee']:.1f}" if means["ieee"] is not None else "N/D",
                "IEEE Xplore source",
            ),
            (
                "📙 Average Elsevier",
                f"{means['elsevier']:.1f}" if means["elsevier"] is not None else "N/D",
                "ScienceDirect source",
            ),
            (
                "👤 Single Author (Solo)",
                f"{solo_pct:.1%}",
                f"{len(with_authors):,} articles analyzed",
            ),
        ]
    )

    col_dist, col_trend = st.columns(2)
    with col_dist:
        dist = (
            with_authors["n_authors"]
            .value_counts()
            .sort_index()
            .rename_axis("authors")
            .reset_index(name="articles")
        )
        fig_dist = px.bar(
            dist,
            x="authors",
            y="articles",
            title="Distribution of the Number of Authors per Article",
            labels={"authors": "Authors per article", "articles": "Number of articles"},
            color_discrete_sequence=[CATEGORICAL_PALETTE[0]],
        )
        fig_dist.update_traces(hovertemplate="%{x} authors: %{y:,} articles<extra></extra>")
        fig_dist.update_layout(xaxis_title="Authors per article", yaxis_title="Number of articles")
        render_chart(
            fig_dist,
            caption="Typical asymmetries of scientific collaboration in electrical engineering.",
        )

    with col_trend:
        trend = with_authors.copy()
        trend["year"] = valid_years(trend)
        trend = trend.dropna(subset=["year"]).astype({"year": int})
        trend = trend[trend["year"] >= 2000]
        if not trend.empty:
            if "source" in trend.columns:
                by_year_auth = (
                    trend.groupby(["year", "source"])["n_authors"].mean().reset_index(name="mean")
                )
                fig_trend = px.line(
                    by_year_auth,
                    x="year",
                    y="mean",
                    color="source",
                    markers=True,
                    title="Evolution of the Average Team Size along the Years",
                    labels={"year": "Year", "mean": "Median authors", "source": "Source"},
                )
            else:
                by_year_auth = (
                    trend.groupby("year")["n_authors"].mean().rename("mean").reset_index()
                )
                fig_trend = px.line(
                    by_year_auth,
                    x="year",
                    y="mean",
                    markers=True,
                    title="Evolution of the Average Team Size along the Years",
                    labels={"year": "Year", "mean": "Median authors"},
                )
            fig_trend.update_layout(xaxis_title="Year", yaxis_title="Median authors/article")
            render_chart(
                fig_trend,
                caption="Time trend of expansion of the average size of the teams.",
            )


def _top_authors(author_rows: pd.DataFrame) -> None:
    st.subheader("")
    count_col = "doi" if "doi" in author_rows.columns else "author_display"
    agg = "nunique" if count_col == "doi" else "size"
    counts = author_rows.groupby("author_display")[count_col].agg(agg)
    top_index = counts.sort_values(ascending=False).head(15).index

    if "source" in author_rows.columns:
        scoped = author_rows[author_rows["author_display"].isin(top_index)]
        by_source = (
            scoped.groupby(["author_display", "source"])[count_col].agg(agg).unstack(fill_value=0)
        )
        for src in ("ieee", "elsevier"):
            if src not in by_source.columns:
                by_source[src] = 0
        by_source["total"] = counts.reindex(top_index)
        by_source = by_source.reindex(top_index).reset_index()
        fig = source_topn_hbar(
            by_source, "author_display", x_title="Number of articles", y_title="Autor"
        )
        fig.update_traces(hovertemplate="<b>%{y}</b><br>%{x:,} articles<extra></extra>")
    else:
        fig = topn_hbar(counts.reindex(top_index), x_title="Number of articles", y_title="Autor")
        fig.update_traces(hovertemplate="<b>%{y}</b><br>%{x:,} articles<extra></extra>")
    render_chart(fig)


def _lead_authors_ranking(author_rows: pd.DataFrame) -> None:
    st.subheader("🥇 Most frequent as first or second author")
    if "position" not in author_rows.columns:
        st.info("Author position in the publication not available in this layer.")
        return

    lead_rows = author_rows[author_rows["position"] <= 1]
    if lead_rows.empty:
        st.info("No author in 1st/2nd position identified in this layer.")
        return

    count_col = "doi" if "doi" in lead_rows.columns else "author_display"
    agg = "nunique" if count_col == "doi" else "size"
    counts = lead_rows.groupby("author_display")[count_col].agg(agg)
    top_index = counts.sort_values(ascending=False).head(15).index

    if "source" in lead_rows.columns:
        scoped = lead_rows[lead_rows["author_display"].isin(top_index)]
        by_source = (
            scoped.groupby(["author_display", "source"])[count_col].agg(agg).unstack(fill_value=0)
        )
        for src in ("ieee", "elsevier"):
            if src not in by_source.columns:
                by_source[src] = 0
        by_source["total"] = counts.reindex(top_index)
        by_source = by_source.reindex(top_index).reset_index()
        fig = source_topn_hbar(
            by_source, "author_display", x_title="Articles as 1st/2nd author", y_title="Autor"
        )
        fig.update_traces(hovertemplate="<b>%{y}</b><br>%{x:,} articles<extra></extra>")
    else:
        fig = topn_hbar(
            counts.reindex(top_index), x_title="Articles as 1st/2nd author", y_title="Autor"
        )
        fig.update_traces(hovertemplate="<b>%{y}</b><br>%{x:,} articles<extra></extra>")
    render_chart(
        fig,
        caption="Combined counting of articles in which the author appears in the 1st OR 2nd position of the list "
        "Authors, in the order registered by the source (it is not alphabetical). "
        "this -- a position; does not indicate a specific role of authorship (the convention on what 1st/2th "
        "position means varies by area, and this corpus does not record papers). "
        "Canonicization of the top of the page.",
    )


def _researchers_by_year(articles_df: pd.DataFrame) -> None:
    st.subheader("👥 Active researchers per year")
    by_year = researchers_by_year(articles_df)
    if by_year.empty:
        st.info("No valid years for this graph.")
        return

    fig = source_bars(by_year, "year", total_line=True)
    fig.update_layout(
        hovermode="x unified",
        xaxis_title="Year of publication",
        yaxis_title="Distinct researchers",
    )
    render_chart(
        fig,
        caption="Distinct canonical authors who published each year, by source. "
        "Publishing in both sources in the same year counts once in each source, but only once in the "
        "total — so the total can be smaller than IEEE + Elsevier added together.",
    )


def _cumulative_researchers_chart(articles_df: pd.DataFrame) -> None:
    st.subheader("📈 Cumulative researchers")
    cum = cumulative_researchers(articles_df)
    if cum.empty:
        st.info("No valid years for the cumulative.")
        return

    fig = source_lines(
        cum,
        "year",
        title="Different researchers cumulative per year",
        y_title="Cumulative researchers",
    )
    fig.update_layout(xaxis_title="Year of publication")
    render_chart(
        fig,
        caption=f"Each researcher is counted once, in the year of their first publication "
        f"identified in the corpus (by source and overall for Total). At the end of the period, the corpus "
        f"contains {int(cum['total'].iloc[-1]):,} distinct researchers "
        f"({int(cum['ieee'].iloc[-1]):,} IEEE, {int(cum['elsevier'].iloc[-1]):,} Elsevier).",
    )


def _lead_authors_by_year(articles_df: pd.DataFrame) -> None:
    st.subheader("🥇 1st/2nd active authors per year")
    by_year = researchers_by_year(articles_df, max_position=1)
    if by_year.empty:
        st.info("No valid years for this graph.")
        return

    fig = source_bars(by_year, "year", total_line=True)
    fig.update_layout(
        hovermode="x unified",
        xaxis_title="Year of publication",
        yaxis_title="Distinct researchers (first/second author)",
    )
    render_chart(
        fig,
        caption="Different canonical authors who appeared as 1st or 2nd author in each year, "
        'on the basis -- it is not the same universe of the graph "Active Researchers per year" above, which '
        "counts any position in the authors list.",
    )


def _cumulative_lead_authors_chart(articles_df: pd.DataFrame) -> None:
    st.subheader("📈 1st/2nd authors cumulative")
    cum = cumulative_researchers(articles_df, max_position=1)
    if cum.empty:
        st.info("No valid years for the cumulative.")
        return

    fig = source_lines(
        cum,
        "year",
        title="1st/2nd different authors cumulative per year",
        y_title="Cumulative researchers",
    )
    fig.update_layout(xaxis_title="Year of publication")
    render_chart(
        fig,
        caption=f"Each researcher is counted once, in the year of their first appearance as first "
        f"or second author. At the end of the period, the corpus contains {int(cum['total'].iloc[-1]):,} "
        f"distinct researchers in this group ({int(cum['ieee'].iloc[-1]):,} IEEE, "
        f"{int(cum['elsevier'].iloc[-1]):,} Elsevier).",
    )


def _production_heatmap(author_rows: pd.DataFrame) -> None:
    st.subheader("Production per year — top authors")
    working = author_rows.copy()
    working["year"] = valid_years(working)
    working = working.dropna(subset=["year"]).astype({"year": int})
    if working.empty:
        st.info("No valid years for heatmap.")
        return

    top_authors = (
        working.groupby("author_display")["doi"]
        .nunique()
        .sort_values(ascending=False)
        .head(TOP_AUTHORS)
        .index
        if "doi" in working.columns
        else working.groupby("author_display")
        .size()
        .sort_values(ascending=False)
        .head(TOP_AUTHORS)
        .index
    )
    scoped = working[working["author_display"].isin(top_authors)]
    pivot = scoped.groupby(["author_display", "year"]).size().unstack(fill_value=0)
    pivot = pivot.reindex(top_authors)

    t = theme_tokens()
    zero_color = t.get("heatmap_zero", "#0b1725")
    fig = px.imshow(
        pivot,
        aspect="auto",
        color_continuous_scale=[zero_color, CATEGORICAL_PALETTE[0], CATEGORICAL_PALETTE[3]],
        labels={"x": "Year of publication", "y": "Autor", "color": "Articles"},
    )
    fig.update_layout(
        xaxis_title="Year of publication",
        yaxis_title="Autor",
        height=max(420, 22 * len(pivot)),
    )
    render_chart(
        fig,
        caption="Rows with recent activity indicate active researchers; rows concentrated in earlier years "
        "old ones indicate who stopped publishing in this line of research.",
    )


def _emerging_vs_established(author_rows: pd.DataFrame) -> None:
    st.subheader("🌱 Emerging vs. established")
    working = author_rows.copy()
    working["year"] = valid_years(working)
    working = working.dropna(subset=["year"]).astype({"year": int})
    if working.empty:
        st.info("No valid years for this analysis.")
        return

    last_year = int(working["year"].max())
    by_author = working.groupby("author_display").agg(
        first_year=("year", "min"),
        total_papers=("year", "size"),
        recent_papers=("year", lambda s: (s >= last_year - RECENT_WINDOW_YEARS + 1).sum()),
    )
    by_author = by_author[by_author["total_papers"] >= 2]
    if by_author.empty:
        st.info("No authors with sufficient production for comparison.")
        return

    fig = px.scatter(
        by_author.reset_index(),
        x="first_year",
        y="recent_papers",
        size="total_papers",
        color="total_papers",
        color_continuous_scale=["#4a3aa7", "#eda100", "#1baf7a"],
        hover_name="author_display",
        labels={
            "first_year": "Year of the first publication in the corpus",
            "recent_papers": f"Articles in the last {RECENT_WINDOW_YEARS} years",
            "total_papers": "Total articles",
        },
    )
    fig.update_layout(coloraxis_showscale=False)
    render_chart(
        fig,
        caption="Right upper quadrant : recent researchers already with high production (on the rise); "
        "left lower quadrant (early debut year, few recent articles) : activity "
        "concentrated in the past in this line of research.",
    )


def _correlation_by_source(author_rows: pd.DataFrame) -> pd.DataFrame:
    """Pearson/Spearman/N for IEEE, Elsevier, and Total, as one small table."""
    sources: list[tuple[str, str | None]] = [("Total", None)]
    if "source" in author_rows.columns:
        sources = [("IEEE", "ieee"), ("Elsevier", "elsevier"), *sources]
    rows = []
    for label, key in sources:
        subset = author_rows[author_rows["source"] == key] if key else author_rows
        stats = output_impact_correlation(subset, "citation_count")
        rows.append(
            {
                "Source": label,
                "Pearson": f"{stats['pearson']:.2f}" if stats["pearson"] is not None else "—",
                "Spearman": f"{stats['spearman']:.2f}" if stats["spearman"] is not None else "—",
                "Authors": stats["n"],
            }
        )
    return pd.DataFrame(rows)


def _volume_vs_impact(author_rows: pd.DataFrame) -> None:
    st.subheader("📊 Volume × impact")
    if "citation_count" not in author_rows.columns:
        st.info("'citation_count' column not available in this layer.")
        return
    stats = output_impact_correlation(author_rows, "citation_count")
    by_author = author_rows.groupby("author_display").agg(
        articles=("author_display", "size"),
        mean_citations=("citation_count", "mean"),
        total_citations=("citation_count", "sum"),
    )
    by_author = by_author[by_author["articles"] >= 2].dropna(subset=["mean_citations"])
    if by_author.empty or stats["pearson"] is None:
        st.info("No sufficient citation data for authors with ≥2 articles.")
        return

    st.dataframe(_correlation_by_source(author_rows), hide_index=True, width="stretch")
    fig = px.scatter(
        by_author.reset_index(),
        x="articles",
        y="mean_citations",
        size="total_citations",
        color="mean_citations",
        color_continuous_scale=["#4a3aa7", "#eda100", "#1baf7a"],
        hover_name="author_display",
        title=f"Volume × impact (Pearson r = {stats['pearson']:.2f})",
        labels={
            "articles": "Articles in the corpus",
            "mean_citations": "Average citations per article",
        },
    )
    fig.update_layout(coloraxis_showscale=False)
    render_chart(
        fig,
        caption="`citation_count` null is treated as 'not collected' and excluded from the average — not as zero. "
        "The bubble size is the total number of citations cumulative by the author. "
        "included because it is more robust to long tail distributions (a few authors with production or "
        "citations far above average), common in bibliometric data.",
    )


def _lotka_law(author_rows: pd.DataFrame) -> None:
    """The section is named for the bibliometric laws; this is the author one.

    Lotka's law predicts how many authors publish x papers. It belongs on this
    page rather than with Bradford/Zipf because its unit is the author, and it
    inherits the same heuristic-identity caveat as every other person-level view.
    """
    from lake_research_map.dashboard.analytics import lotka_law_analysis

    st.subheader("📐 Lotka's law of author productivity")
    if author_rows.empty or "author_key" not in author_rows.columns:
        st.info("Author productivity is not available for this population.")
        return

    # `author_rows` is one row per author-article pair, so productivity is the
    # count of DISTINCT DOIs per canonical key -- matching the page's own
    # per-author counts rather than counting a co-authored paper twice.
    per_author = (
        author_rows.groupby("author_key")["doi"].nunique()
        if "doi" in author_rows.columns
        else author_rows.groupby("author_key").size()
    )
    result = lotka_law_analysis(per_author)
    table = result["table"]
    if table.empty:
        st.info("Too few distinct productivity levels to fit Lotka's law.")
        return

    metric_row(
        [
            (
                "⚡ Exponent (α)",
                f"{result['alpha']:.2f}",
                "Lotka's inverse-square case is α = 2",
            ),
            ("📊 Fit R²", f"{result['r2']:.3f}", "Weighted log-log regression"),
            (
                "👥 Authors covered",
                f"{int(per_author.gt(0).sum()):,}",
                f"{len(table)} distinct productivity levels",
            ),
        ]
    )

    display = table.rename(
        columns={
            "papers_x": "Articles published",
            "empirical_authors": "Authors (observed)",
            "theoretical_authors": "Authors (Lotka)",
            "empirical_share": "Observed share",
            "theoretical_share": "Lotka share",
        }
    )
    st.dataframe(display, hide_index=True, width="stretch")
    st.caption(
        "α near 2 reproduces the classic inverse-square pattern: most authors contribute "
        "one article and a few contribute many. The fit is descriptive and corpus-scoped — "
        "it counts only articles present in this collection, so a prolific researcher indexed "
        "mostly elsewhere appears here as an occasional author. Author names are canonicalised "
        "heuristically, which both merges homonyms and splits spelling variants."
    )


def _full_output_table(articles_df: pd.DataFrame) -> pd.DataFrame:
    st.subheader("📋 Complete production by author and year")
    matrix = loaders.author_year_matrix_cached(loaders.filter_signature())
    if matrix.empty:
        st.info("No valid years to set up the table.")
        return matrix

    st.caption(
        "One line per canonical author (see warning at the top of the page), one column per year "
        "valid publication, plus `total`, `ieee_total` and `elsevier_total` (break of historical total "
        "by source). Distinct DOI counting is used when available to avoid counting an article twice "
        "co-authored and signed by the same author. **Sorted from highest to lowest total.**"
    )
    st.dataframe(matrix, hide_index=True, width="stretch")
    st.download_button(
        "▁Baixar",
        data=matrix.to_csv(index=False).encode("utf-8"),
        file_name="producao_por_autor_ano.csv",
        mime="text/csv",
        key="dl_author_year_matrix",
    )
    return matrix


def _gini_interpretation(gini: float) -> str:
    if gini < 0.3:
        return "low concentration — production is relatively distributed among authors"
    if gini > 0.6:
        return "High concentration — production is dominated by few very prolific authors"
    return "moderate concentration"


def _concentration_analysis(matrix: pd.DataFrame) -> None:
    st.subheader("📐 Production concentration (Gini / Lorenz curve)")
    if matrix.empty:
        st.info("No sufficient data for this analysis.")
        return

    gini_ieee = gini_coefficient(matrix["ieee_total"])
    gini_elsevier = gini_coefficient(matrix["elsevier_total"])
    gini_total = gini_coefficient(matrix["total"])
    metric_row(
        [
            ("📐 Gini — IEEE", f"{gini_ieee:.2f}", _gini_interpretation(gini_ieee)),
            ("📐 Gini — Elsevier", f"{gini_elsevier:.2f}", _gini_interpretation(gini_elsevier)),
            ("📐 Gini — Total", f"{gini_total:.2f}", _gini_interpretation(gini_total)),
        ]
    )

    fig = lorenz_chart(
        {
            "ieee": lorenz_curve(matrix["ieee_total"]),
            "elsevier": lorenz_curve(matrix["elsevier_total"]),
            "total": lorenz_curve(matrix["total"]),
        }
    )
    render_chart(
        fig,
        caption="Gini index calculated on the historical total per author, separated by source (0 = "
        "All publish the same, 1 = a single author concentrates all the production). "
        "observed curve is away from the diagonal of perfect equity, more concentrated is production "
        "in that source.",
    )


def _productivity_trend(matrix: pd.DataFrame) -> None:
    st.subheader("📈 Productivity trend — top authors")
    if matrix.empty:
        st.info("No sufficient data for this analysis.")
        return

    trend_df = author_productivity_trend(matrix, top_n=TOP_AUTHORS)
    st.caption(
        f"Least-squares trend line (`numpy.polyfit`, degree 1) for articles per year for each "
        f"of the {TOP_AUTHORS} most prolific authors, using only years in which the author "
        "published. "
        "Series of this size (less active years) are not "
        "support a reliable statistical significance test — it is a directional indicator, not "
        "a prediction."
    )
    display = trend_df.rename(
        columns={
            "author": "Autor",
            "total": "Historical total",
            "first_year": "First year",
            "last_year": "Last year",
            "active_years": "Active years",
            "slope": "Inclination (articles/year)",
            "trend": "Trend",
        }
    )
    st.dataframe(display, hide_index=True, width="stretch")


def _network_null_model(graph, net_metrics: dict) -> None:
    """Compare observed clustering against degree-preserving rewirings.

    A co-authorship graph is clustered simply because papers have several
    authors, so a raw clustering coefficient says little on its own. The null
    keeps every author's degree and rewires the ties, which separates real
    collaborative structure from an artefact of corpus size and team sizes.
    """
    from lake_research_map.dashboard.analytics import network_null_model_diagnostics

    result = network_null_model_diagnostics(graph)
    if not result.get("valid"):
        return

    with st.expander("Structure vs. degree-preserving null model", expanded=False):
        p_value = result["empirical_p_value"]
        metric_row(
            [
                (
                    "🕸️ Observed clustering",
                    f"{result['observed_clustering']:.3f}",
                    f"null mean {result['null_mean']:.3f} ± {result['null_std']:.3f}",
                ),
                (
                    "📐 Z-score",
                    f"{result['z_score']:+.2f}",
                    f"{result['simulations']} degree-preserving rewirings",
                ),
                (
                    "🧪 Empirical p-value",
                    f"{p_value:.3f}",
                    "Clustering exceeds chance"
                    if p_value < 0.05
                    else "Not distinguishable from the null",
                ),
            ]
        )
        assortativity = result.get("observed_assortativity")
        assort_z = result.get("assortativity_z_score")
        targeted = result.get("robustness_targeted")
        random_removal = result.get("robustness_random")
        if assortativity is not None or targeted is not None:
            metric_row(
                [
                    (
                        "🔗 Degree assortativity",
                        "n/a" if assortativity is None else f"{assortativity:+.3f}",
                        "n/a" if assort_z is None else f"z {assort_z:+.2f} vs. the same rewirings",
                    ),
                    (
                        "💥 Giant component, hubs removed",
                        "n/a" if targeted is None else f"{targeted:.0%}",
                        f"{result.get('robustness_removed', 0)} highest-degree authors",
                    ),
                    (
                        "🎲 Giant component, random removal",
                        "n/a" if random_removal is None else f"{random_removal:.0%}",
                        "Same count removed at random, averaged over 20 draws",
                    ),
                ]
            )
        st.caption(
            "The null preserves each author's degree and rewires the ties, so a high z-score "
            "means the clustering is not just a consequence of how many co-authors each "
            "person has. Assortativity is scored against those same rewirings, because the "
            "degree sequence alone forces part of it. The two removal figures are only "
            "meaningful as a pair: a network that fragments when its hubs go but shrugs off "
            "the same number of random losses is one held together by a few people, while "
            "two similar numbers mean the structure is distributed. Path length and the "
            "small-world σ above are restricted to the largest connected component, while "
            "density and the centralities cover the whole graph including fragments — the "
            "two are not on the same population. Every claim here is also bounded by "
            "heuristic author identity: homonyms merge and spelling variants split."
        )


def _periodized_ties(author_rows: pd.DataFrame) -> None:
    """Show whether the field keeps recruiting collaborators or has closed up.

    A static recurrent-edge count cannot tell those apart: both produce the
    same number of repeat pairs. Splitting by the period in which a tie first
    appears is what separates them.
    """
    from lake_research_map.dashboard.analytics import periodized_collaboration_ties

    ties = periodized_collaboration_ties(author_rows)
    if ties.empty or len(ties) < 2:
        return

    with st.expander("New vs. returning collaborations over time", expanded=False):
        shown = ties.rename(
            columns={
                "period": "Period",
                "new_ties": "New ties",
                "repeated_ties": "Returning ties",
                "total_ties": "Active ties",
                "new_share": "New share",
            }
        )
        shown["New share"] = (shown["New share"] * 100).round(1)
        st.dataframe(shown, hide_index=True, width="stretch")
        first, last = ties.iloc[0]["new_share"], ties.iloc[-1]["new_share"]
        direction = (
            "the field is still recruiting new collaborators"
            if last >= first
            else "collaboration is consolidating into established pairs"
        )
        st.caption(
            "A tie is *new* in the period containing its first-ever collaboration and "
            "*returning* thereafter. The share of new ties moved from "
            f"{first:.0%} to {last:.0%} across the periods shown, so on this corpus "
            f"{direction}. Periods are equal splits of the observed years, not calendar "
            "decades, and every count inherits the limits of heuristic author identity."
        )


def _coauthorship_network(author_rows: pd.DataFrame) -> None:
    st.subheader("Co-authorship network (top authors)")
    if "doi" not in author_rows.columns:
        st.info("Column 'doi' not available to reconstruct the network.")
        return

    counts = author_rows.groupby("author_display")["doi"].nunique()
    # A force-directed layout was tried here and, even after several rounds of
    # tuning, still put too many authors too close together to read at 60
    # nodes. A much smaller, fixed circular layout trades "shows everyone" for
    # "every connection is actually legible" -- no physics, no randomness, no
    # possible overlap (evenly spaced points on a circle can't collide).
    top_authors = (
        counts[counts >= MIN_PAPERS_FOR_NETWORK]
        .sort_values(ascending=False)
        .head(TOP_NETWORK_AUTHORS)
        .index
    )
    if len(top_authors) < 3:
        st.info(
            f"Too few authors with ≥{MIN_PAPERS_FOR_NETWORK} articles to build a readable network."
        )
        return

    scoped = author_rows[author_rows["author_display"].isin(top_authors)]
    by_doi = scoped.groupby("doi")["author_display"].apply(list)

    graph = nx.Graph()
    graph.add_nodes_from(top_authors)
    for authors in by_doi:
        unique_authors = sorted(set(authors))
        for i in range(len(unique_authors)):
            for j in range(i + 1, len(unique_authors)):
                a, b = unique_authors[i], unique_authors[j]
                if graph.has_edge(a, b):
                    graph[a][b]["weight"] += 1
                else:
                    graph.add_edge(a, b, weight=1)

    # Authors in the top-N by volume with no coauthor also in the top-N add
    # nothing to a *network* view -- drop them rather than scatter meaningless
    # isolated dots around the circle.
    graph.remove_nodes_from(list(nx.isolates(graph)))
    if graph.number_of_edges() == 0:
        st.info("No co-author found among the most productive authors.")
        return

    n = graph.number_of_nodes()
    # Order nodes by a depth-first walk of the graph (starting from the most
    # connected author) rather than alphabetically or by rank -- neighbors in
    # the walk tend to be actual collaborators, so placing them next to each
    # other around the circle keeps most edges short instead of criss-crossing
    # the whole diagram. Any node a DFS from one root can't reach (a separate
    # component) is appended afterwards.
    root = max(graph.degree, key=lambda kv: kv[1])[0]
    order = list(nx.dfs_preorder_nodes(graph, source=root))
    order += [node for node in graph.nodes() if node not in order]

    radius = 9.0
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    pos = {
        node: np.array([radius * np.cos(a), radius * np.sin(a)])
        for node, a in zip(order, angles, strict=True)
    }

    degree = dict(graph.degree())
    nodes = list(graph.nodes())

    communities = coauthorship_community_detection(graph)
    net_metrics = graph_advanced_metrics(graph)
    n_comms = len(set(communities.values())) if communities else 1

    partners_df = analyze_coauthorship_partners(graph, author_rows, communities)
    partners_by_author = (
        partners_df.set_index("author").to_dict(orient="index") if not partners_df.empty else {}
    )

    recurrent_edges = [
        (u, v, int(d.get("weight", 1)))
        for u, v, d in graph.edges(data=True)
        if d.get("weight", 1) >= 2
    ]
    n_recurrent = len(recurrent_edges)
    top_pair = max(recurrent_edges, key=lambda x: x[2]) if recurrent_edges else None
    top_pair_note = (
        f"Mais forte: {top_pair[0]} & {top_pair[1]} ({top_pair[2]} arts)"
        if top_pair
        else "None with ≥2 articles"
    )

    sw_value = (
        f"{net_metrics['small_world_sigma']:.2f}" if net_metrics.get("small_world_sigma") else "—"
    )
    sw_note = (
        f"L={net_metrics['avg_path_length']:.2f} (Topologia Small-World)"
        if net_metrics.get("small_world_sigma") and net_metrics["small_world_sigma"] > 1.0
        else f"Densidade: {net_metrics['density']:.3f}"
    )

    metric_row(
        [
            ("👥 Connected researchers", f"{n}", f"{n_comms} Louvain communities"),
            (
                "🔗 Single Connections (pairs)",
                f"{graph.number_of_edges()}",
                f"{graph.number_of_edges() - n_recurrent} ocasionais (1 art.)",
            ),
            ("🔁 Recurring partnerships (≥2 articles)", f"{n_recurrent}", top_pair_note),
            ("🌐 Coef. Pequeno Mundo (σ)", sw_value, sw_note),
        ]
    )

    # Edges as separate line segments:
    # Single-article edges are rendered subtly; recurrent partnerships (>= 2 papers)
    # are rendered with higher opacity and thickness to stand out.
    edge_traces = []
    max_w = max((d["weight"] for _, _, d in graph.edges(data=True)), default=1)

    # 1. Single-article connections (w == 1)
    for a, b, d in graph.edges(data=True):
        if d.get("weight", 1) == 1:
            x0, y0 = pos[a]
            x1, y1 = pos[b]
            edge_traces.append(
                go.Scatter(
                    x=[x0, x1],
                    y=[y0, y1],
                    mode="lines",
                    line=dict(
                        color="rgba(94,169,255,0.20)",
                        width=1.2,
                    ),
                    hoverinfo="skip",
                    showlegend=False,
                )
            )

    # 2. Recurrent connections (w >= 2)
    for a, b, d in graph.edges(data=True):
        w = d.get("weight", 1)
        if w >= 2:
            x0, y0 = pos[a]
            x1, y1 = pos[b]
            edge_traces.append(
                go.Scatter(
                    x=[x0, x1],
                    y=[y0, y1],
                    mode="lines",
                    line=dict(
                        color="rgba(235,104,52,0.80)",
                        width=2.5 + 4 * (w / max_w),
                    ),
                    hoverinfo="skip",
                    showlegend=False,
                )
            )

    # 3. Interactive midpoints on recurrent edges to show partner details on hover
    if recurrent_edges:
        mid_x = [(pos[a][0] + pos[b][0]) / 2 for a, b, _ in recurrent_edges]
        mid_y = [(pos[a][1] + pos[b][1]) / 2 for a, b, _ in recurrent_edges]
        recurrent_custom = [[a, b, w] for a, b, w in recurrent_edges]
        edge_traces.append(
            go.Scatter(
                x=mid_x,
                y=mid_y,
                mode="markers",
                marker=dict(
                    size=7,
                    color="#eb6834",
                    symbol="diamond",
                    line=dict(width=1, color="white"),
                ),
                customdata=recurrent_custom,
                hovertemplate=(
                    "🔁 <b>Recurring partnership</b><br> "
                    "👥 %{customdata[0]} ↔ %{customdata[1]}<br> "
                    "📚 <b>%{customdata[2]} articles</b> in co-authorship in the network "
                    "<extra></extra>"
                ),
                name="Recurring partnerships",
                showlegend=False,
            )
        )

    node_colors = [
        CATEGORICAL_PALETTE[communities.get(a, 0) % len(CATEGORICAL_PALETTE)] for a in nodes
    ]

    customdata = []
    for a in nodes:
        p = partners_by_author.get(a, {})
        customdata.append(
            [
                p.get("community", f"#{communities.get(a, 0) + 1}"),
                p.get("articles", 0),
                p.get("network_unique_count", degree[a]),
                p.get("network_recurrent_count", 0),
                p.get("network_recurrent_names", "—"),
                p.get("global_unique_count", 0),
                p.get("global_recurrent_count", 0),
                p.get("global_top_partners", "—"),
            ]
        )

    fig = go.Figure(data=edge_traces)
    fig.add_trace(
        go.Scatter(
            x=[pos[a][0] for a in nodes],
            y=[pos[a][1] for a in nodes],
            mode="markers+text",
            text=nodes,
            textposition="top center",
            textfont=dict(size=10, color=theme_tokens()["chart_text"]),
            marker=dict(
                size=[10 + 4 * degree[a] for a in nodes],
                color=node_colors,
                line=dict(width=1.5, color="rgba(255,255,255,0.4)"),
            ),
            customdata=customdata,
            hovertemplate=(
                "<b>%{text}</b><br> "
                "🏘️ Louvain community: <b>%{customdata[0]}</b><br> "
                "📄 Total articles in the corpus: <b>%{customdata[1]}</b><br> "
                "<br> "
                "🕸️ <b>In the top-author network:</b><br> "
                "• Single co-authors: <b>%{customdata[2]}</b><br> "
                "• Repeated partnerships (≥2 articles): <b>%{customdata[3]}</b><br> "
                "• Partners in the network: %{customdata[4]}<br> "
                "<br> "
                "🌐 <b>No Corpus Global:</b><br> "
                "• Single co-authors: <b>%{customdata[5]}</b><br> "
                "• Repeated partnerships: <b>%{customdata[6]}</b><br> "
                "• Top partners in the corpus: %{customdata[7]} "
                "<extra></extra>"
            ),
            showlegend=False,
        )
    )
    # The layout is a fixed circle: node position encodes nothing, so ticks and
    # gridlines are hidden (`scaleanchor` keeps the circle round). Axis names
    # state this explicitly instead of appearing anonymous.
    fig.update_layout(
        xaxis=dict(
            title="Position in the circular layout (no unit)",
            showticklabels=False,
            showgrid=False,
            zeroline=False,
            ticks="",
            scaleanchor="y",
            scaleratio=1,
        ),
        yaxis=dict(
            title="Position in the circular layout (no unit)",
            showticklabels=False,
            showgrid=False,
            zeroline=False,
            ticks="",
        ),
        showlegend=False,
        height=650,
        hovermode="closest",
    )
    render_chart(
        fig,
        caption=f"Fixed circular layout (no force simulation): the {n} authors with at least one coauthorship among the "
        f"{TOP_NETWORK_AUTHORS} most productive authors (≥{MIN_PAPERS_FOR_NETWORK} articles) are equally "
        "spaced around the circle — the position does not indicate proximity, and the order follows a walk "
        "by graph from the most connected author, to keep most connections as short lines "
        "instead of crossing the whole drawing. Orange lines with a central diamond mark "
        "**recurring partnerships** (≥2 articles), while thin blue lines represent one-off "
        "co-authorships (1 article). Node size reflects the number of distinct co-authors.",
    )

    _network_null_model(graph, net_metrics)
    _periodized_ties(author_rows)

    if not partners_df.empty:
        st.divider()
        st.markdown("#### 👥 Connections with Unique Authors vs. Recurrents (Top Authors)")
        st.caption(
            "Quantitative and nominal detailing of scientific collaborations. "
            "The section **In the Network** restricts the analysis to the top authors represented in the graph above; "
            "The section **In the Global Corpus** covers the totality of articles and collaborators registered in the database."
        )
        pr_map = net_metrics.get("pagerank", {})
        close_map = net_metrics.get("closeness", {})
        partners_df["pagerank"] = (
            partners_df["author"].map(pr_map).fillna(0.0).apply(lambda x: f"{x:.4f}")
        )
        partners_df["closeness"] = (
            partners_df["author"].map(close_map).fillna(0.0).apply(lambda x: f"{x:.3f}")
        )

        display_df = partners_df[
            [
                "author",
                "articles",
                "community",
                "network_unique_count",
                "network_recurrent_count",
                "network_recurrent_names",
                "pagerank",
                "closeness",
                "global_unique_count",
                "global_recurrent_count",
                "global_top_partners",
            ]
        ].rename(
            columns={
                "author": "Researcher",
                "articles": "Total Articles",
                "community": "Community",
                "network_unique_count": "Single Co-authors (Network)",
                "network_recurrent_count": "Repeated partnerships (network ≥2)",
                "network_recurrent_names": "Who are the partners (Network)",
                "pagerank": "PageRank",
                "closeness": "Closeness",
                "global_unique_count": "Single Co-authors (Global)",
                "global_recurrent_count": "Repeated partnerships (global ≥2)",
                "global_top_partners": "Top partners in the corpus",
            }
        )
        st.dataframe(display_df, hide_index=True, width="stretch")


def _cognitive_distance_analysis(articles_df: pd.DataFrame, author_rows: pd.DataFrame) -> None:
    st.subheader("🧠 Cognitive distance in co-authorships vs. citation impact")
    st.caption(
        "The cognitive distance measures the conceptual dispersion among the co-authors of an article "
        "allows empirically testing if interdisciplinary partnerships "
        "reach greater scientific repercussion."
    )
    signals = loaders.semantics()
    if signals.empty or "map_x" not in signals.columns or "map_y" not in signals.columns:
        st.info(
            "Semantic signals are unavailable for cognitive-distance calculation. Run `--stage semantic`."
        )
        return

    scoped = loaders.with_semantics(articles_df)
    valid_articles = scoped.dropna(subset=["map_x", "map_y", "citation_count"]).copy()
    if len(valid_articles) < 10:
        st.info("Insufficient articles with data sets of semantics and citations.")
        return

    merged_author_art = author_rows.merge(valid_articles[["doi", "map_x", "map_y"]], on="doi")
    author_pos = merged_author_art.groupby("author_key")[["map_x", "map_y"]].mean()

    records = []
    for doi, grp in merged_author_art.groupby("doi"):
        authors = grp["author_key"].unique()
        if len(authors) >= 2:
            matched = author_pos.index.intersection(authors)
            if len(matched) >= 2:
                coords = author_pos.loc[matched].to_numpy()
                diffs = coords[:, None, :] - coords[None, :, :]
                dists = np.sqrt(np.sum(diffs**2, axis=-1))
                i_upper = np.triu_indices(len(coords), k=1)
                mean_dist = float(np.mean(dists[i_upper]))
                row = valid_articles[valid_articles["doi"] == doi].iloc[0]
                records.append(
                    {
                        "doi": doi,
                        "title": str(row.get("title", "—"))[:80],
                        "year": row.get("year"),
                        "cognitive_distance": round(mean_dist, 3),
                        "citation_count": int(row.get("citation_count", 0)),
                        "team_size": len(authors),
                    }
                )

    if len(records) < 5:
        st.info("Few articles with multiple authors positioned in the semantic space.")
        return

    dist_df = pd.DataFrame(records)
    r_pearson = float(
        dist_df["cognitive_distance"].corr(dist_df["citation_count"], method="pearson")
    )
    r_spearman = float(
        dist_df["cognitive_distance"].corr(dist_df["citation_count"], method="spearman")
    )

    metric_row(
        [
            ("👥 Multi-authorship Articles Evaluated", f"{len(dist_df):,}", None),
            (
                "📐 Mean cognitive distance",
                f"{dist_df['cognitive_distance'].mean():.2f}",
                None,
            ),
            (
                "📈 Pearson's correlation (r)",
                f"{r_pearson:+.3f}",
                "Linear relationship",
            ),
            (
                "📊 Spearman correlation (ρ)",
                f"{r_spearman:+.3f}",
                "Monotonic relationship",
            ),
        ]
    )

    fig = px.scatter(
        dist_df,
        x="cognitive_distance",
        y="citation_count",
        size="team_size",
        hover_name="title",
        hover_data={
            "cognitive_distance": True,
            "citation_count": True,
            "team_size": True,
            "year": True,
        },
        title="Cognitive distance between co-authors vs. citations received",
        labels={
            "cognitive_distance": "Cognitive Distance of the Team (Semantic Dispersion)",
            "citation_count": "Citations Received",
            "team_size": "Authors",
        },
        color_discrete_sequence=[CATEGORICAL_PALETTE[0]],
        opacity=0.7,
    )
    render_chart(
        fig,
        caption="Each point represents a co-authorship article. "
        "The degree of conceptual complementarity between the research histories of its authors.",
    )


def _research_line_selector(
    author_rows: pd.DataFrame, articles_df: pd.DataFrame
) -> tuple[str | None, pd.DataFrame, set]:
    """Returns `(selected_keyword, scoped_authors, dois_with_kw)`; `selected_keyword` is None when there's nothing to show (missing data or no valid selection)."""
    st.subheader("🔎 Who leads this line of research")
    kw_exploded = explode_keywords(articles_df)
    if kw_exploded.empty:
        st.info("Keywords column not available in this layer.")
        return None, author_rows.iloc[0:0], set()

    top_keywords = kw_exploded["keyword"].value_counts().head(60).index.tolist()
    selected = st.selectbox("Select a keyword:", options=top_keywords)
    if not selected:
        return None, author_rows.iloc[0:0], set()

    dois_with_kw = set(kw_exploded.loc[kw_exploded["keyword"] == selected, "doi"].dropna())
    scoped_authors = (
        author_rows[author_rows["doi"].isin(dois_with_kw)]
        if "doi" in author_rows.columns
        else author_rows.iloc[0:0]
    )
    if scoped_authors.empty:
        st.info("No author associated with this term in this layer.")
        return None, scoped_authors, dois_with_kw

    return selected, scoped_authors, dois_with_kw


def _research_line_top_authors(selected: str, scoped_authors: pd.DataFrame) -> None:
    leaders = (
        scoped_authors.groupby("author_display")["doi"]
        .nunique()
        .sort_values(ascending=False)
        .head(10)
    )
    fig = topn_hbar(
        leaders,
        title=f"Most productive authors in '{selected}'",
        x_title="Number of articles",
        y_title="Autor",
    )
    render_chart(fig)


def _research_line_trend(selected: str, scoped_authors: pd.DataFrame) -> None:
    trend_df = scoped_authors.copy()
    trend_df["year"] = valid_years(trend_df)
    trend_df = trend_df.dropna(subset=["year"]).astype({"year": int})
    if trend_df.empty:
        st.info("No valid years for the trajectory.")
        return

    by_year = trend_df.groupby("year")["doi"].nunique().reset_index(name="articles")
    fig = px.line(
        by_year,
        x="year",
        y="articles",
        markers=True,
        title=f"Annual trajectory of '{selected}'",
        labels={"year": "Year", "articles": "Articles"},
    )
    fig.update_traces(line_color=CATEGORICAL_PALETTE[2])
    render_chart(fig)


def _research_line_researchers_by_year(
    selected: str, articles_df: pd.DataFrame, dois_with_kw: set
) -> None:
    keyword_articles = articles_df[articles_df["doi"].isin(dois_with_kw)]
    kw_by_year = researchers_by_year(keyword_articles)
    if kw_by_year.empty:
        st.info("No valid years for this graph.")
        return

    fig = source_bars(kw_by_year, "year", total_line=True)
    fig.update_layout(
        hovermode="x unified",
        xaxis_title="Year of publication",
        yaxis_title="Distinct researchers",
    )
    render_chart(
        fig,
        caption=f"Distinct researchers who published in '{selected}' each year, by source.",
    )


def _research_line_researchers_cumulative(
    selected: str, articles_df: pd.DataFrame, dois_with_kw: set
) -> None:
    keyword_articles = articles_df[articles_df["doi"].isin(dois_with_kw)]
    kw_cum = cumulative_researchers(keyword_articles)
    if kw_cum.empty:
        st.info("No valid years for the cumulative.")
        return

    fig = source_lines(
        kw_cum,
        "year",
        title=f"Cumulative researchers in '{selected}'",
        y_title="Cumulative researchers",
    )
    fig.update_layout(xaxis_title="Year of publication")
    render_chart(
        fig,
        caption="Cumulative total of different researchers who have already published in Portugal"
        f"'{selected}' through each year ({int(kw_cum['total'].iloc[-1]):,} at the end of the period).",
    )


def _research_line_articles(
    articles_df: pd.DataFrame, scoped_authors: pd.DataFrame, selected: str
) -> None:
    top_dois = scoped_authors["doi"].unique()
    subset = articles_df[articles_df["doi"].isin(top_dois)]
    if "citation_count" in subset.columns:
        subset = subset.sort_values("citation_count", ascending=False, na_position="last")
    article_table(
        subset,
        ["title", "year", "venue", "source", "citation_count", "doi"],
        download_key=f"lideres_{selected.replace(' ', '_')}",
    )


def _author_keyword_selector(author_rows: pd.DataFrame) -> str | None:
    """Shared author selector for the "Keyword Profile"/"Focus Change" sub-tabs."""
    st.subheader("🏷️ Keyword profile by author")
    counts = (
        author_rows.groupby("author_display")["doi"].nunique()
        if "doi" in author_rows.columns
        else author_rows.groupby("author_display").size()
    )
    eligible = counts[counts >= 3].sort_values(ascending=False)
    if eligible.empty:
        st.info("No author with at least 3 articles in this layer.")
        return None

    selected_author = st.selectbox(
        "Select an author (minimum 3 articles):", options=eligible.index.tolist()
    )
    return selected_author or None


def _author_keyword_working(
    selected_author: str, author_rows: pd.DataFrame, articles_df: pd.DataFrame
) -> pd.DataFrame | None:
    author_dois = set(
        author_rows.loc[author_rows["author_display"] == selected_author, "doi"].dropna()
    )
    subset = articles_df[articles_df["doi"].isin(author_dois)]
    kw_exploded = explode_keywords(subset)
    if kw_exploded.empty:
        st.info(f"No keyword recorded for {selected_author}.")
        return None

    working = kw_exploded.copy()
    working["year"] = valid_years(working)
    return working.dropna(subset=["year"]).astype({"year": int})


def _author_keyword_overview(selected_author: str, working: pd.DataFrame) -> None:
    top_terms = working["keyword"].value_counts().head(10)
    fig = topn_hbar(
        top_terms,
        title=f"Dominant keywords for {selected_author}",
        x_title="Number of mentions",
        y_title="Keyword",
    )
    render_chart(fig)


def _author_keyword_shift(working: pd.DataFrame) -> None:
    if working["year"].nunique() < 2:
        st.info("Insufficient years to compare beginning vs. end of career in the corpus.")
        return

    split_year = int(working["year"].median())
    early = working[working["year"] <= split_year]["keyword"].value_counts()
    late = working[working["year"] > split_year]["keyword"].value_counts()
    all_terms = set(early.index) | set(late.index)
    compare = pd.DataFrame(
        {
            "early": early.reindex(all_terms, fill_value=0),
            "late": late.reindex(all_terms, fill_value=0),
        }
    )
    compare = (
        compare[(compare["early"] + compare["late"]) > 0]
        .sort_values("late", ascending=False)
        .head(10)
    )
    fig = go.Figure()
    fig.add_bar(
        x=compare.index,
        y=compare["early"],
        name=f"through {split_year}",
        marker_color=CATEGORICAL_PALETTE[0],
    )
    fig.add_bar(
        x=compare.index,
        y=compare["late"],
        name=f"after {split_year}",
        marker_color=CATEGORICAL_PALETTE[2],
    )
    fig.update_layout(
        barmode="group",
        title="Focus change: beginning vs. end of career in the corpus",
        xaxis_title="Keyword",
        yaxis_title="Mentions in the period",
    )
    render_chart(fig)
