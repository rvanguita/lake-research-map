"""It maps the taxonomy of mathematical optimization methods and complexity spectrum (MILP, SOCP, MINLP, IA), objective functions and multi-criteria formulations (Pareto), uncertainty modeling paradigms (stochastic, robust, fuzzy, DRO), time horizons (dynamic expansion vs. representative days), validation in electrical test systems (IEEE 33, 69, 123-bus and real networks), computational solvers toolling (GAMS, CPLEX, MATLAB and OpenDSS)."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from lake_research_map.dashboard import loaders
from lake_research_map.dashboard.analytics import (
    benchmark_feeders_analysis,
    computational_solvers_analysis,
    mathematical_complexity_spectrum,
    objective_functions_taxonomy,
    optimization_methods_taxonomy,
    planning_time_horizons_analysis,
    uncertainty_paradigms_analysis,
)
from lake_research_map.dashboard.components import (
    page_header,
    render_chart,
    summary_card_row,
)
from lake_research_map.dashboard.theme import CATEGORICAL_PALETTE


def render() -> None:
    page_header(
        "🔬",
        "Engineering evidence",
        "Mathematical formulation, objective functions, uncertainty, temporal horizons, IEEE validation and solvers.",
    )

    articles_df = loaders.require_articles()
    df = loaders.with_semantics(articles_df)

    section = st.selectbox(
        "Evidence dimension",
        [
            "Methods and complexity",
            "Objective functions",
            "Modelagem da incerteza",
            "Horizontes temporais",
            "Benchmarks IEEE e redes",
            "Ferramental e solvers",
        ],
    )

    if section == "Methods and complexity":
        _render_methods_tab(df)
    elif section == "Objective functions":
        _render_objectives_tab(df)
    elif section == "Modelagem da incerteza":
        _render_uncertainty_tab(df)
    elif section == "Horizontes temporais":
        _render_horizons_tab(df)
    elif section == "Benchmarks IEEE e redes":
        _render_feeders_tab(df)
    elif section == "Ferramental e solvers":
        _render_solvers_tab(df)


# ---------------------------------------------------------------------------
# Tab 1: Methods & mathematical complexity
# ---------------------------------------------------------------------------


def _render_methods_tab(df: pd.DataFrame) -> None:
    st.markdown("##⚙ Ta Taxonomy of Paradigms of Optimization & Mathematical Complexity")
    st.caption(
        "It identifies and quantifies the mathematical methods used in the distribution planning literature. "
        "Evidences the historical transition from classical metaheuristics to exact programming (MILP, SOCP), "
        "optimization under uncertainty (Robusta, Stochastic) and artificial intelligence."
    )

    res = optimization_methods_taxonomy(df)
    summary_df = res["summary_df"]
    temporal_df = res["temporal_df"]

    if summary_df.empty:
        st.info("No optimization method detected in the corpus.")
        return

    spec_res = mathematical_complexity_spectrum(df)
    spectrum_df = spec_res["spectrum_df"]
    temp_spec = spec_res["temporal_spectrum"]

    top_method = summary_df.iloc[0]["method"]
    top_cites_method = summary_df.sort_values(by="mean_citations", ascending=False).iloc[0][
        "method"
    ]
    top_recent_method = summary_df.sort_values(by="pct_recent", ascending=False).iloc[0]["method"]

    summary_card_row(
        [
            ("🏆 Most frequent method", top_method, f"{summary_df.iloc[0]['articles']} articles"),
            ("🚀 Maior Momentum Recente", top_recent_method, "Higher proportion ≥ 2021"),
            ("💡 Highest citation impact", top_cites_method, "Average citations/article"),
            ("📐 Paradigms mapped", str(len(summary_df)), "Resolution families"),
        ]
    )

    col_bar, col_line = st.columns([1, 1])

    with col_bar:
        st.markdown("### 📊 Total Volume by Method")
        bar_data = summary_df.sort_values(by="articles", ascending=True)
        fig_bar = px.bar(
            bar_data,
            x="articles",
            y="method",
            orientation="h",
            color="pct_recent",
            color_continuous_scale="Viridis",
            labels={
                "articles": "Articles",
                "method": "Paradigma",
                "pct_recent": "% Recente (≥2021)",
            },
        )
        fig_bar.update_layout(
            xaxis_title="Number of Articles",
            yaxis_title="Paradigma",
            height=440,
            margin={"l": 20, "r": 20, "t": 30, "b": 30},
        )
        render_chart(fig_bar)

    with col_line:
        st.markdown("### 📈 Temporal Evolution of Main Methods")
        if not temporal_df.empty:
            top_cols = summary_df.head(5)["method"].tolist()
            plot_temp = temporal_df[[c for c in top_cols if c in temporal_df.columns]].reset_index()
            fig_temp = px.line(
                plot_temp,
                x="year",
                y=[c for c in top_cols if c in temporal_df.columns],
                labels={
                    "value": "In Articles / Year",
                    "year": "Year of Publication",
                    "variable": "Method",
                },
                color_discrete_sequence=CATEGORICAL_PALETTE,
            )
            fig_temp.update_layout(
                xaxis_title="Year of Publication",
                yaxis_title="In Articles / Year",
                height=440,
                legend={"orientation": "h", "y": -0.25, "x": 0.0},
                margin={"l": 20, "r": 20, "t": 30, "b": 30},
            )
            render_chart(fig_temp)

    st.markdown("### 📐 Mathematical Complexity Spectrum of Formulations")
    col_c1, col_c2 = st.columns([1, 1])
    with col_c1:
        if not spectrum_df.empty:
            fig_spec = px.bar(
                spectrum_df.sort_values(by="articles", ascending=True),
                x="articles",
                y="complexity_class",
                orientation="h",
                color="pct_recent",
                color_continuous_scale="Plasma",
                labels={
                    "articles": "Articles",
                    "complexity_class": "Classe de Complexidade",
                    "pct_recent": "% Recente (≥2021)",
                },
            )
            fig_spec.update_layout(
                xaxis_title="Articles in Corpus",
                yaxis_title="Classe de Complexidade",
                height=380,
                margin={"l": 20, "r": 20, "t": 30, "b": 30},
            )
            render_chart(fig_spec)

    with col_c2:
        if not temp_spec.empty:
            classes_avail = [c for c in temp_spec.columns if c != "year"]
            fig_tspec = px.line(
                temp_spec,
                x="year",
                y=classes_avail,
                labels={
                    "value": "Articles / Year",
                    "year": "Year of Publication",
                    "variable": "Classe",
                },
                color_discrete_sequence=CATEGORICAL_PALETTE,
            )
            fig_tspec.update_layout(
                xaxis_title="Year of Publication",
                yaxis_title="Articles / Year",
                height=380,
                legend={"orientation": "h", "y": -0.25, "x": 0.0},
                margin={"l": 20, "r": 20, "t": 30, "b": 30},
            )
            render_chart(fig_tspec)

    st.markdown("### 📋 Analytical Table of Optimization Methods")
    display_sum = summary_df.copy()
    display_sum["pct_recent"] = display_sum["pct_recent"].astype(str) + "%"
    display_sum.columns = [
        "Resolution paradigm",
        "Articles in Corpus",
        "Publications ≥ 2021",
        "Average citations",
    ]
    st.dataframe(display_sum, hide_index=True, width="stretch")


# ---------------------------------------------------------------------------
# Tab 2: Objective functions & co-optimization
# ---------------------------------------------------------------------------


def _render_objectives_tab(df: pd.DataFrame) -> None:
    st.markdown("## 🎯 Objective Functions & Multi-Criteria Formulations")
    st.caption(
        "Maps which criteria the literature seeks to optimize: economic costs of investment and operation (CAPEX/OPEX), "
        "Reliability and supply continuity (SAIDI, SAIFI, ENS), technical losses ($I^2R$), voltage profile, "
        "decarbonization and emissions, and resilience to extreme climatic events. "
        "The co-occurrence matrix reveals which pairs of objectives are often solved together."
    )

    obj_res = objective_functions_taxonomy(df)
    summary_df = obj_res["summary_df"]
    co_matrix = obj_res["co_matrix"]
    multi_ratio = obj_res["multi_obj_ratio"]
    temporal_multiobj = obj_res["temporal_multiobj"]

    if summary_df.empty:
        st.info("No objective function identified in the corpus.")
        return

    top_obj = summary_df.iloc[0]["objective"]
    top_cites = summary_df.sort_values(by="mean_citations", ascending=False).iloc[0]["objective"]

    summary_card_row(
        [
            ("🏆 Most frequent objective", top_obj, f"{summary_df.iloc[0]['articles']} articles"),
            ("🌐 Multi-objective rate", f"{multi_ratio:.1f}%", "Studies with ≥ 2 objectives"),
            ("💡 Highest citation impact", top_cites, "Average citations/article"),
            ("📐 Objective Families", str(len(summary_df)), "Fundamental criteria"),
        ]
    )

    col_bar, col_heat = st.columns([1, 1.1])

    with col_bar:
        st.markdown("#### 📊 Frequency and Currentity of Objective Functions")
        fig_obj = px.bar(
            summary_df.sort_values(by="articles", ascending=True),
            x="articles",
            y="objective",
            orientation="h",
            color="pct_recent",
            color_continuous_scale="Tealgrn",
            labels={
                "articles": "Articles",
                "objective": "Criteria/Objective Function",
                "pct_recent": "% Recente (≥2021)",
            },
        )
        fig_obj.update_layout(
            xaxis_title="Articles in Corpus",
            yaxis_title="Criteria/Objective Function",
            height=460,
            margin={"l": 20, "r": 20, "t": 30, "b": 30},
        )
        render_chart(fig_obj)

    with col_heat:
        st.markdown("### 🧬 Goal Co-Optimization Matrix")
        if not co_matrix.empty:
            fig_co = px.imshow(
                co_matrix,
                text_auto=True,
                aspect="auto",
                color_continuous_scale="Purples",
                labels={
                    "x": "Objetivo Concorrente",
                    "y": "Primary Objective",
                    "color": "Joint Articles",
                },
            )
            fig_co.update_layout(
                xaxis_title="Objetivo Concorrente",
                yaxis_title="Primary Objective",
                height=460,
                margin={"l": 20, "r": 20, "t": 30, "b": 30},
            )
            render_chart(fig_co)

    st.markdown("### 📈 Historical Transition: Mono-Objective vs. Multi-Objective")
    if not temporal_multiobj.empty:
        fig_mo = go.Figure()
        fig_mo.add_trace(
            go.Bar(
                x=temporal_multiobj["year"],
                y=temporal_multiobj["mono_objective"],
                name="Mono-Objetivo",
                marker={"color": "#636EFA"},
            )
        )
        fig_mo.add_trace(
            go.Bar(
                x=temporal_multiobj["year"],
                y=temporal_multiobj["multi_objective"],
                name="Multi-Objective (≥2 criteria)",
                marker={"color": "#00CC96"},
            )
        )
        fig_mo.add_trace(
            go.Scatter(
                x=temporal_multiobj["year"],
                y=temporal_multiobj["pct_multi"],
                yaxis="y2",
                mode="lines+markers",
                name="% multi-objective",
                line={"color": "#FFA15A", "width": 2.5},
            )
        )
        fig_mo.update_layout(
            barmode="stack",
            xaxis_title="Year of Publication",
            yaxis_title="Volume of Articles",
            yaxis2={
                "title": "Multi-Objective Participation (%)",
                "overlaying": "y",
                "side": "right",
                "showgrid": False,
            },
            legend={"orientation": "h", "y": -0.2, "x": 0.0},
            height=420,
            margin={"l": 20, "r": 20, "t": 30, "b": 30},
        )
        render_chart(fig_mo)

    st.markdown("#### 📋 Detailing of Objective Functions")
    disp_obj = summary_df.copy()
    disp_obj["pct_recent"] = disp_obj["pct_recent"].astype(str) + "%"
    disp_obj.columns = [
        "Objective-function / Criterion",
        "Articles in Corpus",
        "Publications ≥ 2021",
        "Average citations",
    ]
    st.dataframe(disp_obj, hide_index=True, width="stretch")


# ---------------------------------------------------------------------------
# Tab 3: Modelagem da Incerteza
# ---------------------------------------------------------------------------


def _render_uncertainty_tab(df: pd.DataFrame) -> None:
    st.markdown("## 🎲 Uncertainty & Stochastic Modeling Paradigms")
    st.caption(
        "Evaluates how the literature deals with the randomness of renewable generation (solar, wind), "
        "load and recharge behavior of electric vehicles. "
        "Counterposes deterministic approaches to stochastic programming with scenarios reduction, "
        "Robust optimization (worst case uncertainty sets), fuzzy logic and distributionally robust optimization (DRO)."
    )

    unc_res = uncertainty_paradigms_analysis(df)
    paradigms_df = unc_res["paradigms_df"]
    cross_resources = unc_res["cross_resources"]
    temp_paradigms = unc_res["temporal_paradigms"]

    if paradigms_df.empty:
        st.info("No uncertainty paradigm detected in the corpus.")
        return

    top_paradigm = paradigms_df.iloc[0]["paradigm"]
    top_cites = paradigms_df.sort_values(by="mean_citations", ascending=False).iloc[0]["paradigm"]

    summary_card_row(
        [
            (
                "🎲 Paradigma Predominante",
                top_paradigm,
                f"{paradigms_df.iloc[0]['articles']} articles",
            ),
            (
                "🚀 Most recent paradigm",
                paradigms_df.sort_values(by="pct_recent", ascending=False).iloc[0]["paradigm"],
                "Higher % post-2021",
            ),
            ("💡 Highest citation impact", top_cites, "Average citations/article"),
            ("📐 Paradigms mapped", str(len(paradigms_df)), "Uncertainty models"),
        ]
    )

    col_pbar, col_pcross = st.columns([1, 1.1])

    with col_pbar:
        st.markdown("#### Distribution of Uncertainty Paradigms")
        fig_p = px.bar(
            paradigms_df.sort_values(by="articles", ascending=True),
            x="articles",
            y="paradigm",
            orientation="h",
            color="pct_recent",
            color_continuous_scale="Plasma",
            labels={
                "articles": "Articles",
                "paradigm": "Paradigma de Incerteza",
                "pct_recent": "% Recente (≥2021)",
            },
        )
        fig_p.update_layout(
            xaxis_title="Articles in Corpus",
            yaxis_title="Paradigma de Incerteza",
            height=440,
            margin={"l": 20, "r": 20, "t": 30, "b": 30},
        )
        render_chart(fig_p)

    with col_pcross:
        st.markdown("### 🧬 Uncertainty × Modeled Physical Resource")
        if not cross_resources.empty:
            fig_cr = px.imshow(
                cross_resources,
                text_auto=True,
                aspect="auto",
                color_continuous_scale="Viridis",
                labels={
                    "x": "Resource / Physical Vector",
                    "y": "Paradigma de Incerteza",
                    "color": "Joint Articles",
                },
            )
            fig_cr.update_layout(
                xaxis_title="Resource / Physical Vector",
                yaxis_title="Paradigma de Incerteza",
                height=440,
                margin={"l": 20, "r": 20, "t": 30, "b": 30},
            )
            render_chart(fig_cr)

    if not temp_paradigms.empty:
        st.markdown("### 📈 Historical Evolution of Uncertainty Paradigms")
        p_cols = [c for c in temp_paradigms.columns if c != "year"]
        fig_tp = px.line(
            temp_paradigms,
            x="year",
            y=p_cols,
            labels={
                "value": "Articles / Year",
                "year": "Year of Publication",
                "variable": "Paradigma",
            },
            color_discrete_sequence=CATEGORICAL_PALETTE,
        )
        fig_tp.update_layout(
            xaxis_title="Year of Publication",
            yaxis_title="Articles / Year",
            height=420,
            legend={"orientation": "h", "y": -0.25, "x": 0.0},
            margin={"l": 20, "r": 20, "t": 30, "b": 30},
        )
        render_chart(fig_tp)

    st.markdown("### 📋 Synthesis of Uncertainty Paradigms")
    disp_p = paradigms_df.copy()
    disp_p["pct_recent"] = disp_p["pct_recent"].astype(str) + "%"
    disp_p.columns = [
        "Paradigma de Incerteza",
        "Articles in Corpus",
        "Publications ≥ 2021",
        "Average citations",
    ]
    st.dataframe(disp_p, hide_index=True, width="stretch")


# ---------------------------------------------------------------------------
# Tab 4: Horizontes Temporais & Escalas
# ---------------------------------------------------------------------------


def _render_horizons_tab(df: pd.DataFrame) -> None:
    st.markdown("## ⏱")
    st.caption(
        "Maps how literature structures the temporal horizon of expansion: "
        "from the **Static Planning** (single target year *snapshot*) to the **Multi-Year Dynamic Expansion** (*Multi-Year*), "
        "and the modern**Co-Optimization Planning + Operation** with hourly representation and representative days of demand / generation."
    )

    h_res = planning_time_horizons_analysis(df)
    horizons_df = h_res["horizons_df"]

    if horizons_df.empty:
        st.info("No temporal horizon detected in the corpus.")
        return

    top_h = horizons_df.iloc[0]["horizon"]
    total_art = horizons_df["articles"].sum()

    summary_card_row(
        [
            ("⏱️ Most adopted structure", top_h, f"{horizons_df.iloc[0]['articles']} articles"),
            (
                "🚀 Maior Momentum Recente",
                horizons_df.sort_values(by="pct_recent", ascending=False).iloc[0]["horizon"],
                "Higher % post-2021",
            ),
            (
                "💡 Highest citation impact",
                horizons_df.sort_values(by="mean_citations", ascending=False).iloc[0]["horizon"],
                "Average citations/article",
            ),
            ("📄 Classified articles", str(total_art), "Horizonte identificado"),
        ]
    )

    col_hbar, col_hcite = st.columns([1, 1])

    with col_hbar:
        st.markdown("#### 📊 Volume by horizon structure")
        fig_h = px.bar(
            horizons_df.sort_values(by="articles", ascending=True),
            x="articles",
            y="horizon",
            orientation="h",
            color="pct_recent",
            color_continuous_scale="Blues",
            labels={
                "articles": "Articles",
                "horizon": "Estrutura Temporal",
                "pct_recent": "% Recente (≥2021)",
            },
        )
        fig_h.update_layout(
            xaxis_title="Articles in Corpus",
            yaxis_title="Estrutura Temporal",
            height=380,
            margin={"l": 20, "r": 20, "t": 30, "b": 30},
        )
        render_chart(fig_h)

    with col_hcite:
        st.markdown("### 💡 Average citations by adopted horizon")
        fig_hc = px.bar(
            horizons_df.sort_values(by="mean_citations", ascending=True),
            x="mean_citations",
            y="horizon",
            orientation="h",
            color="mean_citations",
            color_continuous_scale="Viridis",
            labels={
                "mean_citations": "Average citations",
                "horizon": "Estrutura Temporal",
            },
        )
        fig_hc.update_layout(
            xaxis_title="Average citations per article",
            yaxis_title="Estrutura Temporal",
            height=380,
            margin={"l": 20, "r": 20, "t": 30, "b": 30},
        )
        render_chart(fig_hc)

    st.markdown(
        "> [!TIP]\n"
        ">**Methodological Evidence**: The transition to decarbonized networks forced literature to migrate from static planning "
        "Assets of short duration (such as BESS batteries and demand response) "
        "> can not be dimensioned without modeling the sub-clockwise operational flexibility and typical days throughout the year."
    )

    st.markdown("#### 📋 Planning-horizon details")
    disp_h = horizons_df.copy()
    disp_h["pct_recent"] = disp_h["pct_recent"].astype(str) + "%"
    disp_h.columns = [
        "Estrutura de Horizonte Temporal",
        "Articles in Corpus",
        "Publications ≥ 2021",
        "Average citations",
    ]
    st.dataframe(disp_h, hide_index=True, width="stretch")


# ---------------------------------------------------------------------------
# Tab 5: Benchmarks IEEE & Redes
# ---------------------------------------------------------------------------


def _render_feeders_tab(df: pd.DataFrame) -> None:
    st.markdown("## ")
    st.caption(
        "Maps which electrical test networks are used to validate the theoretical models. "
        "Differentiation of standardized IEEE radial feeders (33, 69 and 123 nodes) of real concessionaire networks."
    )

    res = benchmark_feeders_analysis(df)
    feeders_df = res["feeders_df"]
    cross_matrix = res["cross_matrix"]

    if feeders_df.empty:
        st.info("No test system identified in the corpus.")
        return

    top_feeder = feeders_df.iloc[0]["feeder"]
    real_nets = feeders_df[feeders_df["feeder"].str.contains("Reais")]["articles"].sum()

    summary_card_row(
        [
            (
                "🔌 Most used benchmark",
                top_feeder,
                f"{feeders_df.iloc[0]['articles']} articles",
            ),
            ("🏢 Validation in real networks", f"{real_nets} articles", "Studies with utilities"),
            ("⚡ Systems mapped", str(len(feeders_df)), "IEEE standards and practical networks"),
        ]
    )

    col_feed, col_cross = st.columns([1, 1.1])

    with col_feed:
        st.markdown("### 🏆 Study Case Distribution")
        fig_f = px.bar(
            feeders_df.sort_values(by="articles", ascending=True),
            x="articles",
            y="feeder",
            orientation="h",
            color="articles",
            color_continuous_scale="Tealgrn",
            labels={"articles": "Articles", "feeder": "Sistema de Teste"},
        )
        fig_f.update_layout(
            xaxis_title="Articles in Corpus",
            yaxis_title="Sistema de Teste",
            height=440,
            margin={"l": 20, "r": 20, "t": 30, "b": 30},
        )
        render_chart(fig_f)

    with col_cross:
        st.markdown("#### 🧬 Cross-matrix: benchmark × modeled resource")
        if not cross_matrix.empty:
            fig_c = px.imshow(
                cross_matrix,
                text_auto=True,
                aspect="auto",
                color_continuous_scale="Purp",
                labels={
                    "x": "Recurso / Tecnologia",
                    "y": "Test network / benchmark",
                    "color": "Joint Articles",
                },
            )
            fig_c.update_layout(
                xaxis_title="Recurso / Tecnologia",
                yaxis_title="Test network / benchmark",
                height=440,
                margin={"l": 20, "r": 20, "t": 30, "b": 30},
            )
            render_chart(fig_c)

    st.markdown("### 📋 Test Systems Catalog")
    disp_f = feeders_df.copy()
    disp_f["pct_recent"] = disp_f["pct_recent"].astype(str) + "%"
    disp_f.columns = [
        "Sistema de Teste / Rede",
        "Articles in Corpus",
        "Publications ≥ 2021",
        "Average citations",
    ]
    st.dataframe(disp_f, hide_index=True, width="stretch")


# ---------------------------------------------------------------------------
# Tab 6: Ferramental & Solvers Computacionais
# ---------------------------------------------------------------------------


def _render_solvers_tab(df: pd.DataFrame) -> None:
    st.markdown("### 💻 Ferramental Computacional, Modeladores & Solvers")
    st.caption(
        "Analyzes mathematical software, algebraic modeling platforms and specialized simulators "
        "that enable the computational implementation of theoretical formulations in the distribution literature."
    )

    s_res = computational_solvers_analysis(df)
    solvers_df = s_res["solvers_df"]
    ecosystem_df = s_res["ecosystem_df"]

    if solvers_df.empty:
        st.info("No explicit mention of computational solvers identified in the corpus.")
        return

    top_tool = solvers_df.iloc[0]["tool"]
    top_cat = ecosystem_df.iloc[0]["category"]

    summary_card_row(
        [
            ("💻 Most cited tool", top_tool, f"{solvers_df.iloc[0]['articles']} articles"),
            (
                "📐 Leader Ecosystem",
                top_cat,
                f"{ecosystem_df.iloc[0]['articles']} references",
            ),
            ("🛠️ Ferramentas Mapeadas", str(len(solvers_df)), "Solvers e simuladores"),
        ]
    )

    col_s1, col_s2 = st.columns([1.1, 0.9])

    with col_s1:
        st.markdown("### 📊 Platform Distribution / Solver")
        fig_s = px.bar(
            solvers_df.sort_values(by="articles", ascending=True),
            x="articles",
            y="tool",
            orientation="h",
            color="category",
            color_discrete_sequence=CATEGORICAL_PALETTE,
            labels={
                "articles": "Articles",
                "tool": "Software / Solver",
                "category": "Ecossistema",
            },
        )
        fig_s.update_layout(
            xaxis_title="Articles in Corpus",
            yaxis_title="Software / Solver",
            height=440,
            margin={"l": 20, "r": 20, "t": 30, "b": 30},
        )
        render_chart(fig_s)

    with col_s2:
        st.markdown("### 🧩 Tooling Category Participation")
        if not ecosystem_df.empty:
            fig_pie = px.pie(
                ecosystem_df,
                names="category",
                values="articles",
                hole=0.45,
                color_discrete_sequence=CATEGORICAL_PALETTE,
            )
            fig_pie.update_layout(
                height=440,
                legend={"orientation": "h", "y": -0.15, "x": 0.0},
                margin={"l": 20, "r": 20, "t": 30, "b": 30},
            )
            render_chart(fig_pie)

    st.markdown("#### 📋 Computational ecosystem breakdown")
    disp_s = solvers_df.copy()
    disp_s["pct_recent"] = disp_s["pct_recent"].astype(str) + "%"
    disp_s.columns = [
        "Software / Ferramenta",
        "Categoria de Software",
        "Articles in Corpus",
        "Publications ≥ 2021",
        "Average citations",
    ]
    st.dataframe(disp_s, hide_index=True, width="stretch")
