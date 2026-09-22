"""🔮 Trends and fronts — publication-volume projection via regression.

The horizon follows the last complete bibliographic year, so it moves with the
corpus rather than being pinned to a calendar year in this docstring.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from lake_research_map.dashboard import loaders
from lake_research_map.dashboard.analytics import technological_burst_detection
from lake_research_map.dashboard.components import (
    hero_banner,
    metric_row,
    page_header,
    render_chart,
)
from lake_research_map.dashboard.forecasting import (
    COVERAGE_HOLDOUT_FOLDS,
    CV_YEARS,
    HOLDOUT_YEAR,
    TRAIN_END_YEAR,
    ForecastResult,
)
from lake_research_map.dashboard.theme import (
    CATEGORICAL_PALETTE,
    SOURCE_COLORS,
    SOURCE_LABELS,
    TOTAL_COLOR,
    hex_to_rgba,
    theme_tokens,
)

TOP_KEYWORDS_FORECAST = 10
TOP_KEYWORD_TRENDS = 5
MIN_KEYWORD_OCCURRENCES = 20

_MODEL_LABELS = {
    "baseline": "Persistence (last value)",
    "linear": "Linear",
    "log_linear": "Log-linear (exponential growth)",
    "none": "—",
}


def render() -> None:
    page_header(
        "🔮",
        "Trends and fronts",
        "Volume projection with temporal validation, training only in complete years and intervals "
        "conformations calibrated in historical errors.",
    )

    hero_banner(
        "Complete historical series",
        "Persistence, linear trend and log-linear trend "
        "The partial year is only monitored; it does not enter the training. "
        "The bands use conformal historical errors and represent uncertainty of the collected corpus.",
    )

    _, articles_df = loaders.articles()
    if articles_df.empty:
        st.warning("Run the pipeline and reload this page.")
        return

    tab_volume, tab_keywords, tab_bass, tab_bursts = st.tabs(
        [
            "Volume",
            "Topic trajectories",
            "Technological diffusion",
            "Concept bursts",
        ],
        on_change="rerun",
        key="forecasting_primary_tab",
    )
    if tab_volume.open:
        with tab_volume:
            source_label = st.segmented_control(
                "Series",
                ["Total", SOURCE_LABELS["ieee"], SOURCE_LABELS["elsevier"]],
                default="Total",
                key="forecast_source",
            )
            sources = {
                "Total": (None, TOTAL_COLOR),
                SOURCE_LABELS["ieee"]: ("ieee", SOURCE_COLORS["ieee"]),
                SOURCE_LABELS["elsevier"]: ("elsevier", SOURCE_COLORS["elsevier"]),
            }
            source, color = sources[source_label or "Total"]
            _render_series_forecast(source_label or "Total", color, _series_forecast(source))
    elif tab_keywords.open:
        with tab_keywords:
            _keyword_growth_ranking()
    elif tab_bass.open:
        with tab_bass:
            _bass_diffusion_analysis()
    elif tab_bursts.open:
        with tab_bursts:
            _render_bursts(articles_df)


def _render_bursts(df: pd.DataFrame) -> None:
    st.subheader("Concept frequency bursts")
    st.caption(
        "Kleinberg's two-state model applied to the annual share of documents mentioning each concept. "
        "The intensity is the log-likelihood gain of the burst state."
    )
    result = technological_burst_detection(df)
    bursts = result["burst_timeline"]
    if bursts.empty:
        st.info("No sustained burst was detected under the current filters.")
        return
    metric_row(
        [
            ("Detected intervals", f"{result['total_bursts']:,}"),
            ("Active in the latest year", f"{len(result['active_frontiers']):,}"),
            ("Highest intensity", f"{bursts.iloc[0]['Intensity']:.2f}"),
        ]
    )
    plot = bursts.sort_values(["Burst's Beginning", "Intensity"])
    figure = go.Figure()
    for _, row in plot.iterrows():
        figure.add_bar(
            y=[row["Technology / Concept"]],
            x=[row["Duration (Years)"]],
            base=[row["Burst's Beginning"]],
            orientation="h",
            marker_color=CATEGORICAL_PALETTE[0]
            if row["Status"] == "Active"
            else CATEGORICAL_PALETTE[6],
            hovertemplate=(
                f"Start: {row["Burst's Beginning"]}<br>Peak: {row['Peak year']}<br>"
                f"End: {row['Burst end']}<br>Intensity: {row['Intensity']:.2f}<extra></extra>"
            ),
            showlegend=False,
        )
    figure.update_layout(xaxis_title="Year", yaxis_title="Concept")
    render_chart(figure)
    st.dataframe(bursts, hide_index=True, width="stretch")


def _series_forecast(source: str | None) -> ForecastResult:
    """Volume forecast for one source (or the whole corpus when `source` is None).

    Cached for the same reason as `_keyword_forecasts`: the fit is a
    rolling-origin CV over 3 candidate models, and it reads the unfiltered
    layer, so it never changes between reruns.
    """
    return loaders.volume_forecast(source)


def _horizon_backtest_table(result) -> None:
    """Score the chosen model separately at every horizon it is asked to predict.

    Every fold used to be one step ahead, so the second forecast year appeared
    on the chart with no validation behind it at all. A two-year claim has to
    be backtested two years out or labelled as unvalidated.
    """
    backtests = getattr(result, "horizon_backtests", None)
    if not backtests or len(backtests) < 2:
        return

    rows = []
    for step, year in enumerate(result.forecast_years, start=1):
        stats = backtests.get(step) or {}
        mase = stats.get("mase")
        rows.append(
            {
                "Horizon": f"{step}-year ({year})",
                "CV MAE": ("n/a" if stats.get("cv_mae") is None else f"{stats['cv_mae']:.1f}"),
                "MASE": "n/a" if mase is None else f"{mase:.2f}",
                "Beats naive": "—" if mase is None else ("yes" if mase < 1 else "no"),
                "Interval coverage": (
                    "not testable" if stats.get("coverage") is None else f"{stats['coverage']:.0%}"
                ),
            }
        )

    with st.expander("Backtest by forecast horizon", expanded=False):
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        st.caption(
            "Each horizon is scored on folds whose training window ends that many years "
            "before the target, so a two-year number never borrows one-year information. "
            "MASE divides the error by the average one-step change in the series, which is "
            "what makes it comparable across series of different size: below 1 beats a "
            'naive carry-forward, above 1 loses to it. "Not testable" means the series '
            "is too short to hold folds back at that horizon — an unknown coverage is "
            "reported as unknown rather than filled in from the shorter horizon."
        )


def _baseline_skill_label(skill: float | None) -> str:
    """A forecast that does not beat last-year-repeated has to say so.

    Skill is 1 - MAE(model) / MAE(persistence) over the rolling-origin folds, so
    zero or less means the naive baseline was at least as good.
    """
    if skill is None:
        return "n/a"
    return f"{skill:+.0%}" if skill > 0 else "No better than naive"


def _baseline_skill_help(skill: float | None) -> str:
    if skill is None:
        return "Not enough complete years to compare against persistence"
    if skill > 0:
        return "Lower validation error than repeating the last complete year"
    return "Persistence matched or beat the fitted model; read the projection with care"


def _render_series_forecast(label: str, color: str, result: ForecastResult) -> None:
    if result.insufficient_data:
        st.info(" ".join(result.notes) or "Insufficient data for a forecast.")
        return

    partial_note = ""
    if result.holdout_actual is not None:
        err = abs(result.holdout_predicted - result.holdout_actual)
        partial_note = f"{err:,.1f} (vs. partial {HOLDOUT_YEAR})"
    metric_row(
        [
            (
                "🧮 Selected model",
                _MODEL_LABELS.get(result.chosen_model, result.chosen_model),
                None,
            ),
            (
                "📉 MAE validation",
                partial_note or "n/a",
                f"CV {CV_YEARS[0]}–{CV_YEARS[-1]}: {result.cv_mae:.1f}"
                if result.cv_mae == result.cv_mae
                else None,
            ),
            (
                "🏁 Skill vs. persistence",
                _baseline_skill_label(result.baseline_skill),
                _baseline_skill_help(result.baseline_skill),
            ),
            (
                "📏 Interval coverage",
                "n/a"
                if result.empirical_interval_coverage is None
                else f"{result.empirical_interval_coverage:.0%}",
                f"{COVERAGE_HOLDOUT_FOLDS} held-out one-step folds in a 90% band"
                if result.empirical_interval_coverage is not None
                else "Series too short to hold folds back",
            ),
            (
                f"🔮 Forecast {result.forecast_years[0]}",
                f"{result.forecast_values[0]:,.0f}",
                f"±{(result.forecast_upper[0] - result.forecast_values[0]):,.0f}",
            ),
            (
                f"🔮 Forecast {result.forecast_years[1]}",
                f"{result.forecast_values[1]:,.0f}",
                f"±{(result.forecast_upper[1] - result.forecast_values[1]):,.0f}",
            ),
        ]
    )

    _horizon_backtest_table(result)

    fig = go.Figure()

    # Observed history (bars) -- includes the partial holdout year.
    fig.add_bar(
        x=result.history.index,
        y=result.history.values,
        name=f"{label} (observed)",
        marker_color=color,
        opacity=0.85,
    )

    # Fitted curve over complete training years only.
    fig.add_trace(
        go.Scatter(
            x=result.fitted_curve.index,
            y=result.fitted_curve.values,
            name="Fitted model",
            mode="lines",
            line=dict(color=color, width=2, dash="dot"),
        )
    )

    # Forecast band (shaded) -- drawn before the forecast line so the line sits on top.
    band_years = list(result.forecast_years)
    fig.add_trace(
        go.Scatter(
            x=band_years + band_years[::-1],
            y=list(result.forecast_upper) + list(result.forecast_lower[::-1]),
            fill="toself",
            fillcolor=hex_to_rgba(TOTAL_COLOR, 0.18),
            line=dict(color="rgba(0,0,0,0)"),
            hoverinfo="skip",
            name="Confidence interval (~95%)",
            showlegend=True,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=band_years,
            y=result.forecast_values,
            name="Prediction",
            mode="lines+markers",
            line=dict(color=TOTAL_COLOR, width=2.5, dash="dash"),
            marker=dict(size=9, symbol="diamond"),
            hovertemplate="%{x} prediction: %{y:,.0f} articles<extra></extra>",
        )
    )

    # Partial next-year actual (e.g. the handful of 2027 records already indexed) --
    # plotted separately so it's never mistaken for the forecast itself.
    next_year = band_years[0]
    if next_year in result.history.index and result.history.loc[next_year] > 0:
        # Fill with the vivid highlight color and outline in the chart's own
        # background -- a fixed white fill (the previous approach) disappears
        # against a white chart background in light mode.
        t = theme_tokens()
        fig.add_trace(
            go.Scatter(
                x=[next_year],
                y=[result.history.loc[next_year]],
                name=f"{next_year} (partial/early)",
                mode="markers",
                marker=dict(
                    size=13,
                    symbol="star",
                    color=TOTAL_COLOR,
                    line=dict(width=2, color=t["chart_bg"]),
                ),
                hovertemplate=f"{next_year} already has %{{y:,.0f}} indexed records (partial)<extra></extra>",
            )
        )

    fig.update_layout(
        xaxis_title="Year of publication",
        yaxis_title="Number of articles",
        hovermode="x unified",
    )
    render_chart(
        fig,
        caption=f"Bars are observed volumes, including partial {HOLDOUT_YEAR}; the dotted line is "
        "the historical model fit. Any records already indexed for the following year are shown "
        "separately because they do not represent that year's final total.",
    )

    with st.expander("📋 Comparison of candidate models"):
        table = result.model_comparison.copy()
        table["model"] = table["model"].map(_MODEL_LABELS)
        table = table.rename(
            columns={
                "model": "Model",
                "holdout_mae": f"MAE vs. partial {HOLDOUT_YEAR}",
                "cv_mae": f"MAE cross-validation ({CV_YEARS[0]}–{CV_YEARS[-1]})",
                "combined_mae": "MAE temporal (usado na escolha)",
            }
        )
        st.dataframe(table, hide_index=True, width="stretch")
        st.caption(
            "The model with the lowest MAE in temporal validation is chosen and readjusted only with years"
            f"complete years. {HOLDOUT_YEAR} remains outside training because it is partial."
        )


def _keyword_trend_lines(
    keywords: list[str], results_by_keyword: dict[str, ForecastResult]
) -> None:
    """Actual trajectory (solid) + forecast continuation (dashed) per topic.

    The ranking bar next to this only shows the net change between two
    points; this shows the real yearly shape leading up to it -- some
    "growing" topics rise steadily, others spike once and plateau, and that
    distinction doesn't survive a single before/after number.
    """
    fig = go.Figure()
    for i, kw in enumerate(keywords):
        result = results_by_keyword.get(kw)
        if result is None or result.insufficient_data:
            continue
        color = CATEGORICAL_PALETTE[i % len(CATEGORICAL_PALETTE)]

        fig.add_trace(
            go.Scatter(
                x=list(result.history.index),
                y=list(result.history.values),
                name=kw,
                legendgroup=kw,
                mode="lines",
                line=dict(color=color, width=2),
                hovertemplate=f"<b>{kw}</b><br>Year %{{x}}: %{{y:.0f}} mentions<extra></extra>",
            )
        )
        # Dashed continuation from the last real point into the forecast, so
        # the line doesn't visually jump -- not shown in the legend, since
        # it's the same topic as the solid trace right above it.
        forecast_x = [result.history.index[-1], *result.forecast_years]
        forecast_y = [result.history.values[-1], *result.forecast_values]
        fig.add_trace(
            go.Scatter(
                x=forecast_x,
                y=forecast_y,
                name=kw,
                legendgroup=kw,
                showlegend=False,
                mode="lines",
                line=dict(color=color, width=2, dash="dash"),
                hovertemplate=f"<b>{kw}</b> (forecast)<br>Year %{{x}}: %{{y:.0f}} mentions<extra></extra>",
            )
        )

    fig.update_layout(
        xaxis_title="Year of publication",
        yaxis_title="Mentions per year",
        hovermode="x unified",
    )
    render_chart(
        fig,
        caption="Solid = observed history; traced : continuation predicted by the same chosen model "
        "Shows the real trajectory behind the next ranking, not only the point of arrival.",
    )


def _keyword_forecasts() -> tuple[str, list[dict], dict[str, ForecastResult], int | None]:
    """Fit a forecast per eligible keyword; returns `(status, rows, results, final_year)`.

    Cached because it fits 3 candidate models per keyword per CV fold —
    hundreds of sklearn fits — and this page reads the unfiltered layer, so
    the result never changes between reruns. Without this it re-ran in full
    on every widget interaction.
    """
    return loaders.keyword_forecasts(MIN_KEYWORD_OCCURRENCES)


def _keyword_growth_ranking() -> None:
    st.subheader("Topics with higher projected growth")
    status, rows, results_by_keyword, final_forecast_year = _keyword_forecasts()

    if status == "no_keywords":
        st.info("Keywords column not available in this layer.")
        return
    if status == "none_eligible":
        st.info(f"No keyword has at least {MIN_KEYWORD_OCCURRENCES} occurrences.")
        return
    if not rows:
        st.info("It was not possible to adjust a model to any eligible keyword.")
        return

    ranking = pd.DataFrame(rows).sort_values("variation", ascending=False)
    ranking["model"] = ranking["model"].map(lambda value: _MODEL_LABELS.get(value, value))
    top = ranking.head(min(TOP_KEYWORDS_FORECAST, len(ranking))).sort_values("variation")

    sub_trend, sub_rank = st.tabs(
        ["📈 Trajectories", "🏆 Growth ranking"], on_change="rerun", key="forecast_keyword_tab"
    )
    if sub_trend.open:
        with sub_trend:
            _keyword_trend_lines(
                ranking.head(TOP_KEYWORD_TRENDS)["keyword"].tolist(), results_by_keyword
            )
    if sub_rank.open:
        with sub_rank:
            fig = go.Figure()
            fig.add_bar(
                x=top["variation"],
                y=top["keyword"],
                orientation="h",
                marker_color=[
                    TOTAL_COLOR if v >= 0 else SOURCE_COLORS["ieee"] for v in top["variation"]
                ],
                hovertemplate=f"<b>%{{y}}</b><br>Projected change through {final_forecast_year}: %{{x:+.1f}} articles/year<extra></extra>",
            )
            fig.update_layout(
                xaxis_title=f"Projected change ({TRAIN_END_YEAR} → {final_forecast_year}, articles/year)",
                yaxis_title="Keyword",
            )
            render_chart(
                fig,
                caption=f"The same forecasting engine used for publication volume, applied to each keyword with at "
                f"least {MIN_KEYWORD_OCCURRENCES} occurrences in the corpus. The same caveat applies: "
                f"{HOLDOUT_YEAR} is partial and the corpus is incomplete, so interpret the projection "
                "as directional rather than exact.",
            )

    with st.expander("📋 Complete table of topics evaluated"):
        st.dataframe(ranking.reset_index(drop=True), hide_index=True, width="stretch")


def _bass_diffusion_analysis() -> None:
    st.subheader("📊 Bass Diffusion Model for Emerging Technologies")
    st.caption(
        "The Bass diffusion model represents the adoption cycle of technological innovations by "
        "separating innovators' external influence (p) from internal imitation effects (q). It also "
        "estimates theoretical saturation (m) and the peak publication year (t*)."
    )
    from lake_research_map.dashboard.forecasting import fit_bass_diffusion_nls

    status, rows, results_by_keyword, _ = _keyword_forecasts()
    if status != "ok" or not results_by_keyword:
        st.info("Insufficient keywords for the diffusion model.")
        return

    bass_records = []
    for kw, f_res in results_by_keyword.items():
        hist = f_res.history
        years = hist.index.to_numpy()
        adoptions = hist.values
        bass = fit_bass_diffusion_nls(years, adoptions)
        if bass.get("valid"):
            bass_records.append(
                {
                    "Technology / Topic": kw,
                    "Current Stage": bass["stage"].title(),
                    "Method": bass.get("method", "nls").upper(),
                    "Coef. Innovation (p)": round(bass["p"], 4),
                    "Coef. Imitation (q)": round(bass["q"], 4),
                    "Saturation Potential (m)": int(round(bass["m"])),
                    "Estimated peak year": (int(round(bass["t_peak"])) if bass["t_peak"] else "—"),
                }
            )

    if bass_records:
        st.dataframe(pd.DataFrame(bass_records), hide_index=True, width="stretch")
        st.caption(
            "Nonlinear continuous adjustment (NLS via `scipy.optimize.curve_fit`) with physical capacity limits. "
            "Topics in 'Growth' stage have not yet reached the apex of scientific production; "
            "Topics in 'Maturity' have already exceeded the estimated peak year and tend to stabilize."
        )
    else:
        st.info("No topic with sufficient history for stable convergence of the Bass model.")
