"""🔮 Tendências & Previsão — projeção de volume de publicações (2027-2028) via regressão."""

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
from lake_research_map.dashboard.forecasting import HOLDOUT_YEAR, ForecastResult
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
    "baseline": "Persistência (último valor)",
    "linear": "Linear",
    "log_linear": "Log-linear (crescimento exponencial)",
    "none": "—",
}


def render() -> None:
    page_header(
        "🔮",
        "Tendências e frentes",
        "Projeção de volume com validação temporal, treino apenas em anos completos e intervalos "
        "conformes calibrados nos erros históricos.",
    )

    hero_banner(
        "Série histórica completa",
        "A previsão ignora o filtro global de ano. Persistência, tendência linear e tendência log-linear "
        "são comparadas por validação temporal. O ano parcial é apenas monitorado; não entra no treino. "
        "As bandas usam erros históricos conformes e representam incerteza do corpus coletado.",
    )

    _, articles_df = loaders.articles()
    if articles_df.empty:
        st.warning("Nenhum dado disponível ainda. Execute o pipeline e recarregue esta página.")
        return

    tab_volume, tab_keywords, tab_bass, tab_bursts = st.tabs(
        [
            "Volume",
            "Trajetórias de tópicos",
            "Difusão tecnológica",
            "Bursts de conceitos",
        ],
        on_change="rerun",
        key="forecasting_primary_tab",
    )
    if tab_volume.open:
        with tab_volume:
            source_label = st.segmented_control(
                "Série",
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
    st.subheader("Bursts de frequência de conceitos")
    st.caption(
        "Modelo de dois estados de Kleinberg aplicado à proporção anual de documentos que menciona "
        "cada conceito. A intensidade é o ganho de log-verossimilhança do estado de burst."
    )
    result = technological_burst_detection(df)
    bursts = result["burst_timeline"]
    if bursts.empty:
        st.info("Nenhum burst sustentado foi detectado neste recorte.")
        return
    metric_row(
        [
            ("Intervalos detectados", f"{result['total_bursts']:,}"),
            ("Ativos no último ano", f"{len(result['active_frontiers']):,}"),
            ("Maior intensidade", f"{bursts.iloc[0]['Intensidade']:.2f}"),
        ]
    )
    plot = bursts.sort_values(["Início do Burst", "Intensidade"])
    figure = go.Figure()
    for _, row in plot.iterrows():
        figure.add_bar(
            y=[row["Tecnologia / Conceito"]],
            x=[row["Duração (Anos)"]],
            base=[row["Início do Burst"]],
            orientation="h",
            marker_color=CATEGORICAL_PALETTE[0]
            if row["Status"] == "Ativo"
            else CATEGORICAL_PALETTE[6],
            hovertemplate=(
                f"Início: {row['Início do Burst']}<br>Pico: {row['Ano de Pico']}<br>"
                f"Fim: {row['Fim do Burst']}<br>Intensidade: {row['Intensidade']:.2f}<extra></extra>"
            ),
            showlegend=False,
        )
    figure.update_layout(xaxis_title="Ano", yaxis_title="Conceito")
    render_chart(figure)
    st.dataframe(bursts, hide_index=True, width="stretch")


def _series_forecast(source: str | None) -> ForecastResult:
    """Volume forecast for one source (or the whole corpus when `source` is None).

    Cached for the same reason as `_keyword_forecasts`: the fit is a
    rolling-origin CV over 3 candidate models, and it reads the unfiltered
    layer, so it never changes between reruns.
    """
    return loaders.volume_forecast(source)


def _render_series_forecast(label: str, color: str, result: ForecastResult) -> None:
    if result.insufficient_data:
        st.info(" ".join(result.notes) or "Dados insuficientes para uma previsão.")
        return

    partial_note = ""
    if result.holdout_actual is not None:
        err = abs(result.holdout_predicted - result.holdout_actual)
        partial_note = f"{err:,.1f} (vs. {HOLDOUT_YEAR}, parcial)"
    metric_row(
        [
            (
                "🧮 Modelo escolhido",
                _MODEL_LABELS.get(result.chosen_model, result.chosen_model),
                None,
            ),
            (
                "📉 MAE de validação",
                partial_note or "N/D",
                f"CV 2023–2025: {result.cv_mae:.1f}" if result.cv_mae == result.cv_mae else None,
            ),
            (
                "📈 R² (ajuste no treino)",
                f"{result.r2_train:.2f}" if result.r2_train == result.r2_train else "N/D",
                None,
            ),
            (
                f"🔮 Previsão {result.forecast_years[0]}",
                f"{result.forecast_values[0]:,.0f}",
                f"±{(result.forecast_upper[0] - result.forecast_values[0]):,.0f}",
            ),
            (
                f"🔮 Previsão {result.forecast_years[1]}",
                f"{result.forecast_values[1]:,.0f}",
                f"±{(result.forecast_upper[1] - result.forecast_values[1]):,.0f}",
            ),
        ]
    )

    fig = go.Figure()

    # Observed history (bars) -- includes the partial holdout year.
    fig.add_bar(
        x=result.history.index,
        y=result.history.values,
        name=f"{label} (observado)",
        marker_color=color,
        opacity=0.85,
    )

    # Fitted curve over complete training years only.
    fig.add_trace(
        go.Scatter(
            x=result.fitted_curve.index,
            y=result.fitted_curve.values,
            name="Modelo ajustado",
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
            name="Intervalo de confiança (~95%)",
            showlegend=True,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=band_years,
            y=result.forecast_values,
            name="Previsão",
            mode="lines+markers",
            line=dict(color=TOTAL_COLOR, width=2.5, dash="dash"),
            marker=dict(size=9, symbol="diamond"),
            hovertemplate="Previsão %{x}: %{y:,.0f} artigos<extra></extra>",
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
                name=f"{next_year} (parcial/antecipado)",
                mode="markers",
                marker=dict(
                    size=13,
                    symbol="star",
                    color=TOTAL_COLOR,
                    line=dict(width=2, color=t["chart_bg"]),
                ),
                hovertemplate=f"{next_year} já tem %{{y:,.0f}} registros indexados (parcial)<extra></extra>",
            )
        )

    fig.update_layout(
        xaxis_title="Ano de publicação",
        yaxis_title="Quantidade de artigos",
        hovermode="x unified",
    )
    render_chart(
        fig,
        caption=f"Barras = observado (inclui {HOLDOUT_YEAR}, parcial). Linha pontilhada = ajuste do modelo "
        "no histórico. Losango tracejado + faixa sombreada = previsão e intervalo de confiança. Estrela = "
        "registros já indexados para o próximo ano, mostrados à parte por não serem o total final dele.",
    )

    with st.expander("📋 Comparação dos modelos candidatos"):
        table = result.model_comparison.copy()
        table["model"] = table["model"].map(_MODEL_LABELS)
        table = table.rename(
            columns={
                "model": "Modelo",
                "holdout_mae": f"MAE vs. {HOLDOUT_YEAR} (parcial)",
                "cv_mae": "MAE validação cruzada (2023–2025)",
                "combined_mae": "MAE temporal (usado na escolha)",
            }
        )
        st.dataframe(table, hide_index=True, width="stretch")
        st.caption(
            "O modelo com menor MAE na validação temporal é escolhido e reajustado somente com anos "
            f"completos. {HOLDOUT_YEAR} permanece fora do treino por ser parcial."
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
                hovertemplate=f"<b>{kw}</b><br>Ano %{{x}}: %{{y:.0f}} menções<extra></extra>",
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
                hovertemplate=f"<b>{kw}</b> (previsto)<br>Ano %{{x}}: %{{y:.0f}} menções<extra></extra>",
            )
        )

    fig.update_layout(
        xaxis_title="Ano de publicação",
        yaxis_title="Menções por ano",
        hovermode="x unified",
    )
    render_chart(
        fig,
        caption="Sólido = histórico observado; tracejado = continuação prevista pelo mesmo modelo escolhido "
        "para cada termo. Mostra a trajetória real por trás do ranking ao lado, não só o ponto de chegada.",
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
    st.subheader("🏷️ Tópicos com maior crescimento projetado")
    status, rows, results_by_keyword, final_forecast_year = _keyword_forecasts()

    if status == "no_keywords":
        st.info("Coluna 'keywords' não disponível nesta camada.")
        return
    if status == "none_eligible":
        st.info(f"Nenhuma palavra-chave com pelo menos {MIN_KEYWORD_OCCURRENCES} ocorrências.")
        return
    if not rows:
        st.info("Não foi possível ajustar um modelo para nenhuma palavra-chave elegível.")
        return

    ranking = pd.DataFrame(rows).sort_values("variação", ascending=False)
    ranking["modelo"] = ranking["modelo"].map(lambda value: _MODEL_LABELS.get(value, value))
    top = ranking.head(min(TOP_KEYWORDS_FORECAST, len(ranking))).sort_values("variação")

    sub_trend, sub_rank = st.tabs(["📈 Trajetórias", "🏆 Ranking de Crescimento"])
    with sub_trend:
        _keyword_trend_lines(
            ranking.head(TOP_KEYWORD_TRENDS)["keyword"].tolist(), results_by_keyword
        )
    with sub_rank:
        fig = go.Figure()
        fig.add_bar(
            x=top["variação"],
            y=top["keyword"],
            orientation="h",
            marker_color=[
                TOTAL_COLOR if v >= 0 else SOURCE_COLORS["ieee"] for v in top["variação"]
            ],
            hovertemplate=f"<b>%{{y}}</b><br>Variação projetada até {final_forecast_year}: %{{x:+.1f}} artigos/ano<extra></extra>",
        )
        fig.update_layout(
            xaxis_title=f"Variação projetada (2025 → {final_forecast_year}, artigos/ano)",
            yaxis_title="Palavra-chave",
        )
        render_chart(
            fig,
            caption=f"Mesmo motor de previsão da série de volume, aplicado a cada palavra-chave com pelo "
            f"menos {MIN_KEYWORD_OCCURRENCES} ocorrências no corpus. Mesmas ressalvas: {HOLDOUT_YEAR} é um "
            "ano parcial e o corpus é incompleto — leia como sinal direcional, não como número exato.",
        )

    with st.expander("📋 Tabela completa de tópicos avaliados"):
        st.dataframe(ranking.reset_index(drop=True), hide_index=True, width="stretch")


def _bass_diffusion_analysis() -> None:
    st.subheader("📊 Modelo de Difusão de Bass para Tecnologias Emergentes")
    st.caption(
        "O Modelo de Difusão de Bass (1969) modela o ciclo de adoção de inovações tecnológicas, "
        "separando a influência externa de inovadores (p) da influência de contágio/imitação interna (q). "
        "Permite estimar a capacidade de saturação teórica (m) e o ano de pico de publicações (t*)."
    )
    from lake_research_map.dashboard.forecasting import fit_bass_diffusion_nls

    status, rows, results_by_keyword, _ = _keyword_forecasts()
    if status != "ok" or not results_by_keyword:
        st.info("Palavras-chave insuficientes para o modelo de difusão.")
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
                    "Tecnologia / Tópico": kw,
                    "Estágio Atual": bass["stage"].title(),
                    "Método": bass.get("method", "nls").upper(),
                    "Coef. Inovação (p)": round(bass["p"], 4),
                    "Coef. Imitação (q)": round(bass["q"], 4),
                    "Potencial de Saturação (m)": int(round(bass["m"])),
                    "Ano de Pico Estimado": (int(round(bass["t_peak"])) if bass["t_peak"] else "—"),
                }
            )

    if bass_records:
        st.dataframe(pd.DataFrame(bass_records), hide_index=True, width="stretch")
        st.caption(
            "Ajuste contínuo não-linear (NLS via `scipy.optimize.curve_fit`) com limites físicos de capacidade. "
            "Tópicos em estágio de 'Crescimento' ainda não atingiram o ápice de produção científica; "
            "tópicos em 'Maturidade' já ultrapassaram o ano de pico estimado e tendem à estabilização."
        )
    else:
        st.info(
            "Nenhum tópico com histórico suficiente para convergência estável do modelo de Bass."
        )
