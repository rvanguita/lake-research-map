"""🧭 Semântica & Relevância — o que o corpus realmente contém, segundo os embeddings."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from lake_research_map.dashboard import loaders
from lake_research_map.dashboard.components import (
    article_table,
    hero_banner,
    metric_row,
    page_header,
    render_chart,
    require_columns,
)
from lake_research_map.dashboard.theme import (
    CATEGORICAL_PALETTE,
    TREND_DOWN_COLOR,
    theme_tokens,
)
from lake_research_map.transform.screening_calibration import (
    calibrate_screening_threshold,
    generate_stratified_screening_sample,
    resolve_review_consensus,
    reviewer_agreement,
    validate_review_labels,
)

LOW_RELEVANCE_PERCENTILE = 10
TOP_REVIEW_ROWS = 40

# How the semantic map can be colored. The theme is the default, but a reviewer
# screening a corpus wants to ask the same picture different questions -- where
# the off-topic mass sits, whether a region is recent or old, which publisher
# indexed it.
MAP_COLOR_OPTIONS = {
    "Tema": "theme_label",
    "Relevância": "relevance_margin",
    "Ano": "year",
    "Fonte": "source_display",
}


def render() -> None:
    page_header(
        "🧭",
        "Triagem e descoberta",
        "Triagem de relevância, temas descobertos automaticamente e quase-duplicatas — tudo "
        "derivado dos embeddings dos resumos.",
    )

    signals = loaders.semantics()
    if signals.empty:
        st.info(
            "A etapa `semantic` ainda não foi executada — rode o botão na barra lateral ou "
            "`uv run lake-research-map --stage semantic`. Ela depende de `gold` e `embed`."
        )
        return

    articles_df = loaders.require_articles()
    scoped = loaders.with_semantics(articles_df)
    if not require_columns(scoped, ["relevance_score"]):
        return
    scored = scoped.dropna(subset=["relevance_score"])

    hero_banner(
        "Por que esta página existe",
        "A busca que gerou este corpus (<i>distribution system planning</i>) é ambígua: casa tanto "
        "com <b>distribuição de energia elétrica</b> quanto com <b>distribuição logística</b>. "
        "Cada artigo recebe aqui um score de proximidade ao tema da revisão, calculado sobre o "
        "embedding do resumo — triagem de relevância é etapa metodológica de uma revisão "
        "sistemática, não um detalhe de implementação.",
    )

    tab_triagem, tab_space, tab_isolation, tab_dupes = st.tabs(
        [
            "Triagem de relevância",
            "Espaço semântico e temas",
            "Isolamento semântico",
            "Quase-duplicatas",
        ],
        on_change="rerun",
        key="semantics_primary_tab",
    )

    if tab_triagem.open:
        with tab_triagem:
            _relevance_screening(scored)
    elif tab_space.open:
        with tab_space:
            _semantic_map(scored)
            _themes(scored)
    elif tab_isolation.open:
        with tab_isolation:
            _semantic_novelty_panel(scored)
    elif tab_dupes.open:
        with tab_dupes:
            _duplicates()


def _relevance_screening(scored: pd.DataFrame) -> None:
    # The margin needs `offtopic_score`, written by the contrastive anchor; a
    # database whose last `semantic` run predates it still gets the old view.
    has_margin = "relevance_margin" in scored.columns and scored["relevance_margin"].notna().any()
    if not has_margin:
        _legacy_relevance_screening(scored)
        return

    st.subheader("Distribuição da margem de relevância")
    margin = scored["relevance_margin"]
    low = scored[margin < 0]

    # Articles without an abstract have a title-only vector -- their score
    # comes from much weaker signal and should not influence percentile stats.
    has_abstract_col = "has_abstract" in scored.columns
    title_only = (
        scored[~scored["has_abstract"].astype(bool)] if has_abstract_col else scored.iloc[0:0]
    )
    stats_base = scored[scored["has_abstract"].astype(bool)] if has_abstract_col else scored
    stats_margin = stats_base["relevance_margin"] if not stats_base.empty else margin

    metrics = [
        ("📄 Artigos com score", f"{len(scored):,}", None),
        ("📊 Margem mediana", f"{stats_margin.median():+.3f}", None),
        (
            "🚩 Fora do escopo (margem < 0)",
            f"{len(low):,}",
            f"{100 * len(low) / len(scored):.1f}% do corpus",
        ),
    ]
    if len(title_only) > 0:
        metrics.append(
            (
                "⚠️ Sem resumo (título-only)",
                f"{len(title_only):,}",
                "excluídos das estatísticas de margem",
            )
        )
    metric_row(metrics)

    fig = px.histogram(
        scored,
        x="relevance_margin",
        nbins=60,
        title="Quanto cada artigo pende para o tema da revisão, e não para logística",
        color_discrete_sequence=[CATEGORICAL_PALETTE[0]],
    )
    fig.update_layout(
        xaxis_title="Margem (relevância − proximidade a logística)",
        yaxis_title="Quantidade de artigos",
        showlegend=False,
    )
    render_chart(
        fig,
        caption="Cada resumo é comparado com **duas** âncoras: o tema da revisão e a leitura "
        "logística da mesma busca. A margem é a diferença, e o **zero é o corte**: à esquerda dele "
        "estão os artigos que o próprio texto coloca mais perto de cadeia de suprimentos do que de "
        "redes de distribuição de energia. Com uma âncora só, as duas distribuições se sobrepunham e "
        "qualquer percentil descartava também trabalho dentro do escopo. Para aplicar o corte a "
        "**todos** os gráficos, use o filtro na barra lateral.",
    )

    st.divider()
    st.subheader(f"Fora do escopo — {len(low):,} artigos para revisão manual")
    st.caption(
        "Ordenados da margem mais negativa para a menos negativa. O score é um auxílio à triagem, "
        "não um veredito: revise antes de descartar."
    )
    review = low.sort_values("relevance_margin").head(TOP_REVIEW_ROWS).copy()
    for column in ("relevance_margin", "relevance_score"):
        review[column] = review[column].round(3)
    if "has_abstract" in review.columns:
        review["nota"] = review["has_abstract"].apply(lambda x: "" if x else "⚠️ título-only")
    display_cols = [
        "relevance_margin",
        "relevance_score",
        "theme_label",
        "title",
        "year",
        "venue",
        "source",
        "doi",
    ]
    if "nota" in review.columns:
        display_cols.insert(0, "nota")
    article_table(
        review,
        display_cols,
        download_key="fora_do_escopo",
    )

    st.divider()
    st.subheader("🤖 Triagem Assistida por Active Learning (Amostragem por Incerteza)")
    st.caption(
        "Artigos onde a margem contrastante está mais próxima de zero (|Δ| ≈ 0) representam a "
        "fronteira de decisão de máxima ambiguidade. Priorizar a inspeção manual destes casos acelera o "
        "refinamento da triagem sistemática (SLR) com o menor esforço de leitura humana."
    )
    uncertain = scored[scored["relevance_margin"].notna()].copy()
    uncertain["abs_margin"] = uncertain["relevance_margin"].abs()
    uncertain = uncertain.sort_values("abs_margin").head(25)
    for col in ("relevance_margin", "relevance_score"):
        uncertain[col] = uncertain[col].round(3)
    article_table(
        uncertain,
        [
            "relevance_margin",
            "relevance_score",
            "theme_label",
            "title",
            "year",
            "venue",
            "source",
            "doi",
        ],
        download_key="active_learning_incerteza",
    )

    _screening_calibration_panel(scored)


def _screening_calibration_panel(scored: pd.DataFrame) -> None:
    """Collect reviewed CSVs in memory and report held-out threshold evidence."""
    st.divider()
    st.subheader("Calibração com revisão humana")
    st.caption(
        "A fila por incerteza prioriza leitura; a amostra estratificada abaixo serve a outra "
        "pergunta: estimar o desempenho do limiar em toda a faixa de margens. O dashboard não "
        "grava rótulos nem aplica exclusões automaticamente."
    )

    sample = generate_stratified_screening_sample(scored, n_samples=100, seed=42)
    st.download_button(
        "Baixar amostra estratificada para revisão",
        data=sample.to_csv(index=False).encode("utf-8"),
        file_name="screening_review_sample.csv",
        mime="text/csv",
        key="screening_review_sample",
        width="stretch",
    )
    uploaded = st.file_uploader(
        "Enviar decisões revisadas (CSV long-form)",
        type=["csv"],
        accept_multiple_files=True,
        key="screening_review_files",
        help=(
            "Campos obrigatórios: doi e manual_label. Use include/1, exclude/0 ou uncertain; "
            "reviewer e protocol_version são recomendados."
        ),
    )
    if not uploaded:
        st.info(
            "Envie decisões independentes dos revisores para calcular concordância e validar "
            "um limiar candidato."
        )
        return

    frames: list[pd.DataFrame] = []
    for file in uploaded:
        try:
            frames.append(pd.read_csv(file))
        except (pd.errors.ParserError, UnicodeDecodeError, ValueError) as exc:
            st.error(f"Não foi possível ler `{file.name}`: {exc}")
    if not frames:
        return

    labels, issues = validate_review_labels(
        pd.concat(frames, ignore_index=True),
        known_dois=set(scored["doi"].dropna().astype(str)),
    )
    if not issues.empty:
        n_errors = int(issues["severity"].eq("error").sum())
        n_warnings = int(issues["severity"].eq("warning").sum())
        message = f"Auditoria do arquivo: {n_errors} erro(s) e {n_warnings} aviso(s)."
        st.error(message) if n_errors else st.warning(message)
        st.dataframe(issues, hide_index=True, width="stretch")
    if labels.empty:
        return

    agreement = reviewer_agreement(labels)
    if not agreement.empty:
        st.markdown("**Concordância entre revisores**")
        st.dataframe(agreement.round(3), hide_index=True, width="stretch")
        if agreement["status"].ne("ok").any():
            st.caption(
                "κ só é reportado com pelo menos 20 decisões binárias compartilhadas e presença "
                "das duas classes; os demais pares permanecem como suporte insuficiente."
            )

    resolved = resolve_review_consensus(labels)
    n_resolved = int(resolved["resolved"].sum())
    n_disagreement = int(resolved["resolution"].eq("disagreement").sum())
    n_single = int(resolved["resolution"].eq("single_reviewer").sum())
    metric_row(
        [
            ("Decisões resolvidas", f"{n_resolved:,}", None),
            ("Divergências pendentes", f"{n_disagreement:,}", None),
            ("Rótulos de um revisor", f"{n_single:,}", "evidência provisória"),
        ]
    )

    calibration = calibrate_screening_threshold(resolved, scored, seed=42)
    if not calibration["valid"]:
        st.warning(
            "Ainda não há suporte para validação holdout. São necessárias ao menos 40 decisões "
            "resolvidas, com 10 inclusões e 10 exclusões. Nenhum limiar é recomendado."
        )
        return

    threshold = float(calibration["threshold"])
    metrics = calibration["metrics"]
    intervals = calibration["confidence_intervals"]

    def _metric_with_ci(name: str) -> str:
        low_ci, high_ci = intervals[name]
        return f"{metrics[name]:.1%} (IC95% {low_ci:.1%}–{high_ci:.1%})"

    st.success(
        f"Limiar candidato: margem ≥ {threshold:+.3f}. Ajustado em "
        f"{calibration['n_calibration']} decisões e avaliado uma vez em "
        f"{calibration['n_holdout']} decisões holdout."
    )
    metric_row(
        [
            ("Sensibilidade", _metric_with_ci("recall"), f"FN: {metrics['fn']}"),
            ("Especificidade", _metric_with_ci("specificity"), None),
            ("Precisão", _metric_with_ci("precision"), None),
            ("F2", _metric_with_ci("f2"), "prioriza sensibilidade"),
            ("Redução de carga", _metric_with_ci("workload_reduction"), None),
        ]
    )

    curve = calibration["curve"]
    fig = px.line(
        curve.sort_values("recall"),
        x="recall",
        y="precision",
        markers=True,
        hover_data={"threshold": ":+.3f", "specificity": ":.1%"},
        labels={
            "recall": "Sensibilidade",
            "precision": "Precisão",
            "threshold": "Limiar",
            "specificity": "Especificidade",
        },
        title="Precisão e sensibilidade no conjunto holdout",
    )
    fig.add_scatter(
        x=[metrics["recall"]],
        y=[metrics["precision"]],
        mode="markers",
        marker={"size": 13, "symbol": "diamond", "color": TREND_DOWN_COLOR},
        name="Limiar candidato",
    )
    fig.update_layout(xaxis_tickformat=".0%", yaxis_tickformat=".0%")
    render_chart(
        fig,
        caption="O ponto destacado é evidência de validação, não autorização para exclusão "
        "automática. Intervalos amplos indicam necessidade de ampliar a revisão humana.",
    )


def _legacy_relevance_screening(scored: pd.DataFrame) -> None:
    """The percentile view, for a database that predates the contrastive anchor."""
    st.subheader("Distribuição do score de relevância")
    st.info(
        "Esta camada foi gerada antes da âncora contrastiva — rode `--stage semantic` para usar a "
        "margem, cujo corte em zero substitui o percentil abaixo."
    )
    threshold = float(scored["relevance_score"].quantile(LOW_RELEVANCE_PERCENTILE / 100))
    low = scored[scored["relevance_score"] < threshold]

    metric_row(
        [
            ("📄 Artigos com score", f"{len(scored):,}", None),
            ("📉 Score mediano", f"{scored['relevance_score'].median():.3f}", None),
            (
                f"🚩 Abaixo do percentil {LOW_RELEVANCE_PERCENTILE}",
                f"{len(low):,}",
                f"corte em {threshold:.3f}",
            ),
        ]
    )

    fig = px.histogram(
        scored,
        x="relevance_score",
        nbins=60,
        title="Quão perto do tema da revisão está cada artigo",
        labels={"relevance_score": "Score de relevância (cosseno)", "count": "Artigos"},
        color_discrete_sequence=[CATEGORICAL_PALETTE[0]],
    )
    fig.add_vline(
        x=threshold,
        line_dash="dash",
        line_color=TREND_DOWN_COLOR,
        annotation_text=f"percentil {LOW_RELEVANCE_PERCENTILE}",
    )
    fig.update_layout(
        xaxis_title="Score de relevância (1,0 = idêntico ao tema-âncora)",
        yaxis_title="Quantidade de artigos",
        showlegend=False,
    )
    render_chart(
        fig,
        caption="O score é o cosseno entre o resumo e um texto-âncora que descreve o escopo da "
        "revisão. A cauda à esquerda concentra os falsos positivos da busca.",
    )

    st.divider()
    st.subheader(f"Cauda de baixa relevância — {len(low):,} artigos para revisão manual")
    st.caption(
        "Ordenados do menos relevante para o mais relevante. O score é um auxílio à triagem, "
        "não um veredito: revise antes de descartar."
    )
    review = low.sort_values("relevance_score").head(TOP_REVIEW_ROWS).copy()
    review["relevance_score"] = review["relevance_score"].round(3)
    article_table(
        review,
        ["relevance_score", "theme_label", "title", "year", "venue", "source", "doi"],
        download_key="baixa_relevancia",
    )


def _semantic_map(scored: pd.DataFrame) -> None:
    st.subheader("Mapa semântico do corpus")
    if not require_columns(scored, ["map_x", "map_y", "theme_label"]):
        return

    plot_df = scored.dropna(subset=["map_x", "map_y"]).copy()

    # Prepara atributos descritivos e amigáveis para tooltip e legendas
    plot_df["title_display"] = plot_df["title"].fillna("Sem título")
    plot_df["title_hover"] = plot_df["title_display"].apply(
        lambda t: (
            "<br>".join([t[i : i + 65] for i in range(0, min(len(t), 195), 65)])
            + ("..." if len(t) > 195 else "")
        )
    )
    plot_df["venue_display"] = plot_df["venue"].fillna("Periódico não informado")
    plot_df["year_display"] = plot_df["year"].fillna("—").astype(str)
    source_map = {"ieee": "IEEE Xplore", "elsevier": "ScienceDirect (Elsevier)"}
    plot_df["source_display"] = plot_df["source"].map(source_map).fillna(plot_df["source"])
    plot_df["theme_display"] = plot_df["theme_label"].fillna("Sem tema atribuído")

    if "relevance_margin" in plot_df.columns:
        plot_df["status_display"] = np.where(
            plot_df["relevance_margin"] >= 0,
            "🟢 In-Scope (Energia / Relevante)",
            "🔴 Off-Topic (Logística / Geral)",
        )
    else:
        plot_df["status_display"] = "—"

    # Se relevance_margin não existir, faz fallback para relevance_score
    available_map = dict(MAP_COLOR_OPTIONS)
    if "relevance_margin" not in plot_df.columns and "relevance_score" in plot_df.columns:
        available_map["Relevância"] = "relevance_score"

    available = {
        label: column
        for label, column in available_map.items()
        if column in plot_df.columns and plot_df[column].notna().any()
    }

    ctrl_col0, ctrl_col1, ctrl_col2, ctrl_col3 = st.columns([2.3, 2.5, 2.5, 2.7])
    with ctrl_col0:
        alternative_projections = loaders.alternative_projections()
        projection_options = ["t-SNE", *alternative_projections]
        proj_choice = (
            st.segmented_control(
                "Projeção",
                options=projection_options,
                default="t-SNE",
                key="sem_proj_choice",
                help="Alterna a técnica de redução dimensional dos vetores 384D.",
            )
            or "t-SNE"
        )
    with ctrl_col1:
        choice = (
            st.segmented_control(
                "Colorir por",
                options=list(available),
                default="Tema",
                key="semantic_map_color",
                help="Altera a dimensão cromática dos pontos no espaço vetorial.",
            )
            or "Tema"
        )
    with ctrl_col2:
        legend_pos = (
            st.segmented_control(
                "Posição da legenda",
                options=["Lateral direita", "Inferior", "Ocultar"],
                default="Lateral direita",
                key="sem_map_legend_pos",
                help="Posicione a legenda para melhor legibilidade ou oculte para expandir o gráfico.",
            )
            or "Lateral direita"
        )
    with ctrl_col3:
        sub_c1, sub_c2 = st.columns(2)
        with sub_c1:
            show_density = st.checkbox(
                "🌊 Densidade (KDE)",
                value=False,
                key="sem_show_density",
                help="Sobrepõe curvas de nível de densidade de probabilidade bidimensional.",
            )
        with sub_c2:
            show_theme_labels = st.checkbox(
                "🏷️ Rótulos no mapa",
                value=True,
                key="sem_show_theme_labels",
                help="Exibe os nomes dos temas sobre os centróides medianos dos clusters em primeiro plano.",
            )

    if proj_choice in ("PCA 2D", "UMAP"):
        if (
            proj_choice in alternative_projections
            and not alternative_projections[proj_choice].empty
        ):
            alt_df = alternative_projections[proj_choice]
            plot_df = (
                plot_df.drop(columns=["map_x", "map_y"], errors="ignore")
                .merge(alt_df[["doi", "map_x", "map_y"]], on="doi", how="left")
                .dropna(subset=["map_x", "map_y"])
            )

    color_column = available[choice]
    continuous = choice in ("Relevância", "Ano")

    fig = px.scatter(
        plot_df,
        x="map_x",
        y="map_y",
        color=color_column,
        custom_data=[
            "title_hover",
            "venue_display",
            "year_display",
            "source_display",
            "theme_display",
            "relevance_score",
            "relevance_margin" if "relevance_margin" in plot_df.columns else "relevance_score",
            "status_display",
        ],
        color_continuous_scale=None if not continuous else ["#e34948", "#eda100", "#1baf7a"],
        color_discrete_sequence=CATEGORICAL_PALETTE,
        opacity=0.75,
    )
    fig.update_traces(
        marker=dict(size=6),
        hovertemplate=(
            "<b>%{customdata[0]}</b><br><br>"
            "🏛️ <b>Veículo:</b> %{customdata[1]}<br>"
            "📅 <b>Ano:</b> %{customdata[2]}  •  🏷️ <b>Base:</b> %{customdata[3]}<br>"
            "🎯 <b>Tema:</b> %{customdata[4]}<br>"
            "📊 <b>Relevância:</b> %{customdata[5]:.3f} (Margem Δ: %{customdata[6]:.3f})<br>"
            "🚦 <b>Triagem:</b> %{customdata[7]}"
            "<extra></extra>"
        ),
    )

    if show_density:
        fig.add_trace(
            go.Histogram2dContour(
                x=plot_df["map_x"],
                y=plot_df["map_y"],
                colorscale="Blues",
                reversescale=True,
                showscale=False,
                opacity=0.35,
                contours=dict(coloring="fill", showlabels=False),
                hoverinfo="skip",
            )
        )
        fig.data = (fig.data[-1],) + fig.data[:-1]

    if show_theme_labels:
        _add_theme_labels(fig, plot_df)

    # Título principal estruturado e eixos conceituais
    fig.update_layout(
        title=dict(
            text="Mapa Semântico do Corpus",
            subtitle=dict(
                text="Projeção 2D dos resumos vetoriais (embeddings) — proximidade espacial indica convergência temático-conceitual"
            ),
        ),
        xaxis=dict(title="Eixo Semântico 1 (Espaço Latente)", showticklabels=False),
        yaxis=dict(title="Eixo Semântico 2 (Espaço Latente)", showticklabels=False),
    )

    # Título dinâmico da legenda categórica
    if choice == "Tema":
        legend_title_text = "<b>Tema Temático</b><br><span style='font-size:10px; color:#888;'>Agrupamento de tópicos</span>"
    elif choice == "Fonte":
        legend_title_text = "<b>Base Indexadora</b>"
    else:
        legend_title_text = ""

    # Posicionamento da legenda ou barra de cores
    if continuous:
        if choice == "Relevância":
            fig.update_layout(
                coloraxis_colorbar=dict(
                    title=dict(
                        text="<b>Margem (Δ)</b><br><span style='font-size:10px;'>Off-topic < 0 < In-scope</span>",
                        side="top",
                    ),
                    tickvals=[-0.2, -0.1, 0.0, 0.1, 0.2],
                    ticktext=["-0.20", "-0.10", "0.00 Limiar", "+0.10", "+0.20"],
                )
            )
        elif choice == "Ano":
            fig.update_layout(
                coloraxis_colorbar=dict(
                    title=dict(text="<b>Ano de Publicação</b>", side="top"),
                    dtick=2,
                )
            )
        chart_margin = dict(l=40, r=120, t=95, b=40)
    else:
        if legend_pos == "Lateral direita":
            fig.update_layout(
                showlegend=True,
                legend=dict(
                    orientation="v",
                    yanchor="top",
                    y=1,
                    xanchor="left",
                    x=1.01,
                    title=dict(text=legend_title_text),
                    font=dict(size=11),
                    itemsizing="constant",
                    tracegroupgap=6,
                ),
            )
            chart_margin = dict(l=40, r=300, t=95, b=40)
        elif legend_pos == "Inferior":
            fig.update_layout(
                showlegend=True,
                legend=dict(
                    orientation="h",
                    yanchor="top",
                    y=-0.16,
                    xanchor="center",
                    x=0.5,
                    title=dict(text=legend_title_text),
                    font=dict(size=11),
                    itemsizing="constant",
                ),
            )
            chart_margin = dict(l=40, r=40, t=95, b=120)
        else:  # "Ocultar"
            fig.update_layout(showlegend=False)
            chart_margin = dict(l=40, r=40, t=95, b=40)

    render_chart(
        fig,
        height=650,
        margin=chart_margin,
        caption="Projeção t-SNE dos embeddings dos resumos, calculada no **mesmo espaço** em que os "
        "temas são descobertos — é isso que faz a cor de um ponto concordar com onde ele caiu. "
        "**Os eixos não têm significado numérico absoluto**: apenas a distância relativa entre pontos importa. "
        "O corpus forma um contínuo denso de planejamento de redes elétricas com **uma** ilha destacada "
        "(a de logística/pesquisa operacional, contaminação da busca pela palavra *distribution*); "
        "os temas são recortes desse contínuo revelados por agrupamento semântico denso.",
    )


def _add_theme_labels(fig, plot_df: pd.DataFrame) -> None:
    """Write each theme's name on the map as an annotation badge in front of points."""
    if "theme_label" not in plot_df.columns:
        return
    centroids = plot_df.groupby("theme_label")[["map_x", "map_y"]].median().reset_index()
    t = theme_tokens()
    labels = centroids["theme_label"].apply(
        lambda s: "<br>".join(s.split(" · ")) if " · " in s else s
    )
    for (_, row), label in zip(centroids.iterrows(), labels, strict=True):
        fig.add_annotation(
            x=row["map_x"],
            y=row["map_y"],
            text=f"<b>{label}</b>",
            showarrow=False,
            font=dict(size=10, color=t["chart_annotation"]),
            bgcolor=t["legend_bg"],
            bordercolor=t["legend_border"],
            borderwidth=1,
            borderpad=4,
            opacity=0.92,
        )


def _semantic_novelty_panel(scored: pd.DataFrame) -> None:
    st.subheader("Isolamento semântico no espaço de embeddings")
    st.caption(
        "Mede a distância média aos $k$-vizinhos mais próximos no espaço vetorial 384D. "
        "Valores altos indicam documentos isolados dos vizinhos do corpus. O indicador não mede, por si "
        "só, inovação ou interdisciplinaridade."
    )

    nov_df = loaders.semantic_novelty_scores()
    if nov_df.empty:
        st.info("Matriz de embeddings não disponível para calcular novidade semântica.")
        return

    merged = pd.merge(scored, nov_df, on="doi", how="inner")
    if merged.empty:
        st.info("Nenhum artigo com score de novidade correspondente.")
        return

    nov = merged["novelty_score"]
    p90 = float(nov.quantile(0.90))

    metric_row(
        [
            ("Isolamento mediano", f"{nov.median():.3f}", None),
            ("Limiar P90", f"{p90:.3f}", "10% mais isolados"),
            ("Maior isolamento", f"{nov.max():.3f}", None),
            ("Total avaliado", f"{len(merged):,}", "Embeddings 384D"),
        ]
    )

    fig = px.scatter(
        merged,
        x="novelty_score",
        y="relevance_score" if "relevance_score" in merged.columns else "novelty_score",
        color="theme_label" if "theme_label" in merged.columns else None,
        hover_data=["title", "year", "venue"],
        labels={
            "novelty_score": "Isolamento semântico (distância k-NN)",
            "relevance_score": "Relevância Temática",
            "theme_label": "Tema",
        },
        color_discrete_sequence=CATEGORICAL_PALETTE,
        title="Isolamento semântico e relevância no corpus",
    )
    fig.add_vline(x=p90, line_dash="dash", line_color="#eb6834", annotation_text="P90")
    fig.update_layout(height=480)
    render_chart(
        fig,
        caption="Pontos à direita estão mais distantes de seus vizinhos semânticos e merecem inspeção.",
    )

    st.markdown("##### Artigos semanticamente mais isolados")
    top_novel = merged.sort_values("novelty_score", ascending=False).head(20).copy()
    top_novel["novelty_score"] = top_novel["novelty_score"].round(3)
    if "relevance_score" in top_novel.columns:
        top_novel["relevance_score"] = top_novel["relevance_score"].round(3)
    cols = ["novelty_score", "relevance_score", "theme_label", "title", "year", "venue", "doi"]
    display_cols = [c for c in cols if c in top_novel.columns]
    article_table(top_novel, display_cols, download_key="top_novidade_semantica")


def _themes(scored: pd.DataFrame) -> None:
    st.subheader("Temas descobertos automaticamente")
    if not require_columns(scored, ["theme_label"]):
        return

    by_theme = (
        scored.groupby("theme_label")
        .agg(artigos=("doi", "size"), relevancia_media=("relevance_score", "mean"))
        .reset_index()
        .sort_values("artigos", ascending=False)
    )

    fig = px.bar(
        by_theme,
        x="artigos",
        y="theme_label",
        orientation="h",
        color="relevancia_media",
        color_continuous_scale=["#e34948", "#eda100", "#1baf7a"],
        title="Tamanho de cada tema e sua proximidade média ao escopo da revisão",
        labels={
            "artigos": "Quantidade de artigos",
            "theme_label": "",
            "relevancia_media": "Relevância",
        },
    )
    fig.update_layout(
        xaxis_title="Quantidade de artigos",
        yaxis_title="Tema",
        yaxis=dict(categoryorder="total ascending"),
    )
    render_chart(
        fig,
        caption="Os rótulos vêm dos termos que cada grupo usa **mais que o resto do corpus** — sem "
        "isso, todo tema sairia rotulado como 'distribution, power, planning'. A cor mostra que o "
        "tema de menor relevância média é justamente o de logística, confirmando por outro caminho "
        "o que o score de relevância indica.",
    )

    st.divider()
    st.subheader("Evolução temporal dos temas de pesquisa")
    if "year" not in scored.columns:
        st.info("Coluna 'year' não disponível nesta camada.")
        return
    yearly = scored.dropna(subset=["year", "theme_label"]).copy()
    yearly["year"] = pd.to_numeric(yearly["year"], errors="coerce")
    yearly = yearly.dropna(subset=["year"]).astype({"year": int})
    if yearly.empty:
        st.info("Sem anos válidos para esta análise.")
        return

    min_corpus_year = int(yearly["year"].min())
    max_corpus_year = int(yearly["year"].max())

    # Controles interativos em barra compacta
    c_time, c_metric, c_smooth = st.columns([3, 3, 3])
    with c_time:
        time_options = []
        if min_corpus_year < 2000:
            time_options.append("Desde 2000 (Recomendado)")
        if min_corpus_year < 1990:
            time_options.append("Desde 1990")
        time_options.append(f"Histórico Completo ({min_corpus_year}–{max_corpus_year})")

        time_choice = (
            st.segmented_control(
                "Horizonte temporal",
                options=time_options,
                default=time_options[0],
                key="theme_evol_horizon",
                help="Filtra o período de análise. O período moderno evita oscilações artificiais de anos esparsos antigos.",
            )
            or time_options[0]
        )

    with c_metric:
        metric_choice = (
            st.segmented_control(
                "Métrica",
                options=["Participação Relativa (%)", "Volume Absoluto (Artigos)"],
                default="Participação Relativa (%)",
                key="theme_evol_metric",
                help="Alterne entre a participação relativa de cada tema no ano e o volume real de publicações.",
            )
            or "Participação Relativa (%)"
        )

    with c_smooth:
        smooth_choice = (
            st.segmented_control(
                "Suavização",
                options=[
                    "Média móvel 3 anos (Suave)",
                    "Média móvel 5 anos",
                    "Sem suavização (Bruto)",
                ],
                default="Média móvel 3 anos (Suave)",
                key="theme_evol_smooth",
                help="Aplica média móvel centralizada para suavizar o ruído anual e revelar tendências estruturais.",
            )
            or "Média móvel 3 anos (Suave)"
        )

    # Determina o ano inicial conforme o filtro
    if "Desde 2000" in time_choice:
        start_year = max(2000, min_corpus_year)
    elif "Desde 1990" in time_choice:
        start_year = max(1990, min_corpus_year)
    else:
        start_year = min_corpus_year

    end_year = max_corpus_year
    filtered_yearly = yearly[(yearly["year"] >= start_year) & (yearly["year"] <= end_year)]

    # 1. Constrói o grid contínuo cartesiano Produto(Anos, Temas) = 0
    all_years = list(range(start_year, end_year + 1))
    all_themes = sorted(filtered_yearly["theme_label"].unique())
    if not all_years or not all_themes:
        st.info("Nenhum dado no período selecionado.")
        return

    grid = (
        pd.MultiIndex.from_product([all_years, all_themes], names=["year", "theme_label"])
        .to_frame()
        .reset_index(drop=True)
    )

    raw_counts = filtered_yearly.groupby(["year", "theme_label"]).size().reset_index(name="artigos")
    complete = pd.merge(grid, raw_counts, on=["year", "theme_label"], how="left").fillna(
        {"artigos": 0}
    )
    pivot = complete.pivot(index="year", columns="theme_label", values="artigos")

    # 2. Configura a janela de suavização
    if "3 anos" in smooth_choice:
        win = 3
    elif "5 anos" in smooth_choice:
        win = 5
    else:
        win = 1

    # 3. Calcula os valores conforme a métrica
    is_relative = "Relativa" in metric_choice
    if is_relative:
        row_sums = pivot.sum(axis=1).replace(0, 1)
        pct = pivot.div(row_sums, axis=0) * 100
        if win > 1:
            smoothed = pct.rolling(window=win, min_periods=1, center=True).mean()
            smoothed_sums = smoothed.sum(axis=1).replace(0, 1)
            smoothed = smoothed.div(smoothed_sums, axis=0) * 100
        else:
            smoothed = pct
        y_col = "percentual"
        y_title = "Participação no ano (%)"
        plot_df = smoothed.reset_index().melt(id_vars="year", value_name=y_col)
    else:
        if win > 1:
            smoothed = pivot.rolling(window=win, min_periods=1, center=True).mean()
        else:
            smoothed = pivot
        y_col = "volume"
        y_title = "Quantidade de artigos publicados"
        plot_df = smoothed.reset_index().melt(id_vars="year", value_name=y_col)

    # Associa contagem real não-suavizada para o tooltip
    raw_vol_map = complete.rename(columns={"artigos": "volume_real"})
    plot_df = pd.merge(plot_df, raw_vol_map, on=["year", "theme_label"], how="left")

    yearly_totals = filtered_yearly.groupby("year").size()
    plot_df["pct_real"] = plot_df.apply(
        lambda r: r["volume_real"] / yearly_totals.get(r["year"], 1) * 100,
        axis=1,
    )

    fig = px.area(
        plot_df.sort_values(["theme_label", "year"]),
        x="year",
        y=y_col,
        color="theme_label",
        line_shape="spline",
        custom_data=["volume_real", "pct_real"],
        color_discrete_sequence=CATEGORICAL_PALETTE,
    )

    hovertemplate = (
        "<b>%{fullData.name}</b><br>"
        "📅 <b>Ano:</b> %{x}<br>"
        + (
            "📊 <b>Participação (suavizada):</b> %{y:.1f}%<br>"
            if is_relative
            else "📚 <b>Volume (suavizado):</b> %{y:.1f} artigos<br>"
        )
        + "📚 <b>Volume real do ano:</b> %{customdata[0]:.0f} artigos (%{customdata[1]:.1f}%)"
        "<extra></extra>"
    )
    fig.update_traces(hovertemplate=hovertemplate)

    fig.update_layout(
        title=dict(
            text="Evolução Temporal dos Temas de Pesquisa",
            subtitle=dict(
                text="Atenção científica por tema ao longo dos anos de publicação (interpolação spline suavizada)"
            ),
        ),
        xaxis=dict(
            title="Ano de publicação",
            dtick=2 if (end_year - start_year) <= 20 else 5,
        ),
        yaxis=dict(
            title=y_title,
            range=[0, 100] if is_relative else None,
        ),
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.01,
            title=dict(text="<b>Tema Temático</b>"),
            font=dict(size=11),
            itemsizing="constant",
            tracegroupgap=6,
        ),
        hovermode="x unified",
    )

    chart_caption = (
        "Evolução temática com preenchimento contínuo e média móvel: elimina distorções de anos esparsos "
        "e revela para onde a atenção científica migrou. As curvas fluidas facilitam visualizar a emergência "
        "de tópicos como mobilidade elétrica e armazenamento distribuído."
        if is_relative
        else "Volume absoluto de artigos por tema em cada ano: revela o crescimento do corpus como um todo "
        "e a expansão acelerada da produção científica nas últimas duas décadas."
    )

    render_chart(
        fig,
        height=580,
        margin=dict(l=40, r=300, t=95, b=40),
        caption=chart_caption,
    )

    st.divider()
    st.subheader("🧭 Deriva Semântica Temporal dos Temas (Thematic Drift)")
    st.caption(
        "Mudança do centro de massa de cada tema ao longo de três épocas históricas "
        "(1990–2010, 2011–2018, 2019–2026). As trajetórias revelam como o foco "
        "conceitual de cada linha de pesquisa se deslocou no plano semântico."
    )
    if (
        "map_x" in scored.columns
        and "map_y" in scored.columns
        and "year" in scored.columns
        and "theme_id" in scored.columns
    ):
        from lake_research_map.transform.semantics import compute_temporal_drift

        valid_drift = scored.dropna(subset=["map_x", "map_y", "year", "theme_id"]).copy()
        valid_drift["year"] = pd.to_numeric(valid_drift["year"], errors="coerce")
        valid_drift = valid_drift.dropna(subset=["year"])
        windows = [(1990, 2010), (2011, 2018), (2019, 2026)]
        drift_data = compute_temporal_drift(
            valid_drift[["map_x", "map_y"]].to_numpy(dtype=float),
            valid_drift["theme_id"].to_numpy(dtype=int),
            valid_drift["year"].to_numpy(dtype=int),
            windows,
        )
        drift_rows = []
        theme_names = dict(zip(valid_drift["theme_id"], valid_drift["theme_label"], strict=False))
        for t_id, pts in drift_data.items():
            t_name = theme_names.get(t_id, f"Tema {t_id}")
            for p in pts:
                drift_rows.append(
                    {
                        "Tema": t_name,
                        "Época": p["name"],
                        "map_x": p["x"],
                        "map_y": p["y"],
                        "Artigos": p["count"],
                    }
                )
        if drift_rows:
            drift_df = pd.DataFrame(drift_rows)
            fig_drift = px.line(
                drift_df,
                x="map_x",
                y="map_y",
                color="Tema",
                text="Época",
                markers=True,
                title="Trajetória dos centróides temáticos no espaço bidimensional",
                color_discrete_sequence=CATEGORICAL_PALETTE,
            )
            fig_drift.update_traces(textposition="top center")
            fig_drift.update_layout(
                xaxis=dict(title="Dimensão 1", showticklabels=False),
                yaxis=dict(title="Dimensão 2", showticklabels=False),
            )
            render_chart(
                fig_drift,
                caption="As linhas conectam os centróides médios em cada época cronológica.",
            )


def _duplicates() -> None:
    st.subheader("Quase-duplicatas que a deduplicação por DOI não pegou")
    pairs = loaders.duplicate_pairs()
    overrides = loaders.duplicate_overrides()

    _, articles_df = loaders.articles()
    titles = (
        articles_df.set_index("doi")["title"]
        if "doi" in articles_df.columns and "title" in articles_df.columns
        else pd.Series(dtype="object")
    )
    st.caption(
        "O DOI é a única chave confiável de deduplicação deste corpus (ver `CLAUDE.md`), então o "
        "mesmo trabalho publicado sob dois DOIs pode sobreviver como dois registros. Estes pares "
        "foram detectados pela similaridade do resumo e aguardam uma decisão humana. O dashboard "
        "permanece somente leitura; registre a decisão com `lake-research-map duplicates`."
    )

    st.markdown("##### Pendentes de revisão")
    if pairs.empty:
        st.success("Nenhum par de resumos quase idênticos aguarda revisão.")
    else:
        table = pairs.copy()
        table["Título A"] = table["doi_a"].map(titles)
        table["Título B"] = table["doi_b"].map(titles)
        table["Similaridade"] = table["similarity"].round(4)
        st.dataframe(
            table[["Similaridade", "Título A", "Título B", "doi_a", "doi_b"]].sort_values(
                "Similaridade", ascending=False
            ),
            hide_index=True,
            width="stretch",
        )
        st.code(
            "uv run lake-research-map duplicates merge --canonical-doi DOI --duplicate-doi DOI "
            '--reason "justificativa"\n'
            "uv run lake-research-map duplicates keep --doi-a DOI --doi-b DOI "
            '--reason "justificativa"',
            language="bash",
        )

    st.markdown("##### Histórico de decisões")
    if overrides.empty:
        st.info("Nenhuma decisão de quase-duplicata foi registrada.")
        return

    history = overrides.copy()
    history["Decisão"] = history["decision"].map({"merge": "Mesclar", "keep": "Manter separados"})
    history["DOI canônico"] = history["canonical_doi"].fillna("—")
    history["Justificativa"] = history["reason"]
    history["Atualizado em"] = history["updated_at"]
    st.dataframe(
        history[
            [
                "Decisão",
                "DOI canônico",
                "doi_a",
                "doi_b",
                "Justificativa",
                "Atualizado em",
            ]
        ].sort_values("Atualizado em", ascending=False),
        hide_index=True,
        width="stretch",
    )
