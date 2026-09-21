"""🧩 Qualidade e RAG — riqueza de metadados, cobertura de texto completo e chunks."""

from __future__ import annotations

import re

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from lake_research_map.dashboard import actions, loaders
from lake_research_map.dashboard.analytics import metadata_coverage_matrix, pdf_selection_bias
from lake_research_map.dashboard.charts import topn_hbar
from lake_research_map.dashboard.components import (
    article_table,
    hero_banner,
    metric_row,
    page_header,
    render_chart,
    require_columns,
)
from lake_research_map.dashboard.search import hybrid_search_rrf, semantic_search
from lake_research_map.dashboard.theme import (
    CATEGORICAL_PALETTE,
    CHART_HEIGHT,
    SOURCE_COLORS,
    TREND_DOWN_COLOR,
    theme_tokens,
)
from lake_research_map.transform.gold_articles import CHUNK_MAX_CHARS

SEARCH_DEMO_MAX_RESULTS = 10


def render() -> None:
    page_header(
        "🧩",
        "Qualidade e RAG",
        "Diagnóstico do corpus como fonte para RAG: metadados, texto completo e fragmentos (chunks).",
    )

    articles_df = loaders.require_articles()
    chunks_df = loaders.filtered_chunks()

    pdf_share = (
        articles_df["has_pdf"].fillna(False).astype(bool).mean()
        if "has_pdf" in articles_df.columns
        else 0.0
    )
    hero_banner(
        "Diagnóstico RAG",
        f"O corpus oferece <b>{pdf_share:.0%}</b> de cobertura de texto completo e "
        f"<b>{len(chunks_df):,}</b> chunks prontos para recuperação.",
    )

    tab_metadata, tab_content, tab_anomalies, tab_search = st.tabs(
        [
            "Metadados e cobertura",
            "Texto, chunks e embeddings",
            "Auditoria e anomalias",
            "Busca híbrida",
        ],
        on_change="rerun",
        key="quality_primary_tab",
    )

    if tab_metadata.open:
        with tab_metadata:
            _metadata_coverage(articles_df)
            _ieee_extras(articles_df)
    elif tab_content.open:
        with tab_content:
            _fulltext_coverage(articles_df, chunks_df)
            _pdf_selection_bias(articles_df)
            if _chunks_intro(chunks_df):
                content_view = st.segmented_control(
                    "Detalhamento dos chunks",
                    ["Tipos", "Tamanho", "Por artigo", "Embeddings"],
                    default="Tipos",
                    key="quality_chunk_view",
                )
                if content_view == "Tipos":
                    _chunk_type_pie(chunks_df)
                elif content_view == "Tamanho":
                    _chunk_length_histogram(chunks_df)
                elif content_view == "Por artigo":
                    _chunks_per_article(chunks_df)
                else:
                    _embedding_readiness(chunks_df)
    elif tab_anomalies.open:
        with tab_anomalies:
            _bibliometric_anomalies_audit(articles_df)
    elif tab_search.open:
        with tab_search:
            _search_demo(chunks_df)


def _ieee_extras(articles_df: pd.DataFrame) -> None:
    """Fields only the IEEE CSV export carries: country, online date, type, licence.

    Every number here is reported against the IEEE subset, never the whole
    corpus: Elsevier's .bib has no affiliation, no online date and no licence
    field at all, so a percentage over 1.831 articles would understate these by
    a factor of six and read as "missing data" rather than "not applicable".
    """
    st.subheader("🔷 Campos exclusivos da base IEEE")

    if "countries" not in articles_df.columns:
        st.info(
            "Colunas de enriquecimento IEEE ainda não existem nesta camada — rode "
            "`uv run lake-research-map --stage bronze` (e silver/gold) para populá-las."
        )
        return

    ieee_only = articles_df[
        articles_df["source"].eq("ieee")
        if "source" in articles_df.columns
        else articles_df.index.notna()
    ]
    n_ieee = len(ieee_only)
    if n_ieee == 0:
        st.info("Nenhum artigo da base IEEE no filtro atual.")
        return

    hero_banner(
        "Cobertura parcial, por natureza da fonte",
        f"Estes campos vêm do export CSV do IEEE Xplore, que a ScienceDirect não fornece. "
        f"A base é de <b>{n_ieee:,} artigos IEEE</b> — cerca de "
        f"{n_ieee / max(len(articles_df), 1):.0%} do corpus filtrado. "
        "Todos os percentuais abaixo usam esse denominador, não o corpus inteiro.",
    )

    countries = ieee_only["countries"].apply(lambda c: c if isinstance(c, list) else [])
    with_country = int(countries.apply(bool).sum())
    metric_row(
        [
            ("🔷 Artigos IEEE", f"{n_ieee:,}", None),
            ("🌍 Com país identificado", f"{with_country:,}", f"{with_country / n_ieee:.0%}"),
            (
                "🗓️ Com data online",
                f"{int(ieee_only['online_date'].notna().sum()):,}"
                if "online_date" in ieee_only.columns
                else "N/D",
                None,
            ),
        ]
    )

    sub_pais, sub_mes, sub_tipo = st.tabs(
        ["🌍 Países", "🗓️ Granularidade Mensal", "📰 Tipo & Licença"]
    )

    with sub_pais:
        exploded = countries.explode().dropna()
        if exploded.empty:
            st.info("Nenhuma afiliação com país identificável.")
        else:
            top = exploded.value_counts().head(15)
            fig = topn_hbar(
                top,
                title="Top 15 países por participação em artigos (base IEEE)",
                x_title="Artigos com ao menos um autor no país",
                y_title="País",
            )
            fig.update_traces(hovertemplate="<b>%{y}</b><br>%{x:,} artigos<extra></extra>")
            render_chart(
                fig,
                caption="Um artigo conta uma vez por país presente entre suas afiliações, então "
                "colaborações internacionais aparecem em mais de um país e a soma das barras "
                f"excede os {n_ieee:,} artigos. Extraído do último segmento de cada afiliação "
                "(`…, cidade, País`), o que também acerta o formato dos EUA (`…, UT, USA`).",
            )

    with sub_mes:
        if "online_date" not in ieee_only.columns or ieee_only["online_date"].isna().all():
            st.info("Coluna 'online_date' indisponível.")
        else:
            dated = ieee_only.dropna(subset=["online_date"]).copy()
            dated["mes"] = pd.to_datetime(dated["online_date"]).dt.to_period("M").dt.to_timestamp()
            by_month = dated.groupby("mes").size().reset_index(name="artigos")
            fig = px.bar(
                by_month,
                x="mes",
                y="artigos",
                title="Publicações por mês de disponibilização online (base IEEE)",
                labels={"mes": "Mês", "artigos": "Artigos"},
                color_discrete_sequence=[SOURCE_COLORS["ieee"]],
            )
            fig.update_layout(
                xaxis_title="Mês de publicação online", yaxis_title="Quantidade de artigos"
            )
            render_chart(
                fig,
                caption="`Online Date` é o **único** campo do corpus com resolução mais fina que o "
                "ano — em todo o resto do dashboard só o ano sobrevive. Serve para ver sazonalidade "
                "e a defasagem entre publicação online e edição formal.",
            )

    with sub_tipo:
        col_tipo, col_lic = st.columns(2)
        with col_tipo:
            if "document_type" in ieee_only.columns and ieee_only["document_type"].notna().any():
                counts = ieee_only["document_type"].dropna().value_counts()
                fig = px.pie(
                    names=counts.index,
                    values=counts.to_numpy(),
                    title="Tipo de veículo (Document Identifier)",
                    color_discrete_sequence=CATEGORICAL_PALETTE,
                )
                render_chart(
                    fig,
                    caption="Revela que o IEEE Xplore hospeda também periódicos de outras "
                    "editoras (CSEE, SGEPRI), além de revistas e capítulos de livro.",
                )
            else:
                st.info("Coluna 'document_type' indisponível.")
        with col_lic:
            if "license" in ieee_only.columns and ieee_only["license"].notna().any():
                counts = ieee_only["license"].dropna().value_counts()
                oa = int(counts.filter(like="CC").sum())
                fig = px.pie(
                    names=counts.index,
                    values=counts.to_numpy(),
                    title="Licença de publicação",
                    color_discrete_sequence=CATEGORICAL_PALETTE,
                )
                render_chart(
                    fig,
                    caption=f"{oa} artigos sob licença Creative Commons (acesso aberto) entre os "
                    f"{int(counts.sum())} com licença declarada.",
                )
            else:
                st.info("Coluna 'license' indisponível.")


def _embedding_readiness(chunks_df: pd.DataFrame) -> None:
    st.subheader("🧠 Prontidão de embeddings")
    total = len(chunks_df)
    with_embedding = (
        int(chunks_df["has_embedding"].sum()) if "has_embedding" in chunks_df.columns else 0
    )
    pending = total - with_embedding
    pct = (with_embedding / total * 100) if total else 0.0
    t = theme_tokens()

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=pct,
            number={"suffix": "%", "font": {"color": t["chart_annotation"]}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": t["chart_text"]},
                "bar": {"color": CATEGORICAL_PALETTE[0]},
                "bgcolor": t["chart_bg"],
                "borderwidth": 0,
            },
            title={
                "text": "% de chunks com embedding gerado",
                "font": {"color": t["chart_text"], "size": 14},
            },
        )
    )
    if with_embedding and "embed_model" in chunks_df.columns:
        model_used = chunks_df["embed_model"].dropna().mode()
        model_caption = (
            f"Gerados com `{model_used.iat[0]}` — a etapa `embed` do pipeline "
            "(`transform/embeddings.py`) roda 100% local via `fastembed`, sem chave de API."
        )
    else:
        model_caption = (
            "A coluna `embedding` existe no esquema (`gold_models.Chunk`) mas nenhum chunk foi processado "
            "ainda — este é o principal bloqueio para usar o corpus como base de um RAG real."
        )
    render_chart(
        fig,
        height=260,
        caption=f"{with_embedding:,} de {total:,} chunks têm embedding. {model_caption}",
    )

    # Per-chunk-type embedding breakdown
    if "chunk_type" in chunks_df.columns:
        by_type = (
            chunks_df.groupby("chunk_type")
            .agg(
                total=("chunk_type", "size"),
                embedded=("has_embedding", "sum")
                if "has_embedding" in chunks_df.columns
                else ("chunk_type", lambda x: 0),
            )
            .reset_index()
        )
        by_type["embedded"] = by_type["embedded"].astype(int)
        by_type["pendente"] = by_type["total"] - by_type["embedded"]
        by_type.columns = ["Tipo", "Total", "Com embedding", "Pendente"]
        st.dataframe(by_type, hide_index=True, width="stretch")

    if pending > 0:
        if st.button(
            f"🚀 Gerar embeddings agora ({pending:,} chunks pendentes)", key="generate_embeddings"
        ):
            _run_embedding_generation(pending)


def _run_embedding_generation(pending: int) -> None:
    progress_bar = st.progress(0.0, text=f"Gerando embeddings (0/{pending})...")

    def on_progress(done: int, total_pending: int) -> None:
        progress_bar.progress(
            min(done / total_pending, 1.0),
            text=f"Gerando embeddings ({done:,}/{total_pending:,})...",
        )

    with st.spinner("Carregando o modelo de embeddings (primeira execução baixa os pesos)..."):
        stats = actions.run_embedding_generation(on_progress=on_progress)

    st.success(f"{stats['embedded']:,} chunks embedados. Atualizando a página...")
    st.cache_data.clear()
    st.rerun()


def _fulltext_coverage(articles_df: pd.DataFrame, chunks_df: pd.DataFrame) -> None:
    st.subheader("📄 Cobertura de texto completo")
    if not require_columns(articles_df, ["has_pdf"]):
        return

    n_total = len(articles_df)
    n_with_pdf = int(articles_df["has_pdf"].fillna(False).astype(bool).sum())

    dois_with_fulltext = set()
    if not chunks_df.empty and "chunk_type" in chunks_df.columns and "doi" in chunks_df.columns:
        dois_with_fulltext = set(
            chunks_df.loc[chunks_df["chunk_type"] == "fulltext", "doi"].dropna()
        )
    n_with_fulltext_chunks = (
        int(articles_df["doi"].isin(dois_with_fulltext).sum())
        if "doi" in articles_df.columns
        else 0
    )
    n_pdf_no_chunks = n_with_pdf - n_with_fulltext_chunks

    funnel_df = pd.DataFrame(
        {
            "stage": ["Todos os artigos", "Com PDF vinculado", "Com chunks de texto completo"],
            "count": [n_total, n_with_pdf, n_with_fulltext_chunks],
        }
    )
    col_funnel, col_metrics = st.columns([2, 1])
    with col_funnel:
        fig = px.funnel(
            funnel_df,
            x="count",
            y="stage",
            title="Funil de disponibilidade de texto completo",
            labels={"count": "Quantidade de artigos", "stage": "Etapa"},
        )
        fig.update_traces(
            marker_color=[CATEGORICAL_PALETTE[0], CATEGORICAL_PALETTE[3], CATEGORICAL_PALETTE[2]]
        )
        render_chart(fig)
    with col_metrics:
        metric_row(
            [
                (
                    "📎 Artigos com PDF",
                    f"{n_with_pdf:,}",
                    f"{n_with_pdf / n_total:.1%}" if n_total else None,
                ),
            ]
        )
        metric_row(
            [
                (
                    "⚠️ PDFs sem chunks de texto",
                    f"{max(n_pdf_no_chunks, 0):,}",
                    "extração falhou ou PDF vazio" if n_pdf_no_chunks > 0 else "todos processados",
                ),
            ]
        )
    st.caption(
        f"Apenas {n_with_pdf / n_total:.1%} dos artigos têm PDF — os PDFs do corpus vêm exclusivamente dos "
        "bulk-downloads do IEEE, então a Elsevier tem 0% de cobertura de texto completo. A extração de PDF "
        "(`gold_articles.py`) engole falhas silenciosamente, então um PDF que falhou é indistinguível de um "
        "PDF sem texto extraível — o card '⚠️ PDFs sem chunks' acima é o sinal disso."
    )


def _pdf_selection_bias(articles_df: pd.DataFrame) -> None:
    """Test whether the full-text subset differs materially from the corpus."""
    st.subheader("Viés de disponibilidade de PDF")
    bias = pdf_selection_bias(articles_df)
    finite = bias.dropna(subset=["smd"])
    if finite.empty:
        st.info("São necessários artigos com e sem PDF e ao menos duas observações por grupo.")
        return

    labels = {
        "year": "Ano",
        "citation_count": "Citações",
        "reference_count": "Referências",
        "team_size": "Tamanho da equipe",
        "abstract_chars": "Tamanho do resumo",
        "keyword_count": "Quantidade de palavras-chave",
    }
    bias = bias.copy()
    bias["Métrica"] = bias["metric"].map(labels)
    finite = finite.copy()
    finite["Métrica"] = finite["metric"].map(labels)
    finite["erro_superior"] = finite["ci_high"] - finite["smd"]
    finite["erro_inferior"] = finite["smd"] - finite["ci_low"]
    colors = [
        TREND_DOWN_COLOR if abs(value) >= 0.1 else CATEGORICAL_PALETTE[0] for value in finite["smd"]
    ]
    fig = go.Figure(
        go.Scatter(
            x=finite["smd"],
            y=finite["Métrica"],
            mode="markers",
            marker={"size": 11, "color": colors},
            error_x={
                "type": "data",
                "array": finite["erro_superior"],
                "arrayminus": finite["erro_inferior"],
                "visible": True,
            },
            customdata=finite[["n_pdf", "n_no_pdf", "transform"]],
            hovertemplate=(
                "<b>%{y}</b><br>SMD: %{x:.2f}<br>Com PDF: %{customdata[0]}"
                "<br>Sem PDF: %{customdata[1]}<br>Transformação: %{customdata[2]}<extra></extra>"
            ),
        )
    )
    fig.add_vline(x=0, line_color=theme_tokens()["grid"], line_width=1)
    fig.add_vrect(x0=-0.1, x1=0.1, fillcolor=CATEGORICAL_PALETTE[0], opacity=0.08, line_width=0)
    fig.update_layout(
        title="Diferença padronizada: artigos com PDF menos artigos sem PDF",
        xaxis_title="Diferença média padronizada (IC bootstrap de 95%)",
        yaxis_title="Métrica",
        showlegend=False,
    )
    render_chart(
        fig,
        caption="Valores positivos indicam média maior no subconjunto com PDF. A faixa |SMD| < 0,10 "
        "representa desequilíbrio pequeno. A comparação é descritiva: disponibilidade de PDF depende "
        "da fonte e não sustenta interpretação causal.",
    )

    audit = bias[
        ["Métrica", "transform", "n_pdf", "n_no_pdf", "smd", "ci_low", "ci_high", "status"]
    ].rename(
        columns={
            "transform": "Transformação",
            "n_pdf": "n com PDF",
            "n_no_pdf": "n sem PDF",
            "smd": "SMD",
            "ci_low": "IC 2,5%",
            "ci_high": "IC 97,5%",
            "status": "Status",
        }
    )
    st.dataframe(audit.round(3), hide_index=True, width="stretch")


def _metadata_coverage(articles_df: pd.DataFrame) -> None:
    """Show one non-redundant view of common-field completeness by source."""
    if not require_columns(articles_df, ["source"]):
        return
    coverage = metadata_coverage_matrix(articles_df)
    if coverage.empty:
        st.info("Não há campos comuns suficientes para medir a cobertura.")
        return

    labels = {
        "doi": "DOI",
        "title": "Título",
        "year": "Ano",
        "venue": "Periódico/evento",
        "authors": "Autores",
        "abstract": "Resumo",
        "keywords": "Palavras-chave",
        "citation_count": "Citações",
        "reference_count": "Referências",
        "has_pdf": "PDF disponível",
    }
    coverage["Campo"] = coverage["field"].map(labels).fillna(coverage["field"])
    matrix = coverage.pivot(index="Campo", columns="source", values="coverage")
    st.subheader("Completude dos campos comuns por fonte")
    fig = px.imshow(
        matrix,
        zmin=0,
        zmax=1,
        text_auto=".0%",
        aspect="auto",
        color_continuous_scale="Blues",
        labels={"x": "Fonte", "y": "Campo", "color": "Cobertura"},
    )
    fig.update_layout(xaxis_title="Fonte", yaxis_title="Campo")
    render_chart(
        fig,
        caption="Cada célula usa como denominador apenas os artigos da fonte indicada. PDF significa "
        "disponibilidade, não simples existência da coluna. Campos exclusivos do IEEE são auditados "
        "separadamente abaixo para evitar classificar ausência estrutural da Elsevier como falha.",
    )
    exact = coverage[["Campo", "source", "n_total", "n_present", "coverage"]].rename(
        columns={
            "source": "Fonte",
            "n_total": "Total",
            "n_present": "Presentes",
            "coverage": "Cobertura",
        }
    )
    st.dataframe(
        exact,
        hide_index=True,
        width="stretch",
        column_config={"Cobertura": st.column_config.ProgressColumn(format="percent")},
    )


def _chunks_intro(chunks_df: pd.DataFrame) -> bool:
    """Guard + shared summary metric row. Returns True if there's data to show."""
    st.subheader("🧩 Fragmentos (chunks) preparados para embedding")
    if chunks_df.empty:
        st.info(
            "`lit_gold.chunks` ainda não foi populada — execute o pipeline completo "
            "(botão na barra lateral ou `uv run lake-research-map --stage all`)."
        )
        return False

    if not require_columns(
        chunks_df, ["chunk_type"], "A tabela de chunks não possui o campo `chunk_type`."
    ):
        return False

    n_abstract = int((chunks_df["chunk_type"] == "abstract").sum())
    n_fulltext = int((chunks_df["chunk_type"] == "fulltext").sum())
    n_dois = chunks_df["doi"].nunique() if "doi" in chunks_df else 0
    n_ft_dois = (
        chunks_df.loc[chunks_df["chunk_type"] == "fulltext", "doi"].nunique()
        if "doi" in chunks_df.columns
        else 0
    )

    metric_row(
        [
            ("🧩 Total de fragmentos (chunks)", f"{len(chunks_df):,}", None),
            ("📝 Chunks de resumo", f"{n_abstract:,}", None),
            ("📄 Chunks de texto completo", f"{n_fulltext:,}", f"{n_ft_dois} artigos com PDF"),
            (
                "📄 DOIs distintos com fragmentos",
                f"{n_dois:,}",
                None,
            ),
        ]
    )

    if n_fulltext > 0 and n_dois > 0:
        st.caption(
            f"⚠️ **Viés de cobertura**: texto completo cobre {n_ft_dois} de {n_dois} artigos "
            f"({100 * n_ft_dois / n_dois:.1f}%), mas gera {n_fulltext:,} de {len(chunks_df):,} "
            f"chunks ({100 * n_fulltext / len(chunks_df):.1f}%). Estatísticas por chunk refletem "
            f"desproporcionalmente esses {100 * n_ft_dois / n_dois:.1f}% do corpus."
        )
    return True


def _chunk_type_pie(chunks_df: pd.DataFrame) -> None:
    by_type = chunks_df["chunk_type"].value_counts().rename_axis("type").reset_index(name="count")
    by_type["label_pt"] = by_type["type"].map(
        {"abstract": "Resumo (Abstract)", "fulltext": "Texto Completo (Fulltext)"}
    )
    fig = px.pie(by_type, names="label_pt", values="count", title="Proporção de chunks por tipo")
    fig.update_traces(
        texttemplate="<b>%{label}</b><br><b>%{value:,} (%{percent})</b>",
        hovertemplate="<b>%{label}</b>: %{value:,} chunks (%{percent})<extra></extra>",
    )
    render_chart(
        fig,
        height=CHART_HEIGHT,
        caption="`abstract` = 1 fragmento por artigo (título + palavras-chave + resumo); `fulltext` = "
        "fragmentos extraídos diretamente dos PDFs dos artigos disponíveis.",
    )


def _chunk_length_histogram(chunks_df: pd.DataFrame) -> None:
    if not require_columns(chunks_df, ["char_len"]):
        return
    length_df = chunks_df.dropna(subset=["char_len"])
    # In overlay mode the last category painted sits on top -- draw
    # the smaller-volume chunk type last so it isn't hidden behind
    # the larger one wherever their bins overlap.
    type_order = (
        length_df["chunk_type"].value_counts().sort_values(ascending=False).index.tolist()
        if "chunk_type" in length_df.columns
        else None
    )
    fig = px.histogram(
        length_df,
        x="char_len",
        color="chunk_type" if "chunk_type" in length_df.columns else None,
        barmode="overlay",
        opacity=0.75,
        nbins=40,
        category_orders={"chunk_type": type_order} if type_order else None,
        title="Tamanho dos chunks (caracteres) por tipo",
        labels={"char_len": "Comprimento em caracteres", "chunk_type": "Tipo"},
    )
    fig.update_traces(hovertemplate="Tamanho: ~%{x} caracteres<br>Chunks: %{y:,}<extra></extra>")
    fig.update_layout(
        xaxis_title=f"Caracteres por fragmento (teto de chunking: {CHUNK_MAX_CHARS:,})",
        yaxis_title="Quantidade de chunks",
    )
    render_chart(
        fig,
        height=CHART_HEIGHT,
        caption=f"O eixo mostra até o teto de {CHUNK_MAX_CHARS:,} caracteres usado ao dividir o "
        "texto completo (`gold_articles.CHUNK_MAX_CHARS`); fragmentos excessivamente curtos perdem "
        "contexto semântico.",
    )


def _chunks_per_article(chunks_df: pd.DataFrame) -> None:
    if "doi" not in chunks_df.columns:
        st.info("Coluna 'doi' não disponível nesta camada.")
        return

    chunks_per_article = chunks_df.groupby("doi").size()
    fulltext_dois = (
        chunks_df.loc[chunks_df["chunk_type"] == "fulltext", "doi"].unique()
        if "chunk_type" in chunks_df.columns
        else []
    )
    fulltext_chunks_per_article = chunks_per_article.reindex(fulltext_dois)
    if fulltext_chunks_per_article.empty:
        st.info("Nenhum artigo com chunks de texto completo nesta camada/filtro.")
        return

    st.markdown("**Chunks por artigo (apenas os que têm texto completo)**")
    fig = px.histogram(
        fulltext_chunks_per_article.rename("n_chunks").reset_index(),
        x="n_chunks",
        nbins=20,
        labels={"n_chunks": "Chunks por artigo (abstract + fulltext)"},
    )
    fig.update_traces(marker_color=CATEGORICAL_PALETTE[1])
    fig.update_layout(yaxis_title="Quantidade de artigos")
    render_chart(
        fig,
        height=CHART_HEIGHT,
        caption="Dimensiona o custo de gerar embeddings: artigos com PDFs longos geram mais chunks "
        "de texto completo.",
    )


def _render_result_card(
    row: pd.Series, term_pattern: re.Pattern | None, score: float | None
) -> None:
    text = str(row["text"])
    if term_pattern:
        match = term_pattern.search(text)
        idx = match.start() if match else 0
        match_len = (match.end() - match.start()) if match else 0
    else:
        idx, match_len = 0, 0
    start = max(idx - 120, 0)
    end = min(idx + match_len + 120, len(text))
    excerpt = ("…" if start > 0 else "") + text[start:end] + ("…" if end < len(text) else "")
    if term_pattern:
        excerpt = term_pattern.sub(lambda m: f"**{m.group(0)}**", excerpt)
    doi = row.get("doi")
    chunk_type = row.get("chunk_type", "")
    with st.container(border=True):
        header = f"**{chunk_type}**"
        if score is not None:
            header += f" · similaridade {score:.2f}"
        if doi:
            header += f" · [{doi}](https://doi.org/{doi})"
        st.markdown(header)
        st.caption(excerpt)


def _search_demo(chunks_df: pd.DataFrame) -> None:
    st.subheader("🔍 Busca nos chunks")
    has_embeddings = "has_embedding" in chunks_df.columns and bool(chunks_df["has_embedding"].any())

    if has_embeddings:
        st.caption(
            "Busca por similaridade vetorial real: a consulta é embedada com o mesmo modelo "
            "(`BAAI/bge-small-en-v1.5` via `fastembed`) usado para os chunks, e os resultados são "
            "ordenados por similaridade de cosseno (`dashboard/search.py`)."
        )
    else:
        st.caption(
            "Isto simula uma recuperação por palavra-chave, **não** uma busca semântica real — a coluna "
            "`embedding` ainda não foi preenchida (ver o indicador acima; rode a etapa `embed` do "
            "pipeline). Serve para mostrar, na prática, o formato dos trechos que um RAG real usaria como "
            "contexto de resposta."
        )
    if chunks_df.empty or "doi" not in chunks_df.columns:
        st.info("Nenhum chunk disponível nesta camada/filtro.")
        return

    search_mode = "Híbrido (Vetorial + BM25 RRF)"
    if has_embeddings:
        col_q, col_m = st.columns([3, 2])
        with col_q:
            query = st.text_input(
                "Buscar (linguagem natural ou termo):",
                placeholder="ex.: distribution network, hosting capacity, IEEE 33-bus, SOCP...",
            )
        with col_m:
            search_mode = st.radio(
                "Algoritmo de Recuperação:",
                options=["Híbrido (Vetorial + BM25 RRF)", "Vetorial Puro (BGE-Small)"],
                horizontal=True,
            )
    else:
        query = st.text_input(
            "Buscar termo nos chunks:",
            placeholder="ex.: distribution network, hosting capacity, monte carlo...",
        )
    if not query:
        return

    # `chunks_df` is the lightweight, filter-scoped frame (no `text`/
    # `embedding` -- see `data.py::load_chunks`); the full columns are only
    # loaded here, lazily, once a query is actually submitted, then scoped to
    # the same filtered DOI set.
    search_df = loaders.chunk_search_data()
    if search_df.empty or "doi" not in search_df.columns:
        st.info("Nenhum chunk disponível para busca nesta camada.")
        return
    scoped = search_df[search_df["doi"].isin(chunks_df["doi"])]
    if not require_columns(
        scoped, ["text"], "Nenhum chunk com texto disponível nesta camada/filtro."
    ):
        return

    if has_embeddings:
        if "Híbrido" in search_mode:
            matches = hybrid_search_rrf(query, scoped, top_k=SEARCH_DEMO_MAX_RESULTS)
            caption_mode = "Busca Híbrida RRF (Dense BGE-Small + BM25 Okapi)"
        else:
            matches = semantic_search(query, scoped, top_k=SEARCH_DEMO_MAX_RESULTS)
            caption_mode = "Busca Vetorial Densa (BGE-Small)"
        st.caption(
            f"{caption_mode} — Top {len(matches):,} chunks mais relevantes (de {len(scoped):,} disponíveis)."
        )
    else:
        mask = scoped["text"].str.contains(query, case=False, na=False, regex=False)
        n_total_matches = int(mask.sum())
        matches = scoped.loc[mask].head(SEARCH_DEMO_MAX_RESULTS)
        st.caption(f"{n_total_matches:,} de {len(scoped):,} chunks contêm o termo buscado.")

    if matches.empty:
        return

    term_pattern = re.compile(re.escape(query), re.IGNORECASE)
    for _, row in matches.iterrows():
        score = float(row["score"]) if has_embeddings and "score" in row else None
        _render_result_card(row, term_pattern, score)

    if not has_embeddings and n_total_matches > SEARCH_DEMO_MAX_RESULTS:
        st.caption(f"Mostrando {SEARCH_DEMO_MAX_RESULTS} de {n_total_matches:,} resultados.")


def _bibliometric_anomalies_audit(articles_df: pd.DataFrame) -> None:
    st.subheader("🕵️ Auditoria Não-Supervisionada de Anomalias (Isolation Forest)")
    st.caption(
        "O algoritmo Isolation Forest isola observações atípicas através de particionamento aleatório do espaço "
        "multidimensional de atributos (ano de publicação, contagem de citações, referências, coautores e "
        "relevância temática). Artigos anômalos requerem menos divisões para serem isolados, revelando "
        "publicações hiper-citadas recentes, mega-equipes incomuns, desvios de metadados ou ruído de indexação."
    )

    signals = loaders.semantics()
    joined = articles_df.copy()
    if not signals.empty and "relevance_score" in signals.columns:
        joined = pd.merge(joined, signals[["doi", "relevance_score"]], on="doi", how="left")

    from lake_research_map.dashboard.analytics import detect_bibliometric_anomalies

    anomalies_df = detect_bibliometric_anomalies(joined, contamination=0.03)
    n_anomalies = int(anomalies_df["is_anomaly"].sum())

    metric_row(
        [
            ("📚 Total de Artigos Auditados", f"{len(anomalies_df):,}", None),
            (
                "🚩 Artigos Atípicos Detectados",
                f"{n_anomalies}",
                f"{n_anomalies / max(len(anomalies_df), 1):.1%} do acervo",
            ),
            ("🎯 Contaminação Assumida", "3.0%", "Limiar estatístico"),
            ("🔍 Algoritmo", "Isolation Forest", "100 estimadores / árvores"),
        ]
    )

    fig = px.scatter(
        anomalies_df,
        x="year",
        y="citation_count",
        color="is_anomaly",
        color_discrete_map={True: "#e34948", False: "#2a78d6"},
        hover_data=["title", "venue", "anomaly_reason", "anomaly_score"],
        labels={
            "year": "Ano de Publicação",
            "citation_count": "Citações",
            "is_anomaly": "Atípico?",
        },
        title="Dispersão Citações × Ano com Marcação de Anomalias Bibliométricas",
    )
    fig.update_layout(height=480)
    render_chart(
        fig,
        caption="Pontos vermelhos indicam artigos cujos vetores de atributos se distanciam significativamente do padrão mediano do corpus.",
    )

    st.markdown("##### 📋 Artigos Auditados como Atípicos")
    outliers = (
        anomalies_df[anomalies_df["is_anomaly"]]
        .sort_values("anomaly_score", ascending=False)
        .copy()
    )
    cols = [
        "title",
        "year",
        "venue",
        "citation_count",
        "source",
        "anomaly_score",
        "anomaly_reason",
        "doi",
    ]
    display_cols = [c for c in cols if c in outliers.columns]
    article_table(outliers, display_cols, download_key="artigos_anomalos_auditoria")
