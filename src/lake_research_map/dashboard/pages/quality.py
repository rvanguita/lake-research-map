"""🧩 Quality and RAG — metadata richness, full text coverage and chunks."""

from __future__ import annotations

import re

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from lake_research_map.dashboard import loaders
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
        "Quality and RAG",
        "Diagnosis of the corpus as a source for RAG: metadata, full text and fragments (chunks).",
    )

    articles_df = loaders.require_articles()
    chunks_df = loaders.filtered_chunks()

    pdf_share = (
        articles_df["has_pdf"].fillna(False).astype(bool).mean()
        if "has_pdf" in articles_df.columns
        else 0.0
    )
    hero_banner(
        "RAG diagnosis",
        f"The corpus offers <b>{pdf_share:.0%}</b> full-text coverage and "
        f"<b>{len(chunks_df):,}</b> chunks ready for retrieval.",
    )

    tab_metadata, tab_content, tab_anomalies, tab_search = st.tabs(
        [
            "Metadata and coverage",
            "Text, chunks and embeddings",
            "Audit and anomalies",
            "Hybrid search",
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
                    "Chunk breakdown",
                    ["Types", "Length", "By article", "Embeddings"],
                    default="Types",
                    key="quality_chunk_view",
                )
                if content_view == "Types":
                    _chunk_type_pie(chunks_df)
                elif content_view == "Length":
                    _chunk_length_histogram(chunks_df)
                elif content_view == "By article":
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
    st.subheader("🔷 IEEE-only fields")

    if "countries" not in articles_df.columns:
        st.info(
            "IEEE enrichment columns do not exist in this layer yet — round "
            "`uv run lake-research-map --stage bronze` (and silver/gold) to populate them."
        )
        return

    ieee_only = articles_df[
        articles_df["source"].eq("ieee")
        if "source" in articles_df.columns
        else articles_df.index.notna()
    ]
    n_ieee = len(ieee_only)
    if n_ieee == 0:
        st.info("No article from the IEEE database in the current filter.")
        return

    hero_banner(
        "Partial coverage by nature of the source",
        f"These fields come from the IEEE Xplore CSV export and are not supplied by ScienceDirect. "
        f"The denominator is <b>{n_ieee:,} IEEE articles</b> — about "
        f"{n_ieee / max(len(articles_df), 1):.0%} of the filtered corpus. "
        "All the percentages below use this denominator, not the entire corpus.",
    )

    countries = ieee_only["countries"].apply(lambda c: c if isinstance(c, list) else [])
    with_country = int(countries.apply(bool).sum())
    metric_row(
        [
            ("🔷 IEEE articles", f"{n_ieee:,}", None),
            ("🌍 With identified country", f"{with_country:,}", f"{with_country / n_ieee:.0%}"),
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
        ["🌍 Countries", "🗓️ Monthly granularity", "& License type"]
    )

    with sub_pais:
        exploded = countries.explode().dropna()
        if exploded.empty:
            st.info("No affiliation with identifiable country.")
        else:
            top = exploded.value_counts().head(15)
            fig = topn_hbar(
                top,
                title="Top 15 countries for participation in articles (IEEE basis)",
                x_title="Articles with at least one author in the country",
                y_title="Country",
            )
            fig.update_traces(hovertemplate="<b>%{y}</b><br>%{x:,} articles<extra></extra>")
            render_chart(
                fig,
                caption="An article counts once a country present among its affiliations, then "
                "International collaborations appear in more than one country and the sum of bars"
                f"exceeds the {n_ieee:,} articles. It is extracted from the final segment of each affiliation "
                "(`…, city, country`) and also handles the U.S. format (`…, UT, USA`).",
            )

    with sub_mes:
        if "online_date" not in ieee_only.columns or ieee_only["online_date"].isna().all():
            st.info("'online_date' column unavailable.")
        else:
            dated = ieee_only.dropna(subset=["online_date"]).copy()
            dated["mes"] = pd.to_datetime(dated["online_date"]).dt.to_period("M").dt.to_timestamp()
            by_month = dated.groupby("mes").size().reset_index(name="articles")
            fig = px.bar(
                by_month,
                x="mes",
                y="articles",
                title="Publications per month of online availability (IEEE database)",
                labels={"mes": "Month", "articles": "Articles"},
                color_discrete_sequence=[SOURCE_COLORS["ieee"]],
            )
            fig.update_layout(
                xaxis_title="Online publication month", yaxis_title="Number of articles"
            )
            render_chart(
                fig,
                caption="`Online Date` is the unique** corpus field with finer resolution than the "
                "year — in all the rest of the dashboard only the year survives. "
                "and the gap between online publication and formal edition.",
            )

    with sub_tipo:
        col_tipo, col_lic = st.columns(2)
        with col_tipo:
            if "document_type" in ieee_only.columns and ieee_only["document_type"].notna().any():
                counts = ieee_only["document_type"].dropna().value_counts()
                fig = px.pie(
                    names=counts.index,
                    values=counts.to_numpy(),
                    title="Vehicle type (document identifier)",
                    color_discrete_sequence=CATEGORICAL_PALETTE,
                )
                render_chart(
                    fig,
                    caption="It reveals that the IEEE Xplore also hosts periodicals from others "
                    "publishers (CSEE, SGEPRI), as well as magazines and book chapters.",
                )
            else:
                st.info("'document_type' column unavailable.")
        with col_lic:
            if "license" in ieee_only.columns and ieee_only["license"].notna().any():
                counts = ieee_only["license"].dropna().value_counts()
                oa = int(counts.filter(like="CC").sum())
                fig = px.pie(
                    names=counts.index,
                    values=counts.to_numpy(),
                    title="Publication license",
                    color_discrete_sequence=CATEGORICAL_PALETTE,
                )
                render_chart(
                    fig,
                    caption=f"{oa} articles under a Creative Commons license (open access) among the "
                    f"{int(counts.sum())} records with a declared license.",
                )
            else:
                st.info("'License' column unavailable.")


def _embedding_readiness(chunks_df: pd.DataFrame) -> None:
    st.subheader("🧠 Embeddings readiness")
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
                "text": "% of chunks with a vector",
                "font": {"color": t["chart_text"], "size": 14},
            },
        )
    )
    if with_embedding and "embed_model" in chunks_df.columns:
        model_used = chunks_df["embed_model"].dropna().mode()
        model_caption = (
            f"Generated with `{model_used.iat[0]}` — the pipeline's `embed` stage runs "
            "fully locally through `fastembed`, with no API key."
        )
    else:
        model_caption = (
            "The vector column exists on `gold_models.DatasetChunk` but no chunk has been embedded "
            "yet — this is the blocker for using the corpus as a RAG source."
        )
    render_chart(
        fig,
        height=260,
        caption=f"{with_embedding:,} of {total:,} chunks have embeddings. {model_caption}",
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
        by_type["pending"] = by_type["total"] - by_type["embedded"]
        by_type.columns = ["Type", "Total", "Embedded", "Pending"]
        st.dataframe(by_type, hide_index=True, width="stretch")

    if pending > 0:
        # Deliberately not a button. This page used to embed synchronously into
        # the live `lit_chunks` table, but publication rebuilds that table from
        # the versioned candidate, so those vectors were discarded at the next
        # publish and never passed `embed_contract`. Embedding belongs to the
        # audited pipeline run like every other stage (ADR-06).
        st.info(
            f"{pending:,} chunks are waiting for a vector. Run the `embed` stage through the "
            "pipeline \u2014 `uv run lake-research-map --stage embed`, or the Airflow trigger on "
            "the Pipeline and provenance page. Embedding from here would write to the live table "
            "that the next publication rebuilds, so the work would be lost and would never pass "
            "the embedding contract."
        )


def _fulltext_coverage(articles_df: pd.DataFrame, chunks_df: pd.DataFrame) -> None:
    st.subheader("📄 Full-text coverage")
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
            "stage": ["All articles", "With a linked PDF", "With full-text chunks"],
            "count": [n_total, n_with_pdf, n_with_fulltext_chunks],
        }
    )
    col_funnel, col_metrics = st.columns([2, 1])
    with col_funnel:
        fig = px.funnel(
            funnel_df,
            x="count",
            y="stage",
            title="Full-text availability funnel",
            labels={"count": "Number of articles", "stage": "Stage"},
        )
        fig.update_traces(
            marker_color=[CATEGORICAL_PALETTE[0], CATEGORICAL_PALETTE[3], CATEGORICAL_PALETTE[2]]
        )
        render_chart(fig)
    with col_metrics:
        metric_row(
            [
                (
                    "📎 PDF articles",
                    f"{n_with_pdf:,}",
                    f"{n_with_pdf / n_total:.1%}" if n_total else None,
                ),
            ]
        )
        metric_row(
            [
                (
                    "",
                    f"{max(n_pdf_no_chunks, 0):,}",
                    "extraction failed or empty PDF"
                    if n_pdf_no_chunks > 0
                    else "todos processados",
                ),
            ]
        )
    st.caption(
        f"Only {n_with_pdf / n_total:.1%} of articles have a PDF — corpus PDFs come exclusively from "
        "IEEE bulk-downloads, so Elsevier has 0% full text coverage. "
        "(`gold_articles.py`) silently swallows faults, so a PDF that failed is indistinguishable from a "
        "PDF without extractable text — the card '"
    )


def _pdf_selection_bias(articles_df: pd.DataFrame) -> None:
    """Test whether the full-text subset differs materially from the corpus."""
    st.subheader("PDF availability biases")
    bias = pdf_selection_bias(articles_df)
    finite = bias.dropna(subset=["smd"])
    if finite.empty:
        st.info("Articles with and without PDF and at least two observations per group are needed.")
        return

    labels = {
        "year": "Year",
        "citation_count": "Citations",
        "reference_count": "References",
        "team_size": "Team size",
        "abstract_chars": "Abstract length",
        "keyword_count": "Number of keywords",
    }
    bias = bias.copy()
    bias["Metrics"] = bias["metric"].map(labels)
    finite = finite.copy()
    finite["Metrics"] = finite["metric"].map(labels)
    finite["erro_superior"] = finite["ci_high"] - finite["smd"]
    finite["erro_inferior"] = finite["smd"] - finite["ci_low"]
    colors = [
        TREND_DOWN_COLOR if abs(value) >= 0.1 else CATEGORICAL_PALETTE[0] for value in finite["smd"]
    ]
    fig = go.Figure(
        go.Scatter(
            x=finite["smd"],
            y=finite["Metrics"],
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
                "<b>%{y}</b><br>SMD: %{x:.2f}<br>With PDF: %{customdata[0]} "
                "<br>No PDF: %{customdata[1]}<br>Transformation: %{customdata[2]}<extra></extra>"
            ),
        )
    )
    fig.add_vline(x=0, line_color=theme_tokens()["grid"], line_width=1)
    fig.add_vrect(x0=-0.1, x1=0.1, fillcolor=CATEGORICAL_PALETTE[0], opacity=0.08, line_width=0)
    fig.update_layout(
        title="Standardized difference: articles with PDF less articles without PDF",
        xaxis_title="Standardized mean difference (IC bootstrap of 95%)",
        yaxis_title="Metrics",
        showlegend=False,
    )
    render_chart(
        fig,
        caption="Positive values indicate a higher mean in the subset with PDF. "
        "The comparison is descriptive: PDF availability depends "
        "of the source and does not support causal interpretation.",
    )

    audit = bias[
        ["Metrics", "transform", "n_pdf", "n_no_pdf", "smd", "ci_low", "ci_high", "status"]
    ].rename(
        columns={
            "transform": "Transformation",
            "n_pdf": "n com PDF",
            "n_no_pdf": "n without PDF",
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
        st.info("There are not enough common fields to measure the coverage.")
        return

    labels = {
        "doi": "DOI",
        "title": "Title",
        "year": "Year",
        "venue": "Periodic/event",
        "authors": "Authors",
        "abstract": "Resumo",
        "keywords": "Keywords",
        "citation_count": "Citations",
        "reference_count": "References",
        "has_pdf": "PDF available",
    }
    coverage["Campo"] = coverage["field"].map(labels).fillna(coverage["field"])
    matrix = coverage.pivot(index="Campo", columns="source", values="coverage")
    st.subheader("Completeness of common fields by source")
    fig = px.imshow(
        matrix,
        zmin=0,
        zmax=1,
        text_auto=".0%",
        aspect="auto",
        color_continuous_scale="Blues",
        labels={"x": "Source", "y": "Campo", "color": "Cobertura"},
    )
    fig.update_layout(xaxis_title="Source", yaxis_title="Campo")
    render_chart(
        fig,
        caption="Each cell uses only the articles of the indicated source as a denominator. "
        "Availability, not simple existence of the column. "
        "separately below to avoid classifying structural absence of Elsevier as a failure.",
    )
    exact = coverage[["Campo", "source", "n_total", "n_present", "coverage"]].rename(
        columns={
            "source": "Source",
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
            "`lit_gold.chunks` has not yet been populated — run the complete pipeline "
            "(sidebar button or `uv run lake-research-map --stage all`)."
        )
        return False

    if not require_columns(
        chunks_df, ["chunk_type"], "The chunk table does not have the `chunk_type` field."
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
            ("🧩 Total chunks", f"{len(chunks_df):,}", None),
            ("📝 Abstract chunks", f"{n_abstract:,}", None),
            ("📄 Full-text chunks", f"{n_fulltext:,}", f"{n_ft_dois} articles with PDFs"),
            (
                "📄 DOIs distintos com fragmentos",
                f"{n_dois:,}",
                None,
            ),
        ]
    )

    if n_fulltext > 0 and n_dois > 0:
        st.caption(
            f"⚠️ **Coverage bias**: full text covers {n_ft_dois} of {n_dois} articles "
            f"({100 * n_ft_dois / n_dois:.1f}%), but produces {n_fulltext:,} of {len(chunks_df):,} "
            f"chunks ({100 * n_fulltext / len(chunks_df):.1f}%). Per-chunk statistics reflect "
            f"disproportionately those {100 * n_ft_dois / n_dois:.1f}% of the corpus."
        )
    return True


def _chunk_type_pie(chunks_df: pd.DataFrame) -> None:
    by_type = chunks_df["chunk_type"].value_counts().rename_axis("type").reset_index(name="count")
    by_type["label_pt"] = by_type["type"].map(
        {"abstract": "Resumo (Abstract)", "fulltext": "Texto Completo (Fulltext)"}
    )
    fig = px.pie(by_type, names="label_pt", values="count", title="Proportion of chunk by type")
    fig.update_traces(
        texttemplate="<b>%{label}</b><br><b>%{value:,} (%{percent})</b>",
        hovertemplate="<b>%{label}</b>: %{value:,} chunks (%{percent})<extra></extra>",
    )
    render_chart(
        fig,
        height=CHART_HEIGHT,
        caption="`abstract` = 1 fragment per article (title + keywords + abstract); `fulltext` = "
        "fragments extracted directly from the PDFs of the available articles.",
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
        title="Chunk length (characters) by type",
        labels={"char_len": "Comprimento em caracteres", "chunk_type": "Tipo"},
    )
    fig.update_traces(hovertemplate="Length: ~%{x} characters<br>Chunks: %{y:,}<extra></extra>")
    fig.update_layout(
        xaxis_title=f"Characters per chunk (chunking cap: {CHUNK_MAX_CHARS:,})",
        yaxis_title="Number of chunks",
    )
    render_chart(
        fig,
        height=CHART_HEIGHT,
        caption=f"The axis extends to the {CHUNK_MAX_CHARS:,}-character ceiling used to split the "
        "texto completo (`gold_articles.CHUNK_MAX_CHARS`); fragmentos excessivamente curtos perdem "
        "Semantic context.",
    )


def _chunks_per_article(chunks_df: pd.DataFrame) -> None:
    if "doi" not in chunks_df.columns:
        st.info("Column 'doi' not available in this layer.")
        return

    chunks_per_article = chunks_df.groupby("doi").size()
    fulltext_dois = (
        chunks_df.loc[chunks_df["chunk_type"] == "fulltext", "doi"].unique()
        if "chunk_type" in chunks_df.columns
        else []
    )
    fulltext_chunks_per_article = chunks_per_article.reindex(fulltext_dois)
    if fulltext_chunks_per_article.empty:
        st.info("No article with full text chunks in this layer/filter.")
        return

    st.markdown("**Chunks per article (only those with full text)**")
    fig = px.histogram(
        fulltext_chunks_per_article.rename("n_chunks").reset_index(),
        x="n_chunks",
        nbins=20,
        labels={"n_chunks": "Leads per article (abstract + fulltext)"},
    )
    fig.update_traces(marker_color=CATEGORICAL_PALETTE[1])
    fig.update_layout(yaxis_title="Number of articles")
    render_chart(
        fig,
        height=CHART_HEIGHT,
        caption="Dimensions the cost of generating embeddings: articles with long PDFs generate more blanks "
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
            "Search for real vector similarity: the query is embedded with the same model "
            "(`BAAI/bge-small-en-v1.5` via `fastembed`) used for chunks, and the results are "
            "ranked by cosine similarity (`dashboard/search.py`)."
        )
    else:
        st.caption(
            "This simulates a keyword retrieval, *not** a real semantic search — the column "
            "`embedding` has not yet been completed (see the indicator above; turn the `embed` step of the "
            "pipeline). "
            "answer context."
        )
    if chunks_df.empty or "doi" not in chunks_df.columns:
        st.info("No chunk available in this layer/filter.")
        return

    search_mode = "Hybrid (Vetorial + BM25 RRF)"
    if has_embeddings:
        col_q, col_m = st.columns([3, 2])
        with col_q:
            query = st.text_input(
                "Search (natural language or term):",
                placeholder="ex.: distribution network, hosting capacity, IEEE 33-bus, SOCP...",
            )
        with col_m:
            search_mode = st.radio(
                "Recovery algorithm:",
                options=["Hybrid (Vetorial + BM25 RRF)", "Vetorial Puro (BGE-Small)"],
                horizontal=True,
            )
    else:
        query = st.text_input(
            "Search for a term in chunks:",
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
        st.info("No chunk available for search in this layer.")
        return
    scoped = search_df[search_df["doi"].isin(chunks_df["doi"])]
    if not require_columns(scoped, ["text"], "No chunk with text available in this layer/filter."):
        return

    if has_embeddings:
        if "Hybrid" in search_mode:
            matches = hybrid_search_rrf(query, scoped, top_k=SEARCH_DEMO_MAX_RESULTS)
            caption_mode = "Hybrid Search RRF (BGE-Small + BM25 Okapi)"
        else:
            matches = semantic_search(query, scoped, top_k=SEARCH_DEMO_MAX_RESULTS)
            caption_mode = "Busca Vetorial Densa (BGE-Small)"
        st.caption(
            f"{caption_mode} — Top {len(matches):,} most relevant chunks (of {len(scoped):,} available)."
        )
    else:
        mask = scoped["text"].str.contains(query, case=False, na=False, regex=False)
        n_total_matches = int(mask.sum())
        matches = scoped.loc[mask].head(SEARCH_DEMO_MAX_RESULTS)
        st.caption(f"{n_total_matches:,} of {len(scoped):,} chunks contain the search term.")

    if matches.empty:
        return

    term_pattern = re.compile(re.escape(query), re.IGNORECASE)
    for _, row in matches.iterrows():
        score = float(row["score"]) if has_embeddings and "score" in row else None
        _render_result_card(row, term_pattern, score)

    if not has_embeddings and n_total_matches > SEARCH_DEMO_MAX_RESULTS:
        st.caption(f"Showing {SEARCH_DEMO_MAX_RESULTS} of {n_total_matches:,} results.")


def _bibliometric_anomalies_audit(articles_df: pd.DataFrame) -> None:
    st.subheader("Isolation Forest (Isolation Forest)")
    st.caption(
        "The Isolation Forest algorithm isolates atypical observations through random partitioning of space "
        "multidimensional attributes (year of publication, citation count, references, co-authors and authors) "
        "Thematic relevance).Anomalous articles require fewer divisions to be isolated, revealing "
        "Recent hyper-cited publications, unusual mega-teams, metadata deviations or indexing noise."
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
            ("📚 Total Audited Articles", f"{len(anomalies_df):,}", None),
            (
                "🚩 Detected Atypical Articles",
                f"{n_anomalies}",
                f"{n_anomalies / max(len(anomalies_df), 1):.1%} of the collection",
            ),
            ("🎯 Contamination", "3.0%", "Statistical threshold"),
            ("🔍 Algoritmo", "Isolation Forest", "100 estimators/trees"),
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
            "year": "Year of Publication",
            "citation_count": "Citations",
            "is_anomaly": "Atypical?",
        },
        title="Citations × Year Dispersion with Bibliometric Anomalies Marking",
    )
    fig.update_layout(height=480)
    render_chart(
        fig,
        caption="Red dots indicate articles whose attribute vectors differ significantly from the median pattern of the corpus.",
    )

    st.markdown("#### 📋 Articles Audited as Atypical")
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
    article_table(outliers, display_cols, download_key="anomalous_articles_audit")
