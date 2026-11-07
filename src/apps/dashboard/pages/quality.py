"""🧩 Quality and RAG — metadata richness, full text coverage and chunks."""

from __future__ import annotations

import re

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from apps.api.api import (
    BENCHMARK_MODES,
    ArticleHit,
    RetrievalUnavailable,
    mode_runner,
)
from apps.api.filters import ArticleFilters
from apps.dashboard import loaders
from apps.dashboard.analytics import metadata_coverage_matrix, pdf_selection_bias
from apps.dashboard.charts import topn_hbar
from apps.dashboard.components import (
    article_table,
    hero_banner,
    metric_row,
    page_header,
    render_chart,
    require_columns,
)
from apps.dashboard.retrieval_eval import (
    benchmark_retrieval_modes,
    evaluate_rankings,
)
from apps.dashboard.theme import (
    CATEGORICAL_PALETTE,
    CHART_HEIGHT,
    FLAG_COLORS,
    SOURCE_COLORS,
    TREND_DOWN_COLOR,
    theme_tokens,
)
from transform.evidence_samples import RETRIEVAL_QUERIES
from transform.gold_articles import CHUNK_MAX_CHARS
from transform.vector_payloads import ABSTRACT_MIN_CHARS, abstract_is_missing

SEARCH_DEMO_MAX_RESULTS = 10
SEARCH_MODES = {
    "Hybrid (dense + BM25, RRF)": "hybrid",
    "Dense (BGE-small)": "dense",
    "BM25 (lexical)": "bm25",
}
SEARCH_SCOPES = {
    "Abstracts": "abstract",
    "All chunks (favours articles with a PDF)": "all",
}
# The pooled judgement sample was drawn at depth 10, so scoring deeper than
# that would count unjudged results as irrelevant and understate every mode.
RETRIEVAL_BENCHMARK_K = 10


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
            "Article search",
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
            _article_search(articles_df)
            _retrieval_benchmark()


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
                "🗓️ With an online date",
                f"{int(ieee_only['online_date'].notna().sum()):,}"
                if "online_date" in ieee_only.columns
                else "n/a",
                None,
            ),
        ]
    )

    sub_country, sub_month, sub_type = st.tabs(
        ["🌍 Countries", "🗓️ Monthly granularity", "📄 License type"],
        on_change="rerun",
        key="quality_ieee_tab",
    )

    if sub_country.open:
        with sub_country:
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
                    caption="An article counts once for each country present among its "
                    "affiliations, so an international collaboration appears under more than one "
                    f"country and the bars sum to more than the {n_ieee:,} articles. The country "
                    "is taken from the final segment of each affiliation (`…, city, country`), "
                    "which also handles the U.S. format (`…, UT, USA`).",
                )

    if sub_month.open:
        with sub_month:
            if "online_date" not in ieee_only.columns or ieee_only["online_date"].isna().all():
                st.info("'online_date' column unavailable.")
            else:
                dated = ieee_only.dropna(subset=["online_date"]).copy()
                dated["mes"] = (
                    pd.to_datetime(dated["online_date"]).dt.to_period("M").dt.to_timestamp()
                )
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
                    caption="`Online Date` is the **only** corpus field with finer resolution than "
                    "year; elsewhere in the dashboard only the publication year survives. It can "
                    "reveal seasonality and the lag between online publication and the formal issue.",
                )

    if sub_type.open:
        with sub_type:
            col_type, col_license = st.columns(2)
            with col_type:
                if (
                    "document_type" in ieee_only.columns
                    and ieee_only["document_type"].notna().any()
                ):
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
            with col_license:
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
            "fully locally through `fastembed`, with no API key, and stores each version's "
            "vectors in its own Qdrant collection (the readiness shown here is the provenance "
            "Gold records for them)."
        )
    else:
        model_caption = (
            "No chunk of this version has been embedded yet — this is the blocker for using "
            "the corpus as a RAG source. Vectors are stored in Qdrant, one collection per "
            "dataset version."
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
            "pipeline \u2014 `uv run lake-research-map --stage embed` \u2014 with Qdrant "
            "running (`docker compose --profile qdrant up -d qdrant`). Embedding from here would bypass "
            "the embedding contract that every published version must pass."
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
                    "📄 PDFs without extracted text",
                    f"{max(n_pdf_no_chunks, 0):,}",
                    "extraction failed or empty PDF" if n_pdf_no_chunks > 0 else "all processed",
                ),
            ]
        )
    st.caption(
        f"Only {n_with_pdf / n_total:.1%} of articles have a PDF — corpus PDFs come exclusively "
        "from IEEE bulk-downloads, so Elsevier has 0% full-text coverage. Text extraction "
        "(`gold_articles.py`) silently swallows faults, so a PDF that failed to parse is "
        "indistinguishable from one that carries no extractable text; the card above counts "
        "both together."
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
    finite["error_upper"] = finite["ci_high"] - finite["smd"]
    finite["error_lower"] = finite["smd"] - finite["ci_low"]
    # |SMD| < 0.1 is the conventional "negligible" band. It used to be drawn
    # as a shaded band with a zero line; the chart contract forbids guide
    # lines, so the band is carried by the marker color, the hover and the
    # caption instead.
    finite["magnitude"] = (
        finite["smd"]
        .abs()
        .ge(0.1)
        .map({True: "|SMD| ≥ 0.1, not negligible", False: "|SMD| < 0.1, negligible"})
    )
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
                "array": finite["error_upper"],
                "arrayminus": finite["error_lower"],
                "visible": True,
            },
            customdata=finite[["n_pdf", "n_no_pdf", "transform", "magnitude"]],
            hovertemplate=(
                "<b>%{y}</b><br>SMD: %{x:.2f} (%{customdata[3]})<br>With PDF: %{customdata[0]} "
                "<br>No PDF: %{customdata[1]}<br>Transformation: %{customdata[2]}<extra></extra>"
            ),
        )
    )
    fig.update_layout(
        title="Standardized difference: articles with PDF less articles without PDF",
        xaxis_title="Standardized mean difference (95% bootstrap CI)",
        yaxis_title="Metrics",
        showlegend=False,
    )
    render_chart(
        fig,
        caption="Zero means no difference; positive values indicate a higher mean in the subset "
        "with PDF. Blue points fall inside the conventional negligible band (|SMD| < 0.1); "
        "purple points fall outside it. The comparison is descriptive: PDF availability "
        "depends on the source and does not support causal interpretation.",
    )

    audit = bias[
        ["Metrics", "transform", "n_pdf", "n_no_pdf", "smd", "ci_low", "ci_high", "status"]
    ].rename(
        columns={
            "transform": "Transformation",
            "n_pdf": "n with PDF",
            "n_no_pdf": "n without PDF",
            "smd": "SMD",
            "ci_low": "CI 2.5%",
            "ci_high": "CI 97.5%",
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
        "venue": "Journal/event",
        "authors": "Authors",
        "abstract": "Abstract",
        "keywords": "Keywords",
        "citation_count": "Citations",
        "reference_count": "References",
        "has_pdf": "PDF available",
    }
    coverage["Field"] = coverage["field"].map(labels).fillna(coverage["field"])
    matrix = coverage.pivot(index="Field", columns="source", values="coverage")
    st.subheader("Completeness of common fields by source")
    fig = px.imshow(
        matrix,
        zmin=0,
        zmax=1,
        text_auto=".0%",
        aspect="auto",
        color_continuous_scale="Blues",
        labels={"x": "Source", "y": "Field", "color": "Completeness"},
    )
    fig.update_layout(xaxis_title="Source", yaxis_title="Field")
    render_chart(
        fig,
        caption="Each cell uses only articles from the indicated source as its denominator and "
        "measures usable values, not merely whether the column exists. IEEE-only fields are reported "
        "separately below so Elsevier's structural absence is not misclassified as a quality failure.",
    )
    exact = coverage[["Field", "source", "n_total", "n_present", "coverage"]].rename(
        columns={
            "source": "Source",
            "n_total": "Total",
            "n_present": "Present",
            "coverage": "Completeness",
        }
    )
    st.dataframe(
        exact,
        hide_index=True,
        width="stretch",
        column_config={"Completeness": st.column_config.ProgressColumn(format="percent")},
    )


def _chunks_intro(chunks_df: pd.DataFrame) -> bool:
    """Guard + shared summary metric row. Returns True if there's data to show."""
    st.subheader("🧩 Chunks prepared for embedding")
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
                "📄 Distinct DOIs with chunks",
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
    by_type["label_pt"] = by_type["type"].map({"abstract": "Abstract", "fulltext": "Full text"})
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
        labels={"char_len": "Length in characters", "chunk_type": "Type"},
    )
    fig.update_traces(hovertemplate="Length: ~%{x} characters<br>Chunks: %{y:,}<extra></extra>")
    fig.update_layout(
        xaxis_title=f"Characters per chunk (chunking cap: {CHUNK_MAX_CHARS:,})",
        yaxis_title="Number of chunks",
    )
    render_chart(
        fig,
        height=CHART_HEIGHT,
        caption=f"The axis extends to the {CHUNK_MAX_CHARS:,}-character ceiling used to split full "
        "text (`gold_articles.CHUNK_MAX_CHARS`). Very short fragments lose semantic context.",
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
        labels={"n_chunks": "Chunks per article (abstract + full text)"},
    )
    fig.update_traces(marker_color=CATEGORICAL_PALETTE[1])
    fig.update_layout(yaxis_title="Number of articles")
    render_chart(
        fig,
        height=CHART_HEIGHT,
        caption="Sizes the cost of generating embeddings: articles with long PDFs produce more "
        "full-text chunks.",
    )


def _sidebar_filters() -> ArticleFilters:
    """The sidebar's filters, applied before ranking -- the same population the pages show.

    Non-articles stay eligible here because the sidebar never excluded them.
    """
    year_range, sources, venues, min_margin, categories = loaders.filter_signature()
    return ArticleFilters(
        year_range=tuple(year_range) if isinstance(year_range, list | tuple) else year_range,
        sources=tuple(sources),
        venues=tuple(venues),
        publication_categories=tuple(categories),
        min_margin=min_margin,
        exclude_non_articles=False,
    )


def _escape_markdown(text: str) -> str:
    # `$` would start LaTeX in st.markdown, and abstracts quote costs in dollars.
    return text.replace("\\", "\\\\").replace("$", "\\$").replace("*", "\\*").replace("_", "\\_")


def _highlight(text: str, query: str) -> str:
    escaped = _escape_markdown(text)
    terms = sorted(
        {t for t in re.findall(r"\w+", query.lower()) if len(t) > 2}, key=len, reverse=True
    )
    if not terms:
        return escaped
    pattern = re.compile("|".join(re.escape(term) for term in terms), re.IGNORECASE)
    return pattern.sub(lambda match: f"**{match.group(0)}**", escaped)


def _render_article_hit(hit: ArticleHit, query: str) -> None:
    with st.container(border=True):
        st.markdown(f"**{hit.rank}. {_escape_markdown(hit.title or hit.doi)}**")
        where = " · ".join(str(part) for part in (hit.year, hit.venue, hit.source) if part)
        st.caption(f"{_escape_markdown(where)} · [{hit.doi}](https://doi.org/{hit.doi})")
        scores = [f"{hit.mode} score {hit.score:.4f}"]
        if hit.dense_rank is not None:
            scores.append(f"dense #{hit.dense_rank} ({hit.dense_score:.3f})")
        if hit.bm25_rank is not None:
            scores.append(f"BM25 #{hit.bm25_rank} ({hit.bm25_score:.2f})")
        if hit.rerank_rank is not None:
            scores.append(f"rerank #{hit.rerank_rank} ({hit.rerank_score:.2f})")
        flags = []
        if hit.scope == "all":
            flags.append(f"matched {hit.chunk_type} chunk {hit.chunk_seq}")
        if hit.abstract_missing:
            flags.append("no usable abstract: ranked on title and keywords")
        if hit.abstract_truncated:
            flags.append("abstract longer than the model window: its tail is not in the vector")
        if hit.relevance_margin is not None and hit.relevance_margin < 0:
            flags.append("closer to the logistics reading (margin < 0)")
        if hit.has_pdf:
            flags.append("full text available")
        st.caption(" · ".join(scores + flags))
        st.markdown(_highlight(hit.excerpt, query))


def _abstract_disclosure(articles_df: pd.DataFrame, retriever) -> None:
    """What the abstract-level ranking cannot see, for the articles in view."""
    total = len(articles_df)
    missing = (
        int(articles_df["abstract"].map(abstract_is_missing).sum())
        if "abstract" in articles_df.columns
        else None
    )
    truncated_text, window_text = "n/a", "the embedding model could not be loaded to measure it"
    try:
        lengths = retriever.abstract_token_lengths()
        if lengths is not None:
            window = retriever.context_window()
            dois = set(articles_df["doi"]) if "doi" in articles_df.columns else set(lengths)
            truncated = sum(1 for doi, n in lengths.items() if doi in dois and n > window)
            truncated_text = f"{truncated:,}"
            window_text = f"tail beyond {window} tokens never reaches the vector"
    except Exception as exc:  # a missing model must not take the page down
        window_text = f"not measured ({exc.__class__.__name__})"
    metric_row(
        [
            (
                "Articles without a usable abstract",
                "n/a" if missing is None else f"{missing:,}",
                None
                if missing is None or not total
                else f"{missing / total:.1%} of the {total:,} in view; under {ABSTRACT_MIN_CHARS} characters",
            ),
            ("Abstracts over the model window", truncated_text, window_text),
        ]
    )


def _article_search(articles_df: pd.DataFrame) -> None:
    st.subheader("🔍 Find the most relevant articles")
    retriever = loaders.retriever()
    if retriever is None:
        st.info(
            "Search needs a published Gold version. Run the pipeline "
            "(`uv run lake-research-map --stage all`) and reload this page."
        )
        return
    st.caption(
        "One result per article. By default each article competes through its abstract "
        "chunk (title, keywords and abstract), so every article is ranked on equal terms; "
        "the sidebar filters apply before ranking. The same ranking is available outside "
        "the dashboard through `lake-research-map search` and the MCP server."
    )
    col_query, col_mode = st.columns([3, 2])
    with col_query:
        query = st.text_input(
            "Query (natural language or exact terms):",
            placeholder="e.g. hosting capacity of feeders with rooftop PV, IEEE 33-bus, SOCP",
            key="quality_search_query",
        )
    with col_mode:
        mode_label = st.radio(
            "Ranking:", list(SEARCH_MODES), horizontal=True, key="quality_search_mode"
        )
    col_scope, col_rerank = st.columns([3, 2])
    with col_scope:
        scope_label = st.segmented_control(
            "Ranked unit:",
            list(SEARCH_SCOPES),
            default="Abstracts",
            required=True,
            key="quality_search_scope",
        )
    with col_rerank:
        rerank = st.checkbox(
            "Rerank with a cross-encoder (experimental)",
            key="quality_search_rerank",
            help="Reorders the candidates with a local MiniLM cross-encoder. Not an approved "
            "default: no mode is, until the WP-13 human relevance labels exist.",
        )
    _abstract_disclosure(articles_df, retriever)
    if not query:
        return

    mode = SEARCH_MODES[mode_label]
    scope = SEARCH_SCOPES[scope_label or "Abstracts"]
    filters = _sidebar_filters()
    try:
        hits = retriever.search(
            query,
            top_k=SEARCH_DEMO_MAX_RESULTS,
            mode=mode,
            scope=scope,
            filters=filters,
            rerank=rerank,
        )
    except RetrievalUnavailable as exc:
        fallback = "bm25" if not rerank or mode == "bm25" else mode
        if rerank and mode != "bm25":
            st.warning(f"The reranker is unavailable ({exc}); showing the ranking without it.")
        else:
            st.warning(
                f"Dense retrieval is unavailable ({exc}). Showing lexical (BM25) results; start "
                "Qdrant with `docker compose --profile qdrant up -d qdrant` for dense search."
            )
        try:
            hits = retriever.search(
                query, top_k=SEARCH_DEMO_MAX_RESULTS, mode=fallback, scope=scope, filters=filters
            )
        except RetrievalUnavailable:
            hits = retriever.search(
                query, top_k=SEARCH_DEMO_MAX_RESULTS, mode="bm25", scope=scope, filters=filters
            )
    if scope == "all":
        st.caption(
            f"Every chunk ranked, abstracts and full text: the {retriever.corpus.fulltext_articles:,} "
            "articles with a linked PDF compete with many chunks each, so this view favours "
            "them. Use it to find passages, not to compare articles."
        )
    if not hits:
        st.info("No article in view matches this query. Widen the sidebar filters or rephrase.")
        return
    for hit in hits:
        _render_article_hit(hit, query)
    table = pd.DataFrame([hit.to_dict() for hit in hits])
    article_table(
        table,
        ["rank", "title", "year", "venue", "source", "score", "dense_rank", "bm25_rank",
         "rerank_rank", "relevance_margin", "abstract_missing", "abstract_truncated", "doi"],
        download_key="article_search_hits",
    )  # fmt: skip


def _bibliometric_anomalies_audit(articles_df: pd.DataFrame) -> None:
    st.subheader("Unsupervised anomaly audit (Isolation Forest)")
    st.caption(
        "Isolation Forest identifies atypical observations by randomly partitioning a multidimensional "
        "feature space: publication year, citation and reference counts, co-author count, and thematic "
        "relevance. Articles isolated with fewer partitions may be recent highly cited papers, unusual "
        "mega-teams, metadata deviations, or indexing noise."
    )

    signals = loaders.semantics()
    joined = articles_df.copy()
    if not signals.empty and "relevance_score" in signals.columns:
        joined = pd.merge(joined, signals[["doi", "relevance_score"]], on="doi", how="left")

    from apps.dashboard.analytics import detect_bibliometric_anomalies

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
            ("🔍 Algorithm", "Isolation Forest", "100 estimators/trees"),
        ]
    )

    fig = px.scatter(
        anomalies_df,
        x="year",
        y="citation_count",
        color="is_anomaly",
        color_discrete_map=FLAG_COLORS,
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


def _benchmark_runners(retriever, probe: str) -> tuple[dict, dict[str, str]]:
    """Every benchmark mode that can run here, and why the others cannot."""
    runners, skipped = {}, {}
    for name, settings in BENCHMARK_MODES.items():
        run = mode_runner(retriever, settings)
        try:
            run(probe, 1)
        except RetrievalUnavailable as exc:
            skipped[name] = str(exc)
            continue
        runners[name] = run
    return runners, skipped


def _retrieval_benchmark() -> None:
    """Score every retrieval mode over the versioned query set — `RQ-08`, `WP-13`.

    The harness (`apps/dashboard/retrieval_eval.py`) and the pooled judgement sampler
    existed before the panel did; the panel is deliberately honest about what
    stands behind the numbers: a benchmark scored against one AI-assisted pass
    is a smoke test of the harness, not evidence that a mode is better.
    """
    st.divider()
    st.subheader("📐 Retrieval benchmark")

    judgements = loaders.review_labels("retrieval")
    if judgements.empty:
        st.info(
            "No retrieval judgements are stored yet. Generate the pooled sample with "
            "`lake-research-map evidence retrieval`, assign it with `reviews setup "
            "--workflow retrieval`, and import the reviewed file. Until then no "
            "Recall@k/MRR/nDCG figure would mean anything."
        )
        return

    reviewers = sorted(judgements["reviewer_id"].dropna().unique())
    independent = [name for name in reviewers if not str(name).endswith("-assisted")]
    relevant_by_query: dict[str, set[str]] = {}
    for row in judgements.itertuples(index=False):
        if str(row.label).strip().casefold() != "relevant":
            continue
        query_id, _, doi = str(row.subject_id).partition("::")
        if doi:
            relevant_by_query.setdefault(query_id, set()).add(doi)
    if not relevant_by_query:
        st.warning("Judgements exist but none marked a result relevant, so no query is scorable.")
        return

    retriever = loaders.retriever()
    if retriever is None:
        st.info("No published version to retrieve over; run the pipeline.")
        return

    queries = [(qid, text) for qid, text in RETRIEVAL_QUERIES if qid in relevant_by_query]
    runners, skipped = _benchmark_runners(retriever, queries[0][1])
    rankings, latency = benchmark_retrieval_modes(runners, queries, top_k=RETRIEVAL_BENCHMARK_K)
    metrics = evaluate_rankings(rankings, relevant_by_query, k=RETRIEVAL_BENCHMARK_K)

    table = pd.DataFrame(
        [
            {
                "Mode": mode,
                f"Recall@{RETRIEVAL_BENCHMARK_K}": values[f"recall@{RETRIEVAL_BENCHMARK_K}"],
                "MRR": values["mrr"],
                f"nDCG@{RETRIEVAL_BENCHMARK_K}": values[f"ndcg@{RETRIEVAL_BENCHMARK_K}"],
                "p50 latency (ms)": latency[mode]["p50_ms"],
                "Queries": int(values["queries"]),
            }
            for mode, values in metrics.items()
        ]
    )
    st.dataframe(table.round(3), hide_index=True, width="stretch")

    pooling = (
        " Every mode returns one result per article. The stored judgements were pooled at "
        "depth 10 from the chunk-level modes (the three 'all chunks' rows), so an "
        "article-level or reranked result outside that pool counts as not relevant until the "
        "pool is regenerated with `lake-research-map evidence retrieval`."
    )
    if skipped:
        pooling += (
            " Not run here: "
            + "; ".join(f"{name} ({reason})" for name, reason in skipped.items())
            + "."
        )
    if independent:
        st.caption(
            f"Scored over {len(queries)} judged queries from {', '.join(independent)}. "
            "The pool was built from several modes, so no mode is scored on results only "
            "it could return." + pooling
        )
    else:
        st.warning(
            f"**Not approved evidence.** Every judgement here comes from "
            f"`{', '.join(reviewers)}` — an AI-assisted pass, not an independent human "
            "review. `WP-13` requires two independent reviewers before a retrieval mode "
            "may be selected as the default on these numbers. The table shows the harness "
            "works; it does not show which mode is better." + pooling
        )
