"""🏗️ Camadas & Pipeline — o funil raw → bronze → silver → gold, camada a camada."""

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
    st.subheader("Proveniência da busca")
    configs = loaders.search_configs()
    if configs.empty:
        st.info("A configuração de busca ainda não foi ingerida na camada raw.")
        return

    st.caption(
        "Os registros abaixo preservam as consultas que originaram os exports. "
        "Eles documentam a coleta e não representam artigos do corpus."
    )
    for _, row in configs.iterrows():
        source = str(row.get("source") or "Fonte")
        with st.expander(SOURCE_LABELS.get(source, source.title())):
            st.code(str(row.get("query_string") or "—"), language=None, wrap_lines=True)
            columns = st.columns(2)
            columns[0].metric("Intervalo de anos", str(row.get("year_range") or "—"))
            columns[1].metric("Filtros", str(row.get("filters") or "—"))
            search_url = row.get("search_url")
            if search_url:
                st.link_button(
                    "Abrir busca original", str(search_url), icon=":material/open_in_new:"
                )
            st.caption(f"Arquivo: {row.get('source_file') or '—'}")


def render() -> None:
    page_header(
        "🏗️",
        "Pipeline e proveniência",
        "O funil raw → bronze → silver → gold: quanto sobrevive em cada etapa, por que, e onde a "
        "cobertura de metadados melhora ou piora.",
    )

    runs_df = loaders.pipeline_runs()
    executions_df = loaders.pipeline_executions()
    versions_df = loaders.dataset_versions()
    publication_df = loaders.publication_state()
    quality_df = loaders.quality_results()
    if runs_df.empty:
        hero_banner(
            "Sem histórico de execuções",
            "Nenhuma execução registrada ainda. Execute o pipeline para popular o histórico — "
            "as estatísticas são registradas automaticamente a cada execução.",
        )
    else:
        last = runs_df.iloc[0]
        active = None
        if not publication_df.empty:
            active = publication_df.iloc[0].get("active_version_id")
        active_text = f" Versão ativa: <code>{str(active)[:12]}</code>." if active else ""
        hero_banner(
            "Histórico de execuções",
            f"Última execução: <b>{last['stage']}</b> em "
            f"<code>{last['finished_at']}</code> — status: <b>{last['status']}</b>."
            f"{active_text}",
        )

    funnel_df = loaders.layer_funnel()
    row_counts = loaders.row_counts()

    if funnel_df.empty or funnel_df[["raw", "bronze", "silver", "gold"]].sum().sum() == 0:
        st.warning(
            "Nenhum dado encontrado em nenhuma camada ainda. Execute o pipeline "
            "(`uv run lake-research-map --stage all`) e recarregue esta página."
        )
        has_pipeline_data = False
    else:
        has_pipeline_data = True
        _headline_metrics(funnel_df)

    tab_flow, tab_quality, tab_audit, tab_operations = st.tabs(
        [
            "Fluxo e retenção",
            "Qualidade entre camadas",
            "Auditoria",
            "Execução e proveniência",
        ],
        on_change="rerun",
        key="pipeline_primary_tab",
    )

    if tab_flow.open:
        if has_pipeline_data:
            _sankey_funnel(funnel_df)
            _retention_by_stage(funnel_df)
        else:
            st.info("Execute a camada raw para iniciar o funil.")
    elif tab_quality.open:
        if not quality_df.empty:
            _render_contract_status(quality_df)
        elif has_pipeline_data:
            st.info(
                "Ainda não há contratos persistidos; o diagnóstico abaixo é apenas uma "
                "verificação legada sobre as tabelas carregadas."
            )
            _drift_check(funnel_df)
        if has_pipeline_data:
            _metadata_coverage_by_layer()
            st.subheader("Contagem bruta por tabela")
            st.dataframe(row_counts, hide_index=True, width="stretch")
        else:
            st.info("Ainda não há camadas para comparar.")
    elif tab_audit.open:
        _render_version_audit(versions_df, publication_df)
        _render_execution_audit(executions_df, runs_df)
        _render_source_changes()
        st.subheader("Registros rejeitados na camada silver")
        rejected_df = loaders.rejected_records()
        if rejected_df.empty:
            st.info("Nenhum registro rejeitado encontrado. Execute `--stage silver` para popular.")
        else:
            st.caption(
                f"{len(rejected_df):,} registros excluídos na camada silver — mantidos aqui para "
                "auditoria da revisão sistemática."
            )
            st.dataframe(rejected_df, hide_index=True, width="stretch")
    elif tab_operations.open:
        render_pipeline_controls()
        _render_search_provenance()


def _render_contract_status(quality_df: pd.DataFrame) -> None:
    st.subheader("Contratos executáveis")
    latest = quality_df.sort_values("checked_at").drop_duplicates(
        ["dataset_version_id", "stage", "check_id"], keep="last"
    )
    failures = latest[(latest["severity"] == "error") & (~latest["passed"].astype(bool))]
    warnings = latest[(latest["severity"] == "warning") & (~latest["passed"].astype(bool))]
    metric_row(
        [
            ("Verificações", f"{len(latest):,}", "último resultado por contrato"),
            ("Bloqueios", f"{len(failures):,}", "severidade error"),
            ("Alertas", f"{len(warnings):,}", "não bloqueantes"),
        ]
    )
    if failures.empty:
        st.success("Nenhuma falha bloqueante no conjunto mais recente de contratos.")
    else:
        st.error(
            f"{len(failures)} contrato(s) bloqueante(s) falharam; a versão candidata não pode "
            "ser publicada."
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
            "stage": "Etapa",
            "check_id": "Contrato",
            "severity": "Severidade",
            "passed": "Aprovado",
            "observed": "Observado",
            "expected": "Esperado",
            "checked_at": "Verificado em",
        }
    )
    st.dataframe(display, hide_index=True, width="stretch")


def _render_version_audit(versions_df: pd.DataFrame, publication_df: pd.DataFrame) -> None:
    st.subheader("Versões do corpus")
    if versions_df.empty:
        st.info("Nenhuma versão determinística foi registrada.")
        return
    active = None
    working = None
    if not publication_df.empty:
        active = publication_df.iloc[0].get("active_version_id")
        working = publication_df.iloc[0].get("working_version_id")
    metric_row(
        [
            ("Versão ativa", str(active)[:12] if active else "—", "Gold publicado"),
            ("Versão de trabalho", str(working)[:12] if working else "—", "pipeline atual"),
            ("Versões registradas", f"{len(versions_df):,}", None),
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
    st.subheader("Execuções e etapas correlacionadas")
    if executions_df.empty:
        if runs_df.empty:
            st.info("Nenhuma execução registrada. Execute o pipeline para popular o histórico.")
        else:
            st.dataframe(runs_df, hide_index=True, width="stretch")
        return
    st.dataframe(executions_df, hide_index=True, width="stretch")
    if not runs_df.empty:
        st.caption("Tentativas de etapa vinculadas às execuções acima.")
        st.dataframe(runs_df, hide_index=True, width="stretch")


def _render_source_changes() -> None:
    st.subheader("Reconciliação dos arquivos de origem")
    changes = loaders.source_changes()
    if changes.empty:
        st.info("Nenhuma reconciliação versionada foi registrada.")
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
                f"−{dropped} sem DOI" if dropped else "sem perdas",
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
        dropped_node = _idx("Descartado (sem DOI)")

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
    fig.update_layout(title="Volume de artigos por etapa e por base, com as perdas rotuladas")
    render_chart(
        fig,
        caption="A perda visível ('Descartado (sem DOI)') acontece no silver: artigos bronze sem DOI "
        "normalizado nunca chegam a formar um registro silver (`silver_articles.py`). Bronze pode conter "
        "mais linhas que a soma direta do raw porque é upsert-only e nunca remove linhas órfãs (ver o "
        "painel de drift abaixo).",
    )


def _retention_by_stage(funnel_df: pd.DataFrame) -> None:
    st.subheader("Retenção por etapa e por base")
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
    fig.update_layout(xaxis_title="Camada", yaxis_title="Quantidade de artigos")
    render_chart(
        fig,
        caption="Bronze deduplica apenas dentro de cada fonte (chave `(source, source_id)`); silver e gold "
        "então convergem para o mesmo total porque não há sobreposição de DOI entre IEEE e Elsevier neste "
        "corpus.",
    )


def _drift_check(funnel_df: pd.DataFrame) -> None:
    st.subheader("⚠️ Verificação de drift: bronze vs. raw")
    drift = funnel_df.copy()
    drift["drift"] = drift["bronze"] - drift["raw"]
    drift_display = drift[["source", "raw", "bronze", "drift"]].copy()
    drift_display["source"] = drift_display["source"].map(lambda s: SOURCE_LABELS.get(s, s))

    has_drift = (drift["drift"] != 0).any()
    st.dataframe(drift_display, hide_index=True, width="stretch")
    if has_drift:
        st.warning(
            "Bronze diverge do raw para pelo menos uma base. Bronze é upsert-only e nunca remove linhas "
            "órfãs (`_upsert` em `bronze_articles.py`) — se um arquivo `.bib`/CSV for removido de `data/`, "
            "suas linhas bronze permanecem. Um `bronze > raw` positivo é esse sintoma; investigue antes de "
            "confiar nas contagens de bronze como espelho fiel do raw atual."
        )
    else:
        st.success("Bronze e raw estão alinhados para as duas bases — sem sinal de drift.")


def _metadata_coverage_by_layer() -> None:
    st.subheader("Cobertura de metadados por camada")
    by_layer = loaders.articles_by_layer()

    fields = ["doi", "abstract", "keywords", "citation_count", "has_pdf"]
    field_labels = {
        "doi": "DOI",
        "abstract": "Resumo",
        "keywords": "Palavras-chave",
        "citation_count": "Citações",
        "has_pdf": "PDF vinculado",
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
        st.info("Nenhuma camada com dados suficientes para comparar cobertura de metadados.")
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
        labels={"field": "Campo", "coverage": "Preenchimento (%)", "layer": "Camada"},
    )
    fig.update_traces(hovertemplate="<b>%{x}</b><br>%{data.name}: %{y:.1f}%<extra></extra>")
    render_chart(
        fig,
        caption="Mostra o que cada camada ganha e perde: gold projeta silver descartando `issn`, `volume`, "
        "`issue`, `pages` e as flags de qualidade — mas mantém `sources` (adicionado nesta refatoração) "
        "para permitir a quebra IEEE/Elsevier também no gold.",
    )
