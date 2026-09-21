"""Multipage Streamlit dashboard for the lake-research-map medallion pipeline.

Run from the repo root with:
    uv run streamlit run main.py

Or via docker compose:
    docker compose up dashboard

`main()` is a function on purpose: Streamlit re-executes the entry script on
every rerun, and a function body re-runs with it. (A module whose top-level code
does the rendering only executes once per process, because Python caches
imports -- every later rerun would silently no-op.)
"""

from __future__ import annotations

import streamlit as st

from lake_research_map.dashboard.components import render_sidebar
from lake_research_map.dashboard.pages import (
    forecasting,
    highlights,
    overview,
    pipeline_layers,
    production,
    quality,
    researchers,
    semantics,
    synthesis,
    topics,
)

# Keep this registry small and stable: the URL slugs are public bookmarks while
# the grouped navigation communicates the analytical workflow.
PAGE_GROUPS = {
    "Visão": [
        (overview.render, "Visão geral", ":material/dashboard:", "overview"),
    ],
    "Análises": [
        (
            production.render,
            "Produção e periódicos",
            ":material/calendar_month:",
            "output-over-time",
        ),
        (topics.render, "Tópicos e estrutura científica", ":material/account_tree:", "topics"),
        (highlights.render, "Impacto e citações", ":material/insights:", "highlights"),
        (researchers.render, "Pesquisadores e colaboração", ":material/groups:", "researchers"),
        (synthesis.render, "Evidências de engenharia", ":material/science:", "synthesis"),
        (forecasting.render, "Tendências e frentes", ":material/trending_up:", "forecast"),
    ],
    "Revisão": [
        (semantics.render, "Triagem e descoberta", ":material/manage_search:", "semantics"),
        (quality.render, "Qualidade e RAG", ":material/fact_check:", "quality"),
    ],
    "Operação": [
        (pipeline_layers.render, "Pipeline e proveniência", ":material/schema:", "pipeline-layers"),
    ],
}

# Flat compatibility view used by lightweight import/registry checks.
PAGES = [page for definitions in PAGE_GROUPS.values() for page in definitions]


def main() -> None:
    st.set_page_config(
        page_title="lake-research-map", page_icon=":material/local_library:", layout="wide"
    )

    is_first = True
    navigation_groups: dict[str, list[st.Page]] = {}
    for group, definitions in PAGE_GROUPS.items():
        pages = []
        for render, title, icon, url_path in definitions:
            pages.append(
                st.Page(
                    render,
                    title=title,
                    icon=icon,
                    url_path=url_path,
                    default=is_first,
                )
            )
            is_first = False
        navigation_groups[group] = pages

    st.sidebar.header("lake-research-map")
    navigation = st.navigation(navigation_groups)
    render_sidebar()
    navigation.run()


if __name__ == "__main__":
    main()
