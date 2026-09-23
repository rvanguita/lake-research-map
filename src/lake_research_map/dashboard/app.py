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
    "Overview": [
        (overview.render, "Overview", ":material/dashboard:", "overview"),
    ],
    "Analytics": [
        (
            production.render,
            "Production and venues",
            ":material/calendar_month:",
            "output-over-time",
        ),
        (topics.render, "Topics and scientific structure", ":material/account_tree:", "topics"),
        (highlights.render, "Impact and citations", ":material/insights:", "highlights"),
        (researchers.render, "Researchers and collaboration", ":material/groups:", "researchers"),
        (synthesis.render, "Engineering evidence", ":material/science:", "synthesis"),
        (forecasting.render, "Trends and fronts", ":material/trending_up:", "forecast"),
    ],
    "Review": [
        (semantics.render, "Screening and discovery", ":material/manage_search:", "semantics"),
        (quality.render, "Quality and RAG", ":material/fact_check:", "quality"),
    ],
    "Operations": [
        (pipeline_layers.render, "Pipeline and provenance", ":material/schema:", "pipeline-layers"),
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
