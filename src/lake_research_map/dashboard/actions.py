"""Pipeline write-actions the dashboard can trigger in-process.

Every page under `dashboard/pages/` is meant to be read-only (loader ->
analytics -> chart -> render), with pipeline execution going through Airflow
via `pipeline_control.py`/`airflow_client.py` instead. The one exception is
generating embeddings from the Data Quality & RAG page, which runs synchronously
so it can drive a live progress bar. That db/transform access is kept here,
out of the page module, so `dashboard/pages/*.py` stays a read-only layer.
"""

from __future__ import annotations

from lake_research_map.db.engines import get_session
from lake_research_map.transform.embeddings import ProgressCallback, build_embeddings


def run_embedding_generation(on_progress: ProgressCallback | None = None) -> dict:
    """Run the `embed` stage synchronously against the gold layer.

    Returns the same stats dict as `transform.embeddings.build_embeddings`.
    """
    session = get_session("gold")
    try:
        return build_embeddings(session, on_progress=on_progress)
    finally:
        session.close()
