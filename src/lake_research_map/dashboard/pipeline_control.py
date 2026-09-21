"""Dashboard-facing pipeline control -- triggers Airflow DAG runs instead of
executing the pipeline in-process. Each medallion stage maps 1:1 to its own
Airflow DAG (see airflow/dags/lake_research_map_dags.py), so a specific stage
can be re-run without redoing the others, plus a combined "run everything"
DAG for convenience.
"""

from __future__ import annotations

from lake_research_map.dashboard import airflow_client

STAGE_LABELS = {
    "raw": "Gross Data (Raw)",
    "bronze": "Bronze (Padronizado)",
    "silver": "Silver (Limpo e Deduplicado)",
    "gold": "Gold (RAG)",
    "embed": "Embeddings (RAG)",
    "semantic": "Semantics (Relevance and Themes)",
    "all": "Todas as Etapas",
}

DAG_IDS = {
    "raw": "lake_research_map_raw",
    "bronze": "lake_research_map_bronze",
    "silver": "lake_research_map_silver",
    "gold": "lake_research_map_gold",
    "embed": "lake_research_map_embed",
    "semantic": "lake_research_map_semantic",
    "all": "lake_research_map_all",
}


def trigger_stage(stage: str) -> dict:
    """Trigger the DAG for `stage` (one of raw/bronze/silver/gold/all).

    Returns a run reference to keep in session state and pass to `poll_run`:
    `{"stage": ..., "dag_id": ..., "dag_run_id": ..., "state": "queued"}`.
    """
    if stage not in DAG_IDS:
        raise ValueError(f"unknown stage {stage!r}, expected one of {list(DAG_IDS)}")
    dag_id = DAG_IDS[stage]
    dag_run_id = airflow_client.trigger_dag(dag_id)
    return {"stage": stage, "dag_id": dag_id, "dag_run_id": dag_run_id, "state": "queued"}


def poll_run(run_ref: dict) -> dict:
    """Refresh a run reference's `state` (and `task_instances` for multi-task
    DAGs) from Airflow. Returns a new dict; does not mutate the input.
    """
    dag_run = airflow_client.get_dag_run(run_ref["dag_id"], run_ref["dag_run_id"])
    updated = dict(run_ref, state=dag_run.get("state", run_ref["state"]))
    if updated["state"] not in ("queued", "running", "scheduled"):
        updated["task_instances"] = airflow_client.get_task_instances(
            run_ref["dag_id"], run_ref["dag_run_id"]
        )
    return updated
