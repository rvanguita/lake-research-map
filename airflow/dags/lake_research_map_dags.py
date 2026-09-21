"""Airflow DAGs for the lake-research-map medallion pipeline.

Six DAGs, matching the six stage buttons in the Streamlit dashboard 1:1:
`lake_research_map_raw/bronze/silver/gold/embed/semantic` (one task each) and
`lake_research_map_all`, which groups the six chained stage tasks into a
single `medallion_pipeline` TaskGroup so the Airflow UI graph reads as one
connected flow -- raw -> bronze -> silver -> gold -> embed -> semantic -- rather than a
bare chain of same-level tasks. Every task just shells out to the same
`uv run lake-research-map --stage <stage>` entrypoint the CLI uses -- this file
intentionally does not import `lake_research_map` directly, so the pipeline
logic (ingest/transform/pipeline.py) needs zero changes to be orchestrated by
Airflow.

The individual per-stage DAGs stay separate (not folded into the group)
because the dashboard's sidebar triggers them 1:1 by DAG id (see
`dashboard/pipeline_control.py::DAG_IDS`) to let a user re-run just one
stage; only the "run everything" DAG needed the flow to read as one unit.

All DAGs are `schedule=None` -- manual/API trigger only, no cron schedule --
since the pipeline is meant to be run on demand from the dashboard.
"""

from __future__ import annotations

from datetime import timedelta

import pendulum
from airflow.providers.standard.operators.bash import BashOperator
from airflow.sdk import DAG, TaskGroup

PROJECT_DIR = "/opt/airflow/project"
START_DATE = pendulum.datetime(2024, 1, 1, tz="UTC")

STAGES = ("raw", "bronze", "silver", "gold", "embed", "semantic")
STAGE_POLICY = {
    "raw": {"retries": 1, "execution_timeout": timedelta(minutes=30)},
    "bronze": {"retries": 0, "execution_timeout": timedelta(minutes=20)},
    "silver": {"retries": 0, "execution_timeout": timedelta(minutes=20)},
    "gold": {"retries": 0, "execution_timeout": timedelta(minutes=30)},
    "embed": {"retries": 1, "execution_timeout": timedelta(hours=6)},
    "semantic": {"retries": 0, "execution_timeout": timedelta(hours=2)},
}


def _bash_command(stage: str, workflow: str) -> str:
    execution_id = "{{ dag.dag_id }}:{{ dag_run.run_id }}"
    return (
        f"cd {PROJECT_DIR} && uv run lake-research-map --stage {stage} "
        f"--execution-id '{execution_id}' --trigger airflow --workflow {workflow}"
    )


default_args = {
    "owner": "lake-research-map",
    "retry_delay": timedelta(minutes=2),
}

# One single-task DAG per stage, mirroring the dashboard's individual buttons.
for stage in STAGES:
    with DAG(
        dag_id=f"lake_research_map_{stage}",
        description=f"Run the {stage} stage of the lake-research-map medallion pipeline.",
        schedule=None,
        start_date=START_DATE,
        catchup=False,
        default_args=default_args,
        tags=["lake-research-map"],
    ):
        BashOperator(
            task_id=stage,
            bash_command=_bash_command(stage, stage),
            **STAGE_POLICY[stage],
        )

# One combined DAG running the full medallion flow as a single grouped unit,
# for the "run all" button.
with DAG(
    dag_id="lake_research_map_all",
    description="The lake-research-map medallion pipeline as one flow: raw->bronze->silver->gold->embed->semantic.",
    schedule=None,
    start_date=START_DATE,
    catchup=False,
    default_args=default_args,
    tags=["lake-research-map", "pipeline"],
):
    with TaskGroup(group_id="medallion_pipeline") as medallion_pipeline:
        tasks = [
            BashOperator(
                task_id=stage,
                bash_command=_bash_command(stage, "all"),
                **STAGE_POLICY[stage],
            )
            for stage in STAGES
        ]
        for upstream, downstream in zip(tasks, tasks[1:], strict=False):
            upstream >> downstream
