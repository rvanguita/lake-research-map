"""Verify the Airflow DAG file is importable and defines the expected DAGs.

The DAG module imports `pendulum`, `airflow.sdk`, and
`airflow.providers.standard.operators.bash`, none of which are project
dependencies (they live inside the Airflow container). This test mocks them
so the module's *own* logic is exercised: every DAG ID the dashboard triggers
must appear, and the file must not raise on import.
"""

from __future__ import annotations

import importlib.util
import sys
import unittest.mock
from pathlib import Path
from unittest.mock import MagicMock


def test_dag_module_imports_and_defines_expected_dags():
    dag_instances: list[dict] = []
    bash_tasks: list[dict] = []

    class FakeDAG:
        def __init__(self, dag_id: str, **kwargs):
            self.dag_id = dag_id
            dag_instances.append({"dag_id": dag_id, "kwargs": kwargs})

        def __enter__(self):
            return self

        def __exit__(self, *a):
            pass

    class FakeTaskGroup:
        def __init__(self, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            pass

    class FakeBashOperator:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            bash_tasks.append(kwargs)

        def __rshift__(self, other):
            return other

    airflow_sdk_mock = MagicMock()
    airflow_sdk_mock.DAG = FakeDAG
    airflow_sdk_mock.TaskGroup = FakeTaskGroup

    bash_mock = MagicMock()
    bash_mock.BashOperator = FakeBashOperator

    pendulum_mock = MagicMock()
    pendulum_mock.datetime.return_value = MagicMock()

    modules_to_mock = {
        "pendulum": pendulum_mock,
        "airflow": MagicMock(),
        "airflow.sdk": airflow_sdk_mock,
        "airflow.providers": MagicMock(),
        "airflow.providers.standard": MagicMock(),
        "airflow.providers.standard.operators": MagicMock(),
        "airflow.providers.standard.operators.bash": bash_mock,
    }

    with unittest.mock.patch.dict(sys.modules, modules_to_mock):
        spec = importlib.util.spec_from_file_location(
            "lake_research_map_dags",
            str(
                Path(__file__).resolve().parents[1]
                / "airflow"
                / "dags"
                / "lake_research_map_dags.py"
            ),
        )
        assert spec is not None and spec.loader is not None
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

    dag_ids = {d["dag_id"] for d in dag_instances}

    # Every per-stage DAG that the dashboard can trigger individually.
    for stage in ("raw", "bronze", "silver", "gold", "embed", "semantic"):
        assert f"lake_research_map_{stage}" in dag_ids, f"missing DAG lake_research_map_{stage}"

    # The combined "run all" DAG.
    assert "lake_research_map_all" in dag_ids

    # Total: 6 per-stage + 1 combined = 7.
    assert len(dag_ids) == 7
    all_commands = [
        task["bash_command"] for task in bash_tasks if "--workflow all" in task["bash_command"]
    ]
    assert len(all_commands) == 6
    assert all("{{ dag.dag_id }}:{{ dag_run.run_id }}" in command for command in all_commands)
    assert all("--trigger airflow" in command for command in all_commands)
