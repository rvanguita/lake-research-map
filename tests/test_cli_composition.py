"""The `pipeline.py` / `runs.py` / `cli/` split keeps its two promises.

The command line moved out of `pipeline.py` into `cli/`, which only routes:
editing it must not mint a dataset version. And `run("all")` bootstraps the
schema once -- `run_all` used to repeat it, and nothing guarded that.
"""

from __future__ import annotations

import contextlib

import pipeline
from db import bootstrap as bootstrap_module
from ingest.snapshots import build_fingerprint


def test_editing_the_cli_does_not_mint_a_new_version(tmp_path):
    command = tmp_path / "src/cli/main.py"
    stage = tmp_path / "src/pipeline.py"
    command.parent.mkdir(parents=True)
    command.write_text("A = 1\n")
    stage.write_text("STAGES = ()\n")
    first = build_fingerprint([], repo_root=tmp_path)

    command.write_text("A = 2\n")
    second = build_fingerprint([], repo_root=tmp_path)

    assert second.code_sha256 == first.code_sha256
    assert second.version_id == first.version_id

    # The control: a stage edit still does, so the exclusion is not hashing nothing.
    stage.write_text("STAGES = ('raw',)\n")
    assert build_fingerprint([], repo_root=tmp_path).code_sha256 != first.code_sha256


def test_run_all_bootstraps_exactly_once(monkeypatch):
    calls: list[str] = []

    def counting_bootstrap():
        calls.append("bootstrap")

    @contextlib.contextmanager
    def unlocked(_execution_id):
        # False, as under SQLite: no exclusivity was proven, so no recovery sweep.
        yield False

    # Both names, so a lazy `from db.bootstrap import bootstrap` added to a
    # stage path would be counted too.
    monkeypatch.setattr(pipeline, "bootstrap", counting_bootstrap)
    monkeypatch.setattr(bootstrap_module, "bootstrap", counting_bootstrap)
    monkeypatch.setattr(pipeline, "_pipeline_lock", unlocked)
    ran: list[str] = []
    for stage in pipeline.STAGES[:-1]:
        monkeypatch.setattr(
            pipeline, f"run_{stage}", lambda *, _stage=stage, **_kwargs: ran.append(_stage)
        )

    pipeline.run("all", execution_id="execution-1")

    assert calls == ["bootstrap"]
    assert ran == list(pipeline.STAGES[:-1])
