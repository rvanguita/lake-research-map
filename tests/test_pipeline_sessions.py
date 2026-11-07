"""Every session a stage opens is closed, whatever fails and wherever.

The stage functions used to open their sessions and then call
`_working_version` / `_ensure_execution` / `_begin_stage` before entering their
`try/finally`, so a failure there leaked the sessions. These tests hand out
fake sessions that record `close()` and fail at those points.
"""

from __future__ import annotations

import pytest

import pipeline
import runs


class _FakeSession:
    def __init__(self, layer: str):
        self.layer = layer
        self.closed = False

    def commit(self):
        pass

    def rollback(self):
        pass

    def close(self):
        self.closed = True


class _Boom(RuntimeError):
    pass


@pytest.fixture
def handed_out(monkeypatch):
    sessions: list[_FakeSession] = []

    def fake_get_session(layer):
        session = _FakeSession(layer)
        sessions.append(session)
        return session

    monkeypatch.setattr(runs, "get_session", fake_get_session)
    monkeypatch.setattr(
        pipeline,
        "_prepare_raw_version",
        lambda *args, **kwargs: ([], "v" * 64, False),
    )
    monkeypatch.setattr(pipeline, "_working_version", lambda session, execution_id: "v" * 64)
    monkeypatch.setattr(pipeline, "_ensure_execution", lambda session, execution_id, **kw: None)
    return sessions


def _raise(*_args, **_kwargs):
    raise _Boom("injected")


_STAGES = {
    "raw": (pipeline.run_raw, {"gold", "raw"}),
    "bronze": (pipeline.run_bronze, {"gold", "raw", "bronze"}),
    "silver": (pipeline.run_silver, {"gold", "raw", "bronze", "silver"}),
    "gold": (pipeline.run_gold, {"gold", "silver"}),
    "embed": (pipeline.run_embed, {"gold"}),
    "semantic": (pipeline.run_semantic, {"gold"}),
}


@pytest.mark.parametrize("stage", list(_STAGES))
def test_begin_stage_failure_closes_every_session(monkeypatch, handed_out, stage):
    runner, layers = _STAGES[stage]
    monkeypatch.setattr(pipeline, "_begin_stage", _raise)

    with pytest.raises(_Boom):
        runner(execution_id="execution-1")

    assert {session.layer for session in handed_out} == layers
    # One session per layer: a stray direct get_session() beside _layer_sessions would show up here.
    assert len(handed_out) == len(layers)
    assert all(session.closed for session in handed_out), [
        session.layer for session in handed_out if not session.closed
    ]


@pytest.mark.parametrize("stage", [name for name in _STAGES if name != "raw"])
def test_working_version_failure_closes_every_session(monkeypatch, handed_out, stage):
    runner, layers = _STAGES[stage]
    monkeypatch.setattr(pipeline, "_working_version", _raise)

    with pytest.raises(_Boom):
        runner(execution_id="execution-1")

    assert {session.layer for session in handed_out} == layers
    assert len(handed_out) == len(layers)
    assert all(session.closed for session in handed_out)


def test_a_failed_open_closes_the_sessions_opened_before_it(monkeypatch):
    sessions: list[_FakeSession] = []

    def fake_get_session(layer):
        if layer == "bronze":
            raise _Boom("cannot connect")
        session = _FakeSession(layer)
        sessions.append(session)
        return session

    monkeypatch.setattr(runs, "get_session", fake_get_session)

    with pytest.raises(_Boom):
        with runs._layer_sessions("gold", "raw", "bronze"):
            raise AssertionError("the block must not run when an open fails")

    assert [session.layer for session in sessions] == ["gold", "raw"]
    assert all(session.closed for session in sessions)


def test_layer_sessions_yields_in_order_and_closes_on_success(monkeypatch):
    monkeypatch.setattr(runs, "get_session", _FakeSession)

    with runs._layer_sessions("gold", "silver") as (gold, silver):
        assert (gold.layer, silver.layer) == ("gold", "silver")
        assert not gold.closed and not silver.closed

    assert gold.closed and silver.closed
