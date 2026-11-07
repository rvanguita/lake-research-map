"""`search`, `screen` and `mcp` on the command line."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from test_retrieval_api import _embed, _publish

import pipeline
from apps.api import cli
from apps.api.api import Retriever
from apps.api.index import load_corpus


@pytest.fixture
def retriever(gold_session, vector_store):
    _publish(gold_session, vector_store)
    return Retriever(
        load_corpus(gold_session),
        vector_store,
        embed_fn=_embed,
        token_len_fn=lambda texts: [10] * len(texts),
        context_window=512,
    )


def _args(**overrides) -> SimpleNamespace:
    fields = dict(
        command="search",
        query="expansion planning",
        subject="expansion planning",
        off_subject=None,
        limit=50,
        top_k=3,
        mode="hybrid",
        scope="abstract",
        rerank=False,
        candidates=50,
        year_min=None,
        year_max=None,
        sources=None,
        venues=None,
        publication_categories=None,
        min_margin=None,
        require_abstract=False,
        include_non_articles=False,
        json=True,
    )
    fields.update(overrides)
    return SimpleNamespace(**fields)


@pytest.mark.parametrize("command", ["search", "screen", "mcp"])
def test_retrieval_commands_never_take_the_pipeline_lock(command, monkeypatch):
    assert pipeline._mutates_published_inputs(SimpleNamespace(command=command)) is False

    def refuse(_execution_id):
        raise AssertionError("a read-only command must not take the pipeline lock")

    called = []
    monkeypatch.setattr(pipeline, "_pipeline_lock", refuse)
    monkeypatch.setattr(cli, "run", called.append)
    argv = [command] if command == "mcp" else [command, "distribution planning"]

    pipeline.main(argv)

    assert [args.command for args in called] == [command]


def test_run_prints_exactly_one_json_document(monkeypatch, capsys, retriever):
    monkeypatch.setattr(cli, "build_retriever", lambda: retriever)

    cli.run(_args())

    payload = json.loads(capsys.readouterr().out)
    assert payload["version_id"] == "v2"
    assert [hit["rank"] for hit in payload["hits"]] == [1, 2, 3]
    assert payload["hits"][0]["doi"] == "10.1/a"


def test_filter_flags_reach_the_ranking(retriever):
    payload = cli.search_payload(retriever, _args(mode="dense", year_min=2021))

    assert payload["filters"]["year_range"][0] == 2021
    assert {hit["doi"] for hit in payload["hits"]} == {"10.1/b", "10.1/d"}


def test_screen_ranks_by_margin_with_an_off_subject(retriever):
    payload = cli.screen_payload(
        retriever, _args(command="screen", off_subject="logistics", limit=0)
    )

    assert [row["doi"] for row in payload["articles"]][0] == "10.1/a"
    assert "subject_margin" in payload["articles"][0]
    assert "ranked by subject_margin" in cli.render_screen(payload)


def test_the_text_rendering_has_one_line_per_hit(retriever):
    payload = cli.search_payload(retriever, _args(json=False))

    lines = cli.render_hits(payload).splitlines()
    assert len(lines) == 1 + len(payload["hits"])
    assert "10.1/a" in lines[1]


def test_evidence_pooling_runs_every_mode_and_names_the_ones_it_skips(
    monkeypatch, tmp_path, caplog, gold_session, vector_store, silver_session
):
    import csv
    import logging

    from apps.api import api
    from db import engines

    _publish(gold_session, vector_store)

    def offline_reranker(_query, _texts):
        raise RuntimeError("no network for the reranker download")

    retriever = Retriever(
        load_corpus(gold_session),
        vector_store,
        embed_fn=_embed,
        rerank_fn=offline_reranker,
        token_len_fn=lambda texts: [10] * len(texts),
        context_window=512,
    )
    monkeypatch.setattr(api, "build_retriever", lambda version_id=None: retriever)
    monkeypatch.setattr(engines, "get_session", lambda layer: silver_session)
    output = tmp_path / "retrieval.csv"

    with caplog.at_level(logging.INFO, logger="pipeline"):
        pipeline._run_evidence_command(
            SimpleNamespace(evidence_action="retrieval", output=str(output), limit=3, seed=0)
        )

    with output.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows and {row["workflow"] for row in rows} == {"retrieval"}
    assert all("::10.1/" in row["subject_id"] for row in rows)
    summary = next(r.getMessage() for r in caplog.records if "pooled_modes" in r.getMessage())
    pooled, skipped = summary.split("skipped_modes")
    assert pooled.count(", all chunks'") == 3 and "'Hybrid RRF, abstracts'" in pooled
    assert "Hybrid RRF + rerank, abstracts" in skipped
