"""`Retriever` over a hand-built published version: SQLite Gold + local Qdrant."""

from __future__ import annotations

import json

import numpy as np
import pytest

from apps.api.api import (
    BENCHMARK_MODES,
    RetrievalUnavailable,
    Retriever,
    mode_runner,
)
from apps.api.filters import ArticleFilters
from apps.api.index import load_corpus
from db.gold_models import (
    DatasetArticle,
    DatasetChunk,
    DatasetSemantics,
    DatasetVersion,
    PublicationState,
)
from db.vector_store import VectorPoint, point_id
from transform.vector_payloads import chunk_payload

LONG = "a sufficiently long abstract text so the abstract does not count as missing. "

# doi -> (year, non_article, abstract, relevance, offtopic, chunks{(type, seq): (text, vector)})
CORPUS = {
    "10.1/a": (
        2020,
        False,
        LONG + "distribution network expansion planning",
        0.80,
        0.60,
        {
            ("abstract", 0): ("Expansion planning of distribution networks", [1.0, 0.0, 0.0]),
            ("fulltext", 1): ("MILP expansion planning model for feeders", [0.97, 0.03, 0.0]),
            ("fulltext", 2): ("Results on a radial test system", [0.30, 0.70, 0.0]),
        },
    ),
    "10.1/b": (
        2022,
        False,
        LONG + "feeder reconfiguration",
        0.75,
        0.62,
        {("abstract", 0): ("Feeder reconfiguration for loss reduction", [0.90, 0.10, 0.0])},
    ),
    "10.1/c": (
        2019,
        False,
        LONG + "cold chain logistics",
        0.60,
        0.72,
        {("abstract", 0): ("Cold chain logistics facility location", [0.0, 1.0, 0.0])},
    ),
    "10.1/d": (
        2024,
        False,
        "short",
        None,
        None,
        {("abstract", 0): ("Distribution planning", [0.0, 0.0, 1.0])},
    ),
    "10.1/e": (
        2021,
        True,
        LONG + "front matter",
        0.70,
        0.65,
        {("abstract", 0): ("Expansion planning front matter", [0.80, 0.20, 0.0])},
    ),
}


def _publish(gold_session, vector_store, version_id: str = "v2") -> None:
    gold_session.add(DatasetVersion(version_id=version_id, status="active"))
    points = []
    for doi, (year, non_article, abstract, rel, off, chunks) in CORPUS.items():
        article = DatasetArticle(
            dataset_version_id=version_id,
            doi=doi,
            sources=["ieee"],
            title=f"Title {doi}",
            year=year,
            venue="Venue",
            abstract=abstract,
            is_non_article=non_article,
            has_pdf=len(chunks) > 1,
            authors=["A. Author"],
        )
        gold_session.add(article)
        semantics = None
        if rel is not None:
            semantics = DatasetSemantics(
                dataset_version_id=version_id,
                doi=doi,
                relevance_score=rel,
                offtopic_score=off,
                theme_id=0,
                theme_label="Planning",
                map_x=0.0,
                map_y=0.0,
            )
            gold_session.add(semantics)
        for (chunk_type, seq), (text, vector) in chunks.items():
            gold_session.add(
                DatasetChunk(
                    dataset_version_id=version_id,
                    doi=doi,
                    chunk_type=chunk_type,
                    seq=seq,
                    text=text,
                    char_len=len(text),
                )
            )
            points.append(
                VectorPoint(
                    point_id(doi, chunk_type, seq),
                    np.asarray(vector, dtype=np.float32),
                    chunk_payload(
                        version_id=version_id,
                        doi=doi,
                        chunk_type=chunk_type,
                        seq=seq,
                        text=text,
                        text_sha256="h",
                        embedded_at=None,
                        article=article,
                        semantics=semantics,
                    ),
                )
            )
    gold_session.merge(PublicationState(id=1, active_version_id=version_id))
    gold_session.commit()
    vector_store.ensure_collection(version_id, 3)
    vector_store.upsert(version_id, points)


QUERY_VECTORS = {
    "expansion planning": [1.0, 0.0, 0.0],
    "logistics": [0.0, 1.0, 0.0],
}


def _embed(text: str):
    return QUERY_VECTORS.get(text, [0.0, 0.0, 1.0])


@pytest.fixture
def retriever(gold_session, vector_store):
    _publish(gold_session, vector_store)
    corpus = load_corpus(gold_session)
    return Retriever(
        corpus,
        vector_store,
        embed_fn=_embed,
        token_len_fn=lambda texts: [600 if "Expansion planning of" in t else 20 for t in texts],
        context_window=512,
    )


def test_load_corpus_reads_only_the_published_version(gold_session, vector_store):
    assert load_corpus(gold_session) is None
    gold_session.add(DatasetVersion(version_id="v1", status="superseded"))
    gold_session.add(DatasetArticle(dataset_version_id="v1", doi="10.1/old", sources=[]))
    gold_session.commit()
    _publish(gold_session, vector_store, "v2")

    corpus = load_corpus(gold_session)

    assert corpus.version_id == "v2"
    assert "10.1/old" not in corpus.articles.index
    assert corpus.articles.loc["10.1/c", "relevance_margin"] == pytest.approx(-0.12)
    assert np.isnan(corpus.articles.loc["10.1/d", "relevance_margin"])
    assert corpus.articles.loc["10.1/d", "abstract_missing"]
    assert corpus.fulltext_articles == 1


def test_dense_search_ranks_one_hit_per_article(retriever):
    hits = retriever.search("expansion planning", mode="dense", top_k=3)

    assert [hit.doi for hit in hits] == ["10.1/a", "10.1/b", "10.1/c"]
    assert all(hit.chunk_type == "abstract" for hit in hits)
    assert hits[0].dense_rank == 1 and hits[0].bm25_rank is None
    # The non-article front matter (0.98 cosine) is excluded by default.
    assert "10.1/e" not in {hit.doi for hit in hits}


def test_the_all_chunks_scope_ranks_an_article_by_its_best_chunk(retriever):
    hits = retriever.search("logistics", mode="dense", scope="all", top_k=5)

    dois = [hit.doi for hit in hits]
    assert len(dois) == len(set(dois))
    # 10.1/a ranks through its results chunk (0.70 to the logistics axis).
    a = next(hit for hit in hits if hit.doi == "10.1/a")
    assert (a.chunk_type, a.chunk_seq) == ("fulltext", 2)
    assert dois[0] == "10.1/c"


def test_filters_apply_before_ranking_so_top_k_is_filled(retriever):
    hits = retriever.search(
        "expansion planning",
        mode="dense",
        top_k=2,
        filters=ArticleFilters(year_range=(2021, 2030)),
    )

    assert [hit.doi for hit in hits] == ["10.1/b", "10.1/d"]


def test_bm25_never_embeds_and_needs_no_store(gold_session, vector_store):
    _publish(gold_session, vector_store)

    def refuse(_text):
        raise AssertionError("a lexical search must not embed the query")

    retriever = Retriever(load_corpus(gold_session), None, embed_fn=refuse)
    hits = retriever.search("feeder reconfiguration", mode="bm25", top_k=3)

    assert hits[0].doi == "10.1/b"
    assert hits[0].abstract_truncated is None
    with pytest.raises(RetrievalUnavailable):
        retriever.search("feeder", mode="dense")


def test_hybrid_fuses_both_rankings_by_reciprocal_rank(retriever):
    hits = retriever.search("expansion planning", mode="hybrid", top_k=5)

    top = hits[0]
    assert top.doi == "10.1/a"
    assert top.score == pytest.approx(1 / (60 + top.dense_rank) + 1 / (60 + top.bm25_rank))
    assert [hit.rank for hit in hits] == list(range(1, len(hits) + 1))


def test_rerank_reorders_the_fused_candidates(gold_session, vector_store):
    _publish(gold_session, vector_store)
    retriever = Retriever(
        load_corpus(gold_session),
        vector_store,
        embed_fn=_embed,
        rerank_fn=lambda _query, texts: [10.0 if "Feeder" in text else 0.0 for text in texts],
        token_len_fn=lambda texts: [10] * len(texts),
        context_window=512,
    )

    hits = retriever.search("expansion planning", mode="hybrid", rerank=True, top_k=3)

    assert hits[0].doi == "10.1/b"
    assert hits[0].rerank_rank == 1 and hits[0].mode == "hybrid+rerank"


def test_hits_flag_missing_and_truncated_abstracts(retriever):
    hits = {hit.doi: hit for hit in retriever.search("expansion planning", mode="dense", top_k=4)}

    assert hits["10.1/a"].abstract_truncated is True
    assert hits["10.1/b"].abstract_truncated is False
    assert hits["10.1/d"].abstract_missing is True


def test_screen_scores_every_eligible_article_and_ranks_by_margin(retriever):
    frame = retriever.screen("expansion planning", off_subject="logistics")

    assert set(frame["doi"]) == {"10.1/a", "10.1/b", "10.1/c", "10.1/d"}
    assert frame["doi"].iloc[0] == "10.1/a"
    assert frame["doi"].iloc[-1] == "10.1/c"
    assert list(frame["rank"]) == [1, 2, 3, 4]
    assert (frame["subject_margin"].diff().dropna() <= 0).all()


def test_evidence_returns_passages_of_one_article_only(retriever):
    passages = retriever.evidence("10.1/a", "expansion planning", top_k=2)

    assert [passage.seq for passage in passages] == [1, 2]
    assert all(passage.doi == "10.1/a" for passage in passages)
    assert retriever.evidence("10.1/b", "expansion planning") == []


def test_status_reports_counts_alias_and_truncation(retriever, vector_store):
    vector_store.point_alias("v2")

    status = retriever.status(measure_truncation=True)

    assert (status["articles"], status["abstract_chunks"], status["fulltext_chunks"]) == (5, 5, 2)
    assert status["abstract_missing"] == 1
    assert status["points"] == 7 and status["alias_in_sync"] is True
    assert (status["abstract_truncated"], status["context_window_tokens"]) == (1, 512)


def test_hits_and_articles_serialize_to_json(retriever):
    hits = retriever.search("expansion planning", mode="hybrid", top_k=5)
    record = retriever.article("10.1/d")

    json.dumps([hit.to_dict() for hit in hits])
    json.dumps(record)
    assert record["relevance_margin"] is None and record["year"] == 2024
    assert retriever.article("10.1/missing") is None


def test_benchmark_modes_search_the_whole_population(retriever):
    run = mode_runner(retriever, BENCHMARK_MODES["Dense, abstracts"])

    # The judgements were pooled over every article, front matter included.
    assert "10.1/e" in run("expansion planning", 5)
    assert len(BENCHMARK_MODES) == 7
