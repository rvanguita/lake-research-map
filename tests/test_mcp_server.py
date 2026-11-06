"""The MCP tools, called directly and through the registered server."""

from __future__ import annotations

import asyncio
import json

import pytest
from test_retrieval_api import _embed, _publish

from lake_research_map.retrieval import mcp_server
from lake_research_map.retrieval.api import Retriever
from lake_research_map.retrieval.index import load_corpus


@pytest.fixture
def configured(gold_session, vector_store):
    _publish(gold_session, vector_store)
    retriever = Retriever(
        load_corpus(gold_session),
        vector_store,
        embed_fn=_embed,
        token_len_fn=lambda texts: [10] * len(texts),
        context_window=512,
    )
    mcp_server.configure(retriever)
    yield retriever
    mcp_server.configure(None)


def test_tools_answer_from_the_injected_retriever(configured):
    found = mcp_server.search_articles("expansion planning", top_k=3, mode="dense")
    screened = mcp_server.screen_articles("expansion planning", off_subject="logistics", limit=2)
    article = mcp_server.get_article("10.1/a")
    evidence = mcp_server.get_evidence("10.1/a", "expansion planning", top_k=1)
    status = mcp_server.corpus_status()

    assert [hit["doi"] for hit in found["hits"]] == ["10.1/a", "10.1/b", "10.1/c"]
    assert [row["doi"] for row in screened["articles"]] == ["10.1/a", "10.1/b"]
    assert article["fulltext_chunks"] == 2
    assert [passage["seq"] for passage in evidence["passages"]] == [1]
    assert status["version_id"] == "v2"
    for payload in (found, screened, article, evidence, status):
        json.dumps(payload)


def test_unavailable_dense_retrieval_is_reported_not_raised(gold_session, vector_store):
    _publish(gold_session, vector_store)
    mcp_server.configure(Retriever(load_corpus(gold_session), None))
    try:
        result = mcp_server.search_articles("expansion planning", mode="dense")
        lexical = mcp_server.search_articles("feeder reconfiguration", mode="bm25")
    finally:
        mcp_server.configure(None)

    assert result["hits"] == [] and "vector store" in result["error"]
    assert lexical["hits"][0]["doi"] == "10.1/b"


def test_an_unknown_doi_is_an_error_payload(configured):
    assert "unknown DOI" in mcp_server.get_article("10.1/nope")["error"]


def test_the_server_registers_the_five_tools():
    server = mcp_server.build_server()

    # asyncio.run directly: the anyio pytest plugin is disabled (-p no:anyio).
    tools = asyncio.run(server.list_tools())

    assert {tool.name for tool in tools} == {
        "search_articles",
        "screen_articles",
        "get_article",
        "get_evidence",
        "corpus_status",
    }
    search = next(tool for tool in tools if tool.name == "search_articles")
    assert "query" in search.inputSchema["required"]
