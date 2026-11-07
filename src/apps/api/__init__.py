"""Article-level retrieval over the published Gold version (FR-14).

Streamlit-free on purpose: the CLI (`search`, `screen`), the MCP server and the
dashboard all call the same `Retriever`, so a RAG client gets exactly the
ranking a reviewer sees. Nothing here imports `apps/dashboard/`.
"""
