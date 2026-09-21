"""Semantic search over `gold.chunks` using the embeddings the `embed` pipeline
stage already produced.

`_rank_by_similarity` is a pure function (no Streamlit, no model loading) so it
can be unit tested directly with a hand-built query vector -- see
`tests/test_search.py`. `semantic_search` wires it up to the real embedding
model, cached across Streamlit reruns with `st.cache_resource` so the ONNX
model is loaded once per process, not once per query.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from lake_research_map.transform.embeddings import EMBED_MODEL_NAME


def _parse_embedding(emb_bin) -> np.ndarray | None:
    """Parse a stored vector. Binary is the only canonical representation.

    The JSON mirror is no longer written (see `transform/versioned_gold.py`),
    and every published version carries the binary column, so falling back to
    JSON only kept a slower parse alive for rows that do not need it.
    """
    if emb_bin is None:
        return None
    return np.frombuffer(emb_bin, dtype=np.float32).copy()


def _rank_by_similarity(
    query_vector: np.ndarray, chunks_df: pd.DataFrame, top_k: int
) -> pd.DataFrame:
    """Rank `chunks_df` by cosine similarity of `embedding_bin` to `query_vector`.

    Rows with no stored vector are excluded. Returns a copy of the top_k
    matching rows with an added `score` column, sorted descending. Empty
    input (no column, no embedded rows) returns an empty frame.
    """
    if "embedding_bin" not in chunks_df.columns:
        return chunks_df.iloc[0:0].copy()

    embedded = chunks_df[chunks_df["embedding_bin"].notna()]
    if embedded.empty:
        return embedded.copy()

    matrix = np.stack(
        [np.frombuffer(value, dtype=np.float32) for value in embedded["embedding_bin"]]
    )
    query = np.asarray(query_vector, dtype=np.float32)
    denominator = np.linalg.norm(matrix, axis=1) * np.linalg.norm(query)
    scores = np.divide(
        matrix @ query,
        denominator,
        out=np.full(len(matrix), -np.inf, dtype=np.float32),
        where=denominator > 0,
    )

    limit = min(max(int(top_k), 0), len(scores))
    if limit == 0:
        return embedded.iloc[0:0].assign(score=pd.Series(dtype=float))
    candidate_indices = np.argpartition(scores, -limit)[-limit:]
    ranked_indices = candidate_indices[np.argsort(scores[candidate_indices])[::-1]]
    result = embedded.iloc[ranked_indices].copy()
    result["score"] = scores[ranked_indices]
    return result.reset_index(drop=True)


@st.cache_resource
def _get_embedding_model():
    # Imported here so pages that never call semantic_search don't pay
    # fastembed's import cost or trigger a model-file check.
    from fastembed import TextEmbedding

    return TextEmbedding(model_name=EMBED_MODEL_NAME)


def semantic_search(query: str, chunks_df: pd.DataFrame, top_k: int = 10) -> pd.DataFrame:
    """Embed `query` with the same model used for `chunks.embedding` and rank
    `chunks_df` by cosine similarity. Returns an empty frame if no chunk in
    `chunks_df` has an embedding yet.
    """
    has_emb = "embedding" in chunks_df.columns and chunks_df["embedding"].notna().any()
    has_bin = "embedding_bin" in chunks_df.columns and chunks_df["embedding_bin"].notna().any()
    if not has_emb and not has_bin:
        return chunks_df.iloc[0:0].copy()

    model = _get_embedding_model()
    query_vector = np.array(next(model.embed([query])))
    return _rank_by_similarity(query_vector, chunks_df, top_k)


def build_vector_index(matrix: np.ndarray):
    """Build an accelerated k-NN vector search index (NearestNeighbors with cosine metric).

    Enables efficient neighborhood indexing for fast vector retrieval at scale.
    """
    from sklearn.neighbors import NearestNeighbors

    if len(matrix) == 0:
        return None
    index = NearestNeighbors(n_neighbors=min(50, len(matrix)), metric="cosine", algorithm="auto")
    index.fit(matrix)
    return index


def search_vector_index(
    index, query_vector: np.ndarray, top_k: int = 10
) -> tuple[np.ndarray, np.ndarray]:
    """Query the accelerated vector index for top_k nearest neighbors.

    Returns:
    - indices: array of integer row indices in the original matrix
    - similarities: array of cosine similarity scores (1.0 - cosine distance)
    """
    if index is None:
        return np.array([], dtype=int), np.array([], dtype=float)

    query = query_vector.reshape(1, -1)
    k = min(top_k, index.n_samples_fit_)
    distances, indices = index.kneighbors(query, n_neighbors=k)
    similarities = 1.0 - distances[0]
    return indices[0], similarities


# --- Hybrid Search: BM25 Okapi & Reciprocal Rank Fusion (RRF) --------------


def bm25_search(
    query: str,
    chunks_df: pd.DataFrame,
    top_k: int = 20,
    k1: float = 1.5,
    b: float = 0.75,
) -> pd.DataFrame:
    """Perform BM25 Okapi lexical search over chunk texts.

    Pure NumPy/Python implementation with zero extra dependencies.
    Ideal for catching exact acronyms, model names (e.g. "IEEE 33-bus", "SOCP", "MILP")
    and technical constants that dense embeddings may overlook.
    """
    import re
    from collections import Counter

    if chunks_df.empty or not query.strip():
        res = chunks_df.iloc[0:0].copy()
        res["bm25_score"] = 0.0
        return res

    text_col = "text" if "text" in chunks_df.columns else chunks_df.columns[0]
    texts = chunks_df[text_col].fillna("").astype(str).tolist()
    n_docs = len(texts)
    if n_docs == 0:
        res = chunks_df.iloc[0:0].copy()
        res["bm25_score"] = 0.0
        return res

    # Tokenize query and docs
    query_tokens = [w for w in re.findall(r"\b\w+\b", query.lower()) if len(w) > 1]
    if not query_tokens:
        res = chunks_df.iloc[0:0].copy()
        res["bm25_score"] = 0.0
        return res

    doc_tokens = [re.findall(r"\b\w+\b", t.lower()) for t in texts]
    doc_lens = np.array([max(len(tokens), 1) for tokens in doc_tokens], dtype=float)
    avgdl = float(np.mean(doc_lens)) if len(doc_lens) > 0 else 1.0

    # Calculate document frequencies
    doc_counters = [Counter(tokens) for tokens in doc_tokens]
    scores = np.zeros(n_docs, dtype=float)

    for q in query_tokens:
        doc_freq = sum(1 for c in doc_counters if q in c)
        if doc_freq == 0:
            continue
        # Okapi BM25 IDF
        idf = float(np.log(1.0 + (n_docs - doc_freq + 0.5) / (doc_freq + 0.5)))
        for i, c in enumerate(doc_counters):
            tf = float(c.get(q, 0))
            if tf > 0:
                denom = tf + k1 * (1.0 - b + b * (doc_lens[i] / avgdl))
                scores[i] += idf * (tf * (k1 + 1.0)) / max(denom, 1e-9)

    matched_indices = np.where(scores > 0)[0]
    if len(matched_indices) == 0:
        res = chunks_df.iloc[0:0].copy()
        res["bm25_score"] = 0.0
        return res

    ranked_order = matched_indices[np.argsort(-scores[matched_indices])][:top_k]
    result = chunks_df.iloc[ranked_order].copy()
    result["bm25_score"] = scores[ranked_order]
    return result.reset_index(drop=True)


def hybrid_search_rrf(
    query: str,
    chunks_df: pd.DataFrame,
    top_k: int = 10,
    rrf_k: int = 60,
    query_vector: np.ndarray | None = None,
) -> pd.DataFrame:
    """Hybrid search combining Dense Cosine Retrieval with Sparse BM25 via RRF.

    Applies Reciprocal Rank Fusion (Cormack et al., SIGIR 2009):
        RRF(d) = sum_{m in {dense, bm25}} 1.0 / (rrf_k + rank_m(d))
    Combines conceptual semantic matching with lexical precision.
    """
    if chunks_df.empty or not query.strip():
        return chunks_df.iloc[0:0].copy()

    # Retrieve top 50 dense candidates
    if query_vector is not None:
        dense_df = _rank_by_similarity(query_vector, chunks_df, top_k=50)
    else:
        dense_df = semantic_search(query, chunks_df, top_k=50)

    # Retrieve top 50 BM25 lexical candidates
    bm25_df = bm25_search(query, chunks_df, top_k=50)

    if dense_df.empty and bm25_df.empty:
        return chunks_df.iloc[0:0].copy()

    # Map candidate IDs/indexes to RRF scores
    candidate_scores: dict[int, float] = {}
    dense_score_map: dict[int, float] = {}
    bm25_score_map: dict[int, float] = {}
    candidate_rows: dict[int, pd.Series] = {}

    for rank, (idx, row) in enumerate(dense_df.iterrows()):
        c_id = int(row.get("id", idx))
        dense_score_map[c_id] = float(row.get("score", 0.0))
        candidate_scores[c_id] = candidate_scores.get(c_id, 0.0) + (1.0 / (rrf_k + rank + 1))
        candidate_rows[c_id] = row

    for rank, (idx, row) in enumerate(bm25_df.iterrows()):
        c_id = int(row.get("id", idx))
        bm25_score_map[c_id] = float(row.get("bm25_score", 0.0))
        candidate_scores[c_id] = candidate_scores.get(c_id, 0.0) + (1.0 / (rrf_k + rank + 1))
        if c_id not in candidate_rows:
            candidate_rows[c_id] = row

    sorted_candidates = sorted(candidate_scores.items(), key=lambda x: -x[1])[:top_k]

    rows = []
    for c_id, rrf_s in sorted_candidates:
        r = candidate_rows[c_id].to_dict()
        r["score"] = round(rrf_s, 5)
        r["dense_score"] = round(dense_score_map.get(c_id, 0.0), 4)
        r["bm25_score"] = round(bm25_score_map.get(c_id, 0.0), 3)
        rows.append(r)

    return pd.DataFrame(rows)
