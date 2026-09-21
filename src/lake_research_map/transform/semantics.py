"""Per-article semantic signals derived from the abstract embeddings.

Runs as `--stage semantic`, after `embed` (which itself needs `gold`): it reads
`lit_chunks.embedding` and writes `lit_semantics` / `lit_duplicate_pairs`.
Re-running `gold` rebuilds the chunks, so this stage must be re-run after it.

Why this exists: the corpus was assembled from a search for "distribution
system planning", which is ambiguous -- it matches electric power distribution
*and* logistics/supply-chain distribution. Roughly a tenth of the corpus turned
out to be facility-location and cold-chain papers, plus book front matter
("Preface", "Index") ingested as if it were an article. Relevance screening is
a core step of a systematic literature review, so instead of hiding that, every
article is scored against *both* readings of the query and the dashboard lets a
reviewer act on the margin between them.

The functions below take plain numpy arrays so they can be tested without a
database or the embedding model; `build_semantics` is the only part that does
I/O.
"""

from __future__ import annotations

import logging

import numpy as np
from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session

from lake_research_map.db.gold_models import Chunk, DuplicateOverride, DuplicatePair, Semantics
from lake_research_map.transform.embeddings import EMBED_MODEL_NAME

logger = logging.getLogger(__name__)

# What the review is actually about. Written as a descriptive passage rather
# than as the raw boolean search string: the chunks were embedded as passages,
# so a passage-shaped anchor sits in the same region of the space. Validated
# against the known off-topic cluster -- it ranks those papers at roughly the
# 9th-30th percentile while genuine distribution-planning papers land at the
# 98th+, ROC AUC 0.96.
ANCHOR_TEXT = (
    "Planning and expansion of electric power distribution systems: distribution network "
    "planning, distributed generation, feeders, substations, voltage, reliability and power "
    "quality in electrical energy distribution grids."
)

# The other meaning of the same search string. Scoring against both anchors and
# keeping the *margin* is what turns screening from a ranking into a decision:
# a single anchor separates the two groups well (ROC AUC 0.96) but their score
# distributions still overlap -- on this corpus the 90th percentile of the
# logistics group (0.705) sits above the 10th percentile of the genuine
# distribution-planning papers (0.694), so any percentile cut throws away
# in-scope work. The margin `relevance - offtopic` reaches AUC 0.99 with no
# overlap (0.011 vs 0.060) and, unlike a percentile, has a meaningful zero:
# "closer to logistics than to the review's topic".
#
# Measured against pseudo-labels taken from the corpus's own logistics cluster,
# so it is indicative, not a hand-labelled gold standard.
OFF_ANCHOR_TEXT = (
    "Logistics and supply chain distribution: warehouse location, vehicle routing, inventory "
    "management, freight transportation, facility location and cold chain distribution networks."
)

MIN_THEMES = 4
MAX_THEMES = 12
DUPLICATE_THRESHOLD = 0.95
THEME_LABEL_TERMS = 3
RANDOM_SEED = 0

# Clustering and the map both run on this PCA space (61% of the variance on
# this corpus) instead of the raw 384 dimensions. Sharing it is the point: the
# themes used to be found in 384-d while the map was built in 2-d from the same
# vectors but independently, so a point's color had no obligation to agree with
# where it landed -- 35% of a point's 15 nearest neighbours on the map carried a
# different theme. Clustering the reduced space raises that agreement to ~70%,
# and t-SNE over 50 dimensions is also markedly faster than over 384.
PCA_COMPONENTS = 50
TSNE_PERPLEXITY = 50

# A label term must appear in at least this share of its theme's documents.
# Without the floor, "distinctive" degenerates into "rare": the top terms came
# out as one-off acronyms (`Nilm · Nice · Ndz`) that name nothing.
LABEL_MIN_DOC_FREQ = 0.15


def relevance_scores(matrix: np.ndarray, anchor: np.ndarray) -> np.ndarray:
    """Cosine similarity of each row of `matrix` to `anchor`.

    Both sides are L2-normalized first, so this is a plain dot product and the
    result is in [-1, 1] regardless of whether the caller pre-normalized.
    """
    if matrix.size == 0:
        return np.zeros(0, dtype="float32")
    normalized = matrix / np.linalg.norm(matrix, axis=1, keepdims=True)
    unit_anchor = anchor / np.linalg.norm(anchor)
    return (normalized @ unit_anchor).astype("float32")


def reduced_space(matrix: np.ndarray, n_components: int = PCA_COMPONENTS) -> np.ndarray:
    """L2-normalize `matrix` and project it onto its leading principal components.

    The normalization is explicit rather than assumed: `fastembed` does return
    unit vectors for this model (verified: norm 1.0000 +/- 0.00000), but every
    distance below only means cosine distance *because* of that, and a future
    model that doesn't normalize would otherwise silently change what the
    clusters and the map are measuring.

    PCA both denoises and speeds the projection up; `n_components` is capped by
    the data so a handful of rows (a fresh corpus, a test) still works.
    """
    from sklearn.decomposition import PCA

    if matrix.size == 0:
        return matrix
    normalized = matrix / np.linalg.norm(matrix, axis=1, keepdims=True)
    components = min(n_components, *normalized.shape)
    if components < 2:
        return normalized.astype("float32")
    projected = PCA(n_components=components, random_state=RANDOM_SEED).fit_transform(normalized)
    return projected.astype("float32")


def theme_labels_from_terms(
    texts: list[str], labels: np.ndarray, n_terms: int = THEME_LABEL_TERMS
) -> dict[int, str]:
    """Name each theme after the terms it over-uses relative to the rest of the corpus.

    Three rules, each one earned by a label that read badly without it:

    - `max_df` drops the terms every theme shares (`distribution`, `power`,
      `planning`), so they can't win a slot just by being everywhere;
    - a term only competes if it appears in `LABEL_MIN_DOC_FREQ` of the theme's
      documents, which is what keeps rare acronyms out of the label;
    - a term that is a substring of one already picked is skipped, so a theme
      doesn't come out as "energy - energy storage" saying one thing twice.

    Purely lexical on purpose: it explains the cluster in the corpus's own
    words, and it stays testable without the embedding model.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer

    labels = np.asarray(labels)
    fallback = {int(t): f"Tema {int(t) + 1}" for t in np.unique(labels)}
    # The floor has to scale with the corpus: 10 documents out of 1.8k is a
    # real floor, out of 20 it empties the vocabulary.
    min_df = int(np.clip(round(0.005 * len(texts)), 2, 10))
    try:
        vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            min_df=min_df,
            max_df=0.5,
            sublinear_tf=True,
        )
        tfidf = vectorizer.fit_transform(texts)
    except ValueError:
        # Too few/too short documents for a vocabulary -- fall back to numbers.
        logger.debug("theme_labels_from_terms: TF-IDF vocabulary empty", exc_info=True)
        return fallback

    terms = np.array(vectorizer.get_feature_names_out())
    corpus_mean = np.asarray(tfidf.mean(axis=0)).ravel()
    present = (tfidf > 0).astype("float32")

    theme_labels: dict[int, str] = {}
    for theme in np.unique(labels):
        members = labels == theme
        member_mean = np.asarray(tfidf[members].mean(axis=0)).ravel()
        doc_freq = np.asarray(present[members].mean(axis=0)).ravel()
        distinctiveness = np.where(
            doc_freq >= LABEL_MIN_DOC_FREQ, member_mean - corpus_mean, -np.inf
        )

        picked: list[str] = []
        for index in np.argsort(distinctiveness)[::-1]:
            if not np.isfinite(distinctiveness[index]):
                break
            term = str(terms[index])
            if any(term in other or other in term for other in picked):
                continue
            picked.append(term)
            if len(picked) == n_terms:
                break
        theme_labels[int(theme)] = " · ".join(p.title() for p in picked) or fallback[int(theme)]
    return theme_labels


def discover_themes(
    matrix: np.ndarray, texts: list[str], k: int | None = None
) -> tuple[np.ndarray, dict[int, str]]:
    """Cluster the reduced space and label themes from distinctive terms.

    Without an explicit ``k``, candidates from 4 through 12 are compared by
    silhouette score. Solutions containing a cluster below 2% of the corpus
    are rejected, and near-ties prefer the smaller candidate.
    """
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    n_samples = len(matrix)
    if n_samples == 0:
        return np.zeros(0, dtype=int), {}
    if k is None:
        candidates = range(MIN_THEMES, min(MAX_THEMES, n_samples - 1) + 1)
        scored: list[tuple[float, int, np.ndarray]] = []
        for candidate in candidates:
            candidate_labels = KMeans(
                n_clusters=candidate, n_init=10, random_state=RANDOM_SEED
            ).fit_predict(matrix)
            counts = np.bincount(candidate_labels)
            if counts.min(initial=n_samples) / n_samples < 0.02:
                continue
            scored.append(
                (float(silhouette_score(matrix, candidate_labels)), candidate, candidate_labels)
            )
        if scored:
            best_score = max(score for score, _, _ in scored)
            _, _, labels = min(
                (item for item in scored if best_score - item[0] <= 0.005),
                key=lambda item: item[1],
            )
        else:
            fallback_k = max(1, min(MIN_THEMES, n_samples))
            labels = KMeans(n_clusters=fallback_k, n_init=10, random_state=RANDOM_SEED).fit_predict(
                matrix
            )
    else:
        k = max(1, min(k, n_samples))
        labels = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_SEED).fit_predict(matrix)
    return labels, theme_labels_from_terms(texts, labels)


def project_2d(matrix: np.ndarray) -> np.ndarray:
    """Project embeddings to 2D for the semantic map.

    t-SNE preserves local neighbourhoods, which is what makes the off-topic
    group read as a visually separate island. Coordinates are only comparable
    within one run.

    Expects the `reduced_space` matrix. Euclidean distance is left as the
    metric because the vectors it came from are unit-length, so euclidean and
    cosine order neighbours identically -- paying for `metric="cosine"` would
    buy nothing. `TSNE_PERPLEXITY` is higher than Plotly's habit of 30: on this
    corpus it measured better (neighbourhood agreement 0.698 vs 0.686), which
    fits a corpus that is one dense continuum rather than distinct blobs.
    """
    from sklearn.manifold import TSNE

    n_samples = len(matrix)
    if n_samples < 3:
        return np.zeros((n_samples, 2), dtype="float32")
    # t-SNE requires perplexity < n_samples.
    perplexity = min(TSNE_PERPLEXITY, max(2, (n_samples - 1) // 3))
    projection = TSNE(
        n_components=2, perplexity=perplexity, init="pca", random_state=RANDOM_SEED
    ).fit_transform(matrix)
    return projection.astype("float32")


def project_pca_2d(matrix: np.ndarray) -> np.ndarray:
    """Project matrix to 2D using PCA for a linear orthogonal perspective."""
    from sklearn.decomposition import PCA

    n_samples = len(matrix)
    if n_samples < 2:
        return np.zeros((n_samples, 2), dtype="float32")
    n_components = min(2, n_samples, matrix.shape[1] if matrix.ndim > 1 else 1)
    if n_components < 2:
        coords = np.zeros((n_samples, 2), dtype="float32")
        if n_components == 1:
            coords[:, 0] = PCA(n_components=1).fit_transform(matrix).ravel()
        return coords
    return PCA(n_components=2, random_state=RANDOM_SEED).fit_transform(matrix).astype("float32")


def project_umap(matrix: np.ndarray, n_neighbors: int = 15, min_dist: float = 0.1) -> np.ndarray:
    """Project embeddings to 2D via UMAP when the optional dependency exists."""
    n_samples = len(matrix)
    if n_samples < 3:
        return np.zeros((n_samples, 2), dtype="float32")
    import umap

    effective_neighbors = min(n_neighbors, max(2, n_samples - 1))
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=effective_neighbors,
        min_dist=min_dist,
        metric="cosine",
        random_state=RANDOM_SEED,
    )
    return reducer.fit_transform(matrix).astype("float32")


def compute_thematic_centroids(matrix: np.ndarray, labels: np.ndarray) -> dict[int, np.ndarray]:
    """Compute the 2D centroid (mean coordinate) for each theme."""
    centroids: dict[int, np.ndarray] = {}
    for label in np.unique(labels):
        mask = labels == label
        if np.any(mask):
            centroids[int(label)] = np.mean(matrix[mask], axis=0).astype("float32")
    return centroids


def compute_temporal_drift(
    coords_2d: np.ndarray,
    labels: np.ndarray,
    years: np.ndarray,
    windows: list[tuple[int, int]],
) -> dict[int, list[dict]]:
    """Track movement of thematic centroids across chronological windows.

    Returns a dictionary mapping theme_id to a list of dicts with:
    {'window': (start, end), 'name': 'start-end', 'x': float, 'y': float, 'count': int}
    """
    drift: dict[int, list[dict]] = {int(label): [] for label in np.unique(labels)}
    for start_yr, end_yr in windows:
        window_mask = (years >= start_yr) & (years <= end_yr)
        window_labels = labels[window_mask]
        window_coords = coords_2d[window_mask]

        for theme_id in drift:
            theme_mask = window_labels == theme_id
            if np.any(theme_mask):
                c = np.mean(window_coords[theme_mask], axis=0)
                drift[theme_id].append(
                    {
                        "window": (start_yr, end_yr),
                        "name": f"{start_yr}–{end_yr}",
                        "x": float(c[0]),
                        "y": float(c[1]),
                        "count": int(np.sum(theme_mask)),
                    }
                )
    return drift


def compute_semantic_novelty(matrix: np.ndarray, k: int = 10) -> np.ndarray:
    """Compute semantic isolation as mean cosine distance to k nearest neighbors."""
    from sklearn.neighbors import NearestNeighbors

    n_samples = len(matrix)
    if n_samples <= 1:
        return np.zeros(n_samples, dtype="float32")
    normed = matrix / np.clip(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-12, None)
    n_neighbors = min(k + 1, n_samples)
    nn = NearestNeighbors(n_neighbors=n_neighbors, metric="cosine").fit(normed)
    distances, _ = nn.kneighbors(normed)
    if distances.shape[1] > 1:
        return np.mean(distances[:, 1:], axis=1).astype("float32")
    return np.zeros(n_samples, dtype="float32")


def near_duplicate_pairs(
    matrix: np.ndarray, dois: list[str], threshold: float = DUPLICATE_THRESHOLD
) -> list[tuple[str, str, float]]:
    """Distinct-DOI pairs whose abstracts are near-identical.

    Returns `(doi_a, doi_b, similarity)` sorted by similarity, each unordered
    pair once. Only the upper triangle is scanned, so a row is never paired
    with itself.
    """
    if len(matrix) < 2:
        return []
    normalized = matrix / np.linalg.norm(matrix, axis=1, keepdims=True)
    similarity = normalized @ normalized.T
    rows, cols = np.triu_indices(len(matrix), k=1)
    hits = similarity[rows, cols] >= threshold
    pairs = [
        (dois[int(i)], dois[int(j)], float(similarity[int(i), int(j)]))
        for i, j in zip(rows[hits], cols[hits], strict=True)
        if dois[int(i)] != dois[int(j)]
    ]
    return sorted(pairs, key=lambda p: p[2], reverse=True)


def unresolved_duplicate_pairs(
    pairs: list[tuple[str, str, float]], reviewed_pairs: set[tuple[str, str]]
) -> list[tuple[str, str, float]]:
    """Remove reviewed unordered DOI pairs from a detected candidate list."""
    reviewed = {tuple(sorted(pair)) for pair in reviewed_pairs}
    return [pair for pair in pairs if tuple(sorted(pair[:2])) not in reviewed]


def _embed_anchors(texts: list[str]) -> np.ndarray:
    # Imported lazily, like transform/embeddings.py does, so stages that never
    # touch the model don't pay fastembed's import cost. Both anchors go
    # through one model load -- it is the expensive part, not the two vectors.
    from fastembed import TextEmbedding

    model = TextEmbedding(model_name=EMBED_MODEL_NAME)
    return np.array(list(model.embed(texts)), dtype="float32")


def build_semantics(
    gold_session: Session,
    anchor_text: str = ANCHOR_TEXT,
    off_anchor_text: str = OFF_ANCHOR_TEXT,
    *,
    anchor_vectors: tuple[np.ndarray, np.ndarray] | None = None,
) -> dict:
    """Rebuild `lit_semantics` and `lit_duplicate_pairs` from the abstract chunks.

    One abstract chunk per article is the unit here: it exists for every gold
    article and is a self-contained summary, whereas fulltext chunks only exist
    for the minority of articles that have a PDF.
    """
    rows = gold_session.execute(
        select(Chunk.doi, Chunk.embedding, Chunk.embedding_bin, Chunk.text)
        .where(Chunk.chunk_type == "abstract")
        .where((Chunk.embedding.is_not(None)) | (Chunk.embedding_bin.is_not(None)))
        .order_by(Chunk.doi)
    ).all()

    # Coverage, not just presence: this stage used to run happily on whatever
    # subset happened to be embedded, so a `gold` rebuild followed by a partial
    # `embed` produced a full-looking `lit_semantics` derived from a fraction of
    # the corpus, with nothing saying so.
    total_abstracts = (
        gold_session.scalar(
            select(func.count()).select_from(Chunk).where(Chunk.chunk_type == "abstract")
        )
        or 0
    )
    coverage = len(rows) / total_abstracts if total_abstracts else 0.0

    if not rows:
        return {
            "articles": 0,
            "themes": 0,
            "duplicate_pairs": 0,
            "embedding_coverage": 0.0,
            "skipped": "no embeddings yet",
        }

    if coverage < 1.0:
        logger.warning(
            "build_semantics: only %d of %d abstract chunks are embedded (%.1f%%) -- "
            "the signals written here describe that subset only; run `--stage embed` first",
            len(rows),
            total_abstracts,
            coverage * 100,
        )

    dois = [r[0] for r in rows]
    matrix = np.array(
        [
            np.frombuffer(r[2], dtype=np.float32)
            if r[2] is not None
            else np.array(r[1], dtype=np.float32)
            for r in rows
        ],
        dtype="float32",
    )
    texts = [r[3] or "" for r in rows]

    if anchor_vectors is not None:
        anchors = np.array(anchor_vectors, dtype="float32")
    else:
        anchors = _embed_anchors([anchor_text, off_anchor_text])
    scores = relevance_scores(matrix, anchors[0])
    offtopic = relevance_scores(matrix, anchors[1])

    # One space for both: the themes are found in it and the map is drawn from
    # it, so a point's color and its position can't disagree the way they did
    # when the clusters lived in 384 dimensions and the map in 2.
    reduced = reduced_space(matrix)
    labels, theme_labels = discover_themes(reduced, texts)
    coords = project_2d(reduced)
    pairs = near_duplicate_pairs(matrix, dois)
    reviewed_pairs = {
        (override.doi_a, override.doi_b)
        for override in gold_session.scalars(select(DuplicateOverride)).all()
    }
    unresolved_pairs = unresolved_duplicate_pairs(pairs, reviewed_pairs)

    gold_session.execute(delete(Semantics))
    gold_session.execute(delete(DuplicatePair))
    gold_session.add_all(
        [
            Semantics(
                doi=doi,
                relevance_score=float(scores[i]),
                offtopic_score=float(offtopic[i]),
                theme_id=int(labels[i]),
                theme_label=theme_labels[int(labels[i])],
                map_x=float(coords[i][0]),
                map_y=float(coords[i][1]),
                embed_model=EMBED_MODEL_NAME,
            )
            for i, doi in enumerate(dois)
        ]
    )
    gold_session.add_all(
        [DuplicatePair(doi_a=a, doi_b=b, similarity=s) for a, b, s in unresolved_pairs]
    )
    gold_session.flush()

    return {
        "articles": len(dois),
        "themes": len(theme_labels),
        "duplicate_pairs": len(unresolved_pairs),
        "duplicate_pairs_reviewed": len(pairs) - len(unresolved_pairs),
        "embedding_coverage": round(coverage, 4),
        "median_relevance": round(float(np.median(scores)), 4),
        # The screening signal: positive means closer to the review's topic
        # than to the logistics reading of the same search string.
        "median_margin": round(float(np.median(scores - offtopic)), 4),
        "offtopic_articles": int((scores - offtopic < 0).sum()),
    }
