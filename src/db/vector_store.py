"""Qdrant vector store: the only place chunk embeddings are written (ADR-08).

Gold (MySQL) remains the source of truth for chunk text, article metadata,
embedding provenance and the publication pointer; Qdrant holds the vectors.
Each dataset version gets its own collection, filled while the version is a
candidate. A published collection is only ever written again to restore points
it lost (`vectors rebuild`), and those are recomputed from the same Gold text
with the same pinned model revision -- so it is as immutable as the
`lit_dataset_*` rows it mirrors, and an exact rollback still only has to move
the pointer.

Point ids are a pure function of the chunk key, so a re-run upserts the same
points instead of duplicating them. Every dense search is exact (brute force):
at ~7.5k vectors that costs nothing, and a systematic review needs the same
query to return the same ranking on every run. Approximate (HNSW) search waits
for the WP-25 scale trigger.

`qdrant_client` is imported lazily, like fastembed, so code paths that never
touch vectors do not pay for it. This module is storage mechanics only; what
goes into a point's payload is decided by `transform/vector_payloads.py`.
"""

from __future__ import annotations

import functools
import logging
import uuid
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from functools import cache
from typing import Any

import numpy as np

from config import get_qdrant_settings

logger = logging.getLogger(__name__)

COLLECTION_PREFIX = "lit_chunks_"
# The alias external RAG clients query. Internal readers never use it: they
# resolve the collection from the publication pointer in Gold, so an alias that
# lags behind a publication cannot make the project disagree with itself.
ACTIVE_ALIAS = "lit_chunks_active"
DENSE_VECTOR = "dense"
# Never change this value: every stored point id was derived from it.
POINT_NAMESPACE = uuid.UUID("3d6c1f0e-8a4b-5c2d-9e7f-1a2b3c4d5e6f")
BATCH_SIZE = 256
START_HINT = (
    "start it with `docker compose --profile qdrant up -d qdrant` and set QDRANT_URL "
    "(http://127.0.0.1:6333 on the host, http://qdrant:6333 inside docker compose)"
)

# Payload fields a server-side filter can use. The server needs an index to
# filter efficiently; the local in-memory client the tests use ignores them.
PAYLOAD_INDEXES: dict[str, str] = {
    "doi": "keyword",
    "chunk_type": "keyword",
    "text_sha256": "keyword",
    "embed_revision": "keyword",
    "year": "integer",
    "source": "keyword",
    "venue": "keyword",
    "publication_category": "keyword",
    "is_non_article": "bool",
    "has_pdf": "bool",
    "abstract_missing": "bool",
    "relevance_margin": "float",
    "theme_id": "integer",
}


class VectorStoreUnavailable(RuntimeError):
    """The vector store is not configured, or it does not answer."""


@dataclass(frozen=True, slots=True)
class VectorPoint:
    id: str
    vector: np.ndarray
    payload: dict[str, Any]


@dataclass(frozen=True, slots=True)
class StoredPoint:
    id: str
    payload: dict[str, Any]
    vector: np.ndarray | None


@dataclass(frozen=True, slots=True)
class DenseHit:
    id: str
    score: float
    payload: dict[str, Any]


def collection_name(version_id: str) -> str:
    name = f"{COLLECTION_PREFIX}{version_id}"
    if name == ACTIVE_ALIAS:
        raise ValueError(f"version id {version_id!r} would collide with the alias {ACTIVE_ALIAS}")
    return name


def version_of(collection: str) -> str | None:
    """Dataset version a collection (or alias target) belongs to, if it is ours."""
    if collection == ACTIVE_ALIAS or not collection.startswith(COLLECTION_PREFIX):
        return None
    return collection[len(COLLECTION_PREFIX) :] or None


def point_id(doi: str, chunk_type: str, seq: int) -> str:
    """Deterministic point id for one Gold chunk key within a version."""
    return str(uuid.uuid5(POINT_NAMESPACE, f"{doi}\x1f{chunk_type}\x1f{int(seq)}"))


def _batches(items: Sequence, size: int = BATCH_SIZE) -> Iterator[Sequence]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def _as_vector(raw: Any) -> np.ndarray | None:
    if raw is None:
        return None
    if isinstance(raw, Mapping):
        raw = raw.get(DENSE_VECTOR)
        if raw is None:
            return None
    return np.asarray(raw, dtype=np.float32)


def _payload_selector(fields: Sequence[str] | bool) -> list[str] | bool:
    return list(fields) if not isinstance(fields, bool) else fields


def _is_local(client: Any) -> bool:
    # The local client searches by brute force and warns that server-only
    # options (payload indexes, search params) have no effect, so skip them.
    return type(getattr(client, "_client", None)).__name__ == "QdrantLocal"


def _is_transport_error(exc: Exception) -> bool:
    import httpx
    from qdrant_client.http.exceptions import ResponseHandlingException

    return isinstance(exc, ResponseHandlingException | httpx.TransportError | ConnectionError)


def _guarded(method):
    """Translate "the server did not answer" into `VectorStoreUnavailable`.

    Callers decide how to degrade on that one exception (a stage fails with the
    start hint, a dashboard page shows a caption), without having to know
    qdrant-client's or httpx's exception hierarchy.
    """

    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        try:
            return method(self, *args, **kwargs)
        except VectorStoreUnavailable:
            raise
        except Exception as exc:
            if _is_transport_error(exc):
                raise VectorStoreUnavailable(
                    f"Qdrant did not answer ({exc}); {START_HINT}"
                ) from exc
            raise

    return wrapper


class QdrantVectorStore:
    """One immutable collection per dataset version, plus the active alias."""

    def __init__(self, client: Any) -> None:
        self.client = client
        self.local = _is_local(client)

    @_guarded
    def ping(self) -> None:
        self.client.get_collections()

    @_guarded
    def has_collection(self, version_id: str) -> bool:
        return bool(self.client.collection_exists(collection_name(version_id)))

    @_guarded
    def versions(self) -> list[str]:
        names = (item.name for item in self.client.get_collections().collections)
        return sorted(version for version in map(version_of, names) if version)

    @_guarded
    def ensure_collection(self, version_id: str, dimension: int) -> bool:
        """Create the version's collection when absent; return whether it was created.

        An existing collection with different vector parameters is an error, not
        something to recreate: it holds vectors another model produced.
        """
        from qdrant_client import models

        name = collection_name(version_id)
        if self.client.collection_exists(name):
            params = self.client.get_collection(name).config.params.vectors
            dense = params.get(DENSE_VECTOR) if isinstance(params, Mapping) else None
            if dense is None or dense.size != dimension or dense.distance != models.Distance.COSINE:
                raise ValueError(f"collection {name} exists with incompatible vectors: {params}")
            return False
        self.client.create_collection(
            name,
            vectors_config={
                DENSE_VECTOR: models.VectorParams(size=dimension, distance=models.Distance.COSINE)
            },
        )
        if not self.local:
            for field, schema in PAYLOAD_INDEXES.items():
                self.client.create_payload_index(
                    name,
                    field_name=field,
                    field_schema=models.PayloadSchemaType(schema),
                    wait=True,
                )
        return True

    @_guarded
    def drop_collection(self, version_id: str) -> None:
        self.client.delete_collection(collection_name(version_id))

    @_guarded
    def upsert(self, version_id: str, points: Sequence[VectorPoint]) -> None:
        from qdrant_client import models

        name = collection_name(version_id)
        for batch in _batches(list(points)):
            self.client.upsert(
                name,
                points=[
                    models.PointStruct(
                        id=point.id,
                        vector={DENSE_VECTOR: np.asarray(point.vector, dtype=np.float32).tolist()},
                        payload=point.payload,
                    )
                    for point in batch
                ],
                wait=True,
            )

    @_guarded
    def fetch(
        self,
        version_id: str,
        ids: Sequence[str],
        *,
        with_vectors: bool = False,
        payload_fields: Sequence[str] | bool = True,
    ) -> dict[str, StoredPoint]:
        """Points by id; ids the collection does not hold are simply absent."""
        name = collection_name(version_id)
        found: dict[str, StoredPoint] = {}
        for batch in _batches(list(ids)):
            records = self.client.retrieve(
                name,
                ids=list(batch),
                with_payload=_payload_selector(payload_fields),
                with_vectors=[DENSE_VECTOR] if with_vectors else False,
            )
            for record in records:
                found[str(record.id)] = StoredPoint(
                    str(record.id),
                    dict(record.payload or {}),
                    _as_vector(record.vector) if with_vectors else None,
                )
        return found

    @_guarded
    def scroll(
        self,
        version_id: str,
        *,
        query_filter: Any = None,
        with_vectors: bool = False,
        payload_fields: Sequence[str] | bool = True,
    ) -> list[StoredPoint]:
        """Every point matching `query_filter` (a list: the corpus is small)."""
        name = collection_name(version_id)
        points: list[StoredPoint] = []
        offset = None
        while True:
            records, offset = self.client.scroll(
                name,
                scroll_filter=query_filter,
                limit=BATCH_SIZE,
                offset=offset,
                with_payload=_payload_selector(payload_fields),
                with_vectors=[DENSE_VECTOR] if with_vectors else False,
            )
            points.extend(
                StoredPoint(
                    str(record.id),
                    dict(record.payload or {}),
                    _as_vector(record.vector) if with_vectors else None,
                )
                for record in records
            )
            if offset is None:
                return points

    def find_by_text_hash(
        self, version_id: str, hashes: Sequence[str], *, model: str, revision: str
    ) -> dict[str, np.ndarray]:
        """Vectors already stored for these texts by the pinned model revision.

        An embedding is a function of (text, model, revision), so any point
        carrying the same three values holds the vector the model would produce
        again -- this is what lets a new version reuse its predecessor's work.
        """
        from qdrant_client import models

        wanted = sorted(set(hashes))
        if not wanted or not self.has_collection(version_id):
            return {}
        found: dict[str, np.ndarray] = {}
        for batch in _batches(wanted):
            query_filter = models.Filter(
                must=[
                    models.FieldCondition(
                        key="text_sha256", match=models.MatchAny(any=list(batch))
                    ),
                    models.FieldCondition(key="embed_model", match=models.MatchValue(value=model)),
                    models.FieldCondition(
                        key="embed_revision", match=models.MatchValue(value=revision)
                    ),
                ]
            )
            for point in self.scroll(
                version_id,
                query_filter=query_filter,
                with_vectors=True,
                payload_fields=["text_sha256"],
            ):
                digest = point.payload.get("text_sha256")
                if digest and point.vector is not None:
                    found.setdefault(digest, point.vector)
        return found

    @_guarded
    def set_payloads(
        self, version_id: str, groups: Sequence[tuple[Sequence[str], Mapping[str, Any]]]
    ) -> None:
        """Merge each payload into its group of points, many groups per request."""
        from qdrant_client import models

        name = collection_name(version_id)
        operations = [
            models.SetPayloadOperation(
                set_payload=models.SetPayload(payload=dict(payload), points=list(ids))
            )
            for ids, payload in groups
            if ids
        ]
        for batch in _batches(operations):
            self.client.batch_update_points(name, update_operations=list(batch), wait=True)

    @_guarded
    def delete_points(self, version_id: str, ids: Sequence[str]) -> None:
        from qdrant_client import models

        name = collection_name(version_id)
        for batch in _batches(list(ids)):
            self.client.delete(
                name, points_selector=models.PointIdsList(points=list(batch)), wait=True
            )

    @_guarded
    def query_dense(
        self,
        version_id: str,
        vector: np.ndarray,
        *,
        query_filter: Any = None,
        limit: int = 10,
        payload_fields: Sequence[str] | bool = True,
    ) -> list[DenseHit]:
        """Exact cosine search; scores are cosine similarities in [-1, 1]."""
        from qdrant_client import models

        extra = {} if self.local else {"search_params": models.SearchParams(exact=True)}
        response = self.client.query_points(
            collection_name(version_id),
            query=np.asarray(vector, dtype=np.float32).tolist(),
            using=DENSE_VECTOR,
            query_filter=query_filter,
            limit=max(int(limit), 1),
            with_payload=_payload_selector(payload_fields),
            with_vectors=False,
            **extra,
        )
        return [
            DenseHit(str(point.id), float(point.score), dict(point.payload or {}))
            for point in response.points
        ]

    @_guarded
    def count(self, version_id: str, *, query_filter: Any = None) -> int:
        return int(
            self.client.count(
                collection_name(version_id), count_filter=query_filter, exact=True
            ).count
        )

    def abstract_vectors(self, version_id: str) -> tuple[list[str], np.ndarray]:
        """(dois, matrix) of the version's abstract vectors, ordered by DOI."""
        from qdrant_client import models

        points = self.scroll(
            version_id,
            query_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="chunk_type", match=models.MatchValue(value="abstract")
                    )
                ]
            ),
            with_vectors=True,
            payload_fields=["doi"],
        )
        points = sorted(
            (point for point in points if point.vector is not None and point.payload.get("doi")),
            key=lambda point: point.payload["doi"],
        )
        if not points:
            return [], np.zeros((0, 0), dtype=np.float32)
        return [point.payload["doi"] for point in points], np.stack(
            [point.vector for point in points]
        ).astype(np.float32)

    @_guarded
    def alias_target(self, alias: str = ACTIVE_ALIAS) -> str | None:
        for item in self.client.get_aliases().aliases:
            if item.alias_name == alias:
                return version_of(item.collection_name)
        return None

    @_guarded
    def point_alias(self, version_id: str, alias: str = ACTIVE_ALIAS) -> None:
        """Move `alias` to the version's collection in one atomic request."""
        from qdrant_client import models

        operations: list[Any] = []
        if any(item.alias_name == alias for item in self.client.get_aliases().aliases):
            operations.append(
                models.DeleteAliasOperation(delete_alias=models.DeleteAlias(alias_name=alias))
            )
        operations.append(
            models.CreateAliasOperation(
                create_alias=models.CreateAlias(
                    collection_name=collection_name(version_id), alias_name=alias
                )
            )
        )
        self.client.update_collection_aliases(change_aliases_operations=operations)


@cache
def _store_for(url: str, api_key: str | None, timeout: int) -> QdrantVectorStore:
    from qdrant_client import QdrantClient

    return QdrantVectorStore(QdrantClient(url=url, api_key=api_key, timeout=timeout))


def get_vector_store() -> QdrantVectorStore:
    """The configured store, verified to answer -- the one access point, like `get_engine`."""
    settings = get_qdrant_settings()
    if not settings.url:
        raise VectorStoreUnavailable(f"QDRANT_URL is not set; {START_HINT}")
    store = _store_for(settings.url, settings.api_key, settings.timeout)
    store.ping()
    return store
