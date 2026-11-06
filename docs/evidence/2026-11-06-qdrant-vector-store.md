# Qdrant vector store (WP-26, ADR-08) — 2026-11-06

## Population and method

- Qdrant server `qdrant/qdrant:v1.20.0` (compose profile `qdrant`), `qdrant-client` 1.20.0.
- Active version before the migration: `2974743a…`, 7,552 chunks (3,115 abstract, 4,437 full text) with
  complete pre-ADR-08 provenance and `embedding_bin` BLOBs.
- Eight dataset versions on record: 1 active, 2 superseded, 5 candidates. Only two hold BLOBs: the active
  one and the superseded legacy import `31a2270b…`, whose 7,552 BLOBs carry no model revision, no text hash
  and no embedding timestamp.

## Backfill

`uv run lake-research-map vectors backfill`:

| Version | Result |
|---|---|
| `2974743a…` (active) | 7,552 points; embed contract passed; `lit_chunks_active` pointed at it |
| `31a2270b…` (superseded legacy import) | skipped: 0/7,552 chunks carry a reusable vector — it cannot pass the embed contract, which it already could not before the migration |

Checks on the copy:

- Every Qdrant vector equals its MySQL BLOB bit for bit (max absolute difference 0 over 7,552 vectors).
- Relevance recomputed from the Qdrant abstract matrix against the anchors matches the stored
  `lit_dataset_semantics.relevance_score` within 5.1e-7 (float rounding) over 3,115 abstracts.

The first backfill attempt ran at ~6.7 s per 64-chunk batch. The cause was the per-batch lookup of BLOBs by
`text_sha256`, which has **no index** on the deployed schema (the column arrived through an additive migration,
which adds none): 7.15 s per 64 hashes, against 0.59 s for the same lookup restricted by the indexed
`dataset_version_id`. Reusable vectors are now resolved once, in bulk, before any batch — Qdrant in pages of 256
hashes, MySQL in one read per legacy version — and the backfill completes in 30 s.

## First publication on Qdrant

`uv run lake-research-map --stage all` (code fingerprint changed, so a new version `4ffe43c7…` was minted):

| Stage | Stats |
|---|---|
| gold | 3,115 articles; 7,552 chunks reused, 0 new, 0 invalidated |
| embed | 0 embedded, **7,552 vectors copied** from the active collection, 0 unresolved, 0 orphans |
| semantic | 3,115 articles, 4 themes, 206 duplicate pairs, median relevance 0.7742, median margin 0.1152, 246 off-topic |

- Publication passed the embed contract on the Qdrant points; `lit_chunks_active` moved to `4ffe43c7…` after the
  commit; `2974743a…` is superseded and keeps its collection, so reactivating it needs no recomputation.
- Semantic signals of the new version equal the previous one's: max relevance difference 0, theme agreement
  100% over 3,115 articles.
- The new version's 7,552 chunk rows hold provenance and no BLOB; the live `lit_chunks` projection holds no
  MySQL vector any more.

## Model snapshot drift found on the way

Re-embedding for the checks above failed the revision pin: fastembed downloaded Hugging Face snapshot
`aa8f8b06…`, not the pinned `52398278…`. Its default cache lives in tmpfs `/tmp`, so every reboot is a fresh
download, and the Hub repository received a commit on 2026-09-24. That commit changed `README.md` only:
`model_optimized.onnx` (LFS sha256 `51f1bd0a…`), the tokenizer files and `config.json` are byte-identical in
both revisions, and re-embedding 200 stored chunks with the new snapshot reproduced their vectors exactly
(max absolute difference 0, min cosine 1.000000).

The pin now also accepts a snapshot whose five model-defining files hash to `EMBED_MODEL_CONTENT_SHA256`
(`9002f600…`) and records the pinned revision; any change to the weights or tokenizer still fails. Before this
fix the old embed stage would have refused the identical model, and the old semantic stage — whose anchors
loaded the model without the check — would have scored with whatever snapshot it found.

## Dashboard reads

- `data.load_chunk_search_data` (7,552 texts from MySQL plus vectors fetched from Qdrant): 5.4 s cold, against
  2.3 s for the BLOB read measured on 2026-09-21. The path is cached per Streamlit process and is replaced by
  direct Qdrant queries in the article-level search (WP-27).
- `data.load_abstract_embeddings_data`: 3,115 vectors in 1.3 s.
- Embedding readiness is read from Gold provenance (7,552/7,552), so no page render calls Qdrant.

## Tests

436 passing, 4 opt-in skipped (2 MySQL, 2 Qdrant server). `RUN_QDRANT_INTEGRATION=1 uv run pytest -m qdrant`
passed against the compose service: payload indexes exist, filtered exact search matches the NumPy cosine
order, and the alias switch is atomic.
