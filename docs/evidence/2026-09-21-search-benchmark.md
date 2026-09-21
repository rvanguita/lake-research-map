# Search benchmark and scale decision — 2026-09-21

## Population and method

- Active immutable Gold version: `31a2270bed86b9fd5058d4de037edb06ed8017e74251ba9edb054cadfaa31f2c`.
- 7,552 chunks, 384 float32 dimensions, 11.06 MiB dense matrix.
- Twenty warm in-process searches using a real stored vector, `top_k=10`, with native numerical libraries limited to two threads.

## Results

| Measurement | Before | After binary-first optimization |
|---|---:|---:|
| Database load | 11.612 s | 2.294 s |
| Linear cosine P50 | 156.44 ms | 37.76 ms |
| Linear cosine P95 | 326.23 ms | 92.72 ms |
| sklearn index build | 58.09 ms | not required |
| sklearn query P50/P95 | 9.25/11.62 ms | reference only |

The load reduction comes from excluding the legacy JSON embedding column. Search now parses canonical binary vectors directly and uses vectorized cosine scoring with partial top-k selection.

## Decision

Keep the local binary/in-process implementation. Its warm-query P95 is below 100 ms and the matrix occupies about 11 MiB, so Faiss or a remote vector service would add dependency, recovery, and operational cost without solving a measured limit. Reconsider when the matrix exceeds 512 MiB, warm P95 exceeds 250 ms, or concurrent search becomes a product requirement.

The cold database load remains visible but is incurred only on submitted searches and is cached by Streamlit. A future benchmark must use labeled queries from WP-13 before changing the default retrieval mode.
