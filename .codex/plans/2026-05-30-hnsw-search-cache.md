# HNSW Search Cache Plan

## Goal

Keep the local HNSW index loaded in RAM for repeated query UI searches instead
of loading `index.bin` from disk on every request.

## Approach

- Add a five-minute default search cache to `LocalHnswVectorStore`.
- Cache both the loaded hnswlib index and the metadata snapshot used for result
  counts.
- Reuse the cached index until the TTL expires.
- Invalidate the cache on writes made by the same process. The query UI runs in
  a separate process from the embedder, so newly embedded vectors may be visible
  up to five minutes late in the UI; restarting the query UI also refreshes it.

## Verification

- Unit-test repeated searches reuse one loaded HNSW index within the TTL.
- Unit-test expiration reloads the index.
- Unit-test local writes invalidate the cache.
- Run focused vector store tests, query UI tests, and then the full main-service
  suite.

## Result

- `LocalHnswVectorStore` keeps one loaded search index and metadata snapshot in
  process memory for five minutes by default.
- The cache is invalidated after writes made by the same process.
- Query UI runs in a different process from embedding, so newly embedded vectors
  may take up to five minutes to appear in search unless the query UI restarts.
