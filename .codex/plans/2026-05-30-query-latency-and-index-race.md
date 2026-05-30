# Query Latency and Index Race Plan

## Goal

Investigate why local text queries can take close to a minute and make the
query/index path safer while embedding is still writing vectors.

## Findings

- A cold query spends significant time loading and warming the Transformers
  model before the first text embedding.
- Query UI and embedding can run at the same time against the same local HNSW
  files.
- The vector store currently saves `index.bin` directly to the live path. A
  query can try to load that file while it is being overwritten, causing
  `Index seems to be corrupted or unsupported`.

## Approach

- Write HNSW index files to a temp file and atomically replace the live file.
- Write metadata JSON through the same temp-and-replace pattern.
- Add query-stage timing logs for text embedding, vector search, DB hydration,
  and total request time.
- Warm the query service at startup so the first browser query does not pay the
  model-load and index-load cost.

## Verification

- Add/update unit tests for atomic vector-store writes.
- Run focused embedding query/vector-store tests.
- Run the full main-service test suite.

## Result

- Cold text embedding on this machine spent about 15-22 seconds loading,
  validating, and warming the Transformers/SigLIP model on MPS.
- HNSW search failed during probing with `Index seems to be corrupted or
  unsupported` while the old embedding worker was still running. That confirms a
  read/write race on the live `index.bin` path.
- `LocalHnswVectorStore` now saves `index.bin` and `metadata.json` to temp files
  and atomically replaces the live files.
- Query UI now logs stage timings and runs a startup warmup query. After warmup,
  normal browser queries should avoid cold model load and cold HNSW load unless
  the query UI restarts or the five-minute cache expires.
