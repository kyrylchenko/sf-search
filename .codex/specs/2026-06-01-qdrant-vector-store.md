# Qdrant Vector Store Spec

## Problem

The embedding service currently writes vectors into a local `hnswlib` index file. That path rewrites local artifacts during embedding and makes production operations harder to observe, scale, and recover. The next server deployment should use Qdrant as the vector database while preserving the current embedding service ownership of vector writes.

## Requirements

- Use Qdrant as the default vector store for embedding and the local query UI.
- Keep `local_hnsw` available by configuration for local experiments and rollback.
- Write vectors directly from the embedding service after image embeddings are produced.
- Treat Qdrant upsert failure as embedding failure. Rows claimed for the batch must be marked failed and queue messages acknowledged through the existing failure path.
- Record enough DB metadata to know which vector store handled an embedding: kind, path, and vector ID.
- Surface Qdrant write timing through existing logs and OpenTelemetry progress events so SigNoz can show whether vector writes are a bottleneck.
- Add Qdrant to Docker Compose with persisted storage on disk.
- Do not commit generated Qdrant data, local `.env`, logs, panorama images, tiles, or vector artifacts.

## Non-Goals

- No production search API work in this milestone.
- No separate async indexer service yet.
- No Qdrant Cloud or authenticated remote Qdrant setup in committed config.
- No migration of existing local HNSW vectors. Existing embeddings can be regenerated into Qdrant when needed.

## Architecture

Embedding keeps the same batch flow:

1. Fetch embedding jobs from NATS.
2. Claim DB rows for the selected model spec.
3. Generate image embeddings.
4. Upsert the batch into Qdrant.
5. Mark embeddings complete only after the upsert returns successfully.

If Qdrant raises an exception, the runner uses its current batch failure branch: mark each claimed embedding failed, acknowledge messages, and emit failure logs/events. This keeps DB state honest and makes Qdrant problems visible.

The vector store interface remains synchronous for now. Qdrant writes are batched with `add_many`, and OpenTelemetry timing around `embedding_vector_store_batch_*` shows whether the blocking write is meaningful compared with model inference. A later indexer/async writer can be added if monitoring proves Qdrant writes are the bottleneck.

## Configuration

New settings:

- `EMBEDDING_VECTOR_STORE_KIND`: `qdrant` by default, `local_hnsw` for the old file store.
- `QDRANT_URL`: local default `http://localhost:6333`; Compose points services at `http://qdrant:6333`.
- `QDRANT_COLLECTION`: default collection for panorama view embeddings.
- `QDRANT_VECTOR_ON_DISK`: default `true` to reduce RAM pressure for large collections.
- `QDRANT_HNSW_ON_DISK`: default `false` so the HNSW graph can stay memory-backed unless RAM pressure requires disk mode.
- `QDRANT_ON_DISK_PAYLOAD`: default `true` because DB remains the source of truth for rich metadata.
- `QDRANT_UPSERT_WAIT`: default `true` so failed writes propagate to embedding status.
- `QDRANT_TIMEOUT_SECONDS`: default request timeout.

## Data Model

Qdrant point ID equals the DB `panorama_view_embeddings.id`. Point payload contains minimal debugging metadata:

- `embedding_id`
- `view_id`
- `model_id`
- `source_image_hash`

The Postgres embedding row stores:

- `vector_store_kind = "qdrant"`
- `vector_store_path = "<qdrant-url>/<collection>"`
- `vector_id = "<embedding-id>"`

## Failure Handling

- Missing image: unchanged, row fails before vector write.
- Model failure: unchanged, row fails before vector write.
- Qdrant create/upsert failure: all claimed rows in that batch fail with the Qdrant error text.
- Duplicate point IDs: Qdrant upsert is idempotent and replaces the point payload/vector for the same embedding ID.

## Alternatives Considered

- Keep local HNSW: fastest to preserve but keeps index rewrite IO in the embedding hot path and is not production-friendly.
- Separate queue-backed indexer: isolates Qdrant writes, but adds another service and makes it possible for DB to say embedded while Qdrant is behind or failed unless we add a more complex state machine.
- Direct Qdrant write from embedding: simplest now, preserves correctness, and monitoring will show if the write path needs to be split later.

## Verification

- Unit tests for Qdrant store collection creation, batched upsert payloads, search result conversion, and exception propagation.
- Unit tests for vector store factory/config.
- Unit test that a vector store batch failure marks embedding rows failed.
- `uv run pytest -q`
- `docker compose config --quiet`
