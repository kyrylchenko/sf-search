# Local HNSW To Qdrant Backfill Plan

**Goal:** Copy locally stored HNSW embeddings into Qdrant so the default Qdrant-backed query UI can search the existing local dataset.

**Non-goals:** Do not re-embed images, change the embedding model, delete the local HNSW index, or change Qdrant production tuning.

**Files likely to change:**

- `services/main/main_service/embedding/vector_store.py`
- `services/main/main_service/ops/vector_backfill.py`
- `services/main/main_service/ops/__main__.py`
- `services/main/tests/embedding/test_vector_store.py`
- `services/main/tests/ops/test_vector_backfill.py`
- `services/main/tests/ops/test_ops_requeue.py`

**Approach:**

1. Add a public batch export method to `LocalHnswVectorStore` that loads the HNSW index, reads labels from `metadata.json`, and yields `VectorStoreRecord` batches using `index.get_items`.
2. Add an ops backfill function that calls the local HNSW exporter, upserts each batch to the configured Qdrant store, then updates matching embedding DB rows to `vector_store_kind='qdrant'` and the Qdrant collection path.
3. Add a CLI command:

```bash
uv run python -m main_service.ops backfill-qdrant-from-local-hnsw --batch-size 256
```

4. Run focused tests, then run the backfill against the local Dockerized Qdrant.

**Why this approach:**

- It preserves the existing HNSW artifact as a fallback.
- It avoids queueing or re-running model inference.
- It keeps DB state aligned with the active Qdrant store after successful upserts.

**Verification commands:**

```bash
uv run pytest tests/embedding/test_vector_store.py tests/ops/test_vector_backfill.py tests/ops/test_ops_requeue.py -q
```

```bash
curl -fsS http://127.0.0.1:6333/collections/panorama_view_embeddings_siglip2
```

```bash
docker compose exec -T postgres psql -U sf_search -d sf_search -c "select vector_store_kind, embedding_status, count(*) from panorama_view_embedding_table group by vector_store_kind, embedding_status order by vector_store_kind, embedding_status;"
```

**Risks and open questions:**

- Qdrant upserts are network and serialization heavy. Use moderate batch sizes so memory does not spike.
- If the command fails mid-run, rerun it. Qdrant point IDs are embedding IDs, so upserts are idempotent.
