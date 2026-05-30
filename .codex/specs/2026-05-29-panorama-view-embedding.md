# Panorama View Embedding

## Problem

Panorama preprocessing now writes perspective view images and durable rows in
`panorama_view_table`. The next ingestion stage needs to consume those processed
views, generate image embeddings, persist enough metadata to rerun or switch
models later, and write vectors to a simple local index while the long-term
vector database is still undecided.

## Requirements

- Processing publishes small embedding jobs after it completes generated views.
- Processing applies backpressure before pulling the next panorama when the
  embedding queue has at least 100 pending jobs by default. It may finish and
  enqueue all views for the current panorama even if that pushes the queue above
  the threshold.
- Embedding consumes durable NATS JetStream jobs from `pano.embedding.requested`.
- Duplicate embedding messages are safe. Postgres is the dedupe gate.
- A view can have embeddings for multiple models or preprocessing versions.
- Store model, source image, vector, timing, status, and error metadata in
  Postgres.
- Store local vector index artifacts under `.local/`, which remains ignored.
- Keep the model implementation swappable. The first production adapter targets
  `google/siglip2-so400m-patch14-384`, but the database must not assume that is
  final.
- Provide a local test-only query UI after embedding works. It should embed text
  queries with the same model, search the local HNSW index, and display matching
  locally stored view images with metadata.

## Non-Goals

- Public search APIs and website querying.
- A permanent vector database decision.
- Re-embedding all historical views for a new model. That can be a later
  backfill/requeue tool.
- Manual migrations. Current local development still uses `Base.metadata`.

## Architecture

Add `panorama_view_embedding_table` keyed by processed view, model identity,
preprocess version, and source image hash. The table records claim status,
attempts, model metadata, vector dimensions, vector store details, and failure
details.

Processing receives a `PanoEmbeddingQueue` dependency. Before fetching more pano
processing jobs, it checks that queue depth. While processing a pano, every
successfully completed view enqueues a small message containing the view row ID,
pano ID, and local image path.

Embedding runs as `python -m main_service.embedding`. It pulls a bounded batch
from NATS, claims one `panorama_view_embedding_table` row per view/model/source,
loads the local view image, generates a normalized vector through an image
embedder interface, writes that vector to a local vector store, then marks the
embedding row complete. Failed jobs are acked after Postgres records the error so
operators can requeue or retry from DB state later.

## Local HNSW Vector Store

The first local vector store is intentionally simple and file-based but should
use HNSW semantics so the query UI exercises the retrieval shape we expect later:

- one directory per model key under `.local/embedding-indexes`;
- an HNSW index file;
- a metadata JSON file mapping vector IDs to embedding row IDs and view IDs;
- a small fallback/test implementation for unit tests that does not require
  heavyweight model downloads.

This is enough to verify embedding and later build a search prototype. A true
managed vector database can replace this behind the same interface once we know
the model and scale constraints better.

## Model Metadata

Each embedding row stores:

- `model_provider`;
- `model_id`;
- `model_revision`;
- `preprocess_version`;
- `source_image_path`;
- `source_image_hash`;
- `source_image_bytes`;
- `embedding_dimension`;
- `embedding_dtype`;
- `embedding_normalized`;
- `vector_store_kind`;
- `vector_store_path`;
- `vector_id`.

This allows comparing or rerunning different models without overwriting earlier
results.

## Failure Handling

- Missing or not-yet-processed views are acked and counted as skipped.
- Already completed embeddings for the same view/model/source are acked and
  counted as skipped.
- Claim failures, model failures, and vector store failures mark the embedding
  row failed with a truncated error.
- The worker remains stateless; durable queue messages plus Postgres status are
  the source of truth.
- Downloader, processing, and embedding CLIs are long-running services by
  default. A bounded one-shot run must be requested explicitly with `--once`.
- After an empty or backpressure-paused batch, services sleep briefly and poll
  again. After skipped duplicate jobs, they immediately fetch again so old queue
  messages do not prevent fresh jobs from being reached.
- Queue backpressure uses durable consumer backlog when possible, not retained
  JetStream stream message count.

## Verification

- Unit tests for embedding queue payloads and NATS publishing.
- Unit tests for embedding job parsing and ack behavior.
- DB tests for embedding table creation, claiming, completion, dedupe, and
  retry.
- Runner tests using fake embedders and fake vector stores.
- Processing runner tests proving embedding backpressure and enqueue behavior.
- Full `uv run pytest -q` from `services/main`.
- A bounded local smoke run against a small set of already downloaded/processed
  panos when local model dependencies and weights are available.
- Manual query UI check: enter a text query, embed it with the same model, search
  local HNSW, and render matching local tile images plus view/model metadata.

## Test Query UI

The UI is local-only and not part of the future public website. It should run as
`python -m main_service.embedding.query_ui`, bind to localhost, and avoid secrets
or external services. It reads Postgres metadata and the local HNSW index, then
shows result cards with:

- generated view image;
- similarity score;
- pano ID;
- viewset and view ID;
- heading, pitch, FOV, rendered dimensions;
- embedding model and vector ID.

The UI must use the same text embedding adapter as the embedding worker uses for
image embeddings, so score behavior matches the production path.

## SigLIP Adapter Notes

For `google/siglip2-so400m-patch14-384`, text inputs must use
`padding="max_length"` with truncation. The Transformers SigLIP implementation
documents this as the training-time text padding mode, and local retrieval tests
showed that plain `padding=True` can move relevant street-level tiles below sky
or unrelated results.

Automatic device selection should prefer CUDA first, then Apple MPS, then CPU.
This keeps local Mac runs off CPU when MPS is available while preserving the same
adapter path for Linux GPU servers.
