# main service

## Docker Stack

The local/server stack is Dockerized around Postgres, NATS JetStream, Qdrant,
and the main pipeline workers. Runtime data is mounted outside the image:

- Postgres: `./.local/postgres-data`
- Qdrant: `./.local/qdrant-storage`
- Panoramas, temporary generated view handoff files, local HNSW fallback indexes:
  `services/main/.local`
- Hugging Face model cache in Docker:
  `services/main/.local/huggingface`

Docker build context safety: the repo root `.dockerignore` excludes `**/.local`,
`**/.env`, logs, virtualenvs, caches, and build outputs. Building the app image
does not copy downloaded panos or temporary generated view tiles into the Docker image.
Large downloads during `docker compose build` are Python/Torch/CUDA image
dependencies, not panorama data.

Embedding containers set `HF_HOME=/app/services/main/.local/huggingface`, which
is inside the mounted `.local` directory. The first model run may download
SigLIP weights; later container runs should reuse that cache instead of
redownloading.

Start infrastructure:

```bash
docker compose up -d postgres nats qdrant
```

Build app containers:

```bash
docker compose build discovery downloader processing embedding monitoring query-ui
```

Start the long-running pipeline:

```bash
docker compose --profile pipeline up -d
```

Start the test query UI:

```bash
docker compose --profile query up -d query-ui
```

Useful control commands:

```bash
docker compose logs -f discovery downloader processing embedding monitoring
docker compose run --rm processing uv run python -m main_service.ops requeue-processing --limit 1000
docker compose run --rm embedding uv run python -m main_service.ops requeue-embedding --limit 10000
docker compose run --rm embedding uv run python -m main_service.embedding --limit 10 --batch-size 10 --once --log-level INFO
```

Use `docker-compose.gpu.yml` for CUDA embedding and
`docker-compose.observability.yml` when SigNoz is running separately.

Qdrant is the default vector store:

```text
EMBEDDING_VECTOR_STORE_KIND=qdrant
QDRANT_URL=http://qdrant:6333
QDRANT_COLLECTION=panorama_view_embeddings_siglip2
QDRANT_UPSERT_WAIT=true
```

`QDRANT_UPSERT_WAIT=true` means an embedding row is marked complete only after
Qdrant accepts the vector upsert. For a local file-backed experiment, set
`EMBEDDING_VECTOR_STORE_KIND=local_hnsw`.

## Local CLI

Run the coverage discovery service from this directory:

```bash
uv run python -m main_service.discovery --log-level INFO
```

It reads the configured boundary GeoJSON, queries Street View coverage tiles,
persists tile/pano state in Postgres, and publishes downloader jobs to
`pano.download.requested`. It keeps polling until stopped; add `--once` for one
bounded discovery pass.

Run the panorama downloader service from this directory:

```bash
uv run python -m main_service.downloader --limit 5
```

The downloader consumes durable NATS JetStream jobs from
`pano.download.requested`, deduplicates through Postgres, writes images under
`.local/panoramas`, and publishes small processing jobs to
`pano.processing.requested`.

Worker commands run continuously until stopped. They log progress to stderr and
write one JSON summary line to stdout per batch. Use `--once` only for an
explicit bounded smoke run. Use `--log-level DEBUG` when you need queue depth
checks, storage paths, and detailed job state.

Run the panorama preprocessing service:

```bash
uv run python -m main_service.processing --limit 1
```

The preprocessor consumes durable NATS JetStream jobs from
`pano.processing.requested`, loads viewsets from `../../docs/data/viewsets` by
default, renders perspective views through the shared processing renderer, saves
temporary handoff images under `.local/panorama-view-tmp`, and records each
generated view in `panorama_view_table`.

The preprocessor also publishes small embedding jobs to
`pano.embedding.requested`. It checks the embedding queue before pulling another
panorama; by default it pauses at 100 queued view jobs, but it can finish and
enqueue all views for the current panorama.

Useful local options:

```bash
uv run python -m main_service.processing \
  --limit 1 \
  --concurrency 4 \
  --max-view-concurrency 4 \
  --log-level INFO \
  --render-scale 2 \
  --viewsets-dir ../../docs/data/viewsets \
  --storage-dir .local/panorama-view-tmp
```

`--concurrency` means simultaneous view renders inside one loaded panorama. Keep
it low for full-resolution Google panoramas; `--max-view-concurrency` defaults
to `4` and caps unsafe manual values because `py360convert` allocates large
native arrays while rendering.

The renderer uses the same `py360convert.e2p` projection and bicubic
interpolation as the visualizer. `opencv-python-headless` is installed so
`py360convert` can use its OpenCV sampler path; processing logs include
`processing_renderer_backend` and should report `backend="opencv"`.

Run the panorama view embedding service:

```bash
uv sync --group embedding
uv run python -m main_service.embedding --limit 10 --log-level INFO
```

The embedder consumes durable NATS JetStream jobs from
`pano.embedding.requested`, claims model-specific rows in
`panorama_view_embedding_table`, embeds temporary generated view images, writes
vectors to Qdrant by default, then deletes the temporary tile files. Full
panorama files remain the durable image source of truth.

For one-shot checks, add `--once`:

```bash
uv run python -m main_service.embedding --limit 10 --once --log-level INFO
```

The default model settings target `google/siglip2-so400m-patch14-384`:

```text
EMBEDDING_MODEL_PROVIDER=transformers
EMBEDDING_MODEL_ID=google/siglip2-so400m-patch14-384
EMBEDDING_PREPROCESS_VERSION=siglip2-384-rgb-v1
EMBEDDING_DIMENSION=1152
EMBEDDING_DTYPE=float16
EMBEDDING_VECTOR_STORE_KIND=qdrant
QDRANT_URL=http://localhost:6333
QDRANT_COLLECTION=panorama_view_embeddings_siglip2
QDRANT_UPSERT_WAIT=true
```

`QDRANT_UPSERT_WAIT=true` keeps DB state strict: an embedding row is marked
complete only after Qdrant accepts the point write. To use the old file-backed
index for a local experiment, set `EMBEDDING_VECTOR_STORE_KIND=local_hnsw`; it
stores indexes under `EMBEDDING_VECTOR_STORE_DIR`.

Run the local test-only query UI:

```bash
uv sync --group embedding
uv run python -m main_service.embedding.query_ui --log-level INFO
```

Open `http://127.0.0.1:8787`. The UI embeds the text query with the same local
model, searches the configured vector store, and renders result tiles on demand
from the stored full panoramas. It uses paged JSON loading and infinite scroll
for local debugging. This is not the future public website.

## Viewset Visualizer

Run the local browser visualizer against a downloaded pano:

```bash
uv run python -m main_service.tools.viewset_visualizer \
  --pano .local/panoramas/example-pano-id.jpg \
  --viewsets ../../docs/data/viewsets
```

Open `http://127.0.0.1:8765`.

The visualizer computes accurate sampled perspective-frustum boundaries in
Python using the same equirectangular coordinate model as `py360convert`. The
browser only draws the returned normalized polygons, so the UI does not carry a
separate pano-projection implementation.

Use `Show all`, `Hide all`, or the compact pitch/heading checkbox matrix to
control overlay noise. Large presets default to hidden overlays so individual
views can be inspected without covering the whole pano.

Select a view in the sidebar, then use `Open selected view` or the search box to
open the server-rendered 2D perspective view in a new browser tab.

That view page can toggle between the local rendered perspective image and a
Google Maps Embed Street View iframe. Set the API key only in your local
environment:

```bash
export GOOGLE_MAPS_EMBED_API_KEY=YOUR_KEY
```

Google expects north-based headings. Viewsets store pano-relative headings; the
visualizer derives Google heading as:

```text
google_heading = (north_offset + relative_heading) % 360
```

By default `north_offset` is read from GPano XMP `PoseHeadingDegrees` in the
downloaded panorama. Override it with `--north-offset` when needed.
