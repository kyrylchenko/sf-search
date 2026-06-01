# Server Runbook

This runbook keeps server-specific values in environment files. Do not commit
real API keys, tokens, private endpoints, private paths, or production
credentials.

## Prerequisites

- Docker and Docker Compose.
- For GPU embedding: NVIDIA driver, NVIDIA Container Toolkit, and a CUDA-capable
  PyTorch install inside the app image.
- Enough persistent disk for:
  - Postgres data;
  - NATS JetStream state;
  - downloaded panos;
  - generated panorama view tiles;
  - Qdrant vector storage;
  - optional local HNSW embedding index files;
  - optional monitoring snapshots and telemetry export state.

## Configure

Copy `.env.example` to `.env` at the repo root for Compose-level values, and copy
`services/main/.env.example` to `services/main/.env` for app values. Keep
server-specific values local.

Important app settings:

```text
DB_HOST=postgres
NATS_URL=nats://nats:4222
EMBEDDING_DEVICE=auto
EMBEDDING_DTYPE=float16
EMBEDDING_BATCH_SIZE=1
EMBEDDING_VECTOR_STORE_KIND=qdrant
QDRANT_URL=http://qdrant:6333
QDRANT_COLLECTION=panorama_view_embeddings_siglip2
QDRANT_VECTOR_ON_DISK=true
QDRANT_HNSW_ON_DISK=false
QDRANT_ON_DISK_PAYLOAD=true
QDRANT_UPSERT_WAIT=true
PANO_DOWNLOAD_STORAGE_DIR=.local/panoramas
PANO_VIEW_STORAGE_DIR=.local/panorama-views
EMBEDDING_VECTOR_STORE_DIR=.local/embedding-indexes
OBSERVABILITY_ENABLED=false
OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317
MONITORING_INTERVAL_SECONDS=15
```

For a CUDA server, set or override:

```text
EMBEDDING_DEVICE=cuda
EMBEDDING_DTYPE=float16
EMBEDDING_BATCH_SIZE=<measure-on-server>
```

Start with a conservative batch size, run the smoke command below, then increase
until throughput stops improving or VRAM gets tight.

## Start Infrastructure

```bash
docker compose up -d postgres nats qdrant
```

Qdrant is also started automatically with the pipeline or query UI profiles. To
start it by itself:

```bash
docker compose up -d qdrant
```

Qdrant HTTP is exposed on `http://localhost:6333`, and its storage is persisted
under `./.local/qdrant-storage`.

## Build App Image

```bash
docker compose build discovery downloader processing embedding query-ui
```

## GPU Smoke Test

Run after at least a few view tiles exist under `services/main/.local`.

Host:

```bash
cd services/main
uv run python -m main_service.embedding.smoke --device cuda --dtype float16 --batch-size 8 --limit 16
```

Docker with GPU override:

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml run --rm embedding \
  uv run python -m main_service.embedding.smoke --device cuda --dtype float16 --batch-size 8 --limit 16
```

The JSON output should show `cuda_available=true`, a CUDA device name, and model
parameters on `cuda:0`.

## Start Pipeline

CPU/auto device:

```bash
docker compose --profile pipeline up -d
```

CUDA:

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml --profile pipeline up -d
```

With external SigNoz observability:

1. Start SigNoz separately with the official SigNoz Docker Compose stack.
2. Confirm the SigNoz collector publishes OTLP gRPC on host port `4317`.
3. Start this app stack with the observability override:

```bash
docker compose -f docker-compose.yml -f docker-compose.observability.yml --profile pipeline up -d --build
```

CUDA plus external SigNoz:

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml -f docker-compose.observability.yml --profile pipeline up -d --build
```

The observability override sets:

```text
OBSERVABILITY_ENABLED=true
OTEL_EXPORTER_OTLP_ENDPOINT=http://host.docker.internal:4317
DEPLOYMENT_ENVIRONMENT=server
```

On Linux, the override also maps `host.docker.internal` to the Docker host
gateway. If SigNoz runs on another host or network, keep the real endpoint only
in local env files or deployment config; do not commit private hostnames.

Query UI:

```bash
docker compose --profile query up -d query-ui
```

With CUDA query embedding:

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml --profile query up -d query-ui
```

Open `http://localhost:8787`.

## Requeue From Postgres

NATS is durable, but Postgres is the source of truth. If queues are recreated or
you move machines, rebuild work from DB.

Requeue downloaded panos that do not have completed views:

```bash
docker compose run --rm processing \
  uv run python -m main_service.ops requeue-processing --limit 1000
```

Requeue completed views missing embeddings for the configured model:

```bash
docker compose run --rm embedding \
  uv run python -m main_service.ops requeue-embedding --limit 10000
```

The downstream services dedupe through Postgres, so duplicate queue messages are
acceptable.

## Logs

```bash
docker compose logs -f discovery downloader processing embedding monitoring
```

Embedding startup logs should include requested device, actual device, dtype,
batch size, vector store kind, and vector store path. Qdrant writes emit
`qdrant_upsert_start` and `qdrant_upsert_complete`; SigNoz timings for
`embedding_vector_store_batch` show if Qdrant writes are slowing embedding.

When observability is enabled, services also export logs through OpenTelemetry to
SigNoz. The monitor service emits snapshots containing:

- map tile counts by discovery status;
- pano counts by download status;
- generated view counts by processing status;
- embedding counts by embedding status;
- NATS download, processing, and embedding queue depths;
- a small coverage summary with pano count and coordinate bounds.

SigNoz is the timing/logs/metrics surface. A real geospatial coverage map is
still a separate UI concern; current map readiness comes from Postgres location
metadata and the monitor's coverage summary.
