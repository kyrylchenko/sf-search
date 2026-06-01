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
  - embedding index files.

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
PANO_DOWNLOAD_STORAGE_DIR=.local/panoramas
PANO_VIEW_STORAGE_DIR=.local/panorama-views
EMBEDDING_VECTOR_STORE_DIR=.local/embedding-indexes
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
docker compose up -d postgres nats
```

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
docker compose logs -f discovery downloader processing embedding
```

Embedding startup logs should include requested device, actual device, dtype,
and batch size.
