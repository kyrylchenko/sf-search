# Deduplicate Compose App Image

## Goal

Avoid rebuilding/exporting the same large sf-search Docker image once per service on normal code updates.

## Non-goals

- Do not split CPU and GPU images yet.
- Do not change service commands, volumes, ports, or runtime behavior.
- Do not remove the existing CUDA/PyTorch compatibility pin.

## Context

All app services use the same `x-main-service` build context and Dockerfile. Without an explicit shared image name, Compose exports separate service images such as `sf-search-discovery`, `sf-search-embedding`, and `sf-search-query-ui`. The layers are shared, but export/unpack is still repeated and dominates rebuild time for the large CUDA/PyTorch image.

## Approach

1. Add `image: sf-search-main:latest` to the shared `x-main-service` anchor.
2. Tag the already-built image as `sf-search-main:latest` to avoid another rebuild during this deployment.
3. Keep a future Dockerfile split as a separate change: non-embedding workers should eventually use a smaller non-CUDA image, while embedding/query-ui keep the CUDA dependencies.

## Verification

- `docker tag sf-search-embedding:latest sf-search-main:latest`
- `docker compose -f docker-compose.yml -f docker-compose.gpu.yml -f docker-compose.observability.yml config --services`
- Start pipeline and verify containers run from `sf-search-main:latest`.
