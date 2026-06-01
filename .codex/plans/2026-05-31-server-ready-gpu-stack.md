# Server-Ready GPU Stack Plan

## Goal

Make the ingestion/processing/embedding stack easier to run on a GPU server
without manually starting every service, while keeping server-specific values
configurable later.

## Approach

- Add explicit embedding runtime config with safe defaults:
  - `EMBEDDING_DEVICE=auto`
  - `EMBEDDING_BATCH_SIZE=1`
- Log the actual model device, dtype, and embedding batch settings at startup.
- Add batched image embedding support while preserving the current single-image
  behavior.
- Add DB-driven requeue/backfill commands:
  - downloaded panos missing processing jobs;
  - completed views missing embeddings.
- Add a GPU/embedding smoke command that reports runtime device details and
  bounded local tile throughput.
- Dockerize app services in Compose so discovery, downloader, processing,
  embedding, and query UI can be started together with Postgres and NATS.
- Add server runbook docs with placeholders for server-specific paths and GPU
  settings.

## Verification

- Unit-test config defaults and parser options.
- Unit-test batched embedding runner behavior with fakes.
- Unit-test requeue/backfill selection.
- Run focused tests for embedding, processing/downloader config, and requeue.
- Run the full main-service test suite.
