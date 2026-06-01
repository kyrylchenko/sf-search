# Qdrant Vector Store Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the default embedding vector store with Qdrant while keeping local HNSW as a configurable fallback.

**Architecture:** Add a Qdrant-backed `VectorStore` implementation and a factory used by the embedding CLI and query UI. The embedding runner already fails claimed rows when `vector_store.add_many()` raises, so Qdrant write failures should use that path unchanged.

**Tech Stack:** Python 3.14, qdrant-client, Qdrant Docker image, SQLAlchemy/Postgres, NATS, pytest, OpenTelemetry progress events.

---

## Files

- Create `services/main/main_service/embedding/qdrant_store.py` for Qdrant vector storage.
- Create `services/main/main_service/embedding/vector_store_factory.py` for config-driven store construction.
- Create `services/main/tests/embedding/test_qdrant_store.py`.
- Create `services/main/tests/embedding/test_vector_store_factory.py`.
- Modify `services/main/main_service/config.py`.
- Modify `services/main/main_service/embedding/__main__.py`.
- Modify `services/main/main_service/embedding/query_ui.py`.
- Modify `services/main/tests/embedding/test_embedding_runner.py`.
- Modify `services/main/tests/embedding/test_embedding_config.py`.
- Modify `services/main/tests/embedding/test_embedding_entrypoint.py`.
- Modify `services/main/pyproject.toml` and `services/main/uv.lock`.
- Modify `docker-compose.yml`, `.env.example`, and `services/main/.env.example`.
- Modify `docs/server-runbook.md` and `services/main/README.md`.

## Tasks

- [ ] Add failing tests for Qdrant config defaults and embedding CLI flags.
- [ ] Add failing tests for Qdrant store behavior using a fake client:
  - collection is created when missing;
  - `add_many()` upserts normalized float32 vectors with point IDs and payloads;
  - `search()` converts Qdrant scored points into `(vector_id, score)` pairs;
  - client exceptions propagate out of `add_many()`.
- [ ] Add a failing runner test proving `vector_store.add_many()` failure marks all claimed batch rows failed.
- [ ] Implement Qdrant settings in `Settings`.
- [ ] Implement `QdrantVectorStore` with lazy `qdrant_client` imports.
- [ ] Implement a vector-store factory for `qdrant` and `local_hnsw`.
- [ ] Wire the factory into embedding CLI and query UI.
- [ ] Add `qdrant-client` to the embedding dependency group and update the lockfile.
- [ ] Add Qdrant service, persisted storage, and env vars to Docker Compose.
- [ ] Update runbook and README commands to explain Qdrant defaults and local HNSW fallback.
- [ ] Run focused tests after each TDD slice, then run `uv run pytest -q`.
- [ ] Validate Compose with `docker compose config --quiet`.
- [ ] Commit with Conventional Commit format.

## Rationale

Direct Qdrant writes are the smallest production-ready step. The runner already has the correct correctness boundary: DB completion happens after vector storage succeeds. Keeping writes direct also makes Qdrant duration visible in the existing `embedding_vector_store_batch` telemetry before adding service complexity.

## Risks

- `QDRANT_UPSERT_WAIT=true` can slow the embedding hot path. This is intentional for correctness first and can be changed after SigNoz confirms write cost.
- Qdrant collection creation must use the configured embedding dimension. If the model changes, use a new collection or recreate the old one.
- Existing local HNSW vectors will not automatically appear in Qdrant.

## Verification Commands

```bash
cd services/main
uv run pytest tests/embedding/test_qdrant_store.py -q
uv run pytest tests/embedding/test_vector_store_factory.py tests/embedding/test_embedding_runner.py tests/embedding/test_embedding_config.py tests/embedding/test_embedding_entrypoint.py -q
uv run pytest -q
cd ../..
docker compose config --quiet
```
