# Panorama View Embedding Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a restart-safe embedding stage that consumes processed panorama views, stores model-specific embedding metadata in Postgres, writes vectors to a local HNSW store, adds processing-to-embedding queue backpressure, and provides a local test query UI.

**Architecture:** Extend the existing NATS queue helper with embedding messages, add a model-specific embedding table, publish embedding jobs from the processing runner, and implement a bounded embedding runner under `main_service.embedding`. The runner uses pluggable image/text embedder and vector store interfaces so SigLIP2, CLIP, and future stores can be swapped without schema churn. The query UI uses the same text embedder and local HNSW index to display locally stored view images.

**Tech Stack:** Python 3.14, SQLAlchemy, Postgres, NATS JetStream, NumPy, Pillow, pytest, optional hnswlib/Transformers/Torch runtime adapters, local filesystem vector artifacts.

---

## Files

- Modify `services/main/main_service/config.py` with embedding queue, model, and local vector store settings.
- Modify `services/main/main_service/ingestion/download_queue.py` with `PanoEmbeddingMessage`, in-memory queue, and NATS queue.
- Modify `services/main/main_service/processing/runner.py` and `processing/__main__.py` to apply embedding backpressure and enqueue completed views.
- Create `services/main/main_service/db/models/panorama_view_embedding.py`.
- Modify `services/main/main_service/db/models/__init__.py` and `db/initialize_engine.py`.
- Create `services/main/main_service/db/services/panorama_view_embedding_service.py`.
- Create `services/main/main_service/embedding/` modules for NATS source, model adapter, vector store, runner, and CLI.
- Create `services/main/main_service/embedding/query_ui.py` for local text query inspection.
- Add focused tests under `services/main/tests/db`, `tests/ingestion`, `tests/processing`, and `tests/embedding`.
- Update `services/main/README.md`.

## Tasks

- [ ] Add failing queue/config tests for embedding queue defaults and payloads.
- [ ] Implement embedding queue settings and queue classes.
- [ ] Add failing DB tests for model-specific panorama view embedding rows.
- [ ] Implement embedding table and DB service.
- [ ] Add failing processing runner tests for embedding queue backpressure and enqueue behavior.
- [ ] Update processing runner and CLI to publish embedding jobs.
- [ ] Add failing embedding NATS source tests.
- [ ] Implement embedding NATS source.
- [ ] Add failing embedding runner tests with fake embedder/vector store.
- [ ] Implement embedding runner, local HNSW vector store, and lazy SigLIP2 adapter.
- [ ] Add embedding CLI tests and implementation.
- [ ] Add failing query UI tests for text query result rendering.
- [ ] Implement local query UI backed by text embeddings and the local HNSW index.
- [ ] Run a bounded local smoke against available processed panos if local model dependencies are available.
- [ ] Update docs, run full tests, and commit.

## Verification Commands

Run from `services/main`:

```bash
uv run pytest tests/ingestion/test_download_queue.py tests/db/test_panorama_view_embedding_service.py tests/processing/test_processing_runner.py tests/embedding -q
uv run pytest -q
```

Manual local checks after dependencies/model weights are available:

```bash
uv run python -m main_service.embedding --limit 10
uv run python -m main_service.embedding.query_ui
```

## Risks

- Real SigLIP2 execution needs optional local ML dependencies and model weights.
  Tests will use fake embedders so pipeline behavior remains verifiable without
  downloading models.
- The first local HNSW store validates persistence and IDs, but it is not the
  final city-scale vector database.
- Backpressure is checked before fetching the next pano, so a single pano can
  push the embedding queue above the threshold. This is intentional.
- The query UI is local-only test tooling. It is not the website.
