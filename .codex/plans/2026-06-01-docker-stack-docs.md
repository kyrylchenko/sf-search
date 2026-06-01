# Docker Stack Docs Plan

## Goal

Make the repo startup/control docs clear after adding Dockerized Qdrant and confirm future agents understand Docker builds do not copy local panorama data.

## Non-Goals

- No service behavior changes.
- No new runtime config defaults.
- No live data or local paths in committed docs.

## Files

- `AGENTS.md`
- `services/main/README.md`
- `docs/server-runbook.md`

## Approach

1. Document that `.dockerignore` excludes `.local`, `.env`, logs, virtualenvs, and generated artifacts from Docker build context.
2. Update agent context to reflect the current default vector store: Qdrant with local HNSW fallback.
3. Add concise startup/control commands for infrastructure, pipeline workers, query UI, requeue commands, and bounded embedding smoke verification.
4. Document Docker model cache behavior and keep `HF_HOME` under ignored `.local` runtime storage.
5. Keep examples generic and safe for a public repo.

## Verification

- `git diff --check`
- Review docs for accidental secrets/private paths.
- `docker compose config --quiet`
- Bounded Dockerized Qdrant embedding run:
  - `docker compose run --rm embedding uv run python -m main_service.ops --log-level INFO requeue-embedding --limit 2`
  - `docker compose run --rm embedding uv run python -m main_service.embedding --limit 2 --batch-size 2 --once --log-level INFO --device cpu`

## Live Verification Result

- Qdrant `/readyz` returned `all shards are ready`.
- Dockerized requeue command enqueued 2 embedding jobs.
- Dockerized embedding command returned `{"embedded": 2, "failed": 0, "skipped": 0}`.
- Qdrant collection `panorama_view_embeddings_siglip2` reported `points_count=2`.
- Postgres rows for the two embeddings were `complete` with
  `vector_store_kind=qdrant`.
