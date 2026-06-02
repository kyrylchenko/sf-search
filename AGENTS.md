# Agent Context

This repo is building `sf-search`: a long-running panorama ingestion and semantic search system.

## Product Direction

The long-term application takes a city boundary, such as San Francisco, then:

1. Converts the boundary into map/coverage tiles.
2. Loads Google Street View panorama IDs for those tiles.
3. Deduplicates panorama IDs and tracks processing state.
4. Downloads actual panorama images.
5. Splits each panorama into many smaller views because full panoramas are too large for a CLIP model.
6. Generates CLIP vectors for those smaller views.
7. Stores panorama, tile/view, embedding, and processing metadata.
8. Builds or updates a vector index over image embeddings. The current default
   vector store is Qdrant; local HNSW remains available for experiments.
9. Later powers a website where users type natural-language queries, such as "green graffiti of a woman" or "U-Haul truck".
10. Embeds the text query and searches the vector index to return matching panorama tiles/views.

The design is intentionally flexible. Architecture, data model, storage, tiling strategy, model choice, indexing approach, and website behavior can change as we learn more.

The active focus is ingestion, panorama processing, embedding, and vector
storage only:

- City boundary input.
- Coverage/map tile generation.
- Google Street View panorama ID discovery.
- Panorama ID deduplication.
- Panorama download and local/object-storage metadata.
- Panorama view/tile generation from downloaded equirectangular images.
- Image embedding for generated panorama views. Rendered view tile files are
  temporary embedding handoff artifacts, not durable storage.
- Qdrant vector persistence with enough metadata to debug search results.
- Resumable state tracking for long-running jobs.

Do not plan or implement the public website, public search APIs, or production
query serving unless the user explicitly asks for that next. The local query UI
is a test/debug tool only.

## Current Repo Shape

- `services/main` is the only service with meaningful implementation today.
- `services/retrieval`, `services/preprocess`, and `services/embedding` are placeholders.
- `services/shared` contains shared logging setup.
- The current main service implements discovery, download, processing,
  embedding, a test query UI, DB models/services, NATS queue adapters, and a
  monitoring snapshot worker.
- `pipeline_manager.py` is scaffolding and is not a finished pipeline.
- `pano_retrieval.py` currently overlaps with `geo.py` and should eventually contain actual panorama retrieval logic or be removed.
- `target.geojson` is the current local target boundary and can change between
  experiments.

## Runtime Stack

The Docker stack currently includes:

- Postgres for durable pipeline state.
- NATS JetStream for durable worker queues.
- Qdrant for default vector storage.
- `discovery`, `downloader`, `processing`, `embedding`, `monitoring`, and
  `query-ui` containers built from `services/main`.

Important commands:

```bash
docker compose up -d postgres nats qdrant
docker compose build discovery downloader processing embedding monitoring query-ui
docker compose --profile pipeline up -d
docker compose --profile query up -d query-ui
```

For GPU embedding:

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml --profile pipeline up -d
```

With external SigNoz:

```bash
docker compose -f docker-compose.yml -f docker-compose.observability.yml --profile pipeline up -d
```

Useful operational commands:

```bash
docker compose run --rm processing uv run python -m main_service.ops requeue-processing --limit 1000
docker compose run --rm embedding uv run python -m main_service.ops requeue-embedding --limit 10000
docker compose run --rm embedding uv run python -m main_service.embedding --limit 10 --batch-size 10 --once --log-level INFO
```

Docker build safety: the root `.dockerignore` excludes `.local`, `.env`, logs,
virtualenvs, cache directories, and build artifacts. Downloaded panoramas,
temporary generated tiles, Qdrant storage, Postgres data, NATS data, and local
HNSW indexes live under ignored runtime paths and must not be copied into Docker
images or committed.

Embedding containers set `HF_HOME=/app/services/main/.local/huggingface`.
Model weights are runtime cache data and must remain under ignored `.local`
storage, never committed. The first Dockerized embedding run can spend several
minutes downloading/loading the model; subsequent runs should reuse the mounted
cache.

Qdrant notes:

- `EMBEDDING_VECTOR_STORE_KIND=qdrant` is the default.
- `QDRANT_UPSERT_WAIT=true` is the default so embedding rows become complete
  only after Qdrant accepts the point write.
- If Qdrant write fails, the embedding service must mark claimed embedding rows
  failed through the existing batch failure path.
- `EMBEDDING_VECTOR_STORE_KIND=local_hnsw` remains a local fallback.

## Reference Project

The sibling project `../pano-analyzer` contains useful prototype snippets and generated artifacts. Treat it as reference material unless the user explicitly asks to edit it.

Useful files there include:

- `src/download_panos.py`: Street View coverage lookup and async panorama download.
- `src/count_panos_sf.py`: scans coverage tiles and deduplicates panorama IDs.
- `src/pano_utils.py`: GPano XMP parsing, uncropping, and equirectangular tiling helpers.
- `src/tile.panos.py`: tile manifest generation.
- `src/embed.py`: OpenCLIP image embedding and HNSW index build. This is deferred reference material.
- `src/query.py` and `src/query.desktop.py`: text embedding and HNSW search examples. This is deferred reference material.
- `src/pano_detector.py`: perspective projection and object detection over panorama views.

Upstream StreetLevel library docs:

- `https://streetlevel.readthedocs.io/en/master/`

Before implementing or changing Street View coverage, metadata resolution, or
panorama download behavior, check these docs and the `../pano-analyzer` snippets
instead of guessing at library APIs.

## Agent Workflow

Before doing actual implementation work, write a plan in `.codex/plans/`.

For larger or ambiguous work, write a spec in `.codex/specs/` first, then a plan in `.codex/plans/`.

All meaningful changes should be documented in specs and/or plans so future agents can understand why something was done, how it was done, what alternatives were considered, and how the result was verified. If the work is small, the plan can be short, but the rationale should still be explicit.

Keep plans and specs checked into git. Do not git-ignore `.codex`.

Recommended filenames:

```text
.codex/specs/YYYY-MM-DD-topic.md
.codex/plans/YYYY-MM-DD-topic.md
```

Plans should include:

- Goal and non-goals.
- Files likely to change.
- Step-by-step implementation approach.
- Why this approach was chosen.
- Verification commands.
- Risks or open questions.

Specs should include:

- Problem and user value.
- Requirements and constraints.
- Proposed architecture and data flow.
- Storage/indexing model.
- Failure handling and resumability.
- Alternatives considered.
- Open questions.

## Git Rules

Hard rule: never perform remote/origin git operations in this repo.

Allowed:

- Inspect local state with commands such as `git status`, `git diff`, `git log`, and `git show`.
- Stage local changes with `git add`.
- Create local commits with `git commit`.
- Commit changes as work progresses, using Conventional Commits format such as `docs: add ingestion spec`, `fix: make pipeline importable`, or `feat: add panorama discovery`.

Forbidden:

- Do not run `git push`.
- Do not run `git pull`.
- Do not run `git fetch`.
- Do not run origin/remote operations such as `git remote update`, `git ls-remote`, or commands targeting `origin`.
- Do not configure, rewrite, or otherwise manipulate git remotes unless the user explicitly changes this project rule in `AGENTS.md`.

Expect local-only work. Future agents may commit locally as they go, but must not synchronize with any remote.

## Public Repo Privacy and Secrets Rules

This is a public repo. Treat everything committed here as visible to the internet.

Hard rule: never commit secrets, private infrastructure details, or private identifiers.

Do not commit:

- API keys, access tokens, refresh tokens, cookies, session IDs, passwords, or private credentials.
- Internal/private API endpoints, database URLs, hostnames, proxy URLs, or service credentials.
- Private filesystem paths, private project paths, machine-specific absolute paths, or local usernames.
- Personal names, private organization names, private customer/user data, or other identifying information unless the user explicitly says that exact value is public and belongs in git.
- `.env` files, local logs containing sensitive data, generated datasets, downloaded panoramas, vector/index artifacts, or other large/private runtime outputs.

When documentation needs an example, use placeholders such as `YOUR_API_KEY`, `https://example.internal`, `/path/to/project`, or `user@example.com`.

## Engineering Notes

- Favor resumable, idempotent processing. This system is expected to run for long periods, potentially months.
- Treat every service as standalone, durable, and restartable. Services should be able to stop and start normally with minimal disturbance: persistent state belongs in durable storage or queues, workers should be stateless where practical, and in-flight work should be recoverable through idempotent status transitions.
- On restart, discovery must use the database as the source of truth. If a map tile is already marked complete, do not re-fetch coverage for that tile; re-push linked pano IDs that are not in terminal download states onto the downloader queue so downstream workers can resume. Duplicate queue messages are acceptable because downstream services must deduplicate/idempotently claim work.
- Preserve enough metadata to debug search results later: pano ID, source tile,
  location, date if available, view spec, temporary image path/blob key when it
  existed, image hash/bytes, embedding model, and indexing version.
- Full downloaded panoramas are the durable image source of truth. Rendered
  panorama view tiles may be written to `.local/panorama-view-tmp` as temporary
  handoff files for embedding, but embedding should delete them after handling a
  job. Query/debug UI should render tiles on demand from stored full panos and
  `panorama_view_table` metadata rather than requiring persisted tile files.
- Avoid assuming the first implementation is final. Keep interfaces clear so crawler, downloader, tiler, embedder, and indexer can be split into separate workers later.
- Do not commit local `.env`, virtualenvs, logs, generated image datasets, or large vector/index artifacts unless explicitly requested.
- Be careful with Google/Street View request volume and retry behavior. Prefer bounded concurrency, checkpointing, and backoff.
