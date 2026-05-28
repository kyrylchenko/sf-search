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
8. Builds or updates an HNSW index over image embeddings.
9. Later powers a website where users type natural-language queries, such as "green graffiti of a woman" or "U-Haul truck".
10. Embeds the text query and searches the HNSW index to return matching panorama tiles/views.

The design is intentionally flexible. Architecture, data model, storage, tiling strategy, model choice, indexing approach, and website behavior can change as we learn more.

The active focus is ingestion and panorama processing only:

- City boundary input.
- Coverage/map tile generation.
- Google Street View panorama ID discovery.
- Panorama ID deduplication.
- Panorama download and local/object-storage metadata.
- Panorama view/tile generation from downloaded equirectangular images.
- Resumable state tracking for long-running jobs.

Do not plan or implement website, text-query UI, public search APIs, or HNSW query serving unless the user explicitly asks for that next. Embeddings and indexing are also separate follow-up stages unless a current plan says otherwise.

## Current Repo Shape

- `services/main` is the only service with meaningful implementation today.
- `services/retrieval`, `services/preprocess`, and `services/embedding` are placeholders.
- `services/shared` contains shared logging setup.
- The current main service has early code for GeoJSON tile generation, SQLAlchemy models, panorama DB service helpers, and panorama view tiling.
- `pipeline_manager.py` is scaffolding and is not a finished pipeline.
- `pano_retrieval.py` currently overlaps with `geo.py` and should eventually contain actual panorama retrieval logic or be removed.
- `target.geojson` is a sample target boundary, not necessarily the final SF boundary.

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
- Preserve enough metadata to debug search results later: pano ID, source tile, location, date if available, view spec, image path/blob key, embedding model, and indexing version.
- Avoid assuming the first implementation is final. Keep interfaces clear so crawler, downloader, tiler, embedder, and indexer can be split into separate workers later.
- Do not commit local `.env`, virtualenvs, logs, generated image datasets, or large vector/index artifacts unless explicitly requested.
- Be careful with Google/Street View request volume and retry behavior. Prefer bounded concurrency, checkpointing, and backoff.
