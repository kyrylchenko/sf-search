# Ingestion and Panorama Processing Spec

## Purpose

Build the backend pipeline that turns a city boundary into a durable set of downloaded Google Street View panoramas and generated panorama views. This is the foundation for later embedding and search work, but this spec intentionally stops before CLIP vectors, HNSW indexing, querying, or a website.

## Scope

In scope:

- Read a target boundary from GeoJSON.
- Generate coverage/map tiles for that boundary.
- Query Google Street View coverage by map tile.
- Deduplicate panorama IDs across coverage tiles.
- Fetch panorama metadata when available.
- Download panoramas with bounded concurrency and retry tracking.
- Generate smaller panorama views from each downloaded equirectangular panorama.
- Persist processing state so the job can resume after crashes or deploys.
- Preserve enough metadata to debug future embeddings and search results.

Out of scope:

- CLIP/OpenCLIP embedding generation.
- HNSW index creation.
- Text query embedding.
- Public APIs and website UI.
- Distributed queue infrastructure unless a later plan introduces it.

## Working Assumptions

- The first production target can run as a single long-running worker on one server.
- The data model should still allow future splitting into crawler, downloader, tiler, embedder, and indexer workers.
- The pipeline may run for months, so all stages must be idempotent and resumable.
- Google Street View calls must use bounded concurrency and backoff.
- Generated images and future vectors can become large, so code should avoid committing artifacts to git.
- This is a public repo. Real API keys, tokens, credentials, private endpoints, private filesystem paths, personal names, private organization names, downloaded panoramas, generated views, vectors, and indexes must not be committed.

## Data Flow

1. Load one GeoJSON feature collection from config.
2. Convert the boundary into Web Mercator map tiles at the configured zoom.
3. Upsert map tile rows with discovery state.
4. For each pending map tile, call Street View coverage lookup.
5. Upsert panorama rows by original Google panorama ID.
6. Record which map tile discovered which panorama.
7. For each pending panorama, fetch metadata and download the panorama image.
8. For each downloaded panorama, load/normalize the equirectangular image and generate configured views.
9. Store one row per generated panorama view with yaw, pitch, roll, FOV, dimensions, and image path.

## Storage Model

Use relational state tables first. A later plan may add object storage, vector stores, or queue-specific tables.

Core entities:

- `MapTile`: coverage tile `(x, y, z)`, discovery status, attempts, timestamps, and last error.
- `Panorama`: Google panorama ID, latitude, longitude, capture date if available, metadata status, download status, image path, attempts, timestamps, and last error.
- `MapTilePanorama`: many-to-many link between map tiles and panorama IDs because one panorama can appear in multiple coverage tiles.
- `PanoramaView`: generated view/tile from a downloaded panorama, including view spec, image path, generation status, and future embedding status fields.

Status values should be explicit strings or enums, such as `pending`, `processing`, `complete`, `failed`, and `skipped`.

## Processing Boundaries

- Boundary/tile generation should stay in `main_service.geo`.
- Street View network access should live behind a small adapter so tests can use fakes.
- Downloading should live separately from coverage discovery.
- View generation should be pure where possible: given an image and a list of view specs, return generated views and metadata.
- Orchestration should coordinate state transitions but not contain low-level Street View or image manipulation details.

## Failure Handling

- Every external call should have bounded retries.
- Failed map tiles and panoramas should keep `last_error` and `attempt_count`.
- The worker should be restartable without deleting successful work.
- Duplicate panorama IDs should not produce duplicate panorama rows.
- Duplicate generated views should be prevented by a uniqueness key based on panorama ID plus view spec.

## Reference Material

Use `../pano-analyzer` as read-only reference:

- `src/download_panos.py` for Street View coverage and async download patterns.
- `src/count_panos_sf.py` for coverage tile scanning and deduplication.
- `src/pano_utils.py` for GPano XMP handling and equirectangular image normalization.
- `src/tile.panos.py` for view manifest fields.
- `src/new/grid_app/other/pano_manipulation_service.py` and `src/pano_detector.py` for perspective projection ideas.

## Open Questions

- Whether downloaded panorama images should initially live on local disk, object storage, or both.
- Whether the first worker should use SQLAlchemy-only persistence or add a queue table abstraction.
- Whether view generation should use fixed equirectangular crops, perspective projections, or both.
- What default view spec set gives useful recall without exploding storage.
- What request rate is acceptable for Street View coverage and download calls.
