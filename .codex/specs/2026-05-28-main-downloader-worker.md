# Main Downloader Runner Spec

## Problem

Discovery now persists pano IDs and publishes small downloader jobs to NATS
JetStream. The next step is to download panorama images and metadata without
losing work on restarts.

The downloader should live in `services/main`. It is not a separate service or
deployment unit right now; it is a simple runner/CLI in the same codebase. It
should still be restartable and durable through Postgres and NATS, and safe
with duplicate queue messages.

## Requirements

- Consume downloader jobs from `pano.download.requested`.
- Use a durable JetStream consumer so unacked jobs survive runner restarts.
- Process up to a configurable number of downloads concurrently; default `5`.
- Deduplicate by pano ID through Postgres before downloading. If a duplicate
  downloader message arrives for a pano that is already `downloaded` with an
  image path/hash, ack and skip it.
- Resolve each pano ID through `streetlevel.streetview.find_panorama_by_id_async`.
- Download the actual panorama image through `streetlevel.streetview.download_panorama_async`.
- Save images under ignored local storage, default `./.local/panoramas`.
- Capture useful metadata from the resolved panorama object.
- Update Postgres with:
  - `download_status`
  - `image_path`
  - `image_hash`
  - latitude/longitude when available
  - metadata JSON when available
  - attempt count and last error on failure
- Publish a small processing job after a successful first-time download.
- Ack the downloader NATS message only after Postgres is updated and the
  processing job is published.
- Avoid committing downloaded images, metadata dumps, or generated outputs.

## Non-Goals

- Do not split this into a separate deployable service or long-running process
  model yet. Keep it as a bounded runner inside `services/main`.
- Do not generate panorama views in this milestone.
- Do not run CLIP embeddings or HNSW indexing.
- Do not implement the website/query path.

## Queue Design

Existing downloader request stream:

```text
stream: PANO_DOWNLOADS
subject: pano.download.requested
```

Downloader input payload:

```json
{
  "pano_id": "example-pano-id",
  "source": "coverage_discovery",
  "discovered_from_tile": {"x": 1, "y": 2, "z": 17}
}
```

New processing stream:

```text
stream: PANO_PROCESSING
subject: pano.processing.requested
```

Processing payload:

```json
{
  "pano_id": "example-pano-id",
  "image_path": ".local/panoramas/example-pano-id.jpg",
  "source": "pano_downloader"
}
```

The processing payload stays small and references local storage. It never
contains image bytes.

## Database Model

Extend `panorama_table` with:

- `image_path: str | None`
- `metadata_json: dict | None`, stored as SQLAlchemy `JSON`
- `downloaded_at: datetime | None`

Keep existing:

- `download_status`
- `image_hash`
- `latitude`
- `longitude`
- `attempt_count`
- `last_error`

Terminal download statuses:

- `downloaded`
- `skipped`

Retryable statuses:

- `pending`
- `queued`
- `downloading`
- `failed`

`downloading` remains retryable because the runner may have died mid-download.
A later lease/heartbeat model can make this more precise.

## Runner Flow

For each fetched NATS message:

1. Parse and validate the pano ID.
2. Ask Postgres to claim the pano for download. This is the dedupe gate.
3. If the pano is already `downloaded` with an existing image path/hash, ack and
   skip without downloading again and without publishing another processing job.
4. Mark the pano `downloading`.
5. Resolve metadata with `find_panorama_by_id_async`.
6. Download image to a temporary file under the configured storage directory.
7. Atomically move the temp file to the final image path.
8. Compute SHA-256 hash.
9. Persist download success metadata and `download_status="downloaded"`.
10. Publish a processing job.
11. Ack the original download message.

On failure:

1. Increment attempt count.
2. Store a bounded error string.
3. Mark `download_status="failed"` unless max attempts is exceeded, in which
   case mark `skipped`.
4. Ack the original message after DB failure state is saved, because retry is
   handled by rediscovery/requeue and explicit failed-status replay rather than
   infinite immediate redelivery.

## Local Storage

Default image path:

```text
.local/panoramas/<pano-id>.jpg
```

The `.local/` directory is git-ignored. The path is intentionally local and can
later be replaced by object storage while keeping the DB field as a storage key.

## Testing

Use offline tests for:

- Message parsing and serialization.
- Processing queue publishing.
- DB status transitions.
- Existing-image idempotency.
- Successful download flow with fake Street View client and fake filesystem.
- Failure flow updates attempt count and status.
- Duplicate input messages are skipped through Postgres state.
- Runner concurrency limit uses configured value.

Use one live verification after unit tests:

- consume a small number of messages from local NATS,
- download one or a few panos from the previously discovered tile,
- confirm image files exist under `.local/panoramas`,
- confirm Postgres rows are `downloaded`,
- confirm processing messages are published.

## Open Decisions

- The first implementation should default to `5` concurrent downloads.
- A separate CLI argument can cap total downloads per run for safe manual tests,
  e.g. `--limit 5`.
- Metadata shape should be best-effort scalar JSON, not a committed raw dump.
