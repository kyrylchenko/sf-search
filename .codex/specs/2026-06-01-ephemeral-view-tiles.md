# Ephemeral View Tiles Spec

## Problem

Persisting every rendered panorama view tile is too expensive. The system still
needs tile images briefly for model embedding, but after the embedding job has
stored its vector and metadata, the tile bytes should be removed. Query/test UI
must continue to display matching tiles without relying on stored tile files.

## Requirements

- Keep downloaded panorama files as the durable image source of truth.
- Processing may write rendered view tiles to a temporary handoff folder.
- Embedding must delete temporary rendered tile files after handling each job.
- Embedding metadata must keep source tile path, hash, bytes, model, vector ID,
  and status for debugging even after the tile file is removed.
- Completed views must remain deduplicated even if their `image_path` is cleared.
- Query UI must generate result thumbnails from stored panoramas and view specs
  instead of serving stored rendered tile files.
- Query UI should fetch results in batches and append more results on scroll.
- Query UI tile generation should be parallel-friendly for batches around 50
  results; the HTTP server can handle multiple tile requests concurrently, and
  generation should reuse cached pano arrays where practical.

## Proposed Flow

1. Downloader stores full panorama images durably.
2. Processing renders configured views into the configured view storage folder,
   stores view metadata and temporary image path in Postgres, and enqueues
   embedding jobs with that temporary path.
3. Embedding claims each view, reads the temporary image, embeds it, writes the
   vector to Qdrant, marks embedding status in Postgres, unlinks the temporary
   tile file, then clears `PanoramaView.image_path` only if it still points to
   that file.
4. Query UI searches Qdrant, hydrates results from Postgres, and returns JSON
   records with `view_id` rather than local tile paths.
5. Browser renders cards using `/tile?view_id=...`; the endpoint loads the
   stored panorama and renders the requested view using the same projection
   code used by processing.

## Storage Model

- `Panorama.image_path`: durable panorama file path.
- `PanoramaView.image_path`: temporary handoff tile path while waiting for
  embedding. It may be `NULL` after embedding.
- `PanoramaView.image_hash` and `image_bytes`: retained for the rendered tile.
- `PanoramaViewEmbedding.source_image_path`: retained historical path for the
  tile that was embedded, even after file deletion.
- Qdrant stores vectors and payload as before.

## Failure Handling

- If embedding succeeds, delete and clear the temp tile.
- If embedding fails after claiming a tile, delete and clear the temp tile. The
  failed view can be reprocessed by explicitly resetting statuses later.
- If deletion fails, log a warning but keep embedding status accurate.
- Duplicate embedding queue messages should be acknowledged and should attempt
  to delete the message's temp path if present.

## Non-Goals

- Do not delete downloaded panorama files.
- Do not add object storage.
- Do not build the final public website.
- Do not redesign the embedding model or Qdrant schema.
