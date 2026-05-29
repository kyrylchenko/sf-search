# Panorama Preprocessing Spec

## Problem

Downloaded equirectangular panoramas are too large and too visually broad for a
CLIP-style image model. The next pipeline stage must consume downloaded pano
jobs, generate the approved perspective views, store those image files locally,
and persist enough metadata to understand exactly how every generated view was
created.

## Scope

This stage starts at `pano.processing.requested` and stops after local view
images plus database rows exist. It does not generate embeddings, build vector
indexes, or publish website/query data.

## Requirements

- Consume durable NATS JetStream jobs from `pano.processing.requested`.
- Parse jobs containing `pano_id` and `image_path`.
- Load viewset JSON from `docs/data/viewsets` by default.
- Render every view through `main_service.processing.view_rendering`, the same
  projection code used by the visualizer.
- Use the current quality path: bicubic projection, configurable `render_scale`,
  JPEG output at high quality by default.
- Store generated view images under ignored local storage by default.
- Persist every generated view in Postgres with rich provenance:
  - pano row id and original Google pano id;
  - viewset name, view id, view kind, description;
  - exact view spec JSON and a stable spec hash;
  - relative heading, pitch, horizontal FOV, base and rendered dimensions;
  - render scale, renderer version, interpolation mode;
  - source pano image path/hash and generated image path/hash/byte size;
  - generation status, attempt count, timestamps, and last error;
  - future embedding status fields.
- Make duplicate queue messages safe. If the same pano/view/spec/source hash is
  already complete with an image path and hash, skip rendering that view.
- Ack processing queue messages after the job has either produced rows or been
  recorded/skipped as failed work. Avoid poison messages that redeliver forever.
- Keep workers stateless and restartable. Durable queue state plus Postgres rows
  must be enough to recover.

## Data Model

Create a new `panorama_view_table` rather than reusing `tile_table`. The existing
`tile_table` requires `embedding_id`, has limited provenance, and represents an
older shape. Keeping a dedicated panorama-view table avoids a risky migration and
gives future embedding/indexing stages a stable input table.

Recommended uniqueness:

```text
panorama_id + viewset_name + view_id + view_spec_hash + render_scale + output_format
```

If viewset definitions change later while keeping the same `view_id`, a new
`view_spec_hash` produces distinct rows instead of silently overwriting old work.

## Storage

Default output directory:

```text
.local/panorama-views/<safe-pano-id>/<safe-viewset>/<safe-view-id>-<hash>-s<scale>.jpg
```

Generated images stay ignored and must not be committed.

## Failure Handling

- Missing downloaded panorama row: count the job as failed and ack it.
- Missing source image file: count the job as failed and ack it.
- Per-view render/save failure: mark that view row `failed`, increment attempts,
  store a truncated error, continue with remaining views when possible.
- Previously failed rows are retryable; claiming them sets status to
  `processing`.

## Alternatives Considered

- Reuse `tile_table`: rejected because it requires embedding rows and lacks
  viewset/source/render provenance.
- Store generated views in NATS: rejected because images are too large for a
  queue payload and should be referenced by path/blob key.
- Publish embedding jobs immediately: deferred until the embedding stage is
  planned. The table will include `embedding_status="pending"` so that stage can
  scan or enqueue from durable DB state.

## Verification

- Unit tests for NATS message parsing.
- DB tests for `panorama_view_table` metadata and idempotent claim/complete
  behavior.
- Runner tests for generating views, skipping duplicates, recording failures,
  and acknowledging jobs.
- Full service test suite.
