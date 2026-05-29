# Random Pano Sampling Plan

## Goal

Create a local-only helper that downloads a small set of panoramas from random
coordinates so viewset coverage can be inspected across varied scenes instead
of adjacent panoramas from one coverage tile.

## Decisions

- Keep the tool under `services/main/main_service/tools/`.
- Default to an approximate San Francisco bounding box, with `--bbox` override.
- Save images and metadata under ignored `.local/random-panoramas` by default.
- Do not touch Postgres or NATS; this is a visual inspection helper, not the
  production ingestion flow.
- Dedupe by resolved Google pano ID so repeated nearby random points do not
  waste slots.
- Treat existing `*.jpg` files in the output directory as already-seen pano IDs
  so manually seeded visualizer samples are not overwritten by a later random
  run.

## Steps

- [x] Add random coordinate sampling and downloader tool.
- [x] Add unit tests for bounds parsing/sampling and dedupe/download behavior.
- [x] Download 15 random-location panos.
- [x] Verify the visualizer can browse the downloaded random pano directory.
- [x] Skip pano IDs that already exist in the output directory.
- [x] Commit changes.
