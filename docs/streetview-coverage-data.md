# Street View Coverage Tile Data

This fixture documents the shape returned by:

```python
streetlevel.streetview.get_coverage_tile(x, y)
```

The sample was captured from the repo's `services/main/target.geojson` boundary by
scanning sorted zoom-17 tiles until a non-empty coverage tile was found.

## Captured Fixture

- File: `docs/data/streetview-coverage-tile-example.json`
- Source tile: `x=21003`, `y=50609`, `z=17`
- Returned objects for that tile at capture time: `28`
- Sample objects committed: `5`

Observed scalar fields include:

- `id`
- `lat`
- `lon`
- `heading`
- `pitch`
- `roll`
- `elevation`
- `date`
- `upload_date`
- `is_third_party`
- `country_code`
- `source`

Many fields may be `null` in coverage-tile results. Treat coverage data as
partial discovery data only; later metadata/download workers can enrich panorama
records when they resolve a specific panorama ID.

## Queue Payload Rule

Downloader queues should carry only small identifiers and references:

```json
{
  "pano_id": "example-pano-id",
  "source": "coverage_discovery",
  "discovered_from_tile": {"x": 1, "y": 2, "z": 17}
}
```

Do not put panorama image bytes, generated view bytes, embeddings, or index
artifacts in NATS.

## Download Probe

See `docs/data/streetview-panorama-download-probe.json` for a live probe of the
StreetLevel Google Street View download path.

Observed behavior:

- Metadata resolution can succeed through `find_panorama_async(...)`.
- Coverage pano IDs may return `None` through `find_panorama_by_id_async(...)`,
  which matches the StreetLevel docs warning that Google pano IDs are not
  stable.
- `streetlevel==0.12.7` uses the newer
  `streetviewpixels-pa.googleapis.com/v1/tile` endpoint.
- In this environment, a plain `aiohttp.ClientSession` still gets HTTP `403`
  from Google tile image requests, while the same StreetLevel call succeeds when
  the session includes browser-like headers including
  `Referer: https://www.google.com/maps/`.
- Downloader code should use those headers for StreetLevel sessions, but still
  treat image download failure as normal retryable/failable work and record it
  in Postgres.
