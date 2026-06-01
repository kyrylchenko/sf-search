# Boundary Pano Count Tool Plan

**Goal:** Add a one-off script to estimate how many unique Street View pano IDs
exist inside the current target boundary.

**Non-goals:** Do not write to Postgres, NATS, Qdrant, local pano storage, or
commit any generated output from the script.

**Files likely to change:**

- `services/main/main_service/tools/count_boundary_panos.py`
- `services/main/tests/tools/test_count_boundary_panos.py`

**Approach:**

1. Reuse `load_map_tiles_from_geojson` so the count uses the same boundary to
   map tile conversion as discovery.
2. Reuse `StreetLevelCoverageClient` and `pano_ids_from_coverage_objects` so the
   script reads coverage the same way discovery does.
3. Use bounded asyncio concurrency around the sync StreetLevel coverage call via
   `asyncio.to_thread`.
4. Return a JSON summary with tile count, scanned tiles, unique pano count,
   non-unique observations, failed tiles, and optional sampled IDs.
5. Keep the CLI safe by default: no DB/queue side effects and no committed
   runtime output.

**Verification commands:**

```bash
uv run pytest tests/tools/test_count_boundary_panos.py
uv run pytest
```

**Run command:**

```bash
uv run python -m main_service.tools.count_boundary_panos --geojson target.geojson --zoom 17 --concurrency 20
```
