# Update Target Boundary Plan

**Goal:** Switch the checked-in discovery target to the user-provided San
Francisco coordinate ring.

**Non-goals:** Do not change discovery logic, queue state, database state, or
generated runtime data.

**Files likely to change:**

- `services/main/target.geojson`
- tests that assert the target boundary's expected tile coverage

**Approach:**

1. Replace the current Haight-Ashbury smoke boundary with the provided
   coordinates.
2. Store the ring as a GeoJSON `Polygon`, not a `LineString`, because discovery
   needs filled area coverage. A `LineString` would only intersect tiles along
   the boundary path.
3. Recalculate the zoom-17 tile coverage expectations.
4. Run the relevant boundary tests and then the full service test suite.

**Verification commands:**

```bash
uv run pytest tests/test_geo.py tests/ingestion/test_boundary_loader.py
uv run pytest
```
