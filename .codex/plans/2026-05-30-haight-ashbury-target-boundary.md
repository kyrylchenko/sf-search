# Haight-Ashbury Target Boundary Plan

## Goal

Change the local discovery target from the current East Bay rectangle to a small
San Francisco Haight-Ashbury area so discovery/download tests run against the
intended city context.

## Approach

- Update `services/main/target.geojson` to an approximate Haight-Ashbury
  rectangle around Haight Street and Ashbury Street.
- Keep this as a small local smoke/test boundary, not a final San Francisco
  citywide boundary.
- Compute zoom-17 tile count after the change.

## Verification

- Load the GeoJSON through the existing boundary loader.
- Report the generated tile count and tile coordinate range.

## Result

- `services/main/target.geojson` now points at an approximate Haight-Ashbury
  rectangle around Haight Street and Ashbury Street.
- The target generates 40 zoom-17 tiles.
- Tile range: `x=20950..20957`, `y=50662..50666`, `z=17`.
