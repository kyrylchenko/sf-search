# Viewset Visualizer Spec

## Problem

Before generating millions of CLIP input views, we need a local visual tool to
inspect proposed viewsets against an actual downloaded panorama. The user needs
to load a pano image, switch between JSON viewset presets, and see which parts
of the equirectangular pano each view covers.

## Requirements

- Provide a local Python tool under `services/main`.
- Accept:
  - path to a downloaded panorama image or a directory of panorama images,
  - path to a folder of `.json` viewset definitions,
  - optional host/port.
- Serve a small browser UI.
- UI must show the panorama image and outline-only overlays for each view in
  the selected viewset. Filled translucent regions are intentionally avoided
  because they hide pano detail when many views are visible.
- UI must list viewsets and view metadata (`relative_heading`, `pitch`, `fov`,
  `view_kind`).
- UI must expose compact per-view toggles as a spatial matrix. The matrix order
  must be derived from Python-computed overlay polygons, not raw heading/pitch
  values, so a checkbox in the top/bottom/left/right area corresponds to the
  approximate same area on the displayed pano.
- Opening a selected view must call the same perspective rendering function that
  processing will use for embedding images.
- Pano canvas clicks must not open views.
- Matrix checkboxes only toggle overlay visibility. They must not have hover
  preview, modifier-key opening, or Open mode behavior.
- Opening individual rendered views is done from the selected-view panel: the
  direct "Open selected view" button, or a fuzzy search box whose results can be
  clicked to open a matching view. Pressing Enter in the search should open the
  top match.
- When `--pano` points to a directory, the UI must expose Previous/Next pano
  controls. The selected pano index must be included in `/api/state`, `/pano`,
  `/view`, and `/api/view-image` links so the canvas image and opened rendered
  views always refer to the same pano.
- When `--pano` points to a single image inside a directory that contains other
  supported pano images, the visualizer should use that directory as the gallery
  and start on the requested image. This keeps Previous/Next useful when the
  user launches with a familiar single-file command.
- View specs must be compatible with later Google Maps Embed usage:
  - `relative_heading` canonical in pano space, normalized to `[0, 360)`;
  - `pitch` clamped/validated against `[-90, 90]`;
  - `fov` clamped/validated against `[10, 100]`.
- Current committed sample presets:
  - `center-no-sky-road`: one center-context view at `fov=91`;
  - `small-object-grid-72`: 72 small-object views at `fov=40`;
  - `v1-wide-center`: six wide-context views at `fov=100` plus one
    center-context view at `fov=77`;
  - `wide-triptych-front-band`: three front-band wide-context views at
    `fov=100`.
- Avoid geometry drift between Python and browser. Python must compute all pano
  to equirectangular 2D overlay coordinates; browser only scales normalized
  coordinates to canvas pixels.

## Viewset JSON Shape

```json
{
  "name": "candidate-v1",
  "description": "Optional human-readable note",
  "views": [
    {
      "id": "wide-000",
      "relative_heading": 0,
      "pitch": 0,
      "fov": 95,
      "view_kind": "wide_context"
    }
  ]
}
```

## Geometry

Overlays must use accurate perspective-frustum boundaries projected onto the
equirectangular pano. Do not approximate the view as a rectangular pano crop.

Python owns this projection:

- sample the perspective view frustum edges;
- use the same `py360convert` coordinate model used for equirectangular to
  perspective conversion;
- convert sampled directions to equirectangular normalized coordinates;
- unwrap seam-crossing polygons around the view center;
- return duplicate shifted polygons where needed so the browser can draw and
  clip without knowing any pano geometry.

The browser must not reimplement pano math. It only receives normalized
polygons and scales them to the displayed image.

## Non-Goals

- Do not generate embeddings.
- Do not persist processing DB rows yet.
- Do not build a production frontend.
- Do not commit downloaded pano images or generated private artifacts.

## Verification

- Unit-test viewset parsing, validation, normalization, and wrap splitting.
- Unit-test API response shape from the local server app.
- Run the visualizer against a locally downloaded pano when available.
