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
- Overlay polygons must use the same relative-heading convention as the
  processing renderer. A view highlighted in the matrix/canvas and the same
  view opened through search must point at the same pano region.
- Pano canvas clicks must not open views.
- Matrix checkboxes only toggle overlay visibility. They must not have hover
  preview, modifier-key opening, or Open mode behavior.
- The matrix must stay above the selected-view open/search panel. The search
  UI can grow with results, so it should not push the matrix down the sidebar.
- Opening individual rendered views is done from the selected-view panel: the
  direct "Open selected view" button, or a fuzzy search box whose results can be
  clicked to open a matching view. Pressing Enter in the search should open the
  top match.
- Local rendered view pages must show a high-resolution preview by default,
  rendered through the same perspective function but with larger output
  dimensions so humans can inspect source detail.
- The high-resolution preview should fit inside the browser viewport without
  showing a second lower-resolution processing image below it.
- Rendered local view images should be served losslessly for inspection. JPEG
  re-encoding can make already-downsampled signs and fine text look worse than
  the actual projection.
- Local perspective rendering should not use direct bilinear sampling as the
  default quality path. Visual comparison showed bilinear as the weakest option
  for small signs and other fine detail, while bicubic/supersampled outputs did
  not show obvious projection corruption.
- Opened high-resolution preview pages must scale the rendered image down to
  fit the browser viewport. Large `scale=4` renders must not be clipped by the
  viewport or hidden behind the page bounds.
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
- `fov` is horizontal FOV. Rendered perspective images must derive vertical
  FOV from `output_width`/`output_height`; passing scalar FOV to projection
  libraries is incorrect for non-square outputs and causes visual drift from
  overlays and Google Embed.
- Current committed sample presets:
  - `center-no-sky-road`: one center-context view at `fov=91`;
  - `small-object-grid-72`: 72 small-object views at `fov=40`;
  - `v1-wide-center`: six wide-context views at `fov=100` plus one
    center-context view at `fov=77`, all pitched 10 degrees up;
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
