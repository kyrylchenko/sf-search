# Viewset Visualizer Implementation Plan

## Goal

Add a local Python/browser visualizer for comparing JSON viewset presets on top
of an actual downloaded equirectangular panorama.

## Files

- Create `services/main/main_service/tools/__init__.py`
- Create `services/main/main_service/tools/viewset_visualizer/__init__.py`
- Create `services/main/main_service/tools/viewset_visualizer/geometry.py`
- Create `services/main/main_service/tools/viewset_visualizer/viewsets.py`
- Create `services/main/main_service/tools/viewset_visualizer/server.py`
- Create `services/main/main_service/tools/viewset_visualizer/__main__.py`
- Create static UI files under
  `services/main/main_service/tools/viewset_visualizer/static/`
- Create tests under `services/main/tests/tools/viewset_visualizer/`
- Add sample public-safe viewsets under `docs/data/viewsets/`
- Update `README.md` or `services/main/README.md`

## Steps

- [x] Write geometry tests for heading/pitch/fov validation and seam wrapping.
- [x] Implement geometry helpers.
- [x] Write viewset parsing tests.
- [x] Implement JSON loader and validation.
- [x] Write server API tests.
- [x] Implement local HTTP server API and static serving.
- [x] Add browser UI with canvas overlays and viewset selector.
- [x] Add sample viewset JSON.
- [x] Document run command.
- [x] Run tests and smoke check server startup.
- [x] Commit changes.
- [x] Replace noisy filled overlays with outline-only projected boundaries.
- [x] Add requested viewset presets for wide, center, and small-object grids.
- [x] Use shared processing perspective rendering when opening an individual
  view.
- [x] Replace the long per-view toggle list with a compact spatial matrix
  ordered by projected polygon centers.
- [x] Remove unreliable pano-click opening.
- [x] Remove matrix hover preview and Open mode after browser behavior proved
  unreliable.
- [x] Keep matrix checkboxes scoped to overlay visibility toggling only.
- [x] Add selected-view fuzzy search for opening rendered views by id, kind,
  heading, pitch, or fov.
- [x] Add multi-pano directory support for the visualizer server.
- [x] Add Previous/Next pano controls that keep opened rendered views pinned to
  the selected pano index.
- [x] Keep the matrix above the selected-view open/search panel so search
  results cannot push tile toggles too far down the sidebar.
- [x] Treat a single pano file path as a gallery rooted at its parent directory
  when sibling pano images exist, starting on the requested file.
- [x] Keep local rendered view pages from stretching processing-sized tiles
  directly to the viewport.
- [x] Align overlay heading projection with the processing perspective renderer
  so canvas/matrix selection and opened view images reference the same region.
- [x] Add high-resolution local preview rendering for opened views.
- [x] Remove the processing-tile image from the opened-view UI and fit the
  high-resolution preview inside the browser viewport.
- [x] Treat view `fov` as horizontal in the processing renderer and derive
  vertical FOV from output aspect ratio to match overlays and Google Embed.
- [x] Move the `v1-wide-center` preset roughly 10 degrees upward to reduce
  road-heavy framing.
- [x] Verify matrix rendering after the duplicate stale JavaScript block was
  removed.
- [x] Commit the matrix fix.
- [x] Switch local perspective rendering away from direct bilinear sampling
  after visual inspection showed bilinear is the weakest option for signs and
  other fine detail.
- [x] Make opened high-resolution previews fit the browser viewport without
  cropping large rendered images.
- [x] Re-run visualizer tests and full service tests.
- [x] Commit the rendering-quality and viewport-fit fix.

## Verification Commands

```bash
cd services/main
uv run pytest tests/tools/viewset_visualizer -q
uv run pytest -q
python3 -m xml.etree.ElementTree ../docs/diagrams/current-ingestion-flow.svg
```

Manual smoke command:

```bash
cd services/main
uv run python -m main_service.tools.viewset_visualizer \
  --pano .local/panoramas/example.jpg \
  --viewsets ../../docs/data/viewsets
```
