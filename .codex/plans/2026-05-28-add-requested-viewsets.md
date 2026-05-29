# Add Requested Viewsets Plan

## Goal

Add separate JSON viewset presets matching the user-requested visualizer options:

- wide left/center/right context views;
- one center crop that avoids most sky/road;
- an around-70-view small-object grid.

## Notes

Google Maps Embed Street View accepts `fov` only up to `100`, so three wide
views cannot cover the full 360 degrees with 50% overlap. The wide triptych
preset is therefore a front-band left/center/right set. Full 360 coverage stays
in the existing combined preset and denser grid presets.

Later adjustment:

- `center-no-sky-road` increased from `fov=65` to `fov=91`, a 40% larger
  coverage area while staying under the Google Embed cap.
- The old combined `v1-wide-center-small-grid` preset was converted to
  `v1-wide-center`: six wide context views plus one center view only. Small
  object views remain in `small-object-grid-72`.
- The wide views increased from `fov=95` to the Google Embed cap of `100`
  instead of `104.5`; the center view increased from `fov=70` to `77`.

## Verification

- Add tests that sample viewset files load successfully.
- Run visualizer tests and full test suite.
