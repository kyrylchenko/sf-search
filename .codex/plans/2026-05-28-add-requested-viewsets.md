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

## Verification

- Add tests that sample viewset files load successfully.
- Run visualizer tests and full test suite.
