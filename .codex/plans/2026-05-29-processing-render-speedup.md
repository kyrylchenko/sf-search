# Processing Render Speedup Plan

## Goal

Reduce panorama preprocessing time enough that a 16-core local machine can keep
embedding fed without unsafe memory growth.

## Evidence

- `py360convert` exposes a fast OpenCV-backed sampler when `cv2` is importable.
- The current local environment does not have `cv2`, so `py360convert` falls
  back to SciPy `map_coordinates`.
- A local benchmark against a full 8192x16384 Google panorama rendered one
  1024x1024 view in roughly 15-24 seconds with the fallback path.
- Current viewsets produce 83 views per pano, so the fallback renderer can spend
  tens of minutes on one pano.

## Approach

- Add `opencv-python-headless` as a runtime dependency so `py360convert` uses
  `cv2.remap`.
- Keep the existing bounded view concurrency cap; faster rendering still
  allocates native arrays and should not be allowed to run at unbounded
  concurrency.
- Add renderer introspection/logging so service logs make it clear whether the
  fast OpenCV path is active.
- Benchmark before/after on the same local panorama.

## Verification

- Verify `py360convert.utils.cv2` is available after dependency sync.
- Re-run the single-view benchmark. Result: same 1024x1024 local pano view went
  from roughly 15-24 seconds on the SciPy fallback to 0.14-0.34 seconds with the
  OpenCV sampler path.
- Run focused rendering/processing tests. Result: `14 passed`.
- Run the main service test suite. Result: `148 passed`.
