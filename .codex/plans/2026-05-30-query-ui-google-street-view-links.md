# Query UI Google Street View Links Plan

## Goal

Make local query UI result tiles open the corresponding Google Maps Street View
panorama at the same view direction as the generated tile.

## Context

- Viewsets store pano-relative heading, pitch, and horizontal FOV.
- Downloaded panorama metadata stores a north offset / pose heading in
  `panorama_table.metadata_json["heading"]`.
- Google Maps URLs use north-based `heading`, plus `pitch` and `fov`.
- The official Maps URL Street View mode supports `api=1`,
  `map_action=pano`, `pano`, `viewpoint`, `heading`, `pitch`, and `fov`.

## Approach

- Extend query UI result metadata with latitude, longitude, and pano north
  heading.
- Convert pano-relative heading to Google north-based heading:
  `google_heading = (pano_heading + relative_heading) % 360`.
- Generate a Google Maps Street View URL for every result with a pano id.
- Wrap the result image in an external link and add a small explicit
  "Open in Google Maps" link in metadata.

## Verification

- Unit-test URL generation and HTML rendering.
- Run focused query UI tests.
- Run the main service test suite if the focused tests pass.
