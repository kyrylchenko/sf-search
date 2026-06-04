# Query UI Image Search Spec

## Problem

Text search is useful for semantic lookup, but visual exploration often starts
from an example image. The local query UI should support searching with an
uploaded image and quickly pivoting from any result tile into visually similar
tiles.

## Requirements

- Add an `Image Search` tab to the local query UI.
- Accept image input through paste, drag/drop, and file picker.
- Show a small preview of the selected/source image while searching and while
  results are displayed.
- Search image queries through the same embedding model and vector store used
  by text query.
- Add a `Similar` action to every result tile. Clicking it starts image search
  using that tile as the query image.
- Tile-based image searches must be reloadable and browser-back friendly through
  `/image?view_id=<id>`.
- Uploaded images are temporary runtime inputs. Do not persist them in the repo,
  database, or durable local storage.

## Proposed Architecture

- Extend `LocalQueryService` with `search_image_path(...)`.
- Add `/api/image-search`:
  - `GET /api/image-search?view_id=<id>` renders that view from the source pano,
    embeds the rendered tile, searches the vector store, and returns hydrated
    results.
  - `POST /api/image-search` accepts multipart image bytes, validates and
    normalizes the image into a temporary JPEG, embeds it, searches, and deletes
    the temp file.
- Keep result hydration and result card metadata shared with text search.
- Add an image tab HTML/JS shell with upload dropzone, preview, loading state,
  result grid, infinite scroll, and a `Load more` fallback.
- Existing result JSON gets a `similar_url` field so both text and image results
  can pivot into image search.

## Failure Handling

- Invalid upload/multipart/image data returns `400`.
- Missing source `view_id` returns `404`.
- Search/vector/model errors return `500` and are logged.
- Uploaded temp files are deleted in a `finally` path.

## Verification

- Unit tests cover image tab markup, multipart image extraction, image-search
  service path, and result JSON `similar_url`.
- Full `services/main` pytest suite must pass.
