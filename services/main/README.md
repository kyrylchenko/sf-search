# main service

Run a bounded panorama downloader batch from this directory:

```bash
uv run python -m main_service.downloader --limit 5
```

The downloader consumes durable NATS JetStream jobs from
`pano.download.requested`, deduplicates through Postgres, writes images under
`.local/panoramas`, and publishes small processing jobs to
`pano.processing.requested`.

Run a bounded panorama preprocessing batch:

```bash
uv run python -m main_service.processing --limit 1
```

The preprocessor consumes durable NATS JetStream jobs from
`pano.processing.requested`, loads viewsets from `../../docs/data/viewsets` by
default, renders perspective views through the shared processing renderer, saves
generated images under `.local/panorama-views`, and records each generated view
in `panorama_view_table`.

The preprocessor also publishes small embedding jobs to
`pano.embedding.requested`. It checks the embedding queue before pulling another
panorama; by default it pauses at 100 queued view jobs, but it can finish and
enqueue all views for the current panorama.

Useful local options:

```bash
uv run python -m main_service.processing \
  --limit 1 \
  --concurrency 1 \
  --render-scale 2 \
  --viewsets-dir ../../docs/data/viewsets \
  --storage-dir .local/panorama-views
```

Run a bounded panorama view embedding batch:

```bash
uv sync --group embedding
uv run python -m main_service.embedding --limit 10
```

The embedder consumes durable NATS JetStream jobs from
`pano.embedding.requested`, claims model-specific rows in
`panorama_view_embedding_table`, embeds local generated view images, and writes
vectors to a local HNSW index under `.local/embedding-indexes`.

The default model settings target `google/siglip2-so400m-patch14-384`:

```text
EMBEDDING_MODEL_PROVIDER=transformers
EMBEDDING_MODEL_ID=google/siglip2-so400m-patch14-384
EMBEDDING_PREPROCESS_VERSION=siglip2-384-rgb-v1
EMBEDDING_DIMENSION=1152
EMBEDDING_DTYPE=float16
```

Run the local test-only query UI:

```bash
uv sync --group embedding
uv run python -m main_service.embedding.query_ui
```

Open `http://127.0.0.1:8787`. The UI embeds the text query with the same local
model, searches the local HNSW index, and displays locally stored generated view
images with pano/view/model metadata. This is not the future public website.

## Viewset Visualizer

Run the local browser visualizer against a downloaded pano:

```bash
uv run python -m main_service.tools.viewset_visualizer \
  --pano .local/panoramas/example-pano-id.jpg \
  --viewsets ../../docs/data/viewsets
```

Open `http://127.0.0.1:8765`.

The visualizer computes accurate sampled perspective-frustum boundaries in
Python using the same equirectangular coordinate model as `py360convert`. The
browser only draws the returned normalized polygons, so the UI does not carry a
separate pano-projection implementation.

Use `Show all`, `Hide all`, or the compact pitch/heading checkbox matrix to
control overlay noise. Large presets default to hidden overlays so individual
views can be inspected without covering the whole pano.

Select a view in the sidebar, then use `Open selected view` or the search box to
open the server-rendered 2D perspective view in a new browser tab.

That view page can toggle between the local rendered perspective image and a
Google Maps Embed Street View iframe. Set the API key only in your local
environment:

```bash
export GOOGLE_MAPS_EMBED_API_KEY=YOUR_KEY
```

Google expects north-based headings. Viewsets store pano-relative headings; the
visualizer derives Google heading as:

```text
google_heading = (north_offset + relative_heading) % 360
```

By default `north_offset` is read from GPano XMP `PoseHeadingDegrees` in the
downloaded panorama. Override it with `--north-offset` when needed.
