# main service

Run a bounded panorama downloader batch from this directory:

```bash
uv run python -m main_service.downloader --limit 5
```

The downloader consumes durable NATS JetStream jobs from
`pano.download.requested`, deduplicates through Postgres, writes images under
`.local/panoramas`, and publishes small processing jobs to
`pano.processing.requested`.

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

Click a view in the sidebar, or click an overlay on the pano, to open the
server-rendered 2D perspective view in a new browser tab.
