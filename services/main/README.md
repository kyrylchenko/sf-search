# main service

Run a bounded panorama downloader batch from this directory:

```bash
uv run python -m main_service.downloader --limit 5
```

The downloader consumes durable NATS JetStream jobs from
`pano.download.requested`, deduplicates through Postgres, writes images under
`.local/panoramas`, and publishes small processing jobs to
`pano.processing.requested`.
