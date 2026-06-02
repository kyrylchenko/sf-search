# SigNoz Dashboard

`sf-search-pipeline-dashboard.json` is the baseline importable SigNoz dashboard
for the local/production ingestion pipeline.

`sf-search-pipeline-expanded-dashboard.json` keeps the baseline panels and adds
extra queue, Qdrant, log, GPU, query UI, and detailed pipeline timing panels.
It intentionally includes more widgets than most runs need so operators can
delete noisy panels after import.

The dashboard expects OpenTelemetry to be enabled and all long-running services
to export to the same SigNoz collector. The DB, NATS, coverage, and Qdrant
panels are emitted by the monitoring service, so those panels will be empty or
stale unless that service is running.

Useful services for the dashboard:

- `sf-search-discovery`
- `sf-search-downloader`
- `sf-search-processing`
- `sf-search-embedding`
- `sf-search-monitoring`
- `sf-search-query-ui`

Trace spans are nested so a high-level operation can be expanded into its
internal work. The intended shape is:

```text
*.batch
  *.fetch
  *.job
    *.claim
    *.resolve / *.download / *.source_load / *.render / *.model_encode
    *.db_update
    *.enqueue_processing / *.enqueue_embedding / *.qdrant_upsert
    *.ack
```

Latency panels use `sf_search_pipeline_duration_seconds_bucket` and the
`operation` label. Trace drilldowns should use the dotted span names such as
`embedding.job`, `embedding.model_encode`, and `embedding.qdrant_upsert`.

Embedding throughput panels use
`sf_search_pipeline_events_total{event="embedding_job_complete"}`. This is the
accurate tile completion signal because each completed embedding job is one
panorama view/tile. The model encode latency panels use
`operation="embedding_model_encode"`; in batched embedding mode that timing is
for the whole batch. The per-tile latency panels use
`operation="embedding_job"`, which is emitted once per tile in both single-image
and batched embedding modes.

No private SigNoz URL, API key, or host-specific value belongs in this file or
the dashboard JSON.
