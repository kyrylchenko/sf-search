# SigNoz Dashboard

`sf-search-pipeline-dashboard.json` is an importable SigNoz dashboard for the
local/production ingestion pipeline.

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

No private SigNoz URL, API key, or host-specific value belongs in this file or
the dashboard JSON.
