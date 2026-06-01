# SigNoz Dashboard JSON Plan

**Goal:** Add and troubleshoot an importable SigNoz dashboard JSON for the
`sf-search` pipeline.

**Non-goals:** Do not change SigNoz deployment, collector configuration, or add
private SigNoz URLs/API keys to the repo.

**Files likely to change:**

- `docs/signoz/sf-search-pipeline-dashboard.json`
- `docs/signoz/README.md`
- `services/main/main_service/observability/progress.py`
- `services/main/main_service/discovery/__main__.py`
- `services/main/main_service/downloader/__main__.py`
- `services/main/main_service/embedding/runner.py`
- `services/main/main_service/processing/runner.py`
- focused observability/runner tests

**Approach:**

1. Base the dashboard JSON structure on the public `SigNoz/dashboards` template format.
2. Use PromQL panels for OTel metrics emitted by this repo:
   - `sf_search_pipeline_duration_seconds`
   - `sf_search_pipeline_events_total`
   - `sf_search_nats_queue_pending`
   - `sf_search_status_rows`
   - `sf_search_coverage`
   - `sf_search_qdrant_collection`
3. Use ClickHouse SQL for log-level aggregation, because SigNoz stores logs in ClickHouse and exposes `severity_text`.
4. Include at least ten useful panels covering latency, throughput, queue pressure, failures, progress, Qdrant, and query UI behavior.
5. Validate JSON with `python3 -m json.tool`.
6. Troubleshoot imported dashboard panels against emitted metrics:
   - Fix label mismatches, especially Query UI duration labels.
   - Add missing explicit duration metrics for service batches that currently
     exist only as trace spans.
   - Make start/complete event identity stable enough for the progress reporter
     to record per-job/per-view durations.
7. Use focused tests before instrumentation changes so missing duration metrics
   are reproducible.

**Why this approach:**

- PromQL keeps the dashboard portable across SigNoz metric-storage details.
- ClickHouse SQL is the most direct way to group logs by severity.
- Keeping this as a dashboard artifact avoids coupling dashboard edits to runtime code.
- Trace spans and metric histograms are different telemetry signals. Dashboard
  latency panels that use `sf_search_pipeline_duration_seconds_bucket` need
  explicit histogram observations; spans alone are not enough.

**Verification commands:**

```bash
python3 -m json.tool docs/signoz/sf-search-pipeline-dashboard.json >/tmp/sf-search-pipeline-dashboard.validated.json
uv run pytest services/main/tests/observability/test_progress.py
```

**References checked:**

- `https://github.com/SigNoz/dashboards`
- `https://raw.githubusercontent.com/SigNoz/dashboards/main/mysql/mysql-otlp-v1.json`
- `https://signoz.io/docs/userguide/logs_clickhouse_queries/`
