# SigNoz Observability

## Problem

The pipeline is now dockerized and can run for long periods, but operators need
visibility into what it is doing without reading raw terminal logs for every
service. We need traces for slow paths, metrics for progress and queue depth,
logs in one place, and persistent local runtime data on disk.

## Decision

SigNoz itself runs outside this repo with the official SigNoz Docker Compose
stack. This repo owns only the application-side observability:

- OpenTelemetry instrumentation in Python services;
- optional OTLP export to the external SigNoz collector;
- a lightweight monitoring reporter service for database totals and NATS queue
  depths;
- Docker Compose wiring to point app containers at SigNoz;
- documentation for running the combined stack.

This avoids vendoring SigNoz's ClickHouse, migrations, collector config, and
version-specific deployment details into this public repo.

## Requirements

- Observability must be optional and disabled by default for local tests.
- When enabled, services export traces, metrics, and logs over OTLP to a
  configured endpoint.
- Docker containers should work with a SigNoz stack running separately on the
  same host through published OTLP ports.
- App data remains persisted through bind mounts:
  - Postgres data under `.local/postgres-data`;
  - NATS JetStream state in the existing Docker volume;
  - downloaded panos, generated views, HNSW indexes, and monitoring outputs
    under `services/main/.local`.
- Monitoring should report:
  - total map tiles by discovery status;
  - total panos by download status;
  - total generated views by processing status;
  - total embeddings by embedding status;
  - NATS download, processing, and embedding queue pending counts;
  - recent throughput counters from pipeline events where available.
- Processing and embedding progress callbacks should record event counters and
  durations for matching start/complete or start/failure pairs.
- Logs should continue going to stderr and should also be exportable to SigNoz
  when observability is enabled.
- Do not commit SigNoz credentials, private endpoints, or production hostnames.

## Non-Goals

- Do not copy the full SigNoz deployment into this repo.
- Do not build a permanent public monitoring website.
- Do not attempt to backfill historical telemetry.
- Do not add high-cardinality metric labels for pano IDs, file paths, or raw
  coordinates.
- Do not rely on SigNoz for durable pipeline state; Postgres and NATS remain the
  source of truth.

## Architecture

Add a `main_service.observability` package with:

- a no-op telemetry implementation used when observability is disabled or the
  OpenTelemetry packages are unavailable;
- an OpenTelemetry implementation that configures trace, metric, and log
  exporters from settings;
- a progress reporter that maps structured pipeline events to counters and
  duration histograms.

Add a `main_service.monitoring` package with a long-running service that:

1. queries Postgres status totals;
2. reads NATS pending queue depths through existing queue helpers;
3. writes gauges to the active telemetry recorder;
4. logs each snapshot as structured JSON.

Docker Compose keeps the base stack runnable without SigNoz. A separate
`docker-compose.observability.yml` override enables observability, sets OTLP
defaults for a separately running SigNoz collector, and adds host gateway access
for Linux Docker hosts.

## Data Flow

```text
pipeline services -> OpenTelemetry SDK -> OTLP gRPC -> external SigNoz collector
monitor service -> Postgres/NATS snapshots -> OpenTelemetry metrics/logs -> SigNoz
Docker stdout/stderr -> normal docker logs, plus optional OTLP log exporter
```

## Coverage Map

SigNoz is the monitoring surface for counts, timings, logs, and queue state. It
is not the primary geospatial UI. For now, coverage map readiness means the
monitoring service can emit bounded coverage metadata and keep enough location
data in Postgres for a map view. A richer map should be a separate small UI or a
query UI extension once the monitoring basics are stable.

## Failure Handling

- If observability is disabled, all instrumentation calls are no-ops.
- If OpenTelemetry dependencies are missing, service startup logs a warning and
  continues without telemetry.
- Exporter failures must not fail pipeline jobs.
- Monitor snapshot failures are logged and retried on the next interval.
- Queue depth errors for one queue should not prevent DB status metrics from
  being emitted.

## Verification

- Unit-test observability config defaults.
- Unit-test progress event duration tracking with a fake clock and recorder.
- Unit-test DB/NATS snapshot aggregation with SQLite and fake queues.
- Unit-test monitor CLI parser.
- Run focused tests for observability and monitoring.
- Run the full main-service test suite.
- Validate base, GPU, and observability Docker Compose configs.
