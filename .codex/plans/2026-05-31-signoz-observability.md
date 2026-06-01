# SigNoz Observability Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add optional OpenTelemetry/SigNoz observability and a dockerized monitor service for pipeline progress, timings, logs, and queue state.

**Architecture:** SigNoz runs separately from the official SigNoz Compose stack. This repo exports OTLP telemetry when enabled, keeps normal local execution no-op by default, and adds a monitor worker that periodically emits Postgres and NATS state.

**Tech Stack:** Python 3.14, OpenTelemetry Python SDK/exporters, SQLAlchemy, NATS JetStream helpers, Docker Compose, SigNoz OTLP gRPC endpoint.

---

### Task 1: Observability Config And Dependencies

**Files:**
- Modify: `services/main/pyproject.toml`
- Modify: `services/main/main_service/config.py`
- Modify: `.env.example`
- Modify: `services/main/.env.example`
- Test: `services/main/tests/test_config.py`
- Test: `services/main/tests/embedding/test_embedding_config.py`

- [ ] Add optional runtime settings with safe defaults:
  - `observability_enabled: bool = False`
  - `otel_exporter_otlp_endpoint: str = "http://localhost:4317"`
  - `otel_exporter_otlp_insecure: bool = True`
  - `otel_exporter_otlp_timeout_seconds: float = 10.0`
  - `otel_metric_export_interval_millis: int = 10000`
  - `otel_service_version: str = "local"`
  - `deployment_environment: str = "local"`
  - `monitoring_interval_seconds: float = 15.0`
- [ ] Add OpenTelemetry packages to `services/main/pyproject.toml` regular dependencies:
  - `opentelemetry-api`
  - `opentelemetry-sdk`
  - `opentelemetry-exporter-otlp-proto-grpc`
- [ ] Update env examples with placeholder endpoint values only.
- [ ] Update config tests to assert defaults.

### Task 2: Telemetry Recorder And Progress Timing

**Files:**
- Create: `services/main/main_service/observability/__init__.py`
- Create: `services/main/main_service/observability/telemetry.py`
- Create: `services/main/main_service/observability/progress.py`
- Test: `services/main/tests/observability/test_progress.py`

- [ ] Write tests for `TimedProgressReporter`:
  - it records every event;
  - it records a duration when `processing_job_start` is followed by `processing_job_complete`;
  - it is safe when a complete event has no matching start.
- [ ] Implement `NoopTelemetry` with `record_event`, `record_duration`, `set_gauge`, `span`, `flush`, and `shutdown` no-op methods.
- [ ] Implement `OpenTelemetryRecorder` behind optional imports. If imports fail or config disables observability, return `NoopTelemetry`.
- [ ] Implement `TimedProgressReporter` using a lock and a monotonic clock.
- [ ] Keep identifiers out of metric attributes except low-cardinality fields such as `service`, `event`, `status`, `queue`, and `table`.

### Task 3: Wire Telemetry Into Services

**Files:**
- Modify: `services/main/main_service/discovery/__main__.py`
- Modify: `services/main/main_service/downloader/__main__.py`
- Modify: `services/main/main_service/processing/__main__.py`
- Modify: `services/main/main_service/embedding/__main__.py`
- Modify: `services/main/main_service/embedding/query_ui.py`
- Test: existing entrypoint parser/config tests

- [ ] Configure telemetry once per CLI after logging is configured.
- [ ] Wrap each `run_batch` call in a service-level span.
- [ ] Pass `TimedProgressReporter` as the progress callback to processing and embedding runners.
- [ ] Record downloader and discovery batch result events from their CLI wrappers.
- [ ] Call `shutdown()` in `finally` blocks so batch processors flush on normal shutdown.

### Task 4: Monitoring Snapshot Service

**Files:**
- Create: `services/main/main_service/monitoring/__init__.py`
- Create: `services/main/main_service/monitoring/snapshot.py`
- Create: `services/main/main_service/monitoring/__main__.py`
- Test: `services/main/tests/monitoring/test_snapshot.py`
- Test: `services/main/tests/monitoring/test_monitoring_entrypoint.py`

- [ ] Write SQLite tests that seed map tile, panorama, view, and embedding rows and assert grouped counts.
- [ ] Write fake queue tests that assert pending counts are included even if one queue raises.
- [ ] Implement `build_pipeline_snapshot(engine, queues)` returning a dataclass with `status_counts`, `queue_depths`, and `coverage`.
- [ ] Implement monitor CLI parser with `--interval-seconds`, `--once`, and `--log-level`.
- [ ] Implement long-running monitor loop that logs snapshots and sets telemetry gauges.

### Task 5: Docker Compose And Docs

**Files:**
- Modify: `docker-compose.yml`
- Create: `docker-compose.observability.yml`
- Modify: `docs/server-runbook.md`
- Test: Docker Compose config validation

- [ ] Add a `monitoring` service to the `pipeline` profile.
- [ ] Add observability environment placeholders to the main service anchor.
- [ ] Add `docker-compose.observability.yml` that sets `OBSERVABILITY_ENABLED=true`, OTLP endpoint defaults, and host gateway access.
- [ ] Document how to start SigNoz separately, then run:
  - `docker compose -f docker-compose.yml -f docker-compose.observability.yml --profile pipeline up -d --build`
- [ ] Document that coverage map data remains in Postgres for now and map UI work is separate.

### Task 6: Verification And Commit

**Files:**
- All files from previous tasks.

- [ ] Run focused observability/monitoring tests.
- [ ] Run full `uv run pytest -q` from `services/main`.
- [ ] Run `git diff --check`.
- [ ] Run `docker compose config --quiet`.
- [ ] Run `docker compose -f docker-compose.yml -f docker-compose.gpu.yml config --quiet`.
- [ ] Run `docker compose -f docker-compose.yml -f docker-compose.observability.yml config --quiet`.
- [ ] Commit with `feat: add signoz observability wiring`.
