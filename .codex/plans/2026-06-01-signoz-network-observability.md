# SigNoz Network Observability

## Goal

Make sf-search containers export OTLP logs, metrics, and traces to the local SigNoz stack reliably under rootless Docker.

## Non-goals

- Do not vendor or embed SigNoz in the sf-search Compose stack.
- Do not expose additional public ports beyond the SigNoz stack's existing `8080`, `4317`, and `4318` mappings.
- Do not change application telemetry instrumentation.

## Context

The observability override originally targets `host.docker.internal:4317`. In this rootless Docker deployment, sf-search containers still receive `StatusCode.UNAVAILABLE` when exporting to that host-published port. SigNoz already runs on Docker network `signoz-net`, so a direct container-to-container route is safer.

## Approach

1. Add external Docker network `signoz-net` to the observability override.
2. Attach sf-search app services to both their default project network and `signoz-net`.
3. Point the local deployment OTLP endpoint at `http://signoz-otel-collector:4317`.
4. Recreate the sf-search app containers and verify exporter errors stop.

## Verification

- `docker network inspect signoz-net`
- Running app container env contains `OTEL_EXPORTER_OTLP_ENDPOINT=http://signoz-otel-collector:4317`.
- Fresh logs no longer show OTLP `StatusCode.UNAVAILABLE`.
- SigNoz UI on `http://localhost:8080` remains healthy.
