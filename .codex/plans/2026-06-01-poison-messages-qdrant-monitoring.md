# Poison Messages And Qdrant Monitoring Implementation Plan

**Goal:** Harden the long-running pipeline so malformed NATS messages do not crash workers and monitoring snapshots include Qdrant collection health.

**Non-goals:** Do not change NATS ack timing, worker retry semantics, Docker image pinning, Qdrant indexing settings, or upstream/downstream enqueue failure behavior.

**Files likely to change:**

- `services/main/main_service/downloader/nats_source.py`
- `services/main/main_service/processing/nats_source.py`
- `services/main/main_service/embedding/nats_source.py`
- `services/main/main_service/monitoring/snapshot.py`
- `services/main/main_service/monitoring/__main__.py`
- NATS source tests under `services/main/tests/*/test_*nats_source.py`
- Monitoring tests under `services/main/tests/monitoring/`

**Approach:**

1. Add tests proving each NATS source skips and acknowledges invalid messages while returning valid messages from the same fetch.
2. Implement per-message decode/parse guards in each NATS source. Invalid payloads are logged and acknowledged so they cannot redeliver forever.
3. Add tests proving monitoring snapshots include Qdrant status when the collection can be read and keep working when Qdrant is unavailable.
4. Add a small Qdrant snapshot source protocol and adapter. Monitoring should include `qdrant` data and `qdrant_errors` without requiring live Qdrant in unit tests.
5. Emit numeric Qdrant fields as telemetry gauges.

**Why this approach:**

- It keeps worker behavior simple and local to queue intake.
- It avoids introducing a dead-letter stream before we need one.
- It keeps Qdrant monitoring optional and failure-isolated so broken monitoring cannot stop the pipeline.

**Verification commands:**

```bash
uv run pytest \
  tests/downloader/test_nats_source.py \
  tests/processing/test_processing_nats_source.py \
  tests/embedding/test_embedding_nats_source.py \
  tests/monitoring/test_snapshot.py \
  tests/monitoring/test_monitoring_entrypoint.py -q
```

```bash
uv run pytest tests/test_imports.py tests/test_config.py -q
```

**Risks and open questions:**

- Acknowledging malformed messages discards them. That is intentional because these jobs cannot be parsed or processed.
- Qdrant client collection metadata can vary across client versions, so the adapter should read fields defensively.
