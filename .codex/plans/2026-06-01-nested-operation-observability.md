# Nested Operation Observability Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make SigNoz show parent/child traces and p50/p95/p99 metrics for each pipeline operation and its internal stages.

**Architecture:** Add one shared `observed_span(...)` helper that creates an OpenTelemetry span and records the same elapsed duration to `sf_search_pipeline_duration_seconds`. Wire it through downloader, processing, embedding, and Qdrant so stage spans are nested under the parent batch/job span. Extend monitoring snapshots with gauges for fully embedded panos and embedding totals.

**Tech Stack:** Python 3.14, OpenTelemetry traces/metrics/logs, SQLAlchemy, NATS, Qdrant, SigNoz PromQL/ClickHouse dashboard JSON.

---

### Task 1: Shared Observed Span Helper

**Files:**
- Modify: `services/main/main_service/observability/telemetry.py`
- Test: `services/main/tests/observability/test_telemetry.py`

- [ ] **Step 1: Write failing helper tests**

Add a fake telemetry class that records span nesting and durations:

```python
class FakeTelemetry:
    def __init__(self) -> None:
        self.durations: list[tuple[str, float, dict[str, object]]] = []
        self.spans: list[tuple[str, dict[str, object] | None]] = []

    def record_duration(self, name: str, seconds: float, attributes: dict[str, object]) -> None:
        self.durations.append((name, seconds, attributes))

    def span(self, name: str, attributes: dict[str, object] | None = None):
        self.spans.append((name, attributes))
        from contextlib import nullcontext
        return nullcontext()
```

Add test:

```python
def test_observed_span_records_trace_span_and_duration() -> None:
    current_time = [1.0]
    telemetry = FakeTelemetry()

    with observed_span(
        telemetry,  # type: ignore[arg-type]
        "embedding.qdrant_upsert",
        "embedding_qdrant_upsert",
        {"service": "embedding", "stage": "qdrant_upsert", "batch_size": 5},
        clock=lambda: current_time[0],
    ):
        current_time[0] = 1.8

    assert telemetry.spans == [
        (
            "embedding.qdrant_upsert",
            {"service": "embedding", "stage": "qdrant_upsert", "batch_size": 5},
        )
    ]
    assert telemetry.durations == [
        (
            "embedding_qdrant_upsert",
            0.8,
            {"service": "embedding", "stage": "qdrant_upsert", "batch_size": 5},
        )
    ]
```

- [ ] **Step 2: Verify test fails**

Run: `uv run pytest tests/observability/test_telemetry.py`

Expected: import failure for `observed_span`.

- [ ] **Step 3: Implement helper**

In `telemetry.py`, add:

```python
@contextlib.contextmanager
def observed_span(
    telemetry: "PipelineTelemetry",
    span_name: str,
    duration_name: str,
    attributes: dict[str, object],
    *,
    clock: Callable[[], float] | None = None,
) -> Iterator[None]:
    with telemetry.span(span_name, attributes):
        with recorded_duration(
            telemetry,
            duration_name,
            attributes,
            clock=clock,
        ):
            yield
```

- [ ] **Step 4: Verify helper tests pass**

Run: `uv run pytest tests/observability/test_telemetry.py`

Expected: all observability telemetry tests pass.

### Task 2: Downloader Nested Spans

**Files:**
- Modify: `services/main/main_service/downloader/runner.py`
- Test: `services/main/tests/downloader/test_runner.py`

- [ ] **Step 1: Extend test fake telemetry/progress**

Add a test using a progress callback and fake telemetry is optional, but assert events exist for these stage names:

```python
assert "downloader_claim_start" in event_names
assert "downloader_resolve_start" in event_names
assert "downloader_download_start" in event_names
assert "downloader_db_update_start" in event_names
assert "downloader_enqueue_processing_start" in event_names
```

- [ ] **Step 2: Verify test fails**

Run: `uv run pytest tests/downloader/test_runner.py::test_runner_reports_progress_events`

Expected: missing stage events.

- [ ] **Step 3: Add downloader stage events and spans**

Add optional `telemetry: PipelineTelemetry | None = None` to `run_downloader_batch`,
`_process_download_job`, and `_download_claimed_panorama`. Use `NoopTelemetry`
when no telemetry is passed. Wrap:

```python
with observed_span(telemetry, "downloader.job", "downloader_job", {"service": "downloader"}):
    ...
with observed_span(telemetry, "downloader.claim", "downloader_claim", {"service": "downloader"}):
    claimed = panorama_service.claim_panorama_for_download(pano_id)
```

Inside `_download_claimed_panorama`, wrap resolve, download, DB update, and enqueue with:

```python
_emit(progress, "downloader_resolve_start", {"pano_id": pano_id.value})
with observed_span(telemetry, "downloader.resolve", "downloader_resolve", {"service": "downloader"}):
    resolved = await streetview_client.resolve(...)
_emit(progress, "downloader_resolve_complete", {"pano_id": pano_id.value})
```

Repeat for `downloader_download`, `downloader_db_update`, and `downloader_enqueue_processing`.

- [ ] **Step 4: Pass telemetry from CLI**

In `services/main/main_service/downloader/__main__.py`, pass `telemetry=telemetry` into `run_downloader_batch`.

- [ ] **Step 5: Verify downloader tests**

Run: `uv run pytest tests/downloader/test_runner.py`

Expected: downloader runner suite passes.

### Task 3: Processing Nested Spans

**Files:**
- Modify: `services/main/main_service/processing/runner.py`
- Test: `services/main/tests/processing/test_processing_runner.py`

- [ ] **Step 1: Add failing progress assertions**

In `test_runner_reports_progress_events`, assert:

```python
assert "processing_source_load_start" in event_names
assert "processing_render_start" in event_names
assert "processing_db_update_start" in event_names
assert "processing_enqueue_embedding_start" in event_names
```

- [ ] **Step 2: Verify test fails**

Run: `uv run pytest tests/processing/test_processing_runner.py::test_runner_reports_progress_events`

Expected: missing stage events.

- [ ] **Step 3: Add processing telemetry parameter and nested spans**

Add optional `telemetry: PipelineTelemetry | None = None` to `run_processing_batch`,
`_process_received_job`, `_process_job`, and `_render_claimed_view`. Use
`NoopTelemetry` when absent.

Wrap:

- `processing.job`
- `processing.source_load`
- `processing.render_pool`
- `processing.render`
- `processing.db_update`
- `processing.enqueue_embedding`

For each wrapper, emit matching `_start`/`_complete` events with `pano_id`,
logical `view_id`, and `viewset_name` when available.

- [ ] **Step 4: Pass telemetry from CLI**

In `processing/__main__.py`, pass `telemetry=telemetry` into `run_processing_batch`.

- [ ] **Step 5: Verify processing tests**

Run: `uv run pytest tests/processing/test_processing_runner.py`

Expected: processing runner suite passes.

### Task 4: Embedding Nested Spans

**Files:**
- Modify: `services/main/main_service/embedding/runner.py`
- Modify: `services/main/main_service/embedding/qdrant_store.py`
- Test: `services/main/tests/embedding/test_embedding_runner.py`
- Test: `services/main/tests/embedding/test_qdrant_store.py`

- [ ] **Step 1: Add failing embedding stage assertions**

In `test_embedding_runner_reports_progress_events`, assert:

```python
assert "embedding_claim_start" in event_names
assert "embedding_model_encode_start" in event_names
assert "embedding_db_update_start" in event_names
```

For batch mode, assert:

```python
assert "embedding_qdrant_upsert_start" in event_names or "embedding_vector_store_batch_start" in event_names
```

- [ ] **Step 2: Verify test fails**

Run: `uv run pytest tests/embedding/test_embedding_runner.py::test_embedding_runner_reports_progress_events`

Expected: missing stage events.

- [ ] **Step 3: Add embedding telemetry parameter and nested spans**

Add optional `telemetry: PipelineTelemetry | None = None` to `run_embedding_batch`,
`_process_received_job`, and `_process_received_job_batch`. Use `NoopTelemetry`
when absent.

Wrap:

- `embedding.batch`
- `embedding.fetch`
- `embedding.job`
- `embedding.claim`
- `embedding.model_encode`
- `embedding.vector_store`
- `embedding.db_update`
- `embedding.ack`

Use duration names with underscores, e.g. `embedding_model_encode`.

- [ ] **Step 4: Instrument Qdrant upsert as child operation**

Modify `QdrantVectorStore` to accept optional `telemetry` in `__init__`. In
`add_many`, wrap the actual client upsert with:

```python
with observed_span(
    self.telemetry,
    "embedding.qdrant_upsert",
    "embedding_qdrant_upsert",
    {"service": "embedding", "collection": self.collection_name, "batch_size": len(points)},
):
    self._get_client().upsert(...)
```

Ensure `vector_store_factory.create_vector_store(...)` passes the embedding
telemetry object down when available.

- [ ] **Step 5: Pass telemetry from CLI**

In `embedding/__main__.py`, pass `telemetry=telemetry` into both
`create_vector_store(...)` and `run_embedding_batch(...)`.

- [ ] **Step 6: Verify embedding tests**

Run: `uv run pytest tests/embedding/test_embedding_runner.py tests/embedding/test_qdrant_store.py tests/embedding/test_vector_store_factory.py`

Expected: all pass.

### Task 5: Fully Embedded Pano Gauges

**Files:**
- Modify: `services/main/main_service/monitoring/snapshot.py`
- Modify: `services/main/main_service/monitoring/__main__.py`
- Test: `services/main/tests/monitoring/test_snapshot.py`
- Test: `services/main/tests/monitoring/test_monitoring_entrypoint.py`

- [ ] **Step 1: Add failing snapshot test**

Create DB rows with one pano, two complete views, and two complete embeddings
for active model. Assert snapshot includes:

```python
assert snapshot.embedding_progress["panos_fully_embedded"] == 1
assert snapshot.embedding_progress["embeddings_complete"] == 2
assert snapshot.embedding_progress["panos_with_multiple_views"] == 1
```

- [ ] **Step 2: Verify test fails**

Run: `uv run pytest tests/monitoring/test_snapshot.py`

Expected: missing `embedding_progress`.

- [ ] **Step 3: Implement snapshot SQL**

Add `embedding_progress: dict[str, int | float | None]` to `PipelineSnapshot`.
Use SQLAlchemy queries to compute:

- panos with more than one complete view.
- panos where all complete views have a complete embedding for the configured
  model spec if model fields are available.
- embeddings complete count.

- [ ] **Step 4: Emit gauges**

In `_emit_snapshot_metrics`, emit:

```python
telemetry.set_gauge("sf_search_embedding_progress", value, {"field": key})
```

- [ ] **Step 5: Verify monitoring tests**

Run: `uv run pytest tests/monitoring/test_snapshot.py tests/monitoring/test_monitoring_entrypoint.py`

Expected: monitoring tests pass.

### Task 6: Dashboard Panels And Docs

**Files:**
- Modify: `docs/signoz/sf-search-pipeline-dashboard.json`
- Modify: `docs/signoz/README.md`

- [ ] **Step 1: Update PromQL panels**

Add or update panels for:

- `embedding_model_encode` p50/p95/p99
- `embedding_qdrant_upsert` p50/p95/p99
- `embedding_db_update` p50/p95/p99
- `processing_render` p50/p95/p99
- `processing_enqueue_embedding` p50/p95/p99
- `downloader_download` p50/p95/p99
- `downloader_resolve` p50/p95/p99
- `sf_search_embedding_progress{field="panos_fully_embedded"}`
- `sf_search_embedding_progress{field="embeddings_complete"}`

- [ ] **Step 2: Document trace breakdown usage**

In `docs/signoz/README.md`, document that traces are nested as:

```text
*.batch
  *.fetch
  *.job
    *.claim
    *.download/render/model_encode
    *.db_update
    *.enqueue/qdrant_upsert/ack
```

- [ ] **Step 3: Validate dashboard JSON**

Run: `python3 -m json.tool docs/signoz/sf-search-pipeline-dashboard.json >/tmp/sf-search-dashboard-check.json`

Expected: exits 0.

### Task 7: Full Verification And Commit

**Files:**
- All modified files.

- [ ] **Step 1: Run focused tests**

Run:

```bash
uv run pytest tests/observability tests/downloader/test_runner.py tests/processing/test_processing_runner.py tests/embedding/test_embedding_runner.py tests/monitoring
```

Expected: all pass.

- [ ] **Step 2: Run full suite**

Run: `uv run pytest`

Expected: all pass.

- [ ] **Step 3: Secret scan changed files**

Run:

```bash
rg -n "PRIVATE_HOST_PATTERN|API[_ -]?KEY|secret|token|password|Users/" .codex/plans/2026-06-01-nested-operation-observability.md docs/signoz services/main/main_service services/main/tests -S
```

Expected: no new private values in changed observability files.

- [ ] **Step 4: Commit**

Run:

```bash
git add .codex/plans/2026-06-01-nested-operation-observability.md docs/signoz services/main/main_service services/main/tests
git commit -m "feat: add nested pipeline observability"
```
