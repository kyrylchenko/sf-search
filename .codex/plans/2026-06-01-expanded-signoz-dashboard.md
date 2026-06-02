# Expanded SigNoz Dashboard Plan

## Goal

Create a second importable SigNoz dashboard JSON with the existing pipeline
panels plus a larger bottom section of extra useful metrics.

Add explicit embedding throughput and latency panels:

- Accurate tiles embedded per second/minute from `embedding_job_complete`.
- Accurate per-tile embedding job latency from `operation="embedding_job"`.
- Separate batch model encode latency from `operation="embedding_model_encode"`
  so batch inference timing is not mistaken for per-tile timing.

## Non-Goals

- Do not overwrite the existing dashboard.
- Do not include private SigNoz URLs, API keys, hostnames, or environment values.
- Do not include private runtime data or generated artifacts.

## Files

- Create: `docs/signoz/sf-search-pipeline-expanded-dashboard.json`
- Modify: `docs/signoz/README.md`
- Modify: `services/main/main_service/embedding/runner.py`
- Modify: `services/main/tests/embedding/test_embedding_runner.py`

## Approach

1. Load `docs/signoz/sf-search-pipeline-dashboard.json`.
2. Preserve all existing widgets.
3. Append extra widgets using the same SigNoz widget schema and PromQL/log
   query style already present in the existing dashboard.
4. Cover latency percentiles, rates, queue pressure, DB status counts, Qdrant,
   embedding progress, errors/skips, temp-tile cleanup, query UI behavior, and
   monitoring health.
5. Add a low-cardinality per-tile `embedding_job` duration metric in the batch
   embedding path. Single-tile mode already emits this duration; batch mode did
   not before this change.
6. Validate the JSON with `python3 -m json.tool`.

## Verification

```bash
uv run pytest services/main/tests/embedding/test_embedding_runner.py
python3 -m json.tool docs/signoz/sf-search-pipeline-expanded-dashboard.json >/tmp/sf-search-expanded-dashboard.json
rg -n "<private-value-patterns>" docs/signoz .codex/plans/2026-06-01-expanded-signoz-dashboard.md -S
git diff --check
```
