# Long-Running Worker Services Plan

## Goal

Make downloader, processing, and embedding CLIs behave like real services that
keep polling for months instead of exiting after one empty queue fetch.

## Evidence

- The current CLIs run one bounded batch and close NATS connections immediately.
- The embedding queue has hundreds of pending durable-consumer messages, while
  fresh processed views are still pending in Postgres.
- Running the embedding worker with a small limit can skip a few duplicate/old
  messages and exit before it reaches fresh view IDs.
- NATS stream retained message count is not the same as downstream backlog; queue
  backpressure should prefer durable consumer `num_pending + num_ack_pending`.

## Approach

- Keep bounded batch runners unchanged for unit tests and smoke probes.
- Add a shared service loop helper for CLI entrypoints.
- Make CLIs loop forever by default and add `--once` for explicit bounded runs.
- Add configurable `--idle-sleep-seconds` / `SERVICE_IDLE_SLEEP_SECONDS`.
- Sleep only after empty or paused batches. If a batch skips old duplicate
  messages, immediately fetch again so the worker can drain to fresh jobs.
- Configure producer-side NATS queues with the downstream durable consumer name
  so backpressure checks use consumer backlog when available.

## Verification

- Unit test the loop helper.
- Unit test CLI parser flags.
- Unit test NATS queue pending-count behavior for stream and durable consumer
  modes.
- Run focused tests, then full `uv run pytest -q`.
