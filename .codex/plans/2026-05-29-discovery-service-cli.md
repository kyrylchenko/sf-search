# Discovery Service CLI Plan

## Goal

Replace the manual Python heredoc for scheduling panorama downloads with a real
service command: `python -m main_service.discovery`.

## Approach

- Add a discovery entrypoint package beside downloader, processing, and
  embedding.
- Reuse existing boundary loading, StreetLevel coverage lookup, Postgres
  discovery persistence, NATS download queue publishing, CLI logging, and service
  loop.
- Add `MAX_DOWNLOADER_QUEUE_DEPTH` config so discovery can pause before
  overfilling downloader jobs.
- Keep `--once` for explicit smoke runs; default behavior keeps polling.
- Document the command in `services/main/README.md`.

## Verification

- Add entrypoint parser tests.
- Add config default test.
- Run focused tests and full `uv run pytest -q`.
