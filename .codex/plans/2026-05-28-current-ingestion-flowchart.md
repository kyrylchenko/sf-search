# Current Ingestion Flowchart Plan

## Goal

Create a concrete, openable flowchart artifact showing the current implemented
path from input coordinates/boundaries through coverage discovery, durable DB
state, NATS queues, full-resolution pano download, and processing job output.

## Non-Goals

- Do not plan the website/query path.
- Do not add implementation code.
- Do not include private local paths, credentials, or generated pano data.

## Output

- `docs/diagrams/current-ingestion-flow.svg`

## Verification

- Validate the SVG XML parses.
- Commit the docs artifact.
