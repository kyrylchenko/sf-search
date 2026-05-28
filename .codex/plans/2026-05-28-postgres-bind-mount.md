# Postgres Bind Mount Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the local Postgres data directory visible on disk while keeping the database port and public-safe local credentials easy to inspect.

**Architecture:** Keep Postgres exposed on `localhost:5432`. Replace the Docker named Postgres volume with a repo-local bind mount at `./.local/postgres-data`, and ignore `.local/` so database files are never committed.

**Tech Stack:** Docker Compose, Postgres 16 Alpine.

---

## Tasks

- [x] Ignore local runtime data directories.
- [x] Change Postgres Compose storage to `./.local/postgres-data`.
- [x] Document connection credentials and disk mount path.
- [x] Validate Compose config.
- [x] Restart Postgres and verify readiness/query.
- [x] Commit changes.
