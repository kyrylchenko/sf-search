# sf-search

## Local Infrastructure

Start Postgres and NATS JetStream:

```bash
docker compose up -d postgres nats
```

Verify Postgres:

```bash
docker compose exec postgres pg_isready -U sf_search -d sf_search
docker compose exec postgres psql -U sf_search -d sf_search -c "select version();"
```

Connect from your host:

```text
host: localhost
port: 5432
database: sf_search
user: sf_search
password: sf_search_dev_password
```

Local Postgres files are bind-mounted at `./.local/postgres-data`. The `.local/`
directory is ignored by git.

Verify NATS:

```bash
docker compose exec nats wget -qO- http://127.0.0.1:8222/healthz
```

## Local Ingestion

Run a bounded downloader batch from the main service:

```bash
cd services/main
uv run python -m main_service.downloader --limit 5
```

Downloaded panoramas are written to `.local/panoramas` by default. The
downloader publishes lightweight processing jobs to `PANO_PROCESSING`; queued
messages contain only the pano ID and local image path, not image bytes.

Inspect downloaded rows:

```bash
docker compose exec postgres psql -U sf_search -d sf_search -c "select orig_id, download_status, image_path, downloaded_at from panorama_table order by downloaded_at desc nulls last limit 10;"
```

Inspect JetStream stream depth:

```bash
docker compose exec nats nats stream info PANO_PROCESSING
```

The committed `.env.example` values are local placeholders. Do not commit real
credentials, private endpoints, downloaded panoramas, vectors, or generated
indexes.
