FROM ghcr.io/astral-sh/uv:python3.14-bookworm

ENV PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

WORKDIR /app

COPY services/shared /app/services/shared
COPY services/main/README.md services/main/pyproject.toml services/main/uv.lock /app/services/main/

WORKDIR /app/services/main
RUN uv sync --frozen --group embedding --no-install-project

COPY services/main /app/services/main
COPY docs /app/docs
RUN uv sync --frozen --group embedding

CMD ["uv", "run", "python", "-m", "main_service"]
