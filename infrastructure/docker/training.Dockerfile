FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

ENV PYTHONUNBUFFERED=1 \
    UV_PYTHON_DOWNLOADS=0 \
    UV_NO_CACHE=1 \
    UV_LINK_MODE=copy

WORKDIR /app

COPY . /app

RUN uv sync --locked --no-dev --no-editable

ENV UV_NO_SYNC=1 \
    PATH="/app/.venv/bin:${PATH}" \
    HOME=/tmp

CMD ["uv", "run", "--locked", "python", "scripts/run_training_pipeline.py"]
