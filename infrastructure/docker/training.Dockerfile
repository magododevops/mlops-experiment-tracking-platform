FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim@sha256:e5b65587bce7de595f299855d7385fe7fca39b8a74baa261ba1b7147afa78e58

ARG TRAINING_UID=1000
ARG TRAINING_GID=1000

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_PYTHON_DOWNLOADS=0 \
    UV_NO_CACHE=1 \
    UV_LINK_MODE=copy

RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --gid "${TRAINING_GID}" trainer \
    && useradd \
        --uid "${TRAINING_UID}" \
        --gid "${TRAINING_GID}" \
        --create-home \
        --shell /usr/sbin/nologin \
        trainer

WORKDIR /app

COPY . /app

RUN uv sync --locked --no-dev --no-editable

ENV UV_NO_SYNC=1 \
    PATH="/app/.venv/bin:${PATH}" \
    HOME=/tmp

USER trainer:trainer

CMD ["uv", "run", "--locked", "python", "scripts/run_training_pipeline.py"]
