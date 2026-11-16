# Training Operations Runbook

## Purpose

Operate the local MLOps experiment tracking platform,
execute reproducible training pipelines, and inspect results.

This is a portfolio laboratory, not a production deployment.

## Prerequisites

- Linux or WSL2 environment
- Docker Engine and Docker Compose
- Python 3.12 and uv
- Git
- Dataset already prepared under `data/`
- Local `.env` configured according to `.env.example`

Commands assume execution from the repository root.

## 1. Confirm repository state

```bash
git status --short --branch
git rev-parse HEAD
```

Use committed code for training.

If running locally, remove a previously exported Docker
revision to avoid incorrect source attribution:

```bash
unset PIPELINE_GIT_COMMIT
```

Do not change historical MLflow metadata to conceal errors.

## 2. Start the platform

```bash
docker compose up -d
docker compose ps
```

Expected services:

- `postgres`
- `silo`
- `mlflow`

All required services should be operational before training.

Check MLflow:

```bash
curl --fail --silent --show-error \
  http://127.0.0.1:5000/health
```

Tracking UI:

http://127.0.0.1:5000

Silo API:

http://127.0.0.1:9000

Silo Console:

http://127.0.0.1:9001

These addresses are intended for local development.

## 3. Verify project quality

```bash
make check
make verify-data
```

The data verification step confirms source integrity,
processed data consistency, and reproducibility.

Do not train against unverified data.

## 4. Execute the local Training Pipeline

This operation intentionally creates three MLflow runs.

Only execute it when a new experiment execution is required.

```bash
unset PIPELINE_GIT_COMMIT

MLFLOW_TRACKING_URI=http://127.0.0.1:5000 \
  uv run --locked python scripts/run_training_pipeline.py
```

Expected result:

```text
TRAINING PIPELINE: PASSED
```

The pipeline records:

- Execution ID
- Git commit
- Dataset SHA-256
- MLflow run identifiers
- Validation metrics
- Execution status

Each execution creates a local manifest:

```text
outputs/pipelines/<execution-id>/manifest.json
```

## 5. Execute the containerized Training Pipeline

First, verify the runtime:

```bash
export TRAINING_UID="$(id -u)"
export TRAINING_GID="$(id -g)"
export PIPELINE_GIT_COMMIT="$(git rev-parse HEAD)"

docker compose \
  -f compose.yaml \
  -f compose.training.yaml \
  run --rm --no-deps \
  training \
  uv run --locked python scripts/smoke_training_container.py
```

Expected:

```text
CONTAINER RUNTIME: PASSED
New training runs: 0
```

The smoke test does not train any models.

To intentionally run a new containerized training:

```bash
docker compose \
  -f compose.yaml \
  -f compose.training.yaml \
  run --rm --no-deps \
  training
```

This creates three new MLflow runs.

If the source code or dependencies have changed, rebuild
the training image before executing it:

```bash
docker compose \
  -f compose.yaml \
  -f compose.training.yaml \
  build training
```

The runtime uses:

- A non-root user
- A read-only root filesystem
- Read-only training data
- A writable output volume
- An internal Docker network connection to MLflow

The host provides the Git commit and UID/GID parameters.
The commit value is currently validated syntactically,
but is not cryptographically tied to the image.

## 6. Observe an existing execution

This command reads existing records and creates no new runs.

```bash
MLFLOW_TRACKING_URI=http://127.0.0.1:5000 \
  uv run --locked python scripts/observe_training_pipeline.py \
  --manifest outputs/pipelines/<execution-id>/manifest.json
```

Replace `<execution-id>` with a real successful execution ID.

Healthy output contains:

```text
PIPELINE_OBSERVATION_STARTED
RUN_VERIFIED
RUN_VERIFIED
RUN_VERIFIED
PIPELINE_HEALTHY
```

Events are JSON-formatted for automated log collection.

`PIPELINE_HEALTHY` means the observed records match
the expected execution evidence. It does not imply
that the model is approved or deployed to production.

## 7. Inspect model registry and inference

Registered model:

```text
predictive-maintenance-classifier
```

Current demonstration candidate:

```text
models:/predictive-maintenance-classifier/1
```

To inspect its demo lifecycle:

```bash
uv run --locked python \
  scripts/manage_model_lifecycle.py status
```

To execute a local inference example:

```bash
uv run --locked python \
  scripts/predict_registered_model.py
```

This loads the existing model and does not retrain it.

A candidate model is not automatically production-approved.

## 8. Inspect service logs

```bash
docker compose logs --tail=100 mlflow
docker compose logs --tail=100 postgres
docker compose logs --tail=100 silo
```

Use `docker compose ps` to inspect service health.

Avoid pasting credentials, secrets, or sensitive values
into issue trackers or public incident reports.

## 9. Stop services

To stop containers without deleting persistent volumes:

```bash
docker compose stop
```

To remove application containers and networks while
preserving persistent volumes:

```bash
docker compose down
```

Do not use `docker compose down -v` for routine cleanup.

The `-v` option removes named volumes, potentially
destroying MLflow metadata, database contents,
and stored artifacts.

## 10. Operational limitations

- This project uses local development infrastructure.
- MLflow is not configured as a production HA service.
- Training runs are created on each intentional execution.
- There is no automatic safe retry or run reconciliation.
- There is no production model serving endpoint.
- Dataset preparation and model tuning are separate tasks.
- The held-out test partition must not be used for
  repeated experimentation or threshold selection.
- No automatic GitHub Actions runs are required by this lab.

See [Incident Response](incident-response.md) for failures.
