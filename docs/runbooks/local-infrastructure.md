# Local Tracking Infrastructure

## Purpose

This environment provides persistent metadata and S3-compatible
artifact storage for the future MLflow Tracking Server.

## Services

- PostgreSQL 17: experiment metadata and model registry backend.
- Silo: S3-compatible object storage for ML artifacts.

MLflow itself will be deployed in the next milestone.

## Start

```bash
docker compose config --quiet
docker compose up -d
docker compose ps
```

## Initialize the artifact bucket

The local bucket must be initialized before configuring MLflow.

```bash
docker compose exec -T silo sh -c \
  'mcli alias set local http://127.0.0.1:9000 "$MINIO_ROOT_USER" "$MINIO_ROOT_PASSWORD" >/dev/null && mcli mb --ignore-existing local/mlflow-artifacts && mcli ls local'
```

## Local access

Silo Console: http://localhost:9001

S3 API: http://localhost:9000

PostgreSQL is reachable only from the Docker backend network.

## Persistence

The containers use named volumes:

- postgres-data
- silo-data

Data survives ordinary container restarts and recreation.

Do not run `docker compose down -v` unless intentionally deleting
local experiment infrastructure data.

## Security considerations

This is a local development environment, not a production
deployment.

- Credentials are generated into a Git-ignored `.env`.
- PostgreSQL is not published to the host.
- S3 and console ports bind to localhost.
- TLS, identity federation, backups, and external authentication
  are intentionally outside the initial milestone.

The future MLflow server must not be exposed publicly without
appropriate authentication and transport security.

## Stop

```bash
docker compose down
```

This stops and removes containers while retaining named volumes.
