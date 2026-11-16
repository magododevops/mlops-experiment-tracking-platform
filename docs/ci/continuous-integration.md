# Continuous Integration

## Goal

Run Python quality checks on demand before merging or releasing.

This pipeline verifies code quality and unit tests without
requiring locally stored datasets or the MLflow infrastructure.

## Triggers

- Manual execution through GitHub Actions (`workflow_dispatch`)
- No automatic execution on push or pull request

## Checks

1. Install Python 3.12 and uv.
2. Restore the uv dependency cache.
3. Sync dependencies from `uv.lock`.
4. Run Ruff lint.
5. Verify Ruff formatting.
6. Run pytest.

## Local equivalent

```bash
make check
```

## Scope limitations

The CI job intentionally does not:

- Download or version full datasets.
- Use the held-out test partition.
- Connect to the local MLflow Tracking Server.
- Access PostgreSQL or S3 credentials.
- Register or promote models.
- Deploy an inference service.

Integration tests, data integrity checks with external data,
training jobs, and model promotion require separate workflows.

## Security

- GitHub Actions token has read-only repository permissions.
- Checkout does not retain Git credentials.
- Third-party actions are pinned by commit SHA.
- No secrets are required for this job.

## Repository protection

After the first successful run, configure branch protection
or a ruleset requiring the `Python Quality / 3.12` check
before merging changes into `main`.
