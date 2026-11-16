# Incident Response Runbook

## Purpose

Diagnose training pipeline failures without losing
experiment metadata or creating unnecessary duplicate runs.

## Incident priorities

1. Preserve evidence.
2. Identify the failed stage.
3. Confirm infrastructure health.
4. Inspect MLflow for partial results.
5. Correct the underlying cause.
6. Decide whether a new execution is necessary.

Do not automatically rerun training after every failure.

## Scenario A: MLflow is unavailable

### Symptoms

- MLflow connection error
- Tracking API unavailable
- Training pipeline cannot reach the tracking server

### Diagnosis

Check containers:

```bash
docker compose ps
```

Check the local MLflow health endpoint:

```bash
curl --fail --silent --show-error \
  http://127.0.0.1:5000/health
```

Inspect logs:

```bash
docker compose logs --tail=100 mlflow
docker compose logs --tail=100 postgres
docker compose logs --tail=100 silo
```

### Recovery

If required services were simply stopped:

```bash
docker compose up -d
docker compose ps
```

Recheck the health endpoint.

Do not clear volumes or recreate databases as a
first troubleshooting action.

## Scenario B: Dataset integrity fails

### Symptoms

- SHA-256 mismatch
- Processed dataset mismatch
- Unexpected schema or split inconsistency

### Diagnosis

```bash
make verify-data
```

### Recovery

Confirm that the source dataset matches the approved
version and that preprocessing steps are deterministic.

Do not edit dataset files merely to make a test pass.

If a dataset version is intentionally changed, perform
a controlled update of the dataset contract, manifests,
and experimental baselines.

## Scenario C: A training execution fails

### Symptoms

- Nonzero exit code
- `PIPELINE_FAILED` event
- `failure.json` created under an execution directory

### Diagnosis

Inspect the recorded evidence:

```bash
find outputs/pipelines -name failure.json -print
```

Inspect a specific file:

```bash
cat outputs/pipelines/<execution-id>/failure.json
```

The failure manifest may contain:

- Execution ID
- Failure stage
- Error type
- Git revision
- Start and failure timestamps
- Retry policy

The current implementation does not automatically
capture every partially created MLflow run.

### Recovery

Identify the actual failure before another training attempt.

If the failure occurred before training, no new runs
should have been started by the training step.

If the failure occurred after training began, inspect
MLflow for runs that may have been created already.

Do not assume retrying is idempotent.

## Scenario D: Observer reports inconsistent metrics

### Symptoms

- `PIPELINE_UNHEALTHY`
- Mismatched run metrics
- Incorrect Git commit tag
- Missing or unexpected run

### Diagnosis

Run the observer using an existing successful manifest:

```bash
MLFLOW_TRACKING_URI=http://127.0.0.1:5000 \
  uv run --locked python scripts/observe_training_pipeline.py \
  --manifest outputs/pipelines/<execution-id>/manifest.json
```

Review the JSON error and the associated Run ID.

Open the run in the MLflow UI and inspect:

- Run status
- Run parameters
- Run metrics
- `pipeline_execution_id`
- `source_git_commit`
- Run source metadata

### Recovery

Confirm whether the manifest or MLflow data is inconsistent.

Do not silently overwrite evidence to make monitoring green.

Preserve existing evidence and document the investigation.

## Scenario E: Incorrect source Git commit

### Symptoms

The recorded Git commit differs from the revision
you expected to execute.

### Diagnosis

```bash
git rev-parse HEAD
printenv PIPELINE_GIT_COMMIT || true
```

### Recovery

For local Python training:

```bash
unset PIPELINE_GIT_COMMIT
```

For container execution:

```bash
export PIPELINE_GIT_COMMIT="$(git rev-parse HEAD)"
```

Rebuild the image after source changes.

Do not rewrite historical manifests to imply that
they were generated from a different source revision.

The current revision injection validates SHA format,
not complete image provenance.

## Scenario F: Demo model cannot be resolved

### Symptoms

- Registry alias not found
- Model version not accessible
- Local inference fails when using an alias

### Diagnosis

```bash
uv run --locked python \
  scripts/manage_model_lifecycle.py status
```

Check the Model Registry in the MLflow UI.

### Recovery

Confirm whether the requested alias is expected to exist.

The demo lifecycle separates:

- Candidate
- Demo-approved
- Staging-demo active
- Staging-demo inactive

Do not create production aliases to fix a demo lookup.

The `rollback` operation in the current lab removes
the staging-demo alias; it does not restore an older
deployed model version.

## Controlled failure injection

The following test intentionally fails before training:

```bash
PIPELINE_TEST_FAIL_AT=before_training \
MLFLOW_TRACKING_URI=http://127.0.0.1:5000 \
  uv run --locked python scripts/run_training_pipeline.py
```

Expected behavior:

- Exit code 1
- `PIPELINE_FAILED` event
- `failure.json` created
- No training step started

Run this only when intentionally testing failure handling.

Do not use the injection setting during normal training.

## Escalation checklist

Before asking another engineer to investigate, collect:

- Execution ID
- Source Git commit
- Failure stage
- Error type
- Relevant logs, with secrets removed
- Service health
- Existing MLflow Run IDs, if any
- Whether the Model Registry was modified
- Whether retry was attempted

## Known gaps

- No automated incident alert delivery
- No running-job heartbeat or timeout supervision
- No automatic reconciliation of partial runs
- No safe automatic retry strategy
- No dedicated persistent incident event database
- No centralized distributed tracing
- No disaster recovery procedure validated for production

These limitations must not be represented as
implemented production capabilities.
