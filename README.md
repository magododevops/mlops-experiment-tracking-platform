# ML Experiment Tracking Platform

A reproducible Machine Learning experimentation platform for
predictive maintenance.

## Project Overview

The project demonstrates production-oriented MLOps practices,
including data validation, reproducible model training,
experiment tracking, model evaluation, and model governance.

**Status:** M1 — Repository Foundation.

## Technology Stack

- Python 3.12
- uv
- pytest
- Ruff

Planned components:

- scikit-learn, Pandas, NumPy
- MLflow Tracking and Model Registry
- PostgreSQL
- S3-compatible object storage
- Docker Compose
- GitHub Actions

## Repository Structure

```text
src/mlops_experiment_tracking/  Python package
tests/unit/                    Unit tests
pyproject.toml                 Project configuration
uv.lock                        Dependency lockfile
Makefile                       Development commands
```

## Getting Started

### Requirements

- Linux or WSL2
- uv
- GNU Make

### Installation

```bash
uv python install 3.12
uv sync --locked
```

### Quality Checks

```bash
make check
```

## Architecture

The initial release provides the Python package foundation.

Tracking infrastructure and architecture diagrams will be
introduced as the corresponding components are implemented.

## Roadmap

- [ ] M1 — Repository Foundation
- [ ] M2 — Dataset and Data Validation
- [ ] M3 — Baseline Model Training
- [ ] M4 — MLflow Tracking Infrastructure
- [ ] M5 — Experiment Comparison and Reproducibility
- [ ] M6 — Model Evaluation and Registry
- [ ] M7 — Automated Testing and CI
- [ ] M8 — Documentation and Architecture Diagrams
- [ ] M9 — Final Review and Portfolio Publication

## Development Standards

- Conventional Commits
- Type hints
- Reproducible dependencies
- Automated quality checks
- No committed secrets or generated datasets

## License

No project license has been selected yet.

## Dataset Preparation

This project uses the synthetic AI4I 2020 Predictive Maintenance
dataset from the UCI Machine Learning Repository.

```bash
uv run --locked python scripts/download_dataset.py
uv run --locked python scripts/prepare_dataset.py
make verify-data
```

The dataset is separated into stratified 60/20/20 training,
validation, and test partitions using a fixed random seed.

Dataset and split manifests record SHA-256 digests.
The verifier regenerates the expected partitions and detects
unexpected changes to local files.

Exploratory statistics are calculated only from training data.
Failure-mode labels and identifiers are excluded from model inputs.

See [Dataset Documentation](docs/data/dataset.md) for provenance,
feature contracts and limitations.
