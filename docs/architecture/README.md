# Architecture Diagrams

This directory contains editable draw.io diagrams.

## Infrastructure Architecture

[infrastructure.drawio](infrastructure.drawio)

Shows the local development environment, Docker Compose
services, MLflow Tracking Server, PostgreSQL, Silo, training
container, and data/output mounts.

## MLOps Lifecycle

[mlops-lifecycle.drawio](mlops-lifecycle.drawio)

Shows data integrity, training, validation, experiment tracking,
model registration, promotion controls, inference, and observation.

## Editing

Open https://app.diagrams.net/ and select:

File > Open From > Device

Select a `.drawio` file.

All architecture boxes and connectors are editable native shapes.

## Accuracy and scope

These diagrams describe the implemented local portfolio laboratory.

They are not production deployment diagrams.

In particular, this project does not claim:
- High availability
- Production inference deployment
- Continuous model drift monitoring
- Automatic retraining or safe retries
- Production-grade model approval controls

Technology symbols use editable text abbreviations rather than
external image assets, so the diagrams work offline.
