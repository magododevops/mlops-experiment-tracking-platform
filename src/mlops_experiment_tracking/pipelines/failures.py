"""Failure evidence and controlled training pipeline error handling."""

import json
import re
from datetime import UTC, datetime
from pathlib import Path


class PipelineStageError(RuntimeError):
    """An error associated with a known pipeline stage."""

    def __init__(self, stage: str, message: str) -> None:
        super().__init__(message)
        self.stage = stage


def build_failure_manifest(
    *,
    execution_id: str,
    started_at: str,
    stage: str,
    error_type: str,
    git_commit: str | None,
) -> dict:
    """Generate structured failure evidence without sensitive error text."""
    if not re.fullmatch(r"[0-9a-f]{32}", execution_id):
        raise ValueError("Invalid execution ID")

    if not stage or not re.fullmatch(r"[a-z][a-z0-9_]*", stage):
        raise ValueError("Invalid failure stage")

    if not error_type or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", error_type):
        raise ValueError("Invalid error type")

    return {
        "schema_version": 1,
        "pipeline": "baseline-training",
        "pipeline_execution_id": execution_id,
        "status": "FAILED",
        "started_at_utc": started_at,
        "failed_at_utc": datetime.now(UTC).isoformat(),
        "failed_stage": stage,
        "error_type": error_type,
        "git_commit": git_commit,
        "automatic_retry": False,
        "recovery_action": "Investigate before running again",
    }


def save_failure_manifest(
    manifest: dict,
    *,
    output_root: Path,
) -> Path:
    """Write failure evidence once, refusing to overwrite an execution."""
    destination = output_root / manifest["pipeline_execution_id"] / "failure.json"
    destination.parent.mkdir(parents=True, exist_ok=False)

    destination.write_text(
        json.dumps(manifest, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return destination
