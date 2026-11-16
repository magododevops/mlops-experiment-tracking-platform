"""Validate the Docker training runtime without executing training."""

import os
import pwd
import shutil
import subprocess
from pathlib import Path
from urllib.request import urlopen

from mlops_experiment_tracking.pipelines.runtime import (
    validate_runtime,
)


def is_readonly(path: Path) -> bool:
    """Inspect the readonly flag of an existing filesystem mount."""
    flags = os.statvfs(path).f_flag
    return bool(flags & os.ST_RDONLY)


def main() -> None:
    root = Path("/app")
    data = root / "data"
    outputs = root / "outputs"

    uid = os.getuid()
    username = pwd.getpwuid(uid).pw_name

    revision = os.environ.get("PIPELINE_GIT_COMMIT", "")
    git_executable = shutil.which("git")

    validate_runtime(
        uid=uid,
        username=username,
        git_executable=git_executable,
        revision=revision,
        root_readonly=is_readonly(root),
        data_readonly=is_readonly(data),
        outputs_writable=os.access(outputs, os.W_OK),
    )

    if (root / ".git").exists():
        raise RuntimeError("Git metadata must not be baked into the image")

    if not (data / "raw" / "ai4i2020.csv").is_file():
        raise FileNotFoundError("Raw training dataset is missing")

    if not (data / "processed" / "seed-42").is_dir():
        raise FileNotFoundError("Processed training dataset is missing")

    git_version = subprocess.check_output(
        ["git", "--version"],
        text=True,
    ).strip()

    tracking_uri = os.environ["MLFLOW_TRACKING_URI"]
    health_url = tracking_uri.rstrip("/") + "/health"

    with urlopen(health_url, timeout=5) as response:
        if response.status != 200:
            raise RuntimeError("MLflow health check failed")

    print()
    print("TRAINING CONTAINER SMOKE TEST")
    print("-----------------------------")
    print(f"User: {username}")
    print(f"UID: {uid}")
    print(f"Git: {git_version}")
    print(f"Source revision: {revision}")
    print("Root filesystem: READ-ONLY")
    print("Data mount: READ-ONLY")
    print("Outputs mount: WRITABLE")
    print("Git repository metadata: EXCLUDED")
    print("MLflow health: PASSED")
    print("New training runs: 0")
    print("CONTAINER RUNTIME: PASSED")


if __name__ == "__main__":
    main()
