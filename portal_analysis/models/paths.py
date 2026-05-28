"""Resolve model artifact paths for training and inference."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from portal_analysis.config import MODELS_DIRECTORY
from portal_analysis.training.artifact import METADATA_FILE

ALL_TASKS = ("finger_tapping", "hand_open_close", "hand_up_down")


def model_dir_for_task(
    task_name: str,
    version: str = "latest",
    models_root: Optional[Path] = None,
) -> Path:
    """Return expected artifact directory: models/<task>/<version>/"""
    root = Path(models_root) if models_root else MODELS_DIRECTORY
    return root / task_name / version


def legacy_model_file(task_name: str, models_root: Optional[Path] = None) -> Path:
    """Legacy single-file layout: models/<task>/<task>_minirocket.joblib"""
    root = Path(models_root) if models_root else MODELS_DIRECTORY
    return root / task_name / f"{task_name}_minirocket.joblib"


def resolve_model_path(
    task_name: str,
    version: str = "latest",
    models_root: Optional[Path] = None,
) -> Path:
    """
    Resolve model path for a task.

    Checks (in order):
    1. Versioned artifact bundle directory (metadata.json present)
    2. Legacy .joblib file
    """
    bundle_dir = model_dir_for_task(task_name, version, models_root)
    if (bundle_dir / METADATA_FILE).exists():
        return bundle_dir

    legacy = legacy_model_file(task_name, models_root)
    if legacy.exists():
        return legacy

    if bundle_dir.exists():
        return bundle_dir

    raise FileNotFoundError(
        f"No model found for task '{task_name}' (version={version}). "
        f"Expected {bundle_dir} or {legacy}. "
        "Train with: python -m portal_analysis.cli train --task {task_name}"
    )
