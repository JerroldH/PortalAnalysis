"""Save and load versioned model artifact bundles."""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import joblib
import sklearn

from portal_analysis.classification.representation_extractor import RocketRepresentationExtractor
from portal_analysis.classification.signal_augmentation import SignalAugmentation
from portal_analysis.training.task_config import TaskConfig

CLASSIFIER_FILE = "classifier.joblib"
ROCKET_FILE = "rocket.joblib"
METADATA_FILE = "metadata.json"


def _git_revision() -> Optional[str]:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
        return result.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def build_metadata(
    task_config: TaskConfig,
    metrics: Optional[Dict[str, float]] = None,
    version: Optional[str] = None,
) -> Dict[str, Any]:
    return {
        "version": version or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_sha": _git_revision(),
        "python_version": sys.version,
        "platform": platform.platform(),
        "sklearn_version": sklearn.__version__,
        "task_config": task_config.to_dict(),
        "metrics": metrics or {},
    }


def save_artifact_bundle(
    output_dir: Path,
    augmenter: SignalAugmentation,
    rocket: RocketRepresentationExtractor,
    classifier,
    task_config: TaskConfig,
    metrics: Optional[Dict[str, float]] = None,
    version: Optional[str] = None,
) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = build_metadata(task_config, metrics=metrics, version=version)
    metadata["augmenter_config"] = {
        "smooth_window_length": augmenter.smooth_window_length,
        "smooth_polyorder": augmenter.smooth_polyorder,
        "gaussian_sigma": augmenter.gaussian_sigma,
        "include_fft": augmenter.include_fft,
        "include_diffs": augmenter.include_diffs,
        "augmentation_method": task_config.augmentation_method,
    }

    joblib.dump(classifier, output_dir / CLASSIFIER_FILE)
    joblib.dump(rocket, output_dir / ROCKET_FILE)

    with open(output_dir / METADATA_FILE, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return output_dir


@dataclass
class ArtifactBundle:
    augmenter: SignalAugmentation
    rocket: RocketRepresentationExtractor
    classifier: Any
    metadata: Dict[str, Any]
    path: Path


def _augmenter_from_metadata(metadata: Dict[str, Any]) -> SignalAugmentation:
    cfg = metadata["augmenter_config"]
    return SignalAugmentation(
        smooth_window_length=cfg["smooth_window_length"],
        smooth_polyorder=cfg["smooth_polyorder"],
        gaussian_sigma=cfg["gaussian_sigma"],
        include_fft=cfg["include_fft"],
        include_diffs=cfg["include_diffs"],
    )


def load_artifact_bundle(model_dir: Path) -> ArtifactBundle:
    model_dir = Path(model_dir)
    if not model_dir.is_dir():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    metadata_path = model_dir / METADATA_FILE
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing {METADATA_FILE} in {model_dir}")

    with open(metadata_path, encoding="utf-8") as f:
        metadata = json.load(f)

    return ArtifactBundle(
        augmenter=_augmenter_from_metadata(metadata),
        rocket=joblib.load(model_dir / ROCKET_FILE),
        classifier=joblib.load(model_dir / CLASSIFIER_FILE),
        metadata=metadata,
        path=model_dir,
    )
