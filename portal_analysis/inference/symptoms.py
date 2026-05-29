"""
Clinical motor sign (symptom) prediction for hand movement tasks.

Four binary signs per task (except both_still), aligned with
Hand-Movement-Analysis symptom classification:

    amplitude_reduction, sequence_effect, slowness, halt_hesitation
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from portal_analysis.models.model_manager import HandMovementModel, ModelManager
from portal_analysis.models.paths import INFERENCE_TASKS, symptom_model_dir
from portal_analysis.training.task_config import TaskConfig

CANONICAL_SYMPTOMS: Tuple[str, ...] = (
    "amplitude_reduction",
    "sequence_effect",
    "slowness",
    "halt_hesitation",
)

# Source label column(s) per task; multiple columns are combined with logical OR.
SYMPTOM_SOURCE_COLUMNS: Dict[str, Dict[str, List[str]]] = {
    "finger_tapping": {
        "amplitude_reduction": ["AMPLITUDE_Predicted_Label"],
        "sequence_effect": ["DECRAMPLITUDE_Predicted_Label"],
        "slowness": ["SLOWNESS_Predicted_Label"],
        "halt_hesitation": ["HALT_HESITATION_Predicted_Label"],
    },
    "hand_open_close": {
        "amplitude_reduction": ["Low Amplitude"],
        "sequence_effect": ["Dec. Amp."],
        "slowness": ["Low speed"],
        "halt_hesitation": ["Halts", "Irregularity"],
    },
    "hand_up_down": {
        "amplitude_reduction": ["dec. amp."],
        "sequence_effect": ["dec. amp."],
        "slowness": ["low speed"],
        "halt_hesitation": ["irregular", "can't turn hand"],
    },
}

# Optional task-level overrides when training symptom models (severity config unchanged).
SYMPTOM_TRAINING_OVERRIDES: Dict[str, Dict[str, object]] = {
    "finger_tapping": {"include_fft": True},
}


def symptom_sources_for_task(task_name: str) -> Dict[str, List[str]]:
    if task_name not in SYMPTOM_SOURCE_COLUMNS:
        raise ValueError(
            f"Task {task_name!r} has no symptom definitions "
            f"(supported: {', '.join(SYMPTOM_SOURCE_COLUMNS)})."
        )
    return SYMPTOM_SOURCE_COLUMNS[task_name]


def load_symptom_labels(
    task_config: TaskConfig,
    symptom_key: str,
    base_dir: Path,
) -> pd.Series:
    """
    Load binary (0/1) labels for one canonical symptom from the task symptoms CSV.

    Multiple source columns are OR-combined (positive if any column is 1).
    """
    sources = symptom_sources_for_task(task_config.task_name)[symptom_key]
    if not task_config.symptoms_file:
        raise ValueError(f"Task {task_config.task_name} has no symptoms_file configured.")

    labels_path = (
        base_dir
        / task_config.task_name
        / task_config.labels_subdirectory
        / task_config.symptoms_file
    )
    df_labels = pd.read_csv(labels_path).set_index("ID")
    df_labels = df_labels[~df_labels.index.duplicated(keep="first")]

    series_list = []
    for col in sources:
        if col not in df_labels.columns:
            continue
        s = pd.to_numeric(df_labels[col], errors="coerce")
        s = s.where(s.isin([0, 1]))
        series_list.append(s)

    if not series_list:
        raise ValueError(
            f"No symptom source columns {sources!r} found in {labels_path} "
            f"for {symptom_key!r}."
        )

    stacked = pd.concat(series_list, axis=1)
    combined = stacked.max(axis=1, skipna=True)
    combined = combined.where(stacked.notna().any(axis=1))
    return combined.dropna()


def load_symptom_models(
    task_name: str,
    version: str = "latest",
    models_root: Optional[Path] = None,
) -> Dict[str, HandMovementModel]:
    """
    Load all available symptom classifiers for a task.

    Returns {canonical_symptom_key: HandMovementModel}. Missing bundles are skipped.
    When two symptoms share the same trained artifact (e.g. hand_up_down dec. amp.),
    the same model object may be loaded once and reused.
    """
    if task_name not in INFERENCE_TASKS:
        return {}

    loaded: Dict[str, HandMovementModel] = {}
    artifact_cache: Dict[Path, HandMovementModel] = {}

    for symptom_key in CANONICAL_SYMPTOMS:
        bundle_dir = symptom_model_dir(task_name, symptom_key, version, models_root)
        if not bundle_dir.exists():
            continue
        if bundle_dir not in artifact_cache:
            artifact_cache[bundle_dir] = ModelManager.load(bundle_dir)
        loaded[symptom_key] = artifact_cache[bundle_dir]

    return loaded


def load_symptom_models_for_tasks(
    tasks: Optional[List[str]] = None,
    version: str = "latest",
    models_root: Optional[Path] = None,
) -> Dict[str, Dict[str, HandMovementModel]]:
    """Load symptom models for each inference task."""
    task_list = tasks or list(INFERENCE_TASKS)
    return {
        task: load_symptom_models(task, version=version, models_root=models_root)
        for task in task_list
        if task in INFERENCE_TASKS
    }
