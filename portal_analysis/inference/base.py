"""
Base inference pipeline for hand movement tasks.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.sequence import pad_sequences

from portal_analysis.models.model_manager import HandMovementModel, ModelManager
from portal_analysis.models.paths import resolve_model_path


@dataclass
class InferenceResult:
    patient_id: str
    severity: int
    symptoms: Dict[str, int] = field(default_factory=dict)
    raw_sequence_length: int = 0

    def as_dict(self) -> Dict[str, Any]:
        """Flat dict for tabular display (DataFrame / console)."""
        return {
            "patient_id": self.patient_id,
            "severity": self.severity,
            "raw_sequence_length": self.raw_sequence_length,
            **self.symptoms,
        }

    def to_json_dict(
        self,
        task: str,
        subtask: str,
    ) -> Dict[str, Any]:
        """Structured payload written to ``<recording_id>_inference.json``."""
        payload: Dict[str, Any] = {
            "patient_id": self.patient_id,
            "task": task,
            "subtask": subtask,
            "severity": self.severity,
            "raw_sequence_length": self.raw_sequence_length,
        }
        if self.symptoms:
            payload["symptoms"] = dict(self.symptoms)
        return payload


class BaseInferencePipeline(abc.ABC):
    """Abstract inference pipeline for one motor task."""

    TASK_NAME: str = ""
    DATA_COLUMN: str = ""
    DEFAULT_MODEL_VERSION: str = "latest"

    MAX_SEQUENCE_LENGTH: int = 450

    def __init__(
        self,
        model_path: Optional[Path] = None,
        model_version: str = "latest",
    ):
        self._model: Optional[HandMovementModel] = None
        self._model_path: Optional[Path] = model_path
        self._model_version = model_version

    def load_model(self, path: Optional[Path] = None) -> None:
        if path is None:
            if self._model_path is not None:
                path = self._model_path
            else:
                path = resolve_model_path(self.TASK_NAME, self._model_version)

        self._model = ModelManager.load(path)
        print(f"[{self.TASK_NAME}] Model loaded from {path}")

    @property
    def model(self) -> HandMovementModel:
        if self._model is None:
            self.load_model()
        return self._model

    def _prepare_sequence(self, distances_csv: Path) -> Optional[np.ndarray]:
        distances_csv = Path(distances_csv)
        if not distances_csv.exists():
            return None

        df = pd.read_csv(distances_csv)
        if self.DATA_COLUMN not in df.columns or len(df) < 5:
            return None

        seq = df[self.DATA_COLUMN].dropna().values.astype(np.float32)
        return pad_sequences(
            [seq],
            maxlen=self.MAX_SEQUENCE_LENGTH,
            dtype="float32",
            padding="post",
            truncating="post",
        )

    def _predict_symptoms(
        self,
        X: np.ndarray,
        symptom_models: Dict[str, HandMovementModel],
    ) -> Dict[str, int]:
        symptoms = {}
        for name, sym_model in symptom_models.items():
            pred = sym_model.predict(X)
            symptoms[name] = int(pred[0])
        return symptoms

    def _save_kinematic_plot(
        self,
        distances_csv: Path,
        plot_path: Optional[Path] = None,
    ) -> None:
        from portal_analysis.preprocessing.kinematic_plots import (
            plot_kinematic_feature_over_time,
        )

        try:
            plot_kinematic_feature_over_time(
                distances_csv, self.DATA_COLUMN, plot_path=plot_path
            )
        except (ValueError, OSError) as exc:
            print(f"[{self.TASK_NAME}] Skipping kinematic plot: {exc}")

    def run_from_csv(
        self,
        patient_id: str,
        distances_csv: Path,
        symptom_models: Optional[Dict[str, HandMovementModel]] = None,
        plot_path: Optional[Path] = None,
    ) -> Optional[InferenceResult]:
        distances_csv = Path(distances_csv)
        if not distances_csv.exists():
            return None

        self._save_kinematic_plot(distances_csv, plot_path=plot_path)

        X = self._prepare_sequence(distances_csv)
        if X is None:
            print(f"[{self.TASK_NAME}] Skipping {patient_id}: invalid distances CSV.")
            return None

        severity = int(self.model.predict(X)[0])
        symptoms = self._predict_symptoms(X, symptom_models) if symptom_models else {}

        raw_len = len(pd.read_csv(distances_csv))
        return InferenceResult(
            patient_id=patient_id,
            severity=severity,
            symptoms=symptoms,
            raw_sequence_length=raw_len,
        )

    def run_from_pose(
        self,
        patient_id: str,
        pose_csv: Path,
        distances_csv: Optional[Path] = None,
        symptom_models: Optional[Dict[str, HandMovementModel]] = None,
        video_width: int = 1920,
        video_height: int = 1080,
        plot_path: Optional[Path] = None,
    ) -> Optional[InferenceResult]:
        """Convert a MediaPipe pose CSV to distances, then run inference."""
        from portal_analysis.preprocessing.distances import DistanceCalculator

        pose_csv = Path(pose_csv)
        if not pose_csv.exists():
            print(f"[{self.TASK_NAME}] Skipping {patient_id}: pose CSV not found ({pose_csv}).")
            return None

        if distances_csv is None:
            distances_csv = pose_csv.parent.parent / "distances" / f"{pose_csv.stem}_distances.csv"
        else:
            distances_csv = Path(distances_csv)

        calc = DistanceCalculator(width=video_width, height=video_height)
        calc.calculate_distances(pose_csv, distances_csv)
        return self.run_from_csv(
            patient_id, distances_csv, symptom_models, plot_path=plot_path
        )

    def run_from_video(
        self,
        patient_id: str,
        video_path: Path,
        pose_output_dir: Path,
        distances_output_dir: Path,
        symptom_models: Optional[Dict[str, HandMovementModel]] = None,
        file_prefix: Optional[str] = None,
        video_width: int = 1920,
        video_height: int = 1080,
        plot_path: Optional[Path] = None,
    ) -> Optional[InferenceResult]:
        from portal_analysis.preprocessing.hand_pose import HandPoseExtractor

        video_path = Path(video_path)
        prefix = file_prefix if file_prefix is not None else patient_id
        base_name = f"{prefix}_{video_path.stem}"
        pose_path = Path(pose_output_dir) / f"{base_name}.csv"
        dist_path = Path(distances_output_dir) / f"{base_name}_distances.csv"

        extractor = HandPoseExtractor()
        ok = extractor.process_video(video_path, pose_path)
        extractor.close()
        if not ok:
            print(f"[{self.TASK_NAME}] No hands detected in {video_path.name}")
            return None

        return self.run_from_pose(
            patient_id,
            pose_path,
            distances_csv=dist_path,
            symptom_models=symptom_models,
            video_width=video_width,
            video_height=video_height,
            plot_path=plot_path,
        )
