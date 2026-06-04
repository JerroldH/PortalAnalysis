"""
Inference pipeline for the Hand Pronation-Supination (Up/Down) task.

Severity classes (MDS-UPDRS Part III, item 3.6):
    0 = Normal, 1 = Slight, 2 = Mild, 3 = Moderate/Severe

Symptoms predicted (with ``--with-symptoms``):
    amplitude_reduction, sequence_effect, slowness, halt_hesitation

Note: This task uses FFT augmentation (include_fft=True) matching the training config.
"""

from pathlib import Path
from typing import Dict, Optional

from portal_analysis.inference.base import BaseInferencePipeline, InferenceResult
from portal_analysis.models.model_manager import HandMovementModel


class HandUpDownPipeline(BaseInferencePipeline):
    """
    End-to-end inference for hand pronation-supination recordings.

    Quick start::

        pipeline = HandUpDownPipeline()
        result = pipeline.run_from_csv(
            "P001_right",
            Path(".../distances/P001_right_up_down_distances.csv"),
        )
        print(result.severity, result.symptoms)
    """

    TASK_NAME = "hand_up_down"
    DATA_COLUMN = "yaw_rad"
    MAX_SEQUENCE_LENGTH = 450

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
        """Convert pose landmarks to palm angles (yaw_rad), then run inference."""
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
        calc.calculate_hand_up_down_distances(pose_csv, distances_csv)
        return self.run_from_csv(
            patient_id, distances_csv, symptom_models, plot_path=plot_path
        )
