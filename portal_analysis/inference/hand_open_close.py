"""
Inference pipeline for the Hand Open/Close task.

Severity classes (MDS-UPDRS Part III, item 3.5):
    0 = Normal, 1 = Slight, 2 = Mild, 3 = Moderate/Severe

Symptoms predicted (with ``--with-symptoms``):
    amplitude_reduction, sequence_effect, slowness, halt_hesitation
"""

from pathlib import Path
from typing import Dict, Optional

from portal_analysis.inference.base import BaseInferencePipeline, InferenceResult
from portal_analysis.models.model_manager import HandMovementModel


class HandOpenClosePipeline(BaseInferencePipeline):
    """
    End-to-end inference for hand open/close recordings.

    Quick start::

        pipeline = HandOpenClosePipeline()
        result = pipeline.run_from_csv(
            "P001_right",
            Path(".../distances/P001_right_open_close_distances.csv"),
        )
        print(result.severity, result.symptoms)
    """

    TASK_NAME = "hand_open_close"
    DATA_COLUMN = "Normalized Hand Sum Finger Distances"
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
        source_video_path: Optional[Path] = None,
    ) -> Optional[InferenceResult]:
        """Convert pose landmarks to open/close distances, then run inference."""
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
        calc.calculate_hand_open_close_distances(pose_csv, distances_csv)
        return self._with_artifact(
            self.run_from_csv(
                patient_id,
                distances_csv,
                symptom_models,
                plot_path=plot_path,
                source_video_path=source_video_path,
            ),
            "pose_csv",
            pose_csv,
        )
