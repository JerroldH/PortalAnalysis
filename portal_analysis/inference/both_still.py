"""
Inference pipeline for the Hand Tremor (Both Still) task.

Severity classes (MDS-UPDRS Part III, item 3.11):
    0 = Normal, 1 = Slight, 2 = Mild, 3 = Moderate/Severe
"""

from pathlib import Path
from typing import Dict, Optional

from portal_analysis.inference.base import BaseInferencePipeline, InferenceResult
from portal_analysis.models.model_manager import HandMovementModel


class BothStillPipeline(BaseInferencePipeline):
    """Inference for both-hands-still tremor recordings (per-hand severity)."""

    TASK_NAME = "both_still"
    DATA_COLUMN = "mean_fingertip_distance_from_center"
    MAX_SEQUENCE_LENGTH = 450
    VIDEO_STEM = "both_still"

    @classmethod
    def hand_distances_stem(cls, patient_id: str, subtask: str) -> str:
        """Booth-style stem: ``{patient_id}_both_still_{left|right}_hand``."""
        return f"{patient_id}_{cls.VIDEO_STEM}_{subtask}_hand"

    @classmethod
    def distances_csv_path(
        cls,
        processed_dir: Path,
        patient_id: str,
        subtask: str,
        *,
        prefer_results: bool = True,
    ) -> Path:
        """Resolve per-hand tremor CSV (results layout, then legacy booth)."""
        filename = f"{cls.hand_distances_stem(patient_id, subtask)}.csv"
        results_path = (
            Path(processed_dir) / "results" / "distances" / subtask / filename
        )
        if prefer_results and results_path.exists():
            return results_path
        legacy = Path(processed_dir) / "both_still" / subtask / "distances" / filename
        if legacy.exists():
            return legacy
        flat_results = Path(processed_dir) / "results" / "distances" / filename
        if flat_results.exists():
            return flat_results
        return results_path

    @classmethod
    def write_tremor_distances_from_pose(
        cls,
        pose_csv: Path,
        processed_dir: Path,
        patient_id: str,
    ) -> None:
        """
        Pose CSV -> tremor metrics under ``results/distances/{left,right}/``.

        HandTremorProcessor initially writes split files next to the combined
        CSV in ``results/distances/``; we move them into per-hand subfolders to
        match Booth training layout.
        """
        from portal_analysis.preprocessing.hand_tremor import HandTremorProcessor

        pose_csv = Path(pose_csv)
        dist_root = Path(processed_dir) / "results" / "distances"
        dist_root.mkdir(parents=True, exist_ok=True)

        combined = dist_root / f"{patient_id}_{cls.VIDEO_STEM}.csv"
        HandTremorProcessor().process_csv_file(
            pose_csv, combined, separate_hands=True
        )

        for subtask in ("left", "right"):
            flat_hand = dist_root / f"{cls.hand_distances_stem(patient_id, subtask)}.csv"
            if not flat_hand.exists():
                continue
            hand_dir = dist_root / subtask
            hand_dir.mkdir(parents=True, exist_ok=True)
            target = hand_dir / flat_hand.name
            flat_hand.replace(target)

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
        *,
        subtask: str = "left",
        processed_dir: Optional[Path] = None,
    ) -> Optional[InferenceResult]:
        del video_width, video_height
        pose_csv = Path(pose_csv)
        if not pose_csv.exists():
            print(f"[{self.TASK_NAME}] Skipping {patient_id}: pose CSV not found ({pose_csv}).")
            return None

        if processed_dir is not None:
            self.write_tremor_distances_from_pose(pose_csv, processed_dir, patient_id)

        if distances_csv is None:
            if processed_dir is None:
                print(
                    f"[{self.TASK_NAME}] Skipping {patient_id}: "
                    "need processed_dir or distances_csv."
                )
                return None
            distances_csv = self.distances_csv_path(
                processed_dir, patient_id, subtask, prefer_results=False
            )
        else:
            distances_csv = Path(distances_csv)

        recording_id = self.hand_distances_stem(patient_id, subtask)
        result = self._with_artifact(
            self.run_from_csv(
                recording_id,
                distances_csv,
                symptom_models,
                plot_path=plot_path,
                source_video_path=source_video_path,
            ),
            "pose_csv",
            pose_csv,
        )
        return result

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
        *,
        subtask: str = "left",
        processed_dir: Optional[Path] = None,
    ) -> Optional[InferenceResult]:
        from portal_analysis.preprocessing.hand_pose import HandPoseExtractor

        video_path = Path(video_path)
        prefix = file_prefix if file_prefix is not None else patient_id
        pose_path = Path(pose_output_dir) / f"{prefix}_{self.VIDEO_STEM}.csv"

        if not pose_path.exists():
            extractor = HandPoseExtractor()
            ok = extractor.process_video(video_path, pose_path)
            extractor.close()
            if not ok:
                print(f"[{self.TASK_NAME}] No hands detected in {video_path.name}")
                return None

        proc_dir = processed_dir if processed_dir is not None else Path(distances_output_dir).parent.parent
        return self._with_artifact(
            self.run_from_pose(
                patient_id,
                pose_path,
                symptom_models=symptom_models,
                video_width=video_width,
                video_height=video_height,
                plot_path=plot_path,
                subtask=subtask,
                processed_dir=proc_dir,
                source_video_path=video_path,
            ),
            "video_path",
            video_path,
        )
