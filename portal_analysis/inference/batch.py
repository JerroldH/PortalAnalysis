"""
Batch inference pipeline: runs all three hand movement tasks for one or more patients
and consolidates results into a single DataFrame / CSV report.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Union

import pandas as pd

from portal_analysis.inference.base import InferenceResult
from portal_analysis.inference.finger_tapping import FingerTappingPipeline
from portal_analysis.inference.hand_open_close import HandOpenClosePipeline
from portal_analysis.inference.hand_up_down import HandUpDownPipeline


class BatchInferencePipeline:
    """
    Run all hand movement inference tasks for a list of patients.

    The pipeline expects processed distances CSVs already computed by
    DistanceCalculator.  For video-based processing use run_from_videos().

    Parameters
    ----------
    model_paths : dict, optional
        {task_name: Path} overrides for individual model files.
        Tasks not listed fall back to their DEFAULT_MODEL_NAME.

    Example
    -------
    ::
        batch = BatchInferencePipeline()
        results_df = batch.run_from_csvs(
            patient_ids=["P001", "P002"],
            distances_dir=Path("N:/Booth_Processed"),
        )
        results_df.to_csv("booth_inference_results.csv", index=False)
    """

    TASK_PIPELINE_MAP = {
        "finger_tapping": FingerTappingPipeline,
        "hand_open_close": HandOpenClosePipeline,
        "hand_up_down": HandUpDownPipeline,
    }

    # Mapping from task_name → (subtask_list, file_separator)
    # Used to locate per-patient distances CSVs in the standard booth directory layout.
    TASK_FILE_INFO = {
        "finger_tapping": (["right", "left"], "_finger"),
        "hand_open_close": (["right", "left"], "_open"),
        "hand_up_down": (["right", "left"], "_up"),
    }

    def __init__(
        self,
        model_paths: Optional[Dict[str, Path]] = None,
        model_version: str = "latest",
    ):
        model_paths = model_paths or {}
        self._pipelines: Dict[str, object] = {
            task: cls(
                model_path=model_paths.get(task),
                model_version=model_version,
            )
            for task, cls in self.TASK_PIPELINE_MAP.items()
        }

    # ------------------------------------------------------------------
    # From pre-computed distances CSVs
    # ------------------------------------------------------------------

    def run_from_csvs(
        self,
        patient_ids: List[str],
        distances_dir: Path,
        tasks: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Run inference for all patients using pre-computed distances CSVs.

        Expects the booth layout::

            distances_dir/
              finger_tapping/right/distances/<patient_id>_finger_distances.csv
              finger_tapping/left/distances/<patient_id>_finger_distances.csv
              hand_open_close/right/distances/<patient_id>_open_distances.csv
              …

        Parameters
        ----------
        patient_ids : list of str
        distances_dir : Path
            Root of the processed data (BASE_PROCESSED_DIRECTORY).
        tasks : list of str, optional
            Subset of tasks to run. Defaults to all three.

        Returns
        -------
        pd.DataFrame  One row per (patient_id, task, subtask).
        """
        distances_dir = Path(distances_dir)
        tasks = tasks or list(self.TASK_PIPELINE_MAP.keys())
        rows = []

        for task_name in tasks:
            pipeline = self._pipelines[task_name]
            subtasks, sep = self.TASK_FILE_INFO[task_name]

            for patient_id in patient_ids:
                for subtask in subtasks:
                    # Construct expected path
                    csv_path = (
                        distances_dir
                        / task_name
                        / subtask
                        / "distances"
                        / f"{patient_id}{sep}_distances.csv"
                    )

                    result: Optional[InferenceResult] = pipeline.run_from_csv(
                        patient_id=f"{patient_id}_{subtask}",
                        distances_csv=csv_path,
                    )

                    if result is not None:
                        row = result.as_dict()
                        row["task"] = task_name
                        row["subtask"] = subtask
                        rows.append(row)
                    else:
                        rows.append({
                            "patient_id": f"{patient_id}_{subtask}",
                            "task": task_name,
                            "subtask": subtask,
                            "severity": None,
                            "raw_sequence_length": 0,
                        })

        df = pd.DataFrame(rows)
        cols = ["patient_id", "task", "subtask", "severity", "raw_sequence_length"]
        extra = [c for c in df.columns if c not in cols]
        return df[cols + extra]

    # ------------------------------------------------------------------
    # From raw videos
    # ------------------------------------------------------------------

    def run_from_videos(
        self,
        patient_ids: List[str],
        raw_video_dir: Path,
        processed_dir: Path,
        tasks: Optional[List[str]] = None,
        video_width: int = 1920,
        video_height: int = 1080,
    ) -> pd.DataFrame:
        """
        Full pipeline: video → pose → distances → inference.

        Expects videos at::

            raw_video_dir/<patient_id>/finger_tapping/left_finger_tapping.mp4
            raw_video_dir/<patient_id>/finger_tapping/right_finger_tapping.mp4
            …

        Intermediate files are written under *processed_dir*.

        Returns
        -------
        pd.DataFrame  One row per (patient_id, task, subtask).
        """
        from portal_analysis.config import TASKS

        raw_video_dir = Path(raw_video_dir)
        processed_dir = Path(processed_dir)
        tasks = tasks or list(self.TASK_PIPELINE_MAP.keys())
        rows = []

        # Map task_name → subtask → video filename
        video_map = {
            "finger_tapping": {
                "right": "right_finger_tapping.mp4",
                "left": "left_finger_tapping.mp4",
            },
            "hand_open_close": {
                "right": "right_open_close.mp4",
                "left": "left_open_close.mp4",
            },
            "hand_up_down": {
                "right": "right_up_down.mp4",
                "left": "left_up_down.mp4",
            },
        }

        for patient_id in patient_ids:
            for task_name in tasks:
                pipeline = self._pipelines[task_name]
                for subtask, video_file in video_map[task_name].items():
                    video_path = raw_video_dir / patient_id / task_name / video_file
                    pose_dir = processed_dir / task_name / subtask / "pose"
                    dist_dir = processed_dir / task_name / subtask / "distances"

                    result = pipeline.run_from_video(
                        patient_id=f"{patient_id}_{subtask}",
                        video_path=video_path,
                        pose_output_dir=pose_dir,
                        distances_output_dir=dist_dir,
                        video_width=video_width,
                        video_height=video_height,
                    )

                    if result is not None:
                        row = result.as_dict()
                        row["task"] = task_name
                        row["subtask"] = subtask
                    else:
                        row = {
                            "patient_id": f"{patient_id}_{subtask}",
                            "task": task_name,
                            "subtask": subtask,
                            "severity": None,
                            "raw_sequence_length": 0,
                        }
                    rows.append(row)

        df = pd.DataFrame(rows)
        cols = ["patient_id", "task", "subtask", "severity", "raw_sequence_length"]
        extra = [c for c in df.columns if c not in cols]
        return df[cols + extra]

    # ------------------------------------------------------------------
    # Convenience: save results
    # ------------------------------------------------------------------

    @staticmethod
    def save_results(df: pd.DataFrame, output_path: Path) -> None:
        """Save results DataFrame to CSV."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"Results saved → {output_path}")
