"""
Batch inference pipeline: runs all three hand movement tasks for one or more patients
and consolidates results into a single DataFrame / CSV report.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd

from portal_analysis.inference.base import InferenceResult
from portal_analysis.inference.finger_tapping import FingerTappingPipeline
from portal_analysis.inference.hand_open_close import HandOpenClosePipeline
from portal_analysis.inference.hand_up_down import HandUpDownPipeline


@dataclass(frozen=True)
class VideoInferenceEntry:
    """One video file to run through the full inference pipeline."""

    patient_id: str
    task_name: str
    subtask: str
    video_path: Path


class BatchInferencePipeline:
    """
    Run all hand movement inference tasks for a list of patients.

    Use run_from_csvs() for pre-computed distances, run_from_poses() for pose CSVs,
    or run_from_videos() for the full video pipeline.

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

    # Tasks whose pose CSVs are converted to distances via DistanceCalculator.
    TASKS_FROM_POSE = frozenset({"finger_tapping"})

    HAND_SIDES = ("left", "right")

    # Video stem per task/side (matches raw MP4 names without extension).
    VIDEO_STEMS = {
        "finger_tapping": {
            "right": "right_finger_tapping",
            "left": "left_finger_tapping",
        },
        "hand_open_close": {
            "right": "right_open_close",
            "left": "left_open_close",
        },
        "hand_up_down": {
            "right": "right_up_down",
            "left": "left_up_down",
        },
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

    @staticmethod
    def _pose_csv_path(
        processed_dir: Path,
        task_name: str,
        subtask: str,
        patient_id: str,
        video_stem: str,
    ) -> Path:
        return processed_dir / task_name / subtask / "pose" / f"{patient_id}_{video_stem}.csv"

    @staticmethod
    def normalize_hands(hands: str = "both") -> List[str]:
        """Return subtask side(s) to run: ``left``, ``right``, or both."""
        if hands == "both":
            return ["left", "right"]
        if hands in BatchInferencePipeline.HAND_SIDES:
            return [hands]
        raise ValueError(f"hands must be 'left', 'right', or 'both', got {hands!r}")

    @classmethod
    def _subtasks_for_hands(
        cls,
        task_name: str,
        hands: str = "both",
    ) -> List[Tuple[str, str]]:
        """(subtask, video_stem) pairs for one task, filtered by hand side."""
        sides = cls.normalize_hands(hands)
        stems = cls.VIDEO_STEMS[task_name]
        return [(subtask, stems[subtask]) for subtask in sides if subtask in stems]

    @classmethod
    def resolve_task_subtask_from_stem(cls, stem: str) -> Optional[Tuple[str, str]]:
        """Map a video filename stem (no extension) to (task_name, subtask)."""
        for task_name, stems in cls.VIDEO_STEMS.items():
            for subtask, video_stem in stems.items():
                if stem == video_stem or stem.endswith(f"_{video_stem}"):
                    return task_name, subtask
        return None

    @classmethod
    def entries_from_video_paths(
        cls,
        patient_id: str,
        video_paths: List[Path],
        tasks: Optional[List[str]] = None,
        hands: str = "both",
    ) -> List[VideoInferenceEntry]:
        """
        Build inference entries from explicit MP4 paths.

        Task and subtask are inferred from each file's stem using VIDEO_STEMS
        (e.g. ``right_finger_tapping.mp4`` or ``P001_right_finger_tapping.mp4``).
        """
        entries: List[VideoInferenceEntry] = []
        for video_path in video_paths:
            video_path = Path(video_path)
            resolved = cls.resolve_task_subtask_from_stem(video_path.stem)
            if resolved is None:
                raise ValueError(
                    f"Cannot infer task/subtask from video name '{video_path.name}'. "
                    f"Use a known stem such as right_finger_tapping, left_open_close, …"
                )
            task_name, subtask = resolved
            if tasks is not None and task_name not in tasks:
                continue
            if subtask not in cls.normalize_hands(hands):
                continue
            entries.append(
                VideoInferenceEntry(
                    patient_id=patient_id,
                    task_name=task_name,
                    subtask=subtask,
                    video_path=video_path,
                )
            )
        return entries

    @staticmethod
    def _distances_csv_path(
        processed_dir: Path,
        task_name: str,
        subtask: str,
        patient_id: str,
        video_stem: str,
    ) -> Path:
        return (
            processed_dir
            / task_name
            / subtask
            / "distances"
            / f"{patient_id}_{video_stem}_distances.csv"
        )

    # ------------------------------------------------------------------
    # From pre-computed distances CSVs
    # ------------------------------------------------------------------

    def run_from_csvs(
        self,
        patient_ids: List[str],
        distances_dir: Path,
        tasks: Optional[List[str]] = None,
        hands: str = "both",
    ) -> pd.DataFrame:
        """
        Run inference for all patients using pre-computed distances CSVs.

        Expects the booth layout::

            distances_dir/
              finger_tapping/right/distances/<patient_id>_right_finger_tapping_distances.csv
              finger_tapping/left/distances/<patient_id>_left_finger_tapping_distances.csv
              hand_open_close/right/distances/<patient_id>_right_open_close_distances.csv
              …

        Parameters
        ----------
        patient_ids : list of str
        distances_dir : Path
            Root of the processed data (BASE_PROCESSED_DIRECTORY).
        tasks : list of str, optional
            Subset of tasks to run. Defaults to all three.
        hands : str
            ``left``, ``right``, or ``both`` (default).

        Returns
        -------
        pd.DataFrame  One row per (patient_id, task, subtask).
        """
        distances_dir = Path(distances_dir)
        tasks = tasks or list(self.TASK_PIPELINE_MAP.keys())
        rows = []

        for task_name in tasks:
            pipeline = self._pipelines[task_name]

            for patient_id in patient_ids:
                for subtask, video_stem in self._subtasks_for_hands(task_name, hands):
                    csv_path = self._distances_csv_path(
                        distances_dir, task_name, subtask, patient_id, video_stem
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
    # From pre-computed pose CSVs
    # ------------------------------------------------------------------

    def run_from_poses(
        self,
        patient_ids: List[str],
        processed_dir: Path,
        tasks: Optional[List[str]] = None,
        hands: str = "both",
        video_width: int = 1920,
        video_height: int = 1080,
    ) -> pd.DataFrame:
        """
        Run inference from MediaPipe pose CSVs (pose → distances → severity).

        Expects pose files at::

            processed_dir/
              finger_tapping/right/pose/<patient_id>_right_finger_tapping.csv
              …

        Distances CSVs are written alongside under ``distances/`` and reused on
        the next run. Only **finger tapping** is converted from pose in this
        package; other tasks need pre-computed distances (use ``run_from_csvs``).
        """
        processed_dir = Path(processed_dir)
        tasks = tasks or list(self.TASK_PIPELINE_MAP.keys())
        rows = []

        for task_name in tasks:
            if task_name not in self.TASKS_FROM_POSE:
                print(
                    f"[batch] Skipping {task_name}: pose mode only supports "
                    f"{sorted(self.TASKS_FROM_POSE)}. Use --mode csv for other tasks."
                )
                continue

            pipeline = self._pipelines[task_name]

            for patient_id in patient_ids:
                for subtask, video_stem in self._subtasks_for_hands(task_name, hands):
                    pose_path = self._pose_csv_path(
                        processed_dir, task_name, subtask, patient_id, video_stem
                    )
                    dist_path = self._distances_csv_path(
                        processed_dir, task_name, subtask, patient_id, video_stem
                    )

                    result: Optional[InferenceResult] = pipeline.run_from_pose(
                        patient_id=f"{patient_id}_{subtask}",
                        pose_csv=pose_path,
                        distances_csv=dist_path,
                        video_width=video_width,
                        video_height=video_height,
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
        if df.empty:
            return pd.DataFrame(
                columns=["patient_id", "task", "subtask", "severity", "raw_sequence_length"]
            )
        cols = ["patient_id", "task", "subtask", "severity", "raw_sequence_length"]
        extra = [c for c in df.columns if c not in cols]
        return df[cols + extra]

    # ------------------------------------------------------------------
    # From raw videos
    # ------------------------------------------------------------------

    def run_from_video_paths(
        self,
        entries: List[VideoInferenceEntry],
        processed_dir: Path,
        video_width: int = 1920,
        video_height: int = 1080,
    ) -> pd.DataFrame:
        """
        Full pipeline from explicit video file paths: video → pose → distances → inference.

        Parameters
        ----------
        entries : list of VideoInferenceEntry
            Each entry specifies patient_id, task, subtask, and the MP4 path.
        processed_dir : Path
            Root for intermediate pose/distances outputs (booth layout).
        """
        processed_dir = Path(processed_dir)
        rows = []

        for entry in entries:
            pipeline = self._pipelines[entry.task_name]
            pose_dir = processed_dir / entry.task_name / entry.subtask / "pose"
            dist_dir = processed_dir / entry.task_name / entry.subtask / "distances"

            result = pipeline.run_from_video(
                patient_id=f"{entry.patient_id}_{entry.subtask}",
                video_path=entry.video_path,
                pose_output_dir=pose_dir,
                distances_output_dir=dist_dir,
                file_prefix=entry.patient_id,
                video_width=video_width,
                video_height=video_height,
            )

            if result is not None:
                row = result.as_dict()
                row["task"] = entry.task_name
                row["subtask"] = entry.subtask
            else:
                row = {
                    "patient_id": f"{entry.patient_id}_{entry.subtask}",
                    "task": entry.task_name,
                    "subtask": entry.subtask,
                    "severity": None,
                    "raw_sequence_length": 0,
                }
            rows.append(row)

        df = pd.DataFrame(rows)
        cols = ["patient_id", "task", "subtask", "severity", "raw_sequence_length"]
        extra = [c for c in df.columns if c not in cols]
        return df[cols + extra]

    def run_from_videos(
        self,
        patient_ids: List[str],
        raw_video_dir: Path,
        processed_dir: Path,
        tasks: Optional[List[str]] = None,
        hands: str = "both",
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
        raw_video_dir = Path(raw_video_dir)
        processed_dir = Path(processed_dir)
        tasks = tasks or list(self.TASK_PIPELINE_MAP.keys())
        rows = []

        for patient_id in patient_ids:
            for task_name in tasks:
                pipeline = self._pipelines[task_name]
                for subtask, video_stem in self._subtasks_for_hands(task_name, hands):
                    video_path = raw_video_dir / patient_id / task_name / f"{video_stem}.mp4"
                    pose_dir = processed_dir / task_name / subtask / "pose"
                    dist_dir = processed_dir / task_name / subtask / "distances"

                    result = pipeline.run_from_video(
                        patient_id=f"{patient_id}_{subtask}",
                        video_path=video_path,
                        pose_output_dir=pose_dir,
                        distances_output_dir=dist_dir,
                        file_prefix=patient_id,
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
