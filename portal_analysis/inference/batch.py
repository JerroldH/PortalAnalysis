"""
Batch inference pipeline: runs all three hand movement tasks for one or more patients
and consolidates results into a single DataFrame / CSV report.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd

from portal_analysis.inference.symptoms import CANONICAL_SYMPTOMS

from portal_analysis.inference.base import InferenceResult
from portal_analysis.inference.finger_tapping import FingerTappingPipeline
from portal_analysis.inference.hand_open_close import HandOpenClosePipeline
from portal_analysis.inference.hand_up_down import HandUpDownPipeline
from portal_analysis.inference.both_still import BothStillPipeline
from portal_analysis.inference.symptoms import load_symptom_models_for_tasks
from portal_analysis.models.model_manager import HandMovementModel


@dataclass(frozen=True)
class VideoInferenceEntry:
    """One video file to run through the full inference pipeline."""

    patient_id: str
    task_name: str
    subtask: str
    video_path: Path


@dataclass(frozen=True)
class DistancesInferenceEntry:
    """One pre-computed distances CSV for inference."""

    patient_id: str
    task_name: str
    subtask: str
    distances_csv: Path


@dataclass(frozen=True)
class PoseInferenceEntry:
    """One MediaPipe pose CSV (pose → distances → inference)."""

    patient_id: str
    task_name: str
    subtask: str
    pose_csv: Path


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
        batch.save_results(results_df, Path("N:/Booth_Processed"))
    """

    TASK_PIPELINE_MAP = {
        "finger_tapping": FingerTappingPipeline,
        "hand_open_close": HandOpenClosePipeline,
        "hand_up_down": HandUpDownPipeline,
        "both_still": BothStillPipeline,
    }

    # Tasks whose pose CSVs are converted to distances before inference.
    TASKS_FROM_POSE = frozenset({
        "finger_tapping",
        "hand_open_close",
        "hand_up_down",
        "both_still",
    })

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
        "both_still": {
            "left": "both_still",
            "right": "both_still",
        },
    }

    def __init__(
        self,
        model_paths: Optional[Dict[str, Path]] = None,
        model_version: str = "latest",
        with_symptoms: bool = False,
        models_root: Optional[Path] = None,
    ):
        model_paths = model_paths or {}
        self._model_version = model_version
        self._with_symptoms = with_symptoms
        self._pipelines: Dict[str, object] = {
            task: cls(
                model_path=model_paths.get(task),
                model_version=model_version,
            )
            for task, cls in self.TASK_PIPELINE_MAP.items()
        }
        self._symptom_models: Dict[str, Dict[str, HandMovementModel]] = {}
        if with_symptoms:
            self._symptom_models = load_symptom_models_for_tasks(
                version=model_version,
                models_root=models_root,
            )
            for task_name, models in self._symptom_models.items():
                if models:
                    print(
                        f"[batch] Loaded {len(models)} symptom model(s) for {task_name}"
                    )
                else:
                    print(
                        f"[batch] Warning: no symptom models found for {task_name} "
                        f"(version={model_version}). Train with: "
                        f"python -m portal_analysis.cli train-symptoms --tasks {task_name}"
                    )

    def _symptom_models_for(self, task_name: str) -> Optional[Dict[str, HandMovementModel]]:
        if not self._with_symptoms:
            return None
        models = self._symptom_models.get(task_name, {})
        return models if models else None

    @staticmethod
    def results_dir(processed_dir: Path) -> Path:
        """Root folder for inference outputs (JSON scores, pose, distances, plots)."""
        return Path(processed_dir) / "results"

    @staticmethod
    def recording_id(patient_id: str, video_stem: str) -> str:
        """Recording key matching pose/distances/plots: ``{patient_id}_{video_stem}``."""
        return f"{patient_id}_{video_stem}"

    @classmethod
    def _patient_recording_id(
        cls,
        patient_id: str,
        task_name: str,
        subtask: str,
    ) -> str:
        """Per-recording id for JSON and plots (both_still uses per-hand stem)."""
        if task_name == "both_still":
            return BothStillPipeline.hand_distances_stem(patient_id, subtask)
        return cls.recording_id(patient_id, cls.VIDEO_STEMS[task_name][subtask])

    @classmethod
    def inference_output_dir(
        cls,
        processed_dir: Path,
        output_dir: Optional[Path] = None,
    ) -> Path:
        """Directory for per-recording JSON scores (default: ``results/inference/``)."""
        if output_dir is not None:
            return Path(output_dir)
        return cls.results_dir(processed_dir) / "inference"

    @classmethod
    def inference_json_path(
        cls,
        processed_dir: Path,
        recording_id: str,
        output_dir: Optional[Path] = None,
    ) -> Path:
        """``results/inference/<recording_id>_inference.json``."""
        return cls.inference_output_dir(processed_dir, output_dir) / (
            f"{recording_id}_inference.json"
        )

    @staticmethod
    def _mapping_value(value: Any) -> Dict[str, Any]:
        if isinstance(value, dict):
            return value
        if isinstance(value, str) and value:
            try:
                parsed = json.loads(value)
            except json.JSONDecodeError:
                return {}
            return parsed if isinstance(parsed, dict) else {}
        return {}

    @classmethod
    def row_to_inference_json(cls, row: Dict[str, Any]) -> Dict[str, Any]:
        """Build JSON object for one recording; nest symptoms when present."""
        severity = row.get("severity")
        if severity is not None and pd.notna(severity):
            severity = int(severity)

        payload: Dict[str, Any] = {
            "patient_id": row["patient_id"],
            "task": row["task"],
            "subtask": row["subtask"],
            "severity": None if pd.isna(severity) else severity,
            "raw_sequence_length": int(row.get("raw_sequence_length") or 0),
        }
        confidence = row.get("confidence")
        if confidence is not None and pd.notna(confidence):
            payload["confidence"] = float(confidence)

        quality_status = row.get("quality_status")
        if quality_status is not None and pd.notna(quality_status):
            payload["quality_status"] = str(quality_status)

        for key in (
            "current_model_name",
            "current_expected_score",
            "ordinal_evidence_score",
            "reference_model_name",
            "reference_prediction",
            "model_agreement",
            "agreement_delta",
            "agreement_weight",
            "quality_weight",
            "evidence_weight",
        ):
            value = row.get(key)
            if value is not None and pd.notna(value):
                payload[key] = value

        for key in (
            "severity_probabilities",
            "quality",
            "clinical_features",
            "artifacts",
        ):
            value = cls._mapping_value(row.get(key))
            if value:
                payload[key] = value

        symptoms = {
            key: int(row[key])
            for key in CANONICAL_SYMPTOMS
            if key in row and row[key] is not None and pd.notna(row[key])
        }
        if symptoms:
            payload["symptoms"] = symptoms
        return payload

    @classmethod
    def _pose_csv_path(
        cls,
        processed_dir: Path,
        task_name: str,
        subtask: str,
        patient_id: str,
        video_stem: str,
    ) -> Path:
        del task_name, subtask  # encoded in video_stem
        return cls.results_dir(processed_dir) / "pose" / f"{patient_id}_{video_stem}.csv"

    @classmethod
    def _legacy_pose_csv_path(
        cls,
        processed_dir: Path,
        task_name: str,
        subtask: str,
        patient_id: str,
        video_stem: str,
    ) -> Path:
        return (
            Path(processed_dir)
            / task_name
            / subtask
            / "pose"
            / f"{patient_id}_{video_stem}.csv"
        )

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
    def _recording_stem(
        cls,
        task_name: str,
        patient_id: str,
        subtask: str,
    ) -> str:
        """Stem used for JSON / plots / per-hand tremor CSV names."""
        if task_name == "both_still":
            return BothStillPipeline.hand_distances_stem(patient_id, subtask)
        return cls.VIDEO_STEMS[task_name][subtask]

    @classmethod
    def resolve_task_subtask_from_stem(cls, stem: str) -> Optional[Tuple[str, str]]:
        """Map a filename stem (no extension) to (task_name, subtask)."""
        if stem.endswith("_both_still_left_hand"):
            return "both_still", "left"
        if stem.endswith("_both_still_right_hand"):
            return "both_still", "right"
        if stem == "both_still" or (
            stem.endswith("_both_still")
            and not stem.endswith("_left_hand")
            and not stem.endswith("_right_hand")
        ):
            return "both_still", "both"

        for task_name, stems in cls.VIDEO_STEMS.items():
            for subtask, video_stem in stems.items():
                if stem == video_stem or stem.endswith(f"_{video_stem}"):
                    return task_name, subtask
        return None

    @staticmethod
    def _stem_for_resolution(stem: str, *, strip_suffix: Optional[str] = None) -> str:
        if strip_suffix and stem.endswith(strip_suffix):
            return stem[: -len(strip_suffix)]
        return stem

    @classmethod
    def _entries_from_named_paths(
        cls,
        patient_id: str,
        paths: List[Path],
        *,
        tasks: Optional[List[str]] = None,
        hands: str = "both",
        task: Optional[str] = None,
        stem_suffix: Optional[str] = None,
        file_label: str = "file",
    ) -> List[Tuple[str, str, Path]]:
        """
        Shared logic for explicit file paths (video, pose, or distances).

        Returns (task_name, subtask, path) tuples.
        """
        if task is not None:
            if task not in cls.TASK_PIPELINE_MAP:
                raise ValueError(
                    f"Unknown task {task!r}. "
                    f"Choose from: {', '.join(cls.TASK_PIPELINE_MAP)}."
                )
            if hands == "both" and task != "both_still":
                raise ValueError(
                    f"With an explicit task, set --hand to left or right "
                    f"(one {file_label} corresponds to one hand)."
                )
            if tasks is not None and task not in tasks:
                return []
            if task == "both_still" and hands == "both":
                return [
                    (task, subtask, Path(path))
                    for path in paths
                    for subtask in cls.normalize_hands(hands)
                ]
            subtask = hands
            return [(task, subtask, Path(path)) for path in paths]

        entries: List[Tuple[str, str, Path]] = []
        for path in paths:
            path = Path(path)
            stem = cls._stem_for_resolution(path.stem, strip_suffix=stem_suffix)
            resolved = cls.resolve_task_subtask_from_stem(stem)
            if resolved is None:
                raise ValueError(
                    f"Cannot infer task/subtask from {file_label} name '{path.name}'. "
                    f"Use a known stem such as right_finger_tapping, left_open_close, …, "
                    f"or pass --task and --hand left|right."
                )
            task_name, subtask = resolved
            if tasks is not None and task_name not in tasks:
                continue
            if task_name == "both_still" and subtask == "both":
                for side in cls.normalize_hands(hands):
                    entries.append((task_name, side, path))
                continue
            if subtask not in cls.normalize_hands(hands):
                continue
            entries.append((task_name, subtask, path))
        return entries

    @classmethod
    def entries_from_distances_paths(
        cls,
        patient_id: str,
        distances_paths: List[Path],
        tasks: Optional[List[str]] = None,
        hands: str = "both",
        task: Optional[str] = None,
    ) -> List[DistancesInferenceEntry]:
        """
        Build inference entries from explicit distances CSV paths.

        If *task* is set, use it with *hands* (``left`` or ``right``) for every
        file. Otherwise task and subtask are inferred from each file's stem
        (e.g. ``right_finger_tapping_distances.csv`` or
        ``SUBJECT_001_right_finger_tapping_distances.csv``).
        """
        tuples = cls._entries_from_named_paths(
            patient_id,
            distances_paths,
            tasks=tasks,
            hands=hands,
            task=task,
            stem_suffix="_distances",
            file_label="distances CSV",
        )
        return [
            DistancesInferenceEntry(
                patient_id=patient_id,
                task_name=task_name,
                subtask=subtask,
                distances_csv=path,
            )
            for task_name, subtask, path in tuples
        ]

    @classmethod
    def entries_from_pose_paths(
        cls,
        patient_id: str,
        pose_paths: List[Path],
        tasks: Optional[List[str]] = None,
        hands: str = "both",
        task: Optional[str] = None,
    ) -> List[PoseInferenceEntry]:
        """
        Build inference entries from explicit pose CSV paths.

        If *task* is set, use it with *hands* for every file. Otherwise task
        and subtask are inferred from each file's stem (e.g.
        ``right_finger_tapping.csv``).
        """
        tuples = cls._entries_from_named_paths(
            patient_id,
            pose_paths,
            tasks=tasks,
            hands=hands,
            task=task,
            file_label="pose CSV",
        )
        return [
            PoseInferenceEntry(
                patient_id=patient_id,
                task_name=task_name,
                subtask=subtask,
                pose_csv=path,
            )
            for task_name, subtask, path in tuples
        ]

    @classmethod
    def entries_from_video_paths(
        cls,
        patient_id: str,
        video_paths: List[Path],
        tasks: Optional[List[str]] = None,
        hands: str = "both",
        task: Optional[str] = None,
    ) -> List[VideoInferenceEntry]:
        """
        Build inference entries from explicit MP4 paths.

        If *task* is set, use it with *hands* (``left`` or ``right``) for every
        video. Otherwise task and subtask are inferred from each file's stem via
        VIDEO_STEMS (e.g. ``right_finger_tapping.mp4``).
        """
        tuples = cls._entries_from_named_paths(
            patient_id,
            video_paths,
            tasks=tasks,
            hands=hands,
            task=task,
            file_label="video",
        )
        return [
            VideoInferenceEntry(
                patient_id=patient_id,
                task_name=task_name,
                subtask=subtask,
                video_path=path,
            )
            for task_name, subtask, path in tuples
        ]

    @classmethod
    def _distances_csv_path(
        cls,
        processed_dir: Path,
        task_name: str,
        subtask: str,
        patient_id: str,
        video_stem: str,
    ) -> Path:
        if task_name == "both_still":
            return BothStillPipeline.distances_csv_path(
                processed_dir, patient_id, subtask, prefer_results=True
            )
        del task_name, subtask
        return (
            cls.results_dir(processed_dir)
            / "distances"
            / f"{patient_id}_{video_stem}_distances.csv"
        )

    @classmethod
    def _legacy_distances_csv_path(
        cls,
        processed_dir: Path,
        task_name: str,
        subtask: str,
        patient_id: str,
        video_stem: str,
    ) -> Path:
        if task_name == "both_still":
            return (
                Path(processed_dir)
                / task_name
                / subtask
                / "distances"
                / f"{BothStillPipeline.hand_distances_stem(patient_id, subtask)}.csv"
            )
        return (
            Path(processed_dir)
            / task_name
            / subtask
            / "distances"
            / f"{patient_id}_{video_stem}_distances.csv"
        )

    @classmethod
    def _resolve_distances_csv_path(
        cls,
        processed_dir: Path,
        task_name: str,
        subtask: str,
        patient_id: str,
        video_stem: str,
    ) -> Path:
        """Prefer ``results/distances/``; fall back to legacy booth layout for reads."""
        results_path = cls._distances_csv_path(
            processed_dir, task_name, subtask, patient_id, video_stem
        )
        if results_path.exists():
            return results_path
        legacy = cls._legacy_distances_csv_path(
            processed_dir, task_name, subtask, patient_id, video_stem
        )
        return legacy if legacy.exists() else results_path

    @classmethod
    def _resolve_pose_csv_path(
        cls,
        processed_dir: Path,
        task_name: str,
        subtask: str,
        patient_id: str,
        video_stem: str,
    ) -> Path:
        results_path = cls._pose_csv_path(
            processed_dir, task_name, subtask, patient_id, video_stem
        )
        if results_path.exists():
            return results_path
        legacy = cls._legacy_pose_csv_path(
            processed_dir, task_name, subtask, patient_id, video_stem
        )
        return legacy if legacy.exists() else results_path

    @classmethod
    def _plot_png_path(
        cls,
        processed_dir: Path,
        patient_id: str,
        video_stem: str,
        *,
        task_name: Optional[str] = None,
        subtask: Optional[str] = None,
    ) -> Path:
        if task_name == "both_still" and subtask is not None:
            stem = BothStillPipeline.hand_distances_stem(patient_id, subtask)
            return cls.results_dir(processed_dir) / "plots" / f"{stem}_distances.png"
        return (
            cls.results_dir(processed_dir)
            / "plots"
            / f"{patient_id}_{video_stem}_distances.png"
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

        Reads distances from ``results/distances/`` when present, otherwise the
        legacy booth layout under each task/subtask. Kinematic plots are written
        to ``results/plots/``.

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
                    csv_path = self._resolve_distances_csv_path(
                        distances_dir, task_name, subtask, patient_id, video_stem
                    )
                    plot_path = self._plot_png_path(
                        distances_dir,
                        patient_id,
                        video_stem,
                        task_name=task_name,
                        subtask=subtask,
                    )

                    recording_id = self._patient_recording_id(
                        patient_id, task_name, subtask
                    )
                    result: Optional[InferenceResult] = pipeline.run_from_csv(
                        patient_id=recording_id,
                        distances_csv=csv_path,
                        symptom_models=self._symptom_models_for(task_name),
                        plot_path=plot_path,
                    )

                    if result is not None:
                        row = result.as_dict()
                        row["task"] = task_name
                        row["subtask"] = subtask
                        rows.append(row)
                    else:
                        rows.append({
                            "patient_id": recording_id,
                            "task": task_name,
                            "subtask": subtask,
                            "severity": None,
                            "raw_sequence_length": 0,
                        })

        df = pd.DataFrame(rows)
        cols = ["patient_id", "task", "subtask", "severity", "raw_sequence_length"]
        extra = [c for c in df.columns if c not in cols]
        return df[cols + extra]

    def run_from_distances_paths(
        self,
        entries: List[DistancesInferenceEntry],
        processed_dir: Optional[Path] = None,
    ) -> pd.DataFrame:
        """
        Run inference from explicit distances CSV paths.

        Parameters
        ----------
        entries : list of DistancesInferenceEntry
            Each entry specifies patient_id, task, subtask, and the distances CSV.
        """
        rows = []

        for entry in entries:
            pipeline = self._pipelines[entry.task_name]
            video_stem = self.VIDEO_STEMS[entry.task_name][entry.subtask]
            recording_id = self._patient_recording_id(
                entry.patient_id, entry.task_name, entry.subtask
            )
            plot_path = None
            if processed_dir is not None:
                plot_path = self._plot_png_path(
                    processed_dir,
                    entry.patient_id,
                    video_stem,
                    task_name=entry.task_name,
                    subtask=entry.subtask,
                )
            result = pipeline.run_from_csv(
                patient_id=recording_id,
                distances_csv=entry.distances_csv,
                symptom_models=self._symptom_models_for(entry.task_name),
                plot_path=plot_path,
            )

            if result is not None:
                row = result.as_dict()
                row["task"] = entry.task_name
                row["subtask"] = entry.subtask
            else:
                row = {
                    "patient_id": recording_id,
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

        Reads pose from ``results/pose/`` or the legacy task/subtask layout.
        Writes distances and plots under ``results/distances/`` and
        ``results/plots/``. All three inference tasks are converted from pose.
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
                    pose_path = self._resolve_pose_csv_path(
                        processed_dir, task_name, subtask, patient_id, video_stem
                    )
                    dist_path = self._distances_csv_path(
                        processed_dir, task_name, subtask, patient_id, video_stem
                    )
                    plot_path = self._plot_png_path(
                        processed_dir,
                        patient_id,
                        video_stem,
                        task_name=task_name,
                        subtask=subtask,
                    )

                    recording_id = self._patient_recording_id(
                        patient_id, task_name, subtask
                    )
                    pose_kwargs = dict(
                        patient_id=patient_id,
                        pose_csv=pose_path,
                        distances_csv=dist_path,
                        symptom_models=self._symptom_models_for(task_name),
                        video_width=video_width,
                        video_height=video_height,
                        plot_path=plot_path,
                    )
                    if task_name == "both_still":
                        pose_kwargs["subtask"] = subtask
                        pose_kwargs["processed_dir"] = processed_dir
                    result: Optional[InferenceResult] = pipeline.run_from_pose(
                        **pose_kwargs
                    )
                    if result is not None and task_name == "both_still":
                        result.patient_id = recording_id

                    if result is not None:
                        row = result.as_dict()
                        row["task"] = task_name
                        row["subtask"] = subtask
                        rows.append(row)
                    else:
                        rows.append({
                            "patient_id": recording_id,
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

    def run_from_pose_paths(
        self,
        entries: List[PoseInferenceEntry],
        processed_dir: Path,
        video_width: int = 1920,
        video_height: int = 1080,
    ) -> pd.DataFrame:
        """
        Run inference from explicit pose CSV paths (pose → distances → severity).

        Distances CSVs and plots are written under *processed_dir/results/*.
        All three inference tasks are supported from pose.
        """
        processed_dir = Path(processed_dir)
        rows = []

        for entry in entries:
            if entry.task_name not in self.TASKS_FROM_POSE:
                print(
                    f"[batch] Skipping {entry.pose_csv.name}: pose mode only supports "
                    f"{sorted(self.TASKS_FROM_POSE)}."
                )
                continue

            pipeline = self._pipelines[entry.task_name]
            video_stem = self.VIDEO_STEMS[entry.task_name][entry.subtask]
            dist_path = self._distances_csv_path(
                processed_dir,
                entry.task_name,
                entry.subtask,
                entry.patient_id,
                video_stem,
            )
            plot_path = self._plot_png_path(
                processed_dir,
                entry.patient_id,
                video_stem,
                task_name=entry.task_name,
                subtask=entry.subtask,
            )
            recording_id = self._patient_recording_id(
                entry.patient_id, entry.task_name, entry.subtask
            )

            pose_kwargs = dict(
                patient_id=entry.patient_id,
                pose_csv=entry.pose_csv,
                distances_csv=dist_path,
                symptom_models=self._symptom_models_for(entry.task_name),
                video_width=video_width,
                video_height=video_height,
                plot_path=plot_path,
            )
            if entry.task_name == "both_still":
                pose_kwargs["subtask"] = entry.subtask
                pose_kwargs["processed_dir"] = processed_dir
            result = pipeline.run_from_pose(**pose_kwargs)
            if result is not None and entry.task_name == "both_still":
                result.patient_id = recording_id

            if result is not None:
                row = result.as_dict()
                row["task"] = entry.task_name
                row["subtask"] = entry.subtask
            else:
                row = {
                    "patient_id": recording_id,
                    "task": entry.task_name,
                    "subtask": entry.subtask,
                    "severity": None,
                    "raw_sequence_length": 0,
                }
            rows.append(row)

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
            Root for intermediate outputs under ``results/pose`` and
            ``results/distances``.
        """
        processed_dir = Path(processed_dir)
        rows = []

        for entry in entries:
            pipeline = self._pipelines[entry.task_name]
            results_root = self.results_dir(processed_dir)
            pose_dir = results_root / "pose"
            dist_dir = results_root / "distances"
            video_stem = self.VIDEO_STEMS[entry.task_name][entry.subtask]
            plot_path = self._plot_png_path(
                processed_dir,
                entry.patient_id,
                video_stem,
                task_name=entry.task_name,
                subtask=entry.subtask,
            )
            recording_id = self._patient_recording_id(
                entry.patient_id, entry.task_name, entry.subtask
            )

            video_kwargs = dict(
                patient_id=entry.patient_id,
                video_path=entry.video_path,
                pose_output_dir=pose_dir,
                distances_output_dir=dist_dir,
                symptom_models=self._symptom_models_for(entry.task_name),
                file_prefix=entry.patient_id,
                video_width=video_width,
                video_height=video_height,
                plot_path=plot_path,
            )
            if entry.task_name == "both_still":
                video_kwargs["subtask"] = entry.subtask
                video_kwargs["processed_dir"] = processed_dir
            result = pipeline.run_from_video(**video_kwargs)
            if result is not None:
                result.patient_id = recording_id

            if result is not None:
                row = result.as_dict()
                row["task"] = entry.task_name
                row["subtask"] = entry.subtask
            else:
                row = {
                    "patient_id": recording_id,
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

        Intermediate files are written under *processed_dir/results/*.

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
                    results_root = self.results_dir(processed_dir)
                    pose_dir = results_root / "pose"
                    dist_dir = results_root / "distances"
                    plot_path = self._plot_png_path(
                        processed_dir,
                        patient_id,
                        video_stem,
                        task_name=task_name,
                        subtask=subtask,
                    )

                    recording_id = self._patient_recording_id(
                        patient_id, task_name, subtask
                    )
                    video_kwargs = dict(
                        patient_id=patient_id,
                        video_path=video_path,
                        pose_output_dir=pose_dir,
                        distances_output_dir=dist_dir,
                        symptom_models=self._symptom_models_for(task_name),
                        file_prefix=patient_id,
                        video_width=video_width,
                        video_height=video_height,
                        plot_path=plot_path,
                    )
                    if task_name == "both_still":
                        video_kwargs["subtask"] = subtask
                        video_kwargs["processed_dir"] = processed_dir
                    result = pipeline.run_from_video(**video_kwargs)
                    if result is not None:
                        result.patient_id = recording_id

                    if result is not None:
                        row = result.as_dict()
                        row["task"] = task_name
                        row["subtask"] = subtask
                    else:
                        row = {
                            "patient_id": recording_id,
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

    @classmethod
    def save_results(
        cls,
        df: pd.DataFrame,
        processed_dir: Path,
        output_dir: Optional[Path] = None,
    ) -> List[Path]:
        """
        Write one JSON file per recording: ``<recording_id>_inference.json``.

        Symptoms are nested under ``symptoms`` when ``--with-symptoms`` produced them.
        """
        if df.empty or "patient_id" not in df.columns:
            return []

        out_dir = cls.inference_output_dir(processed_dir, output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        written: List[Path] = []

        for row in df.to_dict(orient="records"):
            recording_id = row["patient_id"]
            path = out_dir / f"{recording_id}_inference.json"
            payload = cls.row_to_inference_json(row)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
                f.write("\n")
            written.append(path)

        print(f"Inference JSON -> {out_dir} ({len(written)} file(s))")
        return written
