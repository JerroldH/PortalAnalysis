"""Unit tests for batch inference entry resolution (video, pose, distances)."""

import json
from pathlib import Path

import pandas as pd
import pytest

from portal_analysis.inference.batch import BatchInferencePipeline


def test_distances_entries_inferred_from_stem():
    entries = BatchInferencePipeline.entries_from_distances_paths(
        patient_id="P001",
        distances_paths=[Path("right_finger_tapping_distances.csv")],
        hands="right",
    )
    assert len(entries) == 1
    assert entries[0].task_name == "finger_tapping"
    assert entries[0].subtask == "right"
    assert entries[0].distances_csv.name == "right_finger_tapping_distances.csv"


def test_distances_entries_with_subject_prefix():
    entries = BatchInferencePipeline.entries_from_distances_paths(
        patient_id="SUBJECT_001",
        distances_paths=[Path("SUBJECT_001_right_open_close_distances.csv")],
        hands="both",
    )
    assert entries[0].task_name == "hand_open_close"
    assert entries[0].subtask == "right"


def test_pose_entries_inferred_from_stem():
    entries = BatchInferencePipeline.entries_from_pose_paths(
        patient_id="P001",
        pose_paths=[Path("left_finger_tapping.csv")],
        hands="left",
    )
    assert entries[0].task_name == "finger_tapping"
    assert entries[0].subtask == "left"


def test_distances_entries_explicit_task_requires_single_hand():
    with pytest.raises(ValueError, match="left or right"):
        BatchInferencePipeline.entries_from_distances_paths(
            patient_id="P001",
            distances_paths=[Path("any_distances.csv")],
            task="finger_tapping",
            hands="both",
        )


def test_entries_explicit_task_and_hands():
    entries = BatchInferencePipeline.entries_from_video_paths(
        patient_id="subject_003",
        video_paths=[Path("FUSBG_PILOT_02_Fingertapping-L-Pre.mp4")],
        task="finger_tapping",
        hands="left",
    )
    assert len(entries) == 1
    assert entries[0].task_name == "finger_tapping"
    assert entries[0].subtask == "left"
    assert entries[0].patient_id == "subject_003"


def test_entries_inferred_from_stem():
    entries = BatchInferencePipeline.entries_from_video_paths(
        patient_id="P001",
        video_paths=[Path("right_finger_tapping.mp4")],
        hands="right",
    )
    assert entries[0].task_name == "finger_tapping"
    assert entries[0].subtask == "right"


def test_entries_explicit_task_requires_single_hand():
    with pytest.raises(ValueError, match="left or right"):
        BatchInferencePipeline.entries_from_video_paths(
            patient_id="P001",
            video_paths=[Path("any.mp4")],
            task="finger_tapping",
            hands="both",
        )


def test_per_recording_inference_json_path(tmp_path: Path):
    path = BatchInferencePipeline.inference_json_path(
        tmp_path,
        "P001_right_finger_tapping",
    )
    assert path == (
        tmp_path
        / "results"
        / "inference"
        / "P001_right_finger_tapping_inference.json"
    )


def test_save_results_writes_json_with_nested_symptoms(tmp_path: Path):
    df = pd.DataFrame(
        [
            {
                "patient_id": "P001_right_finger_tapping",
                "task": "finger_tapping",
                "subtask": "right",
                "severity": 2,
                "severity_probabilities": {"0": 0.1, "1": 0.2, "2": 0.6, "3": 0.1},
                "confidence": 0.6,
                "raw_sequence_length": 100,
                "quality_status": "VALID",
                "quality": {"raw_sequence_length": 100, "valid_signal_count": 100},
                "clinical_features": {"signal_range": 0.3},
                "artifacts": {"distances_csv": "distances.csv"},
                "amplitude_reduction": 0,
                "slowness": 1,
            }
        ]
    )
    written = BatchInferencePipeline.save_results(df, tmp_path)
    assert len(written) == 1
    payload = json.loads(written[0].read_text(encoding="utf-8"))
    assert payload["severity"] == 2
    assert payload["severity_probabilities"]["2"] == 0.6
    assert payload["confidence"] == 0.6
    assert payload["quality_status"] == "VALID"
    assert payload["quality"]["valid_signal_count"] == 100
    assert payload["clinical_features"]["signal_range"] == 0.3
    assert payload["artifacts"]["distances_csv"] == "distances.csv"
    assert payload["symptoms"] == {"amplitude_reduction": 0, "slowness": 1}
    assert "sequence_effect" not in payload["symptoms"]


def test_clinical_summary_handles_empty_signal():
    from portal_analysis.inference.finger_tapping import FingerTappingPipeline

    pipeline = FingerTappingPipeline()
    frame = pd.DataFrame({"Finger Normalized Distance": [None, None]})

    status, quality, clinical = pipeline._quality_and_clinical_features(frame)

    assert status == "TOO_SHORT"
    assert quality["valid_signal_count"] == 0
    assert clinical == {}


def test_run_from_csv_adds_evidence_fields(tmp_path: Path):
    import numpy as np

    from portal_analysis.inference.finger_tapping import FingerTappingPipeline

    distances_csv = tmp_path / "distances.csv"
    pd.DataFrame({"Finger Normalized Distance": [0.1, 0.2, 0.4, 0.3, 0.2, 0.1]}).to_csv(
        distances_csv, index=False
    )

    class Model:
        classes = np.array([0, 1, 2, 3])

        def predict(self, X):
            return np.array([2])

        def predict_proba(self, X):
            return np.array([[0.1, 0.2, 0.6, 0.1]])

    pipeline = FingerTappingPipeline()
    pipeline._model = Model()
    pipeline._save_kinematic_plot = lambda distances_csv, plot_path=None: None

    result = pipeline.run_from_csv(
        "P001_right_finger_tapping",
        distances_csv,
        plot_path=tmp_path / "plot.png",
    )

    assert result.severity == 2
    assert result.severity_probabilities["2"] == 0.6
    assert result.confidence == 0.6
    assert result.quality_status == "VALID"
    assert result.quality["valid_signal_count"] == 6
    assert result.clinical_features["signal_range"] == pytest.approx(0.3)
    assert result.artifacts["distances_csv"] == str(distances_csv)
    assert result.artifacts["plot_path"] == str(tmp_path / "plot.png")


def test_inference_artifact_paths_under_results(tmp_path: Path):
    processed = tmp_path / "Booth_Processed"
    paths = BatchInferencePipeline._distances_csv_path(
        processed,
        "finger_tapping",
        "right",
        "P001",
        "right_finger_tapping",
    )
    assert paths == (
        processed
        / "results"
        / "distances"
        / "P001_right_finger_tapping_distances.csv"
    )
    plot = BatchInferencePipeline._plot_png_path(
        processed, "P001", "right_finger_tapping"
    )
    assert plot.parent.name == "plots"
    assert plot.parent.parent.name == "results"
