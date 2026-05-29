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
                "raw_sequence_length": 100,
                "amplitude_reduction": 0,
                "slowness": 1,
            }
        ]
    )
    written = BatchInferencePipeline.save_results(df, tmp_path)
    assert len(written) == 1
    payload = json.loads(written[0].read_text(encoding="utf-8"))
    assert payload["severity"] == 2
    assert payload["symptoms"] == {"amplitude_reduction": 0, "slowness": 1}
    assert "sequence_effect" not in payload["symptoms"]


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
