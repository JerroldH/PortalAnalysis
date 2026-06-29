"""Tests for clinical motor sign (symptom) inference helpers."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from portal_analysis.inference.batch import BatchInferencePipeline
from portal_analysis.inference.symptoms import (
    CANONICAL_SYMPTOMS,
    load_symptom_labels,
    symptom_sources_for_task,
)
from portal_analysis.training.task_config import TaskConfig


def test_canonical_symptoms_count():
    assert len(CANONICAL_SYMPTOMS) == 4
    for task in ("finger_tapping", "hand_open_close", "hand_up_down"):
        assert set(symptom_sources_for_task(task).keys()) == set(CANONICAL_SYMPTOMS)


def test_load_symptom_labels_or_combination(tmp_path):
    docs = tmp_path / "hand_open_close" / "docs"
    docs.mkdir(parents=True)
    pd.DataFrame(
        {
            "ID": ["a", "b", "c"],
            "Halts": [0, 1, 0],
            "Irregularity": [0, 0, 1],
        }
    ).to_csv(docs / "symptoms_open_close.csv", index=False)

    cfg = TaskConfig(
        task_name="hand_open_close",
        data_column_name="x",
        file_id_separator="_open",
        symptoms_file="symptoms_open_close.csv",
    )
    y = load_symptom_labels(cfg, "halt_hesitation", tmp_path)
    assert y.loc["a"] == 0
    assert y.loc["b"] == 1
    assert y.loc["c"] == 1


def test_batch_with_symptoms_passes_models():
    mock_models = {"finger_tapping": {"slowness": MagicMock()}}
    with patch(
        "portal_analysis.inference.batch.load_symptom_models_for_tasks",
        return_value=mock_models,
    ):
        batch = BatchInferencePipeline(with_symptoms=True)
        assert batch._symptom_models_for("finger_tapping") == mock_models["finger_tapping"]
        assert batch._symptom_models_for("hand_up_down") is None


def test_inference_result_symptom_columns():
    from portal_analysis.inference.base import InferenceResult

    result = InferenceResult(
        patient_id="P001_right_finger_tapping",
        severity=2,
        symptoms={
            "amplitude_reduction": 1,
            "sequence_effect": 0,
            "slowness": 1,
            "halt_hesitation": 0,
        },
        severity_probabilities={"0": 0.1, "1": 0.2, "2": 0.6, "3": 0.1},
        confidence=0.6,
        quality_status="VALID",
        quality={"raw_sequence_length": 100, "valid_signal_count": 100},
        clinical_features={"signal_range": 0.3},
        artifacts={"distances_csv": "distances.csv"},
    )
    d = result.as_dict()
    assert d["amplitude_reduction"] == 1
    assert d["confidence"] == 0.6
    json_payload = result.to_json_dict("finger_tapping", "right")
    assert json_payload["symptoms"]["halt_hesitation"] == 0
    assert json_payload["severity_probabilities"]["2"] == 0.6
    assert json_payload["quality_status"] == "VALID"
    assert json_payload["clinical_features"]["signal_range"] == 0.3
    assert json_payload["artifacts"]["distances_csv"] == "distances.csv"
    assert "amplitude_reduction" not in json_payload


def test_recording_id_matches_artifact_stem():
    rid = BatchInferencePipeline.recording_id("SUBJECT_001", "right_finger_tapping")
    assert rid == "SUBJECT_001_right_finger_tapping"
    assert (
        BatchInferencePipeline._distances_csv_path(
            Path("N:/Booth_Processed"),
            "finger_tapping",
            "right",
            "SUBJECT_001",
            "right_finger_tapping",
        ).stem
        == f"{rid}_distances"
    )
