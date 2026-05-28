"""
Inference pipeline for the Hand Open/Close task.

Severity classes (MDS-UPDRS Part III, item 3.5):
    0 = Normal, 1 = Slight, 2 = Mild, 3 = Moderate/Severe

Symptoms predicted:
    Low Amplitude, Dec. Amp., Low speed, Halts, Irregularity
"""

from portal_analysis.inference.base import BaseInferencePipeline


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
