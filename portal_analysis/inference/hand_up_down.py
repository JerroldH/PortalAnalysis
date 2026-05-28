"""
Inference pipeline for the Hand Pronation-Supination (Up/Down) task.

Severity classes (MDS-UPDRS Part III, item 3.6):
    0 = Normal, 1 = Slight, 2 = Mild, 3 = Moderate/Severe

Symptoms predicted:
    dec. amp., low speed, irregular, can't turn hand

Note: This task uses FFT augmentation (include_fft=True) matching the training config.
"""

from portal_analysis.inference.base import BaseInferencePipeline


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
