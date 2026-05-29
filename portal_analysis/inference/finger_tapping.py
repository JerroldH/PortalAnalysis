"""
Inference pipeline for the Finger Tapping task.

Severity classes (MDS-UPDRS Part III, item 3.4):
    0 = Normal, 1 = Slight, 2 = Mild, 3 = Moderate/Severe

Symptoms predicted (with ``--with-symptoms``):
    amplitude_reduction, sequence_effect, slowness, halt_hesitation
"""

from portal_analysis.inference.base import BaseInferencePipeline


class FingerTappingPipeline(BaseInferencePipeline):
    """
    End-to-end inference for finger tapping recordings.

    Quick start::

        pipeline = FingerTappingPipeline()
        result = pipeline.run_from_pose(
            "P001_right",
            Path(".../pose/P001_right_finger_tapping.csv"),
        )
        print(result.severity, result.symptoms)
    """

    TASK_NAME = "finger_tapping"
    DATA_COLUMN = "Finger Normalized Distance"
    MAX_SEQUENCE_LENGTH = 450
