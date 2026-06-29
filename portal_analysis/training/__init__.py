"""Training pipeline, artifacts, and task configuration."""

from portal_analysis.training.artifact import ArtifactBundle, load_artifact_bundle, save_artifact_bundle
from portal_analysis.training.task_config import TaskConfig

__all__ = [
    "ArtifactBundle",
    "TaskConfig",
    "evaluate_saved_model",
    "load_artifact_bundle",
    "predict_from_model",
    "predict_sequences",
    "save_artifact_bundle",
    "train_pipeline",
]


def __getattr__(name):
    if name in {
        "evaluate_saved_model",
        "predict_from_model",
        "predict_sequences",
        "train_pipeline",
    }:
        from portal_analysis.training import pipeline

        return getattr(pipeline, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
