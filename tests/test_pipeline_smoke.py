"""Smoke test: fit, save, load, predict without network data."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from portal_analysis.training.artifact import load_artifact_bundle, save_artifact_bundle
from portal_analysis.training.pipeline import fit_pipeline, predict_sequences
from portal_analysis.training.task_config import TaskConfig


@pytest.fixture
def task_config():
    return TaskConfig(
        task_name="smoke_test",
        data_column_name="x",
        file_id_separator="_x",
        n_kernels=100,
        classifier_cv=3,
        include_fft=False,
        include_diffs=False,
        rocket_augment_data=True,
    )


def test_fit_save_load_predict(task_config):
    rng = np.random.default_rng(42)
    X_train = rng.normal(size=(12, 50)).astype(np.float32)
    y_train = np.repeat([0, 1, 2], 4)

    augmenter, rocket, classifier = fit_pipeline(X_train, y_train, task_config)
    assert hasattr(classifier, "predict_proba")

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "model"
        save_artifact_bundle(
            out,
            augmenter,
            rocket,
            classifier,
            task_config,
            version="test",
            dataset={"n_train": 12, "n_test": 3, "n_evaluated": 3},
        )
        from portal_analysis.training.artifact import ArtifactBundle

        bundle = load_artifact_bundle(out)
        assert bundle.metadata["dataset"] == {
            "n_train": 12,
            "n_test": 3,
            "n_evaluated": 3,
        }

    X_test = rng.normal(size=(3, 50)).astype(np.float32)
    preds = predict_sequences(bundle, X_test)
    probabilities = bundle.classifier.predict_proba(bundle.rocket.transform(X_test))
    assert preds.shape == (3,)
    assert probabilities.shape == (3, 4)
    assert np.allclose(probabilities.sum(axis=1), 1.0)
