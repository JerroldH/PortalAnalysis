"""Train and inference pipelines for hand movement severity classification."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import RidgeClassifierCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from portal_analysis.classification.representation_extractor import RocketRepresentationExtractor
from portal_analysis.classification.signal_augmentation import SignalAugmentation
from portal_analysis.data import TimeSeriesDataLoader
from portal_analysis.training.artifact import ArtifactBundle, load_artifact_bundle, save_artifact_bundle
from portal_analysis.training.metrics import print_results
from portal_analysis.training.settings import get_base_processed_directory
from portal_analysis.training.task_config import TaskConfig


def _build_augmenter(task_config: TaskConfig) -> SignalAugmentation:
    return SignalAugmentation(
        smooth_window_length=task_config.smooth_window_length,
        smooth_polyorder=task_config.smooth_polyorder,
        gaussian_sigma=task_config.gaussian_sigma,
        include_fft=task_config.include_fft,
        include_diffs=task_config.include_diffs,
    )


def _build_rocket(task_config: TaskConfig) -> RocketRepresentationExtractor:
    use_signal_aug = task_config.include_fft or task_config.include_diffs
    return RocketRepresentationExtractor(
        method=task_config.rocket_method,
        n_kernels=task_config.n_kernels,
        random_state=task_config.random_state,
        augment_data=task_config.rocket_augment_data if not use_signal_aug else False,
    )


class OrdinalCalibratedRidgeClassifier:
    """Ordinal threshold classifier with calibrated Ridge probabilities."""

    def __init__(self, alphas=None, class_weight=None, cv: int = 3):
        self.alphas = alphas
        self.class_weight = class_weight
        self.cv = cv
        self.classes_ = np.array([0, 1, 2, 3])
        self.thresholds_ = [1, 2, 3]
        self.models_ = []

    def fit(self, X, y):
        y = np.asarray(y)
        alphas = self.alphas if self.alphas is not None else np.logspace(-3, 3, 10)
        self.models_ = []
        for threshold in self.thresholds_:
            binary_y = (y >= threshold).astype(int)
            counts = np.bincount(binary_y, minlength=2)
            if counts.min() < self.cv:
                self.models_.append(float(binary_y.mean()))
                continue
            ridge_kwargs = {"alphas": alphas}
            if self.class_weight is not None:
                ridge_kwargs["class_weight"] = self.class_weight
            base_classifier = make_pipeline(
                StandardScaler(with_mean=False),
                RidgeClassifierCV(**ridge_kwargs),
            )
            model = CalibratedClassifierCV(
                base_classifier,
                method="sigmoid",
                cv=self.cv,
            )
            model.fit(X, binary_y)
            self.models_.append(model)
        return self

    def _cumulative_probabilities(self, X):
        columns = []
        for model in self.models_:
            if isinstance(model, float):
                columns.append(np.full(X.shape[0], model))
            else:
                positive_index = int(np.where(model.classes_ == 1)[0][0])
                columns.append(model.predict_proba(X)[:, positive_index])
        cumulative = np.clip(np.vstack(columns).T, 0.0, 1.0)
        cumulative[:, 1] = np.minimum(cumulative[:, 1], cumulative[:, 0])
        cumulative[:, 2] = np.minimum(cumulative[:, 2], cumulative[:, 1])
        return cumulative

    def predict_proba(self, X):
        cumulative = self._cumulative_probabilities(X)
        probabilities = np.empty((X.shape[0], 4))
        probabilities[:, 0] = 1.0 - cumulative[:, 0]
        probabilities[:, 1] = cumulative[:, 0] - cumulative[:, 1]
        probabilities[:, 2] = cumulative[:, 1] - cumulative[:, 2]
        probabilities[:, 3] = cumulative[:, 2]
        return np.clip(probabilities, 0.0, 1.0)

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


def _build_calibrated_classifier(task_config: TaskConfig):
    """Build the ordinal calibrated Ridge classifier used by severity models."""
    return OrdinalCalibratedRidgeClassifier(
        alphas=np.logspace(-3, 3, 10),
        class_weight=task_config.class_weight,
        cv=3,
    )


def load_training_dataset(
    task_config: TaskConfig,
    base_dir: Optional[Path] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    base_dir = base_dir or get_base_processed_directory()
    data_loader = TimeSeriesDataLoader(
        task_name=task_config.task_name,
        tasks=task_config.resolved_tasks(),
        data_column_name=task_config.data_column_name,
        file_id_separator=task_config.file_id_separator,
        file_id_strip=task_config.file_id_strip,
        data_subdirectory=task_config.data_subdirectory,
        base_dir=base_dir,
    )

    y = data_loader.load_labels(
        label_column=task_config.label_column,
        labels_file=task_config.labels_file,
        labels_subdirectory=task_config.labels_subdirectory,
        label_merge_rules=task_config.label_merge_rules,
    )

    df_combined = data_loader.load_time_series_data(y.index)
    X, y_aligned, valid_ids = data_loader.prepare_sequences(
        df_combined, y,
        maxlen=task_config.max_sequence_length,
        random_state=task_config.shuffle_random_state,
    )
    return data_loader.split_train_test(
        X,
        y_aligned,
        valid_ids,
        test_set_file=task_config.test_set_file,
        test_set_subdirectory=task_config.test_set_subdirectory,
    )


def fit_pipeline(
    X_train: np.ndarray,
    y_train: np.ndarray,
    task_config: TaskConfig,
) -> Tuple[SignalAugmentation, RocketRepresentationExtractor, Any]:
    augmenter = _build_augmenter(task_config)
    rocket = _build_rocket(task_config)
    classifier = _build_calibrated_classifier(task_config)

    use_signal_aug = task_config.include_fft or task_config.include_diffs
    if use_signal_aug:
        X_train_aug = augmenter.transform(X_train, method=task_config.augmentation_method)
    else:
        X_train_aug = X_train

    X_train_repr = rocket.fit_transform(X_train_aug)
    classifier.fit(X_train_repr, y_train)
    return augmenter, rocket, classifier


def predict_sequences(
    bundle: ArtifactBundle,
    X: np.ndarray,
    augmentation_method: Optional[str] = None,
) -> np.ndarray:
    aug_cfg = bundle.metadata.get("augmenter_config", {})
    method = augmentation_method or aug_cfg.get("augmentation_method", "simple")
    use_signal_aug = bundle.augmenter.include_fft or bundle.augmenter.include_diffs
    if use_signal_aug:
        X = bundle.augmenter.transform(X, method=method)
    X_repr = bundle.rocket.transform(X)
    return bundle.classifier.predict(X_repr)


def train_pipeline(
    task_config: TaskConfig,
    output_dir: Path,
    base_dir: Optional[Path] = None,
    version: Optional[str] = None,
    evaluate: bool = True,
) -> Dict:
    print(f"Training pipeline for task: {task_config.task_name}")

    X_train, X_test, y_train, y_test, test_ids = load_training_dataset(
        task_config, base_dir=base_dir
    )
    print(f"Train set: {len(X_train)}, Test set: {len(X_test)}")

    if len(X_train) < 10:
        raise ValueError(
            f"Not enough training samples for {task_config.task_name} ({len(X_train)})"
        )

    augmenter, rocket, classifier = fit_pipeline(X_train, y_train, task_config)

    n_train = len(X_train)
    n_test = len(X_test)

    metrics = None
    evaluated = False
    if evaluate and n_test > 0:
        bundle = ArtifactBundle(augmenter, rocket, classifier, {}, Path("."))
        y_pred = predict_sequences(
            bundle, X_test, augmentation_method=task_config.augmentation_method
        )
        metrics = print_results(y_test, y_pred, test_ids)
        evaluated = True

    artifact_path = save_artifact_bundle(
        output_dir=output_dir,
        augmenter=augmenter,
        rocket=rocket,
        classifier=classifier,
        task_config=task_config,
        metrics=metrics,
        version=version,
        dataset={
            "n_train": n_train,
            "n_test": n_test,
            "n_evaluated": n_test if evaluated else 0,
        },
    )
    print(f"Saved model artifacts to: {artifact_path}")
    return {"artifact_path": artifact_path, "metrics": metrics}


def evaluate_saved_model(
    model_dir: Path,
    base_dir: Optional[Path] = None,
) -> Dict:
    bundle = load_artifact_bundle(model_dir)
    task_config = TaskConfig.from_dict(bundle.metadata["task_config"])

    _, X_test, _, y_test, test_ids = load_training_dataset(task_config, base_dir=base_dir)
    y_pred = predict_sequences(bundle, X_test)
    return print_results(y_test, y_pred, test_ids)


def predict_from_model(model_dir: Path, X: np.ndarray) -> np.ndarray:
    return predict_sequences(load_artifact_bundle(model_dir), X)
