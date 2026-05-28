"""Train and inference pipelines for hand movement severity classification."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeClassifierCV

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


def _build_classifier(task_config: TaskConfig) -> RidgeClassifierCV:
    kwargs = {
        "cv": task_config.classifier_cv,
        "scoring": task_config.classifier_scoring,
    }
    if task_config.class_weight is not None:
        kwargs["class_weight"] = task_config.class_weight
    return RidgeClassifierCV(**kwargs)


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
        df_combined, y, maxlen=task_config.max_sequence_length
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
) -> Tuple[SignalAugmentation, RocketRepresentationExtractor, RidgeClassifierCV]:
    augmenter = _build_augmenter(task_config)
    rocket = _build_rocket(task_config)
    classifier = _build_classifier(task_config)

    use_signal_aug = task_config.include_fft or task_config.include_diffs
    if use_signal_aug:
        X_train_aug = augmenter.transform(X_train, method=task_config.augmentation_method)
    else:
        X_train_aug = X_train

    X_train_repr = rocket.fit_transform(X_train_aug)
    classifier.fit(X_train_repr, y_train)
    return augmenter, rocket, classifier


def predict_sequences(bundle: ArtifactBundle, X: np.ndarray) -> np.ndarray:
    method = bundle.metadata["augmenter_config"]["augmentation_method"]
    use_signal_aug = (
        bundle.metadata["augmenter_config"]["include_fft"]
        or bundle.metadata["augmenter_config"]["include_diffs"]
    )
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

    metrics = None
    if evaluate and len(X_test) > 0:
        bundle = ArtifactBundle(augmenter, rocket, classifier, {}, Path("."))
        y_pred = predict_sequences(bundle, X_test)
        metrics = print_results(y_test, y_pred, test_ids)

    artifact_path = save_artifact_bundle(
        output_dir=output_dir,
        augmenter=augmenter,
        rocket=rocket,
        classifier=classifier,
        task_config=task_config,
        metrics=metrics,
        version=version,
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
