"""Train binary classifiers for clinical motor signs (symptoms) per task."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from portal_analysis.data import TimeSeriesDataLoader
from portal_analysis.inference.symptoms import (
    CANONICAL_SYMPTOMS,
    SYMPTOM_TRAINING_OVERRIDES,
    load_symptom_labels,
    symptom_sources_for_task,
)
from portal_analysis.models.paths import symptom_model_dir
from portal_analysis.training.artifact import save_artifact_bundle
from portal_analysis.training.metrics import print_results
from portal_analysis.training.pipeline import fit_pipeline, predict_sequences
from portal_analysis.training.settings import get_base_processed_directory
from portal_analysis.training.task_config import TaskConfig


def _training_ids_for_symptom(
    data_loader: TimeSeriesDataLoader,
    task_config: TaskConfig,
    y_symptom: pd.Series,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Align sequences to binary symptom labels and split train/test."""
    y_symptom = y_symptom[~y_symptom.index.duplicated(keep="first")]
    y_symptom = y_symptom.dropna()
    if y_symptom.empty:
        raise ValueError("No valid binary symptom labels.")

    df_combined = data_loader.load_time_series_data(y_symptom.index)
    X, y_aligned, valid_ids = data_loader.prepare_sequences(
        df_combined,
        y_symptom,
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


def _symptom_task_config(task_config: TaskConfig) -> TaskConfig:
    overrides = SYMPTOM_TRAINING_OVERRIDES.get(task_config.task_name, {})
    if not overrides:
        return task_config
    return TaskConfig.from_dict({**task_config.to_dict(), **overrides})


def train_symptom_model(
    task_config: TaskConfig,
    symptom_key: str,
    output_dir: Path,
    base_dir: Optional[Path] = None,
    version: Optional[str] = None,
    evaluate: bool = True,
) -> Dict:
    """Train one binary symptom classifier and save an artifact bundle."""
    task_config = _symptom_task_config(task_config)
    base_dir = base_dir or get_base_processed_directory()
    y_symptom = load_symptom_labels(task_config, symptom_key, base_dir)

    data_loader = TimeSeriesDataLoader(
        task_name=task_config.task_name,
        tasks=task_config.resolved_tasks(),
        data_column_name=task_config.data_column_name,
        file_id_separator=task_config.file_id_separator,
        file_id_strip=task_config.file_id_strip,
        data_subdirectory=task_config.data_subdirectory,
        base_dir=base_dir,
    )

    X_train, X_test, y_train, y_test, test_ids = _training_ids_for_symptom(
        data_loader, task_config, y_symptom
    )
    print(
        f"  [{symptom_key}] Train: {len(X_train)}, Test: {len(X_test)} "
        f"(positives train: {int(np.sum(y_train))})"
    )

    if len(X_train) < 10:
        raise ValueError(
            f"Not enough training samples for {task_config.task_name}/{symptom_key} "
            f"({len(X_train)})"
        )

    augmenter, rocket, classifier = fit_pipeline(X_train, y_train, task_config)

    metrics = None
    evaluated = False
    if evaluate and len(X_test) > 0:
        from portal_analysis.training.artifact import ArtifactBundle

        bundle = ArtifactBundle(augmenter, rocket, classifier, {}, Path("."))
        y_pred = predict_sequences(
            bundle, X_test, augmentation_method=task_config.augmentation_method
        )
        metrics = print_results(y_test, y_pred, test_ids)
        evaluated = True

    symptom_task_config = TaskConfig.from_dict(
        {
            **task_config.to_dict(),
            "label_column": symptom_key,
            "labels_file": task_config.symptoms_file,
        }
    )
    artifact_path = save_artifact_bundle(
        output_dir=output_dir,
        augmenter=augmenter,
        rocket=rocket,
        classifier=classifier,
        task_config=symptom_task_config,
        metrics=metrics,
        version=version,
        dataset={
            "n_train": len(X_train),
            "n_test": len(X_test),
            "n_evaluated": len(X_test) if evaluated else 0,
            "symptom_key": symptom_key,
            "source_columns": symptom_sources_for_task(task_config.task_name)[symptom_key],
        },
    )
    return {"artifact_path": artifact_path, "metrics": metrics}


def _dedupe_symptom_keys(task_name: str) -> List[Tuple[str, List[str]]]:
    """Order symptoms to train; skip duplicate source-column sets."""
    sources = symptom_sources_for_task(task_name)
    seen: Dict[Tuple[str, ...], str] = {}
    order: List[Tuple[str, List[str]]] = []
    for key in CANONICAL_SYMPTOMS:
        cols = tuple(sources[key])
        if cols in seen:
            order.append((key, cols))  # copy after train
        else:
            seen[cols] = key
            order.append((key, cols))
    return order


def train_all_symptom_models(
    task_config: TaskConfig,
    version: str = "latest",
    base_dir: Optional[Path] = None,
    models_root: Optional[Path] = None,
    evaluate: bool = True,
) -> Dict[str, Dict]:
    """Train all four canonical symptoms for one task."""
    if task_config.task_name == "both_still":
        raise ValueError("both_still has no symptom models.")

    base_dir = base_dir or get_base_processed_directory()
    results: Dict[str, Dict] = {}
    trained_dirs: Dict[Tuple[str, ...], Path] = {}

    for symptom_key in CANONICAL_SYMPTOMS:
        cols = tuple(symptom_sources_for_task(task_config.task_name)[symptom_key])
        out_dir = symptom_model_dir(
            task_config.task_name, symptom_key, version, models_root
        )

        if cols in trained_dirs:
            src = trained_dirs[cols]
            if out_dir.exists():
                shutil.rmtree(out_dir)
            shutil.copytree(src, out_dir)
            print(f"  [{symptom_key}] Reused artifact from {trained_dirs[cols].name}")
            results[symptom_key] = {"artifact_path": out_dir, "metrics": None, "reused": True}
            continue

        print(f"\n--- Symptom: {symptom_key} ---")
        payload = train_symptom_model(
            task_config=task_config,
            symptom_key=symptom_key,
            output_dir=out_dir,
            base_dir=base_dir,
            version=version,
            evaluate=evaluate,
        )
        trained_dirs[cols] = out_dir
        results[symptom_key] = payload

    return results
