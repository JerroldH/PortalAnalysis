"""
Model management: train, save, load, and predict.

Supports:
  - Versioned artifact bundles (directory with metadata.json)
  - Legacy single .joblib dict files
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import joblib
import numpy as np

from portal_analysis.classification.representation_extractor import RocketRepresentationExtractor
from portal_analysis.classification.signal_augmentation import SignalAugmentation
from portal_analysis.training.artifact import ArtifactBundle, load_artifact_bundle, save_artifact_bundle
from portal_analysis.training.task_config import TaskConfig


@dataclass
class HandMovementModel:
    """Container for a trained hand movement severity model."""

    transformer: Any
    classifier: Any
    augmenter: Optional[SignalAugmentation]
    augment_method: Optional[str]
    task_config: Dict[str, Any]
    classes: np.ndarray = field(default_factory=lambda: np.array([]))
    _rocket_extractor: Optional[RocketRepresentationExtractor] = None

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.classifier.predict(self._extract(X))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X_feat = self._extract(X)
        if hasattr(self.classifier, "predict_proba"):
            return self.classifier.predict_proba(X_feat)
        preds = self.classifier.predict(X_feat)
        n = len(preds)
        proba = np.zeros((n, len(self.classes)))
        for i, p in enumerate(preds):
            idx = np.where(self.classes == p)[0]
            if idx.size:
                proba[i, idx[0]] = 1.0
        return proba

    def _extract(self, X: np.ndarray) -> np.ndarray:
        if self._rocket_extractor is not None:
            if self.augmenter is not None and self.augment_method is not None:
                cfg = self.task_config
                use_signal = cfg.get("include_fft") or cfg.get("include_diffs")
                if use_signal:
                    X = self.augmenter.transform(X, method=self.augment_method)
            return self._rocket_extractor.transform(X)

        if self.augmenter is not None and self.augment_method is not None:
            X = self.augmenter.transform(X, method=self.augment_method)
        elif X.ndim == 2:
            X = X[:, np.newaxis, :]
        return self.transformer.transform(X)

    @classmethod
    def from_artifact_bundle(cls, bundle: ArtifactBundle) -> "HandMovementModel":
        task_config = bundle.metadata.get("task_config", {})
        aug_cfg = bundle.metadata.get("augmenter_config", {})
        classes = (
            bundle.classifier.classes_
            if hasattr(bundle.classifier, "classes_")
            else np.array([])
        )
        return cls(
            transformer=bundle.rocket._transformer,
            classifier=bundle.classifier,
            augmenter=bundle.augmenter,
            augment_method=aug_cfg.get("augmentation_method", "simple"),
            task_config=task_config,
            classes=np.asarray(classes),
            _rocket_extractor=bundle.rocket,
        )


class ModelManager:
    @staticmethod
    def from_task_config(task_config: TaskConfig) -> Dict[str, Any]:
        return task_config.to_dict()

    @staticmethod
    def save_bundle(
        bundle_dir: Path,
        augmenter: SignalAugmentation,
        rocket: RocketRepresentationExtractor,
        classifier,
        task_config: TaskConfig,
        **kwargs,
    ) -> Path:
        return save_artifact_bundle(
            bundle_dir, augmenter, rocket, classifier, task_config, **kwargs
        )

    @staticmethod
    def load(path: Path, task_config: Optional[Dict[str, Any]] = None) -> HandMovementModel:
        path = Path(path)

        if path.is_dir() and (path / "metadata.json").exists():
            return HandMovementModel.from_artifact_bundle(load_artifact_bundle(path))

        if not path.exists():
            raise FileNotFoundError(f"Model not found: {path}")

        payload = joblib.load(path)
        if isinstance(payload, dict) and "classifier" in payload:
            transformer = payload.get("transformer")
            classifier = payload.get("classifier")
            augmenter = payload.get("augmenter")
            augment_method = payload.get("augment_method")
            cfg = payload.get("task_config") or task_config or {}
            classes = payload.get("classes", np.array([]))
        else:
            transformer = payload.named_steps.get("rocket") or payload.steps[0][1]
            classifier = payload.named_steps.get("clf") or payload.steps[-1][1]
            augmenter = None
            augment_method = None
            cfg = task_config or {}
            classes = (
                classifier.classes_ if hasattr(classifier, "classes_") else np.array([])
            )

        return HandMovementModel(
            transformer=transformer,
            classifier=classifier,
            augmenter=augmenter,
            augment_method=augment_method,
            task_config=cfg,
            classes=np.asarray(classes),
        )

    @staticmethod
    def save_legacy(model: HandMovementModel, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "transformer": model.transformer,
                "classifier": model.classifier,
                "augmenter": model.augmenter,
                "augment_method": model.augment_method,
                "task_config": model.task_config,
                "classes": model.classes,
            },
            path,
        )
        print(f"  Saved legacy model -> {path}")
