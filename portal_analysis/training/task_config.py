"""Task configuration for train/inference pipelines."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class TaskConfig:
    """Configuration for one hand-movement classification task."""

    task_name: str
    data_column_name: str
    file_id_separator: str
    label_column: str = "snorkel_label"
    labels_file: str = "weak_supervision_final.csv"
    labels_subdirectory: str = "docs"
    label_merge_rules: Optional[Dict[int, int]] = field(default_factory=lambda: {4: 3})
    test_set_file: str = "test-set-balanced.csv"
    test_set_subdirectory: str = "docs"
    data_subdirectory: str = "distances"
    tasks: Optional[List[str]] = None
    max_sequence_length: int = 450
    # Signal augmentation
    smooth_window_length: int = 20
    smooth_polyorder: int = 2
    gaussian_sigma: float = 1.0
    include_fft: bool = False
    include_diffs: bool = False
    augmentation_method: str = "simple"
    # Rocket
    rocket_method: str = "minirocket"
    n_kernels: int = 10000
    random_state: int = 42
    rocket_augment_data: bool = True
    # Classifier (None = sklearn default)
    class_weight: Optional[str] = "balanced"
    classifier_cv: int = 10
    classifier_scoring: str = "neg_mean_absolute_error"
    # Symptom columns (for future symptom-model training)
    symptom_columns: Optional[List[str]] = None
    symptoms_file: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TaskConfig":
        label_merge = data.get("label_merge_rules")
        if label_merge is not None:
            data = {
                **data,
                "label_merge_rules": {int(k): int(v) for k, v in label_merge.items()},
            }
        known = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in data.items() if k in known})

    @classmethod
    def from_json(cls, path: Path) -> "TaskConfig":
        with open(path, encoding="utf-8") as f:
            return cls.from_dict(json.load(f))

    def resolved_tasks(self) -> List[str]:
        return self.tasks if self.tasks is not None else ["right", "left"]
