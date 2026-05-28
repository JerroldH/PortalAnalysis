"""
Generic time series data loader for classification tasks.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from pathlib import Path
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tqdm import tqdm

from portal_analysis.training.settings import get_base_processed_directory


class TimeSeriesDataLoader:
    """Load labels, time series CSVs, pad sequences, and split train/test."""

    MINIMUM_FRAMES = 5
    DEFAULT_MAX_SEQUENCE_LENGTH = 450

    def __init__(
        self,
        task_name: str = "finger_tapping",
        tasks: Optional[List[str]] = None,
        data_column_name: str = "Finger Normalized Distance",
        file_id_separator: str = "_finger",
        data_subdirectory: str = "distances",
        base_dir: Optional[Path] = None,
    ):
        self.task_name = task_name
        self.tasks = tasks if tasks is not None else ["right", "left"]
        self.data_column_name = data_column_name
        self.file_id_separator = file_id_separator
        self.data_subdirectory = data_subdirectory
        self.base_dir = Path(base_dir) if base_dir else get_base_processed_directory()

    def load_labels(
        self,
        label_column: str = "snorkel_label_final",
        labels_file: str = "weak_supervision_final.csv",
        labels_subdirectory: str = "docs",
        label_merge_rules: Optional[Dict[int, int]] = None,
    ) -> pd.Series:
        labels_path = (
            self.base_dir / self.task_name / labels_subdirectory / labels_file
        )
        df_labels = pd.read_csv(labels_path).set_index("ID")
        y = df_labels[label_column]
        y = y[~y.index.duplicated(keep="first")]
        y.dropna(inplace=True)

        if label_merge_rules:
            for source_label, target_label in label_merge_rules.items():
                y[y == source_label] = target_label

        return y

    def _extract_file_id(self, filename: str) -> str:
        if self.file_id_separator in filename:
            return filename.split(self.file_id_separator)[0]
        return filename

    def _load_time_series_sequence(self, file_path: Path) -> Optional[np.ndarray]:
        try:
            data_df = pd.read_csv(file_path)
            if len(data_df) < self.MINIMUM_FRAMES:
                return None
            if self.data_column_name not in data_df.columns:
                return None
            return data_df[self.data_column_name].values
        except (FileNotFoundError, pd.errors.EmptyDataError, KeyError):
            return None

    def _get_data_directory_path(self, task: str) -> Path:
        return self.base_dir / self.task_name / task / self.data_subdirectory

    def _process_task_directory(
        self,
        data_dir: Path,
        valid_ids: pd.Index,
        task_name: str,
    ) -> Dict[str, np.ndarray]:
        if not data_dir.exists():
            return {}

        task_data = {}
        csv_files = list(data_dir.glob("*.csv"))
        with tqdm(total=len(csv_files), desc=f"Loading {task_name} data") as pbar:
            for file_path in csv_files:
                time_series_sequence = self._load_time_series_sequence(file_path)
                file_id = self._extract_file_id(file_path.stem)
                if file_id in valid_ids:
                    task_data[file_id] = time_series_sequence
                pbar.update(1)

        return task_data

    def load_time_series_data(
        self,
        label_ids: pd.Index,
        labels: Optional[pd.Series] = None,
        label_column_name: str = "label",
    ) -> pd.DataFrame:
        label_ids = pd.Index(label_ids)
        if label_ids.has_duplicates:
            label_ids = label_ids[~label_ids.duplicated(keep="first")]

        combined_data = pd.DataFrame(index=label_ids, columns=["values"], dtype=object)

        for task in self.tasks:
            data_dir = self._get_data_directory_path(task)
            task_data = self._process_task_directory(data_dir, label_ids, task)
            for file_id, time_series_sequence in task_data.items():
                combined_data.at[file_id, "values"] = time_series_sequence

        if labels is not None:
            combined_data[label_column_name] = labels.reindex(combined_data.index)

        return combined_data

    def prepare_sequences(
        self,
        df_combined: pd.DataFrame,
        labels: pd.Series,
        maxlen: Optional[int] = None,
        shuffle: bool = True,
        random_state: int = 42,
    ) -> Tuple[np.ndarray, np.ndarray, pd.Index]:
        if maxlen is None:
            maxlen = self.DEFAULT_MAX_SEQUENCE_LENGTH

        if shuffle:
            df_combined = df_combined.copy().sample(frac=1, random_state=random_state)

        valid_ids = df_combined["values"].dropna().index.intersection(labels.index)
        sequences = df_combined.loc[valid_ids, "values"].tolist()
        X_padded = pad_sequences(
            sequences,
            maxlen=maxlen,
            dtype="float32",
            padding="post",
            truncating="post",
        )
        y_aligned = labels.loc[valid_ids].values
        return X_padded, y_aligned, valid_ids

    def split_train_test(
        self,
        X: np.ndarray,
        y: np.ndarray,
        valid_ids: pd.Index,
        test_set_file: str = "test-set-balanced.csv",
        test_set_subdirectory: str = "docs",
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        test_set_path = (
            self.base_dir / self.task_name / test_set_subdirectory / test_set_file
        )
        test_file_names = pd.read_csv(test_set_path)["file_name"]
        test_mask = valid_ids.isin(test_file_names)
        test_ids = valid_ids[test_mask].values

        return (
            X[~test_mask],
            X[test_mask],
            y[~test_mask],
            y[test_mask],
            test_ids,
        )
