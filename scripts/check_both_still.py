import numpy as np
import os
os.environ['OMP_NUM_THREADS'] = '1'

from portal_analysis.training.settings import get_base_processed_directory
from portal_analysis.training.task_config import TaskConfig
from portal_analysis.training.pipeline import load_training_dataset, _build_augmenter, _build_rocket
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeClassifierCV
from pathlib import Path

cfg = TaskConfig.from_json(Path("configs/both_still.json"))
base = get_base_processed_directory()

X_train, X_test, y_train, y_test, test_ids = load_training_dataset(cfg, base_dir=base)
valid = y_train >= 0
X_train, y_train = X_train[valid], y_train[valid]

augmenter = _build_augmenter(cfg)
X_aug = augmenter.transform(X_train, method=cfg.augmentation_method)
rocket = _build_rocket(cfg)
X_repr = rocket.fit_transform(X_aug)
print(f"X_repr shape: {X_repr.shape}, finite: {np.all(np.isfinite(X_repr))}", flush=True)

print("Scaling...", flush=True)
scaler = StandardScaler(with_mean=False)
X_scaled = scaler.fit_transform(X_repr)
print(f"Scaled shape: {X_scaled.shape}, finite: {np.all(np.isfinite(X_scaled))}", flush=True)

# Check for zero-variance features
n_zero = np.sum(scaler.scale_ == 0)
print(f"Zero-variance features: {n_zero}", flush=True)

print("Fitting RidgeClassifierCV (cv=5)...", flush=True)
clf = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10), cv=5)
clf.fit(X_scaled, y_train)
print("Done!", flush=True)
