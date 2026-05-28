"""
Time series feature extraction using Rocket-based transformations (MiniRocket / Rocket / MultiRocket).
"""

from typing import Optional, Tuple
import numpy as np
from scipy.ndimage import gaussian_filter1d


def _augment_with_fft_and_diffs(X: np.ndarray) -> np.ndarray:
    """Stack original, FFT, 1st diff, and 2nd diff → (n_samples, 4, series_length)."""
    X_diff = np.diff(X, axis=1, prepend=X[:, [0]])
    X_diff2 = np.diff(X_diff, axis=1, prepend=X_diff[:, [0]])
    X_fft = gaussian_filter1d(np.abs(np.fft.fft(X, axis=1)), sigma=1, axis=1)
    X_diff_smooth = gaussian_filter1d(X_diff, sigma=1, axis=1)
    X_diff2_smooth = gaussian_filter1d(X_diff2, sigma=1, axis=1)
    return np.stack([X, X_fft, X_diff_smooth, X_diff2_smooth], axis=1)


class RocketRepresentationExtractor:
    """
    Extracts fixed-length feature vectors from time series using MiniRocket/Rocket/MultiRocket.

    Typical usage:
        extractor = RocketRepresentationExtractor(method='minirocket', n_kernels=10000)
        X_train_feat = extractor.fit_transform(X_train)   # shape (n, n_features)
        X_test_feat  = extractor.transform(X_test)
    """

    def __init__(
        self,
        method: str = "minirocket",
        n_kernels: int = 10000,
        random_state: Optional[int] = 42,
        augment_data: bool = True,
    ):
        self.method = method.lower()
        self.n_kernels = n_kernels
        self.random_state = random_state
        self.augment_data = augment_data
        self._transformer = None
        self._is_fitted = False

    def _build_transformer(self):
        if self.method == "minirocket":
            from aeon.transformations.collection.convolution_based import MiniRocket
            return MiniRocket(n_kernels=self.n_kernels, random_state=self.random_state)
        elif self.method == "rocket":
            from aeon.transformations.collection.convolution_based import Rocket
            return Rocket(n_kernels=self.n_kernels, random_state=self.random_state)
        elif self.method == "multirocket":
            from aeon.transformations.collection.convolution_based import MultiRocket
            return MultiRocket(n_kernels=self.n_kernels, random_state=self.random_state)
        raise ValueError(f"Unknown method '{self.method}'. Choose minirocket, rocket, or multirocket.")

    def _prepare(self, X: np.ndarray) -> np.ndarray:
        """Optionally augment and ensure shape (n_samples, n_channels, series_length)."""
        if X.ndim == 2:
            return _augment_with_fft_and_diffs(X) if self.augment_data else X[:, np.newaxis, :]
        return X  # already (n, c, t)

    def fit(self, X: np.ndarray) -> "RocketRepresentationExtractor":
        if self._transformer is None:
            self._transformer = self._build_transformer()
        self._transformer.fit(self._prepare(X))
        self._is_fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if not self._is_fitted:
            raise RuntimeError("Call fit() before transform().")
        return self._transformer.transform(self._prepare(X))

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

    def extract_features_split(
        self, X_train: np.ndarray, X_test: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Fit on train, transform both train and test."""
        print(f"Fitting {self.method.upper()} ({self.n_kernels} kernels)...")
        X_train_feat = self.fit_transform(X_train)
        print(f"Transforming test set... Feature shape: {X_train_feat.shape}")
        return X_train_feat, self.transform(X_test)
