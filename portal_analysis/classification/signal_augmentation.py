"""
Signal augmentation for time series data.
Provides smoothing, FFT, and difference-based feature augmentation.
"""

from typing import Tuple
import numpy as np
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter1d


class SignalAugmentation:
    """
    Augmentation for time series signals: smoothing, FFT, and differences.
    """

    def __init__(
        self,
        smooth_window_length: int = 20,
        smooth_polyorder: int = 2,
        gaussian_sigma: float = 1.0,
        include_fft: bool = True,
        include_diffs: bool = True,
    ):
        self.smooth_window_length = smooth_window_length
        self.smooth_polyorder = smooth_polyorder
        self.gaussian_sigma = gaussian_sigma
        self.include_fft = include_fft
        self.include_diffs = include_diffs

    def smooth_signal(self, signal: np.ndarray) -> np.ndarray:
        if len(signal) < self.smooth_window_length:
            return signal
        return savgol_filter(signal, window_length=self.smooth_window_length, polyorder=self.smooth_polyorder)

    def smooth_2d(self, X: np.ndarray) -> np.ndarray:
        return np.array([self.smooth_signal(X[i]) for i in range(X.shape[0])])

    def compute_differences(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        X_diff = np.diff(X, axis=1, prepend=X[:, [0]])
        X_diff2 = np.diff(X_diff, axis=1, prepend=X_diff[:, [0]])
        return X_diff, X_diff2

    def compute_fft(self, X: np.ndarray, apply_smoothing: bool = True) -> np.ndarray:
        X_fft = np.abs(np.fft.fft(X, axis=1))
        if apply_smoothing:
            X_fft = gaussian_filter1d(X_fft, sigma=self.gaussian_sigma, axis=1)
        return X_fft

    def augment_simple(self, X: np.ndarray) -> np.ndarray:
        """Original signal + FFT magnitude → shape (n_samples, 2, series_length)."""
        X_fft = np.abs(np.fft.fft(X, axis=1))
        return np.stack([X, X_fft], axis=1)

    def augment_full(self, X: np.ndarray) -> np.ndarray:
        """Original + FFT + 1st diff + 2nd diff → shape (n_samples, 4, series_length)."""
        X_diff, X_diff2 = self.compute_differences(X)
        X_diff_smooth = gaussian_filter1d(X_diff, sigma=self.gaussian_sigma, axis=1)
        X_diff2_smooth = gaussian_filter1d(X_diff2, sigma=self.gaussian_sigma, axis=1)

        features = [X]
        if self.include_fft:
            features.append(self.compute_fft(X, apply_smoothing=True))
        if self.include_diffs:
            features.append(X_diff_smooth)
            features.append(X_diff2_smooth)

        return np.stack(features, axis=1) if len(features) > 1 else X[:, np.newaxis, :]

    def transform(self, X: np.ndarray, method: str = "simple") -> np.ndarray:
        """
        Args:
            method: 'simple' (original + FFT) or 'full' (original + FFT + diffs)
        """
        if method == "simple":
            return self.augment_simple(X)
        elif method == "full":
            return self.augment_full(X)
        raise ValueError(f"Unknown method: {method}. Use 'simple' or 'full'.")
