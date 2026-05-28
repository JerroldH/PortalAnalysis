"""
Tap segmentation and trimming for finger tapping recordings.
Identifies individual tap cycles, extracts best segment, and optionally saves per-tap video clips.
"""

from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from scipy.signal import argrelextrema, find_peaks

FINGER_NORMALIZED_DISTANCE = "Finger Normalized Distance"


class TapTrimmer:
    """
    Identifies tap events in a finger-tapping distance signal and segments the video accordingly.

    Parameters
    ----------
    video_path : str or Path
        Source MP4 video.
    output_dir : str or Path
        Directory where per-tap MP4 clips are saved.
    """

    def __init__(self, video_path, output_dir):
        self.video_path = Path(video_path)
        self.output_dir = Path(output_dir)

    # ------------------------------------------------------------------
    # Tap detection
    # ------------------------------------------------------------------

    def count_taps(
        self,
        distances,
        min_prominence: float = 0.15,
        distance: int = 10,
        height_multiplier: float = 0.3,
    ):
        """
        Detect tap events (local minima in the distance signal).

        Returns
        -------
        tap_count : int
        taps_indices : np.ndarray
            Frame indices of detected taps.
        """
        d = np.asarray(distances, dtype=float)
        local_min = argrelextrema(d, np.less)[0]
        peaks, _ = find_peaks(
            -d,
            prominence=min_prominence,
            distance=distance,
            height=-height_multiplier * d.max(),
        )
        taps_indices = np.intersect1d(local_min, peaks)
        return len(taps_indices), taps_indices

    def find_best_segment_indices(self, distances, taps_indices) -> tuple:
        """Return (start, end) frame indices spanning the detected taps."""
        if taps_indices.size == 0:
            return 0, len(distances) - 1
        return int(taps_indices[0]), int(taps_indices[-1])

    # ------------------------------------------------------------------
    # Feature extraction + video segmentation
    # ------------------------------------------------------------------

    def extract_features_from_distances(self, distances: pd.DataFrame) -> None:
        """
        Trim to the active tapping segment and save per-tap video clips.

        Parameters
        ----------
        distances : pd.DataFrame
            Must contain a 'Finger Normalized Distance' column.
        """
        signal = distances[FINGER_NORMALIZED_DISTANCE].values
        tap_count, taps_indices = self.count_taps(signal)
        if tap_count == 0:
            print("  No taps detected — skipping.")
            return

        start, end = self.find_best_segment_indices(signal, taps_indices)
        trimmed = signal[start : end + 1]
        _, taps_trimmed = self.count_taps(trimmed)
        self.segment_and_save_taps(start, end, taps_trimmed + start)

    def segment_and_save_taps(self, start_index: int, last_index: int, taps_indices) -> None:
        """
        Write one MP4 clip per inter-tap interval.

        Parameters
        ----------
        taps_indices : array-like
            Absolute frame indices of tap events (already offset to global frame numbers).
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)
        cap = cv2.VideoCapture(str(self.video_path))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")

        for i in range(len(taps_indices) - 1):
            seg_start = int(taps_indices[i])
            seg_end = int(taps_indices[i + 1])
            if seg_start < start_index or seg_end > last_index:
                continue

            out_path = self.output_dir / f"tap_segment_{i + 1}.mp4"
            writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))
            cap.set(cv2.CAP_PROP_POS_FRAMES, seg_start)
            for _ in range(seg_start, seg_end):
                ok, frame = cap.read()
                if ok:
                    writer.write(frame)
                else:
                    break
            writer.release()
            print(f"  Saved {out_path.name} (frames {seg_start}–{seg_end})")

        cap.release()
