"""
Distance and angular feature calculation from MediaPipe hand pose CSVs.

Supports finger tapping (thumb-index distance) and hand movement tasks.
Output CSV columns:
    Frame, Finger Distance, Finger Normalized Distance, Angular Distance,
    Wrist Coordinate, Hand BBox Width, Hand BBox Height
"""

import math
import csv
from pathlib import Path

import numpy as np
import pandas as pd
import mediapipe as mp


class DistanceCalculator:
    """
    Compute kinematic features from a hand pose CSV (output of HandPoseExtractor).

    Parameters
    ----------
    width, height : int
        Pixel dimensions used to de-normalize MediaPipe's [0,1] coordinates.
    """

    DISTANCE_COLUMNS = [
        "Frame",
        "Finger Distance",
        "Finger Normalized Distance",
        "Angular Distance",
        "Wrist Coordinate",
        "Hand BBox Width",
        "Hand BBox Height",
    ]

    _lm = mp.solutions.hands.HandLandmark

    def __init__(self, width: int = 1920, height: int = 1080):
        self.width = width
        self.height = height

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _scale(self, raw: list) -> np.ndarray:
        """Scale a [x, y, z] landmark from MediaPipe normalized coords."""
        return np.array([raw[0] * self.width, raw[1] * self.height, raw[2] * self.width])

    def _lm_vec(self, landmarks: list, idx: int) -> np.ndarray:
        return self._scale(landmarks[idx])

    # ------------------------------------------------------------------
    # Per-frame metrics
    # ------------------------------------------------------------------

    def finger_distance(self, landmarks: list) -> float:
        """3-D Euclidean distance between thumb tip and index finger tip."""
        return float(np.linalg.norm(
            self._lm_vec(landmarks, self._lm.THUMB_TIP) - self._lm_vec(landmarks, self._lm.INDEX_FINGER_TIP)
        ))

    def normalized_finger_distance(self, landmarks: list) -> float:
        """finger_distance normalised by (wrist→index MCP) + (index MCP→index tip)."""
        thumb = self._lm_vec(landmarks, self._lm.THUMB_TIP)
        index_tip = self._lm_vec(landmarks, self._lm.INDEX_FINGER_TIP)
        wrist = self._lm_vec(landmarks, self._lm.WRIST)
        mcp = self._lm_vec(landmarks, self._lm.INDEX_FINGER_MCP)

        dist = np.linalg.norm(thumb - index_tip)
        norm = np.linalg.norm(wrist - mcp) + np.linalg.norm(mcp - index_tip)
        return float(dist / norm) if norm > 0 else 0.0

    def angular_distance(self, landmarks: list) -> float:
        """Angle (degrees) at the wrist formed by thumb tip and index finger tip."""
        wrist = self._lm_vec(landmarks, self._lm.WRIST)
        thumb = self._lm_vec(landmarks, self._lm.THUMB_TIP)
        index_tip = self._lm_vec(landmarks, self._lm.INDEX_FINGER_TIP)

        vt = thumb - wrist
        vi = index_tip - wrist
        denom = np.linalg.norm(vt) * np.linalg.norm(vi)
        if denom == 0:
            return 0.0
        cos_a = np.clip(np.dot(vt, vi) / denom, -1.0, 1.0)
        return float(math.degrees(math.acos(cos_a)))

    def wrist_coordinates(self, landmarks: list) -> tuple:
        """Scaled wrist (x, y, z)."""
        w = self._lm_vec(landmarks, self._lm.WRIST)
        return tuple(w)

    # ------------------------------------------------------------------
    # Batch processing
    # ------------------------------------------------------------------

    def calculate_distances(self, pose_csv: Path, output_path: Path = None) -> Path:
        """
        Read a pose CSV produced by HandPoseExtractor and write a distances CSV.

        Parameters
        ----------
        pose_csv : Path
            Input pose landmarks CSV.
        output_path : Path, optional
            Where to write the output. Defaults to
            *pose_csv.parent.parent/distances/<stem>_distances.csv*.

        Returns
        -------
        Path
            Path of the written distances CSV.
        """
        pose_csv = Path(pose_csv)
        if output_path is None:
            output_path = pose_csv.parent.parent / "distances" / f"{pose_csv.stem}_distances.csv"
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        df = pd.read_csv(pose_csv)

        with open(output_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(self.DISTANCE_COLUMNS)

            for frame_no in df["frame_number"].unique():
                row_data = df[df["frame_number"] == frame_no].iloc[0]
                landmarks = [[row_data[f"x_{i}"], row_data[f"y_{i}"], row_data[f"z_{i}"]] for i in range(21)]

                if all(x == 0 and y == 0 and z == 0 for x, y, z in landmarks):
                    continue

                writer.writerow([
                    frame_no,
                    self.finger_distance(landmarks),
                    self.normalized_finger_distance(landmarks),
                    self.angular_distance(landmarks),
                    self.wrist_coordinates(landmarks),
                    row_data["hand_width"] * self.width,
                    row_data["hand_height"] * self.height,
                ])

        print(f"  Distances → {output_path}")
        return output_path

    def process_task(self, base_processed_dir: Path, task: str, subtask: str) -> None:
        """
        Compute distances for all pose CSVs in
        *base_processed_dir/task/subtask/pose/*.
        """
        pose_dir = Path(base_processed_dir) / task / subtask / "pose"
        for pose_file in sorted(pose_dir.glob("*.csv")):
            self.calculate_distances(pose_file)

    def read_distances(self, distance_file: Path) -> pd.DataFrame:
        return pd.read_csv(distance_file)[self.DISTANCE_COLUMNS]
