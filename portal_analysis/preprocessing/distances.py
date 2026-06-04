"""
Distance and angular feature calculation from MediaPipe hand pose CSVs.

Finger tapping writes thumb–index metrics. Hand open/close writes wrist-to-
fingertip sums used for UPDRS 3.5 inference (``Normalized Hand Sum Finger Distances``).
"""

import math
import csv
from pathlib import Path

import numpy as np
import pandas as pd

from portal_analysis.preprocessing import hand_landmarks as hl


class DistanceCalculator:
    """
    Compute kinematic features from a hand pose CSV (output of HandPoseExtractor).

    Parameters
    ----------
    width, height : int
        Pixel dimensions used to de-normalize MediaPipe's [0,1] coordinates.
    """

    FINGER_TAPPING_COLUMNS = [
        "Frame",
        "Finger Distance",
        "Finger Normalized Distance",
        "Angular Distance",
        "Wrist Coordinate",
        "Hand BBox Width",
        "Hand BBox Height",
    ]

    HAND_OPEN_CLOSE_COLUMNS = [
        "Frame",
        "Finger Normalized Distance",
        "Angular Distance",
        "Normalized Hand Sum Finger Distances",
        "Hand Sum Finger Distances",
    ]

    # Backwards-compatible alias for finger tapping.
    DISTANCE_COLUMNS = FINGER_TAPPING_COLUMNS

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
            self._lm_vec(landmarks, hl.THUMB_TIP) - self._lm_vec(landmarks, hl.INDEX_FINGER_TIP)
        ))

    def normalized_finger_distance(self, landmarks: list) -> float:
        """finger_distance normalised by (wrist→index MCP) + (index MCP→index tip)."""
        thumb = self._lm_vec(landmarks, hl.THUMB_TIP)
        index_tip = self._lm_vec(landmarks, hl.INDEX_FINGER_TIP)
        wrist = self._lm_vec(landmarks, hl.WRIST)
        mcp = self._lm_vec(landmarks, hl.INDEX_FINGER_MCP)

        dist = np.linalg.norm(thumb - index_tip)
        norm = np.linalg.norm(wrist - mcp) + np.linalg.norm(mcp - index_tip)
        return float(dist / norm) if norm > 0 else 0.0

    def angular_distance(self, landmarks: list) -> float:
        """Angle (degrees) at the wrist formed by thumb tip and index finger tip."""
        wrist = self._lm_vec(landmarks, hl.WRIST)
        thumb = self._lm_vec(landmarks, hl.THUMB_TIP)
        index_tip = self._lm_vec(landmarks, hl.INDEX_FINGER_TIP)

        vt = thumb - wrist
        vi = index_tip - wrist
        denom = np.linalg.norm(vt) * np.linalg.norm(vi)
        if denom == 0:
            return 0.0
        cos_a = np.clip(np.dot(vt, vi) / denom, -1.0, 1.0)
        return float(math.degrees(math.acos(cos_a)))

    def wrist_coordinates(self, landmarks: list) -> tuple:
        """Scaled wrist (x, y, z)."""
        w = self._lm_vec(landmarks, hl.WRIST)
        return tuple(w)

    def calculate_normalized_finger_palm_distance(self, landmarks: list) -> float:
        """
        Mean fingertip-to-palm-centre distance normalised by palm size.

        Matches ``BoothReports`` ``calculate_normalized_finger_palm_distance``
        (index/middle/ring/pinky tips only; landmarks in MediaPipe [0, 1] coords).
        """
        ts = pd.DataFrame(landmarks).values.reshape(1, 21, 3)
        palm_centre = (ts[:, 0] + ts[:, 5] + ts[:, 17]) / 3
        palm_vector = (
            (ts[:, 5] - ts[:, 0])
            + (ts[:, 9] - ts[:, 0])
            + (ts[:, 13] - ts[:, 0])
            + (ts[:, 17] - ts[:, 0])
        ) / 4
        palm_size = np.linalg.norm(palm_vector, axis=1).reshape(-1, 1)
        fingertip_idxs = [8, 12, 16, 20]
        dist = ts[:, fingertip_idxs] - palm_centre.reshape(
            palm_centre.shape[0], 1, palm_centre.shape[1]
        )
        dist = np.linalg.norm(dist, axis=2)
        norm_dist = dist / palm_size
        return float(norm_dist.mean())

    def calculate_hand_movement_angular_distance(self, landmarks: list) -> float:
        """
        2-D angle (degrees) at middle MCP between wrist→middle MCP and MCP→middle tip.

        Matches ``BoothReports`` ``calculate_angular_distance`` (integer pixel coords).
        """
        wrist_x = int(landmarks[hl.WRIST][0] * self.width)
        wrist_y = int(landmarks[hl.WRIST][1] * self.height)
        middle_mcp_x = int(landmarks[hl.MIDDLE_FINGER_MCP][0] * self.width)
        middle_mcp_y = int(landmarks[hl.MIDDLE_FINGER_MCP][1] * self.height)
        middle_tip_x = int(landmarks[hl.MIDDLE_FINGER_TIP][0] * self.width)
        middle_tip_y = int(landmarks[hl.MIDDLE_FINGER_TIP][1] * self.height)

        vector_wt = (wrist_x - middle_mcp_x, wrist_y - middle_mcp_y)
        vector_wi = (middle_tip_x - middle_mcp_x, middle_tip_y - middle_mcp_y)

        dot = vector_wt[0] * vector_wi[0] + vector_wt[1] * vector_wi[1]
        denom1 = math.sqrt(vector_wt[0] ** 2 + vector_wt[1] ** 2)
        denom2 = math.sqrt(vector_wi[0] ** 2 + vector_wi[1] ** 2)

        if denom1 < 1e-6 or denom2 < 1e-6:
            return 0.0

        cos_x = dot / (denom1 * denom2)
        cos_x = np.minimum(cos_x, 1.0)
        return float((math.acos(cos_x) * 180) / math.pi)

    def hand_sum_finger_distances(self, landmarks: list) -> float:
        """
        Sum of 3-D wrist-to-fingertip distances in pixel units (five digits).

        Matches ``BoothReports`` ``calculalate_hand_sum_finger_distances``.
        """
        wrist = np.array([
            landmarks[hl.WRIST][0] * self.width,
            landmarks[hl.WRIST][1] * self.height,
            landmarks[hl.WRIST][2] * self.width,
        ])
        distances = []
        for tip in hl.FINGER_TIPS:
            tip_pt = np.array([
                landmarks[tip][0] * self.width,
                landmarks[tip][1] * self.height,
                landmarks[tip][2] * self.width,
            ])
            distances.append(np.linalg.norm(tip_pt - wrist))
        return float(np.sum(distances))

    def normalized_hand_sum_finger_distances(self, landmarks: list) -> float:
        """
        Raw hand-sum distance normalised by wrist→thumb CMC (landmarks 0–1).

        Matches ``BoothReports`` ``calculalate_normalized_hand_sum_finger_distances``.
        """
        try:
            raw_sum = self.hand_sum_finger_distances(landmarks)
            wrist = np.array([
                landmarks[hl.WRIST][0] * self.width,
                landmarks[hl.WRIST][1] * self.height,
                landmarks[hl.WRIST][2] * self.width,
            ])
            thumb_cmc = np.array([
                landmarks[hl.THUMB_CMC][0] * self.width,
                landmarks[hl.THUMB_CMC][1] * self.height,
                landmarks[hl.THUMB_CMC][2] * self.width,
            ])
            ref_dist = np.linalg.norm(thumb_cmc - wrist)
            if ref_dist < 1e-6:
                return float("nan")
            return float(raw_sum / ref_dist)
        except Exception:
            return float("nan")

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

    def calculate_hand_open_close_distances(
        self,
        pose_csv: Path,
        output_path: Path = None,
    ) -> Path:
        """
        Read a pose CSV and write hand open/close distances for UPDRS 3.5 inference.

        Output columns and formulas match ``BoothReports`` ``hand_movement_distances.py``.
        """
        pose_csv = Path(pose_csv)
        if output_path is None:
            output_path = pose_csv.parent.parent / "distances" / f"{pose_csv.stem}_distances.csv"
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        df = pd.read_csv(pose_csv)

        with open(output_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(self.HAND_OPEN_CLOSE_COLUMNS)

            for frame_no in df["frame_number"].unique():
                row_data = df[df["frame_number"] == frame_no].iloc[0]
                landmarks = [
                    [row_data[f"x_{i}"], row_data[f"y_{i}"], row_data[f"z_{i}"]]
                    for i in range(21)
                ]

                if all(x == 0 and y == 0 and z == 0 for x, y, z in landmarks):
                    continue

                writer.writerow([
                    frame_no,
                    self.calculate_normalized_finger_palm_distance(landmarks),
                    self.calculate_hand_movement_angular_distance(landmarks),
                    self.normalized_hand_sum_finger_distances(landmarks),
                    self.hand_sum_finger_distances(landmarks),
                ])

        print(f"  Distances → {output_path}")
        return output_path

    def calculate_hand_up_down_distances(
        self,
        pose_csv: Path,
        output_path: Path = None,
    ) -> Path:
        """
        Read a pose CSV and write hand up/down (pronation–supination) angles CSV.

        Output matches ``BoothReports`` ``HandMovementAnglesProcessor`` /
        ``Booth_Processed/hand_up_down/.../distances/*.csv`` (includes ``yaw_rad``).
        """
        from portal_analysis.preprocessing.hand_movement_angles import (
            HandMovementAnglesProcessor,
        )

        pose_csv = Path(pose_csv)
        if output_path is None:
            output_path = pose_csv.parent.parent / "distances" / f"{pose_csv.stem}_distances.csv"
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        processor = HandMovementAnglesProcessor(width=self.width, height=self.height)
        processor.process_csv_file(pose_csv, output_path)
        print(f"  Distances → {output_path}")
        return output_path

    def process_task(self, base_processed_dir: Path, task: str, subtask: str) -> None:
        """
        Compute distances for all pose CSVs in
        *base_processed_dir/task/subtask/pose/*.
        """
        pose_dir = Path(base_processed_dir) / task / subtask / "pose"
        if task == "hand_open_close":
            calc = self.calculate_hand_open_close_distances
        elif task == "hand_up_down":
            calc = self.calculate_hand_up_down_distances
        else:
            calc = self.calculate_distances
        for pose_file in sorted(pose_dir.glob("*.csv")):
            calc(pose_file)

    def read_distances(self, distance_file: Path) -> pd.DataFrame:
        return pd.read_csv(distance_file)[self.DISTANCE_COLUMNS]
