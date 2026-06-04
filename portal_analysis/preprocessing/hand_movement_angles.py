"""
Palm rotation and hand pose metrics from MediaPipe landmark CSVs.

Ported from BoothReports ``hand_movement_angles_processor.py`` for pronation–
supination (hand up/down) inference on ``yaw_rad``.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


class HandMovementAnglesProcessor:
    """
    Compute palm rotation matrices, quaternions, Euler angles, and pose metrics
    from hand pose CSV rows (MediaPipe landmarks).
    """

    WRIST = 0
    INDEX_MCP = 5
    PINKY_MCP = 17

    def __init__(self, width: int = 1920, height: int = 1080):
        self.width = width
        self.height = height

    @staticmethod
    def _normalize(v: np.ndarray, eps: float = 1e-8) -> np.ndarray:
        n = np.linalg.norm(v)
        if n < eps:
            return np.zeros_like(v)
        return v / n

    @staticmethod
    def _rotation_matrix_from_axes(
        x_hat: np.ndarray, y_hat: np.ndarray, z_hat: np.ndarray
    ) -> np.ndarray:
        B = np.stack([x_hat, y_hat, z_hat], axis=1)
        x = HandMovementAnglesProcessor._normalize(B[:, 0])
        y = HandMovementAnglesProcessor._normalize(B[:, 1] - np.dot(B[:, 1], x) * x)
        z = HandMovementAnglesProcessor._normalize(np.cross(x, y))
        y = HandMovementAnglesProcessor._normalize(np.cross(z, x))
        B = np.stack([x, y, z], axis=1)
        return B.T

    @staticmethod
    def _rotmat_to_quaternion(R: np.ndarray) -> Tuple[float, float, float, float]:
        m00, m01, m02 = R[0, 0], R[0, 1], R[0, 2]
        m10, m11, m12 = R[1, 0], R[1, 1], R[1, 2]
        m20, m21, m22 = R[2, 0], R[2, 1], R[2, 2]
        tr = m00 + m11 + m22
        if tr > 0:
            S = math.sqrt(tr + 1.0) * 2.0
            w = 0.25 * S
            x = (m21 - m12) / S
            y = (m02 - m20) / S
            z = (m10 - m01) / S
        elif (m00 > m11) and (m00 > m22):
            S = math.sqrt(1.0 + m00 - m11 - m22) * 2.0
            w = (m21 - m12) / S
            x = 0.25 * S
            y = (m01 + m10) / S
            z = (m02 + m20) / S
        elif m11 > m22:
            S = math.sqrt(1.0 + m11 - m00 - m22) * 2.0
            w = (m02 - m20) / S
            x = (m01 + m10) / S
            y = 0.25 * S
            z = (m12 + m21) / S
        else:
            S = math.sqrt(1.0 + m22 - m00 - m11) * 2.0
            w = (m10 - m01) / S
            x = (m02 + m20) / S
            y = (m12 + m21) / S
            z = 0.25 * S
        q = np.array([w, x, y, z], dtype=float)
        q /= np.linalg.norm(q) + 1e-12
        return tuple(q.tolist())

    @staticmethod
    def _rotmat_to_euler_zyx(R: np.ndarray) -> Tuple[float, float, float]:
        r20 = np.clip(-R[2, 0], -1.0, 1.0)
        pitch = math.asin(r20)
        if abs(r20) < 0.999999:
            yaw = math.atan2(R[1, 0], R[0, 0])
            roll = math.atan2(R[2, 1], R[2, 2])
        else:
            yaw = math.atan2(-R[0, 1], R[1, 1])
            roll = 0.0
        return yaw, pitch, roll

    def _extract_landmark_point(self, row: pd.Series, idx: int) -> np.ndarray:
        return np.array([row[f"x_{idx}"], row[f"y_{idx}"], row[f"z_{idx}"]], dtype=float)

    def compute_palm_rotation_from_row(
        self, row: pd.Series, handedness_col: str = "hand_label"
    ) -> Dict:
        wrist = self._extract_landmark_point(row, self.WRIST)
        idx_mcp = self._extract_landmark_point(row, self.INDEX_MCP)
        pky_mcp = self._extract_landmark_point(row, self.PINKY_MCP)

        x_hat = self._normalize(idx_mcp - wrist)
        v1 = idx_mcp - wrist
        v2 = pky_mcp - wrist
        n_hat = self._normalize(np.cross(v1, v2))

        handed = str(row.get(handedness_col, "")).lower()
        if "left" in handed:
            n_hat = -n_hat

        y_hat = self._normalize(np.cross(n_hat, x_hat))
        z_hat = self._normalize(np.cross(x_hat, y_hat))

        R = self._rotation_matrix_from_axes(x_hat, y_hat, z_hat)
        w, x, y, z = self._rotmat_to_quaternion(R)
        yaw, pitch, roll = self._rotmat_to_euler_zyx(R)

        return {
            "R00": R[0, 0], "R01": R[0, 1], "R02": R[0, 2],
            "R10": R[1, 0], "R11": R[1, 1], "R12": R[1, 2],
            "R20": R[2, 0], "R21": R[2, 1], "R22": R[2, 2],
            "quat_w": w, "quat_x": x, "quat_y": y, "quat_z": z,
            "yaw_rad": yaw, "pitch_rad": pitch, "roll_rad": roll,
        }

    def calculate_finger_angles(self, row: pd.Series) -> Dict:
        finger_angles = {}
        fingers = {
            "thumb": [1, 2, 3, 4],
            "index": [5, 6, 7, 8],
            "middle": [9, 10, 11, 12],
            "ring": [13, 14, 15, 16],
            "pinky": [17, 18, 19, 20],
        }

        for finger_name, landmarks in fingers.items():
            try:
                for i in range(len(landmarks) - 2):
                    p1 = self._extract_landmark_point(row, landmarks[i])
                    p2 = self._extract_landmark_point(row, landmarks[i + 1])
                    p3 = self._extract_landmark_point(row, landmarks[i + 2])
                    v1 = p1 - p2
                    v2 = p3 - p2
                    cos_angle = np.dot(v1, v2) / (
                        np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8
                    )
                    cos_angle = np.clip(cos_angle, -1.0, 1.0)
                    angle_deg = np.degrees(np.arccos(cos_angle))
                    joint_names = ["mcp_pip", "pip_dip"]
                    if i < len(joint_names):
                        finger_angles[f"{finger_name}_{joint_names[i]}_angle"] = angle_deg
            except (KeyError, ValueError):
                continue

        return finger_angles

    def calculate_hand_pose_metrics(self, row: pd.Series) -> Dict:
        metrics = {}
        try:
            thumb_tip = self._extract_landmark_point(row, 4)
            pinky_tip = self._extract_landmark_point(row, 20)
            metrics["hand_span"] = np.linalg.norm(thumb_tip - pinky_tip)

            wrist = self._extract_landmark_point(row, 0)
            middle_tip = self._extract_landmark_point(row, 12)
            metrics["hand_length"] = np.linalg.norm(middle_tip - wrist)

            index_mcp = self._extract_landmark_point(row, 5)
            pinky_mcp = self._extract_landmark_point(row, 17)
            metrics["hand_width"] = np.linalg.norm(index_mcp - pinky_mcp)

            if metrics["hand_length"] > 0:
                metrics["hand_aspect_ratio"] = metrics["hand_width"] / metrics["hand_length"]
            else:
                metrics["hand_aspect_ratio"] = 0.0
        except (KeyError, ValueError):
            metrics.update({
                "hand_span": 0.0,
                "hand_length": 0.0,
                "hand_width": 0.0,
                "hand_aspect_ratio": 0.0,
            })
        return metrics

    def process_csv_row(
        self,
        row: pd.Series,
        frame_col: str = "frame_number",
        hand_id_col: str = "hand_id",
        handedness_col: str = "hand_label",
    ) -> Dict:
        result = {
            frame_col: row.get(frame_col, 0),
            hand_id_col: row.get(hand_id_col, ""),
            handedness_col: row.get(handedness_col, ""),
        }
        try:
            result.update(self.compute_palm_rotation_from_row(row, handedness_col))
            result.update(self.calculate_finger_angles(row))
            result.update(self.calculate_hand_pose_metrics(row))
        except Exception as exc:
            print(f"Error processing row: {exc}")
        return result

    def process_csv_file(
        self,
        input_csv_path: str | Path,
        output_csv_path: str | Path | None = None,
        frame_col: str = "frame_number",
        hand_id_col: str = "hand_id",
        handedness_col: str = "hand_label",
    ) -> pd.DataFrame:
        input_path = Path(input_csv_path)
        if not input_path.exists():
            raise FileNotFoundError(f"Input CSV file not found: {input_csv_path}")

        df = pd.read_csv(input_path)

        for idx in (self.WRIST, self.INDEX_MCP, self.PINKY_MCP):
            for axis in ("x", "y", "z"):
                col = f"{axis}_{idx}"
                if col not in df.columns:
                    raise ValueError(f"Expected column '{col}' not found in input CSV.")

        results = []
        for i, row in df.iterrows():
            landmarks = [
                [row[f"x_{j}"], row[f"y_{j}"], row[f"z_{j}"]] for j in range(21)
            ]
            if all(x == 0 and y == 0 and z == 0 for x, y, z in landmarks):
                continue
            try:
                results.append(
                    self.process_csv_row(row, frame_col, hand_id_col, handedness_col)
                )
            except Exception as exc:
                print(f"Skipping row {i}: {exc}")

        output_df = pd.DataFrame(results)

        if output_csv_path:
            output_path = Path(output_csv_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_df.to_csv(output_path, index=False)

        return output_df
