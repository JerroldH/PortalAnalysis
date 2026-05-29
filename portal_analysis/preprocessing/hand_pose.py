"""
Hand pose extraction from video using MediaPipe Tasks (HandLandmarker).
Produces per-frame CSV with 21 hand landmark coordinates and bounding-box dimensions.
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

import cv2
import mediapipe as mp
import pandas as pd
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

_HAND_LANDMARKER_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)
_DEFAULT_MODEL_PATH = Path(__file__).resolve().parent / "hand_landmarker.task"


def _ensure_hand_landmarker_model(path: Path | None = None) -> Path:
    path = Path(path or _DEFAULT_MODEL_PATH)
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading hand landmarker model to {path} ...")
    urllib.request.urlretrieve(_HAND_LANDMARKER_MODEL_URL, path)
    return path


class HandPoseExtractor:
    """
    Extract hand pose (21 landmarks) from MP4 video files using MediaPipe.

    Output CSV columns:
        frame_number, hand_id, hand_label, hand_width, hand_height,
        x_0..x_20, y_0..y_20, z_0..z_20
    """

    LANDMARK_COLUMNS = (
        ["frame_number", "hand_id", "hand_label", "hand_width", "hand_height"]
        + [c for i in range(21) for c in (f"x_{i}", f"y_{i}", f"z_{i}")]
    )

    def __init__(
        self,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        model_path: Path | str | None = None,
    ):
        model_path = _ensure_hand_landmarker_model(
            Path(model_path) if model_path is not None else None
        )
        options = vision.HandLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=min_detection_confidence,
            min_hand_presence_confidence=min_tracking_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._detector = vision.HandLandmarker.create_from_options(options)

    def _bbox(self, landmarks) -> tuple:
        xs = [lm.x for lm in landmarks]
        ys = [lm.y for lm in landmarks]
        return max(xs) - min(xs), max(ys) - min(ys)

    def process_video(self, video_path: Path, output_path: Path) -> bool:
        """
        Extract hand landmarks from *video_path* and save to *output_path*.

        Returns True if successful, False if no hands were detected.
        Skips processing if output_path already exists.
        """
        video_path = Path(video_path)
        output_path = Path(output_path)

        if output_path.exists():
            return True

        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        rows = []
        frame_number = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp_ms = int(frame_number * 1000.0 / fps)
            results = self._detector.detect_for_video(mp_image, timestamp_ms)

            if results.hand_landmarks and results.handedness:
                for hand_id, (hand_lms, handedness) in enumerate(
                    zip(results.hand_landmarks, results.handedness)
                ):
                    # MediaPipe labels are mirrored for front-facing camera
                    label = handedness[0].category_name
                    hand_label = "Left" if label == "Right" else "Right"
                    w, h = self._bbox(hand_lms)
                    row = [frame_number, hand_id, hand_label, w, h]
                    for lm in hand_lms:
                        row.extend([lm.x, lm.y, lm.z])
                    rows.append(row)

            frame_number += 1

        cap.release()

        if not rows:
            return False

        output_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows, columns=self.LANDMARK_COLUMNS).to_csv(output_path, index=False)
        return True

    def process_task(self, base_processed_dir: Path, task: str, subtask: str) -> None:
        """
        Process all MP4 videos in *base_processed_dir/task/subtask/videos/* and write
        pose CSVs to *base_processed_dir/task/subtask/pose/*.
        """
        video_dir = Path(base_processed_dir) / task / subtask / "videos"
        output_dir = Path(base_processed_dir) / task / subtask / "pose"
        output_dir.mkdir(parents=True, exist_ok=True)

        for video_file in sorted(video_dir.glob("*.mp4")):
            out = output_dir / f"{video_file.stem}.csv"
            ok = self.process_video(video_file, out)
            status = "OK" if ok else "NO HANDS"
            print(f"  [{status}] {video_file.name}")

    def close(self) -> None:
        self._detector.close()
