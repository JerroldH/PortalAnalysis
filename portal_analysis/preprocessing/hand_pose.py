"""
Hand pose extraction from video using MediaPipe.
Produces per-frame CSV with 21 hand landmark coordinates and bounding-box dimensions.
"""

from pathlib import Path
import cv2
import mediapipe as mp
import pandas as pd


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
    ):
        self._mp_hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

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
        rows = []
        frame_number = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = self._mp_hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            if results.multi_hand_landmarks and results.multi_handedness:
                for hand_id, (hand_lms, handedness) in enumerate(
                    zip(results.multi_hand_landmarks, results.multi_handedness)
                ):
                    # MediaPipe labels are mirrored for front-facing camera
                    hand_label = "Left" if handedness.classification[0].label == "Right" else "Right"
                    w, h = self._bbox(hand_lms.landmark)
                    row = [frame_number, hand_id, hand_label, w, h]
                    for lm in hand_lms.landmark:
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
        self._mp_hands.close()
