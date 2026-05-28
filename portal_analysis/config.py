"""
Path configuration for PortalAnalysis (Booth Inference).
Mirrors the Hand Movement Analysis config for shared data directories.
"""

import platform
from pathlib import Path


PORTAL_DATA_DIR = Path(r"N:/Booth_Processed")
system = platform.system()

if system == "Windows":
    CAMERA_DIRECTORY = Path(r"N:/")
elif system == "Darwin":
    CAMERA_DIRECTORY = Path("/Volumes/Camera")
elif system == "Linux":
    CAMERA_DIRECTORY = Path("/mnt/Camera")
else:
    raise EnvironmentError(f"Unsupported OS: {system}")

BASE_RAW_DIRECTORY = CAMERA_DIRECTORY / "CAMERA Booth Data" / "Booth"
BASE_PROCESSED_DIRECTORY = CAMERA_DIRECTORY / "Booth_Processed"
BASE_RESULTS_DIRECTORY = CAMERA_DIRECTORY / "Booth_Results"

# Local models directory (pre-trained model files)
MODELS_DIRECTORY = Path(__file__).parent.parent / "models"
MODELS_DIRECTORY.mkdir(parents=True, exist_ok=True)

TASKS = {
    "hand_movement": {
        "both_still": "both_still.mp4",
        "left_open_close": "left_open_close.mp4",
        "right_open_close": "right_open_close.mp4",
        "left_up_down": "left_up_down.mp4",
        "right_up_down": "right_up_down.mp4",
    },
    "finger_tapping": {
        "left_finger_tapping": "left_finger_tapping.mp4",
        "right_finger_tapping": "right_finger_tapping.mp4",
    },
}

