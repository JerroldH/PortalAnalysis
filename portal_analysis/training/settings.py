"""Runtime settings: environment variables or local config override."""

import os
from pathlib import Path


def get_base_processed_directory() -> Path:
    """
    Resolve Booth processed data directory.

    Priority:
    1. PORTAL_DATA_DIR or HAND_MOVEMENT_DATA_DIR environment variable
    2. portal_analysis.config.BASE_PROCESSED_DIRECTORY
    3. config.py at repo root (copy from config.example.py)
    """
    for env_name in ("PORTAL_DATA_DIR", "HAND_MOVEMENT_DATA_DIR"):
        env_dir = os.environ.get(env_name)
        if env_dir:
            return Path(env_dir)

    try:
        from portal_analysis.config import BASE_PROCESSED_DIRECTORY
        return Path(BASE_PROCESSED_DIRECTORY)
    except ImportError:
        pass

    try:
        from config import BASE_PROCESSED_DIRECTORY
        return Path(BASE_PROCESSED_DIRECTORY)
    except ImportError as exc:
        raise ImportError(
            "Data directory not configured. Set PORTAL_DATA_DIR, use "
            "portal_analysis.config.BASE_PROCESSED_DIRECTORY, or copy "
            "config.example.py to config.py."
        ) from exc
