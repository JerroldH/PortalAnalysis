"""
Optional local overrides. Copy to config.py to customize paths on your machine.

    copy config.example.py config.py

Or set environment variables (no config.py needed):
    PORTAL_DATA_DIR=N:/Booth_Processed
"""

from pathlib import Path

# Override processed data root (optional; portal_analysis.config has OS defaults)
BASE_PROCESSED_DIRECTORY = Path(r"N:/Booth_Processed")
