"""
Train severity models for hand movement tasks.

Examples::

    python scripts/train_models.py
    python scripts/train_models.py --tasks finger_tapping
    python scripts/train_models.py --version v1.0.0 --processed-dir N:/Booth_Processed
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from portal_analysis.cli import main


if __name__ == "__main__":
    if len(sys.argv) == 1:
        main(["train", "--tasks", "all"])
    else:
        main(["train"] + sys.argv[1:])
