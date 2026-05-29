"""Plot kinematic time-series features from distances CSVs."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import matplotlib.pyplot as plt
import pandas as pd

_FRAME_COLUMN_CANDIDATES: Sequence[str] = (
    "Frame",
    "frame_number",
    "frame",
    "FrameNumber",
)


def plot_path_for_distances_csv(distances_csv: Path) -> Path:
    """``…/distances/foo_distances.csv`` → ``…/plots/foo_distances.png``.

    Works for both ``results/distances/`` and legacy ``<task>/<side>/distances/``.
    """
    distances_csv = Path(distances_csv)
    return distances_csv.parent.parent / "plots" / f"{distances_csv.stem}.png"


def _resolve_frame_column(distances_df: pd.DataFrame) -> str:
    for name in _FRAME_COLUMN_CANDIDATES:
        if name in distances_df.columns:
            return name
    raise ValueError(
        f"No frame column found in distances CSV (tried {_FRAME_COLUMN_CANDIDATES})."
    )


def plot_kinematic_feature_over_time(
    distances_csv: Path,
    value_column: str,
    plot_path: Optional[Path] = None,
) -> Path:
    """
    Plot a kinematic feature vs frame and save to *plot_path*.

    Defaults to ``plots/<distances_stem>.png`` beside the ``distances/`` folder.
    """
    distances_csv = Path(distances_csv)
    if plot_path is None:
        plot_path = plot_path_for_distances_csv(distances_csv)
    else:
        plot_path = Path(plot_path)

    distances_df = pd.read_csv(distances_csv)
    frame_column = _resolve_frame_column(distances_df)

    if value_column not in distances_df.columns:
        raise ValueError(
            f"'{value_column}' column not found in distances CSV: {distances_csv}"
        )

    x_values = distances_df[frame_column]
    y_values = distances_df[value_column]

    plot_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 4))
    plt.plot(x_values, y_values, label=value_column, linewidth=2)
    plt.xlabel(frame_column)
    plt.ylabel(value_column)
    plt.title(f"{value_column} Over Time")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Kinematic plot saved to {plot_path}")
    return plot_path
