"""Tests for kinematic feature plots."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import pandas as pd

from portal_analysis.preprocessing.kinematic_plots import (
    plot_kinematic_feature_over_time,
    plot_path_for_distances_csv,
)


def test_plot_path_beside_distances(tmp_path: Path):
    distances_csv = tmp_path / "finger_tapping" / "right" / "distances" / "P001_distances.csv"
    assert plot_path_for_distances_csv(distances_csv) == (
        tmp_path / "finger_tapping" / "right" / "plots" / "P001_distances.png"
    )


def test_plot_path_under_results(tmp_path: Path):
    distances_csv = tmp_path / "results" / "distances" / "P001_right_finger_tapping_distances.csv"
    assert plot_path_for_distances_csv(distances_csv) == (
        tmp_path / "results" / "plots" / "P001_right_finger_tapping_distances.png"
    )


def test_plot_kinematic_feature_over_time(tmp_path: Path):
    distances_dir = tmp_path / "finger_tapping" / "right" / "distances"
    distances_dir.mkdir(parents=True)
    distances_csv = distances_dir / "P001_distances.csv"
    pd.DataFrame(
        {
            "Frame": [0, 1, 2],
            "Finger Normalized Distance": [0.5, 0.8, 0.6],
        }
    ).to_csv(distances_csv, index=False)

    plot_path = plot_kinematic_feature_over_time(
        distances_csv,
        "Finger Normalized Distance",
    )

    assert plot_path.exists()
    assert plot_path.parent.name == "plots"
